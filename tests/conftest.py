"""Shared fixtures for fwd_model_tools tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import pytest

from fwd_model_tools.fields import DensityField, FieldStatus
from fwd_model_tools.fields.units import DensityUnit


@pytest.fixture(scope="session", autouse=True)
def enable_x64():
    """Enable float64 for all tests (required for lossless parquet round-trips)."""
    print(f"Enabling float64 precision for all tests.")
    jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def cosmology():
    """Planck18 cosmology instance."""
    return jc.Planck18()


@pytest.fixture(scope="session", autouse=True)
def patch_fastpm_growth(cosmology):
    """Patch FastPM's MatterDominated to use JaxCosmo/JaxPM growth factors."""
    from unittest.mock import patch

    import fastpm.background
    import jax_cosmo.background as jc_bg
    from jaxpm.growth import dGf2a, dGfa

    class JaxCosmoMatterDominated:

        def __init__(self, Omega0_m, Omega0_lambda=None, Omega0_k=0, a=None, a_normalize=1.0):
            self.cosmo = cosmology
            self.a_normalize = a_normalize

        def D1(self, a):
            a = jnp.atleast_1d(a)
            D = jc_bg.growth_factor(self.cosmo, a)
            norm = jc_bg.growth_factor(self.cosmo, jnp.atleast_1d(self.a_normalize))
            return np.array(D / norm)

        def f1(self, a):
            a = jnp.atleast_1d(a)
            return np.array(jc_bg.growth_rate(self.cosmo, a))

        def D2(self, a):
            a = jnp.atleast_1d(a)
            D = jc_bg.growth_factor_second(self.cosmo, a)
            norm = jc_bg.growth_factor_second(self.cosmo, jnp.atleast_1d(self.a_normalize))
            return np.array(D / norm)

        def f2(self, a):
            a = jnp.atleast_1d(a)
            return np.array(jc_bg.growth_rate_second(self.cosmo, a))

        def E(self, a):
            a = jnp.atleast_1d(a)
            return np.array(jc_bg.Esqr(self.cosmo, a)**0.5)

        def Gf(self, a):
            return self.f1(a) * self.D1(a) * a**2 * self.E(a)

        def Gp(self, a):
            return self.D1(a)

        def Gp2(self, a):
            return self.D2(a)

        def gp(self, a):
            return self.f1(a) * self.D1(a) / a

        def gp2(self, a):
            return self.f2(a) * self.D2(a) / a

        def Gf2(self, a):
            return self.f2(a) * self.D2(a) * a**2 * self.E(a)

        def gf(self, a):
            return np.array(dGfa(self.cosmo, jnp.atleast_1d(a)))

        def gf2(self, a):
            return np.array(dGf2a(self.cosmo, jnp.atleast_1d(a)))

    # Patch both locations to be safe
    p1 = patch('fastpm.background.MatterDominated', JaxCosmoMatterDominated)
    # fastpm.core imports MatterDominated, so we should patch it there too if possible,
    # but since we are patching at session start, hopefully fastpm.core hasn't been imported yet.
    # If it has, we might need to patch fastpm.core.MatterDominated too.
    # We'll patch fastpm.core just in case it gets imported/cached early.
    p2 = patch('fastpm.core.MatterDominated', JaxCosmoMatterDominated)

    with p1, p2:
        yield


# ---------------------------------------------------------------------------
# Simulation configuration fixtures for fastpm comparison tests
# ---------------------------------------------------------------------------


@pytest.fixture(
    scope="session",
    params=[
        ((32, 32, 32), (256., 256., 256.)),
        ((32, 32, 64), (256., 256., 512.)),
    ],
)
def simulation_config(request):
    return request.param


@pytest.fixture(scope="session", params=[0.1, 0.2])
def lpt_scale_factor(request):
    return request.param


@pytest.fixture(scope="session", params=[10, 50, 80])
def steps(request):
    """Number of steps for n-body tests."""
    return request.param


@pytest.fixture(scope="session")
def particle_mesh(simulation_config):
    from pmesh.pm import ParticleMesh

    mesh_shape, box_shape = simulation_config
    return ParticleMesh(BoxSize=box_shape, Nmesh=mesh_shape, dtype="f8")


@pytest.fixture(scope="session")
def fpm_initial_conditions(cosmology, particle_mesh):
    """Generate initial conditions using fastpm's pmesh (whitenoise + power spectrum)."""
    grid = particle_mesh.generate_uniform_particle_grid(shift=0).astype(np.float64)

    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(cosmology, k)

    def pk_fn(x):
        return jnp.interp(x.reshape([-1]), k, pk).reshape(x.shape)

    whitec = particle_mesh.generate_whitenoise(42, type="complex", unitary=False)
    lineark = whitec.apply(lambda k, v: jnp.sqrt(pk_fn(jnp.sqrt(sum(ki**2 for ki in k)))) * v * jnp.sqrt(
        (1 / v.BoxSize).prod()))
    init_mesh = lineark.c2r().value

    return lineark, grid, init_mesh


@pytest.fixture(scope="session")
def initial_conditions(fpm_initial_conditions, simulation_config):
    """Wrap the raw init_mesh into a DensityField with INITIAL_FIELD status."""
    _, _, init_mesh = fpm_initial_conditions
    mesh_shape, box_shape = simulation_config

    return DensityField(
        array=jnp.asarray(init_mesh),
        mesh_size=mesh_shape,
        box_size=box_shape,
        status=FieldStatus.INITIAL_FIELD,
        unit=DensityUnit.DENSITY,
    )


@pytest.fixture(scope="session")
def fpm_solver(cosmology, particle_mesh):
    from fastpm.core import Cosmology as FastPMCosmology
    from fastpm.core import Solver

    ref_cosmo = FastPMCosmology(cosmology)
    return Solver(particle_mesh, ref_cosmo, B=1)


@pytest.fixture(scope="session")
def fpm_lpt1(fpm_solver, fpm_initial_conditions, lpt_scale_factor):
    lineark, grid, _ = fpm_initial_conditions
    return fpm_solver.lpt(lineark, grid, lpt_scale_factor, order=1)


@pytest.fixture(scope="session")
def fpm_lpt1_field(fpm_lpt1, particle_mesh):
    return particle_mesh.paint(fpm_lpt1.X).value


@pytest.fixture(scope="session")
def fpm_lpt2(fpm_solver, fpm_initial_conditions, lpt_scale_factor):
    lineark, grid, _ = fpm_initial_conditions
    return fpm_solver.lpt(lineark, grid, lpt_scale_factor, order=2)


@pytest.fixture(scope="session")
def fpm_lpt2_field(fpm_lpt2, particle_mesh):
    return particle_mesh.paint(fpm_lpt2.X).value


@pytest.fixture(scope="session")
def nbody_from_lpt1(fpm_solver, fpm_lpt1, particle_mesh, lpt_scale_factor, steps):
    from fastpm.core import leapfrog

    stages = np.linspace(lpt_scale_factor, 1.0, steps, endpoint=True)
    finalstate = fpm_solver.nbody(fpm_lpt1, leapfrog(stages))
    return particle_mesh.paint(finalstate.X).value


@pytest.fixture(scope="session")
def nbody_from_lpt2(fpm_solver, fpm_lpt2, particle_mesh, lpt_scale_factor, steps):
    from fastpm.core import leapfrog

    stages = np.linspace(lpt_scale_factor, 1.0, steps, endpoint=True)
    finalstate = fpm_solver.nbody(fpm_lpt2, leapfrog(stages))
    return particle_mesh.paint(finalstate.X).value
