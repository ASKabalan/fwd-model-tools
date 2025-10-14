from functools import partial

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpyro
import numpyro.distributions as dist
from diffrax import ODETerm, RecursiveCheckpointAdjoint, SaveAt, diffeqsolve
from jax_cosmo.scipy.integrate import simps
from jaxpm.distributed import fft3d, ifft3d
from jaxpm.lensing import (convergence_Born, density_plane_fn,
                           spherical_density_fn)
from jaxpm.ode import symplectic_ode
from jaxpm.pm import lpt

from fwd_model_tools.fields import DistributedNormal, linear_field
from fwd_model_tools.solvers.integrate import \
    integrate as reverse_adjoint_integrate
from fwd_model_tools.solvers.semi_implicit_euler import SemiImplicitEuler


def E(cosmo, a):
    """
    Hubble expansion function E(a) = H(a)/H0.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology object.
    a : float or jax.Array
        Scale factor.

    Returns
    -------
    float or jax.Array
        Dimensionless Hubble parameter E(a).
    """
    return jnp.sqrt(jc.background.Esqr(cosmo, a))


def Planck18(**kwargs):
    """
    Planck 2018 fiducial cosmology.

    This function returns a jax-cosmo Cosmology object with Planck 2018 parameters
    as default values. Any parameter can be overridden via kwargs.

    Parameters
    ----------
    **kwargs : dict
        Cosmological parameters to override defaults.

    Returns
    -------
    jax_cosmo.Cosmology
        Cosmology object with Planck18 parameters.

    Notes
    -----
    Default parameters:
    - Omega_c = 0.2607
    - Omega_b = 0.0490
    - Omega_k = 0.0
    - h = 0.6766
    - n_s = 0.9665
    - sigma8 = 0.8102
    - w0 = -1.0
    - wa = 0.0
    """
    defaults = {
        "Omega_c": 0.2607,
        "Omega_b": 0.0490,
        "Omega_k": 0.0,
        "h": 0.6766,
        "n_s": 0.9665,
        "sigma8": 0.8102,
        "w0": -1.0,
        "wa": 0.0,
    }
    defaults.update(kwargs)
    return jc.Cosmology(**defaults)


def integrate(terms, solver, t0, t1, dt0, y0, args, saveat, adjoint):
    """
    Run ODE integration with diffrax or reverse-mode adjoint integrator.

    Parameters
    ----------
    terms : tuple of ODETerm
        ODE terms defining the system dynamics.
    solver : AbstractSolver
        Diffrax solver instance.
    t0 : float
        Initial time.
    t1 : float
        Final time.
    dt0 : float
        Initial time step.
    y0 : PyTree
        Initial state.
    args : PyTree
        Additional arguments for the ODE.
    saveat : SaveAt
        Snapshot configuration.
    adjoint : RecursiveCheckpointAdjoint or custom
        Adjoint method for gradient computation.

    Returns
    -------
    tuple
        (solution, times) where solution contains the ODE solution at saveat times.
    """
    if isinstance(adjoint, RecursiveCheckpointAdjoint):
        solution = diffeqsolve(terms,
                               solver,
                               t0,
                               t1,
                               dt0,
                               y0,
                               args,
                               saveat=saveat,
                               adjoint=adjoint)
        return solution.ys, saveat.subs.ts
    else:
        solution = reverse_adjoint_integrate(terms, solver, t0, t1, dt0, y0,
                                             args, saveat)
        return solution, saveat.subs.ts


def make_full_field_model(
    field_size,
    field_npix,
    box_shape,
    box_size,
    density_plane_width=None,
    density_plane_npix=None,
    density_plane_smoothing=0.1,
    nside=None,
    adjoint=RecursiveCheckpointAdjoint(5),
    t0=0.1,
    t1=1.0,
    dt0=0.05,
    min_redshift=0.01,
    max_redshift=3.0,
    sharding=None,
    halo_size=0,
    geometry="spherical",
    observer_position=None,
):
    """
    Create the full forward model: linear field -> lensing convergence maps.

    This function constructs a JIT-compiled forward model that:
    1. Generates linear initial conditions from a power spectrum
    2. Runs JAXPM N-body simulation forward in time
    3. Computes lensing convergence maps via Born approximation

    Parameters
    ----------
    field_size : float
        Angular size of the field in degrees.
    field_npix : int
        Number of pixels along one side of the field.
    box_shape : tuple
        Shape of the simulation box (nx, ny, nz).
    box_size : tuple or list
        Physical size of the box in each dimension (Mpc/h).
    density_plane_width : int
        Width of density planes for ray tracing (Mpc/h).
    density_plane_npix : int
        Number of pixels per density plane side.
    density_plane_smoothing : float, default=0.1
        Smoothing scale for density planes (Mpc/h).
    nside : int, optional
        HEALPix nside parameter for spherical geometry.
    adjoint : RecursiveCheckpointAdjoint, default=RecursiveCheckpointAdjoint(5)
        Adjoint method for gradient computation.
    t0 : float, default=0.1
        Initial scale factor.
    t1 : float, default=1.0
        Final scale factor.
    dt0 : float, default=0.05
        Initial time step.
    min_redshift : float, default=0.01
        Minimum redshift for source integration.
    max_redshift : float, default=3.0
        Maximum redshift for source integration.
    sharding : Sharding, optional
        JAX sharding specification.
    halo_size : int, default=0
        Halo exchange size for distributed operations.
    geometry : str, default="spherical"
        Coordinate system: "spherical" (HEALPix) or "flat" (Cartesian).
    observer_position : tuple or list, optional
        Observer position as fraction of box size (x, y, z) between 0 and 1.
        If None, defaults to box center (0.5, 0.5, 0.5).

    Returns
    -------
    callable
        JIT-compiled forward model function with signature:
        forward_model(cosmo, nz_shear, initial_conditions) -> (convergence_maps, lightcone, lin_field)

    Raises
    ------
    ImportError
        If required jaxpm dependencies are not installed.
    AssertionError
        If geometry is not "spherical" or "flat", or if required parameters are None.

    Notes
    -----
    - Requires jaxpm for N-body simulation
    - Uses relative position painting (paint_absolute_pos=False) for better accuracy
    - Lightcone is constructed by saving density planes at specific scale factors
    """
    assert geometry in [
        "spherical", "flat"
    ], f"geometry must be 'spherical' or 'flat', got {geometry}"
    assert density_plane_width is not None
    assert density_plane_npix is not None

    if observer_position is None:
        observer_position = (0.5, 0.5, 0.5)

    observer_position_mpc = jnp.array([
        observer_position[0] * box_size[0],
        observer_position[1] * box_size[1],
        observer_position[2] * box_size[2],
    ])

    def forward_model(cosmo, nz_shear, initial_conditions):
        k = jnp.logspace(-4, 1, 128)
        pk = jc.power.linear_matter_power(cosmo, k)
        cosmo._workspace = {}

        def pk_fn(x):
            return jc.scipy.interpolate.interp(x.reshape([-1]), k,
                                               pk).reshape(x.shape)

        lin_field = linear_field(box_shape, box_size, pk_fn,
                                 initial_conditions)
        cosmo._workspace = {}
        dx, p, f = lpt(
            cosmo,
            lin_field,
            particles=None,
            a=t0,
            order=1,
            halo_size=halo_size,
            sharding=sharding,
        )
        cosmo._workspace = {}

        drift, kick = symplectic_ode(box_shape,
                                     paint_absolute_pos=False,
                                     halo_size=halo_size,
                                     sharding=sharding)
        ode_fn = ODETerm(kick), ODETerm(drift)

        a_init = t0
        max_radius = float(min(box_size) / 2.0)
        n_lens = int(max_radius // float(density_plane_width))
        r_edges = jnp.linspace(0.0,
                               float(n_lens) * float(density_plane_width),
                               n_lens + 1)
        r_center = 0.5 * (r_edges[1:] + r_edges[:-1])
        a_center = jc.background.a_of_chi(cosmo, r_center)
        cosmo._workspace = {}

        solver = SemiImplicitEuler()

        if geometry == "spherical":
            saveat = SaveAt(
                ts=a_center[::-1],
                fn=lambda t, y, args: spherical_density_fn(box_shape,
                                                           box_size,
                                                           nside,
                                                           observer_position_mpc,
                                                           density_plane_width,
                                                           sharding=sharding)
                (t, y[1], args),
            )
        else:
            saveat = SaveAt(
                ts=a_center[::-1],
                fn=lambda t, y, args: density_plane_fn(box_shape,
                                                       box_size,
                                                       density_plane_width,
                                                       density_plane_npix,
                                                       sharding=sharding)
                (t, y[1], args),
            )
        y0 = (p, dx)
        args = cosmo

        solution, _ = integrate(
            ode_fn,
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=y0,
            args=args,
            saveat=saveat,
            adjoint=adjoint,
        )

        lightcone = solution[::-1]

        if geometry == "spherical":
            convergence_maps = [
                simps(
                    lambda z: nz(z).reshape([-1, 1]) * convergence_Born(
                        cosmo, lightcone, r_center, a_center, z,
                        density_plane_width),
                    min_redshift,
                    max_redshift,
                    N=32,
                ) for nz in nz_shear
            ]
        else:
            dx = box_size[0] / density_plane_npix
            xgrid, ygrid = jnp.meshgrid(
                jnp.linspace(0, field_size, density_plane_npix,
                             endpoint=False),
                jnp.linspace(0, field_size, density_plane_npix,
                             endpoint=False),
            )
            coords = jnp.array(
                (jnp.stack([xgrid, ygrid], axis=0)) * (jnp.pi / 180))
            convergence_maps = [
                simps(
                    lambda z: nz(z).reshape([-1, 1, 1]) * convergence_Born(
                        cosmo,
                        lightcone,
                        r_center,
                        a_center,
                        z,
                        density_plane_width,
                        dx=dx,
                        coords=coords),
                    min_redshift,
                    max_redshift,
                    N=32,
                ) for nz in nz_shear
            ]

            convergence_maps = [
                kmap.reshape([
                    field_npix,
                    density_plane_npix // field_npix,
                    field_npix,
                    density_plane_npix // field_npix,
                ]).mean(axis=1).mean(axis=-1) for kmap in convergence_maps
            ]

        return convergence_maps, lightcone, lin_field

    return jax.jit(forward_model)


def full_field_probmodel(config):
    """
    Define the full-field forward model and observation likelihood.

    This function creates a NumPyro probabilistic model that can be used for
    Bayesian inference with HMC/NUTS sampling. It samples cosmological parameters
    and initial conditions, runs the forward model, and conditions on observations.

    Parameters
    ----------
    config : Configurations
        Configuration object containing all simulation parameters.

    Returns
    -------
    callable
        NumPyro model function that samples parameters and returns observed maps.

    Notes
    -----
    - Samples cosmological parameters from config.priors
    - Samples initial conditions as DistributedNormal (whitened Gaussian field)
    - Applies forward model to generate convergence maps
    - Observes maps with Gaussian likelihood (shape noise + shot noise)
    - Records lightcone and initial conditions as deterministic variables
    """

    def model():
        forward_model = make_full_field_model(
            config.field_size,
            config.field_npix,
            config.box_shape,
            config.box_size,
            config.density_plane_width,
            config.density_plane_npix,
            config.density_plane_smoothing,
            config.nside,
            adjoint=config.adjoint,
            t0=config.t0,
            dt0=config.dt0,
            t1=config.t1,
            min_redshift=config.min_redshift,
            max_redshift=config.max_redshift,
            sharding=config.sharding,
            halo_size=config.halo_size,
            geometry=config.geometry,
            observer_position=config.observer_position,
        )

        cosmo = config.fiducial_cosmology(**{
            k: numpyro.sample(k, v)
            for k, v in config.priors.items()
        })

        initial_conditions = numpyro.sample(
            "initial_conditions",
            DistributedNormal(jnp.zeros(config.box_shape),
                              jnp.ones(config.box_shape), config.sharding),
        )

        convergence_maps, lc, lin_field = forward_model(
            cosmo, config.nz_shear, initial_conditions)
        numpyro.deterministic("lightcone", lc)
        numpyro.deterministic("ic", lin_field)

        observed_maps = [
            numpyro.sample(
                f"kappa_{i}",
                dist.Normal(
                    k,
                    config.sigma_e /
                    jnp.sqrt(config.nz_shear[i].gals_per_arcmin2 *
                             (config.field_size * 60 / config.field_npix)**2),
                ),
            ) for i, k in enumerate(convergence_maps)
        ]

        return observed_maps

    return model
