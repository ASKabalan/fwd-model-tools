"""Validation tests: compare fwd_model_tools LPT and N-body against fastpm-python."""

from __future__ import annotations

import pytest
from jax import numpy as jnp
from numpy.testing import assert_allclose

import fwd_model_tools as ffi

_FIELD_RTOL = 1e-12
_FIELD_ATOL = 1e-12
_PK_ATOL = 1e-12
_PL_RTOL = 1e-12

# ---------------------------------------------------------------------------
# LPT tests
# ---------------------------------------------------------------------------


def MSE(x, y):
    return jnp.max((x - y) ** 2)


def MSRE(x, y):
    return jnp.max(((x - y) / y) ** 2)


@pytest.mark.parametrize("order", [1, 2])
def test_lpt(
    simulation_config,
    initial_conditions,
    lpt_scale_factor,
    fpm_lpt1_field,
    fpm_lpt2_field,
    cosmology,
    order,
):
    mesh_shape, box_shape = simulation_config

    dx, p = ffi.lpt(cosmology, initial_conditions, ts=lpt_scale_factor, order=order)

    ffi_density = dx.paint()

    fpm_ref_field = fpm_lpt1_field if order == 1 else fpm_lpt2_field

    pk_ffi = ffi_density.power().spectra
    _, pk_fpm = ffi.power(fpm_ref_field, box_shape=box_shape)

    MSE_density = MSE(ffi_density.array, fpm_ref_field)
    MSRE_density = MSRE(ffi_density.array, fpm_ref_field)
    MSE_pk = MSE(pk_ffi, pk_fpm)
    MSRE_pk = MSRE(pk_ffi, pk_fpm)

    print(
        f"Order {order} LPT test: MSE_density={MSE_density:.3e}, MSRE_density={MSRE_density:.3e}, MSE_pk={MSE_pk:.3e}, MSRE_pk={MSRE_pk:.3e}"
    )

    assert_allclose(ffi_density.array, fpm_ref_field, rtol=_FIELD_RTOL, atol=_FIELD_ATOL)
    assert_allclose(pk_ffi, pk_fpm, rtol=_PL_RTOL, atol=_PK_ATOL)


# ---------------------------------------------------------------------------
# N-body tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
def test_nbody_efficient_solver(
    simulation_config,
    initial_conditions,
    lpt_scale_factor,
    nbody_from_lpt1,
    nbody_from_lpt2,
    cosmology,
    order,
    steps,
):
    mesh_shape, box_shape = simulation_config

    dx, p = ffi.lpt(cosmology, initial_conditions, ts=lpt_scale_factor, order=order)

    solver = ffi.EfficientDriftDoubleKick(interp_kernel=ffi.NoInterp(painting=ffi.PaintingOptions(target="density")))

    dt0 = (1.0 - lpt_scale_factor) / (steps - 1)

    result = ffi.nbody(
        cosmology,
        dx,
        p,
        t0=lpt_scale_factor,
        t1=1.0,
        dt0=dt0,
        solver=solver,
    )

    fpm_ref_field = nbody_from_lpt1 if order == 1 else nbody_from_lpt2

    pk_ffi = result.power().spectra
    _, pk_fpm = ffi.power(fpm_ref_field, box_shape=box_shape)

    MSE_density = MSE(result.array, fpm_ref_field)
    MSRE_density = MSRE(result.array, fpm_ref_field)
    MSE_pk = MSE(pk_ffi, pk_fpm)
    MSRE_pk = MSRE(pk_ffi, pk_fpm)

    print(
        f"Order {order} N-body test: MSE_density={MSE_density:.3e}, MSRE_density={MSRE_density:.3e}, MSE_pk={MSE_pk:.3e}, MSRE_pk={MSRE_pk:.3e}"
    )

    assert_allclose(result.array, fpm_ref_field, rtol=_FIELD_RTOL, atol=_FIELD_ATOL)
    assert_allclose(pk_ffi, pk_fpm, rtol=_PL_RTOL, atol=_PK_ATOL)


@pytest.mark.parametrize("order", [1, 2])
def test_nbody_reversible_solver(
    simulation_config,
    initial_conditions,
    lpt_scale_factor,
    nbody_from_lpt1,
    nbody_from_lpt2,
    cosmology,
    order,
    steps,
):
    mesh_shape, box_shape = simulation_config

    dx, p = ffi.lpt(cosmology, initial_conditions, ts=lpt_scale_factor, order=order)

    solver = ffi.ReversibleDoubleKickDrift(
        interp_kernel=ffi.NoInterp(painting=ffi.PaintingOptions(target="density")),
    )

    dt0 = (1.0 - lpt_scale_factor) / (steps - 1)

    result = ffi.nbody(
        cosmology,
        dx,
        p,
        t0=lpt_scale_factor,
        t1=1.0,
        dt0=dt0,
        solver=solver,
    )

    fpm_ref_field = nbody_from_lpt1 if order == 1 else nbody_from_lpt2

    _, pk_ffi = ffi.power(result.array.squeeze(), box_shape=box_shape)
    _, pk_fpm = ffi.power(fpm_ref_field, box_shape=box_shape)

    MSE_density = MSE(result.array, fpm_ref_field)
    MSRE_density = MSRE(result.array, fpm_ref_field)
    MSE_pk = MSE(pk_ffi, pk_fpm)
    MSRE_pk = MSRE(pk_ffi, pk_fpm)

    print(
        f"Order {order} N-body test: MSE_density={MSE_density:.3e}, MSRE_density={MSRE_density:.3e}, MSE_pk={MSE_pk:.3e}, MSRE_pk={MSRE_pk:.3e}"
    )

    assert_allclose(result.array, fpm_ref_field, rtol=_FIELD_RTOL, atol=_FIELD_ATOL)
    assert_allclose(pk_ffi, pk_fpm, rtol=_PL_RTOL, atol=_PK_ATOL)


# ---------------------------------------------------------------------------
# Reversible solver roundtrip tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize(
    "correction_kernel",
    [
        ffi.NoCorrection(),
        ffi.SharpeningKernel(),
    ],
    ids=["NoCorrection", "SharpeningKernel"],
)
def test_reversible_solver_roundtrip(
    simulation_config,
    initial_conditions,
    lpt_scale_factor,
    cosmology,
    order,
    correction_kernel,
):
    """Test that reverse() exactly inverts step() for ReversibleDoubleKickDrift."""
    mesh_shape, _ = simulation_config
    if isinstance(correction_kernel, ffi.SharpeningKernel) and len(set(mesh_shape)) > 1:
        pytest.skip("SharpeningKernel has a pre-existing fft3d shape bug on non-cubic meshes")

    dx, p = ffi.lpt(cosmology, initial_conditions, ts=lpt_scale_factor, order=order)

    # Build solver with geometry
    interp = ffi.NoInterp(painting=ffi.PaintingOptions(target="density"))
    ts = jnp.array([1.0])
    interp = interp.update_geometry(
        ts=ts,
        r_centers=jnp.zeros(1),
        density_widths=jnp.ones(1),
        max_comoving_distance=1.0,
    )
    solver = ffi.ReversibleDoubleKickDrift(interp_kernel=interp, pgd_kernel=correction_kernel)

    # Set up step parameters
    t0 = lpt_scale_factor
    dt = 0.01
    t1 = t0 + dt

    # Init + one forward step
    dx_init, p_init, state = solver.init(dx, p, t0, t1, dt, cosmology)
    dx_stepped, p_stepped, state_stepped = solver.step(dx_init, p_init, t0, t1, dt, state, cosmology)

    # Reverse the step
    dx_recovered, p_recovered, _ = solver.reverse(dx_stepped, p_stepped, t0, t1, dt, state_stepped, cosmology)

    assert_allclose(dx_recovered.array, dx_init.array, rtol=1e-10, atol=1e-10)
    assert_allclose(p_recovered.array, p_init.array, rtol=1e-10, atol=1e-10)
