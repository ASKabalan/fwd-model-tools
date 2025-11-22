import numpy as np
import pytest

import jax
import jax.numpy as jnp
import jax_cosmo as jc

from fwd_model_tools.fields import (DensityField, DensityStatus, FieldStatus,
                                   FlatDensity, SphericalDensity)
from fwd_model_tools.probabilistic_models.full_field_model import Planck18
from fwd_model_tools.power import (PowerSpectrum, compute_flat_cl, compute_pk,
                                   compute_spherical_cl, compute_theory_cl)


def _base_density_field() -> DensityField:
    return DensityField(
        array=jnp.zeros((4, 4, 4)),
        mesh_size=(4, 4, 4),
        box_size=(50.0, 50.0, 50.0),
        observer_position=(0.5, 0.5, 0.5),
        nside=2,
        flatsky_npix=(4, 4),
        field_size=2.0,
        halo_size=0,
        status=FieldStatus.DENSITY_FIELD,
    )


def _flat_density() -> FlatDensity:
    density_field = _base_density_field()
    return FlatDensity.FromDensityMetadata(
        array=jnp.arange(16.0).reshape(4, 4),
        density_field=density_field,
        status=DensityStatus.LIGHTCONE,
    )


def _spherical_density() -> SphericalDensity:
    density_field = _base_density_field()
    nside = density_field.nside
    npix = 12 * nside**2
    return SphericalDensity.FromDensityMetadata(
        array=jnp.linspace(0.0, 1.0, npix),
        density_field=density_field,
        status=DensityStatus.LIGHTCONE,
    )


def test_power_spectrum_pytree_roundtrip_and_ops():
    spec = PowerSpectrum(
        k=jnp.linspace(0.1, 1.0, 4),
        pk=jnp.linspace(1.0, 4.0, 4),
        label="demo",
    )
    flat, aux = jax.tree_util.tree_flatten(spec)
    rebuilt = jax.tree_util.tree_unflatten(aux, flat)
    assert jnp.allclose(rebuilt.k, spec.k)
    assert jnp.allclose(rebuilt.pk, spec.pk)
    assert rebuilt.label == "demo"

    scaled = 2.0 * spec
    assert jnp.allclose(scaled.pk, 2.0 * spec.pk)

    offset = spec + 1.0
    assert jnp.allclose(offset.pk, spec.pk + 1.0)


def test_compute_pk_matches_density_method():
    density_field = _base_density_field().replace(
        array=jnp.arange(64.0).reshape(4, 4, 4)
    )
    direct = compute_pk(density_field)
    via_method = density_field.compute_power_spectrum()

    assert direct.k.shape == direct.pk.shape
    assert direct.k.shape == via_method.k.shape
    assert jnp.allclose(direct.pk, via_method.pk)


def test_compute_spherical_cl_matches_method():
    spherical = _spherical_density()
    direct = compute_spherical_cl(spherical, lmax=5)
    via_method = spherical.compute_power_spectrum(lmax=5)

    assert direct.k[0] == 0
    assert jnp.allclose(direct.k, via_method.k)
    assert jnp.allclose(direct.pk, via_method.pk)


def test_compute_flat_cl_matches_method_and_requires_geometry():
    flat = _flat_density()
    direct = compute_flat_cl(flat)
    via_method = flat.compute_power_spectrum()

    assert direct.k.shape == via_method.k.shape
    assert np.all(np.isfinite(direct.pk))
    assert jnp.allclose(direct.pk, via_method.pk)

    with pytest.raises(ValueError):
        compute_flat_cl(jnp.ones((4, 4)))


def test_compute_theory_cl_accepts_scalar_and_nz():
    cosmo = Planck18()
    ell = jnp.arange(2, 20)
    cl_scalar = compute_theory_cl(cosmo, ell, z_source=1.0)
    nz = jc.redshift.delta_nz(1.0)
    cl_nz = compute_theory_cl(cosmo, ell, z_source=nz)

    assert cl_scalar.shape == ell.shape
    assert jnp.allclose(cl_scalar, cl_nz)
