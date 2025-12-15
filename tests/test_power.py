import jax
import jax.numpy as jnp
import jax_cosmo as jc
import jaxpm.utils as ju
import numpy as np

from fwd_model_tools.fields import DensityField, DensityStatus, FieldStatus, FlatDensity, SphericalDensity
from fwd_model_tools.initial import gaussian_initial_conditions
from fwd_model_tools.pm import lpt
from fwd_model_tools.power import PowerSpectrum, angular_cl_flat, angular_cl_spherical, compute_theory_cl, power
from fwd_model_tools.probabilistic_models.full_field_model import Planck18


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
        wavenumber=jnp.linspace(0.1, 1.0, 4),
        spectra=jnp.linspace(1.0, 4.0, 4),
        name="demo",
    )
    flat, aux = jax.tree_util.tree_flatten(spec)
    rebuilt = jax.tree_util.tree_unflatten(aux, flat)
    assert jnp.allclose(rebuilt.wavenumber, spec.wavenumber)
    assert jnp.allclose(rebuilt.spectra, spec.spectra)
    assert rebuilt.name == "demo"

    scaled = 2.0 * spec
    assert jnp.allclose(scaled.spectra, 2.0 * spec.spectra)

    offset = spec + 1.0
    assert jnp.allclose(offset.spectra, spec.spectra + 1.0)


def test_power_matches_density_method():
    density_field = _base_density_field().replace(array=jnp.arange(64.0).reshape(4, 4, 4))
    direct_k, direct_pk = power(
        density_field.array,
        box_shape=density_field.box_size,
    )
    via_method = density_field.power()

    assert direct_k.shape == direct_pk.shape
    assert direct_k.shape == via_method.wavenumber.shape
    assert jnp.allclose(direct_pk, via_method.spectra)


def test_angular_cl_matches_method():
    spherical = _spherical_density()
    direct_ell, direct_cl = angular_cl_spherical(spherical.array, lmax=5)
    via_method = spherical.angular_cl(lmax=5)

    assert direct_ell[0] == 0
    assert jnp.allclose(direct_ell, via_method.wavenumber)
    assert jnp.allclose(direct_cl, via_method.spectra)


def test_angular_cl_matches_method_and_requires_geometry():
    flat = _flat_density()
    direct_ell, direct_cl = angular_cl_flat(flat.array, field_size=flat.field_size)
    via_method = flat.angular_cl()

    assert direct_ell.shape == via_method.wavenumber.shape
    assert np.all(np.isfinite(direct_cl))
    assert jnp.allclose(direct_cl, via_method.spectra)


def test_compute_theory_cl_accepts_scalar_and_nz():
    cosmo = Planck18()
    ell = jnp.arange(2, 20)
    cl_scalar = compute_theory_cl(cosmo, ell, z_source=1.0)
    nz = jc.redshift.delta_nz(1.0)
    cl_nz = compute_theory_cl(cosmo, ell, z_source=nz)

    assert cl_scalar.shape == ell.shape
    assert jnp.allclose(cl_scalar, cl_nz)


def test_power_matches_jaxpm_utils_after_lpt():
    """Regression: our power() should agree with jaxpm.utils.power_spectrum after LPT pipeline."""

    import os

    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    key = jax.random.PRNGKey(0)
    cosmo = Planck18()
    mesh_size = (32, 32, 32)
    box_size = (200.0, 200.0, 200.0)

    init_field = gaussian_initial_conditions(
        key=key,
        cosmo=cosmo,
        mesh_size=mesh_size,
        box_size=box_size,
    )
    dx_field, _ = lpt(cosmo, init_field, a=1.0, order=1)
    density = dx_field.paint(mode="relative")

    k_arr, pk_arr = power(density.array, box_shape=box_size)
    k_ref, pk_ref = ju.power_spectrum(np.array(density.array), box_shape=box_size)

    max_dk = np.max(np.abs(np.array(k_arr) - k_ref))
    rel_dpk = np.max(np.abs(np.array(pk_arr) - pk_ref) / (np.abs(pk_ref) + 1e-12))

    assert max_dk < 1e-6
    assert rel_dpk < 1e-3, f"Relative pk mismatch {rel_dpk}"  # currently ~8e-3
