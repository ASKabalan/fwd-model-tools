import jax
import pytest

from fwd_model_tools.fields import FlatDensity, SphericalDensity
from fwd_model_tools.initial import gaussian_initial_conditions
from fwd_model_tools.pm import lpt, nbody
from fwd_model_tools.probabilistic_models.full_field_model import Planck18
from fwd_model_tools.utils import compute_snapshot_scale_factors


@pytest.fixture(scope="module")
def small_simulation():
    cosmo = Planck18()
    mesh_size = (8, 8, 8)
    box_size = (100.0, 100.0, 100.0)
    key = jax.random.PRNGKey(0)
    density_field = gaussian_initial_conditions(
        key,
        mesh_size,
        box_size,
        cosmo=cosmo,
        nside=2,
        flatsky_npix=(8, 8),
    )
    dx_field, p_field = lpt(cosmo, density_field, a=0.1, order=1)
    ts = compute_snapshot_scale_factors(cosmo, density_field, nb_shells=2)
    return cosmo, density_field, dx_field, p_field, ts


def test_nbody_single_geometry_returns_field(small_simulation):
    cosmo, _, dx_field, p_field, ts = small_simulation
    spherical_lightcone = nbody(
        cosmo,
        dx_field,
        p_field,
        t1=1.0,
        dt0=0.1,
        ts=ts,
        geometry="spherical",
    )

    assert isinstance(spherical_lightcone, SphericalDensity)
    assert spherical_lightcone.array.shape[0] == ts.shape[0]
    assert spherical_lightcone.scale_factors.shape[0] == ts.shape[0]


def test_nbody_multi_geometry_tuple_returns_tuple(small_simulation):
    cosmo, _, dx_field, p_field, ts = small_simulation
    spherical_lightcone, flat_lightcone = nbody(
        cosmo,
        dx_field,
        p_field,
        t1=1.0,
        dt0=0.1,
        ts=ts,
        geometry=("spherical", "flat"),
    )

    assert isinstance(spherical_lightcone, SphericalDensity)
    assert isinstance(flat_lightcone, FlatDensity)
    assert spherical_lightcone.array.shape[0] == ts.shape[0]
    assert flat_lightcone.array.shape[0] == ts.shape[0]
    assert spherical_lightcone.scale_factors.shape == ts.shape
    assert flat_lightcone.scale_factors.shape == ts.shape
