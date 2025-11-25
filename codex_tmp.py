import jax
import jax.numpy as jnp
import numpy as np
from lenstools import ConvergenceMap
import astropy.units as u

from src.fwd_model_tools._src.power._compute import _flat_cl


def _legendre(mu, ell: int):
    """Simple Legendre evaluator for non-negative integer ell."""
    ell = int(ell)
    if ell == 0:
        return jnp.ones_like(mu)
    if ell == 1:
        return mu
    P0 = jnp.ones_like(mu)
    P1 = mu
    for n in range(2, ell + 1):
        Pn = ((2 * n - 1) * mu * P1 - (n - 1) * P0) / n
        P0, P1 = P1, Pn
    return P1


def _initialize_pk(mesh_shape, box_shape, kedges, los):
    """Create k-bins, counts, averages, and mu grid; kedges optional."""
    mesh_shape_np = np.array(mesh_shape)
    box_shape_np = np.array(box_shape)

    if kedges is None:
        kmax = np.pi * np.min(mesh_shape_np / box_shape_np)
        dk = 2 * np.pi / np.min(box_shape_np) * 2
        kedges_np = np.arange(dk, kmax, dk) + dk / 2
        if kedges_np.size < 2:
            kedges_np = np.linspace(kmax / 4, kmax * 0.9, 2)
        kedges = jnp.asarray(kedges_np)
    else:
        kedges = jnp.asarray(kedges)

    kshapes = np.eye(len(mesh_shape_np), dtype=np.int32) * -2 + 1
    kvec = [
        (2 * np.pi * m / l) * jnp.fft.fftfreq(m).reshape(kshape)
        for m, l, kshape in zip(mesh_shape_np, box_shape_np, kshapes)
    ]
    kmesh = jnp.sqrt(sum(jnp.asarray(ki) ** 2 for ki in kvec))

    dig = jnp.digitize(kmesh.reshape(-1), kedges)
    nbins = kedges.shape[0] + 1
    kcount = jnp.bincount(dig, length=nbins)

    kavg = jnp.bincount(dig, weights=kmesh.reshape(-1), length=nbins)
    kavg = kavg / jnp.where(kcount == 0, 1, kcount)
    kavg = kavg[1:-1]

    if los is None:
        mumesh = 1.0
    else:
        los_np = np.array(los)
        mumesh = sum(jnp.asarray(ki) * losi for ki, losi in zip(kvec, los_np))
        kmesh_nozeros = jnp.where(kmesh == 0, 1, kmesh)
        mumesh = jnp.where(kmesh == 0, 0, mumesh / kmesh_nozeros)

    return dig, kcount, kavg, mumesh, kedges


def power_spec_3d_jittable(
    mesh,
    mesh2=None,
    *,
    box_shape=(1.0, 1.0, 1.0),
    kedges=None,
    multipoles=0,
    los=(0.0, 0.0, 1.0),
):
    box_shape = tuple(box_shape)
    poles = multipoles if isinstance(multipoles, (list, tuple)) else (multipoles,)
    los = None if multipoles == 0 else tuple(los)

    mesh_shape = tuple(mesh.shape)
    dig, kcount, kavg, mumesh, kedges = _initialize_pk(
        mesh_shape, box_shape, kedges, los
    )
    n_bins = kedges.shape[0] + 1

    meshk = jnp.fft.fftn(mesh, norm="ortho")
    if mesh2 is None:
        mmk = meshk.real**2 + meshk.imag**2
    else:
        meshk2 = jnp.fft.fftn(mesh2, norm="ortho")
        mmk = meshk * meshk2.conj()

    pk_list = []
    for ell in poles:  # poles is static (Python tuple) under jit
        ell_int = int(ell)
        w_ell = _legendre(mumesh, ell_int)
        weights = (mmk * (2 * ell_int + 1) * w_ell).reshape(-1)
        psum = jnp.bincount(dig, weights=weights, length=n_bins)
        pk_list.append(psum)
    pk = jnp.stack(pk_list, axis=0)

    norm = jnp.where(kcount > 0, kcount, 1)
    pk = (pk / norm)[..., 1:-1] * (jnp.asarray(box_shape) / jnp.asarray(mesh_shape)).prod()

    if len(poles) == 1:
        return kavg, pk[0]
    return kavg, pk


power_spec_3d_jit = jax.jit(
    power_spec_3d_jittable, static_argnames=("box_shape", "multipoles", "los")
)


def main():
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent / "EXAMPLES"))
    import utils as ex_utils

    mesh = jnp.arange(64.0).reshape(4, 4, 4)
    box = (50.0, 50.0, 50.0)

    # Build kedges once so both implementations use the same binning.
    _, _, _, _, kedges = _initialize_pk(mesh.shape, box, kedges=None, los=None)
    kedges_np = np.array(kedges)

    k_ref, pk_ref = ex_utils.power_spectrum(np.array(mesh), box_shape=np.array(box), kedges=kedges_np)
    k, pk = power_spec_3d_jit(mesh, box_shape=box, kedges=kedges_np)
    print("max |Δk|", float(jnp.max(jnp.abs(k - k_ref))))
    rel = float(jnp.max(jnp.abs(pk - pk_ref) / (np.abs(pk_ref) + 1e-12)))
    print("max rel Δpk", rel)

    mesh2 = mesh * 2
    kx, pkx = power_spec_3d_jit(mesh, mesh2, box_shape=box, kedges=kedges_np)
    kx_ref, pkx_ref = ex_utils.power_spectrum(np.array(mesh), np.array(mesh2), box_shape=np.array(box), kedges=kedges_np)
    print("cross max |Δk|", float(jnp.max(jnp.abs(kx - kx_ref))))
    relx = float(jnp.max(jnp.abs(pkx - pkx_ref) / (np.abs(pkx_ref) + 1e-12)))
    print("cross max rel Δpk", relx)


if __name__ == "__main__":
    # Validate _flat_cl against LensTools ConvergenceMap.powerSpectrum using identical ell_edges.
    nx = ny = 128
    field_size_deg = 10.0
    field_size_rad = np.deg2rad(field_size_deg)
    pixel_size = field_size_rad / nx

    rng = np.random.default_rng(0)
    m1_raw = rng.normal(size=(nx, ny)).astype(np.float32)
    m2_raw = rng.normal(size=(nx, ny)).astype(np.float32)
    # Convert to normalized overdensity-like fields to avoid near-zero spectra
    def normalize_map(m):
        m = m - m.mean()
        norm = np.mean(np.abs(m)) + 1e-6
        return m / norm
    m1 = normalize_map(m1_raw)
    m2 = normalize_map(m2_raw)

    ell_edges = np.linspace(0.0, np.pi / pixel_size, 50 + 1)

    ell_my, cl_my = _flat_cl(m1, None, pixel_size=pixel_size, ell_edges=ell_edges)
    ell_cross, cl_cross = _flat_cl(m1, m2, pixel_size=pixel_size, ell_edges=ell_edges)

    cm1 = ConvergenceMap(m1, angle=field_size_deg * u.deg)
    cm2 = ConvergenceMap(m2, angle=field_size_deg * u.deg)
    ell_lt, cl_lt = cm1.powerSpectrum(l_edges=ell_edges)
    ell_lt_cross, cl_lt_cross = cm1.cross(cm2, l_edges=ell_edges)

    rel_auto = float(np.max(np.abs(cl_my - cl_lt) / (np.abs(cl_lt) + 1e-12)))
    rel_cross = float(np.max(np.abs(cl_cross - cl_lt_cross) / (np.abs(cl_lt_cross) + 1e-12)))

    print("flat auto max rel diff", rel_auto)
    print("flat cross max rel diff", rel_cross)
