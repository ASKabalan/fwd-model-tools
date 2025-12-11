import healpy as hp
import jax.core
import jax.numpy as jnp
import jax_healpy as jhp
import numpy as np


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
    """Initialize k-bins and meshes; kedges may be None (handled internally)."""
    mesh_shape = tuple(mesh_shape)
    box_shape = tuple(box_shape)

    mesh_shape_np = np.array(mesh_shape)
    box_shape_np = np.array(box_shape)

    if kedges is None:
        kmax = np.pi * np.min(mesh_shape_np / box_shape_np)
        dk = 2 * np.pi / np.min(box_shape_np) * 2  # twice fundamental
        kedges_np = np.arange(dk, kmax, dk) + dk / 2
        if kedges_np.size < 2:
            kedges_np = np.linspace(kmax / 4, kmax * 0.9, 2)
        kedges = jnp.asarray(kedges_np)
    else:
        kedges = jnp.asarray(kedges)

    ndim = len(mesh_shape)
    kvec = []
    for i, (m, b) in enumerate(zip(mesh_shape, box_shape)):
        freq = jnp.fft.fftfreq(m)
        shape = [1] * ndim
        shape[i] = m
        kvec.append((2 * jnp.pi * m / b) * freq.reshape(shape))
    kmesh = jnp.sqrt(sum(ki**2 for ki in kvec))

    dig = jnp.digitize(kmesh.reshape(-1), kedges)
    nbins = kedges.shape[0] + 1
    kcount = jnp.bincount(dig, length=nbins)

    kavg = jnp.bincount(dig, weights=kmesh.reshape(-1), length=nbins)
    kavg = kavg / jnp.where(kcount == 0, 1, kcount)
    kavg = kavg[1:-1]

    if los is None:
        mumesh = 1.0
    else:
        mumesh = sum(ki * losi for ki, losi in zip(kvec, los))
        kmesh_nozeros = jnp.where(kmesh == 0, 1, kmesh)
        mumesh = jnp.where(kmesh == 0, 0, mumesh / kmesh_nozeros)

    return dig, kcount, kavg, mumesh, kedges


def _power(
    mesh,
    mesh2=None,
    *,
    box_shape=None,
    kedges=None,
    multipoles=0,
    los=jnp.array([0.0, 0.0, 1.0]),
):
    """Compute auto/cross 3D power spectrum using distributed FFTs (no batching)."""

    mesh_shape = tuple(mesh.shape)
    mesh_shape_arr = jnp.asarray(mesh_shape)
    box_shape = tuple(box_shape) if box_shape is not None else mesh_shape
    box_shape_arr = jnp.asarray(box_shape)
    poles = multipoles if isinstance(multipoles, (list, tuple)) else (multipoles,)
    los = None if multipoles == 0 else tuple(np.asarray(los) / np.linalg.norm(los))

    meshk = jnp.fft.fftn(mesh, norm="ortho")

    dig, kcount, kavg, mumesh, kedges = _initialize_pk(mesh_shape, box_shape, kedges, los)
    n_bins = kedges.shape[0] + 1

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

        if mesh2 is None:
            psum = jnp.bincount(dig, weights=weights, length=n_bins)
        else:
            psum_real = jnp.bincount(dig, weights=weights.real, length=n_bins)
            psum_imag = jnp.bincount(dig, weights=weights.imag, length=n_bins)
            psum = (psum_real**2 + psum_imag**2) ** 0.5

        pk_list.append(psum)

    pk = jnp.stack(pk_list, axis=0)

    norm = jnp.where(kcount > 0, kcount, 1)
    pk = (pk / norm)[:, 1:-1] * (box_shape_arr / mesh_shape_arr).prod()

    if len(poles) == 1:
        return kavg, pk[0]
    return kavg, pk


def _flat_cl(map2d, map2=None, *, pixel_size=None, field_size=None, ell_edges=None):
    nx, ny = map2d.shape

    # pixel_size [radians per pixel] (scalar or (px, py)) or derived from field_size
    if pixel_size is None:
        if field_size is None:
            raise ValueError("pixel_size or field_size must be provided for flat-sky Cl")
        field_x, field_y = field_size
        px, py = field_x / nx, field_y / ny
    else:
        px, py = pixel_size

    if map2 is None:
        map2 = map2d
    elif map2.shape != map2d.shape:
        raise ValueError("map2 must have the same shape as map2d for cross Cl")

    # FFTs
    map_fft = jnp.fft.fft2(map2d)
    map2_fft = jnp.fft.fft2(map2)

    # flat-sky normalization: A_pix / N_pix = (px * py) / (nx * ny)
    norm = (px * py) / (nx * ny)
    pk_2d = (map_fft * map2_fft.conj()) * norm

    # ℓ-grid
    # ℓx, ℓy in 1/rad
    lx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=px)
    ly = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=py)
    LX, LY = jnp.meshgrid(lx, ly, indexing="ij")
    ell_grid = jnp.sqrt(LX**2 + LY**2)

    # bin edges
    if ell_edges is None:
        ell_max = ell_grid.max()
        ell_edges = jnp.linspace(0.0, ell_max, 50)
    ell_edges = jnp.asarray(ell_edges)

    # radial binning
    dig = jnp.digitize(ell_grid.reshape(-1), ell_edges)
    nbins = ell_edges.shape[0] + 1

    ell_count = jnp.bincount(dig, length=nbins)
    ell_sum = jnp.bincount(dig, weights=ell_grid.reshape(-1), length=nbins)
    cl_sum = jnp.bincount(dig, weights=pk_2d.reshape(-1).real, length=nbins)

    denom = jnp.where(ell_count > 0, ell_count, 1)
    ell_avg = (ell_sum / denom)[1:-1]  # drop underflow/overflow bins
    cl = (cl_sum / denom)[1:-1]

    return ell_avg, cl


def _spherical_cl(map_sphere, map_sphere2=None, *, lmax=None, method="jax"):
    """Spherical (HEALPix) angular power spectrum using jax_healpy.anafast."""
    if method == "healpy":
        if not jax.core.is_concrete(map_sphere):
            raise ValueError("method='healpy' requires concrete (non-jax) arrays")
        if map_sphere2 is not None and not jax.core.is_concrete(map_sphere2):
            raise ValueError("method='healpy' requires concrete (non-jax) arrays")

        map_sphere_np = np.asarray(map_sphere)
        map_sphere2_np = None if map_sphere2 is None else np.asarray(map_sphere2)
        cl = hp.anafast(map_sphere_np, map_sphere2_np, lmax=lmax, pol=False)
        ell_out = np.arange(cl.shape[0])
        return jnp.asarray(ell_out), jnp.asarray(cl)

    cl = jhp.anafast(map_sphere, map_sphere2, lmax=lmax, pol=False, method=method)
    ell_out = jnp.arange(cl.shape[-1])
    return ell_out, jnp.asarray(cl)


def _transfer(mesh0, mesh1, *, box_shape, kedges=None):
    """Monopole transfer function sqrt(P1/P0)."""
    k, pk0 = _power(mesh0, None, box_shape=box_shape, kedges=kedges, multipoles=0)
    _, pk1 = _power(mesh1, None, box_shape=box_shape, kedges=kedges, multipoles=0)
    return k, (pk1 / pk0) ** 0.5


def _coherence(mesh0, mesh1, *, box_shape, kedges=None):
    """Monopole coherence pk01 / sqrt(pk0 pk1)."""
    k, pk01 = _power(mesh0, mesh1, box_shape=box_shape, kedges=kedges, multipoles=0)
    _, pk0 = _power(mesh0, None, box_shape=box_shape, kedges=kedges, multipoles=0)
    _, pk1 = _power(mesh1, None, box_shape=box_shape, kedges=kedges, multipoles=0)
    return k, pk01 / (pk0 * pk1) ** 0.5
