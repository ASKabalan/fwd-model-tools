from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import jax.numpy as jnp
import numpy as np
from jaxpm.utils import power_spectrum as jaxpm_power_spectrum

from fwd_model_tools.power.power_spec import PowerSpectrum

if TYPE_CHECKING:
    from fwd_model_tools.fields import DensityField, FlatDensity, SphericalDensity

WindowSpec = Optional[Callable[[jnp.ndarray], jnp.ndarray] | jnp.ndarray]

__all__ = ["compute_pk", "compute_spherical_cl", "compute_flat_cl"]


def compute_pk(
    field_or_array: "DensityField" | jnp.ndarray,
    *,
    kedges: Optional[jnp.ndarray | np.ndarray] = None,
    box_size: Optional[Sequence[float]] = None,
    axis: Optional[Sequence[int]] = None,
    window: WindowSpec = None,
    second_field: Optional[DensityField | jnp.ndarray] = None,
    multipoles: int = 0,
    los: Sequence[float] = (0.0, 0.0, 1.0),
) -> PowerSpectrum:
    """
    Estimate the 3D matter power spectrum P(k) using :func:`jaxpm.utils.power_spectrum`.
    """
    mesh, inferred_box = _prepare_volume(field_or_array, axis)
    mesh2 = None
    second_box = None
    if second_field is not None:
        mesh2, second_box = _prepare_volume(second_field, axis)

    final_box = tuple(box_size if box_size is not None else inferred_box)
    if second_box is not None and box_size is None:
        final_box = tuple(second_box)

    if len(final_box) != 3:
        raise ValueError("box_size must be a length-3 sequence of physical box lengths.")

    if window is not None:
        mesh = _apply_window(mesh, window)
        if mesh2 is not None:
            mesh2 = _apply_window(mesh2, window)

    k, pk = jaxpm_power_spectrum(
        mesh,
        mesh2,
        box_shape=np.asarray(final_box, dtype=np.float32),
        kedges=kedges,
        multipoles=multipoles,
        los=np.asarray(los, dtype=np.float32),
    )

    if pk.ndim > 1:
        raise ValueError(
            "compute_pk currently supports a single multipole; "
            f"received power spectrum with shape {pk.shape}."
        )

    return PowerSpectrum(k=jnp.asarray(k), pk=jnp.asarray(pk))


def compute_spherical_cl(
    field_or_array: "SphericalDensity" | jnp.ndarray | np.ndarray,
    *,
    map_b: SphericalDensity | jnp.ndarray | np.ndarray | None = None,
    nside: Optional[int] = None,
    lmax: Optional[int] = None,
) -> PowerSpectrum:
    """Estimate angular power spectra C_ell for HEALPix maps via :func:`healpy.anafast`."""
    import healpy as hp

    map_a, nside_a = _prepare_healpix_map(field_or_array, nside)
    map_b_arr = None
    if map_b is not None:
        map_b_arr, nside_b = _prepare_healpix_map(map_b, nside)
        if nside_a != nside_b:
            raise ValueError(f"nside mismatch: {nside_a} vs {nside_b}")

    if lmax is None:
        lmax = 3 * nside_a - 1

    cl = hp.anafast(map_a, map_b_arr, lmax=lmax)
    ell = np.arange(cl.shape[0])
    return PowerSpectrum(k=jnp.asarray(ell), pk=jnp.asarray(cl))


def compute_flat_cl(
    field_or_array: "FlatDensity" | jnp.ndarray | np.ndarray,
    *,
    field_size: Optional[float] = None,
    pixel_size: Optional[float] = None,
    nbins: int = 50,
    lmax: float | None = None,
    binning: str = "log",
) -> PowerSpectrum:
    """
    Estimate flat-sky angular power spectra by FFT-ing 2D maps and binning in |ℓ|.
    """
    map_array, reference_field = _prepare_flat_map(field_or_array)
    if map_array.ndim != 2:
        raise ValueError("compute_flat_cl expects a single 2D map (ny, nx).")

    ny, nx = map_array.shape
    pixel_deg = _infer_pixel_size_deg(reference_field, (ny, nx), field_size, pixel_size)
    pixel_rad = np.deg2rad(pixel_deg)

    fft_map = jnp.fft.fft2(jnp.asarray(map_array))
    power2d = (jnp.abs(fft_map) ** 2) / (ny * nx)
    power2d = power2d * (pixel_rad**2)

    fy = np.fft.fftfreq(ny, d=pixel_rad)
    fx = np.fft.fftfreq(nx, d=pixel_rad)
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    ell_grid = 2 * np.pi * np.sqrt(fx_grid**2 + fy_grid**2)

    ell_values = ell_grid.ravel()
    power_values = np.asarray(power2d).ravel()

    if lmax is None:
        lmax = float(ell_values.max())

    mask = ell_values > 0
    mask &= ell_values <= lmax
    ell_values = ell_values[mask]
    power_values = power_values[mask]

    if ell_values.size == 0:
        raise ValueError("No Fourier modes survived selection; check lmax/pixel_size.")

    if binning not in {"log", "linear"}:
        raise ValueError("binning must be 'log' or 'linear'.")

    ell_min = ell_values.min()
    if binning == "log":
        edges = np.geomspace(ell_min, lmax, nbins + 1)
    else:
        edges = np.linspace(ell_min, lmax, nbins + 1)

    sums, _ = np.histogram(ell_values, bins=edges, weights=power_values)
    counts, _ = np.histogram(ell_values, bins=edges)
    ell_centers = 0.5 * (edges[:-1] + edges[1:])

    valid = counts > 0
    cls = np.zeros_like(ell_centers)
    cls[valid] = sums[valid] / counts[valid]

    return PowerSpectrum(k=jnp.asarray(ell_centers[valid]), pk=jnp.asarray(cls[valid]))


def _prepare_volume(
    obj: Any,
    axis: Optional[Sequence[int]],
) -> tuple[jnp.ndarray, Optional[tuple[float, float, float]]]:
    array = jnp.asarray(obj.array if _is_density_field(obj) else obj)

    if axis is None:
        if array.ndim != 3:
            raise ValueError(
                f"Expected a 3D array for the density field, got shape {array.shape}."
            )
        mesh = array
    else:
        axis = tuple(axis)
        if len(axis) != 3:
            raise ValueError("axis must contain exactly three entries.")
        mesh = jnp.moveaxis(array, axis, (0, 1, 2))
        if mesh.ndim > 3:
            mesh = mesh.reshape((*mesh.shape[:3], -1)).mean(axis=-1)

    box = tuple(obj.box_size) if _is_density_field(obj) else None
    return mesh, box


def _prepare_healpix_map(
    obj: Any,
    nside: Optional[int],
) -> tuple[np.ndarray, int]:
    import healpy as hp

    if _is_spherical_density(obj):
        array = np.asarray(obj.array)
        inferred_nside = obj.nside
    else:
        array = np.asarray(obj)
        inferred_nside = nside

    if inferred_nside is None:
        raise ValueError("nside must be provided when passing raw HEALPix arrays.")

    expected = hp.nside2npix(inferred_nside)
    if array.shape[-1] != expected:
        raise ValueError(
            f"Array length {array.shape[-1]} does not match npix={expected} "
            f"for nside={inferred_nside}."
        )

    if array.ndim != 1:
        raise ValueError("Only single HEALPix maps are supported; pass array.ndim == 1.")

    return array.astype(float), inferred_nside


def _prepare_flat_map(
    obj: Any,
) -> tuple[np.ndarray, Optional[Any]]:
    if _is_flat_density(obj):
        return np.asarray(obj.array), obj
    return np.asarray(obj), None


def _infer_pixel_size_deg(
    reference: Optional[Any],
    spatial_shape: tuple[int, int],
    field_size: Optional[float],
    pixel_size: Optional[float],
) -> float:
    if pixel_size is not None:
        return float(pixel_size)
    if field_size is not None:
        return float(field_size) / max(spatial_shape)
    if reference is not None and reference.field_size is not None:
        return float(reference.field_size) / max(reference.flatsky_npix)
    raise ValueError(
        "Provide either field_size (total degrees) or pixel_size (deg/pixel) "
        "when computing flat-sky power spectra."
    )


def _apply_window(arr: jnp.ndarray, window: WindowSpec) -> jnp.ndarray:
    if callable(window):
        return window(arr)
    return arr * jnp.asarray(window)


def _is_density_field(obj: Any) -> bool:
    return hasattr(obj, "array") and hasattr(obj, "box_size") and hasattr(obj, "mesh_size")


def _is_flat_density(obj: Any) -> bool:
    return hasattr(obj, "array") and hasattr(obj, "flatsky_npix")


def _is_spherical_density(obj: Any) -> bool:
    return hasattr(obj, "array") and hasattr(obj, "nside")
