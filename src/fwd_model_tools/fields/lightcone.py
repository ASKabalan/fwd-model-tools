from __future__ import annotations

from math import ceil
from typing import Any, Iterable, Optional, Sequence, Tuple

import healpy as hp
import jax
import jax.core
import jax.numpy as jnp
import jax_healpy as jhp
import matplotlib.pyplot as plt
import numpy as np
from jax.image import resize
from jaxtyping import Array , Float
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fwd_model_tools.power import PowerSpectrum, angular_cl_flat, angular_cl_spherical

from .density import DensityField, DensityStatus, FieldStatus


@jax.tree_util.register_pytree_node_class
class FlatDensity(DensityField):
    """Flat-sky (2D) density or shear maps derived from volumetric simulations."""

    __slots__ = ()
    STATUS_ENUM = DensityStatus

    def __init__(
        self,
        *,
        array: Array,
        mesh_size: Tuple[int, int, int],
        box_size: Tuple[float, float, float],
        observer_position: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        sharding: Optional[Any] = None,
        nside: Optional[int] = None,
        flatsky_npix: Optional[Tuple[int, int]] = None,
        field_size: Optional[float] = None,
        halo_size: int | Tuple[int, int] = 0,
        z_source: Optional[Any] = None,
        status: DensityStatus = DensityStatus.LIGHTCONE,
        scale_factors: float = 1.0,
    ):
        arr = jnp.asarray(array)
        if arr.ndim == 2:
            spatial_shape = arr.shape
        elif arr.ndim == 3:
            spatial_shape = arr.shape[-2:]
        else:
            raise ValueError("FlatDensity array must have shape (ny, nx) or (n_planes, ny, nx).")

        if flatsky_npix is None:
            raise ValueError("FlatDensity requires `flatsky_npix`.")

        if spatial_shape != tuple(flatsky_npix):
            raise ValueError(f"Array spatial shape {spatial_shape} does not match flatsky_npix {flatsky_npix}.")

        super().__init__(
            array=arr,
            mesh_size=mesh_size,
            box_size=box_size,
            observer_position=observer_position,
            sharding=sharding,
            nside=nside,
            flatsky_npix=flatsky_npix,
            field_size=field_size,
            halo_size=halo_size,
            z_source=z_source,
            status=status,
            scale_factors=scale_factors,
        )

    def angular_cl(
        self,
        mesh2: Optional["FlatDensity"] = None,
        *,
        field_size: Optional[float] = None,
        pixel_size: Optional[float] = None,
        ell_edges: Iterable[float] | None = None,
        batch_size: Optional[int] = None,
    ) -> "PowerSpectrum":
        """Compute a flat-sky angular power spectrum C_ell (auto or cross)."""

        effective_field_size = field_size or self.field_size
        data1 = self.array
        data2 = mesh2.array if mesh2 is not None else None

        if data1.ndim == 2:
            data1 = data1[None, ...]
            data2 = data2[None, ...] if data2 is not None else None
        elif data1.ndim != 3:
            raise ValueError("FlatDensity.angular_cl expects array shape (ny,nx) or (B,ny,nx)")

        if data2 is not None and data2.shape != data1.shape:
            raise ValueError("mesh2 must match shape for cross Cl")

        def _compute(pair):
            m1, m2 = pair
            return angular_cl_flat(
                m1,
                m2,
                pixel_size=pixel_size,
                field_size=effective_field_size,
                ell_edges=ell_edges,
            )

        ell_stack, spectra_stack = jax.lax.map(_compute , (data1, data2) , batch_size=batch_size)
        wavenumber = ell_stack[0]
        spectra = spectra_stack if self.array.ndim == 3 else spectra_stack[0]
        return PowerSpectrum(wavenumber=wavenumber, spectra=spectra, name="cl")

    def plot(
        self,
        *,
        ax: Optional[plt.Axes | Sequence[plt.Axes]] = None,
        cmap: str = "magma",
        figsize: Optional[Tuple[float, float]] = None,
        ncols: int = 3,
        titles: Optional[Sequence[str]] = None,
        show_colorbar: bool = True,
        show_ticks: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        """
        Visualize one or more flat-sky maps using matplotlib.
        """
        if not jax.core.is_concrete(self.array):
            raise ValueError("Cannot plot/show traced arrays. Use outside of jit context.")

        data = jnp.asarray(self.array)
        if data.ndim == 2:
            data = data[None, ...]
        elif data.ndim != 3:
            raise ValueError("FlatDensity.plot expects array shape (ny, nx) or (n, ny, nx).")

        n_maps = data.shape[0]

        def _flatten_axes(axes_obj):
            if axes_obj is None:
                return None
            if isinstance(axes_obj, np.ndarray):
                return axes_obj.ravel()
            if isinstance(axes_obj, Sequence):
                return np.array(axes_obj, dtype=object).ravel()
            return np.array([axes_obj], dtype=object)

        axes_flat = _flatten_axes(ax)

        if axes_flat is None:
            ncols_eff = max(1, min(ncols, n_maps))
            nrows = ceil(n_maps / ncols_eff)
            if figsize is None:
                figsize = (5 * ncols_eff, 5 * nrows)
            fig, axes_created = plt.subplots(nrows, ncols_eff, figsize=figsize, squeeze=False)
            axes_flat = axes_created.ravel()
        else:
            if axes_flat.size < n_maps:
                raise ValueError("Provided axes array is too small for number of maps")
            fig = axes_flat[0].get_figure()
            axes_created = axes_flat  # keep user-provided layout as-is

        for idx, ax_i in enumerate(axes_flat):
            if idx < n_maps:
                im = ax_i.imshow(data[idx], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                if show_colorbar:
                    divider = make_axes_locatable(ax_i)
                    cax = divider.append_axes("right", size="3%", pad=0.03)
                    fig.colorbar(im, cax=cax)
                if not show_ticks:
                    ax_i.set_xticks([])
                    ax_i.set_yticks([])
                if titles and idx < len(titles):
                    ax_i.set_title(titles[idx])
            else:
                ax_i.axis("off")

        fig.tight_layout()

        axes_out = axes_flat if ax is not None else axes_created
        return fig, axes_out

    def __getitem__(self, key) -> "FlatDensity":
        if self.array.ndim < 3:
            raise ValueError(f"Indexing only supported for batched FlatDensity (3D array), "
                             f"got array with {self.array.ndim} dimensions")
        return super().__getitem__(key)

    def show(
        self,
        *,
        ax: Optional[plt.Axes | Sequence[plt.Axes]] = None,
        cmap: str = "magma",
        figsize: Optional[Tuple[float, float]] = None,
        ncols: int = 3,
        titles: Optional[Sequence[str]] = None,
        show_colorbar: bool = True,
        show_ticks: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs,
    ) -> None:
        """Plot and display flat maps using matplotlib."""
        self.plot(
            ax=ax,
            cmap=cmap,
            figsize=figsize,
            ncols=ncols,
            titles=titles,
            show_colorbar=show_colorbar,
            show_ticks=show_ticks,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
        plt.show()

    def ud_sample(self, new_npix):
        """
        Resample to new resolution using jax.image.resize.
        """
        if self.array.ndim == 3:
            new_shape = (self.array.shape[0], new_npix, new_npix)
        else:
            new_shape = (new_npix, new_npix)

        resampled = resize(self.array, new_shape, method="bilinear")

        return self.replace(array=resampled, flatsky_npix=(new_npix, new_npix))


@jax.tree_util.register_pytree_node_class
class SphericalDensity(DensityField):
    """Spherical (HEALPix) density or shear maps produced from simulations."""

    __slots__ = ()
    STATUS_ENUM = DensityStatus

    def __init__(
        self,
        *,
        array: Array,
        mesh_size: Tuple[int, int, int],
        box_size: Tuple[float, float, float],
        observer_position: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        sharding: Optional[Any] = None,
        nside: Optional[int] = None,
        flatsky_npix: Optional[Tuple[int, int]] = None,
        field_size: Optional[float] = None,
        halo_size: int | Tuple[int, int] = 0,
        z_source: Optional[Any] = None,
        status: DensityStatus = DensityStatus.LIGHTCONE,
        scale_factors: float = 1.0,
    ):
        if nside is None:
            raise ValueError("SphericalDensity requires `nside`.")

        arr = jnp.asarray(array)
        npix = jhp.nside2npix(nside)
        if arr.shape[-1] != npix:
            raise ValueError(f"Array last dimension {arr.shape[-1]} does not match HEALPix npix "
                             f"{npix} for nside {nside}.")

        super().__init__(
            array=arr,
            mesh_size=mesh_size,
            box_size=box_size,
            observer_position=observer_position,
            sharding=sharding,
            nside=nside,
            flatsky_npix=flatsky_npix,
            field_size=field_size,
            halo_size=halo_size,
            z_source=z_source,
            status=status,
            scale_factors=scale_factors,
        )

    def angular_cl(
        self,
        mesh2: Optional["SphericalDensity"] = None,
        *,
        lmax: Optional[int] = None,
        method: str = "jax",
        batch_size: Optional[int] = None,
    ) -> "PowerSpectrum":
        """Compute a spherical (HEALPix) angular power spectrum C_ell (auto or cross)."""

        def _compute(pair):
            m1, m2 = pair
            return angular_cl_spherical(m1, m2, lmax=lmax, method=method)

        data1 = self.array
        data2 = mesh2.array if mesh2 is not None else None

        if method == "healpy":
            if data1.ndim > 2:
                raise ValueError("SphericalDensity.angular_cl with method='healpy' only supports unbatched arrays.")
            ell , spectra = angular_cl_spherical(data1, data2, lmax=lmax, method=method)
            return PowerSpectrum(wavenumber=ell, spectra=spectra, name="cl")

        if data1.ndim == 1:
            data1 = data1[None, ...]
            data2 = data2[None, ...] if data2 is not None else None
        elif data1.ndim != 2:
            raise ValueError("SphericalDensity.angular_cl expects array shape (npix) or (B,npix)")

        if data2 is not None and data2.shape != data1.shape:
            raise ValueError("mesh2 must match shape for cross Cl")




        ell_stack, spectra_stack = jax.lax.map(_compute , (data1, data2) , batch_size=batch_size)
        wavenumber = ell_stack[0]
        spectra = spectra_stack if self.array.ndim == 2 else spectra_stack[0]
        return PowerSpectrum(wavenumber=wavenumber, spectra=spectra, name="cl")

    def plot(
        self,
        *,
        ax: Optional[plt.Axes | Sequence[plt.Axes]] = None,
        cmap: str = "magma",
        figsize: Tuple[float, float] | None = None,
        ncols: int = 3,
        titles: Optional[Sequence[str]] = None,
        show_colorbar: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        """
        Visualize one or more spherical maps using ``healpy.mollview``.
        """
        if not jax.core.is_concrete(self.array):
            raise ValueError("Cannot plot/show traced arrays. Use outside of jit context.")

        data = jnp.asarray(self.array)
        if data.ndim == 1:
            data = data[None, ...]
        elif data.ndim < 1:
            raise ValueError("SphericalDensity array rank must be ≥1.")
        else:
            data = data.reshape((-1, data.shape[-1]))

        n_maps = data.shape[0]

        def _flatten_axes(axes_obj):
            if axes_obj is None:
                return None
            if isinstance(axes_obj, np.ndarray):
                return axes_obj.ravel()
            if isinstance(axes_obj, Sequence):
                return np.array(axes_obj, dtype=object).ravel()
            return np.array([axes_obj], dtype=object)

        axes_flat = _flatten_axes(ax)

        if axes_flat is None:
            requested_cols = ncols or 3
            ncols_eff = max(1, min(requested_cols, n_maps))
            nrows_eff = ceil(n_maps / ncols_eff)
            if figsize is None:
                figsize = (4 * ncols_eff, 4 * nrows_eff)
            fig, axes_created = plt.subplots(nrows_eff, ncols_eff, figsize=figsize, squeeze=False)
            axes_flat = axes_created.ravel()
        else:
            if axes_flat.size < n_maps:
                raise ValueError("Provided axes array is too small for number of maps")
            fig = axes_flat[0].get_figure()
            axes_created = axes_flat  # keep user-provided layout as-is

        def _sub_from_ax(ax_obj):
            sp = ax_obj.get_subplotspec()
            gs = sp.get_gridspec()
            row = sp.rowspan.start
            col = sp.colspan.start
            return gs.nrows, gs.ncols, row * gs.ncols + col + 1

        def _attach_delegate(ax_obj, delegate):
            ax_obj._healpy_delegate = delegate  # type: ignore[attr-defined]
            ax_obj.set_title = delegate.set_title  # type: ignore[assignment]
            ax_obj.get_title = delegate.get_title  # type: ignore[assignment]

        for idx, ax_i in enumerate(axes_flat):
            if idx < n_maps:
                title = titles[idx] if titles and idx < len(titles) else ""
                map_np = np.asarray(data[idx])

                ax_i.axis("off")
                sub = _sub_from_ax(ax_i)
                delegate = hp.mollview(
                    map_np,
                    fig=fig,
                    sub=sub,
                    cmap=cmap,
                    title=title,
                    bgcolor=(0.0, ) * 4,
                    cbar=show_colorbar,
                    min=vmin if vmin is not None else 0,
                    max=vmax if vmax is not None else
                    (np.percentile(map_np[map_np > 0], 95) if np.any(map_np > 0) else np.max(map_np)),
                )
                if delegate is None:
                    delegate = next(
                        (ax for ax in fig.axes if isinstance(ax, hp.projaxes.HpxMollweideAxes)),
                        None,
                    )
                if delegate is None:
                    raise RuntimeError("healpy.mollview did not return a Mollweide axes.")
                _attach_delegate(ax_i, delegate)
            else:
                ax_i.axis("off")

        axes_out = axes_created
        return fig

    def show(
        self,
        *,
        ax: Optional[plt.Axes | Sequence[plt.Axes]] = None,
        cmap: str = "magma",
        figsize: Tuple[float, float] | None = None,
        ncols: int = 3,
        titles: Optional[Sequence[str]] = None,
        apply_log: bool = True,
        show_colorbar: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot and display spherical maps using healpy.
        """
        if not jax.core.is_concrete(self.array):
            raise ValueError("Cannot plot/show traced arrays. Use outside of jit context.")

        self.plot(
            ax=ax,
            cmap=cmap,
            figsize=figsize,
            ncols=ncols,
            titles=titles,
            show_colorbar=show_colorbar,
            **kwargs,
        )
        plt.show()

    def ud_sample(self, new_nside):
        """
        Change HEALPix resolution using jax_healpy.ud_grade.
        """
        resampled = jhp.ud_grade(self.array, new_nside)
        return self.replace(array=resampled, nside=new_nside)
