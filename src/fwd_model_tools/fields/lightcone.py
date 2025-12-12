from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Optional

import jax
import jax.core
import jax.numpy as jnp
import jax_healpy as jhp
import matplotlib.pyplot as plt
from jax.image import resize
from jaxtyping import Array

from fwd_model_tools.power import PowerSpectrum, angular_cl_flat, angular_cl_spherical

from .._src.base._core import AbstractField
from .._src.base._enums import FieldStatus, PhysicalUnit
from .._src.fields._plotting import (
    generate_titles,
    plot_flat_density,
    plot_spherical_density,
    prepare_axes,
)
from .units import DensityUnit, convert_units


class FlatDensity(AbstractField):
    """Flat-sky (2D) density or shear maps derived from volumetric simulations."""

    STATUS_ENUM = FieldStatus

    def __check_init__(self):
        """Validation hook called after Equinox auto-initialization."""
        super().__check_init__()
        # Validate array shape
        if self.array.ndim == 2:
            spatial_shape = self.array.shape
        elif self.array.ndim == 3:
            spatial_shape = self.array.shape[-2:]
        else:
            raise ValueError("FlatDensity array must have shape (ny, nx) or (n_planes, ny, nx).")
        # Validate required fields
        if self.flatsky_npix is None:
            raise ValueError("FlatDensity requires `flatsky_npix`.")
        if self.density_width is None:
            raise ValueError("FlatDensity requires `density_width`.")
        # Validate spatial shape matches flatsky_npix
        if spatial_shape != tuple(self.flatsky_npix):
            raise ValueError(f"Array spatial shape {spatial_shape} does not match flatsky_npix {self.flatsky_npix}.")

    def __getitem__(self, key) -> FlatDensity:
        if self.array.ndim < 3:
            raise ValueError(
                f"Indexing only supported for batched FlatDensity (3D array), "
                f"got array with {self.array.ndim} dimensions"
            )
        return super().__getitem__(key)

    def to(
        self,
        unit: DensityUnit,
        *,
        omega_m: Optional[float] = None,
        h: Optional[float] = None,
        mean_density: Optional[float] = None,
    ) -> FlatDensity:
        """
        Convert the flat-sky map to a different density unit.

        Parameters
        ----------
        unit : DensityUnit
            Target unit for conversion.
        omega_m : float, optional
            Matter density parameter. Required for MSUN_H_PER_MPC3 conversions.
        h : float, optional
            Hubble parameter (H0 / 100 km/s/Mpc). Required for MSUN_H_PER_MPC3 conversions.
        mean_density : float, optional
            Mean density ρ̄ in particles per (Mpc/h)^3. Required when converting
            FROM OVERDENSITY to other units.

        Returns
        -------
        FlatDensity
            New FlatDensity with converted array and updated unit.
        """
        if self.unit == unit:
            return self

        ny_pix, nx_pix = self.flatsky_npix
        Lx, Ly, _ = self.box_size  # last dim is always Z
        volume_element = (Lx / nx_pix) * (Ly / ny_pix) * self.density_width

        new_array = convert_units(
            array=self.array,
            origin=self.unit,
            destination=unit,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            h=h,
            omega_m=omega_m,
            mean_density=mean_density,
            volume_element=volume_element,
            sharding=self.sharding,
        )

        return self.replace(array=new_array, unit=unit)

    def plot(
        self,
        *,
        ax: Optional[plt.Axes | Sequence[plt.Axes]] = None,
        cmap: str = "magma",
        figsize: Optional[tuple[float, float]] = None,
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

        if titles is None:
            titles = generate_titles("Flat Sky Density", self.scale_factors, n_maps)

        fig, axes = prepare_axes(ax, n_maps, ncols, figsize=figsize)

        for idx, ax_i in enumerate(axes):
            if idx < n_maps:
                plot_flat_density(
                    ax_i,
                    data[idx],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    show_colorbar=show_colorbar,
                    show_ticks=show_ticks,
                    title=titles[idx] if titles and idx < len(titles) else None,
                )
            else:
                ax_i.axis("off")

        fig.tight_layout()
        return fig, axes

    def show(
        self,
        *,
        ax: Optional[plt.Axes | Sequence[plt.Axes]] = None,
        cmap: str = "magma",
        figsize: Optional[tuple[float, float]] = None,
        ncols: int = 3,
        titles: Optional[Sequence[str]] = None,
        show_colorbar: bool = True,
        show_ticks: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        **kwargs,
    ) -> None:
        """Plot and display flat maps using matplotlib."""
        import matplotlib.pyplot as plt

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

    def angular_cl(
        self,
        mesh2: Optional[FlatDensity] = None,
        *,
        field_size: Optional[float] = None,
        pixel_size: Optional[float] = None,
        ell_edges: Iterable[float] | None = None,
        batch_size: Optional[int] = None,
    ) -> PowerSpectrum:
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

        ell_stack, spectra_stack = jax.lax.map(_compute, (data1, data2), batch_size=batch_size)
        wavenumber = ell_stack[0]
        spectra = spectra_stack if self.array.ndim == 3 else spectra_stack[0]
        return PowerSpectrum(
            wavenumber=wavenumber, array=spectra, name="cl", scale_factors=self.scale_factors
        )


class SphericalDensity(AbstractField):
    """Spherical (HEALPix) density or shear maps produced from simulations."""

    STATUS_ENUM = FieldStatus

    def __check_init__(self):
        """Validation hook called after Equinox auto-initialization."""
        super().__check_init__()
        # Validate nside is provided
        if self.nside is None:
            raise ValueError("SphericalDensity requires `nside`.")
        # Validate array shape matches HEALPix npix
        if self.array is not None:
            array_shape = getattr(self.array, "shape", ())
            npix = jhp.nside2npix(self.nside)
            if array_shape != () and array_shape[-1] != npix:
                raise ValueError(
                    f"Array last dimension {array_shape[-1]} does not match HEALPix npix {npix} for nside {self.nside}."
                )

    def __getitem__(self, key) -> SphericalDensity:
        if self.array.ndim < 2:
            raise ValueError(
                f"Indexing only supported for batched SphericalDensity (2D array), "
                f"got array with {self.array.ndim} dimensions"
            )
        return super().__getitem__(key)

    def to(
        self,
        unit: DensityUnit,
        *,
        omega_m: Optional[float] = None,
        h: Optional[float] = None,
        mean_density: Optional[float] = None,
    ) -> SphericalDensity:
        """
        Convert the spherical (HEALPix) map to a different density unit.

        Parameters
        ----------
        unit : DensityUnit
            Target unit for conversion.
        omega_m : float, optional
            Matter density parameter. Required for MSUN_H_PER_MPC3 conversions.
        h : float, optional
            Hubble parameter (H0 / 100 km/s/Mpc). Required for MSUN_H_PER_MPC3 conversions.
        mean_density : float, optional
            Mean density in particles per (Mpc/h)^3, required when converting
            FROM OVERDENSITY to other units.

        Returns
        -------
        SphericalDensity
            New SphericalDensity with converted array and updated unit.
        """
        if self.unit == unit:
            return self

        if self.comoving_centers is None or self.density_width is None:
            raise ValueError("comoving_centers and density_width metadata are required for unit conversion.")

        # Calculate volume per pixel in spherical shell (exact shell volume)
        R_max, R_min = self.comoving_centers + self.density_width / 2, self.comoving_centers - self.density_width / 2
        npix = jhp.nside2npix(self.nside)
        pixel_solid_angle = 4 * jnp.pi / npix  # steradians per pixel
        shell_volume_per_pixel = pixel_solid_angle * (R_max**3 - R_min**3) / 3.0

        new_array = convert_units(
            array=self.array,
            origin=self.unit,
            destination=unit,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            h=h,
            omega_m=omega_m,
            mean_density=mean_density,
            volume_element=shell_volume_per_pixel,
            sharding=self.sharding,
        )

        return self.replace(array=new_array, unit=unit)

    def plot(
        self,
        *,
        ax: Optional[plt.Axes | Sequence[plt.Axes]] = None,
        cmap: str = "magma",
        figsize: tuple[float, float] | None = None,
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
            raise ValueError("SphericalDensity array rank must be >= 1.")
        else:
            data = data.reshape((-1, data.shape[-1]))

        n_maps = data.shape[0]

        if titles is None:
            titles = generate_titles("Spherical Density", self.scale_factors, n_maps)

        fig, axes = prepare_axes(ax, n_maps, ncols, figsize=figsize)

        for idx, ax_i in enumerate(axes):
            if idx < n_maps:
                title = titles[idx] if titles and idx < len(titles) else ""
                plot_spherical_density(
                    ax_i,
                    data[idx],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    show_colorbar=show_colorbar,
                    title=title,
                )
            else:
                ax_i.axis("off")

        return fig

    def show(
        self,
        *,
        ax: Optional[plt.Axes | Sequence[plt.Axes]] = None,
        cmap: str = "magma",
        figsize: tuple[float, float] | None = None,
        ncols: int = 3,
        titles: Optional[Sequence[str]] = None,
        apply_log: bool = True,
        show_colorbar: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot and display spherical maps using healpy.
        """
        import matplotlib.pyplot as plt

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

    def angular_cl(
        self,
        mesh2: Optional[SphericalDensity] = None,
        *,
        lmax: Optional[int] = None,
        method: str = "jax",
        batch_size: Optional[int] = None,
    ) -> PowerSpectrum:
        """Compute a spherical (HEALPix) angular power spectrum C_ell (auto or cross)."""

        def _compute(pair):
            m1, m2 = pair
            return angular_cl_spherical(m1, m2, lmax=lmax, method=method)

        data1 = self.array
        data2 = mesh2.array if mesh2 is not None else None

        if data1.ndim == 1:
            data1 = data1[None, ...]
            data2 = data2[None, ...] if data2 is not None else None
        elif data1.ndim != 2:
            raise ValueError("SphericalDensity.angular_cl expects array shape (npix) or (B,npix)")

        if method == "healpy":
            spectras = []
            for i in range(data1.shape[0]):
                map1 = data1[i]
                map2 = data2[i] if data2 is not None else None
                ell, spectra = angular_cl_spherical(map1, map2, lmax=lmax, method=method)
                spectras.append(spectra)
            spectra = jnp.stack(spectras, axis=0)
            spectra = spectra if self.array.ndim == 2 else spectra[0]
            return PowerSpectrum(
                wavenumber=ell, array=spectra, name="cl", scale_factors=self.scale_factors
            )

        if data2 is not None and data2.shape != data1.shape:
            raise ValueError("mesh2 must match shape for cross Cl")

        ell_stack, spectra_stack = jax.lax.map(_compute, (data1, data2), batch_size=batch_size)
        wavenumber = ell_stack[0]
        spectra = spectra_stack if self.array.ndim == 2 else spectra_stack[0]
        return PowerSpectrum(
            wavenumber=wavenumber, array=spectra, name="cl", scale_factors=self.scale_factors
        )
