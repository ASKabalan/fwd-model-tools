"""
Convergence (kappa) and shear (gamma) field classes for weak lensing.

This module provides Flat/SphericalKappaField and Flat/SphericalShearField
classes that inherit from the density field stack and carry lensing-specific
metadata such as source redshifts.
"""

from typing import Optional

import jax.numpy as jnp

from fwd_model_tools.power import compute_flat_cl, compute_spherical_cl

from .density import DensityField, DensityStatus, FlatDensity, SphericalDensity

__all__ = [
    "FlatKappaField",
    "SphericalKappaField",
    "FlatShearField",
    "SphericalShearField",
]


class FlatKappaField(FlatDensity):
    """
    Convergence map in flat-sky (Cartesian) geometry.

    Inherits from FlatDensity, adding lensing-specific methods for
    resampling, power spectrum computation, and shear derivation.

    Attributes
    ----------
    z_source : float | jnp.ndarray
        Source redshift(s) for the convergence map
    array : Array
        Convergence values, shape (ny, nx) or (n_sources, ny, nx)
    """

    def __init__(
        self,
        *,
        array,
        mesh_size,
        box_size,
        observer_position=(0.5, 0.5, 0.5),
        sharding=None,
        nside=None,
        flatsky_npix=None,
        field_size=None,
        halo_size=0,
        z_source=None,
        status=DensityStatus.KAPPA,
        scale_factors=1.0,
    ):
        """
        Initialize FlatKappaField with explicit metadata, mirroring DensityField.__init__.
        """
        super().__init__(
            array=array,
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

    @classmethod
    def FromDensityMetadata(
        cls,
        *,
        array,
        density_field: DensityField,
        status=DensityStatus.KAPPA,
        z_source=None,
        scale_factors=None,
    ) -> "FlatKappaField":
        """
        Construct FlatKappaField from a reference DensityField and convergence array.
        """
        return super().FromDensityMetadata(
            array=array,
            density_field=density_field,
            status=status,
            z_source=z_source,
            scale_factors=scale_factors,
        )

    def compute_power_spectrum(
        self,
        *,
        field_size: Optional[float] = None,
        pixel_size: Optional[float] = None,
        **kwargs,
    ):
        """Compute a flat-sky angular power spectrum C_ℓ for convergence maps."""
        effective_field_size = field_size or self.field_size
        return compute_flat_cl(
            self,
            field_size=effective_field_size,
            pixel_size=pixel_size,
            **kwargs,
        )

    def get_shear(self, cosmo=None):
        """
        Compute shear (γ1, γ2) from convergence via Kaiser-Squires inversion.

        Parameters
        ----------
        cosmo : jax_cosmo.Cosmology, optional
            Cosmology object if needed for calculations

        Returns
        -------
        tuple[Array, Array]
            Shear components (γ1, γ2)

        Raises
        ------
        NotImplementedError
            This method is not yet implemented

        Notes
        -----
        Future implementation will use FFT-based Kaiser-Squires inversion
        to derive shear from convergence.
        """
        raise NotImplementedError("Shear computation coming soon")


class SphericalKappaField(SphericalDensity):
    """
    Convergence map in spherical (HEALPix) geometry.

    Inherits from SphericalDensity, adding lensing-specific methods for
    resampling, power spectrum computation, and shear derivation.

    Attributes
    ----------
    z_source : float | jnp.ndarray
        Source redshift(s) for the convergence map
    array : Array
        Convergence values, shape (npix,) or (n_sources, npix)
    """

    def __init__(
        self,
        *,
        array,
        mesh_size,
        box_size,
        observer_position=(0.5, 0.5, 0.5),
        sharding=None,
        nside=None,
        flatsky_npix=None,
        field_size=None,
        halo_size=0,
        z_source=None,
        status=DensityStatus.KAPPA,
        scale_factors=1.0,
    ):
        """
        Initialize SphericalKappaField with explicit metadata, mirroring DensityField.__init__.
        """
        super().__init__(
            array=array,
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

    @classmethod
    def FromDensityMetadata(
        cls,
        *,
        array,
        density_field: DensityField,
        status=DensityStatus.KAPPA,
        z_source=None,
        scale_factors=None,
    ) -> "SphericalKappaField":
        """
        Construct SphericalKappaField from a reference DensityField and convergence array.
        """
        return super().FromDensityMetadata(
            array=array,
            density_field=density_field,
            status=status,
            z_source=z_source,
            scale_factors=scale_factors,
        )

    def compute_power_spectrum(self, *, lmax: Optional[int] = None, **kwargs):
        """Compute a spherical angular power spectrum C_ℓ for convergence maps."""
        return compute_spherical_cl(self, lmax=lmax, **kwargs)

    def get_shear(self, cosmo=None):
        """
        Compute shear via spin-2 spherical harmonic transform.

        Parameters
        ----------
        cosmo : jax_cosmo.Cosmology, optional
            Cosmology object if needed for calculations

        Returns
        -------
        tuple[Array, Array]
            Shear components (E-mode, B-mode)

        Raises
        ------
        NotImplementedError
            This method is not yet implemented

        Notes
        -----
        Future implementation will use spin-weighted spherical harmonic
        transforms to derive shear from convergence on the sphere.
        """
        raise NotImplementedError("Shear computation coming soon")


class FlatShearField(FlatDensity):
    """
    Shear map (γ1, γ2) in flat-sky (Cartesian) geometry.
    """

    def __init__(
        self,
        *,
        array,
        mesh_size,
        box_size,
        observer_position=(0.5, 0.5, 0.5),
        sharding=None,
        nside=None,
        flatsky_npix=None,
        field_size=None,
        halo_size=0,
        z_source=None,
        status=DensityStatus.GAMMA,
        scale_factors=1.0,
    ):
        super().__init__(
            array=array,
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

    @classmethod
    def FromDensityMetadata(
        cls,
        *,
        array,
        density_field: DensityField,
        status=DensityStatus.GAMMA,
        z_source=None,
        scale_factors=None,
    ) -> "FlatShearField":
        """
        Construct FlatShearField from a reference DensityField and shear array.
        """
        return super().FromDensityMetadata(
            array=array,
            density_field=density_field,
            status=status,
            z_source=z_source,
            scale_factors=scale_factors,
        )

    def compute_power_spectrum(
        self,
        *,
        field_size: Optional[float] = None,
        pixel_size: Optional[float] = None,
        **kwargs,
    ):
        """Compute the shear flat-sky power spectrum (same as density maps)."""
        effective_field_size = field_size or self.field_size
        return compute_flat_cl(
            self,
            field_size=effective_field_size,
            pixel_size=pixel_size,
            **kwargs,
        )


class SphericalShearField(SphericalDensity):
    """
    Shear map (γ1, γ2) in spherical (HEALPix) geometry.
    """

    def __init__(
        self,
        *,
        array,
        mesh_size,
        box_size,
        observer_position=(0.5, 0.5, 0.5),
        sharding=None,
        nside=None,
        flatsky_npix=None,
        field_size=None,
        halo_size=0,
        z_source=None,
        status=DensityStatus.GAMMA,
        scale_factors=1.0,
    ):
        super().__init__(
            array=array,
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

    @classmethod
    def FromDensityMetadata(
        cls,
        *,
        array,
        density_field: DensityField,
        status=DensityStatus.GAMMA,
        z_source=None,
        scale_factors=None,
    ) -> "SphericalShearField":
        """
        Construct SphericalShearField from a reference DensityField and shear array.
        """
        return super().FromDensityMetadata(
            array=array,
            density_field=density_field,
            status=status,
            z_source=z_source,
            scale_factors=scale_factors,
        )

    def compute_power_spectrum(self, *, lmax: Optional[int] = None, **kwargs):
        """Compute the shear spherical power spectrum."""
        return compute_spherical_cl(self, lmax=lmax, **kwargs)
