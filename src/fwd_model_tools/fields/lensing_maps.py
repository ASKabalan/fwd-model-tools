"""
Convergence (kappa) and shear (gamma) field classes for weak lensing.

This module provides Flat/SphericalKappaField and Flat/SphericalShearField
classes that inherit from the density field stack and carry lensing-specific
metadata such as source redshifts.
"""

from typing import Optional

import jax.numpy as jnp

from fwd_model_tools.power import compute_flat_cl, compute_spherical_cl

from .density import DensityStatus, FlatDensity, SphericalDensity

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

    def __init__(self, *, array, density_field, status=DensityStatus.KAPPA, z_source):
        """
        Initialize FlatKappaField.

        Parameters
        ----------
        array : Array
            Convergence array
        density_field : DensityField
            Reference field with metadata
        status : DensityStatus, default=DensityStatus.KAPPA
            Field status
        z_source : float | jnp.ndarray
            Source redshift(s)
        """
        super().__init__(
            array=array,
            density_field=density_field,
            status=status,
            z_source=z_source,
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

    def __init__(self, *, array, density_field, status=DensityStatus.KAPPA, z_source):
        """
        Initialize SphericalKappaField.

        Parameters
        ----------
        array : Array
            Convergence array
        density_field : DensityField
            Reference field with metadata
        status : DensityStatus, default=DensityStatus.KAPPA
            Field status
        z_source : float | jnp.ndarray
            Source redshift(s)
        """
        super().__init__(
            array=array,
            density_field=density_field,
            status=status,
            z_source=z_source,
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
        density_field,
        status=DensityStatus.GAMMA,
        z_source,
    ):
        super().__init__(
            array=array,
            density_field=density_field,
            status=status,
            z_source=z_source,
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
        density_field,
        status=DensityStatus.GAMMA,
        z_source,
    ):
        super().__init__(
            array=array,
            density_field=density_field,
            status=status,
            z_source=z_source,
        )

    def compute_power_spectrum(self, *, lmax: Optional[int] = None, **kwargs):
        """Compute the shear spherical power spectrum."""
        return compute_spherical_cl(self, lmax=lmax, **kwargs)
