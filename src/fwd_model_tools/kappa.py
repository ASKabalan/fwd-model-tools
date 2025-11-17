"""
Convergence (kappa) field classes for weak lensing.

This module provides FlatKappaField and SphericalKappaField classes
that inherit from FlatDensity and SphericalDensity, adding lensing-specific
methods like resampling, power spectrum computation, and shear derivation.
"""

import jax.numpy as jnp

from fwd_model_tools.field import FlatDensity, SphericalDensity, DensityStatus


class FlatKappaField(FlatDensity):
    """
    Convergence map in flat-sky (Cartesian) geometry.

    Inherits from FlatDensity, adding lensing-specific methods for
    resampling, power spectrum computation, and shear derivation.

    Attributes
    ----------
    z_source : float | jnp.ndarray
        Source redshift(s) for the convergence map
    array : jax.Array
        Convergence values, shape (ny, nx) or (n_sources, ny, nx)
    """

    def __init__(self, *, array, density_field, status=DensityStatus.KAPPA, z_source):
        """
        Initialize FlatKappaField.

        Parameters
        ----------
        array : jax.Array
            Convergence array
        density_field : DensityField
            Reference field with metadata
        status : DensityStatus, default=DensityStatus.KAPPA
            Field status
        z_source : float | jnp.ndarray
            Source redshift(s)
        """
        super().__init__(array=array, density_field=density_field, status=status)
        self.z_source = z_source

    def replace(self, **updates):
        """
        Override replace to handle z_source attribute.

        Returns
        -------
        FlatKappaField
            New instance with updated attributes
        """
        params = {
            "array": self.array,
            "mesh_size": self.mesh_size,
            "box_size": self.box_size,
            "observer_position": self.observer_position,
            "sharding": self.sharding,
            "nside": self.nside,
            "flatsky_npix": self.flatsky_npix,
            "field_size": self.field_size,
            "halo_size": self.halo_size,
            "status": self.status,
            "scale_factors": self.scale_factors,
            "z_source": self.z_source,
        }
        allowed_keys = set(params.keys())
        unknown = set(updates) - allowed_keys
        if unknown:
            raise TypeError(f"Unknown FlatKappaField attribute(s): {unknown}")
        params.update(updates)

        # Create using object.__new__ to bypass __init__
        instance = object.__new__(type(self))
        for key, value in params.items():
            setattr(instance, key, value)
        return instance

    def compute_power_spectrum(self, ell_bins, cosmo=None):
        """
        Compute angular power spectrum C_ℓ via 2D FFT.

        Parameters
        ----------
        ell_bins : jax.Array
            Multipole bins for power spectrum
        cosmo : jax_cosmo.Cosmology, optional
            Cosmology object if needed for calculations

        Returns
        -------
        jax.Array
            Power spectrum C_ℓ

        Raises
        ------
        NotImplementedError
            This method is not yet implemented

        Notes
        -----
        Future implementation will use 2D FFT to compute the power spectrum
        in flat-sky approximation.
        """
        raise NotImplementedError("Power spectrum computation coming soon")

    def get_shear(self, cosmo=None):
        """
        Compute shear (γ1, γ2) from convergence via Kaiser-Squires inversion.

        Parameters
        ----------
        cosmo : jax_cosmo.Cosmology, optional
            Cosmology object if needed for calculations

        Returns
        -------
        tuple[jax.Array, jax.Array]
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
    array : jax.Array
        Convergence values, shape (npix,) or (n_sources, npix)
    """

    def __init__(self, *, array, density_field, status=DensityStatus.KAPPA, z_source):
        """
        Initialize SphericalKappaField.

        Parameters
        ----------
        array : jax.Array
            Convergence array
        density_field : DensityField
            Reference field with metadata
        status : DensityStatus, default=DensityStatus.KAPPA
            Field status
        z_source : float | jnp.ndarray
            Source redshift(s)
        """
        super().__init__(array=array, density_field=density_field, status=status)
        self.z_source = z_source

    def replace(self, **updates):
        """
        Override replace to handle z_source attribute.

        Returns
        -------
        SphericalKappaField
            New instance with updated attributes
        """
        params = {
            "array": self.array,
            "mesh_size": self.mesh_size,
            "box_size": self.box_size,
            "observer_position": self.observer_position,
            "sharding": self.sharding,
            "nside": self.nside,
            "flatsky_npix": self.flatsky_npix,
            "field_size": self.field_size,
            "halo_size": self.halo_size,
            "status": self.status,
            "scale_factors": self.scale_factors,
            "z_source": self.z_source,
        }
        allowed_keys = set(params.keys())
        unknown = set(updates) - allowed_keys
        if unknown:
            raise TypeError(f"Unknown SphericalKappaField attribute(s): {unknown}")
        params.update(updates)

        # Create using object.__new__ to bypass __init__
        instance = object.__new__(type(self))
        for key, value in params.items():
            setattr(instance, key, value)
        return instance

    def compute_power_spectrum(self, ell_bins, cosmo=None):
        """
        Compute angular power spectrum C_ℓ via healpy.anafast.

        Parameters
        ----------
        ell_bins : jax.Array
            Multipole bins for power spectrum
        cosmo : jax_cosmo.Cosmology, optional
            Cosmology object if needed for calculations

        Returns
        -------
        jax.Array
            Power spectrum C_ℓ

        Raises
        ------
        NotImplementedError
            This method is not yet implemented

        Notes
        -----
        Future implementation will use healpy.anafast or spherical harmonic
        transforms to compute the power spectrum on the sphere.
        """
        raise NotImplementedError("Power spectrum computation coming soon")

    def get_shear(self, cosmo=None):
        """
        Compute shear via spin-2 spherical harmonic transform.

        Parameters
        ----------
        cosmo : jax_cosmo.Cosmology, optional
            Cosmology object if needed for calculations

        Returns
        -------
        tuple[jax.Array, jax.Array]
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
