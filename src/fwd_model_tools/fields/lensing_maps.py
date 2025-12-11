from __future__ import annotations

"""
Convergence (kappa) and shear (gamma) field classes for weak lensing.

This module provides Flat/SphericalKappaField and Flat/SphericalShearField
classes that inherit from the density/lightcone stack and carry
lensing-specific metadata such as source redshifts.
"""

from typing import Any

from .lightcone import FlatDensity, SphericalDensity
from .units import ConvergenceUnit, convert_units
import jax

__all__ = [
    "FlatKappaField",
    "SphericalKappaField",
    "FlatShearField",
    "SphericalShearField",
]


# --------------------------------------------------------------------------- #
# Flat-sky convergence                                                        #
# --------------------------------------------------------------------------- #
@jax.tree_util.register_pytree_node_class
class FlatKappaField(FlatDensity):
    """
    Convergence map in flat-sky (Cartesian) geometry.

    Inherits from FlatDensity, adding lensing-specific methods for
    shear derivation.

    Attributes
    ----------
    z_sources : float | jnp.ndarray | Any
        Source redshift(s) for the convergence map.
    array : Array
        Convergence values, shape (ny, nx) or (n_planes, ny, nx).
    """

    def to(
        self,
        unit: ConvergenceUnit,
    ) -> FlatKappaField:
        """
        Convert convergence units (currently a no-op, but wired via convert_units).

        Both DIMENSIONLESS and EFFECTIVE_DENSITY are numerically κ = Σ/Σ_crit;
        this is mostly semantic for now.
        """
        if self.unit == unit:
            return self

        new_array = convert_units(
            array=self.array,
            origin=self.unit,
            destination=unit,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            # convergence conversions currently ignore these
            h=None,
            omega_m=None,
            mean_density=None,
            volume_element=None,
            sharding=self.sharding,
        )
        return self.replace(array=new_array, unit=unit)

    def get_shear(self, cosmo: Any | None = None):
        """
        Compute shear (γ1, γ2) from convergence via Kaiser-Squires inversion.
        """
        raise NotImplementedError("Shear computation from kappa (flat-sky) not implemented yet.")


# --------------------------------------------------------------------------- #
# Spherical convergence                                                       #
# --------------------------------------------------------------------------- #
@jax.tree_util.register_pytree_node_class
class SphericalKappaField(SphericalDensity):
    """
    Convergence map in spherical (HEALPix) geometry.

    Inherits from SphericalDensity, adding lensing-specific methods for
    shear derivation.

    Attributes
    ----------
    z_sources : float | jnp.ndarray | Any
        Source redshift(s) for the convergence map.
    array : Array
        Convergence values, shape (npix,) or (n_planes, npix).
    """

    def to(
        self,
        unit: ConvergenceUnit,
    ) -> SphericalKappaField:
        """
        Convert convergence units (currently a no-op, but wired via convert_units).
        """
        if self.unit == unit:
            return self

        new_array = convert_units(
            array=self.array,
            origin=self.unit,
            destination=unit,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            h=None,
            omega_m=None,
            mean_density=None,
            volume_element=None,
            sharding=self.sharding,
        )
        return self.replace(array=new_array, unit=unit)

    def get_shear(self, cosmo: Any | None = None):
        """
        Compute shear via spin-2 spherical harmonic transform.
        """
        raise NotImplementedError("Shear computation from kappa (spherical) not implemented yet.")


# --------------------------------------------------------------------------- #
# Flat-sky shear                                                              #
# --------------------------------------------------------------------------- #
@jax.tree_util.register_pytree_node_class
class FlatShearField(FlatDensity):
    """
    Shear map (γ1, γ2) in flat-sky (Cartesian) geometry.

    By convention you can store γ1, γ2 either as:
      - separate FlatShearField instances, or
      - an extra leading / trailing dimension in `array`.
    """

    def to(
        self,
        unit: ConvergenceUnit,
    ) -> FlatShearField:
        """
        Convert shear units (currently just semantic – same as convergence units).
        """
        if self.unit == unit:
            return self

        new_array = convert_units(
            array=self.array,
            origin=self.unit,
            destination=unit,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            h=None,
            omega_m=None,
            mean_density=None,
            volume_element=None,
            sharding=self.sharding,
        )
        return self.replace(array=new_array, unit=unit)

    def get_convergence(self, cosmo: Any | None = None):
        """
        Compute convergence from shear via Kaiser-Squires inversion.
        """
        raise NotImplementedError("Convergence from shear (flat-sky) not implemented yet.")


# --------------------------------------------------------------------------- #
# Spherical shear                                                             #
# --------------------------------------------------------------------------- #
@jax.tree_util.register_pytree_node_class
class SphericalShearField(SphericalDensity):
    """
    Shear map (γ1, γ2) in spherical (HEALPix) geometry.
    """

    def to(
        self,
        unit: ConvergenceUnit,
    ) -> SphericalShearField:
        """
        Convert shear units (currently just semantic – same as convergence units).
        """
        if self.unit == unit:
            return self

        new_array = convert_units(
            array=self.array,
            origin=self.unit,
            destination=unit,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            h=None,
            omega_m=None,
            mean_density=None,
            volume_element=None,
            sharding=self.sharding,
        )
        return self.replace(array=new_array, unit=unit)

    def get_convergence(self, cosmo: Any | None = None):
        """
        Compute convergence from shear via spin-2 spherical harmonic transform.
        """
        raise NotImplementedError("Convergence from shear (spherical) not implemented yet.")
