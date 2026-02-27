"""Distributed probability distributions used in sampling workflows."""

import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.scipy.special import ndtr, ndtri
from jaxpm.distributed import fft3d, ifft3d, normal_field
from jaxpm.kernels import fftk, interpolate_power_spectrum
from jaxtyping import Key
from numpyro.distributions import Normal, TransformedDistribution, constraints
from numpyro.distributions.transforms import AffineTransform, Transform
from numpyro.distributions.util import promote_shapes
from numpyro.util import is_prng_key

from ..fields import (
    AbstractField,
    DensityField,
    DensityUnit,
    FieldStatus,
    FlatDensity,
    FlatKappaField,
    SphericalDensity,
    SphericalKappaField,
)


class PowerSpectrumTransform(Transform):
    """
    NumPyro Transform that deterministically maps N(0,1) white noise
    to a cosmological density field via Fourier-space interpolation.
    """

    domain = constraints.real
    codomain = constraints.real

    def __init__(self, mesh_size, box_size, cosmo=None, pk_fn=None, sharding=None):
        self.mesh_size = mesh_size
        self.box_size = box_size
        self.cosmo = cosmo
        self.pk_fn = pk_fn
        self.sharding = sharding

    def _get_pkmesh(self, field):
        """Helper to run your exact power spectrum logic."""
        if self.pk_fn is None:
            if self.cosmo is None:
                raise ValueError("Either pk_fn or cosmo must be provided to compute the power spectrum.")

            # We import here to avoid global dependencies if pk_fn is provided
            k = jnp.logspace(-4, 1, 256)
            pk = jc.power.linear_matter_power(self.cosmo, k)

            pk_fn = lambda x: interpolate_power_spectrum(x, k, pk, self.sharding)
        else:
            pk_fn = self.pk_fn

        kvec = fftk(field)
        kmesh = sum((kk / self.box_size[i] * self.mesh_size[i]) ** 2 for i, kk in enumerate(kvec)) ** 0.5

        pkmesh = (
            pk_fn(kmesh)
            * (self.mesh_size[0] * self.mesh_size[1] * self.mesh_size[2])
            / (self.box_size[0] * self.box_size[1] * self.box_size[2])
        )

        # Safeguard: The k=0 mode (mean density) often has P(k)=0.
        # We replace it with 1.0 safely to prevent NaN gradients or division by zero.
        pkmesh = jnp.where(kmesh == 0.0, 1.0, pkmesh)
        return pkmesh

    def __call__(self, x):
        """Forward transform: White Noise -> Density Field"""
        is_field = isinstance(x, DensityField)
        arr = x.array if is_field else x
        field = fft3d(arr)
        pkmesh = self._get_pkmesh(field)
        field = field * jnp.sqrt(pkmesh)
        result = ifft3d(field).real
        return x.replace(array=result) if is_field else result

    def _inverse(self, y):
        """Inverse transform: Density Field -> White Noise"""
        is_field = isinstance(y, DensityField)
        arr = y.array if is_field else y
        field = fft3d(arr)
        pkmesh = self._get_pkmesh(field)
        field = field / jnp.sqrt(pkmesh)
        result = ifft3d(field).real
        return y.replace(array=result) if is_field else result

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        """
        The determinant of the linear Fourier transform.
        Required mathematically by NumPyro, but ignored by NUTS when reparameterized!
        """
        arr = x.array if isinstance(x, DensityField) else x
        field = fft3d(arr)
        pkmesh = self._get_pkmesh(field)
        return 0.5 * jnp.sum(jnp.log(pkmesh))

    # ---------------------------------------------------------
    # PyTree registration to survive JAX compilation
    # ---------------------------------------------------------
    def tree_flatten(self):
        # Cosmo goes in params because it is traced by MCMC.
        # Everything else is static metadata.
        return (self.cosmo,), (self.mesh_size, self.box_size, self.pk_fn, self.sharding)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        (cosmo,) = params
        mesh_size, box_size, pk_fn, sharding = aux_data
        return cls(mesh_size, box_size, cosmo, pk_fn, sharding)


class DistributedIC(TransformedDistribution):
    """
    Cosmological Initial Conditions Distribution.

    Subclasses TransformedDistribution to naturally bridge DistributedNormal(0,1)
    with the Fourier-space power spectrum geometry.
    """

    arg_constraints = {}

    def __init__(
        self,
        mesh_size,
        box_size,
        observer_position=(0.5, 0.5, 0.5),
        halo_size=(0, 0),
        flatsky_npix=None,
        nside=None,
        field_size=None,
        cosmo=None,
        pk_fn=None,
        sharding=None,
        validate_args=None,
    ):
        self.mesh_size = mesh_size
        self.box_size = box_size
        self.observer_position = observer_position
        self.halo_size = halo_size
        self.flatsky_npix = flatsky_npix
        self.nside = nside
        self.field_size = field_size
        self.cosmo = cosmo
        self.sharding = sharding

        # 1. Base is the sharded N(0,1) noise
        base_dist = DistributedNormal(
            loc=0.0,
            scale=1.0,
            mesh_size=mesh_size,
            box_size=box_size,
            observer_position=observer_position,
            halo_size=halo_size,
            flatsky_npix=flatsky_npix,
            nside=nside,
            field_size=field_size,
            sharding=sharding,
            field_type="density",
        )

        # 2. Transform applies the cosmology and FFT interpolation
        transform = PowerSpectrumTransform(
            mesh_size=mesh_size,
            box_size=box_size,
            cosmo=cosmo,
            pk_fn=pk_fn,
            sharding=sharding,
        )

        super().__init__(base_dist, [transform], validate_args=validate_args)


# ==========================================
# 1. Distributed White Noise
# ==========================================

_FIELD_CLS = {
    "density": DensityField,
    "flat": FlatDensity,
    "spherical": SphericalDensity,
    "flat_kappa": FlatKappaField,
    "spherical_kappa": SphericalKappaField,
}


class DistributedNormal(Normal):
    """
    Sharded normal distribution for distributed initial conditions and kappa maps.

    The ``field_type`` parameter controls what ``sample()`` returns:

    - ``"density"`` (default): returns a ``DensityField`` (3D IC mesh).
    - ``"flatsky"``: returns a ``FlatDensity`` (2D flat-sky map).
    - ``"spherical"``: returns a ``SphericalDensity`` (1D HEALPix map).
    - ``None``: returns a plain JAX array (for observation sites).

    Sharding is automatically trimmed to the array's rank via
    ``get_sharding_for_shape``, so a 3-D ``P('x','y')`` spec works
    correctly for 1-D HEALPix arrays without manual adjustment.
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    reparametrized_params = ["loc", "scale"]

    def __init__(
        self,
        loc=0.0,
        scale=1.0,
        mesh_size=None,
        box_size=None,
        observer_position=None,
        halo_size=None,
        flatsky_npix=None,
        nside=None,
        field_size=None,
        sharding=None,
        field_type=None,
        *,
        validate_args=None,
    ):
        is_loc_field = isinstance(loc, AbstractField)
        is_scale_field = isinstance(scale, AbstractField)
        # 1. Strict Type Guarding
        if is_scale_field and not is_loc_field:
            raise ValueError(
                "If scale is a field, loc must also be a field. (loc=array, scale=field is not supported)."
            )

        if is_loc_field:
            if is_scale_field and type(loc) is not type(scale):
                raise ValueError(f"scale field type {type(scale)} does not match loc field type {type(loc)}.")

            # 2. Infer and Validate field_type
            inferred_type = next((k for k, v in _FIELD_CLS.items() if type(loc) is v), None)
            if field_type is not None and field_type != inferred_type:
                raise ValueError(
                    f"Explicit field_type '{field_type}' does not match inferred type '{inferred_type}' from loc."
                )
            field_type = inferred_type

            # 3. Enforce all explicit metadata args are None
            metadata_args = {
                "mesh_size": mesh_size,
                "box_size": box_size,
                "observer_position": observer_position,
                "halo_size": halo_size,
                "flatsky_npix": flatsky_npix,
                "nside": nside,
                "field_size": field_size,
                "sharding": sharding,
            }
            for arg_name, arg_val in metadata_args.items():
                if arg_val is not None:
                    raise ValueError(
                        f"Metadata argument '{arg_name}' must be None when 'loc' is a field. Metadata is extracted directly from the field."
                    )

            # 4. Extract metadata directly from the field
            mesh_size = loc.mesh_size
            box_size = loc.box_size
            observer_position = loc.observer_position
            halo_size = loc.halo_size
            flatsky_npix = loc.flatsky_npix
            nside = loc.nside
            field_size = loc.field_size
            sharding = loc.sharding

        else:
            # Apply standard defaults if loc is just an array
            if field_type is None:
                field_type = "density"
            if observer_position is None:
                observer_position = (0.5, 0.5, 0.5)
            if halo_size is None:
                halo_size = (0, 0)

        # 5. Shape Promotion (safely preserves AbstractFields if present)
        self.loc, self.scale = promote_shapes(loc, scale)

        # 6. Scalar Broadcasting (Skipped safely if loc is an AbstractField)
        if not is_loc_field and (jnp.isscalar(self.loc) or (self.loc.ndim == 0 and self.scale.ndim == 0)):
            if field_type == "density":
                assert mesh_size is not None, "mesh_size must be provided for density fields"
                self.loc = jnp.broadcast_to(self.loc, mesh_size)
                self.scale = jnp.broadcast_to(self.scale, mesh_size)
            elif field_type == "flat" or field_type == "flat_kappa":
                assert flatsky_npix is not None, "flatsky_npix must be provided for flatsky fields"
                self.loc = jnp.broadcast_to(self.loc, flatsky_npix)
                self.scale = jnp.broadcast_to(self.scale, flatsky_npix)
            elif field_type == "spherical" or field_type == "spherical_kappa":
                assert nside is not None, "nside must be provided for spherical fields"
                npix = 12 * nside**2
                self.loc = jnp.broadcast_to(self.loc, (npix,))
                self.scale = jnp.broadcast_to(self.scale, (npix,))
        else:
            self.loc = jnp.asarray(self.loc)
            self.scale = jnp.asarray(self.scale)

        # 7. Store final resolved metadata
        self.mesh_size = mesh_size
        self.box_size = box_size
        self.observer_position = observer_position
        self.halo_size = halo_size
        self.flatsky_npix = flatsky_npix
        self.nside = nside
        self.field_size = field_size
        self.sharding = sharding
        self.field_type = field_type

        super().__init__(self.loc, self.scale, validate_args=validate_args)

    def sample(self, key: Key, sample_shape=()):
        assert is_prng_key(key)
        eps_shape = sample_shape + self.batch_shape + self.event_shape
        eps = normal_field(key, eps_shape, self.sharding)
        arr = self.loc + eps * self.scale

        cls = _FIELD_CLS.get(self.field_type)
        if cls is None:
            return arr
        return cls(
            array=arr,
            mesh_size=self.mesh_size,
            box_size=self.box_size,
            observer_position=self.observer_position,
            halo_size=self.halo_size,
            flatsky_npix=self.flatsky_npix,
            nside=self.nside,
            field_size=self.field_size,
            status=FieldStatus.INITIAL_FIELD,
            unit=DensityUnit.DENSITY,
            sharding=self.sharding,
        )

    def log_prob(self, value):
        if isinstance(value, AbstractField):
            value = value.array
        return super().log_prob(value)

    # ---------------------------------------------------------
    # PyTree registration to survive MCMC compilation
    # ---------------------------------------------------------
    def tree_flatten(self):
        return (self.loc, self.scale), (
            self.mesh_size,
            self.box_size,
            self.observer_position,
            self.halo_size,
            self.flatsky_npix,
            self.nside,
            self.field_size,
            self.sharding,
            self.field_type,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        loc, scale = params
        (
            mesh_size,
            box_size,
            observer_position,
            halo_size,
            flatsky_npix,
            nside,
            field_size,
            sharding,
            field_type,
        ) = aux_data
        return cls(
            loc=loc,
            scale=scale,
            mesh_size=mesh_size,
            box_size=box_size,
            observer_position=observer_position,
            halo_size=halo_size,
            flatsky_npix=flatsky_npix,
            nside=nside,
            field_size=field_size,
            sharding=sharding,
            field_type=field_type,
        )


# ==========================================
# 2. Preconditioned Uniform boundaries
# ==========================================


class ProbitTransform(Transform):
    domain = constraints.real
    codomain = constraints.unit_interval

    def __call__(self, x):
        return ndtr(x)

    def _inverse(self, y):
        return ndtri(y)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return -0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.square(x)

    def tree_flatten(self):
        return (), ((), {})

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        return cls()


class PreconditionnedUniform(TransformedDistribution):
    """
    Uniform distribution explicitly backed by a Gaussian space for HMC preconditioning.
    NUTS traverses a standard Normal space, avoiding hard boundaries.
    """

    arg_constraints = {"low": constraints.real, "high": constraints.real}
    reparametrized_params = ["low", "high"]

    def __init__(self, low=0.0, high=1.0, validate_args=None):
        self.low, self.high = promote_shapes(low, high)
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(self.low), jnp.shape(self.high))
        base_dist = Normal(0.0, 1.0).expand(batch_shape)
        transforms = [ProbitTransform(), AffineTransform(loc=self.low, scale=self.high - self.low)]
        super().__init__(base_dist, transforms, validate_args=validate_args)

    @property
    def support(self):
        return constraints.interval(self.low, self.high)
