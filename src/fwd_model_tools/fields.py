import jax
import jax.numpy as jnp
from jaxpm.distributed import fft3d, ifft3d, normal_field
from jaxpm.kernels import fftk
from numpyro.distributions import Normal, constraints
from numpyro.distributions.util import promote_shapes
from numpyro.util import is_prng_key


def linear_field(mesh_shape, box_size, pk, field):
    """
    Generate a linear matter field using input power spectrum.

    Parameters
    ----------
    mesh_shape : tuple
        Shape of the mesh (nx, ny, nz).
    box_size : tuple or list
        Physical size of the box in each dimension.
    pk : callable
        Power spectrum function P(k).
    field : jax.Array
        Input white noise field.

    Returns
    -------
    jax.Array
        Linear matter field in real space.
    """
    field = fft3d(field)
    kvec = fftk(field)
    kmesh = sum((kk / box_size[i] * mesh_shape[i])**2
                for i, kk in enumerate(kvec))**0.5
    pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (
        box_size[0] * box_size[1] * box_size[2])
    field = field * (pkmesh)**0.5
    return ifft3d(field)


def lognormal_field(mesh_shape, box_size, pk, field, shift=1.0):
    """
    Generate a lognormal density field from Gaussian white noise.

    The lognormal transformation provides a more realistic approximation of the
    matter density field in the mildly non-linear regime, compared to Gaussian fields.
    This follows the prescription from Coles & Jones (1991) and Xavier et al. (2016).

    The transformation is: δ_ln(x) = exp(δ_g(x) - σ²/2) - shift
    where δ_g is the Gaussian field and σ² is its variance.

    Parameters
    ----------
    mesh_shape : tuple
        Shape of the mesh (nx, ny, nz).
    box_size : tuple or list
        Physical size of the box in each dimension.
    pk : callable
        Power spectrum function P(k).
    field : jax.Array
        Input white noise field (Gaussian random field).
    shift : float, default=1.0
        Shift parameter to control the mean density offset. Default is 1.0,
        which ensures the mean density fluctuation is approximately zero.

    Returns
    -------
    jax.Array
        Lognormal density field in real space.

    References
    ----------
    - Coles, P., & Jones, B. (1991). A lognormal model for the cosmological
      mass distribution. MNRAS, 248, 1.
    - Xavier, H. S., et al. (2016). Improving lognormal models for cosmological
      fields. MNRAS, 459, 3693-3710. arXiv:1602.08503
    """
    field_fft = fft3d(field)
    kvec = fftk(field_fft)
    kmesh = sum((kk / box_size[i] * mesh_shape[i])**2
                for i, kk in enumerate(kvec))**0.5
    pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (
        box_size[0] * box_size[1] * box_size[2])

    gaussian_field = ifft3d(field_fft * (pkmesh)**0.5)

    sigma_sq = jnp.var(gaussian_field)

    lognormal = jnp.exp(gaussian_field - sigma_sq / 2.0) - shift

    return lognormal


class DistributedNormal(Normal):
    """
    Sharded normal distribution for distributed initial conditions.

    This class extends NumPyro's Normal distribution to support distributed
    sampling across multiple devices using JAX's sharding mechanism.

    Parameters
    ----------
    loc : float or jax.Array, default=0.0
        Mean of the normal distribution.
    scale : float or jax.Array, default=1.0
        Standard deviation of the normal distribution.
    sharding : jax.sharding.Sharding, optional
        Sharding specification for distributed sampling.
    validate_args : bool, optional
        Whether to validate input arguments.
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.real
    reparametrized_params = ["loc", "scale"]

    def __init__(self,
                 loc=0.0,
                 scale=1.0,
                 sharding=None,
                 *,
                 validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        self.sharding = sharding
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(loc),
                                               jnp.shape(scale))
        super(Normal, self).__init__(batch_shape=batch_shape,
                                     validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        """
        Sample from the distributed normal distribution.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        sample_shape : tuple, default=()
            Shape of samples to generate.

        Returns
        -------
        jax.Array
            Samples from the normal distribution.
        """
        assert is_prng_key(key)

        eps = normal_field(key,
                           sample_shape + self.batch_shape + self.event_shape,
                           self.sharding)

        return self.loc + eps * self.scale


class DistributedLogNormal(Normal):
    """
    Sharded lognormal distribution for distributed initial conditions.

    This class extends NumPyro's Normal distribution to support distributed
    sampling of lognormal fields across multiple devices using JAX's sharding mechanism.

    The samples are generated by first sampling from a normal distribution,
    then applying the lognormal transformation: exp(N(loc, scale)) - shift.

    Parameters
    ----------
    loc : float or jax.Array, default=0.0
        Mean of the underlying normal distribution (before exp transform).
    scale : float or jax.Array, default=1.0
        Standard deviation of the underlying normal distribution.
    shift : float, default=1.0
        Shift applied after the exponential transformation.
    sharding : jax.sharding.Sharding, optional
        Sharding specification for distributed sampling.
    validate_args : bool, optional
        Whether to validate input arguments.
    """

    arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
    support = constraints.positive
    reparametrized_params = ["loc", "scale"]

    def __init__(self,
                 loc=0.0,
                 scale=1.0,
                 shift=1.0,
                 sharding=None,
                 *,
                 validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        self.shift = shift
        self.sharding = sharding
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(loc),
                                               jnp.shape(scale))
        super(Normal, self).__init__(batch_shape=batch_shape,
                                     validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        """
        Sample from the distributed lognormal distribution.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        sample_shape : tuple, default=()
            Shape of samples to generate.

        Returns
        -------
        jax.Array
            Samples from the lognormal distribution.
        """
        assert is_prng_key(key)

        eps = normal_field(key,
                           sample_shape + self.batch_shape + self.event_shape,
                           self.sharding)

        gaussian_sample = self.loc + eps * self.scale

        sigma_sq = self.scale**2

        return jnp.exp(gaussian_sample - sigma_sq / 2.0) - self.shift
