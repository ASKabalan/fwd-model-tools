"""Distributed probability distributions used in sampling workflows."""

import jax
import jax.numpy as jnp
from jaxpm.distributed import normal_field
from jaxtyping import Key
from jax.scipy.special import ndtr
from numpyro.distributions import Normal, Uniform, constraints
from numpyro.distributions.util import promote_shapes
from numpyro.util import is_prng_key


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

    def __init__(self, loc=0.0, scale=1.0, sharding=None, *, validate_args=None):
        self.loc, self.scale = promote_shapes(loc, scale)
        self.sharding = sharding
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
        super(Normal, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key: Key, sample_shape=()):
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

        eps = normal_field(
            key,
            sample_shape + self.batch_shape + self.event_shape,
            self.sharding,
        )

        return self.loc + eps * self.scale


class PreconditionnedUniform(Uniform):
    """
    Uniform distribution backed by Gaussian sampling for HMC preconditioning.

    Samples z ~ Normal(0, 1), transforms via CDF: u = ndtr(z),
    then scales to [low, high]. NUTS sees smooth Gaussian density
    instead of hard uniform boundaries.

    Parameters
    ----------
    low : float or jax.Array, default=0.0
        Lower bound of the uniform distribution.
    high : float or jax.Array, default=1.0
        Upper bound of the uniform distribution.
    validate_args : bool, optional
        Whether to validate input arguments.
    """

    arg_constraints = {"low": constraints.real, "high": constraints.real}
    reparametrized_params = ["low", "high"]

    def __init__(self, low=0.0, high=1.0, validate_args=None):
        self.low, self.high = promote_shapes(low, high)
        batch_shape = jax.lax.broadcast_shapes(jnp.shape(self.low), jnp.shape(self.high))
        super(Uniform, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        """
        Sample from the preconditioned uniform distribution.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        sample_shape : tuple, default=()
            Shape of samples to generate.

        Returns
        -------
        jax.Array
            Samples from the uniform distribution.
        """
        shape = sample_shape + self.batch_shape
        z = jax.random.normal(key, shape=shape)
        u = ndtr(z)
        return self.low + u * (self.high - self.low)
