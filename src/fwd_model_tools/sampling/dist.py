"""Distributed probability distributions used in sampling workflows."""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Key
from numpyro.distributions import Normal, constraints
from numpyro.distributions.util import promote_shapes
from numpyro.util import is_prng_key

from jaxpm.distributed import normal_field


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

