from jax.sharding import NamedSharding

from fwd_model_tools.sampling.persistency import load_sharded as _load_sharded
from fwd_model_tools.sampling.persistency import save_sharded as _save_sharded


def get_axis_size(sharding: NamedSharding, index: int) -> int:
    """
    Returns the size of the axis for a given sharding spec.

    Parameters
    ----------
    sharding : NamedSharding
        The sharding specification.
    index : int
        The index of the axis.

    Returns
    -------
    int
        The size of the axis.
    """
    axis_name = sharding.spec[index]
    if axis_name is None:
        return 1
    else:
        return sharding.mesh.shape[sharding.spec[index]]


def get_pdims_from_sharding(sharding: NamedSharding):
    """
    Returns the processor dimensions from a sharding specification.

    Parameters
    ----------
    sharding : NamedSharding
        The sharding specification.

    Returns
    -------
    tuple
        A tuple of processor dimensions.
    """
    return tuple([get_axis_size(sharding, i) for i in range(len(sharding.spec))])


def save_sharded(pytree, path, overwrite=True, dump_structure=True):
    """Shim around fwd_model_tools.sampling.persistency.save_sharded."""
    return _save_sharded(pytree, path, overwrite=overwrite, dump_structure=dump_structure)


def load_sharded(path, abstract_pytree=None):
    """Shim around fwd_model_tools.sampling.persistency.load_sharded."""
    return _load_sharded(path, abstract_pytree=abstract_pytree)
