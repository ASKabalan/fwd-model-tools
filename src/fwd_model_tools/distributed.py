import pickle
from pathlib import Path

import jax
import orbax.checkpoint as ocp
from jax.sharding import NamedSharding


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
    return tuple(
        [get_axis_size(sharding, i) for i in range(len(sharding.spec))])


def save_sharded(pytree, path, overwrite=True, dump_structure=True):
    """
    Saves a sharded PyTree to disk using Orbax Checkpoint.

    This function saves a distributed JAX PyTree (dict/array/nested structure) to disk,
    preserving sharding information. This allows for efficient storage and loading of
    large distributed data structures.

    Parameters
    ----------
    pytree : PyTree
        The sharded PyTree to save (can be dict, array, or nested structure).
    path : str or Path
        Directory path where checkpoint will be saved.
    overwrite : bool, default=True
        If True, overwrite existing checkpoint. If False, raise error if path exists.
    dump_structure : bool, default=True
        If True, save structure pickle file. If False, skip structure dump (useful when
        structure will be reconstructed at load time to avoid Device pickle issues).

    Notes
    -----
    - Uses Orbax AsyncCheckpointer for efficient saving
    - Preserves sharding information automatically
    - All processes must call this function (collective operation)
    """
    structure_path = Path(f"{path}_structure.pkl").absolute()
    path = Path(path).absolute()
    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())

    checkpointer.save(path,
                      args=ocp.args.StandardSave(pytree),
                      force=overwrite)
    checkpointer.wait_until_finished()

    if dump_structure:

        def to_shape_dtype_struct_safe(x):
            if hasattr(x, "shape") and hasattr(x, "dtype"):
                return ocp.utils.to_shape_dtype_struct(x)
            else:
                return x

        def strip_sharding(x):
            if hasattr(x, "shape") and hasattr(x, "dtype"):
                return jax.ShapeDtypeStruct(x.shape, x.dtype)
            else:
                return x

        abstract_pytree = jax.tree.map(to_shape_dtype_struct_safe, pytree)
        abstract_pytree_no_sharding = jax.tree.map(strip_sharding,
                                                   abstract_pytree)

        with open(structure_path, "wb") as f:
            pickle.dump(abstract_pytree_no_sharding, f)


def load_sharded(path, abstract_pytree=None):
    """
    Loads a sharded PyTree from disk using Orbax Checkpoint.

    This function reconstructs a distributed JAX PyTree from a checkpoint saved by
    save_sharded(). It automatically restores the correct sharding configuration.

    Parameters
    ----------
    path : str or Path
        Directory path where checkpoint is stored.
    abstract_pytree : PyTree, optional
        Abstract structure with shape/dtype/sharding information for restoration.
        If provided, uses this structure directly (useful to avoid Device pickle issues).
        If None, loads structure from pickled file (default behavior).
        Can be created with: jax.tree.map(ocp.utils.to_shape_dtype_struct, pytree)

    Returns
    -------
    PyTree
        The reconstructed sharded PyTree with the same structure as the original.

    Notes
    -----
    - Uses Orbax AsyncCheckpointer for efficient loading
    - Automatically restores sharding information
    - All processes must call this function (collective operation)
    """
    path = Path(path).absolute()

    if abstract_pytree is None:
        structure_path = Path(f"{path}_structure.pkl").absolute()
        with open(structure_path, "rb") as f:
            abstract_pytree = pickle.load(f)

    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    restored = checkpointer.restore(
        path, args=ocp.args.StandardRestore(abstract_pytree))
    checkpointer.wait_until_finished()
    return restored
