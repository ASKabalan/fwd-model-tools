"""Orbax-based helpers to save and load sharded PyTrees."""

import pickle
from pathlib import Path

import jax
import orbax.checkpoint as ocp


def save_sharded(pytree, path, overwrite: bool = True, dump_structure: bool = True):
    """
    Saves a (possibly sharded) PyTree to disk using Orbax Checkpoint.

    Parameters
    ----------
    pytree : PyTree
        The PyTree to save (can be dict, array, or nested structure).
    path : str or Path
        Directory path where checkpoint will be saved.
    overwrite : bool, default=True
        If True, overwrite existing checkpoint. If False, raise error if path exists.
    dump_structure : bool, default=True
        If True, save a separate pickled abstract structure file.
    """
    structure_path = Path(f"{path}_structure.pkl").absolute()
    path = Path(path).absolute()
    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())

    checkpointer.save(path, args=ocp.args.StandardSave(pytree), force=overwrite)
    checkpointer.wait_until_finished()

    if dump_structure:

        def to_shape_dtype_struct_safe(x):
            if hasattr(x, "shape") and hasattr(x, "dtype"):
                return ocp.utils.to_shape_dtype_struct(x)
            return x

        def strip_sharding(x):
            if hasattr(x, "shape") and hasattr(x, "dtype"):
                return jax.ShapeDtypeStruct(x.shape, x.dtype)
            return x

        abstract_pytree = jax.tree.map(to_shape_dtype_struct_safe, pytree)
        abstract_pytree_no_sharding = jax.tree.map(strip_sharding, abstract_pytree)

        with open(structure_path, "wb") as f:
            pickle.dump(abstract_pytree_no_sharding, f)


def load_sharded(path, abstract_pytree=None):
    """
    Loads a (possibly sharded) PyTree from disk using Orbax Checkpoint.

    Parameters
    ----------
    path : str or Path
        Directory path where checkpoint is stored.
    abstract_pytree : PyTree, optional
        Abstract structure with shape/dtype/sharding information for restoration.

    Returns
    -------
    PyTree
        The reconstructed PyTree with the same structure as the original.
    """
    path = Path(path).absolute()

    if abstract_pytree is None:
        structure_path = Path(f"{path}_structure.pkl").absolute()
        with open(structure_path, "rb") as f:
            abstract_pytree = pickle.load(f)

    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    restored = checkpointer.restore(path, args=ocp.args.StandardRestore(abstract_pytree))
    checkpointer.wait_until_finished()
    return restored
