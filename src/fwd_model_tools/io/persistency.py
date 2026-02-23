"""Orbax-based helpers to save and load sharded PyTrees."""

import pickle
import warnings
from collections.abc import Callable
from pathlib import Path

import jax
import orbax.checkpoint as ocp


def _get_default_sharding() -> jax.sharding.SingleDeviceSharding:
    """Return default single-device sharding for jax.devices()[0]."""
    return jax.sharding.SingleDeviceSharding(jax.devices()[0])


def _apply_sharding_to_abstract_pytree(abstract_pytree, sharded_leaves, sharding):
    """Apply sharding only to sharded leaves in the abstract pytree."""
    """ All others get _get_default_sharding() """

    def apply_sharding(struct, is_sharded):
        if is_sharded:
            return jax.ShapeDtypeStruct(struct.shape, struct.dtype, sharding=sharding)
        else:
            return jax.ShapeDtypeStruct(struct.shape, struct.dtype, sharding=_get_default_sharding())

    return jax.tree.map(apply_sharding, abstract_pytree, sharded_leaves)


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

        def get_sharded_leaves(x):
            return (isinstance(x, jax.Array) and x.sharding is not None
                    and not x.sharding.is_equivalent_to(_get_default_sharding(), 1))

        abstract_pytree = jax.tree.map(to_shape_dtype_struct_safe, pytree)
        abstract_pytree_no_sharding = jax.tree.map(strip_sharding, abstract_pytree)
        sharded_leaves = jax.tree.map(get_sharded_leaves, pytree)

        with open(structure_path, "wb") as f:
            pickle.dump((abstract_pytree_no_sharding, sharded_leaves), f)


def load_sharded(
    path,
    abstract_pytree=None,
    sharding: jax.sharding.Sharding | Callable[[jax.ShapeDtypeStruct], jax.sharding.Sharding] | None = None,
):
    """
    Loads a (possibly sharded) PyTree from disk using Orbax Checkpoint.

    Parameters
    ----------
    path : str or Path
        Directory path where checkpoint is stored.
    abstract_pytree : PyTree, optional
        Abstract structure with shape/dtype information for restoration.
        If None, loads from the companion structure file.
    sharding : jax.sharding.Sharding, Callable, or None, optional
        Target sharding for restored arrays:
        - None (default): Use SingleDeviceSharding on jax.devices()[0]
        - jax.sharding.Sharding: Apply this sharding to all arrays
        - Callable[[jax.ShapeDtypeStruct], jax.sharding.Sharding]: Function
          that returns sharding for each leaf based on its shape/dtype

    Returns
    -------
    PyTree
        The reconstructed PyTree with the specified sharding.

    Notes
    -----
    When loading checkpoints saved on a different device topology (e.g., saved
    on 8 CPUs, loaded on 1 GPU), you must specify the target sharding to ensure
    correct restoration. The default behavior places all arrays on the first
    available device.

    Examples
    --------
    >>> # Load with default single-device sharding
    >>> data = load_sharded("checkpoint_path")

    >>> # Load with explicit sharding
    >>> from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
    >>> mesh = Mesh(jax.devices(), ('x',))
    >>> sharding = NamedSharding(mesh, P('x'))
    >>> data = load_sharded("checkpoint_path", sharding=sharding)

    >>> # Load with per-leaf sharding logic
    >>> def get_sharding(struct):
    ...     if struct.shape[0] > 1000:
    ...         return distributed_sharding
    ...     return single_device_sharding
    >>> data = load_sharded("checkpoint_path", sharding=get_sharding)
    """
    path = Path(path).absolute()
    loaded_from_pkl = False
    sharded_leaves = None

    if abstract_pytree is None:
        structure_path = Path(f"{path}_structure.pkl").absolute()
        with open(structure_path, "rb") as f:
            abstract_pytree, sharded_leaves = pickle.load(f)
        loaded_from_pkl = True

    # Apply sharding to abstract pytree (only possible when sharded_leaves is available)
    if sharding is not None and sharded_leaves is not None:
        abstract_pytree = _apply_sharding_to_abstract_pytree(abstract_pytree, sharded_leaves, sharding)

    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())

    if not loaded_from_pkl:
        # abstract_pytree was provided by the caller (e.g. batched_sampling).
        # Do NOT swallow errors — structural mismatches must propagate so the
        # caller can provide a clear error message.
        restored = checkpointer.restore(path, args=ocp.args.StandardRestore(abstract_pytree))
        checkpointer.wait_until_finished()
        return restored

    # abstract_pytree was loaded from the pkl structure file — retry with
    # default sharding on failure (handles topology changes like 8-CPU → 1-GPU).
    try:
        restored = checkpointer.restore(path, args=ocp.args.StandardRestore(abstract_pytree))
        checkpointer.wait_until_finished()
        return restored
    except Exception as e:
        warnings.warn(
            f"Failed to load checkpoint from {path} with error: {e}. "
            "Will retry with default sharding on jax.devices()[0].",
            stacklevel=2,
        )

    abstract_pytree = _apply_sharding_to_abstract_pytree(abstract_pytree, sharded_leaves, _get_default_sharding())

    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    restored = checkpointer.restore(path, args=ocp.args.StandardRestore(abstract_pytree))
    checkpointer.wait_until_finished()
    return restored
