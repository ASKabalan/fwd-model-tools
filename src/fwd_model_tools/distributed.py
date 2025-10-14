import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
from jax import make_array_from_callback
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
    return tuple([get_axis_size(sharding, i) for i in range(len(sharding.spec))])


def save_sharded_array(array, prefix="shard_data"):
    """
    Saves addressable shards and mesh pdims to disk for later reconstruction.

    This function saves each addressable shard of a distributed JAX array to disk,
    along with metadata about the sharding configuration. This allows for efficient
    storage and loading of large distributed arrays.

    Parameters
    ----------
    array : jax.Array
        The sharded array to save. Must have a sharding attribute.
    prefix : str, default="shard_data"
        Directory prefix where shards will be saved.

    Notes
    -----
    - Creates one .npy file per addressable shard: {prefix}/shard_{rank}_{i}.npy
    - Saves metadata (pdims, global_shape) to {prefix}/info.pkl (rank 0 only)
    - All processes must call this function (collective operation)
    """
    assert hasattr(array, "sharding"), "Array must have a sharding spec."

    rank = jax.process_index()
    shards = array.addressable_shards
    pdims = get_pdims_from_sharding(array.sharding)
    global_shape = array.shape

    os.makedirs(prefix, exist_ok=True)

    for i, shard in enumerate(shards):
        np.save(f"{prefix}/shard_{rank}_{i}.npy", shard.data)

    if rank == 0:
        with open(f"{prefix}/info.pkl", "wb") as f:
            pickle.dump((pdims, global_shape), f)

    print(f"Rank {rank}: saved {len(shards)} shards.")


def load_sharded_array(sharding, prefix="shard_data"):
    """
    Loads a sharded array from saved shard files using given sharding.

    This function reconstructs a distributed JAX array from shard files saved by
    save_sharded_array(). It verifies that the current sharding configuration matches
    the saved configuration.

    Parameters
    ----------
    sharding : NamedSharding
        The sharding specification to use for loading. Must match the saved sharding.
    prefix : str, default="shard_data"
        Directory prefix where shards are stored.

    Returns
    -------
    jax.Array
        The reconstructed sharded array.

    Raises
    ------
    ValueError
        If the current sharding does not match the saved sharding configuration.

    Notes
    -----
    - All processes must call this function (collective operation)
    - Automatically handles multi-dimensional sharding
    """
    rank = jax.process_index()

    with open(f"{prefix}/info.pkl", "rb") as f:
        saved_pdims, saved_global_shape = pickle.load(f)

    current_pdims = get_pdims_from_sharding(sharding)
    if saved_pdims != current_pdims:
        raise ValueError(f"Mesh shape mismatch: saved {saved_pdims}, got {current_pdims}")

    saved_pdims = saved_pdims + (1,) * (len(saved_global_shape) - len(saved_pdims))

    nb_slices = tuple(saved_global_shape[i] // saved_pdims[i] for i in range(len(saved_pdims)))

    def index_to_flat(index, pdims):
        """Maps JAX slice-based index to flat shard index based on mesh dims."""
        flat_index = 0
        flatted_pdims = pdims[1:] + (1,)
        for i, (idx, size) in enumerate(zip(index, nb_slices)):
            if idx.start is None:
                continue

            flat_index += (idx.start // size) * flatted_pdims[i]

        return 0

    def load_callback(index):
        local_index = index_to_flat(index, saved_pdims)
        fname = f"{prefix}/shard_{rank}_{local_index}.npy"
        return np.load(fname)

    restored = make_array_from_callback(
        saved_global_shape,
        sharding,
        data_callback=load_callback,
    )
    return restored
