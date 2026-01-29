"""
Utilities for computing upper triangular pair operations.

This module provides efficient functions for mapping operations over all unique
pairs (i, j) where i <= j from a batched array, useful for computing cross-spectra.
"""

import jax.numpy as jnp
from jax import lax


def linear_to_triangular(k, n):
    """
    Convert linear index k to upper triangular indices (i, j) where i <= j.

    For upper triangle layout (row-major order):
      Row i=0: (0,0), (0,1), (0,2), (0,3), ...
      Row i=1: (1,1), (1,2), (1,3), ...
      Row i=2: (2,2), (2,3), ...
      Row i=3: (3,3), ...

    Linear ordering:
      (0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)

    This matches the ordering produced by jnp.triu_indices.

    Row i contains (n - i) pairs.
    Pairs before row i: T_i = i*n - i*(i-1)/2

    Given k, we find row i using:
      i = floor(((2*n + 1) - sqrt((2*n + 1)² - 8*k)) / 2)
    Then j = i + (k - T_i)

    Parameters
    ----------
    k : int or jax.Array
        Linear index in range [0, n*(n+1)/2)
    n : int
        Number of elements in the array

    Returns
    -------
    i, j : tuple of int or jax.Array
        Upper triangular indices where i <= j
    """
    kf = k.astype(jnp.float64)
    nf = jnp.float64(n)

    # Solve for row i using quadratic formula
    # We want i such that T_i <= k < T_{i+1}
    # where T_i = i*n - i*(i-1)/2
    discriminant = (2.0 * nf + 1.0)**2 - 8.0 * kf
    i = jnp.floor(((2.0 * nf + 1.0) - jnp.sqrt(discriminant)) / 2.0).astype(jnp.int32)

    # Compute T_i = i*n - i*(i-1)/2
    t_i = i * n - i * (i - 1) // 2

    # Column j within row i
    j = i + (k - t_i)

    return i, j


def tri_map(a, pair_fn, batch_size=None):
    """
    Map a function over all upper triangular pairs of array elements.

    Computes pair_fn(a[i], a[j]) for all pairs (i,j) where i <= j, using
    lax.map with analytical index computation for memory efficiency.

    Memory: O(1) temporary space - no index arrays materialized
    Best for: Automatic parallelization across pairs

    For an array of n elements, returns K = n*(n+1)/2 results in upper
    triangular order:
      (0,0), (0,1), ..., (0,n-1), (1,1), (1,2), ..., (n-1,n-1)

    Parameters
    ----------
    a : jax.Array
        Input array of shape (n,) or (n, ...) where n is the batch size
    pair_fn : callable
        Function to apply to each pair (a[i], a[j])
        Should accept two array elements and return a result
    batch_size : int, optional
        Batch size for lax.map processing. None means no batching.
        Use smaller values to reduce memory usage.

    Returns
    -------
    jax.Array
        Array of results with shape (K, ...) where K = n*(n+1)/2
        and ... matches the output shape of pair_fn

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> a = jnp.array([1.0, 2.0, 3.0])
    >>> result = tri_map(a, lambda x, y: x + y)
    >>> result
    Array([2., 3., 4., 4., 5., 6.], dtype=float32)
    # Corresponds to: (1+1), (1+2), (1+3), (2+2), (2+3), (3+3)
    """
    n = a.shape[0]
    K = n * (n + 1) // 2

    def body(k):
        i, j = linear_to_triangular(k, n)
        return pair_fn(a[i], a[j])

    return lax.map(body, jnp.arange(K, dtype=jnp.int32), batch_size=batch_size)


__all__ = ["linear_to_triangular", "tri_map"]
