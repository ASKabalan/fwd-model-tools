from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np

from .fields import DensityField, ParticleField, PositionUnit

__all__ = [
    "compute_box_size_from_redshift",
    "compute_max_redshift_from_box_size",
    "compute_lightcone_shells",
    "distances",
    "centers",
    "compute_particle_scale_factors",
]


@partial(jax.jit, static_argnames=["max_redshift", "observer_position"])
def compute_box_size_from_redshift(cosmo, max_redshift, observer_position):
    """
    Compute simulation box size from maximum redshift and observer position.

    The box size is determined by the comoving distance to max_redshift and
    scaled by the observer position to ensure the observable volume is fully
    contained within the simulation box. For observers not at the box center,
    the box must be larger to accommodate the maximum line-of-sight distance.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology object.
    max_redshift : float
        Maximum redshift for the lightcone.
    observer_position : tuple or array_like
        Observer position as fraction of box size (x, y, z) with values in [0, 1].
        Example: (0.5, 0.5, 0.5) for box center, (0.5, 0.5, 0.0) for edge.

    Returns
    -------
    tuple
        Box size (Lx, Ly, Lz) in Mpc/h for each dimension.

    Notes
    -----
    The scaling factor for each dimension is computed as:
        factor = 1.0 + 2.0 * min(position, 1.0 - position)

    This ensures:
    - Centered observer (0.5): factor = 2.0 → box size = 2 * r_comoving
    - Edge observer (0.0 or 1.0): factor = 2.0 → box size = 2 * r_comoving
    - Off-center: factor > 2.0 → larger box to contain full light cone

    Examples
    --------
    >>> cosmo = Planck18()
    >>> box_size = compute_box_size_from_redshift(cosmo, 1.0, (0.5, 0.5, 0.5))
    >>> # Returns symmetric box sized to contain lightcone to z=1.0
    """
    r_comoving = jc.background.radial_comoving_distance(cosmo, jc.utils.z2a(max_redshift)).squeeze()

    observer_position = np.asarray(observer_position)
    factors = np.clip(observer_position, 0.0, 1.0)
    factors = 1.0 + 2.0 * np.minimum(factors, 1.0 - factors)

    box_size = jax.tree.map(lambda f, r: f * r, factors, r_comoving)

    return box_size


@partial(jax.jit, static_argnames=["box_size", "observer_position"])
def compute_max_redshift_from_box_size(cosmo, box_size, observer_position):
    """
    Compute maximum redshift from simulation box size and observer position.

    The maximum redshift is determined by the comoving distance that can be
    accommodated within the simulation box, given the observer's position.
    For observers not at the box center, the effective line-of-sight distance
    is reduced, leading to a lower maximum redshift.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology object.
    box_size : tuple or array_like
        Simulation box size (Lx, Ly, Lz) in Mpc/h for each dimension.
    observer_position : tuple or array_like
        Observer position as fraction of box size (x, y, z) with values in [0, 1].
        Example: (0.5, 0.5, 0.5) for box center, (0.5, 0.5, 0.0) for edge.

    Returns
    -------
    float
        Maximum redshift that can be accommodated within the box.

    Notes
    -----
    The effective comoving distance is computed as:
        r_effective = min(Li / factor_i)
    where factor_i = 1.0 + 2.0 * min(position_i, 1.0 - position_i) for each dimension i.

    Examples
    --------
    >>> cosmo = Planck18()
    >>> max_redshift = compute_max_redshift_from_box_size(cosmo, (500.0, 500.0, 500.0), (0.5, 0.5, 0.5))
    >>> # Returns maximum redshift accommodated by the box size
    """
    box_size = jnp.asarray(box_size)
    observer_position = jnp.asarray(observer_position)

    factors = jnp.clip(observer_position, 0.0, 1.0)
    factors = 1.0 + 2.0 * jnp.minimum(factors, 1.0 - factors)

    r_effective = jnp.min(box_size / factors)

    max_redshift = jc.utils.a2z(jc.background.a_of_chi(cosmo, r_effective)).squeeze()

    return max_redshift


@partial(jax.jit, static_argnames=["nb_shells"])
def compute_lightcone_shells(
    cosmo,
    field: DensityField,
    nb_shells,
) -> tuple[jax.Array, jax.Array, float]:
    """Return comoving shell centers, scale factors, and shell width.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology object for distance computations.
    field : DensityField
        Field containing box_size, observer_position, and nb_shells metadata.
    nb_shells : int
        Number of radial shells to divide the lightcone into.

    Returns
    -------
    tuple
        ``(r_center, a_center, density_plane_width)`` where ``r_center`` and
        ``a_center`` have shape ``(nb_shells,)``.

    Notes
    -----
    This mirrors the logic from :func:`compute_snapshot_scale_factors` but keeps
    both the comoving distance grid and the corresponding scale factors, which
    is convenient for painting functions that require centers in Mpc.
    """
    max_radius = field.max_comoving_radius
    density_plane_width = max_radius / nb_shells
    n_lens = int(max_radius // float(density_plane_width))

    r_edges = jnp.linspace(
        0.0,
        float(n_lens) * float(density_plane_width),
        n_lens + 1,
    )[::-1]
    r_center = 0.5 * (r_edges[1:] + r_edges[:-1])
    a_center = jc.background.a_of_chi(cosmo, r_center)

    return r_center, a_center


@jax.jit
def compute_particle_scale_factors(
    cosmo,
    field: ParticleField,
) -> jax.Array:
    assert isinstance(field, ParticleField), "field must be a ParticleField instance."
    distances = field.to(PositionUnit.MPC_H)
    distances = distances - jnp.array(field.observer_position_mpc)
    r = jnp.linalg.norm(distances.array, axis=-1)
    a = jc.background.a_of_chi(cosmo, r.reshape(-1, r.shape[-1])).reshape(r.shape)

    return a


@jax.jit
def edges(centers, r_left=None):
    """
    Computes (2, N) edges from centers.
    Guarantees identical physical grids for ascending and descending inputs.
    Row 0: Left edges (minimums)
    Row 1: Right edges (maximums)
    """
    is_descending = centers[0] > centers[-1]

    # 1. Flip to always work left-to-right (ascending)
    c_asc = jnp.where(is_descending, centers[::-1], centers)

    # 2. Calculate the physical left boundary anchor
    if r_left is None:
        r_left = 1.5 * c_asc[0] - 0.5 * c_asc[1]

    # 3. Compute sequentially left-to-right
    def step_fn(r_prev, c):
        r_next = 2 * c - r_prev
        return r_next, r_next

    _, edges_tail = jax.lax.scan(step_fn, r_left, c_asc)
    edges_1d = jnp.concatenate([jnp.atleast_1d(r_left), edges_tail])

    # 4. Extract left and right pairs
    left_asc = edges_1d[:-1]
    right_asc = edges_1d[1:]

    # 5. Reverse the pairs back if the input was descending
    left_final = jnp.where(is_descending, left_asc[::-1], left_asc)
    right_final = jnp.where(is_descending, right_asc[::-1], right_asc)

    # Returns shape (2, N)
    return jnp.stack([left_final, right_final], axis=0)


@jax.jit
def distances(centers, r_left=None):
    """
    Computes absolute cell widths from centers.
    Output order perfectly matches the input centers.
    """
    e = edges(centers, r_left)

    # Edges guarantees Row 1 is always the right edge, Row 0 is the left.
    return e[1] - e[0]


@jax.jit
def centers(distances, r_left, is_descending=False):
    """
    Computes centers from absolute distances.
    Builds the grid left-to-right to maintain physical symmetry.
    """
    # 1. Ensure we are building the distances from left-to-right
    d_asc = jnp.where(is_descending, distances[::-1], distances)

    # 2. Build the contiguous edges
    r_edges = jnp.concatenate([jnp.atleast_1d(r_left), r_left + jnp.cumsum(d_asc)])

    # 3. Calculate ascending midpoints
    c_asc = 0.5 * (r_edges[1:] + r_edges[:-1])

    # 4. Flip back to match original order
    return jnp.where(is_descending, c_asc[::-1], c_asc)
