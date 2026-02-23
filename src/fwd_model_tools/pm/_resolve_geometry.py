"""Private helper to resolve ts / nb_shells / density_widths into canonical lightcone geometry."""

from __future__ import annotations

import jax.numpy as jnp
import jax_cosmo as jc

from ..fields import DensityField
from ..utils import compute_lightcone_shells, distances


def resolve_ts_geometry(cosmo, field: DensityField, painting, ts=None, nb_shells=None, density_widths=None):
    """Resolve the ``ts`` / ``nb_shells`` / ``density_widths`` triple into canonical geometry.

    Parameters
    ----------
    cosmo : jax_cosmo.Cosmology
        Cosmology for distance calculations.
    field : DensityField
        Field carrying box/observer metadata (needed for ``compute_lightcone_shells``).
    ts : scalar, 1-D array, or 2-D ``(2, N)`` array, optional
        Scale-factor specification.  Mutually exclusive with *nb_shells*.
    nb_shells : int, optional
        Number of radial shells.  Mutually exclusive with *ts*.
    density_widths : scalar or array, optional
        Override shell widths.  When *ts* is a scalar this triggers lightcone
        mode for a single shell.
    t1 : scalar, 1-D array, or 2-D ``(2, N)`` array, optional
        End time or detailed time specification. Used if *ts* and *nb_shells* are None.
        Defaults to 1.0.

    Returns
    -------
    ts_resolved : jax.Array or None
        Scale factors at shell centres (or ``None`` for snapshot mode).
    r_centers : jax.Array or None
        Comoving distances at shell centres.
    density_widths : jax.Array or None
        Shell widths in comoving distance.
    is_lightcone : bool
        ``True`` when the output should be treated as a lightcone (multi-shell
        or single-shell with explicit *density_widths*).

    Raises
    ------
    ValueError
        If both *ts* and *nb_shells* are set, or if *ts* has an unsupported shape.
    """
    if ts is None and nb_shells is None:
        raise ValueError("Cannot specify both `ts` and `nb_shells`.")

    is_projecting = painting.target in ("flat", "spherical")

    if ts is not None and nb_shells is not None:
        raise ValueError("Cannot specify both `ts` and `nb_shells`.")

    # --- nb_shells path ---
    if nb_shells is not None:
        r_centers, ts_resolved = compute_lightcone_shells(cosmo, field, nb_shells=nb_shells)
        if density_widths is not None:
            try:
                density_widths = jnp.broadcast_to(density_widths, r_centers.shape)
            except ValueError:
                raise ValueError("density_widths must be broadcastable to the shape of r_centers.")
        else:
            density_widths = distances(r_centers)
        return ts_resolved, r_centers, density_widths, True

    ts = jnp.asarray(ts)

    # --- scalar ts ---
    if ts.ndim == 0 or (ts.ndim == 1 and ts.size == 1):
        ts = jnp.atleast_1d(ts).squeeze()
        r_center = jc.background.radial_comoving_distance(cosmo, ts)
        if density_widths is not None:
            # Single-shell lightcone
            density_widths = jnp.asarray(density_widths).squeeze()
            assert jnp.isscalar(density_widths), "For single-shell mode, density_widths must be a scalar."

            density_widths = jnp.atleast_1d(density_widths)
        elif is_projecting:
            raise ValueError("Projection painting requires explicit density_widths for single-shell mode.")
        else:
            density_widths = jnp.array([0.0])
        # Snapshot mode
        return jnp.atleast_1d(ts), jnp.atleast_1d(r_center), density_widths, False

    # --- 1-D array: shell centres ---
    if ts.ndim == 1:
        r_centers = jc.background.radial_comoving_distance(cosmo, ts)
        if density_widths is not None:
            try:
                density_widths = jnp.broadcast_to(density_widths, r_centers.shape)
            except ValueError:
                raise ValueError("density_widths must be broadcastable to the shape of ts.")
            density_widths = density_widths
        else:
            density_widths = distances(r_centers)
        return ts, r_centers, density_widths, True

    # --- 2-D array (2, N): near/far ---
    if ts.ndim == 2 and ts.shape[0] == 2:
        a_near, a_far = ts[0], ts[1]
        r_near, r_far = jc.background.radial_comoving_distance(cosmo, jnp.array([a_near, a_far]))
        r_centers = 0.5 * (r_near + r_far)
        ts_resolved = jc.background.a_of_chi(cosmo, r_centers)
        density_widths = jnp.abs(r_far - r_near)
        return ts_resolved, r_centers, density_widths, True

    raise ValueError(f"ts has unsupported shape {ts.shape}. Expected scalar, 1-D, or (2, N).")
