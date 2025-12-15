from __future__ import annotations

__all__ = ["raytrace"]


def raytrace(
    cosmo,
    lightcone,
    nz_shear,
    min_z=0.01,
    max_z=3.0,
    n_integrate=32,
):
    """Placeholder for multi-plane ray-tracing solver.

    Parameters mirror :func:`fwd_model_tools.lensing.born` so callers can
    switch between Born and ray-tracing pipelines without API churn.
    """
    raise NotImplementedError("raytrace() is not implemented yet.")
