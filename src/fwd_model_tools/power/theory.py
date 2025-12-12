from __future__ import annotations

import numbers

import jax.numpy as jnp
import jax_cosmo as jc

__all__ = ["compute_theory_cl"]


def compute_theory_cl(
    cosmo: jc.Cosmology,
    ell: jnp.ndarray,
    z_source: float | jc.redshift.redshift_distribution,
) -> jnp.ndarray:
    """
    Compute the theoretical weak-lensing angular power spectrum C_ell.
    """
    nz = _normalize_z_source(z_source)
    tracer = jc.probes.WeakLensing([nz], sigma_e=0.0)
    cl_matrix = jnp.asarray(jc.angular_cl.angular_cl(cosmo, jnp.asarray(ell), [tracer]))

    if cl_matrix.ndim == 1:
        return cl_matrix

    return cl_matrix[0]


def _normalize_z_source(z_source: float | jc.redshift.redshift_distribution, ) -> jc.redshift.redshift_distribution:
    if isinstance(z_source, numbers.Real):
        return jc.redshift.delta_nz(float(z_source))
    if isinstance(z_source, jc.redshift.redshift_distribution):
        return z_source
    raise TypeError("z_source must be a scalar redshift or a jax_cosmo redshift distribution.")
