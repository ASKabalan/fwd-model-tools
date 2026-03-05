"""Shared helpers used across multiple fli-* CLI scripts."""

from __future__ import annotations

import math
import re
import warnings
from argparse import Namespace

import jax
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

import jax_fli as jfli

__all__ = ["_try_parse_s3", "_resolve_nz_shear", "_build_sharding"]


# ---------------------------------------------------------------------------
# nz_shear resolver
# ---------------------------------------------------------------------------


def _try_parse_s3(token: str):
    """Parse s3/stage3 with an optional bin selector. Returns None if token is not s3.

    Supported forms:
      s3            → all 4 Stage-3 bins
      s3[0]         → first bin only (wrapped in a list)
      s3[1:3]       → bins 1 and 2
      s3[:2]        → first two bins
      s3[::2]       → every other bin
    """
    m = re.fullmatch(r"(?:stage3|s3)(?:\[([^\]]*)\])?", token, re.IGNORECASE)
    if m is None:
        return None
    distributions = jfli.io.get_stage3_nz_shear()
    selector = m.group(1)
    if selector is None:
        return distributions
    # Integer index → wrap in list for uniform downstream handling
    if re.fullmatch(r"-?\d+", selector):
        return [distributions[int(selector)]]
    # Slice notation  start:stop  or  start:stop:step
    parts = selector.split(":")
    if 2 <= len(parts) <= 3:
        opt = lambda s: int(s) if s else None  # noqa: E731
        slc = slice(opt(parts[0]), opt(parts[1]), opt(parts[2]) if len(parts) == 3 else None)
        return distributions[slc]
    raise ValueError(f"Cannot parse s3 selector '[{selector}]'. Use s3[i] or s3[start:stop[:step]].")


def _resolve_nz_shear(args: Namespace):
    """Return nz_shear list from CLI --nz-shear values."""
    nz_shear = getattr(args, "nz_shear", None)
    if nz_shear is None:
        return None
    values = nz_shear
    if len(values) == 1:
        s3 = _try_parse_s3(values[0])
        if s3 is not None:
            return s3
    # Otherwise parse as floats
    try:
        import jax.numpy as jnp

        return jnp.array(values, dtype=jnp.float32)
    except ValueError as exc:
        raise ValueError(f"--nz-shear values must be floats or s3/s3[i]/s3[start:stop]: {values}") from exc


# ---------------------------------------------------------------------------
# Sharding setup
# ---------------------------------------------------------------------------


def _build_sharding(args: Namespace):
    """Return sharding or None for single-device runs.

    Warns if the product of pdim dimensions does not match the available device count.
    """
    print(f"jax devices: {jax.devices()}")
    pdim = tuple(args.pdim)

    n_devices = jax.device_count()
    if math.prod(pdim) != n_devices:
        warnings.warn(
            f"--pdim {pdim} implies {math.prod(pdim)} devices but jax.device_count() == {n_devices}. "
            "Results may be incorrect on a misconfigured device mesh.",
            stacklevel=2,
        )

    if pdim == (1, 1):
        return None

    mesh = jax.make_mesh(pdim, ("x", "y"), axis_types=(AxisType.Auto, AxisType.Auto))
    sharding = NamedSharding(mesh, P("x", "y"))
    return sharding
