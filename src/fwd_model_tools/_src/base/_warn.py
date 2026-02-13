import warnings

import jax
import jax.lax as lax


def _issue_warning(msg, *args):
    """The python function that runs on the host."""
    warnings.warn(msg.format(*args), stacklevel=4)


def warning_if(x, pred, msg, *args):
    """
    1. Checks 'pred' on the device (GPU/TPU).
    2. If True, triggers a callback to the host to print the warning.
    3. Returns 'x' unchanged (identity).

    Parameters
    ----------
    x : any
        Passthrough value (returned unchanged).
    pred : bool-like
        Traced predicate evaluated on device.
    msg : str
        A ``str.format()``-style template (e.g. ``"value is {:.2f}"``).
    *args
        Traced JAX values forwarded through ``jax.debug.callback``
        and materialized on the host before formatting.
    """

    def _trigger_warn(_):
        jax.debug.callback(_issue_warning, msg, *args)

    lax.cond(pred, _trigger_warn, lambda _: None, operand=None)
    return x
