import warnings

import jax
import jax.lax as lax


def _issue_warning(msg):
    """The python function that runs on the host."""
    warnings.warn(msg, stacklevel=4)


def warning_if(x, pred, msg):
    """
    1. Checks 'pred' on the device (GPU/TPU).
    2. If True, triggers a callback to the host to print the warning.
    3. Returns 'x' unchanged (identity).
    """

    def _trigger_warn(_):
        # This sends the 'msg' string to the host and runs _issue_warning
        jax.debug.callback(_issue_warning, msg)

    # lax.cond handles the "if" logic on the device
    lax.cond(
        pred,
        _trigger_warn,  # Run if True
        lambda _: None,  # Do nothing if False
        operand=None)
    return x
