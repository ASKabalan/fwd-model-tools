"""Batched MCMC sampling utilities."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import partial, wraps
from typing import ParamSpec, TypeVar

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jaxtyping import Key, PyTree

from ..io import load_sharded, save_sharded
from .sample_converter import default_save

_Param = ParamSpec("_Param")
_Return = TypeVar("_Return")

__all__ = ["batched_sampling", "requires_samplers"]


def requires_samplers(func: Callable[_Param, _Return]) -> Callable[_Param, _Return]:
    """Decorator that raises ImportError when 'blackjax' is not installed."""
    try:
        import blackjax  # noqa: F401
        import numpyro  # noqa: F401

        return func
    except ImportError:
        pass

    @wraps(func)
    def _deferred(*args: _Param.args, **kwargs: _Param.kwargs) -> _Return:
        raise ImportError("Missing optional dependency 'blackjax'. Install with: pip install jax-fli[sampling]")

    return _deferred


@requires_samplers
def batched_sampling(
    model,
    path: str,
    rng_key: Key,
    num_warmup: int = 500,
    num_samples: int = 1000,
    batch_count: int = 5,
    save: bool = True,
    sampler: str = "NUTS",
    backend: str = "numpyro",
    init_params: PyTree | None = None,
    progress_bar: bool = True,
    save_callback: Callable[[dict, dict, str, int], None] = default_save,
    *model_args,
    **model_kwargs,
):
    """
    Run MCMC in batches, checkpointing after each batch to disk.

    Parameters mirror the underlying NumPyro/BlackJAX samplers; results are
    saved under ``path`` as ``samples_*`` and ``sampling_state``.

    Examples
    --------
    >>> import jax.random as jr
    >>> import numpyro
    >>> def toy_model():  # zero-arg model for brevity
    ...     numpyro.sample("x", numpyro.distributions.Normal(0, 1))
    >>> batched_sampling(
    ...     toy_model,
    ...     path=\"output/toy\",
    ...     rng_key=jr.PRNGKey(0),
    ...     num_warmup=10,
    ...     num_samples=20,
    ...     batch_count=2,
    ...     backend=\"numpyro\",
    ...     progress_bar=False,
    ... )
    """
    import blackjax
    import numpyro

    os.makedirs(path, exist_ok=True)
    state_path = f"{path}/sampling_state"
    samples_prefix = f"{path}/samples"
    nb_samples = 0
    init_params = jax.tree.map(jnp.asarray, init_params)

    assert backend in {"numpyro", "blackjax"}, "Backend must be 'numpyro' or 'blackjax'"
    if backend == "numpyro":
        assert sampler in {"NUTS", "HMC"}, "Only NUTS and HMC supported by numpyro"
    if sampler == "MCLMC":
        assert backend == "blackjax", "MCLMC is only supported by blackjax"

    rng_key, init_key, warmup_key, run_key = jax.random.split(rng_key, 4)
    if backend == "blackjax":
        kwargs = {}
        if init_params is not None:
            kwargs["init_strategy"] = partial(numpyro.infer.init_to_value, values=init_params)

        init_params_obj, potential_fn, postprocess_fn, _ = numpyro.infer.util.initialize_model(
            init_key, model, model_args=model_args, model_kwargs=model_kwargs, dynamic_args=True, **kwargs
        )
        logdensity_fn = lambda position: -potential_fn(*model_args, **model_kwargs)(position)
        initial_position = init_params_obj.z
    else:
        logdensity_fn = None
        initial_position = None
        postprocess_fn = None

    state_exists = os.path.exists(state_path)

    if state_exists:
        # ── Resume from checkpoint: build cheap dummy state for Orbax shape inference ──
        print("Found existing sampling state, skipping warmup compilation...")
        if backend == "blackjax":
            assert logdensity_fn is not None, "logdensity_fn must be defined for blackjax backend"
            assert initial_position is not None, "initial_position must be defined for blackjax backend"
            if sampler in ("NUTS", "HMC"):
                init_fn = blackjax.nuts.init if sampler == "NUTS" else blackjax.hmc.init
                last_state = init_fn(initial_position, logdensity_fn)
                flat_positions = jax.tree.leaves(initial_position)
                num_params = sum(p.size for p in flat_positions)
                parameters = {
                    "step_size": jnp.array(0.0),
                    "inverse_mass_matrix": jnp.zeros(num_params),
                }
            elif sampler == "MCLMC":
                last_state = blackjax.mcmc.mclmc.init(
                    position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
                )
                parameters = {"L": jnp.array(0.0), "step_size": jnp.array(0.0)}
            else:
                raise ValueError(f"Unsupported sampler: {sampler}")
        elif backend == "numpyro":
            numpyro_kwargs = {}
            if init_params is not None:
                numpyro_kwargs["init_strategy"] = partial(numpyro.infer.init_to_value, values=init_params)
            kernel = (
                numpyro.infer.NUTS(model, **numpyro_kwargs)
                if sampler == "NUTS"
                else numpyro.infer.HMC(model, **numpyro_kwargs)
            )
            last_state = kernel.init(warmup_key, 0, init_params=None, model_args=model_args, model_kwargs=model_kwargs)
            parameters = {}
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        abstract_state = jax.tree.map(
            ocp.tree.to_shape_dtype_struct,
            {"nb_samples": jnp.array(0), "last_state": last_state, "parameters": parameters},
        )
        try:
            saved_state = load_sharded(state_path, abstract_pytree=abstract_state)
        except Exception as e:
            raise RuntimeError(
                f"Saved sampling state at '{state_path}' was created by a different model "
                "configuration (e.g., different priors, kappa bins, or mesh size). "
                "Delete the directory or use a different output path."
            ) from e
        nb_samples, last_state, parameters = (
            saved_state["nb_samples"],
            saved_state["last_state"],
            saved_state["parameters"],
        )
    else:
        # ── Fresh run: full warmup / adaptation ──
        if backend == "blackjax":
            assert logdensity_fn is not None, "logdensity_fn must be defined for blackjax backend"
            assert initial_position is not None, "initial_position must be defined for blackjax backend"
            if sampler == "NUTS":
                adapt = blackjax.window_adaptation(
                    blackjax.nuts, logdensity_fn, progress_bar=progress_bar, target_acceptance_rate=0.8
                )
                (last_state, parameters), _ = adapt.run(warmup_key, initial_position, num_warmup)
            elif sampler == "HMC":
                adapt = blackjax.window_adaptation(
                    blackjax.hmc,
                    logdensity_fn,
                    progress_bar=progress_bar,
                    target_acceptance_rate=0.8,
                    num_integration_steps=10,
                )
                (last_state, parameters), _ = adapt.run(warmup_key, initial_position, num_warmup)
            elif sampler == "MCLMC":
                initial_state = blackjax.mcmc.mclmc.init(
                    position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
                )
                kernel_builder = lambda imm: blackjax.mcmc.mclmc.build_kernel(
                    logdensity_fn=logdensity_fn,
                    integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                    inverse_mass_matrix=imm,
                )
                print("Tuning MCLMC parameters (L and step_size)...")
                tuned_state, tuned_params, _ = blackjax.mclmc_find_L_and_step_size(
                    mclmc_kernel=kernel_builder,
                    num_steps=num_warmup,
                    state=initial_state,
                    rng_key=warmup_key,
                    diagonal_preconditioning=False,
                    desired_energy_var=1e-3,
                )
                last_state = tuned_state
                parameters = {"L": tuned_params.L, "step_size": tuned_params.step_size}
            else:
                raise ValueError(f"Unsupported sampler: {sampler}")
        elif backend == "numpyro":
            numpyro_kwargs = {}
            if init_params is not None:
                numpyro_kwargs["init_strategy"] = partial(numpyro.infer.init_to_value, values=init_params)
            mcmc = numpyro.infer.MCMC(
                numpyro.infer.NUTS(model, **numpyro_kwargs)
                if sampler == "NUTS"
                else numpyro.infer.HMC(model, **numpyro_kwargs),
                num_warmup=num_warmup,
                num_samples=num_samples,
                progress_bar=True,
            )
            mcmc.warmup(warmup_key, *model_args, **model_kwargs)
            last_state = mcmc.last_state
            parameters = {}
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        assert last_state is not None, "last_state must be defined to save initial state"
        assert parameters is not None, "parameters must be defined to save initial state"
        if save:
            inference_state = {"nb_samples": jnp.array(0), "last_state": last_state, "parameters": parameters}
            save_sharded(inference_state, state_path, overwrite=True, dump_structure=False)

    if backend == "blackjax":
        assert logdensity_fn is not None, "logdensity_fn must be defined for blackjax backend"
        if sampler == "NUTS":
            sampler_fn = blackjax.nuts(logdensity_fn, **parameters)
        elif sampler == "HMC":
            sampler_fn = blackjax.hmc(logdensity_fn, **parameters)
        elif sampler == "MCLMC":
            sampler_fn = blackjax.mclmc(logdensity_fn, **parameters)
        else:
            raise ValueError(f"Unsupported sampler: {sampler}")
    else:
        sampler_fn = None  # Not used for numpyro since it handles sampling internally

    start_batch = nb_samples // num_samples
    if start_batch >= batch_count:
        print(f"All {batch_count} batches already completed. Exiting.")
        return

    for i in range(start_batch, batch_count):
        print(f"Sampling batch {i + 1}/{batch_count} using {sampler} with {backend}...")
        print(f"At sample batch {i + 1}, total samples so far: {nb_samples}")

        run_key, batch_key = jax.random.split(run_key)

        if backend == "blackjax":

            def transform(state, info):
                position = (
                    postprocess_fn(*model_args, **model_kwargs)(state.position)
                    if postprocess_fn is not None
                    else state.position
                )
                return position, info

            assert sampler_fn is not None, "sampler_fn must be defined for blackjax backend"
            last_state, (samples, infos) = blackjax.util.run_inference_algorithm(
                rng_key=batch_key,
                initial_state=last_state,
                inference_algorithm=sampler_fn,
                num_steps=num_samples,
                transform=transform,
                progress_bar=progress_bar,
            )
            if sampler == "MCLMC":
                metrics = {
                    "mean_num_steps": None,
                    "total_divergences": None,
                    "mean_accept_prob": float(jnp.mean(infos.acceptance_rate)),
                }
            else:
                metrics = {
                    "mean_num_steps": float(jnp.mean(infos.num_integration_steps)),
                    "total_divergences": int(jnp.sum(infos.is_divergent)),
                    "mean_accept_prob": float(jnp.mean(infos.acceptance_rate)),
                }

        elif backend == "numpyro":
            mcmc = numpyro.infer.MCMC(
                numpyro.infer.NUTS(model) if sampler == "NUTS" else numpyro.infer.HMC(model),
                num_warmup=0,
                num_samples=num_samples,
                progress_bar=progress_bar,
            )

            mcmc.post_warmup_state = last_state
            mcmc.run(batch_key, *model_args, **model_kwargs, extra_fields=("num_steps", "diverging", "accept_prob"))
            samples = mcmc.get_samples()
            extra = mcmc.get_extra_fields()
            metrics = {
                "mean_num_steps": float(jnp.mean(extra["num_steps"])),
                "total_divergences": int(jnp.sum(extra["diverging"])),
                "mean_accept_prob": float(jnp.mean(extra["accept_prob"])),
            }
            last_state = mcmc.last_state
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        nb_samples += num_samples

        if save:
            print(f"Saving batch {i + 1} samples and state...")
            save_callback(samples, metrics, samples_prefix, i)
            inference_state = {"nb_samples": jnp.array(nb_samples), "last_state": last_state, "parameters": parameters}
            save_sharded(inference_state, state_path, overwrite=True, dump_structure=False)
        del samples
