"""Batched MCMC sampling utilities."""

import os
from collections.abc import Callable
from functools import partial

import blackjax
import jax
import jax.numpy as jnp
import numpyro
import orbax.checkpoint as ocp
from jaxtyping import Key, PyTree
from numpyro.infer import HMC, MCMC, NUTS
from numpyro.infer.util import initialize_model

from ..io import load_sharded, save_sharded


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
    save_callback: Callable[[dict, str, int], None] | None = None,
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

        init_params_obj, potential_fn, postprocess_fn, _ = initialize_model(init_key,
                                                                            model,
                                                                            model_args=model_args,
                                                                            model_kwargs=model_kwargs,
                                                                            dynamic_args=True,
                                                                            **kwargs)
        logdensity_fn = lambda position: -potential_fn(*model_args, **model_kwargs)(position)
        initial_position = init_params_obj.z
    else:
        logdensity_fn = None
        initial_position = None
        postprocess_fn = None

    state_exists = os.path.exists(state_path)
    if state_exists:
        num_warmup = 1  # No warmup when resuming

    if backend == "blackjax":
        assert logdensity_fn is not None, "logdensity_fn must be defined for blackjax backend"
        assert initial_position is not None, "initial_position must be defined for blackjax backend"
        if sampler == "NUTS":
            adapt = blackjax.window_adaptation(blackjax.nuts,
                                               logdensity_fn,
                                               progress_bar=progress_bar,
                                               target_acceptance_rate=0.8)
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
            initial_state = blackjax.mcmc.mclmc.init(position=initial_position,
                                                     logdensity_fn=logdensity_fn,
                                                     rng_key=init_key)
            if state_exists:
                parameters = {
                    "L": jax.ShapeDtypeStruct((),
                                              jnp.asarray(0.0).dtype),
                    "step_size": jax.ShapeDtypeStruct((),
                                                      jnp.asarray(0.0).dtype),
                }
                tuned_state = initial_state
            else:
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
                parameters = {"L": tuned_params.L, "step_size": tuned_params.step_size}
            sampler_fn = blackjax.mclmc(logdensity_fn, **parameters)
            last_state = tuned_state
        else:
            sampler_fn = None
    elif backend == "numpyro":
        kwargs = {}
        if init_params is not None:
            kwargs["init_strategy"] = partial(numpyro.infer.init_to_value, values=init_params)

        mcmc = MCMC(
            NUTS(model, **kwargs) if sampler == "NUTS" else HMC(model, **kwargs),
            num_warmup=num_warmup,
            num_samples=num_samples,
            progress_bar=True,
        )
        mcmc.warmup(warmup_key, *model_args, **model_kwargs)
        last_state = mcmc.last_state
        parameters = {}
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    if save and not state_exists:
        inference_state = {"nb_samples": jnp.array(0), "last_state": last_state, "parameters": parameters}
        save_sharded(inference_state, state_path, overwrite=True, dump_structure=False)

    if state_exists:
        abstract_state = jax.tree.map(
            ocp.tree.to_shape_dtype_struct,
            {
                "nb_samples": jnp.array(0),
                "last_state": last_state,
                "parameters": parameters
            },
        )

        try:
            saved_state = load_sharded(state_path, abstract_pytree=abstract_state)
        except Exception as e:
            raise RuntimeError(f"Saved sampling state at '{state_path}' was created by a different model "
                               "configuration (e.g., different priors, kappa bins, or mesh size). "
                               "Delete the directory or use a different output path.") from e
        nb_samples, last_state, parameters = (
            saved_state["nb_samples"],
            saved_state["last_state"],
            saved_state["parameters"],
        )

    if backend == "blackjax":
        assert logdensity_fn is not None, "logdensity_fn must be defined for blackjax backend"
        if sampler == "NUTS":
            sampler_fn = blackjax.nuts(logdensity_fn, **parameters)
        elif sampler == "HMC":
            sampler_fn = blackjax.hmc(logdensity_fn, **parameters)
        elif sampler == "MCLMC":
            sampler_fn = blackjax.mclmc(logdensity_fn, **parameters)

    start_batch = nb_samples // num_samples
    if start_batch >= batch_count:
        print(f"All {batch_count} batches already completed. Exiting.")
        return

    for i in range(start_batch, batch_count):
        print(f"Sampling batch {i + 1}/{batch_count} using {sampler} with {backend}...")
        print(f"At sample batch {i + 1}, total samples so far: {nb_samples}")

        run_key, batch_key = jax.random.split(run_key)

        if backend == "blackjax":
            transform = (lambda x, _: x.position
                         if postprocess_fn is None else postprocess_fn(*model_args, **model_kwargs)(x.position))
            last_state, samples = blackjax.util.run_inference_algorithm(
                rng_key=batch_key,
                initial_state=last_state,
                inference_algorithm=sampler_fn,
                num_steps=num_samples,
                transform=transform,
                progress_bar=progress_bar,
            )
            nb_evals = 0  # BlackJAX does not expose the number of evaluations

        elif backend == "numpyro":
            mcmc = MCMC(
                NUTS(model) if sampler == "NUTS" else HMC(model),
                num_warmup=0,
                num_samples=num_samples,
                progress_bar=progress_bar,
            )

            mcmc.post_warmup_state = last_state
            mcmc.run(batch_key, *model_args, **model_kwargs, extra_fields=("num_steps", ))
            samples = mcmc.get_samples()
            nb_evals = mcmc.get_extra_fields()["num_steps"]
            last_state = mcmc.last_state
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        nb_samples += num_samples

        if save:
            print(f"Saving batch {i + 1} samples and state...")
            samples["num_steps"] = jnp.array(nb_evals)
            if save_callback is not None:
                save_callback(samples, samples_prefix, i)
            else:
                save_sharded(samples, f"{samples_prefix}_{i}", overwrite=True)
            inference_state = {"nb_samples": jnp.array(nb_samples), "last_state": last_state, "parameters": parameters}
            save_sharded(inference_state, state_path, overwrite=True, dump_structure=False)
        del samples


