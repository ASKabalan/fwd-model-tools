import os
from functools import partial
from pathlib import Path

import blackjax
import jax
import jax.numpy as jnp
import numpyro
import orbax.checkpoint as ocp
from jaxtyping import Key, PyTree
from numpyro.infer import HMC, MCMC, NUTS
from numpyro.infer.util import initialize_model

from .distributed import load_sharded, save_sharded


def reshard_numpyro_state(state, sharding):

    def reshard_dict(d, sharding):
        result = {}
        for key, value in d.items():
            if value.ndim > 0:
                result[key] = jax.lax.with_sharding_constraint(value, sharding)
            else:
                result[key] = value
        return result

    resharded_z = jax.jit(reshard_dict, static_argnums=(1, ))(state.z,
                                                              sharding)
    resharded_z_grad = jax.jit(reshard_dict,
                               static_argnums=(1, ))(state.z_grad, sharding)

    return state._replace(z=resharded_z, z_grad=resharded_z_grad)


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
    *model_args,
    **model_kwargs,
):
    os.makedirs(path, exist_ok=True)
    state_path = f"{path}/sampling_state"
    samples_prefix = f"{path}/samples"
    nb_samples = 0
    init_params = jax.tree.map(jnp.asarray, init_params)

    assert backend in {"numpyro",
                       "blackjax"}, "Backend must be 'numpyro' or 'blackjax'"
    if backend == "numpyro":
        assert sampler in {"NUTS",
                           "HMC"}, "Only NUTS and HMC supported by numpyro"
    if sampler == "MCLMC":
        assert backend == "blackjax", "MCLMC is only supported by blackjax"

    rng_key, init_key, warmup_key, run_key = jax.random.split(rng_key, 4)
    if backend == "blackjax":
        kwargs = {}
        if init_params is not None:
            kwargs["init_strategy"] = partial(numpyro.infer.init_to_value,
                                              values=init_params)

        init_params_obj, potential_fn, postprocess_fn, _ = initialize_model(
            init_key,
            model,
            model_args=model_args,
            model_kwargs=model_kwargs,
            dynamic_args=True,
            **kwargs)
        logdensity_fn = lambda position: -potential_fn(*model_args, **
                                                       model_kwargs)(position)
        initial_position = init_params_obj.z
    else:
        logdensity_fn = None
        initial_position = None
        postprocess_fn = None

    state_exists = os.path.exists(state_path)

    nb_samples = 0

    print(
        f"{'▶️ Resuming' if state_exists else '🔁 Starting fresh with warmup'} for {sampler} using {backend}..."
    )

    if backend == "blackjax":
        assert logdensity_fn is not None, "logdensity_fn must be defined for blackjax backend"
        assert initial_position is not None, "initial_position must be defined for blackjax backend"
        if sampler == "NUTS":
            adapt = blackjax.window_adaptation(blackjax.nuts,
                                               logdensity_fn,
                                               progress_bar=progress_bar,
                                               target_acceptance_rate=0.8)
            (last_state, parameters), _ = adapt.run(warmup_key,
                                                    initial_position,
                                                    num_warmup)
            sampler_fn = blackjax.nuts(logdensity_fn, **parameters)

        elif sampler == "HMC":
            adapt = blackjax.window_adaptation(
                blackjax.hmc,
                logdensity_fn,
                progress_bar=progress_bar,
                target_acceptance_rate=0.8,
                num_integration_steps=10,
            )
            (last_state, parameters), _ = adapt.run(warmup_key,
                                                    initial_position,
                                                    num_warmup)
            sampler_fn = blackjax.hmc(logdensity_fn, **parameters)

        elif sampler == "MCLMC":
            initial_state = blackjax.mcmc.mclmc.init(
                position=initial_position,
                logdensity_fn=logdensity_fn,
                rng_key=init_key)
            kernel_builder = lambda imm: blackjax.mcmc.mclmc.build_kernel(
                logdensity_fn=logdensity_fn,
                integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                inverse_mass_matrix=imm,
            )
            print("🔧 Tuning MCLMC parameters (L and step_size)...")

            tuned_state, tuned_params, _ = blackjax.mclmc_find_L_and_step_size(
                mclmc_kernel=kernel_builder,
                num_steps=num_warmup,
                state=initial_state,
                rng_key=warmup_key,
                diagonal_preconditioning=False,
                desired_energy_var=1e-3,
            )
            parameters = {
                "L": tuned_params.L,
                "step_size": tuned_params.step_size
            }
            sampler_fn = blackjax.mclmc(logdensity_fn, **parameters)
            last_state = tuned_state

    elif backend == "numpyro":
        kwargs = {}
        if init_params is not None:
            kwargs["init_strategy"] = partial(numpyro.infer.init_to_value,
                                              values=init_params)

        mcmc = MCMC(
            NUTS(model, **kwargs) if sampler == "NUTS" else HMC(
                model, **kwargs),
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
        inference_state = {
            "nb_samples": jnp.array(0),
            "last_state": last_state,
            "parameters": parameters
        }
        save_sharded(inference_state,
                     state_path,
                     overwrite=True,
                     dump_structure=False)

    if state_exists:
        abstract_state = jax.tree.map(
            ocp.tree.to_shape_dtype_struct,
            {
                "nb_samples": jnp.array(0),
                "last_state": last_state,
                "parameters": parameters
            },
        )

        saved_state = load_sharded(state_path, abstract_pytree=abstract_state)
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
        print(f"✅ All {batch_count} batches already completed. Exiting.")
        return

    for i in range(start_batch, batch_count):
        print(
            f"📦 Sampling batch {i + 1}/{batch_count} using {sampler} with {backend}...\n"
        )
        print(f"at sample batch {i + 1}, total samples: {nb_samples}")

        run_key, batch_key = jax.random.split(run_key)

        if backend == "blackjax":
            last_state, raw_samples = blackjax.util.run_inference_algorithm(
                rng_key=batch_key,
                initial_state=last_state,
                inference_algorithm=sampler_fn,
                num_steps=num_samples,
                transform=lambda x, _: x.position,
                progress_bar=progress_bar,
            )
            if postprocess_fn is not None:
                samples = jax.vmap(postprocess_fn(*model_args,
                                                  **model_kwargs))(raw_samples)
            else:
                samples = raw_samples
            nb_evals = 0  # Don't know how to get the number of evaluations in blackjax

        elif backend == "numpyro":
            mcmc = MCMC(
                NUTS(model) if sampler == "NUTS" else HMC(model),
                num_warmup=0,
                num_samples=num_samples,
                progress_bar=progress_bar,
            )

            mcmc.post_warmup_state = last_state
            mcmc.run(batch_key,
                     *model_args,
                     **model_kwargs,
                     extra_fields=("num_steps", ))
            samples = mcmc.get_samples()
            nb_evals = mcmc.get_extra_fields()["num_steps"]
            last_state = mcmc.last_state
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        print("\n")
        nb_samples += num_samples

        if save:
            print(f"💾 Saving batch {i + 1} samples and state...")
            samples["num_steps"] = jnp.array(nb_evals)
            save_sharded(samples, f"{samples_prefix}_{i}", overwrite=True)
            inference_state = {
                "nb_samples": jnp.array(nb_samples),
                "last_state": last_state,
                "parameters": parameters
            }
            save_sharded(inference_state,
                         state_path,
                         overwrite=True,
                         dump_structure=False)
        del samples


def load_samples(path: str,
                 param_names: list[str] = None,
                 last_n_batches: int = None) -> dict:
    """
    Efficiently load and concatenate parameter samples from saved batches.

    Parameters
    ----------
    path : str
        Directory where samples are saved (e.g., "output/mcmc_run").
    param_names : list of str, optional
        List of parameter names to load. If None, load all available parameters.
    last_n_batches : int, optional
        Number of last batches to load. If None, load all batches.

    Returns
    -------
    concatenated : dict
        Dictionary mapping parameter names to concatenated JAX arrays.
    """
    path = Path(path)
    checkpoint_dirs = sorted([d for d in path.glob("samples_*") if d.is_dir()])

    if len(checkpoint_dirs) == 0:
        raise FileNotFoundError(f"No sample batches found in {path}")

    if last_n_batches is not None and last_n_batches > 0:
        checkpoint_dirs = checkpoint_dirs[-last_n_batches:]

    print(f"Loading {len(checkpoint_dirs)} sample batch(es) from {path}")

    all_samples = []
    for i, checkpoint_dir in enumerate(checkpoint_dirs):
        print(
            f"  Loading batch {i + 1}/{len(checkpoint_dirs)}: {checkpoint_dir.name}"
        )
        batch_samples = load_sharded(str(checkpoint_dir))
        all_samples.append(batch_samples)

    if len(all_samples) == 0:
        raise ValueError(f"No samples loaded from {path}")

    all_available_params = set()
    for samples in all_samples:
        all_available_params.update(samples.keys())

    if param_names is not None:
        missing_params = set(param_names) - all_available_params
        if missing_params:
            raise ValueError(
                f"Requested parameters {missing_params} not found in samples. Available: {all_available_params}"
            )
        params_to_load = param_names
    else:
        params_to_load = list(all_available_params)

    concatenated = {}
    for param in params_to_load:
        param_arrays = [
            jnp.asarray(samples[param]) for samples in all_samples
            if param in samples
        ]

        if len(param_arrays) == 0:
            continue

        if param_arrays[0].ndim == 0:
            concatenated[param] = jnp.stack(param_arrays, axis=0)
        else:
            concatenated[param] = jnp.concatenate(param_arrays, axis=0)

    print(
        f"Loaded {len(concatenated)} parameter(s): {list(concatenated.keys())}"
    )
    print(f"Total samples: {concatenated[params_to_load[0]].shape[0]}")

    return concatenated
