import os
import pickle
from functools import partial
from glob import glob

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.experimental.multihost_utils import process_allgather
from jaxtyping import Key, PyTree
from numpyro.infer import HMC, MCMC, NUTS
from numpyro.infer.util import initialize_model

all_gather = partial(process_allgather, tiled=True)


def batched_sampling(
    model,
    path: str,
    rng_key: Key,
    num_warmup: int = 500,
    num_samples: int = 1000,
    batch_count: int = 5,
    save: bool = True,
    sampler: str = "NUTS",  # NUTS, HMC, MCLMC
    backend: str = "numpyro",  # numpyro or blackjax
    init_params: PyTree | None = None,
    *model_args,
    **model_kwargs,
):
    os.makedirs(path, exist_ok=True)
    state_path = f"{path}/sampling_state.pkl"
    samples_prefix = f"{path}/samples"
    nb_samples = 0

    assert backend in {"numpyro", "blackjax"}, "Backend must be 'numpyro' or 'blackjax'"
    if backend == "numpyro":
        assert sampler in {"NUTS", "HMC"}, "Only NUTS and HMC supported by numpyro"
    if sampler == "MCLMC":
        assert backend == "blackjax", "MCLMC is only supported by blackjax"

    rng_key, init_key, warmup_key, run_key = jax.random.split(rng_key, 4)
    if backend == "blackjax":
        init_params_obj, potential_fn, postprocess_fn, _ = initialize_model(
            init_key, model, model_args=model_args, model_kwargs=model_kwargs, dynamic_args=True
        )
        logdensity_fn = lambda position: -potential_fn(*model_args, **model_kwargs)(position)
        initial_position = init_params if init_params is not None else init_params_obj.z
    else:
        logdensity_fn = None
        initial_position = None
        postprocess_fn = None

    if not os.path.exists(state_path):
        print(f"🔁 Starting fresh with warmup for {sampler} using {backend}...")
        if backend == "blackjax":
            assert logdensity_fn is not None, "logdensity_fn must be defined for blackjax backend"
            assert initial_position is not None, "initial_position must be defined for blackjax backend"
            if sampler == "NUTS":
                adapt = blackjax.window_adaptation(
                    blackjax.nuts, logdensity_fn, progress_bar=True, target_acceptance_rate=0.8
                )
                (last_state, parameters), _ = adapt.run(warmup_key, initial_position, num_warmup)
                sampler_fn = blackjax.nuts(logdensity_fn, **parameters)

            elif sampler == "HMC":
                adapt = blackjax.window_adaptation(
                    blackjax.hmc,
                    logdensity_fn,
                    progress_bar=True,
                    target_acceptance_rate=0.8,
                    num_integration_steps=10,
                )
                (last_state, parameters), _ = adapt.run(warmup_key, initial_position, num_warmup)
                sampler_fn = blackjax.hmc(logdensity_fn, **parameters)

            elif sampler == "MCLMC":
                initial_state = blackjax.mcmc.mclmc.init(
                    position=initial_position, logdensity_fn=logdensity_fn, rng_key=init_key
                )
                kernel_builder = lambda imm: blackjax.mcmc.mclmc.build_kernel(
                    logdensity_fn=logdensity_fn,
                    integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
                    inverse_mass_matrix=imm,
                )
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

        if save:
            with open(state_path, "wb") as f:
                pickle.dump((0, last_state, parameters), f)
    else:
        print(f"▶️ Resuming from saved warmup state for {sampler} using {backend}...")
        with open(state_path, "rb") as f:
            saved_state = pickle.load(f)
        nb_samples, last_state, parameters = saved_state[:3]
        if backend == "blackjax" and len(saved_state) > 3:
            # Legacy state files might contain a stored postprocess function; ignore it.
            pass

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
        print(f"📦 Sampling batch {i + 1}/{batch_count} using {sampler} with {backend}...\n")
        print(f"at sample batch {i + 1}, total samples: {nb_samples}")

        run_key, batch_key = jax.random.split(run_key)

        if backend == "blackjax":
            last_state, raw_samples = blackjax.util.run_inference_algorithm(
                rng_key=batch_key,
                initial_state=last_state,
                inference_algorithm=sampler_fn,
                num_steps=num_samples,
                transform=lambda x, _: x.position,
                progress_bar=True,
            )
            if postprocess_fn is not None:
                samples = jax.vmap(postprocess_fn(*model_args, **model_kwargs))(raw_samples)
            else:
                samples = raw_samples
            nb_evals = 0  # Don't know how to get the number of evaluations in blackjax

        elif backend == "numpyro":
            mcmc = MCMC(
                NUTS(model) if sampler == "NUTS" else HMC(model),
                num_warmup=0,
                num_samples=num_samples,
                progress_bar=True,
            )
            mcmc.post_warmup_state = last_state
            mcmc.run(batch_key, *model_args, **model_kwargs, extra_fields=("num_steps",))
            samples = mcmc.get_samples()
            nb_evals = mcmc.get_extra_fields()["num_steps"]
            last_state = mcmc.last_state
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        print("\n")
        nb_samples += num_samples

        if save:
            host_samples = all_gather(samples)
            host_samples["num_steps"] = nb_evals
            np.savez(f"{samples_prefix}_{i}.npz", **host_samples)
            del host_samples
            with open(state_path, "wb") as f:
                pickle.dump((nb_samples, last_state, parameters), f)
        del samples


def load_samples(path: str, param_names: list[str] = None) -> dict:
    """
    Efficiently load and concatenate parameter samples from saved batches.

    Parameters
    ----------
    path : str
        Directory where samples are saved (e.g., "output/mcmc_run").
    param_names : list of str, optional
        List of parameter names to load. If None, load all available parameters.

    Returns
    -------
    concatenated : dict
        Dictionary mapping parameter names to concatenated JAX arrays.
    """
    files = sorted(glob(os.path.join(path, "*samples_*.npz")))
    if not files:
        raise FileNotFoundError(f"No sample files found in path: {path}")

    if param_names is None:
        with np.load(files[0]) as sample_file:
            param_names = list(sample_file.keys())

    collected = {name: [] for name in param_names}

    for file in files:
        data = np.load(file)
        for name in param_names:
            if name in data:
                collected[name].append(jnp.atleast_1d(jnp.array(data[name])))

    return {k: jnp.concatenate(v, axis=0) for k, v in collected.items() if v}
