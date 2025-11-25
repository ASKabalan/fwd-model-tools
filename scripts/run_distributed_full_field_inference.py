#!/usr/bin/env python

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["EQX_ON_ERROR"] = "nan"
os.environ["JAX_ENABLE_X64"] = "False"

if int(os.environ.get("SLURM_NTASKS", 0)) > 1 or int(
        os.environ.get("SLURM_NTASKS_PER_NODE", 0)) > 1:
    os.environ["VSCODE_PROXY_URI"] = ""
    os.environ["no_proxy"] = ""
    os.environ["NO_PROXY"] = ""
    del os.environ["VSCODE_PROXY_URI"]
    del os.environ["no_proxy"]
    del os.environ["NO_PROXY"]
    import jax

    jax.distributed.initialize()

import jax

jax.config.update("jax_enable_x64", False)

import argparse
import time
from pathlib import Path

import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib

matplotlib.use("Agg")
import numpy as np
import numpyro.distributions as dist
from diffrax import RecursiveCheckpointAdjoint
from jaxpm.distributed import normal_field
from numpyro.handlers import condition, seed, trace
from scipy.stats import norm

from fwd_model_tools import Configurations, Planck18, full_field_probmodel
from fwd_model_tools.plotting import plot_ic, plot_kappa, plot_lightcone
from fwd_model_tools.sampling import batched_sampling, load_samples
from fwd_model_tools.sampling.plot import plot_posterior
from fwd_model_tools.utils import (compute_box_size_from_redshift,
                                   reconstruct_full_sphere)


def setup_output_dir(output_dir):
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    samples_dir = output_dir / "samples"
    data_dir = output_dir / "data"

    plots_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, plots_dir, samples_dir, data_dir


def setup_sharding(pdims=(4, 2)):
    if jax.device_count() > 1:
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        mesh = jax.make_mesh(pdims, ("x", "y"))
        sharding = NamedSharding(mesh, P("x", "y"))
        print(f"Using sharding with mesh: {pdims}")
    else:
        sharding = None
        print("Single device mode - no sharding")

    return sharding


def create_redshift_distribution(max_redshift):
    z = jnp.linspace(0, max_redshift, 1000)
    z_centers = jnp.linspace(0.2, max_redshift - 0.01, 4)
    z_centers = jnp.round(z_centers, 3)
    print(f"z_centers = {z_centers}")

    nz_shear = [
        jc.redshift.kde_nz(
            z,
            norm.pdf(z, loc=z_center, scale=0.12),
            bw=0.01,
            zmax=max_redshift,
            gals_per_arcmin2=g,
        ) for z_center, g in zip(z_centers, [7, 8.5, 7.5, 7])
    ]
    nbins = len(nz_shear)

    return nz_shear, nbins


def generate_synthetic_observations(config, fiducial_cosmology,
                                    initial_conditions, data_dir, plots_dir):
    print("\nGenerating synthetic observations")

    # TODO: update to new API full_field_probmodel(template_field, config)
    full_field_basemodel = full_field_probmodel(config)

    fiducial_model = condition(
        full_field_basemodel,
        {
            "Omega_c": fiducial_cosmology.Omega_c,
            "sigma8": fiducial_cosmology.sigma8,
            "initial_conditions": initial_conditions,
        },
    )

    print("Tracing fiducial model to generate observables...")
    start_time = time.time()
    model_trace = trace(seed(fiducial_model, 0)).get_trace()
    elapsed = time.time() - start_time
    print(f"Fiducial model traced in {elapsed:.2f}s")

    nbins = len(config.nz_shear)
    kappa_keys = [f"kappa_{i}" for i in range(nbins)]
    true_kappas_visible = {
        key: model_trace[key]["value"]
        for key in kappa_keys
    }

    if config.geometry == "spherical":
        true_kappas_full = reconstruct_full_sphere(true_kappas_visible,
                                                   config.nside,
                                                   config.observer_position)
    else:
        true_kappas_full = true_kappas_visible

    np.savez(
        data_dir / "true_kappas.npz",
        **true_kappas_visible,
        Omega_c=fiducial_cosmology.Omega_c,
        sigma8=fiducial_cosmology.sigma8,
    )

    true_ic = np.asarray(model_trace["ic"]["value"])
    np.save(data_dir / "true_ic.npy", true_ic)

    true_lightcone = np.asarray(model_trace["lightcone"]["value"])
    np.save(data_dir / "true_lightcone.npy", true_lightcone)

    print("Saved observations to disk")

    plot_lightcone(true_lightcone,
                   plots_dir,
                   spherical=(config.geometry == "spherical"))
    print("Plotted lightcone")

    kappa_array = np.stack([true_kappas_full[k] for k in kappa_keys])
    plot_kappa(kappa_array,
               plots_dir,
               spherical=(config.geometry == "spherical"))
    print("Plotted kappa maps")

    return true_kappas_visible


def run_mcmc_inference(config, true_kappas_visible, samples_dir, args,
                       init_params):
    print("\nSetting up MCMC inference")

    config_inference = Configurations(
        density_plane_smoothing=config.density_plane_smoothing,
        nz_shear=config.nz_shear,
        fiducial_cosmology=config.fiducial_cosmology,
        sigma_e=config.sigma_e,
        priors=config.priors,
        t0=config.t0,
        dt0=config.dt0,
        t1=config.t1,
        adjoint=config.adjoint,
        min_redshift=config.min_redshift,
        max_redshift=config.max_redshift,
        geometry=config.geometry,
        log_lightcone=False,
        log_ic=True,
        ells=config.ells,
        number_of_shells=config.number_of_shells,
        lensing=config.lensing,
        lpt_order=config.lpt_order,
    )
    # TODO: update to new API full_field_probmodel(template_field, config)
    full_field_basemodel = full_field_probmodel(config_inference)

    nbins = len(config.nz_shear)
    observed_model = condition(
        full_field_basemodel,
        {
            f"kappa_{i}": true_kappas_visible[f"kappa_{i}"]
            for i in range(nbins)
        },
    )

    print(f"Sampling with {args.sampler} using {args.backend} backend")
    print(
        f"Warmup: {args.num_warmup}, Samples: {args.num_samples}, Batches: {args.batch_count}"
    )

    batched_sampling(
        model=observed_model,
        path=str(samples_dir),
        rng_key=jax.random.PRNGKey(args.seed),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        batch_count=args.batch_count,
        sampler=args.sampler,
        backend=args.backend,
        save=True,
        init_params=init_params,
        progress_bar=True,
    )

    print("MCMC sampling completed")


def analyze_results(samples_dir, data_dir, plots_dir):
    print("\nLoading samples and analyzing results")

    scalar_samples = load_samples(str(samples_dir),
                                  param_names=["Omega_c", "sigma8"])
    print(f"Loaded {len(scalar_samples['Omega_c'])} samples")
    print(f"Parameters: {list(scalar_samples.keys())}")

    true_data = np.load(data_dir / "true_kappas.npz")
    true_Omega_c = float(true_data["Omega_c"])
    true_sigma8 = float(true_data["sigma8"])

    print("\nPosterior Statistics:")
    print(f"True Omega_c: {true_Omega_c:.4f}")
    print(
        f"Inferred Omega_c: {scalar_samples['Omega_c'].mean():.4f} ± {scalar_samples['Omega_c'].std():.4f}"
    )
    print(f"True sigma8: {true_sigma8:.4f}")
    print(
        f"Inferred sigma8: {scalar_samples['sigma8'].mean():.4f} ± {scalar_samples['sigma8'].std():.4f}"
    )

    print("Loading IC field statistics...")
    ic_mean, ic_std = load_samples(str(samples_dir),
                                   param_names=["ic"],
                                   transform=("mean", "std"))
    if "ic" in ic_mean:
        true_ic = np.load(data_dir / "true_ic.npy")
        plot_ic(
            true_ic,
            ic_mean["ic"],
            ic_std["ic"],
            plots_dir,
            titles=("True", "Mean", "Std", "Diff"),
        )
        print("Plotted IC comparison")

    param_samples = {
        "Omega_c": scalar_samples["Omega_c"],
        "sigma8": scalar_samples["sigma8"]
    }
    true_param_values = {"Omega_c": true_Omega_c, "sigma8": true_sigma8}

    labels = {"Omega_c": r"\Omega_c", "sigma8": r"\sigma_8"}

    plot_posterior(
        param_samples,
        plots_dir,
        params=("Omega_c", "sigma8"),
        true_values=true_param_values,
        labels=labels,
        filled=True,
        title_limit=1,
        width_inch=7,
    )
    print("Plotted posteriors")


def main():
    parser = argparse.ArgumentParser(
        description="Distributed Full-Field Inference")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_02_distributed",
        help="Output directory for plots and samples",
    )
    parser.add_argument(
        "--box-shape",
        type=int,
        nargs=3,
        default=[16, 16, 16],
        help="Simulation box shape (nx ny nz)",
    )
    parser.add_argument(
        "--max-redshift",
        type=float,
        default=0.5,
        help="Maximum redshift for the simulation",
    )
    parser.add_argument(
        "--number-of-shells",
        type=int,
        default=8,
        help="Number of density shells in lightcone",
    )
    parser.add_argument(
        "--geometry",
        type=str,
        choices=["flat", "spherical"],
        default="spherical",
        help="Geometry type",
    )
    parser.add_argument(
        "--observer-position",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 1.0],
        help="Observer position in box coordinates (half-sky)",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=20,
        help="Number of warmup steps for MCMC",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples per batch",
    )
    parser.add_argument(
        "--batch-count",
        type=int,
        default=2,
        help="Number of batches to run",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["NUTS", "HMC", "MCLMC"],
        default="MCLMC",
        help="MCMC sampler to use",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["numpyro", "blackjax"],
        default="blackjax",
        help="Sampling backend",
    )
    parser.add_argument(
        "--sigma-e",
        type=float,
        default=0.3,
        help="Intrinsic ellipticity dispersion",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--pdims",
        type=int,
        nargs=2,
        default=[4, 2],
        help="Processor dimensions for sharding mesh",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from existing samples",
    )

    args = parser.parse_args()

    print(f"JAX devices: {jax.device_count()}")
    print(f"JAX backend: {jax.default_backend()}")

    output_dir, plots_dir, samples_dir, data_dir = setup_output_dir(
        args.output_dir)

    if args.plot_only:
        print(
            "\nPlot-only mode: Loading existing samples and generating plots..."
        )
        analyze_results(samples_dir, data_dir, plots_dir)
        return

    sharding = setup_sharding(tuple(args.pdims))

    fiducial_cosmology = Planck18()
    observer_position = tuple(args.observer_position)
    # Physical box size inferred from max redshift and observer position
    box_size = compute_box_size_from_redshift(fiducial_cosmology,
                                              args.max_redshift,
                                              observer_position)

    print(f"Box size: {box_size} Mpc/h")
    print(f"Max redshift: {args.max_redshift}")
    print(f"Observer position (half-sky): {args.observer_position}")

    nz_shear, nbins = create_redshift_distribution(args.max_redshift)

    config = Configurations(
        density_plane_smoothing=0.1,
        nz_shear=nz_shear,
        fiducial_cosmology=Planck18,
        sigma_e=args.sigma_e,
        priors={
            "Omega_c": dist.Uniform(0.24, 0.28),
            "sigma8": dist.Uniform(0.78, 0.82),
        },
        t0=0.1,
        dt0=0.1,
        t1=1.0,
        adjoint=RecursiveCheckpointAdjoint(4),
        min_redshift=0.01,
        max_redshift=args.max_redshift,
        geometry=args.geometry,
        log_lightcone=True,
        log_ic=True,
        number_of_shells=args.number_of_shells,
    )

    # Attach geometry/box metadata that no longer lives in Configurations
    config.nside = args.box_shape[0]
    config.observer_position = observer_position
    config.box_size = box_size

    print("Configuration created")

    initial_conditions = normal_field(jax.random.key(args.seed),
                                      box_size,
                                      sharding=sharding)
    print("Initial conditions generated")

    true_kappas_visible = generate_synthetic_observations(
        config, fiducial_cosmology, initial_conditions, data_dir, plots_dir)

    init_params = {
        "Omega_c": fiducial_cosmology.Omega_c,
        "sigma8": fiducial_cosmology.sigma8,
        "initial_conditions": initial_conditions,
    }
    init_params = jax.tree.map(jnp.asarray, init_params)

    run_mcmc_inference(config, true_kappas_visible, samples_dir, args,
                       init_params)

    analyze_results(samples_dir, data_dir, plots_dir)

    print("\nWorkflow completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
