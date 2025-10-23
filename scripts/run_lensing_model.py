#!/usr/bin/env python
"""
Lensing Bayesian inference workflow script.

Complete workflow:
1. Trace fiducial model to generate synthetic observations (with log_lightcone=True, log_ic=True)
2. Save true kappa, IC, lightcone and plot lightcone
3. Run MCMC sampling to infer cosmological parameters (with log_lightcone=False, log_ic=True)
4. Samples saved automatically by batched_sampling
5. Load samples and plot IC comparison + posterior
"""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["EQX_ON_ERROR"] = "nan"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
# =============================================================================
# 1. If running on a distributed system, initialize JAX distributed
# =============================================================================
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
# =============================================================================

import jax

jax.config.update("jax_enable_x64", False)

print(f"JAX devices: {jax.device_count()}")
print(f"JAX backend: {jax.default_backend()}")

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
from fwd_model_tools.lensing_model import (compute_box_size_from_redshift,
                                           compute_max_redshift_from_box_size)
from fwd_model_tools.plotting import (plot_ic, plot_kappa, plot_lightcone,
                                      plot_posterior)
from fwd_model_tools.sampling import batched_sampling, load_samples


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


def create_redshift_distribution(
    cosmo,
    box_size=None,
    observer_position=(0.5, 0.5, 0.5),
    geometry="spherical",
    max_redshift=None,
):
    print("\n" + "=" * 60)
    print("Creating redshift distribution")
    print("=" * 60)

    if box_size is None and max_redshift is None:
        raise ValueError("Either box_size or max_redshift must be provided")

    if box_size is None:
        box_size = compute_box_size_from_redshift(cosmo, max_redshift,
                                                  observer_position)
        print(
            f"Auto-computed box size: {box_size} Mpc/h for max redshift {max_redshift}"
        )
    elif max_redshift is None:
        max_redshift = compute_max_redshift_from_box_size(
            cosmo, box_size, observer_position)
        print(
            f"Auto-computed max redshift: {max_redshift} for box size {box_size} Mpc/h"
        )

    z = jnp.linspace(0, max_redshift, 1000)
    z_centers = jnp.linspace(0.2, max_redshift - 0.01, 4)
    print(f"z_centers = {z_centers}")
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

    return nz_shear, nbins, max_redshift, box_size


def generate_synthetic_observations(config, fiducial_cosmology,
                                    initial_conditions, data_dir, plots_dir):
    print("\n" + "=" * 60)
    print("Step 1: Generating synthetic observations")
    print("=" * 60)

    config_with_logging = config._replace(log_lightcone=True, log_ic=True)
    full_field_basemodel = full_field_probmodel(config_with_logging)

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
    print(f"✓ Fiducial model traced in {elapsed:.2f}s")

    nbins = len(config.nz_shear)
    kappa_keys = [f"kappa_{i}" for i in range(nbins)]
    true_kappas = {key: model_trace[key]["value"] for key in kappa_keys}

    print("\n" + "=" * 60)
    print("Step 2: Saving true data and plotting")
    print("=" * 60)

    np.savez(
        data_dir / "true_kappas.npz",
        **true_kappas,
        Omega_c=fiducial_cosmology.Omega_c,
        sigma8=fiducial_cosmology.sigma8,
    )
    print(f"✓ Saved true kappas to {data_dir / 'true_kappas.npz'}")

    true_ic = np.asarray(model_trace["ic"]["value"])
    np.save(data_dir / "true_ic.npy", true_ic)
    print(f"✓ Saved true IC to {data_dir / 'true_ic.npy'}")

    true_lightcone = np.asarray(model_trace["lightcone"]["value"])
    np.save(data_dir / "true_lightcone.npy", true_lightcone)
    print(f"✓ Saved true lightcone to {data_dir / 'true_lightcone.npy'}")

    plot_lightcone(true_lightcone,
                   plots_dir,
                   spherical=(config.geometry == "spherical"))
    print(f"✓ Plotted lightcone to {plots_dir / 'lightcone.png'}")

    kappa_array = np.stack([true_kappas[k] for k in kappa_keys])
    plot_kappa(kappa_array,
               plots_dir,
               spherical=(config.geometry == "spherical"))
    print(f"✓ Plotted kappa maps to {plots_dir / 'kappa_maps.png'}")

    return true_kappas


def run_mcmc_inference(config,
                       true_kappas,
                       samples_dir,
                       args,
                       init_params=None):
    print("\n" + "=" * 60)
    print("Step 3: Running MCMC inference")
    print("=" * 60)

    config_inference = config._replace(log_lightcone=False, log_ic=True)
    full_field_basemodel = full_field_probmodel(config_inference)

    nbins = len(config.nz_shear)
    observed_model = condition(
        full_field_basemodel,
        {f"kappa_{i}": true_kappas[f"kappa_{i}"]
         for i in range(nbins)},
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
    )

    print("✓ MCMC sampling completed")


def analyze_results(samples_dir, data_dir, plots_dir, n_samples_plot=10):
    print("\n" + "=" * 60)
    print("Step 5: Loading samples and plotting results")
    print("=" * 60)

    samples = load_samples(str(samples_dir))
    if n_samples_plot > 0:
        samples = jax.tree.map(lambda x: x[-n_samples_plot:], samples)
        print(f"Using last {n_samples_plot} samples for plotting")
    else:
        print("Using all samples for plotting")
    print(f"Loaded parameters: {list(samples.keys())}")

    true_data = np.load(data_dir / "true_kappas.npz")
    true_Omega_c = float(true_data["Omega_c"])
    true_sigma8 = float(true_data["sigma8"])

    print("\nPosterior Statistics:")
    print(f"True Omega_c: {true_Omega_c:.4f}")
    print(
        f"Inferred Omega_c: {samples['Omega_c'].mean():.4f} ± {samples['Omega_c'].std():.4f}"
    )
    print(f"True sigma8: {true_sigma8:.4f}")
    print(
        f"Inferred sigma8: {samples['sigma8'].mean():.4f} ± {samples['sigma8'].std():.4f}"
    )

    if "ic" in samples:
        true_ic = np.load(data_dir / "true_ic.npy")
        plot_ic(true_ic, samples["ic"], plots_dir)
        print(f"✓ Plotted IC comparison to {plots_dir / 'ic_comparison.png'}")

    param_samples = {
        "Omega_c": samples["Omega_c"],
        "sigma8": samples["sigma8"]
    }
    plot_posterior(param_samples, plots_dir, params=("Omega_c", "sigma8"))
    print(
        f"✓ Plotted posteriors to {plots_dir / 'posterior_trace.png'} and {plots_dir / 'posterior_pair.png'}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run lensing Bayesian inference workflow")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for plots and samples",
    )
    parser.add_argument(
        "--box-shape",
        type=int,
        nargs=3,
        default=[256, 256, 256],
        help="Simulation box shape (nx ny nz)",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        nargs=3,
        default=None,
        help=
        "Simulation box size in Mpc/h (Lx Ly Lz). If not provided, computed automatically from max redshift and observer position.",
    )
    parser.add_argument(
        "--max-redshift",
        type=float,
        default=None,
        help="Maximum redshift for the simulation",
    )
    parser.add_argument(
        "--geometry",
        type=str,
        choices=["flat", "spherical"],
        default="spherical",
        help="Geometry type: 'flat' for Cartesian, 'spherical' for HEALPix",
    )
    parser.add_argument(
        "--observer-position",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.0],
        help="Observer position in box coordinates (x y z) between 0 and 1",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=50,
        help="Number of warmup steps for MCMC",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
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
        "--plot-only",
        action="store_true",
        help=
        "Only generate plots from existing samples (skip observation generation and MCMC)",
    )
    parser.add_argument(
        "--n-samples-plot",
        type=int,
        default=10,
        help=
        "Number of last samples to use for plotting (default: 10, use -1 for all samples)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LENSING BAYESIAN INFERENCE WORKFLOW")
    print("=" * 60)

    output_dir, plots_dir, samples_dir, data_dir = setup_output_dir(
        args.output_dir)

    if args.plot_only:
        print(
            "\nPlot-only mode: Loading existing samples and generating plots..."
        )
        if not samples_dir.exists() or not any(
                samples_dir.glob("samples_*.npz")):
            raise FileNotFoundError(
                f"No sample files found in {samples_dir}. Run without --plot-only first to generate samples."
            )
        analyze_results(samples_dir,
                        data_dir,
                        plots_dir,
                        n_samples_plot=args.n_samples_plot)
    else:
        sharding = setup_sharding()

        fiducial_cosmology = Planck18()

        nz_shear, nbins, max_redshift, box_size = create_redshift_distribution(
            fiducial_cosmology,
            args.box_size,
            observer_position=tuple(args.observer_position),
            geometry=args.geometry,
            max_redshift=args.max_redshift,
        )

        config = Configurations(
            field_size=9.6,
            field_npix=args.box_shape[0] if args.geometry == "flat" else 64,
            box_shape=tuple(args.box_shape),
            box_size=box_size,
            density_plane_width=100.0,
            density_plane_npix=args.box_shape[0],
            nside=args.box_shape[0],
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
            min_redshift=0.01,
            max_redshift=max_redshift,
            sharding=sharding,
            halo_size=0 if sharding is None else args.box_shape[0] // 8,
            adjoint=RecursiveCheckpointAdjoint(4),
            geometry=args.geometry,
            observer_position=tuple(args.observer_position),
            log_lightcone=False,
            log_ic=False,
        )

        print("\nGenerating initial conditions...")
        initial_conditions = normal_field(jax.random.key(args.seed),
                                          config.box_shape,
                                          sharding=sharding)
        print("✓ Initial conditions generated with sharding\n")
        jax.debug.visualize_array_sharding(initial_conditions[:, :, 0])

        true_kappas = generate_synthetic_observations(config,
                                                      fiducial_cosmology,
                                                      initial_conditions,
                                                      data_dir, plots_dir)
        print("\n")

        init_params = {
            "Omega_c": fiducial_cosmology.Omega_c,
            "sigma8": fiducial_cosmology.sigma8,
            "initial_conditions": initial_conditions,
        }
        if args.num_warmup == 0:
            return
        run_mcmc_inference(config,
                           true_kappas,
                           samples_dir,
                           args,
                           init_params=init_params)

        analyze_results(samples_dir,
                        data_dir,
                        plots_dir,
                        n_samples_plot=args.n_samples_plot)

    print("\n" + "=" * 60)
    print("Workflow completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
