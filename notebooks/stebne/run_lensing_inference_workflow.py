#!/usr/bin/env python3
"""
Lensing Bayesian Inference Workflow Script

This script demonstrates a complete Bayesian inference workflow for weak gravitational lensing:
1. Generate synthetic observations (convergence maps and lightcone)
2. Compute gradients of the forward model w.r.t. cosmological parameters
3. Infer cosmological parameters from observations using NUTS/HMC/MCLMC
4. Analyze results and plot posteriors

The script allows resuming from the MCMC sampling section without regenerating observations.
"""

import argparse
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
import numpyro.distributions as dist
from diffrax import RecursiveCheckpointAdjoint
from jaxpm.distributed import normal_field
from jaxpm.spherical import spherical_visibility_mask
from numpyro.handlers import condition, seed, trace
from scipy.stats import norm

from fwd_model_tools import Configurations, Planck18, full_field_probmodel
from fwd_model_tools.lensing_model import (compute_box_size_from_redshift,
                                           compute_max_redshift_from_box_size,
                                           make_full_field_model)
from fwd_model_tools.plotting import (plot_gradient_analysis, plot_ic,
                                      plot_kappa, plot_lightcone,
                                      plot_posterior)
from fwd_model_tools.sampling import batched_sampling, load_samples
from fwd_model_tools.utils import reconstruct_full_sphere


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


def generate_observations(config, fiducial_cosmology, initial_conditions,
                          data_dir):
    print("\n" + "=" * 60)
    print("Generating synthetic observations")
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

    # Keep visible-pixel kappas for inference
    true_kappas_visible = {
        key: model_trace[key]["value"]
        for key in kappa_keys
    }

    # Prepare full maps only for plotting (spherical geometry)
    if config.geometry == "spherical":
    true_kappas_full = reconstruct_full_sphere(
        true_kappas_visible, config.nside, config.observer_position
    )
    else:
        true_kappas_full = true_kappas_visible

    np.savez(
        data_dir / "true_kappas.npz",
        **true_kappas_visible,  # Save visible kappas for inference
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

    return true_kappas_visible, true_kappas_full, true_lightcone, kappa_keys


def compute_mse_loss(
    param_val,
    param_name,
    cosmo,
    nz_shear,
    ic,
    kappa_obs,
    forward_model,
    visible_indices,
):
    cosmo_dict = {
        "Omega_c": cosmo.Omega_c,
        "Omega_b": cosmo.Omega_b,
        "h": cosmo.h,
        "n_s": cosmo.n_s,
        "sigma8": cosmo.sigma8,
        "Omega_k": cosmo.Omega_k,
        "w0": cosmo.w0,
        "wa": cosmo.wa,
    }
    cosmo_dict[param_name] = param_val
    test_cosmo = jc.Cosmology(**cosmo_dict)
    test_cosmo._workspace = {}

    kappas, _, _ = forward_model(test_cosmo, nz_shear, ic)
    kappas_visible = [k[visible_indices] for k in kappas]

    mse = sum(
        jnp.mean((k - kappa_obs[f"kappa_{i}"])**2)
        for i, k in enumerate(kappas_visible))
    return mse / len(kappas_visible)


def compute_loss_and_gradient(
    param_val,
    param_name,
    cosmo,
    nz_shear,
    ic,
    kappa_obs,
    forward_model,
    visible_indices,
):
    loss_fn = lambda pval: compute_mse_loss(pval, param_name, cosmo, nz_shear,
                                            ic, kappa_obs, forward_model,
                                            visible_indices)
    loss_value = float(loss_fn(param_val))
    grad_value = float(jax.grad(loss_fn)(param_val))
    return loss_value, grad_value


def compute_gradients(
    config,
    fiducial_cosmology,
    nz_shear,
    initial_conditions,
    true_kappas_visible,
    gradient_offset_omega_c,
    gradient_offset_sigma8,
    plots_dir,
):
    print("\n" + "=" * 60)
    print("Computing parameter gradients")
    print("=" * 60)

    visible_mask = spherical_visibility_mask(config.nside,
                                             config.observer_position)
    visible_indices, = jnp.where(visible_mask == 1)

    forward_model = make_full_field_model(
        field_size=config.field_size,
        field_npix=config.field_npix,
        box_size=config.box_size,
        box_size=config.box_size,
        density_plane_width=config.density_plane_width,
        density_plane_npix=config.density_plane_npix,
        density_plane_smoothing=config.density_plane_smoothing,
        nside=config.nside,
        adjoint=config.adjoint,
        t0=config.t0,
        dt0=config.dt0,
        t1=config.t1,
        min_redshift=config.min_redshift,
        max_redshift=config.max_redshift,
        sharding=config.sharding,
        halo_size=config.halo_size,
        geometry=config.geometry,
        observer_position=config.observer_position,
    )
    print("Forward model created")

    params_to_test = {
        "Omega_c": {
            "fiducial": fiducial_cosmology.Omega_c,
            "offset": gradient_offset_omega_c,
        },
        "sigma8": {
            "fiducial": fiducial_cosmology.sigma8,
            "offset": gradient_offset_sigma8,
        },
    }

    results = {}

    for param_name, param_info in params_to_test.items():
        print(f"\nComputing gradients for {param_name}")
        print("-" * 60)

        fiducial_val = param_info["fiducial"]
        offset = param_info["offset"]

        offsets = np.array([-2 * offset, -offset, 0.0, offset, 2 * offset])
        values = fiducial_val + offsets

        losses = []
        gradients = []
        for i, (off, val) in enumerate(zip(offsets, values)):
            print(
                f"  [{i+1}/5] {param_name} = {val:.4f} (offset = {off:+.4f})")

            loss_val, grad_val = compute_loss_and_gradient(
                val, param_name, fiducial_cosmology, nz_shear,
                initial_conditions, true_kappas_visible, forward_model,
                visible_indices)
            losses.append(loss_val)
            gradients.append(grad_val)

            print(f"        MSE Loss = {loss_val:.6e}")
            print(f"        d(MSE)/d({param_name}) = {grad_val:.6e}")

        results[param_name] = {
            "offsets": offsets,
            "losses": np.array(losses),
            "gradients": np.array(gradients)
        }

    print("\n" + "=" * 60)
    print("Gradient computation completed")
    print("=" * 60)
    print(
        "Expected: Quadratic loss (parabola) and linear gradient (through 0)")

    plot_gradient_analysis(results,
                           params_to_test,
                           plots_dir,
                           output_format="png",
                           dpi=600)
    print(
        f"\nGradient analysis plots saved to {plots_dir / 'gradient_analysis.png'}"
    )


def run_mcmc_sampling(
    config,
    fiducial_cosmology,
    initial_conditions,
    data_dir,
    samples_dir,
    num_warmup,
    num_samples,
    batch_count,
    sampler,
    backend,
    magick_seed,
):
    print("\n" + "=" * 60)
    print("Setting up MCMC inference")
    print("=" * 60)

    nbins = len(config.nz_shear)
    true_data = np.load(data_dir / "true_kappas.npz")
    true_kappas_loaded = {
        f"kappa_{i}": true_data[f"kappa_{i}"]
        for i in range(nbins)
    }

    config_inference = config._replace(log_lightcone=False, log_ic=True)
    full_field_basemodel = full_field_probmodel(config_inference)

    observed_model = condition(
        full_field_basemodel,
        {f"kappa_{i}": true_kappas_loaded[f"kappa_{i}"]
         for i in range(nbins)},
    )

    init_params = {
        "Omega_c": fiducial_cosmology.Omega_c,
        "sigma8": fiducial_cosmology.sigma8,
        "initial_conditions": initial_conditions,
    }
    init_params = jax.tree.map(jnp.asarray, init_params)

    print(f"Sampling with {sampler} using {backend} backend")
    print(
        f"Warmup: {num_warmup}, Samples: {num_samples}, Batches: {batch_count}"
    )

    batched_sampling(
        model=observed_model,
        path=str(samples_dir),
        rng_key=jax.random.PRNGKey(magick_seed),
        num_warmup=num_warmup,
        num_samples=num_samples,
        batch_count=batch_count,
        sampler=sampler,
        backend=backend,
        save=True,
        init_params=init_params,
        progress_bar=True,
    )

    print("✓ MCMC sampling completed")


def analyze_results(data_dir, samples_dir, plots_dir, n_samples_plot=-1):
    print("\n" + "=" * 60)
    print("Loading samples and analyzing results")
    print("=" * 60)

    scalar_samples = load_samples(str(samples_dir),
                                  param_names=["Omega_c", "sigma8"])
    if n_samples_plot > 0:
        scalar_samples = jax.tree.map(lambda x: x[-n_samples_plot:],
                                      scalar_samples)
        print(f"Using last {n_samples_plot} samples for plotting")
    else:
        print(
            f"Using all {len(scalar_samples['Omega_c'])} samples for plotting")

    print(f"Loaded parameters: {list(scalar_samples.keys())}")

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

    print("\nLoading IC field statistics...")
    ic_mean, ic_std = load_samples(str(samples_dir),
                                   param_names=["ic"],
                                   transform=("mean", "std"))
    if "ic" in ic_mean:
        true_ic = np.load(data_dir / "true_ic.npy")
        plot_ic(true_ic,
                ic_mean["ic"],
                ic_std["ic"],
                plots_dir,
                output_format="png",
                dpi=600)
        print(f"✓ Plotted IC comparison to {plots_dir / 'ic_comparison.png'}")

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
        output_format="png",
        filled=True,
        title_limit=1,
        width_inch=7,
        dpi=600,
    )
    print(f"✓ Plotted posteriors to {plots_dir / 'posterior.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="Lensing Bayesian Inference Workflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output-dir",
                        type=str,
                        default="output_workflow",
                        help="Output directory")
    parser.add_argument("--box-shape",
                        type=int,
                        nargs=3,
                        default=[16, 16, 16],
                        help="Box shape (nx ny nz)")
    parser.add_argument("--box-size",
                        type=float,
                        nargs=3,
                        default=None,
                        help="Box size in Mpc/h (x y z)")
    parser.add_argument("--max-redshift",
                        type=float,
                        default=0.5,
                        help="Maximum redshift")
    parser.add_argument("--geometry",
                        type=str,
                        default="spherical",
                        choices=["spherical", "flat"],
                        help="Geometry type")
    parser.add_argument(
        "--observer-position",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 1.0],
        help="Observer position as fraction of box size",
    )
    parser.add_argument("--num-warmup",
                        type=int,
                        default=100,
                        help="Number of warmup samples")
    parser.add_argument("--num-samples",
                        type=int,
                        default=10,
                        help="Number of samples per batch")
    parser.add_argument("--batch-count",
                        type=int,
                        default=2,
                        help="Number of batches")
    parser.add_argument("--sampler",
                        type=str,
                        default="MCLMC",
                        choices=["NUTS", "HMC", "MCLMC"],
                        help="Sampler type")
    parser.add_argument("--backend",
                        type=str,
                        default="blackjax",
                        choices=["blackjax", "numpyro"],
                        help="Sampling backend")
    parser.add_argument("--sigma-e",
                        type=float,
                        default=0.3,
                        help="Shape noise")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pdims",
                        type=int,
                        nargs=2,
                        default=[4, 2],
                        help="Processor mesh dimensions")
    parser.add_argument("--gradient-offset-omega-c",
                        type=float,
                        default=0.1,
                        help="Gradient offset for Omega_c")
    parser.add_argument("--gradient-offset-sigma8",
                        type=float,
                        default=0.1,
                        help="Gradient offset for sigma8")
    parser.add_argument("--skip-observations",
                        action="store_true",
                        help="Skip observation generation (use existing data)")
    parser.add_argument("--skip-gradients",
                        action="store_true",
                        help="Skip gradient computation")
    parser.add_argument("--skip-sampling",
                        action="store_true",
                        help="Skip MCMC sampling")
    parser.add_argument("--skip-analysis",
                        action="store_true",
                        help="Skip results analysis")
    parser.add_argument("--jax-platform",
                        type=str,
                        default="cpu",
                        choices=["cpu", "gpu"],
                        help="JAX platform")
    parser.add_argument("--n-samples-plot",
                        type=int,
                        default=-1,
                        help="Number of samples to plot (-1 for all)")

    args = parser.parse_args()

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
    os.environ["EQX_ON_ERROR"] = "nan"
    os.environ["JAX_ENABLE_X64"] = "False"
    os.environ["JAX_PLATFORM_NAME"] = args.jax_platform
    os.environ["JAX_PLATFORMS"] = args.jax_platform

    jax.config.update("jax_enable_x64", False)

    print(f"JAX devices: {jax.device_count()}")
    print(f"JAX backend: {jax.default_backend()}")

    output_dir_path, plots_dir, samples_dir, data_dir = setup_output_dir(
        args.output_dir)
    sharding = setup_sharding(tuple(args.pdims))
    fiducial_cosmology = Planck18()

    nz_shear, nbins, max_redshift, box_size = create_redshift_distribution(
        fiducial_cosmology,
        tuple(args.box_size) if args.box_size else None,
        observer_position=tuple(args.observer_position),
        geometry=args.geometry,
        max_redshift=args.max_redshift,
    )

    config = Configurations(
        field_size=9.6,
        field_npix=args.box_size[0],
        box_size=tuple(args.box_size),
        box_size=box_size,
        density_plane_width=100.0,
        density_plane_npix=args.box_size[0],
        nside=args.box_size[0],
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
        halo_size=0 if sharding is None else args.box_size[0] // 8,
        adjoint=RecursiveCheckpointAdjoint(4),
        geometry=args.geometry,
        observer_position=tuple(args.observer_position),
        log_lightcone=False,
        log_ic=False,
    )

    print("\nGenerating initial conditions...")
    initial_conditions = normal_field(jax.random.key(args.seed),
                                      config.box_size,
                                      sharding=sharding)
    print("✓ Initial conditions generated")

    if not args.skip_observations:
        true_kappas_visible, true_kappas_full, true_lightcone, kappa_keys = generate_observations(
            config, fiducial_cosmology, initial_conditions, data_dir)

        plot_lightcone(true_lightcone,
                       plots_dir,
                       spherical=(args.geometry == "spherical"),
                       output_format="png",
                       dpi=600)
        print(f"✓ Plotted lightcone to {plots_dir / 'lightcone.png'}")

        kappa_array = np.stack([true_kappas_full[k] for k in kappa_keys])
        plot_kappa(kappa_array,
                   plots_dir,
                   spherical=(args.geometry == "spherical"),
                   output_format="png",
                   dpi=600)
        print(f"✓ Plotted kappa maps to {plots_dir / 'kappa_maps.png'}")
    else:
        print("\nSkipping observation generation (using existing data)")
        true_data = np.load(data_dir / "true_kappas.npz")
        true_kappas_visible = {
            f"kappa_{i}": true_data[f"kappa_{i}"]
            for i in range(nbins)
        }

    if not args.skip_gradients:
        compute_gradients(
            config,
            fiducial_cosmology,
            nz_shear,
            initial_conditions,
            true_kappas_visible,
            args.gradient_offset_omega_c,
            args.gradient_offset_sigma8,
            plots_dir,
        )
    else:
        print("\nSkipping gradient computation")

    if not args.skip_sampling:
        run_mcmc_sampling(
            config,
            fiducial_cosmology,
            initial_conditions,
            data_dir,
            samples_dir,
            args.num_warmup,
            args.num_samples,
            args.batch_count,
            args.sampler,
            args.backend,
            args.seed,
        )
    else:
        print("\nSkipping MCMC sampling")

    if not args.skip_analysis:
        analyze_results(data_dir, samples_dir, plots_dir, args.n_samples_plot)
    else:
        print("\nSkipping results analysis")

    print("\n" + "=" * 60)
    print("Workflow completed!")
    print("=" * 60)
    print(f"Output directory: {output_dir_path}")
    print(f"  - plots/gradient_analysis.png: Loss and gradient analysis")
    print(f"  - plots/kappa_maps.png: Convergence maps")
    print(f"  - plots/lightcone.png: Density planes")
    print(f"  - plots/ic_comparison.png: Initial conditions comparison")
    print(f"  - plots/posterior.png: Posterior distributions")
    print(f"  - samples/: MCMC samples (can be reloaded)")
    print(f"  - data/: True observations and initial conditions")


if __name__ == "__main__":
    main()
