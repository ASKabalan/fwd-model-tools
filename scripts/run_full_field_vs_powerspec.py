#!/usr/bin/env python

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["EQX_ON_ERROR"] = "nan"
os.environ["JAX_ENABLE_X64"] = "False"
os.environ["JAX_PLATFORMS"] = "cpu"

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

import healpy as hp
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
from diffrax import RecursiveCheckpointAdjoint
from getdist import MCSamples
from getdist import plots as gdplots
from jaxpm.distributed import normal_field
from numpyro.handlers import condition, seed, trace
from scipy.stats import norm

from fwd_model_tools import Configurations, Planck18, full_field_probmodel, reconstruct_full_kappa
from fwd_model_tools.lensing_model import compute_box_size_from_redshift
from fwd_model_tools.plotting import plot_kappa, plot_lightcone
from fwd_model_tools.powerspec_model import powerspec_probmodel
from fwd_model_tools.sampling import batched_sampling, load_samples


def setup_output_dir(output_dir):
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    samples_dir_ff = output_dir / "samples_fullfield"
    samples_dir_ps = output_dir / "samples_powerspec"
    data_dir = output_dir / "data"

    plots_dir.mkdir(parents=True, exist_ok=True)
    samples_dir_ff.mkdir(parents=True, exist_ok=True)
    samples_dir_ps.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, plots_dir, samples_dir_ff, samples_dir_ps, data_dir


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
    print(f"Number of redshift bins: {nbins}")

    return nz_shear, nbins


def generate_synthetic_observations(config, fiducial_cosmology,
                                    initial_conditions, data_dir, plots_dir):
    print("\nGenerating synthetic observations")

    full_field_basemodel = full_field_probmodel(config)

    fiducial_model = condition(
        full_field_basemodel,
        {
            "Omega_c": fiducial_cosmology.Omega_c,
            "sigma8": fiducial_cosmology.sigma8,
            "initial_conditions": initial_conditions,
        },
    )

    print("Tracing fiducial model...")
    start_time = time.time()
    model_trace = trace(seed(fiducial_model, 0)).get_trace()
    elapsed = time.time() - start_time
    print(f"Fiducial model traced in {elapsed:.2f}s")

    nbins = len(config.nz_shear)
    kappa_keys = [f"kappa_{i}" for i in range(nbins)]
    true_kappas_visible = {key: model_trace[key]["value"] for key in kappa_keys}

    true_kappas_full = reconstruct_full_kappa(
        true_kappas_visible, config.nside, config.observer_position
    )

    np.savez(
        data_dir / "true_kappas.npz",
        **true_kappas_visible,
        Omega_c=fiducial_cosmology.Omega_c,
        sigma8=fiducial_cosmology.sigma8,
    )

    true_lightcone = np.asarray(model_trace["lightcone"]["value"])
    np.save(data_dir / "true_lightcone.npy", true_lightcone)

    print("Saved observations to disk")

    plot_lightcone(true_lightcone, plots_dir, spherical=True)
    print("Plotted lightcone")

    kappa_array = np.stack([true_kappas_full[k] for k in kappa_keys])
    plot_kappa(kappa_array, plots_dir, spherical=True)
    print("Plotted kappa maps")

    return true_kappas_visible, true_kappas_full, kappa_keys


def compute_power_spectra(true_kappas_full, kappa_keys, nbins, nside, data_dir, plots_dir):
    print("\nComputing power spectra with healpy.anafast")

    observed_cls = {}
    ell_max = 3 * nside - 1
    ell = np.arange(2, ell_max + 1)

    for i in range(nbins):
        for j in range(i, nbins):
            kappa_i = np.asarray(true_kappas_full[f"kappa_{i}"])
            kappa_j = np.asarray(true_kappas_full[f"kappa_{j}"])

            cl_full = hp.anafast(kappa_i, kappa_j, lmax=ell_max)
            cl_obs = cl_full[2:]

            observed_cls[(i, j)] = jnp.array(cl_obs)
            print(f"  C_ell^{{{i},{j}}}: {len(cl_obs)} multipoles")

    print(f"\nTotal power spectra computed: {len(observed_cls)}")
    print(f"Expected: 4 auto + 6 cross = 10 spectra")

    np.savez(
        data_dir / "observed_cls.npz",
        **{f"cl_{i}_{j}": observed_cls[(i, j)] for i, j in observed_cls.keys()},
        ell=ell,
    )

    fig, axes = plt.subplots(2, 5, figsize=(25, 8))
    axes = axes.ravel()

    for idx, (i, j) in enumerate(observed_cls.keys()):
        ax = axes[idx]
        cl = observed_cls[(i, j)]
        ax.loglog(ell, np.abs(cl), label=f"C_ell^{{{i},{j}}}")
        ax.set_xlabel("ell")
        ax.set_ylabel("C_ell")
        ax.set_title(f"Power Spectrum {i},{j}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / "power_spectra.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Plotted power spectra")

    return observed_cls, ell


def run_full_field_inference(config, true_kappas_visible, samples_dir_ff, args, init_params):
    print("\nSetting up full-field MCMC inference")

    config_inference = config._replace(log_lightcone=False, log_ic=False)
    full_field_basemodel = full_field_probmodel(config_inference)

    nbins = len(config.nz_shear)
    observed_model = condition(
        full_field_basemodel,
        {f"kappa_{i}": true_kappas_visible[f"kappa_{i}"] for i in range(nbins)},
    )

    print(f"Sampling with {args.sampler} using {args.backend} backend")

    batched_sampling(
        model=observed_model,
        path=str(samples_dir_ff),
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

    print("Full-field MCMC sampling completed")


def run_powerspec_inference(config, observed_cls, ell, samples_dir_ps, args, fiducial_cosmology):
    print("\nSetting up power spectrum MCMC inference")

    ell_jax = jnp.array(ell)

    def powerspec_model():
        powerspec_probmodel(config, ell_jax, observed_cls)

    init_params_ps = {
        "Omega_c": fiducial_cosmology.Omega_c,
        "sigma8": fiducial_cosmology.sigma8,
    }
    init_params_ps = jax.tree.map(jnp.asarray, init_params_ps)

    print(f"Sampling with {args.sampler} using {args.backend} backend")

    batched_sampling(
        model=powerspec_model,
        path=str(samples_dir_ps),
        rng_key=jax.random.PRNGKey(args.seed + 1),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        batch_count=args.batch_count,
        sampler=args.sampler,
        backend=args.backend,
        save=True,
        init_params=init_params_ps,
        progress_bar=True,
    )

    print("Power spectrum MCMC sampling completed")


def analyze_and_compare_results(samples_dir_ff, samples_dir_ps, data_dir, plots_dir):
    print("\nLoading and comparing samples")

    samples_ff = load_samples(str(samples_dir_ff))
    print(f"Loaded {len(samples_ff['Omega_c'])} full-field samples")

    samples_ps = load_samples(str(samples_dir_ps))
    print(f"Loaded {len(samples_ps['Omega_c'])} power spectrum samples")

    true_data = np.load(data_dir / "true_kappas.npz")
    true_Omega_c = float(true_data["Omega_c"])
    true_sigma8 = float(true_data["sigma8"])

    print("\nFull-Field Posterior Statistics:")
    print(f"Omega_c: {samples_ff['Omega_c'].mean():.4f} ± {samples_ff['Omega_c'].std():.4f}")
    print(f"sigma8: {samples_ff['sigma8'].mean():.4f} ± {samples_ff['sigma8'].std():.4f}")

    print("\nPower Spectrum Posterior Statistics:")
    print(f"Omega_c: {samples_ps['Omega_c'].mean():.4f} ± {samples_ps['Omega_c'].std():.4f}")
    print(f"sigma8: {samples_ps['sigma8'].mean():.4f} ± {samples_ps['sigma8'].std():.4f}")

    params = ("Omega_c", "sigma8")
    labels = [r"\Omega_c", r"\sigma_8"]

    samples_ff_array = np.column_stack([samples_ff[p] for p in params])
    mc_samples_ff = MCSamples(samples=samples_ff_array, names=params, labels=labels, label="Full-Field")

    samples_ps_array = np.column_stack([samples_ps[p] for p in params])
    mc_samples_ps = MCSamples(samples=samples_ps_array, names=params, labels=labels, label="Power Spectrum")

    markers_dict = {"Omega_c": true_Omega_c, "sigma8": true_sigma8}

    gdplt = gdplots.get_subplot_plotter(width_inch=8)

    gdplt.triangle_plot(
        [mc_samples_ff, mc_samples_ps],
        filled=True,
        contour_colors=["blue", "red"],
        markers=markers_dict,
        title_limit=1,
        legend_labels=["Full-Field", "Power Spectrum"],
    )

    plt.savefig(plots_dir / "comparison_posterior.png", dpi=600, bbox_inches="tight")
    plt.close()
    print("Plotted comparison posteriors")


def main():
    parser = argparse.ArgumentParser(
        description="Full-Field Sampling vs Power Spectrum Inference"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_03_comparison",
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
        help="Geometry type (must be spherical for power spectra)",
    )
    parser.add_argument(
        "--observer-position",
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.5],
        help="Observer position in box coordinates (full-sky)",
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

    if args.geometry != "spherical":
        raise ValueError("Geometry must be 'spherical' for power spectrum computation")

    print(f"JAX devices: {jax.device_count()}")
    print(f"JAX backend: {jax.default_backend()}")

    output_dir, plots_dir, samples_dir_ff, samples_dir_ps, data_dir = setup_output_dir(
        args.output_dir
    )

    if args.plot_only:
        print("\nPlot-only mode: Loading existing samples and generating plots...")
        analyze_and_compare_results(samples_dir_ff, samples_dir_ps, data_dir, plots_dir)
        return

    sharding = setup_sharding(tuple(args.pdims))

    fiducial_cosmology = Planck18()
    box_size = compute_box_size_from_redshift(
        fiducial_cosmology, args.max_redshift, tuple(args.observer_position)
    )

    print(f"Box size: {box_size} Mpc/h")
    print(f"Max redshift: {args.max_redshift}")
    print(f"Observer position (full-sky): {args.observer_position}")

    nz_shear, nbins = create_redshift_distribution(args.max_redshift)

    config = Configurations(
        field_size=9.6,
        field_npix=args.box_shape[0],
        box_shape=tuple(args.box_shape),
        box_size=box_size,
        number_of_shells=args.number_of_shells,
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
        max_redshift=args.max_redshift,
        sharding=sharding,
        halo_size=0 if sharding is None else args.box_shape[0] // 8,
        adjoint=RecursiveCheckpointAdjoint(4),
        geometry=args.geometry,
        observer_position=tuple(args.observer_position),
        log_lightcone=True,
        log_ic=False,
    )

    print("Configuration created")

    initial_conditions = normal_field(
        jax.random.key(args.seed), config.box_shape, sharding=sharding
    )
    print("Initial conditions generated")

    true_kappas_visible, true_kappas_full, kappa_keys = generate_synthetic_observations(
        config, fiducial_cosmology, initial_conditions, data_dir, plots_dir
    )

    observed_cls, ell = compute_power_spectra(
        true_kappas_full, kappa_keys, nbins, config.nside, data_dir, plots_dir
    )

    init_params = {
        "Omega_c": fiducial_cosmology.Omega_c,
        "sigma8": fiducial_cosmology.sigma8,
        "initial_conditions": initial_conditions,
    }
    init_params = jax.tree.map(jnp.asarray, init_params)

    run_full_field_inference(config, true_kappas_visible, samples_dir_ff, args, init_params)

    run_powerspec_inference(config, observed_cls, ell, samples_dir_ps, args, fiducial_cosmology)

    analyze_and_compare_results(samples_dir_ff, samples_dir_ps, data_dir, plots_dir)

    print("\nWorkflow completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
