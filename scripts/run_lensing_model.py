#!/usr/bin/env python
"""
Lensing Bayesian inference workflow script.

Complete workflow:
1. Create probabilistic model with priors
2. Trace fiducial model to generate synthetic observations
3. Run MCMC sampling to infer cosmological parameters
4. Visualize and save results
"""

import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
os.environ['EQX_ON_ERROR'] = 'nan'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax

jax.config.update('jax_enable_x64', False)

print(f'JAX devices: {jax.device_count()}')
print(f'JAX backend: {jax.default_backend()}')

import argparse
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib

matplotlib.use('Agg')
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from diffrax import RecursiveCheckpointAdjoint
from jaxpm.distributed import normal_field
from numpyro.handlers import condition, seed, trace
from scipy.stats import norm

from fwd_model_tools import Configurations, Planck18, full_field_probmodel
from fwd_model_tools.sampling import batched_sampling, load_samples


def setup_output_dir(output_dir):
    """Create output directory structure."""
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    samples_dir = output_dir / 'samples'
    true_data_dir = output_dir / 'true_data'

    plots_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    true_data_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, plots_dir, samples_dir, true_data_dir


def setup_sharding(pdims=(4, 2)):
    """Setup distributed sharding if multiple devices available."""
    if jax.device_count() > 1:
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        mesh = jax.make_mesh(pdims, ('x', 'y'))
        sharding = NamedSharding(mesh, P('x', 'y'))
        print(f'Using sharding with mesh: {pdims}')
    else:
        sharding = None
        print('Single device mode - no sharding')

    return sharding


def create_redshift_distribution(
    cosmo,
    box_size,
    plots_dir,
    observer_position=(0.5, 0.5, 0.5),
    geometry="spherical",
    los_axis=2,
):
    """Create and visualize redshift distribution.

    Parameters
    - cosmo: jax_cosmo.Cosmology
    - box_size: sequence of 3 floats (Mpc/h)
    - plots_dir: Path to save plots
    - observer_position: (x, y, z) in [0, 1] fractions of box
    - geometry: 'spherical' or 'flat'
    - los_axis: line-of-sight axis index for flat sky (default: 2 -> z)
    """
    print('\n' + '=' * 60)
    print('Creating redshift distribution')
    print('=' * 60)

    # Compute a geometry-aware maximum comoving distance based on observer position
    bs = np.asarray(box_size, dtype=float)
    op = np.asarray(observer_position, dtype=float)

    # Apply the user's linear factor rule along the chosen LOS axis for both geometries
    #   factor(p) = 2 - |2p - 1|, with p in [0,1]
    #   p=0.5 -> factor=2 (half box); p=0 or 1 -> factor=1 (full box)
    p = float(np.clip(op[int(los_axis)], 0.0, 1.0))
    factor = 2.0 - abs(2.0 * p - 1.0)
    factor = float(np.clip(factor, 1.0, 2.0))
    max_comoving_distance = float(bs[int(los_axis)] / factor)

    max_redshift = (1 / jc.background.a_of_chi(cosmo, max_comoving_distance) -
                    1).squeeze()
    z = jnp.linspace(0, max_redshift, 1000)
    z_centers = jnp.linspace(0.2, max_redshift - 0.1, 4)
    z_centers = jnp.round(z_centers, 2)
    print(f'z_centers = {z_centers}')

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

    z_plot = np.linspace(0, 3.0, 128)
    plt.figure(figsize=(10, 6))
    for i in range(nbins):
        plt.plot(
            z_plot,
            nz_shear[i](z_plot) * nz_shear[i].gals_per_arcmin2,
            color=f'C{i}',
            label=f'Bin {i}',
        )
    plt.legend()
    plt.xlim(0, 3)
    plt.xlabel('Redshift z')
    plt.ylabel('Galaxies per arcmin²')
    plt.title('Source Redshift Distribution')
    plt.grid(True, alpha=0.3)
    plt.savefig(plots_dir / 'redshift_distribution.png',
                dpi=150,
                bbox_inches='tight')
    plt.close()
    print('✓ Saved redshift distribution plot')

    return nz_shear, nbins, max_redshift


def generate_synthetic_observations(config, fiducial_cosmology, initial_conditions, true_data_dir, plots_dir):
    """Generate synthetic observations by tracing the fiducial model."""
    print('\n' + '=' * 60)
    print('Generating synthetic observations')
    print('=' * 60)

    full_field_basemodel = full_field_probmodel(config)

    fiducial_model = condition(
        full_field_basemodel,
        {
            "Omega_c": fiducial_cosmology.Omega_c,
            "sigma8": fiducial_cosmology.sigma8,
            "initial_conditions": initial_conditions
        },
    )

    print('Tracing fiducial model to generate observables...')
    start_time = time.time()
    model_trace = trace(seed(fiducial_model, 0)).get_trace()
    elapsed = time.time() - start_time
    print(f'✓ Fiducial model traced in {elapsed:.2f}s')
    print(f'Logged parameters: {list(model_trace.keys())}')

    nbins = len(config.nz_shear)
    kappa_keys = [f"kappa_{i}" for i in range(nbins)]
    true_kappas = {key: model_trace[key]["value"] for key in kappa_keys}

    np.savez(
        true_data_dir / "true_kappas.npz",
        **true_kappas,
        Omega_c=fiducial_cosmology.Omega_c,
        sigma8=fiducial_cosmology.sigma8,
    )
    print(f'✓ Saved true kappas to {true_data_dir / "true_kappas.npz"}')

    print('Creating true kappa maps visualization...')
    plot_kappa_maps(true_kappas, nbins, config.geometry, plots_dir / 'true_kappa_maps.png', title_prefix='True')

    return model_trace, true_kappas


def plot_kappa_maps(kappas_dict, nbins, geometry, save_path, title_prefix=''):
    """Plot kappa maps for all redshift bins."""
    kappas = [kappas_dict[f"kappa_{i}"] for i in range(nbins)]

    all_kappas = np.concatenate([np.ravel(k) for k in kappas])
    all_kappas_valid = all_kappas[np.isfinite(all_kappas)]
    if len(all_kappas_valid) > 0:
        kappa_vmin = np.percentile(all_kappas_valid, 2)
        kappa_vmax = np.percentile(all_kappas_valid, 98)
    else:
        kappa_vmin = 0.0
        kappa_vmax = 1.0

    plt.figure(figsize=(15, 4))

    if geometry == 'spherical':
        for i in range(nbins):
            k_min = np.min(kappas[i])
            k_max = np.max(kappas[i])
            k_mean = np.mean(kappas[i])
            k_std = np.std(kappas[i])
            hp.mollview(
                kappas[i],
                title=
                f'{title_prefix} Kappa bin {i}\nmin={k_min:.2e}, max={k_max:.2e}\nmean={k_mean:.2e}, std={k_std:.2e}',
                sub=(1, nbins, i + 1),
                cmap='viridis',
                min=kappa_vmin,
                max=kappa_vmax,
                bgcolor=(0, ) * 4,
                cbar=True,
            )
    else:
        for i in range(nbins):
            k_min = np.min(kappas[i])
            k_max = np.max(kappas[i])
            k_mean = np.mean(kappas[i])
            k_std = np.std(kappas[i])
            ax = plt.subplot(1, nbins, i + 1)
            im = ax.imshow(
                kappas[i],
                cmap='viridis',
                origin='lower',
                vmin=kappa_vmin,
                vmax=kappa_vmax,
            )
            ax.set_title(
                f'{title_prefix} Kappa bin {i}\nmin={k_min:.2e}, max={k_max:.2e}\nmean={k_mean:.2e}, std={k_std:.2e}'
            )
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved kappa maps to {save_path}')


def run_mcmc_inference(config, true_kappas, samples_dir, args):
    """Run MCMC sampling to infer cosmological parameters."""
    print('\n' + '=' * 60)
    print('Running MCMC inference')
    print('=' * 60)

    full_field_basemodel = full_field_probmodel(config)

    nbins = len(config.nz_shear)
    observed_model = condition(
        full_field_basemodel,
        {f"kappa_{i}": true_kappas[f"kappa_{i}"] for i in range(nbins)},
    )

    print(f'Sampling with {args.sampler} using {args.backend} backend')
    print(f'Warmup: {args.num_warmup}, Samples: {args.num_samples}, Batches: {args.batch_count}')

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
    )

    print('✓ MCMC sampling completed')


def analyze_results(samples_dir, true_data_dir, plots_dir, config, args):
    """Load samples and create diagnostic plots."""
    print('\n' + '=' * 60)
    print('Analyzing results')
    print('=' * 60)

    print('Loading samples...')
    samples = load_samples(str(samples_dir))
    print(f'Loaded parameters: {list(samples.keys())}')
    print(f'Sample shapes: {dict((k, v.shape) for k, v in samples.items())}')

    true_data = np.load(true_data_dir / "true_kappas.npz")
    true_Omega_c = float(true_data['Omega_c'])
    true_sigma8 = float(true_data['sigma8'])

    print('\n' + '-' * 60)
    print('Posterior Statistics')
    print('-' * 60)
    print(f'True Omega_c: {true_Omega_c:.4f}')
    print(f'Inferred Omega_c: {samples["Omega_c"].mean():.4f} ± {samples["Omega_c"].std():.4f}')
    print(f'True sigma8: {true_sigma8:.4f}')
    print(f'Inferred sigma8: {samples["sigma8"].mean():.4f} ± {samples["sigma8"].std():.4f}')

    print('\nCreating posterior plots...')
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(samples["Omega_c"], bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(true_Omega_c, color='red', linestyle='--', linewidth=2, label='True value')
    plt.xlabel('Omega_c')
    plt.ylabel('Frequency')
    plt.title('Posterior: Omega_c')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(samples["sigma8"], bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(true_sigma8, color='red', linestyle='--', linewidth=2, label='True value')
    plt.xlabel('sigma8')
    plt.ylabel('Frequency')
    plt.title('Posterior: sigma8')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plots_dir / 'posterior_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ Saved posterior histograms')

    plt.figure(figsize=(8, 8))
    plt.scatter(samples["Omega_c"], samples["sigma8"], alpha=0.3, s=10)
    plt.scatter([true_Omega_c], [true_sigma8], color='red', s=100, marker='*',
                label='True values', zorder=10)
    plt.xlabel('Omega_c')
    plt.ylabel('sigma8')
    plt.title('Joint Posterior: Omega_c vs sigma8')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plots_dir / 'posterior_joint.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ Saved joint posterior plot')

    if 'initial_conditions' in samples:
        print('\nAnalyzing initial conditions recovery...')
        true_ic_path = true_data_dir / 'true_ic.npy'
        if true_ic_path.exists():
            true_ic = np.load(true_ic_path)
            mean_ic = samples["initial_conditions"].mean(axis=0)
            std_ic = samples["initial_conditions"].std(axis=0)

            slice_idx = true_ic.shape[-1] // 2
            plot_ic_comparison(true_ic[..., slice_idx], mean_ic[..., slice_idx],
                             std_ic[..., slice_idx], plots_dir / 'ic_comparison.png')


def plot_ic_comparison(true_ic_slice, mean_ic_slice, std_ic_slice, save_path):
    """Plot comparison of true vs recovered initial conditions."""
    residual_ic = mean_ic_slice - true_ic_slice

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, img, title in zip(
        axes,
        [true_ic_slice, mean_ic_slice, std_ic_slice, residual_ic],
        ["True IC", "Mean IC", "Std Dev IC", "Residuals IC"],
    ):
        im = ax.imshow(img, cmap="viridis")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved IC comparison to {save_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Run lensing Bayesian inference workflow')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for plots and samples',
    )
    parser.add_argument(
        '--box-shape',
        type=int,
        nargs=3,
        default=[256, 256, 256],
        help='Simulation box shape (nx ny nz)',
    )
    parser.add_argument(
        '--box-size',
        type=float,
        nargs=3,
        default=[2000.0, 2000.0, 2000.0],
        help='Simulation box size in Mpc/h (Lx Ly Lz)',
    )
    parser.add_argument(
        '--geometry',
        type=str,
        choices=['flat', 'spherical'],
        default='spherical',
        help="Geometry type: 'flat' for Cartesian, 'spherical' for HEALPix",
    )
    parser.add_argument(
        '--observer-position',
        type=float,
        nargs=3,
        default=[0.5, 0.5, 0.0],
        help='Observer position in box coordinates (x y z) between 0 and 1',
    )   
    parser.add_argument(
        '--num-warmup',
        type=int,
        default=50,
        help='Number of warmup steps for MCMC',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=50,
        help='Number of samples per batch',
    )
    parser.add_argument(
        '--batch-count',
        type=int,
        default=2,
        help='Number of batches to run',
    )
    parser.add_argument(
        '--sampler',
        type=str,
        choices=['NUTS', 'HMC', 'MCLMC'],
        default='MCLMC',
        help='MCMC sampler to use',
    )
    parser.add_argument(
        '--backend',
        type=str,
        choices=['numpyro', 'blackjax'],
        default='blackjax',
        help='Sampling backend',
    )
    parser.add_argument(
        '--sigma-e',
        type=float,
        default=0.3,
        help='Intrinsic ellipticity dispersion',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )

    args = parser.parse_args()

    print('=' * 60)
    print('LENSING BAYESIAN INFERENCE WORKFLOW')
    print('=' * 60)

    output_dir, plots_dir, samples_dir, true_data_dir = setup_output_dir(args.output_dir)
    sharding = setup_sharding()

    fiducial_cosmology = Planck18()

    nz_shear, nbins, max_redshift = create_redshift_distribution(
        fiducial_cosmology,
        args.box_size,
        plots_dir,
        observer_position=tuple(args.observer_position),
        geometry=args.geometry,
        los_axis=2,
    )

    config = Configurations(
        field_size=9.6,
        field_npix=args.box_shape[0] if args.geometry == 'flat' else 64,
        box_shape=tuple(args.box_shape),
        box_size=args.box_size,
        density_plane_width=100.0,
        density_plane_npix=args.box_shape[0],
        nside=args.box_shape[0],
        density_plane_smoothing=0.1,
        nz_shear=nz_shear,
        fiducial_cosmology=Planck18,
        sigma_e=args.sigma_e,
        priors={
            "Omega_c": dist.Uniform(0.2, 0.4),
            "sigma8": dist.Uniform(0.6, 1.0),
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
    )

    print('\nGenerating initial conditions...')
    initial_conditions = normal_field(jax.random.key(args.seed),
                                      config.box_shape,
                                      sharding=sharding)

    model_trace, true_kappas = generate_synthetic_observations(
        config, fiducial_cosmology, initial_conditions, true_data_dir, plots_dir)

    run_mcmc_inference(config, true_kappas, samples_dir, args)

    analyze_results(samples_dir, true_data_dir, plots_dir, config, args)

    print('\n' + '=' * 60)
    print('Workflow completed successfully!')
    print(f'Results saved to: {output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()
