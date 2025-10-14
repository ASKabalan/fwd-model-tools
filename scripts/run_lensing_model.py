#!/usr/bin/env python
"""
Lensing inference workflow script.

Converts the 03-lensing_model.ipynb notebook into a standalone script
that saves all plots as PNG files and provides progress logging.
"""

import os

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.95'
os.environ['EQX_ON_ERROR'] = 'nan'
import jax

jax.config.update('jax_enable_x64', False)

print(f'JAX devices: {jax.device_count()}')
print(f'JAX backend: {jax.default_backend()}')

import argparse
import itertools
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib

matplotlib.use('Agg')
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from diffrax import RecursiveCheckpointAdjoint
from scipy.stats import norm

from fwd_model_tools import Planck18, make_full_field_model
from jaxpm.distributed import normal_field


def setup_output_dir(output_dir):
    """Create output directory structure."""
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    samples_dir = output_dir / 'samples'

    plots_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, plots_dir, samples_dir


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


def create_redshift_distribution(cosmo, box_size, plots_dir):
    """Create and visualize redshift distribution."""
    print('\n' + '=' * 60)
    print('Creating redshift distribution')
    print('=' * 60)

    max_comoving_distance = float(min(box_size) / 2.0)
    max_redshift = (1 / jc.background.a_of_chi(cosmo, max_comoving_distance) - 1).squeeze()
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
        )
        for z_center, g in zip(z_centers, [7, 8.5, 7.5, 7])
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
    plt.savefig(plots_dir / 'redshift_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ Saved redshift distribution plot')

    return nz_shear, nbins, max_redshift


def run_forward_model(config, cosmo, nz_shear, initial_conditions, plots_dir):
    """Run forward model and create visualizations."""
    print('\n' + '=' * 60)
    print('Running forward model')
    print('=' * 60)

    forward_model = make_full_field_model(
        field_size=config['field_size'],
        field_npix=config['field_npix'],
        box_shape=config['box_shape'],
        box_size=config['box_size'],
        density_plane_width=config['density_plane_width'],
        density_plane_npix=config['density_plane_npix'],
        density_plane_smoothing=config['density_plane_smoothing'],
        nside=config['nside'],
        adjoint=RecursiveCheckpointAdjoint(checkpoints=4),
        t0=config['t0'],
        t1=config['t1'],
        dt0=config['dt0'],
        max_redshift=config['max_redshift'],
        sharding=config['sharding'],
        halo_size=0 if config['sharding'] is None else config['box_shape'][0] // 8,
        geometry=config['geometry'],
    )

    start_time = time.time()
    print('Computing convergence maps...')
    kappas, lc, ic = forward_model(cosmo, nz_shear, initial_conditions)
    elapsed = time.time() - start_time
    print(f'✓ Forward model completed in {elapsed:.2f}s')
    print(f'  Lightcone shape: {lc.shape}')
    print(f'  Kappa maps: {len(kappas)} bins')

    print('Creating lightcone visualization...')
    n_planes = lc.shape[0]
    n_show = min(4, n_planes)
    n_cols = 3
    n_rows = n_show // n_cols + (n_show % n_cols > 0)
    print(f'  Showing last {n_show} of {n_planes} planes')
    print(f'n_rows = {n_rows}, n_cols = {n_cols} for n_show = {n_show} Geometry = {config["geometry"]}')

    lc_vmin = np.percentile(lc, 2)
    lc_vmax = np.percentile(lc, 98)
    print(f'  Lightcone range: [{lc_vmin:.3e}, {lc_vmax:.3e}] (2nd-98th percentile)')

    plt.figure(figsize=(12, 4 * n_rows))

    if config['geometry'] == 'spherical':
        for i in range(0, n_show):
            idx = n_planes - n_show + i
            plane_min = np.min(lc[idx])
            plane_max = np.max(lc[idx])
            plane_mean = np.mean(lc[idx])
            hp.mollview(
                lc[idx],
                title=f'Lightcone plane {idx}\nmin={plane_min:.2e}, max={plane_max:.2e}, mean={plane_mean:.2e}',
                sub=(n_rows, n_cols, i + 1),
                cmap='magma',
                min=lc_vmin,
                max=lc_vmax,
                bgcolor=(0,) * 4,
                cbar=True,
            )
    else:
        for i in range(0, n_show):
            idx = n_planes - n_show + i
            plane_min = np.min(lc[idx])
            plane_max = np.max(lc[idx])
            plane_mean = np.mean(lc[idx])
            ax = plt.subplot(n_rows, n_cols, i + 1)
            im = ax.imshow(lc[idx], cmap='magma', origin='lower', vmin=lc_vmin, vmax=lc_vmax)
            ax.set_title(f'Lightcone plane {idx}\nmin={plane_min:.2e}, max={plane_max:.2e}, mean={plane_mean:.2e}')
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(
        plots_dir / f'{config["geometry"]}_lightcone_planes.png',
        dpi=150,
        bbox_inches='tight',
    )
    plt.close()
    print('✓ Saved lightcone visualization')

    print('Creating kappa maps visualization...')
    nbins = len(kappas)

    all_kappas = np.concatenate([np.ravel(k) for k in kappas])
    all_kappas_valid = all_kappas[np.isfinite(all_kappas)]
    if len(all_kappas_valid) > 0:
        kappa_vmin = np.percentile(all_kappas_valid, 2)
        kappa_vmax = np.percentile(all_kappas_valid, 98)
    else:
        kappa_vmin = 0.0
        kappa_vmax = 1.0
    print(f'  Kappa range: [{kappa_vmin:.3e}, {kappa_vmax:.3e}] (2nd-98th percentile)')
    print(
        f'  Valid pixels: {len(all_kappas_valid)}/{len(all_kappas)} ({100 * len(all_kappas_valid) / len(all_kappas):.1f}%)'
    )

    plt.figure(figsize=(15, 4))

    if config['geometry'] == 'spherical':
        for i in range(nbins):
            k_min = np.min(kappas[i])
            k_max = np.max(kappas[i])
            k_mean = np.mean(kappas[i])
            k_std = np.std(kappas[i])
            hp.mollview(
                kappas[i],
                title=f'Kappa bin {i}\nmin={k_min:.2e}, max={k_max:.2e}\nmean={k_mean:.2e}, std={k_std:.2e}',
                sub=(1, nbins, i + 1),
                cmap='viridis',
                min=kappa_vmin,
                max=kappa_vmax,
                bgcolor=(0,) * 4,
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
            ax.set_title(f'Kappa bin {i}\nmin={k_min:.2e}, max={k_max:.2e}\nmean={k_mean:.2e}, std={k_std:.2e}')
            plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(plots_dir / f'{config["geometry"]}_kappa_maps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('✓ Saved kappa maps visualization')

    return forward_model, kappas, lc, ic


def main():
    parser = argparse.ArgumentParser(description='Run lensing inference workflow and save plots')
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
        default=[4000.0, 4000.0, 4000.0],
        help='Simulation box size in Mpc/h (Lx Ly Lz)',
    )
    parser.add_argument(
        '--geometry',
        type=str,
        choices=['flat', 'spherical'],
        default='spherical',
        help="Geometry type: 'flat' for Cartesian, 'spherical' for HEALPix",
    )

    args = parser.parse_args()

    print('=' * 60)
    print('LENSING INFERENCE WORKFLOW')
    print('=' * 60)

    output_dir, plots_dir, samples_dir = setup_output_dir(args.output_dir)
    sharding = setup_sharding()

    config = {
        'box_shape': tuple(args.box_shape),
        'box_size': args.box_size,
        'field_size': 9.6,
        'field_npix': int(args.box_shape[0]),
        'density_plane_width': 100.0,
        'density_plane_npix': int(args.box_shape[0]),
        'density_plane_smoothing': 0.1,
        'nside': 256,
        't0': 0.1,
        't1': 1.0,
        'dt0': 0.1,
        'sharding': sharding,
        'geometry': args.geometry,
    }

    cosmo = Planck18()

    nz_shear, nbins, max_redshift = create_redshift_distribution(cosmo, config['box_size'], plots_dir)
    config['max_redshift'] = max_redshift

    print('\nGenerating initial conditions...')
    initial_conditions = normal_field(jax.random.key(0), config['box_shape'], sharding=sharding)

    forward_model, kappas, lc, ic = run_forward_model(config, cosmo, nz_shear, initial_conditions, plots_dir)

    print('\n' + '=' * 60)
    print('Workflow completed successfully!')
    print(f'Results saved to: {output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()
