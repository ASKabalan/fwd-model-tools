#!/usr/bin/env python

import argparse
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from fwd_model_tools.sampling import load_samples


def plot_initial_conditions_comparison(samples, output_dir):
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    initial_conditions = samples['initial_conditions']

    true_ic = initial_conditions[0]
    mean_ic = initial_conditions.mean(axis=0)
    std_ic = initial_conditions.std(axis=0)
    residual_ic = mean_ic - true_ic

    slice_idx = true_ic.shape[-1] // 2

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    data_slices = [
        (true_ic[..., slice_idx], "True Initial Conditions"),
        (mean_ic[..., slice_idx], "Mean Sampled IC"),
        (std_ic[..., slice_idx], "Std Dev Sampled IC"),
        (residual_ic[..., slice_idx], "Residual (Mean - True)")
    ]

    for ax, (data, title) in zip(axes.flat, data_slices):
        im = ax.imshow(data, cmap='viridis', origin='lower')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        stats_text = f'min={np.min(data):.3e}\nmax={np.max(data):.3e}\nmean={np.mean(data):.3e}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_path = plots_dir / 'initial_conditions_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved IC comparison to {save_path}')


def plot_posterior_diagnostics(samples, true_values, output_dir):
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_samples = len(samples['Omega_c'])
    n_chains = 1
    if n_samples > 1:
        n_chains = 2
        samples_per_chain = n_samples // 2
        samples_dict = {
            'Omega_c': samples['Omega_c'][:samples_per_chain * 2].reshape(n_chains, samples_per_chain),
            'sigma8': samples['sigma8'][:samples_per_chain * 2].reshape(n_chains, samples_per_chain)
        }
    else:
        samples_dict = {
            'Omega_c': samples['Omega_c'].reshape(1, -1),
            'sigma8': samples['sigma8'].reshape(1, -1)
        }

    idata = az.from_dict(posterior=samples_dict)

    print('\n' + '=' * 60)
    print('ArviZ Summary Statistics')
    print('=' * 60)
    print(az.summary(idata, var_names=['Omega_c', 'sigma8']))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    az.plot_trace(idata, var_names=['Omega_c'], axes=axes[0, :])
    axes[0, 1].axvline(true_values['Omega_c'], color='red', linestyle='--',
                       linewidth=2, label='True value')
    axes[0, 1].legend()

    az.plot_trace(idata, var_names=['sigma8'], axes=axes[1, :])
    axes[1, 1].axvline(true_values['sigma8'], color='red', linestyle='--',
                       linewidth=2, label='True value')
    axes[1, 1].legend()

    plt.tight_layout()
    save_path = plots_dir / 'posterior_trace_arviz.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved trace plots to {save_path}')

    fig = plt.figure(figsize=(14, 5))

    ax1 = plt.subplot(1, 2, 1)
    az.plot_posterior(idata, var_names=['Omega_c'], ax=ax1, ref_val=true_values['Omega_c'])
    ax1.set_title('Posterior: Omega_c')

    ax2 = plt.subplot(1, 2, 2)
    az.plot_posterior(idata, var_names=['sigma8'], ax=ax2, ref_val=true_values['sigma8'])
    ax2.set_title('Posterior: sigma8')

    plt.tight_layout()
    save_path = plots_dir / 'posterior_distributions_arviz.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved posterior distributions to {save_path}')

    fig, ax = plt.subplots(figsize=(10, 8))

    Omega_c_flat = samples['Omega_c']
    sigma8_flat = samples['sigma8']

    H, xedges, yedges = np.histogram2d(Omega_c_flat, sigma8_flat, bins=30)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    ax.contourf(H.T, extent=extent, levels=15, cmap='Blues', alpha=0.6)
    ax.contour(H.T, extent=extent, levels=10, colors='black', linewidths=0.5, alpha=0.3)

    ax.scatter(Omega_c_flat, sigma8_flat, alpha=0.2, s=5, color='steelblue', label='Samples')
    ax.scatter([true_values['Omega_c']], [true_values['sigma8']],
               color='red', s=200, marker='*', edgecolors='black', linewidths=1.5,
               label='True values', zorder=10)

    ax.set_xlabel('Omega_c', fontsize=12)
    ax.set_ylabel('sigma8', fontsize=12)
    ax.set_title('Joint Posterior: Omega_c vs sigma8', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = plots_dir / 'posterior_joint_arviz.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved joint posterior to {save_path}')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    az.plot_autocorr(idata, var_names=['Omega_c'], ax=axes[0])
    axes[0].set_title('Autocorrelation: Omega_c')

    az.plot_autocorr(idata, var_names=['sigma8'], ax=axes[1])
    axes[1].set_title('Autocorrelation: sigma8')

    plt.tight_layout()
    save_path = plots_dir / 'autocorrelation_arviz.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved autocorrelation plots to {save_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Plot MCMC results: initial conditions and posteriors using ArviZ'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory containing samples and true_data subdirectories'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / 'samples'
    true_data_dir = output_dir / 'true_data'

    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    if not true_data_dir.exists():
        raise FileNotFoundError(f"True data directory not found: {true_data_dir}")

    print('=' * 60)
    print('PLOTTING MCMC RESULTS')
    print('=' * 60)

    print('\nLoading samples...')
    samples = load_samples(str(samples_dir))
    print(f'Loaded parameters: {list(samples.keys())}')
    print(f'Sample shapes: {dict((k, v.shape) for k, v in samples.items())}')

    print('\nLoading true values...')
    true_data = np.load(true_data_dir / 'true_kappas.npz')
    true_values = {
        'Omega_c': float(true_data['Omega_c']),
        'sigma8': float(true_data['sigma8'])
    }
    print(f'True Omega_c: {true_values["Omega_c"]:.4f}')
    print(f'True sigma8: {true_values["sigma8"]:.4f}')

    if 'initial_conditions' in samples:
        print('\n' + '-' * 60)
        print('Plotting Initial Conditions Comparison')
        print('-' * 60)
        plot_initial_conditions_comparison(samples, output_dir)
    else:
        print('\n⚠ Warning: No initial_conditions found in samples')

    print('\n' + '-' * 60)
    print('Plotting Posterior Diagnostics with ArviZ')
    print('-' * 60)
    plot_posterior_diagnostics(samples, true_values, output_dir)

    print('\n' + '=' * 60)
    print('Plotting completed successfully!')
    print(f'Plots saved to: {output_dir / "plots"}')
    print('=' * 60)


if __name__ == '__main__':
    main()
