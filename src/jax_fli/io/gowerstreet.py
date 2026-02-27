"""Load GowerStreet lightcone data."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import healpy as hp
import jax_cosmo as jc
import numpy as np
from jax_cosmo.parameters import Planck18

from .._src.base._enums import FieldStatus
from ..fields import SphericalDensity
from ..fields.units import DensityUnit
from .catalog import Catalog

PathLike = Union[str, Path]


def _parse_control_par(control_file: Path) -> dict:
    """Parse control.par file for cosmology parameters."""
    keys = ["h", "dOmega0", "dOmegaDE", "dSigma8", "dSpectral", "w0", "dBoxSize", "nGrid"]

    with open(control_file) as f:
        content = f.read()

    local_scope = {}
    try:
        exec(content, {}, local_scope)
    except Exception as e:
        raise RuntimeError(f"Failed to parse {control_file}: {e}") from e

    values = {key: local_scope[key] for key in keys if key in local_scope}

    return values


def load_gowerstreet(
    path: PathLike,
    omega_b: float | None = None,
    max_shells: int | None = None,
    max_redshift: float | None = None,
    max_comoving_distance: float | None = None,
    ud_nside: int | None = None,
) -> Catalog:
    """Load GowerStreet lightcone data.

    Parameters
    ----------
    path : str or Path
        Path to a GowerStreet simulation directory (e.g., sim00001/).
    omega_b : float, optional
        Baryon density parameter. If not provided, uses default Planck 2018 value (0.0493).
    max_shells : int, optional
        Maximum number of shells to load. If None, loads all shells.
        Useful for testing or when memory is limited.

    Returns
    -------
    Catalog
        Contains SphericalDensity field and jc.Cosmology.
    """
    # Only one of max_redshift or max_comoving_distance can be provided
    if (max_redshift, max_comoving_distance, max_shells).count(None) < 2:
        raise ValueError("Only one of max_redshift, max_comoving_distance, or max_shells can be provided")

    sim_dir = Path(path).resolve()

    # Load cosmology from control.par
    control_file = sim_dir / "control.par"
    if not control_file.exists():
        raise FileNotFoundError(f"control.par not found in {sim_dir}")
    z_file = sim_dir / "z_values.txt"
    if not z_file.exists():
        raise FileNotFoundError(f"z_values.txt not found in {sim_dir}")

    params = _parse_control_par(control_file)

    # Load z_values.txt
    coord_data = np.loadtxt(z_file, delimiter=",", comments="#")[:, :7]
    steps = coord_data[:, 0].astype(int)
    z_far, z_near = coord_data[:, 1], coord_data[:, 2]
    comoving_far, comoving_near = coord_data[:, 3], coord_data[:, 4]
    density_widths = coord_data[:, 5]

    # Creating mask
    # If max_shells, take the last max_shells shells
    if max_shells is not None and max_shells > 0:
        mask = np.arange(len(z_far)) >= (len(z_far) - max_shells)
    elif max_redshift is not None:
        mask = z_near <= max_redshift
    elif max_comoving_distance is not None:
        mask = comoving_near <= max_comoving_distance
    else:
        mask = np.ones_like(z_far, dtype=bool)

    print(f"[{z_far.shape}, {z_near.shape}, {comoving_far.shape}, {comoving_near.shape}, {density_widths.shape}]")
    (indices,) = np.where(mask)
    steps = steps[indices]
    z_far = z_far[indices]
    z_near = z_near[indices]
    z_centers = 0.5 * (z_near + z_far)
    comoving_far = comoving_far[indices]
    comoving_near = comoving_near[indices]
    comoving_centers = 0.5 * (comoving_near + comoving_far)
    shell_width = density_widths[indices]
    print(f"[{z_far.shape}, {z_near.shape}, {comoving_far.shape}, {comoving_near.shape}, {density_widths.shape}]")
    # Extract step numbers from filenames
    sph_maps = []

    def find_lc_from_step(step: int) -> Path:
        """Find lightcone file corresponding to a given step number."""
        # Format can be run.00099.lightcone.npy
        # or run.00099.incomplete.lightcone.npy
        # with 99 as step here
        pattern = rf"run.{step:05d}.*[lightcone][incomplete].npy"
        match = sim_dir.glob(pattern)
        match = list(match)
        if len(match) == 0:
            raise FileNotFoundError(f"No lightcone file found for step {step} in {sim_dir}")
        return match[0]

    for i, step in enumerate(steps):
        lc_file = find_lc_from_step(step)
        if "incomplete" in lc_file.name:
            print(f"Skipping incomplete lightcone file {lc_file}")
            continue
        # Load map
        map_data = np.load(lc_file)
        sph_map = SphericalDensity(
            array=np.asarray(map_data),
            mesh_size=(int(params["nGrid"]), int(params["nGrid"]), int(params["nGrid"])),
            box_size=(float(params["dBoxSize"]), float(params["dBoxSize"]), float(params["dBoxSize"])),
            observer_position=(0.5, 0.5, 0.5),
            sharding=None,
            halo_size=(0, 0),
            nside=hp.npix2nside(map_data.shape[-1]),
            z_sources=np.asarray([z_centers[i]]),
            scale_factors=np.array(jc.utils.z2a(np.asarray([z_centers[i]]))),
            comoving_centers=np.asarray(comoving_centers[i]),
            density_width=np.asarray(shell_width[i]),
            status=FieldStatus.LIGHTCONE,
            unit=DensityUnit.COUNTS,
        )
        if ud_nside is None:
            ud_nside = sph_map.nside
        sph_maps.append(sph_map.ud_sample(new_nside=ud_nside))

    all_spherical_maps = SphericalDensity.stack(sph_maps)

    # Determine Omega_b
    if omega_b is None:
        omega_b = Planck18().Omega_b

    # Calculate cosmological parameters
    # dOmega0 is total matter (Omega_m)
    omega_m = float(params["dOmega0"])
    omega_de = float(params["dOmegaDE"])
    omega_c = omega_m - omega_b

    # Calculate Omega_k (curvature)
    # Omega_k = 1 - Omega_m - Omega_de
    omega_k = 1.0 - (omega_m + omega_de)

    # Other parameters
    h_param = float(params["h"])
    n_s = float(params["dSpectral"])
    sigma8 = float(params["dSigma8"])
    w0_param = float(params["w0"])

    # Build jax_cosmo Cosmology
    cosmo = jc.Cosmology(
        Omega_c=omega_c,
        Omega_b=omega_b,
        h=h_param,
        n_s=n_s,
        sigma8=sigma8,
        Omega_k=omega_k,
        w0=w0_param,
        wa=0.0,
    )

    return Catalog(field=all_spherical_maps, cosmology=cosmo)
