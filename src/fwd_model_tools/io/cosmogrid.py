"""Load CosmoGrid lightcone data."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import h5py
import jax.numpy as jnp
import jax_cosmo as jc
import jax_healpy as jhp
import numpy as np

from .._src.base._enums import FieldStatus
from ..fields import SphericalDensity
from ..fields.units import DensityUnit
from .catalog import Catalog

PathLike = Union[str, Path]


def get_stage3_nz_shear(
    gals_per_arcmin2: list[float] | None = None,
    bw: float = 0.01,
    zmax: float | None = None,
) -> list:
    """Load Stage 3 weak lensing redshift distributions.

    Parameters
    ----------
    gals_per_arcmin2 : list of float, optional
        Galaxy densities for each of the 4 tomographic bins.
        Default: [7, 8.5, 7.5, 6]
    bw : float, default=0.01
        KDE bandwidth parameter.
    zmax : float, optional
        Maximum redshift. If None, uses max from data files.

    Returns
    -------
    list of jc.redshift.kde_nz
        List of 4 redshift distribution objects.
    """
    from importlib.resources import files

    if gals_per_arcmin2 is None:
        gals_per_arcmin2 = [7.0, 8.5, 7.5, 6.0]

    data_dir = files("fwd_model_tools.io").joinpath("data")
    nz_files = sorted(data_dir.glob("nz_stage3_*.txt"))

    nz_shear = []
    for nz_file, g in zip(nz_files, gals_per_arcmin2):
        z, nz = np.loadtxt(nz_file, unpack=True)
        nz_shear.append(
            jc.redshift.kde_nz(
                jnp.asarray(z),
                jnp.asarray(nz),
                bw=bw,
                zmax=zmax if zmax is not None else float(z.max()),
                gals_per_arcmin2=g,
            ))

    return nz_shear


def load_cosmogrid_lc(
    path: PathLike,
    baryonified: bool = False,
    max_shells: int | None = None,
    max_redshift: float | None = None,
    max_comoving_distance: float | None = None,
    ud_nside: int | None = None,
) -> Catalog:
    """Load raw CosmoGrid lightcone shells.

    Parameters
    ----------
    path : str or Path
        Path to a CosmoGrid raw simulation directory (e.g., raw/cosmo_000001/run_0/).
    baryonified : bool, default=False
        If True, raises NotImplementedError (baryonified shells not yet supported).
    max_shells : int, optional
        Maximum number of shells to load (takes the nearest shells first).
    max_redshift : float, optional
        Maximum redshift to load shells up to.
    max_comoving_distance : float, optional
        Maximum comoving distance to load shells up to.
    ud_nside : int, optional
        Target NSIDE for up/down-sampling. If None, keeps original NSIDE.

    Returns
    -------
    Catalog
        Contains SphericalDensity field and jc.Cosmology.
    """
    if baryonified:
        raise NotImplementedError("Baryonified shells not yet supported")

    # Only one filter allowed
    if (max_shells, max_redshift, max_comoving_distance).count(None) < 2:
        raise ValueError("Only one of max_shells, max_redshift, or max_comoving_distance can be provided")

    run_dir = Path(path).resolve()

    # Load compressed shells
    npz_file = run_dir / "compressed_shells.npz"
    if not npz_file.exists():
        raise FileNotFoundError(f"compressed_shells.npz not found in {run_dir}")
    if ud_nside == 512:
        npz_file = run_dir / "shells_nside=512.npz"

    data = np.load(npz_file)
    shells = data["shells"]
    shell_info = data["shell_info"]

    # Parse params.yml inline
    params_file = run_dir / "params.yml"
    if not params_file.exists():
        raise FileNotFoundError(f"params.yml not found in {run_dir}")

    params = {}
    with open(params_file, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                try:
                    params[key.strip()] = float(val.strip())
                except ValueError:
                    params[key.strip()] = val.strip()

    # Extract shell metadata
    lower_z = shell_info["lower_z"]
    upper_z = shell_info["upper_z"]
    lower_com = shell_info["lower_com"]
    upper_com = shell_info["upper_com"]
    shell_com = shell_info["shell_com"]

    # Box size from max comoving distance
    box_size = float(np.max(upper_com))

    # Create mask based on filter
    if max_shells is not None:
        mask = np.arange(len(lower_z)) < max_shells
    elif max_redshift is not None:
        mask = lower_z <= max_redshift
    elif max_comoving_distance is not None:
        mask = lower_com <= max_comoving_distance
    else:
        mask = np.ones(len(lower_z), dtype=bool)

    # Apply mask
    indices = np.where(mask)[0]
    shells = shells[indices]
    lower_z = lower_z[indices]
    upper_z = upper_z[indices]
    lower_com = lower_com[indices]
    upper_com = upper_com[indices]
    shell_com = shell_com[indices]

    z_centers = 0.5 * (lower_z + upper_z)
    comoving_centers = 0.5 * (lower_com + upper_com)
    shell_widths = np.abs(upper_com - lower_com)

    # Get NSIDE from data
    nside = jhp.npix2nside(shells.shape[-1])
    if ud_nside is None:
        ud_nside = nside

    # Build SphericalDensity for each shell
    mesh_size = (int(box_size), int(box_size), int(box_size))
    box_tuple = (box_size, box_size, box_size)

    sph_maps = []
    for i in range(len(shells)):
        sph_map = SphericalDensity(
            array=jnp.asarray(shells[i], dtype=jnp.float32),
            mesh_size=mesh_size,
            box_size=box_tuple,
            observer_position=(0.5, 0.5, 0.5),
            sharding=None,
            halo_size=(0, 0),
            nside=nside,
            z_sources=jnp.asarray([z_centers[i]]),
            scale_factors=jnp.asarray([1.0 / (1.0 + z_centers[i])]),
            comoving_centers=jnp.asarray(comoving_centers[i]),
            density_width=jnp.asarray(shell_widths[i]),
            status=FieldStatus.LIGHTCONE,
            unit=DensityUnit.COUNTS,
        )
        sph_maps.append(sph_map.ud_sample(new_nside=ud_nside))

    all_spherical_maps = SphericalDensity.stack(sph_maps)

    # Build jax_cosmo Cosmology
    h_param = float(params["H0"]) / 100.0

    cosmo = jc.Cosmology(
        Omega_c=float(params["O_cdm"]),
        Omega_b=float(params["Ob"]),
        h=h_param,
        n_s=float(params["ns"]),
        sigma8=float(params["s8"]),
        w0=float(params["w0"]),
        wa=float(params["wa"]),
        Omega_k=0.0,
        Omega_nu=float(params["O_nu"]),
    )

    return Catalog(field=all_spherical_maps, cosmology=cosmo)


def load_cosmogrid_kappa(
    path: PathLike,
    **kwargs,
) -> Catalog:
    """Load CosmoGrid kappa maps (not yet implemented)."""
    raise NotImplementedError("load_cosmogrid_kappa not yet implemented")
