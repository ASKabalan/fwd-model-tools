"""Load CosmoGrid lightcone data."""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Union

import h5py
import jax.numpy as jnp
import jax_cosmo as jc
import jax_healpy as jhp
import numpy as np

from .._src.base._enums import FieldStatus
from ..fields import SphericalDensity
from ..fields.lensing_maps import SphericalKappaField
from ..fields.units import ConvergenceUnit, DensityUnit
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
            )
        )

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
    archive_name = "baryonified_shells.npz" if baryonified else "compressed_shells.npz"

    # Only one filter allowed
    if (max_shells, max_redshift, max_comoving_distance).count(None) < 2:
        raise ValueError("Only one of max_shells, max_redshift, or max_comoving_distance can be provided")

    run_dir = Path(path).resolve()

    # Load compressed shells
    npz_file = run_dir / archive_name
    if not npz_file.exists():
        raise FileNotFoundError(f"{archive_name} not found in {run_dir}")
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
    with open(params_file) as f:
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

    # Parse cosmology.par for simulation specs
    cosmo_par_file_tar = run_dir / "param_files.tar.gz"
    cosmo_par_file_flat = run_dir / "param_files" / "cosmology.par"

    sim_specs = {}

    if cosmo_par_file_flat.exists():
        with open(cosmo_par_file_flat) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if "=" in line:
                    key, val = line.split("=", 1)
                    sim_specs[key.strip()] = val.strip()
    elif cosmo_par_file_tar.exists():
        with tarfile.open(cosmo_par_file_tar, "r:gz") as tar:
            try:
                member = tar.getmember("cosmology.par")
                f = tar.extractfile(member)
                if f is not None:
                    content = f.read().decode("utf-8")
                    for line in content.splitlines():
                        line = line.split("#")[0].strip()
                        if "=" in line:
                            key, val = line.split("=", 1)
                            sim_specs[key.strip()] = val.strip()
            except KeyError:
                pass

    # Box size and grid size from params
    # Fallback to max comoving distance for box size if not found
    d_box_size = float(sim_specs.get("dBoxSize", np.max(upper_com)))
    n_grid = int(sim_specs.get("nGrid", 0))

    box_size = d_box_size

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
    if n_grid > 0:
        mesh_size = (n_grid, n_grid, n_grid)
    else:
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


def _nz_metadata(nz_obj, cosmo: jc.Cosmology) -> tuple[float, float, float]:
    """Compute physical metadata from a single-bin ``kde_nz`` and cosmology.

    Parameters
    ----------
    nz_obj : jc.redshift.kde_nz
        A single (unbatched) redshift distribution.
    cosmo : jc.Cosmology
        Cosmology for distance calculations.

    Returns
    -------
    tuple of (scale_factor, comoving_center, density_width)
        All as Python floats.
    """
    zcat = np.asarray(nz_obj.params[0])
    weight = np.asarray(nz_obj.params[1])

    # Effective redshift: z_eff = ∫ z·n(z) dz / ∫ n(z) dz
    z_eff = float(np.trapezoid(zcat * weight, zcat) / np.trapezoid(weight, zcat))
    scale_factor = 1.0 / (1.0 + z_eff)

    comoving_center = float(jc.background.radial_comoving_distance(cosmo, jnp.atleast_1d(scale_factor))[0])

    z_min, z_max = _find_nz_support(zcat, weight)

    chi_far = float(jc.background.radial_comoving_distance(cosmo, jnp.atleast_1d(1.0 / (1.0 + z_max)))[0])
    chi_near = float(jc.background.radial_comoving_distance(cosmo, jnp.atleast_1d(1.0 / (1.0 + z_min)))[0])
    density_width = chi_far - chi_near

    return z_eff, scale_factor, comoving_center, density_width


def _find_nz_support(zcat: np.ndarray, weight: np.ndarray, frac: float = 1e-3) -> tuple[float, float]:
    """Find the redshift support edges via CDF percentiles.

    Uses the cumulative distribution to find where the CDF crosses
    ``frac`` from below (z_min) and ``1 - frac`` from above (z_max).
    This robustly captures the shell containing ``1 - 2*frac`` of the
    source galaxies, regardless of tail behavior.

    Parameters
    ----------
    zcat : np.ndarray
        Redshift values, shape ``(N,)``.
    weight : np.ndarray
        n(z) weights, shape ``(N,)``.
    frac : float, default=1e-3
        CDF fraction for the lower/upper bounds (0.1% / 99.9%).

    Returns
    -------
    tuple of (z_min, z_max)
        Redshift bounds of the non-zero support.
    """
    cdf = np.cumsum(0.5 * (weight[:-1] + weight[1:]) * np.diff(zcat))
    cdf = np.concatenate([[0.0], cdf])
    cdf /= cdf[-1]

    z_min = float(zcat[np.searchsorted(cdf, frac)])
    z_max = float(zcat[np.searchsorted(cdf, 1.0 - frac)])
    return z_min, z_max


def _resolve_cosmogrid_cosmology(path: Path) -> jc.Cosmology:
    """Resolve cosmology from ``CosmoGridV1_metainfo.h5`` given a stage3 path.

    Parameters
    ----------
    path : Path
        Resolved path inside a CosmoGrid tree, e.g.
        ``.../CosmoGrid/stage3_forecast/cosmo_000001/perm_0000/``

    Returns
    -------
    jc.Cosmology
        The cosmology for the given simulation.
    """
    import re

    # Extract cosmo_XXXXXX from the path
    match = re.search(r"(cosmo_\d+)", str(path))
    if match is None:
        raise ValueError(f"Cannot extract cosmo_XXXXXX identifier from path: {path}")
    cosmo_id = match.group(1)

    # Walk up the directory tree to find CosmoGridV1_metainfo.h5
    metainfo_path = None
    for parent in path.parents:
        candidate = parent / "CosmoGridV1_metainfo.h5"
        if candidate.exists():
            metainfo_path = candidate
            break
    if metainfo_path is None:
        raise FileNotFoundError(f"CosmoGridV1_metainfo.h5 not found in any parent of {path}")

    # Look up the cosmology row by matching path_par containing the cosmo ID
    with h5py.File(metainfo_path, "r") as f:
        params = f["parameters/grid"]
        path_pars = params["path_par"]
        row = None
        for i in range(len(path_pars)):
            pp = path_pars[i]
            if isinstance(pp, bytes):
                pp = pp.decode()
            if cosmo_id in pp:
                row = params[i]
                break

    if row is None:
        raise ValueError(f"Cosmology '{cosmo_id}' not found in {metainfo_path}")

    h_param = float(row["H0"]) / 100.0
    return jc.Cosmology(
        Omega_c=float(row["O_cdm"]),
        Omega_b=float(row["Ob"]),
        h=h_param,
        n_s=float(row["ns"]),
        sigma8=float(row["s8"]),
        w0=float(row["w0"]),
        wa=float(row["wa"]),
        Omega_k=0.0,
        Omega_nu=float(row["O_nu"]),
    )


def load_cosmogrid_kappa(
    path: PathLike,
    *,
    baryonified: bool = False,
    probe: str = "kg",
    bins: tuple[int, ...] | list[int] | None = None,
    ud_nside: int | None = None,
) -> Catalog:
    """Load CosmoGrid Stage 3 projected kappa (or IA/galaxy density) maps.

    Parameters
    ----------
    path : str or Path
        Path to a CosmoGrid stage3_forecast directory
        (e.g., ``stage3_forecast/cosmo_000001/perm_0000/``).
    cosmology : jc.Cosmology, Catalog, or None
        Explicit cosmology. If a ``Catalog`` is given its ``.cosmology`` is used.
        If ``None``, auto-resolved from ``CosmoGridV1_metainfo.h5``.
    baryonified : bool, default=False
        If True, load the baryonified HDF5 variant.
    probe : str, default="kg"
        Probe group inside the HDF5 file (``"kg"`` for kappa, ``"ia"``, ``"dg"``).
    bins : tuple or list of int, optional
        Which tomographic bins (1-4) to load. Default is all four.
    ud_nside : int, optional
        Target NSIDE for up/down-sampling. If None, keeps original NSIDE (512).

    Returns
    -------
    Catalog
        Contains a ``SphericalKappaField`` (stacked over bins) and ``jc.Cosmology``.
    """
    path = Path(path).resolve()

    # Select the correct HDF5 file
    h5_name = "projected_probes_maps_baryonified512.h5" if baryonified else "projected_probes_maps_nobaryons512.h5"
    h5_file = path / h5_name
    if not h5_file.exists():
        raise FileNotFoundError(f"{h5_name} not found in {path}")

    # Resolve cosmology
    cosmo = _resolve_cosmogrid_cosmology(path)

    # Determine bins
    if bins is None:
        bins = (1, 2, 3, 4)

    # Load n(z) distributions for each bin
    nz_shear = get_stage3_nz_shear()

    maps = []
    with h5py.File(h5_file, "r") as f:
        for bin_idx in bins:
            key = f"{probe}/stage3_lensing{bin_idx}"
            if key not in f:
                raise KeyError(f"Dataset '{key}' not found in {h5_file}")
            data = f[key][:]
            nside = jhp.npix2nside(len(data))

            nz_bin = nz_shear[bin_idx - 1]  # bins are 1-indexed
            z_eff, a_eff, chi_center, dwidth = _nz_metadata(nz_bin, cosmo)

            kappa_map = SphericalKappaField(
                array=jnp.asarray(data, dtype=jnp.float32),
                mesh_size=(1, 1, 1),
                box_size=(1.0, 1.0, 1.0),
                observer_position=(0.5, 0.5, 0.5),
                sharding=None,
                halo_size=(0, 0),
                nside=nside,
                z_sources=z_eff,
                scale_factors=a_eff,
                comoving_centers=chi_center,
                density_width=dwidth,
                status=FieldStatus.KAPPA,
                unit=ConvergenceUnit.DIMENSIONLESS,
            )
            if ud_nside is not None:
                kappa_map = kappa_map.ud_sample(new_nside=ud_nside)
            maps.append(kappa_map)

    full_field = SphericalKappaField.stack(maps)

    return Catalog(field=full_field, cosmology=cosmo)
