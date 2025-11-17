# Plan

## Goal
- Simplify `notebooks/04-lensing-example.ipynb` to showcase only the spherical (HEALPix) weak-lensing pipeline from initial conditions through convergence.
- For a single source redshift, compare the measured spherical convergence power spectrum (via `SphericalDensity.compute_power_spectrum`) against the theoretical `C_\ell` from `compute_theory_cl`.
- Keep notebook cells modular (markdown to set context, code to execute work) and ensure it executes end-to-end via `nbconvert`.

## Files to modify and modifications

- `notebooks/04-lensing-example.ipynb`
  - Remove flat-sky, tomography, resampling, and duplicate z-source sections so the notebook focuses solely on the spherical pipeline for one `z_source`.
  - Adopt the following cell structure (markdown/code):
    1. Markdown — Title + brief description of the spherical Born example.
    2. Code — Imports, environment setup (`JAX_PLATFORM_NAME`, plotting defaults).
    3. Markdown — Simulation setup overview.
    4. Code — Define mesh/box parameters, random key, cosmology; print summary.
    5. Code — Generate Gaussian initial conditions.
    6. Code — Run first-order LPT, report shell metadata.
    7. Code — Run `nbody` with particle geometry and paint only the spherical lightcone.
    8. Code — Compute convergence via `born(..., z_source=1.0)`; print shapes/info.
    9. Markdown — Visualization blurb.
    10. Code — Display spherical convergence map (`kappa_spherical.show()` or similar).
    11. Markdown — Power-spectrum comparison intro.
    12. Code — Call `kappa_spherical.compute_power_spectrum(lmax=...)` to obtain measured `C_\ell`.
    13. Code — Build matching `ell` array, call `compute_theory_cl(cosmo, ell, z_source=1.0)`, and plot measured vs. theory (log-log) with optional ratio text.
  - Ensure inline comments clarify any non-obvious choices (e.g., selecting `lmax`, units).
  - Execute the notebook via `jupyter nbconvert --execute --to notebook ...` and overwrite the original with the executed result.

## Summary of plan
1. Rewrite `notebooks/04-lensing-example.ipynb` following the single-source spherical workflow outlined above, removing flat/tomographic/resampling content.
2. Implement the measured vs. theory power-spectrum comparison using `SphericalDensity.compute_power_spectrum` and `compute_theory_cl`, plus a visualization cell.
3. Run the notebook with `nbconvert` to produce the fully executed artifact, then proceed with further steps after plan approval.

## Questions for you
- None; constraints are clear.
