#!/bin/bash
# raytrace_script.sh — example usage of fli-raytrace
#
# Post-processes existing lightcone parquet files with Born/raytrace lensing.
# nside is read from the parquet metadata; no re-simulation needed.

INPUT="simulations/results/cosmology_runs/*.parquet"
OUTPUT_DIR="simulations/results/lensing"
NZ_SHEAR="s3"
LENSING="born"   # options: born | raytrace | both
MAX_Z=3.0
N_INTEGRATE=32

fli-raytrace \
    --input "$INPUT" \
    --output "$OUTPUT_DIR" \
    --lensing "$LENSING" \
    --nz-shear "$NZ_SHEAR" \
    --max-z "$MAX_Z" \
    --n-integrate "$N_INTEGRATE"
