#!/bin/bash

# Configuration
ACCOUNT="XXX"
CONSTRAINT="h100"
GPUS_PER_NODE=4
CPUS_PER_NODE=16
TASKS_PER_NODE=$GPUS_PER_NODE
PDIMS="2 1"
NODES=1
INPUT_DIR="results/cosmology_runs"
OUTPUT_DIR="results/lensing"

mkdir -p "$OUTPUT_DIR"

if [ -z "$SLURM_SCRIPT" ]; then
    echo "Error: SLURM_SCRIPT environment variable is not set."
    exit 1
fi

# Raytrace parameters
NZ_SHEAR="s3"
LENSING="born"      # born | raytrace | both
MAX_Z=3.0
N_INTEGRATE=32

BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT --time=02:00:00 \
  --gres=gpu:$GPUS_PER_NODE --cpus-per-task=$((CPUS_PER_NODE / TASKS_PER_NODE)) \
  --gpus-per-task=1 --nodes=$NODES --tasks-per-node=$TASKS_PER_NODE --exclusive"

read -r PX PY <<< "$PDIMS"

echo "Submitting fli-raytrace job for $INPUT_DIR/*.parquet"

sbatch $BASE_SBATCH_ARGS \
    --job-name="fli_raytrace" \
    $SLURM_SCRIPT LOGS fli-raytrace \
    --input "$INPUT_DIR/*.parquet" \
    --output "$OUTPUT_DIR" \
    --lensing $LENSING \
    --nz-shear $NZ_SHEAR \
    --max-z $MAX_Z \
    --n-integrate $N_INTEGRATE
