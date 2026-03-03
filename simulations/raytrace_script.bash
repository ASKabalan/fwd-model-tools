#!/bin/bash
# Submits a SLURM job to apply raytrace/Born lensing post-processing to
# pre-computed simulation outputs using `fli-raytrace`.

# --- SLURM / Cluster configuration ---
ACCOUNT="XXX"
CONSTRAINT="h100"
GPUS_PER_NODE=4
CPUS_PER_NODE=16
TASKS_PER_NODE=$GPUS_PER_NODE
PDIMS="1 1"
NODES=1
QOS="qos_gpu_h100-t3"
TIME_LIMIT="00:30:00"

# --- I/O paths ---
INPUT_DIR="results/cosmology_runs"
OUTPUT_DIR="results/lensing"

# --- Lensing parameters ---
NZ_SHEAR="s3"
LENSING="both"      # born | raytrace | both
MAX_Z=3.0
N_INTEGRATE=2

CPUS_PER_TASK=$((CPUS_PER_NODE / TASKS_PER_NODE))
echo "CPUS_PER_TASK: $CPUS_PER_TASK"
mkdir -p "$OUTPUT_DIR"

if [ -z "$SLURM_SCRIPT" ]; then
    echo "Error: SLURM_SCRIPT environment variable is not set."
    exit 1
fi

BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT --time=$TIME_LIMIT \
  --gres=gpu:$GPUS_PER_NODE --cpus-per-task=$CPUS_PER_TASK \
  --gpus-per-task=1 --nodes=$NODES --tasks-per-node=$TASKS_PER_NODE --qos=$QOS"

read -r PX PY <<< "$PDIMS"

echo "Submitting fli-raytrace job for $INPUT_DIR/*.parquet"

sbatch $BASE_SBATCH_ARGS \
    --job-name="fli_raytrace" \
    --output="SLURM_LOGS/%x_%j.out" \
    --error="SLURM_LOGS/%x_%j.err" \
    $SLURM_SCRIPT FLI_RAYTRACE fli-raytrace \
    --pdim $PX $PY \
    --nodes $NODES \
    --input "$INPUT_DIR/*.parquet" \
    --output "$OUTPUT_DIR" \
    --lensing $LENSING \
    --nz-shear $NZ_SHEAR \
    --max-z $MAX_Z \
    --n-integrate $N_INTEGRATE
