#!/bin/bash

# Configuration
ACCOUNT="XXX"
CONSTRAINT="h100"
GPUS_PER_NODE=1
CPUS_PER_NODE=16
TASKS_PER_NODE=$GPUS_PER_NODE
NODES=1
PDIMS="1 1"          # e.g. "2 1" for 2-GPU mesh
TIME_LIMIT="24:00:00"

OBSERVABLE_DIR="results/observables"
OUTPUT_DIR="results/inference_runs"

CPUS_PER_TASK=$((CPUS_PER_NODE / TASKS_PER_NODE))
echo "CPUS_PER_TASK: $CPUS_PER_TASK"
mkdir -p "$OUTPUT_DIR"

# Check for SLURM_SCRIPT environment variable
if [ -z "$SLURM_SCRIPT" ]; then
    echo "Error: SLURM_SCRIPT environment variable is not set."
    exit 1
fi

# Physics / MCMC parameters
MESH_SIZE="256 256 256"
BOX_SIZE="1000.0 1000.0 1000.0"
NB_SHELLS=10
HALO_SIZE="0 0"      # set to e.g. "64 64" when PDIMS != "1 1"
T0=0.1
T1=1.0
NUM_STEPS=40
DT0=$(echo "scale=4; ($T1 - $T0) / ($NUM_STEPS - 1)" | bc)
LPT_ORDER=2
ADJOINT="checkpointed"
CHECKPOINTS=10
NUM_WARMUP=500
NUM_SAMPLES=100
BATCH_COUNT=5
SAMPLER="NUTS"       # NUTS | HMC | MCLMC
BACKEND="numpyro"    # numpyro | blackjax
SIGMA_E=0.26
INITIAL_CONDITION="" # path to IC parquet; empty = don't pass
SAMPLE="cosmo ic"    # what to sample

# Fiducial / ground-truth cosmology (for reference and job naming; not passed to fli-infer)
OMEGA_C=0.2589
SIGMA_8=0.8159
H=0.6774

# Observable catalog (parquet filename relative to OBSERVABLE_DIR)
OBSERVABLE="obs_seed0.parquet"

# Seed for the MCMC chain
SEED=0

# Common SBATCH arguments (no --qos: inference jobs may need long queues)
BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT --time=$TIME_LIMIT --gres=gpu:$GPUS_PER_NODE --cpus-per-task=$CPUS_PER_TASK --nodes=$NODES --tasks-per-node=$TASKS_PER_NODE --qos=qos_gpu_h100-t3"

# Extract Pdim_X and Pdim_Y
read -r PX PY <<< "$PDIMS"

OBS_PATH="$OBSERVABLE_DIR/$OBSERVABLE"
OBS_NAME="${OBSERVABLE%.*}"
JOB_NAME="${CONSTRAINT}_infer_${OBS_NAME}_Oc${OMEGA_C}_S8${SIGMA_8}_s${SEED}"
OUT_PATH="$OUTPUT_DIR/${JOB_NAME}"

echo "Submitting $JOB_NAME"
echo "  -> Observable: $OBS_PATH | Seed: $SEED | Mesh: $MESH_SIZE"

sbatch $BASE_SBATCH_ARGS \
    --job-name="$JOB_NAME" \
    --output="DEL/LOGS/%x_%j.out" \
    --error="DEL/LOGS/%x_%j.err" \
    $SLURM_SCRIPT LOGS fli-infer \
    --observable "$OBS_PATH" \
    --path "$OUT_PATH" \
    --mesh-size $MESH_SIZE \
    --box-size $BOX_SIZE \
    --nb-shells $NB_SHELLS \
    --pdim $PX $PY \
    $([ -n "$HALO_SIZE" ] && [ "$PDIMS" != "1 1" ] && echo "--halo-size $HALO_SIZE") \
    --t0 $T0 --t1 $T1 --dt0 $DT0 \
    --lpt-order $LPT_ORDER \
    --adjoint $ADJOINT \
    --checkpoints $CHECKPOINTS \
    --num-warmup $NUM_WARMUP \
    --num-samples $NUM_SAMPLES \
    --batch-count $BATCH_COUNT \
    --sampler $SAMPLER \
    --backend $BACKEND \
    --sigma-e $SIGMA_E \
    --sample $SAMPLE \
    $([ -n "$INITIAL_CONDITION" ] && echo "--initial-condition $INITIAL_CONDITION") \
    --seed $SEED \
    --no-progress-bar
