#!/bin/bash
# Submits an MCMC inference job (NUTS/HMC/MCLMC) on a pre-computed observable
# via SLURM using `fli-infer`.

# --- SLURM / Cluster configuration ---
RUN_LOCALLY=false # (true, false, or dryrun)
# If set to false then it is launched with sbatch, if set to true then it is launched locally, if set to dryrun then it prints the sbatch command without executing it.
ACCOUNT="XXX"
CONSTRAINT="h100"
GPUS_PER_NODE=1
CPUS_PER_NODE=16
TASKS_PER_NODE=$GPUS_PER_NODE
NODES=1
PDIMS="1 1"          # e.g. "2 1" for 2-GPU mesh
QOS="qos_gpu_h100-t3"
TIME_LIMIT="00:30:00"

# --- I/O paths ---
OBSERVABLE_DIR="results/observables"
OUTPUT_DIR="results/inference_runs"

# --- Simulation parameters ---
MESH_SIZE="256 256 256"
BOX_SIZE="1000.0 1000.0 1000.0"
LPT_ORDER=2
INTERP="none"

# --- Integration parameters ---
T0=0.1
T1=1.0
NB_STEPS=40
DRIFT_ON_LC=false      # set to "true" to pass --drift-on-lightcone
EQUAL_VOL=false        # set to "true" to enable equal-volume shells
MIN_WIDTH=50.0

# --- Shell / Lightcone parameters ---
NB_SHELLS=10
HALO_FRACTION=8

# --- Lensing parameters ---
LENSING="born"         # born | raytrace
MIN_Z=0.01
MAX_Z=1.5
N_INTEGRATE=32

# --- Sampling / MCMC parameters ---
ADJOINT="checkpointed"
CHECKPOINTS=10
NUM_WARMUP=500
NUM_SAMPLES=100
BATCH_COUNT=5
SAMPLER="NUTS"       # NUTS | HMC | MCLMC
BACKEND="numpyro"    # numpyro | blackjax
SIGMA_E=0.26
INITIAL_CONDITION="" # path to IC parquet; empty = don't pass
INIT_COSMO=false     # set to true to warm-start cosmology from observable (only if --sample includes 'ic' but not 'cosmo')
SAMPLE="cosmo ic"    # what to sample

# --- Fiducial cosmology ---
OMEGA_C=0.2589
SIGMA_8=0.8159
H=0.6774

# --- Job settings ---
OBSERVABLE="obs_seed0.parquet"
SEED=0

CPUS_PER_TASK=$((CPUS_PER_NODE / TASKS_PER_NODE))
TOTAL_GPUS=$((GPUS_PER_NODE * NODES))
echo "CPUS_PER_TASK: $CPUS_PER_TASK"
echo "Total GPUs: $TOTAL_GPUS"
mkdir -p "$OUTPUT_DIR"

dry_run_submit() {
    echo "======================================================="
    echo "Submitting job $JOB_NAME"
    echo "======================================================="
    printf "%-16s | %s\n" "ACCOUNT"        "$ACCOUNT"
    printf "%-16s | %s\n" "CONSTRAINT"     "$CONSTRAINT"
    printf "%-16s | %s\n" "TIME_LIMIT"     "$TIME_LIMIT"
    printf "%-16s | %s\n" "GPUS_PER_NODE"  "$GPUS_PER_NODE"
    printf "%-16s | %s\n" "CPUS_PER_TASK"  "$CPUS_PER_TASK"
    printf "%-16s | %s\n" "NODES"          "$NODES"
    printf "%-16s | %s\n" "TASKS_PER_NODE" "$TASKS_PER_NODE"
    printf "%-16s | %s\n" "QOS"            "$QOS"
    echo "*******************************************************"
    echo "$@"
    echo "*******************************************************"
    echo "======= end of job ======="
    echo
}

# Check for SLURM_SCRIPT environment variable
if [ "$RUN_LOCALLY" = false ] && [ -z "$SLURM_SCRIPT" ]; then
    echo "Error: SLURM_SCRIPT environment variable is not set."
    exit 1
fi

# Common SBATCH arguments (no --qos: inference jobs may need long queues)
BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT --time=$TIME_LIMIT --gres=gpu:$GPUS_PER_NODE --cpus-per-task=$CPUS_PER_TASK --nodes=$NODES --tasks-per-node=$TASKS_PER_NODE --qos=$QOS"

# Extract Pdim_X and Pdim_Y
read -r PX PY <<< "$PDIMS"

OBS_PATH="$OBSERVABLE_DIR/$OBSERVABLE"
OBS_NAME="${OBSERVABLE%.*}"
JOB_NAME="${CONSTRAINT}_infer_${OBS_NAME}_Oc${OMEGA_C}_S8${SIGMA_8}_s${SEED}"
OUT_PATH="$OUTPUT_DIR/${JOB_NAME}"

echo "Submitting $JOB_NAME"
echo "  -> Observable: $OBS_PATH | Seed: $SEED | Mesh: $MESH_SIZE"

if [ "$RUN_LOCALLY" = true ]; then
    if [ "$TOTAL_GPUS" -eq 1 ]; then
        SBATCH_CMD=""
    else
        SBATCH_CMD="mpirun -n $TOTAL_GPUS --oversubscribe"
    fi         
elif [ "$RUN_LOCALLY" = dryrun ]; then
    SBATCH_CMD=dry_run_submit
else
    SBATCH_CMD="sbatch $BASE_SBATCH_ARGS --job-name=$JOB_NAME --output=SLURM_LOGS/%x_%j.out --error=SLURM_LOGS/%x_%j.err $SLURM_SCRIPT FLI_INFERENCE"
fi

$SBATCH_CMD fli-infer \
    --observable "$OBS_PATH" \
    --path "$OUT_PATH" \
    --mesh-size $MESH_SIZE \
    --box-size $BOX_SIZE \
    --nb-shells $NB_SHELLS \
    --pdim $PX $PY \
    --nodes $NODES \
    --halo-fraction $HALO_FRACTION \
    --t0 $T0 --t1 $T1 \
    --nb-steps $NB_STEPS \
    --lpt-order $LPT_ORDER \
    --interp $INTERP \
    $([ "$DRIFT_ON_LC" = "true" ] && echo "--drift-on-lightcone") \
    $([ "$EQUAL_VOL" = "true" ] && echo "--equal-vol") \
    --min-width $MIN_WIDTH \
    --lensing $LENSING \
    --min-z $MIN_Z \
    --max-z $MAX_Z \
    --n-integrate $N_INTEGRATE \
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
    $([ "$INIT_COSMO" = "true" ] && echo "--init-cosmo") \
    --seed $SEED
