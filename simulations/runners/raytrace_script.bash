#!/bin/bash
# Submits a SLURM job to apply raytrace/Born lensing post-processing to
# pre-computed simulation outputs using `fli-raytrace`.

# --- SLURM / Cluster configuration ---
RUN_LOCALLY=true # (true, false, or dryrun)
# If set to false then it is launched with sbatch, if set to true then it is launched locally, if set to dryrun then it prints the sbatch command without executing it.
ACCOUNT="XXX"
CONSTRAINT="h100"
GPUS_PER_NODE=1
CPUS_PER_NODE=16
TASKS_PER_NODE=$GPUS_PER_NODE
PDIMS="1 1"
NODES=1
QOS="qos_gpu_h100-t3"
TIME_LIMIT="00:30:00"

# --- I/O paths ---
INPUT_DIR="results/grid_runs"
OUTPUT_DIR="results/lensing"

# --- Precision ---
ENABLE_X64=false       # set to "true" to enable JAX 64-bit precision

# --- Lensing parameters ---
NZ_SHEAR="s3"
LENSING="both"      # born | raytrace | both
MIN_Z=0.01          # minimum redshift for n(z) integration (default: 0.01)
MAX_Z=1.5           # maximum redshift for n(z) integration (default: 1.5)
N_INTEGRATE=32       # Simpson quadrature points for n(z) distributions (default: 32)

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

# IF CONTRAIN is CPU then no contrain and no gres gpu
if [ "$CONSTRAINT" = "cpu" ]; then
    CONST_STR=""
    GPU_STR=""
else
    CONST_STR="-C $CONSTRAINT"
    GPU_STR="--gres=gpu:$GPUS_PER_NODE"
fi
BASE_SBATCH_ARGS="--account=$ACCOUNT $CONST_STR --time=$TIME_LIMIT $GPU_STR --cpus-per-task=$CPUS_PER_TASK \
  --nodes=$NODES --tasks-per-node=$TASKS_PER_NODE --qos=$QOS"

read -r PX PY <<< "$PDIMS"

echo "Submitting fli-raytrace job for $INPUT_DIR/*.parquet"

JOB_NAME="fli_raytrace"

if [ "$RUN_LOCALLY" = true ]; then
    if [ "$TOTAL_GPUS" -eq 1 ]; then
        SBATCH_CMD=""
    else
        SBATCH_CMD="mpirun -n $TOTAL_GPUS --oversubscribe"
    fi
elif [ "$RUN_LOCALLY" = dryrun ]; then
    SBATCH_CMD=dry_run_submit
else
    SBATCH_CMD="sbatch $BASE_SBATCH_ARGS --job-name=$JOB_NAME --output=SLURM_LOGS/%x_%j.out --error=SLURM_LOGS/%x_%j.err $SLURM_SCRIPT FLI_RAYTRACE"
fi

$SBATCH_CMD fli-raytrace \
    --pdim $PX $PY \
    --nodes $NODES \
    --input "$INPUT_DIR/*.parquet" \
    --output "$OUTPUT_DIR" \
    --lensing $LENSING \
    --nz-shear $NZ_SHEAR \
    --min-z $MIN_Z \
    --max-z $MAX_Z \
    --n-integrate $N_INTEGRATE \
    $([ "$ENABLE_X64" = "true" ] && echo "--enable-x64")
