#!/bin/bash
# Streams MCMC catalog parquet files and computes per-chain statistics
# via SLURM using `fli-extract`.

# --- SLURM / Cluster configuration ---
RUN_LOCALLY=true  # (true, false, or dryrun)
# If set to false then it is launched with sbatch, if set to true then it is launched locally, if set to dryrun then it prints the sbatch command without executing it.
ACCOUNT="XXX"
CONSTRAINT="h100"
GPUS_PER_NODE=4
CPUS_PER_NODE=16
TASKS_PER_NODE=$GPUS_PER_NODE
NODES=4
PDIMS="16 1"          # e.g. "2 1" for 2-GPU mesh
QOS="qos_gpu_h100-t3"
TIME_LIMIT="00:30:00"

# --- I/O paths ---
# Source: provide either INPUT_DIR (local) or REPO_ID + CONFIG (HF Hub), not both.
INPUT_DIR="test_fli_samples"         # local root dir (chain_N/samples layout); leave empty to use HF Hub
REPO_ID=""           # HuggingFace Hub repo ID, e.g. "ASKabalan/jax-fli-experiments"
CONFIG=""            # space-separated config names, one per chain, e.g. "01-chain_0 01-chain_1"
TRUTH_PARQUET="test_fli_samples/chain_0/samples/samples_0.parquet"     # path to truth Catalog parquet; leave empty to skip
OUTPUT_FILE="results/extracts/extract.parquet"

# --- Extract parameters ---
COSMO_KEYS="Omega_c sigma8"
SET_NAME="my_extract"
FIELD_STATISTIC=true   # set to true to compute per-chain mean/std fields
POWER_STATISTIC=true   # set to true to compute per-chain transfer/coherence (requires TRUTH_PARQUET)
DDOF=0

CPUS_PER_TASK=$((CPUS_PER_NODE / TASKS_PER_NODE))
TOTAL_GPUS=$((GPUS_PER_NODE * NODES))
echo "CPUS_PER_TASK: $CPUS_PER_TASK"
echo "Total GPUs: $TOTAL_GPUS"
mkdir -p "$(dirname "$OUTPUT_FILE")"

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

# Validate source: exactly one of INPUT_DIR or REPO_ID must be set
if [ -z "$INPUT_DIR" ] && [ -z "$REPO_ID" ]; then
    echo "Error: set either INPUT_DIR (local) or REPO_ID (HF Hub)."
    exit 1
fi
if [ -n "$INPUT_DIR" ] && [ -n "$REPO_ID" ]; then
    echo "Error: set only one of INPUT_DIR or REPO_ID, not both."
    exit 1
fi

# Validate: REPO_ID requires CONFIG
if [ -n "$REPO_ID" ] && [ -z "$CONFIG" ]; then
    echo "Error: CONFIG must be set when REPO_ID is used."
    exit 1
fi

# Common SBATCH arguments
BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT --time=$TIME_LIMIT --gres=gpu:$GPUS_PER_NODE --cpus-per-task=$CPUS_PER_TASK --nodes=$NODES --tasks-per-node=$TASKS_PER_NODE --qos=$QOS"

# Extract Pdim_X and Pdim_Y
read -r PX PY <<< "$PDIMS"

JOB_NAME="${CONSTRAINT}_extract_${SET_NAME}"

echo "Submitting $JOB_NAME"

if [ "$RUN_LOCALLY" = true ]; then
    if [ "$TOTAL_GPUS" -eq 1 ]; then
        SBATCH_CMD=""
    else
        SBATCH_CMD="mpirun -n $TOTAL_GPUS --oversubscribe"
    fi
elif [ "$RUN_LOCALLY" = dryrun ]; then
    SBATCH_CMD=dry_run_submit
else
    SBATCH_CMD="sbatch $BASE_SBATCH_ARGS --job-name=$JOB_NAME --output=SLURM_LOGS/%x_%j.out --error=SLURM_LOGS/%x_%j.err $SLURM_SCRIPT FLI_EXTRACT"
fi

$SBATCH_CMD fli-extract \
    $([ -n "$INPUT_DIR" ] && echo "--path $INPUT_DIR") \
    $([ -n "$REPO_ID" ] && echo "--repo-id $REPO_ID") \
    $([ -n "$CONFIG" ] && echo "--config $CONFIG") \
    --set-name "$SET_NAME" \
    --output "$OUTPUT_FILE" \
    --cosmo-keys $COSMO_KEYS \
    $([ -n "$TRUTH_PARQUET" ] && echo "--truth $TRUTH_PARQUET") \
    $([ "$FIELD_STATISTIC" = "true" ] && echo "--field-statistic") \
    $([ "$POWER_STATISTIC" = "true" ] && echo "--power-statistic") \
    --ddof $DDOF \
    --pdim $PX $PY \
    --nodes $NODES
