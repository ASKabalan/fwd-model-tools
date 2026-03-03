#!/bin/bash
# Generates FLI forward-model samples in batches across multiple chains using
# `fli-samples`, submitting each (chain, batch) pair as a SLURM job.

# --- SLURM / Cluster configuration ---
RUN_LOCALLY=false # (true, false, or dryrun)
# If set to false then it is launched with sbatch, if set to true then it is launched locally, if set to dryrun then it prints the sbatch command without executing it.
ACCOUNT="XXX"
CONSTRAINT="h100"
GPUS_PER_NODE=1
CPUS_PER_NODE=16
TASKS_PER_NODE=$GPUS_PER_NODE
NODES=1
PDIMS="1 1"
QOS="qos_gpu_h100-t3"
TIME_LIMIT="00:30:00"

# --- I/O paths ---
OUTPUT_DIR="test_fli_samples"

# --- Simulation parameters ---
MODEL="mock"
MESH_SIZE="64 64 64"
BOX_SIZE="250 250 250"
NSIDE=4
LPT_ORDER=2

# --- Integration parameters ---
T0=0.01
T1=1.0
NB_STEPS=100

# --- Shell / Lightcone parameters ---
NB_SHELLS=8
HALO_FRACTION=8
OBSERVER_POSITION="0.5 0.5 0.5"

# --- Lensing parameters ---
NZ_SHEAR="s3"
LENSING="born"
INTERP="none"

# --- Sampling parameters ---
NUM_SAMPLES=10

# --- Job settings ---
CHAINS=(0)
# Make 20 batches
BATCHES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

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

BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT --time=$TIME_LIMIT --gres=gpu:$GPUS_PER_NODE --cpus-per-task=$CPUS_PER_TASK --nodes=$NODES --tasks-per-node=$TASKS_PER_NODE --qos=$QOS"

read -r PX PY <<< "$PDIMS"

for chain in "${CHAINS[@]}"; do
    for batch in "${BATCHES[@]}"; do
        JOB_NAME="${CONSTRAINT}_samples_chain${chain}_batch${batch}"
        echo "Submitting $JOB_NAME"

        if [ "$RUN_LOCALLY" = true ]; then
            SBATCH_CMD=""
        elif [ "$RUN_LOCALLY" = dryrun ]; then
            SBATCH_CMD=dry_run_submit
        else
            SBATCH_CMD="sbatch $BASE_SBATCH_ARGS --job-name=$JOB_NAME --output=SLURM_LOGS/%x_%j.out --error=SLURM_LOGS/%x_%j.err $SLURM_SCRIPT FLI_SAMPLES"
        fi

        $SBATCH_CMD fli-samples \
            --model $MODEL \
            --mesh-size $MESH_SIZE \
            --box-size $BOX_SIZE \
            --nside $NSIDE \
            --lpt-order $LPT_ORDER \
            --t0 $T0 \
            --t1 $T1 \
            --nb-steps $NB_STEPS \
            --nb-shells $NB_SHELLS \
            --halo-fraction $HALO_FRACTION \
            --observer-position $OBSERVER_POSITION \
            --nz-shear $NZ_SHEAR \
            --lensing $LENSING \
            --interp $INTERP \
            --num-samples $NUM_SAMPLES \
            --seed $batch \
            --path "$OUTPUT_DIR/chain_$chain" \
            --batch-id $batch
    done
done
