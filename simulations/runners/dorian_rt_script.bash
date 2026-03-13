#!/bin/bash
# Submits a SLURM job to apply dorian ray-tracing post-processing to
# pre-computed simulation outputs using `fli-dorian-rt` via MPI.

# --- SLURM / Cluster configuration ---
RUN_LOCALLY=true # (true, false, or dryrun)
# If set to false then it is launched with sbatch, if set to true then it is launched locally, if set to dryrun then it prints the sbatch command without executing it.
ACCOUNT="XXX"
CONSTRAINT="cpu"
CPUS_PER_NODE=64
TASKS_PER_NODE=4
NODES=1
QOS="qos_cpu"
TIME_LIMIT="01:00:00"

# --- I/O paths ---
INPUT_DIR="/home/wassim/Projects/NBody/jax-fli-result/results/02-density_width_shell_selection/catalogs/multi_shell"
OUTPUT_DIR="results/lensing/multi_shell_raytrace"

# --- Lensing parameters ---
NZ_SHEAR="s3"
MIN_Z=0.01          # minimum redshift for n(z) integration (default: 0.01)
MAX_Z=1.5           # maximum redshift for n(z) integration (default: 1.5)
N_INTEGRATE=32       # Simpson quadrature points for n(z) distributions (default: 32)
RT_INTERP="bilinear" # bilinear | ngp | nufft
NO_PARALLEL_TRANSPORT=false  # set to "true" to disable parallel transport

CPUS_PER_TASK=$((CPUS_PER_NODE / TASKS_PER_NODE))
TOTAL_TASKS=$((TASKS_PER_NODE * NODES))
echo "CPUS_PER_TASK: $CPUS_PER_TASK"
echo "Total MPI tasks: $TOTAL_TASKS"
mkdir -p "$OUTPUT_DIR"

dry_run_submit() {
    echo "======================================================="
    echo "Submitting job $JOB_NAME"
    echo "======================================================="
    printf "%-16s | %s\n" "ACCOUNT"        "$ACCOUNT"
    printf "%-16s | %s\n" "CONSTRAINT"     "$CONSTRAINT"
    printf "%-16s | %s\n" "TIME_LIMIT"     "$TIME_LIMIT"
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

BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT --time=$TIME_LIMIT --cpus-per-task=$CPUS_PER_TASK \
  --nodes=$NODES --tasks-per-node=$TASKS_PER_NODE --qos=$QOS"

echo "Submitting fli-dorian-rt job for $INPUT_DIR/*.parquet"

JOB_NAME="fli_dorian_rt"

if [ "$RUN_LOCALLY" = true ]; then
    SBATCH_CMD="mpirun -n $TOTAL_TASKS --oversubscribe"
elif [ "$RUN_LOCALLY" = dryrun ]; then
    SBATCH_CMD=dry_run_submit
else
    SBATCH_CMD="sbatch $BASE_SBATCH_ARGS --job-name=$JOB_NAME --output=SLURM_LOGS/%x_%j.out --error=SLURM_LOGS/%x_%j.err $SLURM_SCRIPT FLI_DORIAN_RT"
fi

$SBATCH_CMD fli-dorian-rt \
    --input "$INPUT_DIR/*.parquet" \
    --output "$OUTPUT_DIR" \
    --nz-shear $NZ_SHEAR \
    --min-z $MIN_Z \
    --max-z $MAX_Z \
    --n-integrate $N_INTEGRATE \
    --rt-interp $RT_INTERP \
    $([ "$NO_PARALLEL_TRANSPORT" = "true" ] && echo "--no-parallel-transport")
