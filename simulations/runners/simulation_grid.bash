#!/bin/bash
# Submits a single SLURM job running the full parameter-grid exploration in one
# process using `fli-grid` (supports range notation).

# --- SLURM / Cluster configuration ---
RUN_LOCALLY=true # (true, false, or dryrun)
# If set to false then it is launched with sbatch, if set to true then it is launched locally, if set to dryrun then it prints the sbatch command without executing it.
ACCOUNT="XXX"
CONSTRAINT="h100"
GPUS_PER_NODE=4
CPUS_PER_NODE=16
TASKS_PER_NODE=$GPUS_PER_NODE
PDIMS="16 1"
NODES=4
QOS="qos_gpu_h100-t3"
TIME_LIMIT="24:00:00"   # fli-grid runs ALL combos in one job — set generously

# --- I/O paths ---
OUTPUT_DIR="results/grid_runs"

# --- Simulation parameters ---

SIMULATION_TYPE='lensing'   # lpt | nbody | lensing
LPT_ORDER=2
INTERP="none"
SCHEME="bilinear"    # ngp | bilinear | rbf_neighbor
PAINT_NSIDE=""       # empty = use --nside value; set integer to override

# --- Integration parameters ---
T0=0.1
T1=1.0
NB_STEPS=18               # fixed, not part of grid
DRIFT_ON_LC="--drift-on-lightcone"
EQUAL_VOL=false
MIN_WIDTH=50.0
# Space-separated scalar density widths (each becomes a separate grid dimension).
# Leave empty for default behaviour.
DENSITY_WIDTHS=""

# --- Shell / Lightcone parameters ---
NB_SHELLS=10
HALO_FRACTION=8

# --- Lensing parameters ---
NZ_SHEAR="s3"           # only used when SIMULATION_TYPE=lensing
MIN_Z=0.01              # minimum redshift for n(z) integration (default: 0.01)
MAX_Z=1.5               # maximum redshift for n(z) integration (default: 1.5)
N_INTEGRATE=32          # Simpson quadrature points for n(z) distributions (default: 32)

# --- Precision ---
ENABLE_X64=false       # set to "true" to enable JAX 64-bit precision

# --- Grid parameters ---
# MESH_SIZES: each element is "MX MY MZ"; all are passed as a flat list to fli-grid.
# BOX_SIZES, OMEGA_C, SIGMA_8, SEED support two styles (can be mixed):
#   Explicit list:  OMEGA_C=(0.2589 0.3 0.4)
#   Range notation: OMEGA_C=("0.25:0.45:0.05")   → 0.25 0.30 0.35 0.40 0.45 (stop inclusive)
#   Seed range:     SEED=("0:9:1")                → seeds 0..9
MESH_SIZES=(
    "64 64 64"
    "128 128 128"
)
BOX_SIZES=(
    "500.0 500.0 500.0"
    "1000.0 1000.0 1000.0"
)
OMEGA_C=(0.2)
SIGMA_8=(0.8)
SEED=(0)
NSIDE=(512)

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

read -r PX PY <<< "$PDIMS"

BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT --time=$TIME_LIMIT --gres=gpu:$GPUS_PER_NODE --cpus-per-task=$CPUS_PER_TASK --nodes=$NODES --tasks-per-node=$TASKS_PER_NODE --qos=$QOS"

echo "Submitting single fli-grid job, time limit $TIME_LIMIT"

JOB_NAME="fli_grid_${SIMULATION_TYPE}"

if [ "$RUN_LOCALLY" = true ]; then
    if [ "$TOTAL_GPUS" -eq 1 ]; then
        SBATCH_CMD=""
    else
        SBATCH_CMD="mpirun -n $TOTAL_GPUS --oversubscribe"
    fi
elif [ "$RUN_LOCALLY" = dryrun ]; then
    SBATCH_CMD=dry_run_submit
else
    SBATCH_CMD="sbatch $BASE_SBATCH_ARGS --job-name=$JOB_NAME --output=SLURM_LOGS/%x_%j.out --error=SLURM_LOGS/%x_%j.err $SLURM_SCRIPT FLI_GRID"
fi

$SBATCH_CMD fli-grid $SIMULATION_TYPE \
    --mesh-size ${MESH_SIZES[*]} \
    --box-size ${BOX_SIZES[*]} \
    --Omega-c ${OMEGA_C[*]} \
    --sigma8 ${SIGMA_8[*]} \
    --seed ${SEED[*]} \
    --nb-shells $NB_SHELLS \
    --nb-steps $NB_STEPS \
    --nside ${NSIDE[*]} \
    --t0 $T0 \
    --t1 $T1 \
    --lpt-order $LPT_ORDER \
    --halo-fraction $HALO_FRACTION \
    --pdim $PX $PY \
    --nodes $NODES \
    --interp $INTERP \
    --scheme $SCHEME \
    $([ -n "$PAINT_NSIDE" ] && echo "--paint-nside $PAINT_NSIDE") \
    $DRIFT_ON_LC \
    --min-width $MIN_WIDTH \
    $([ "$EQUAL_VOL" = "true" ] && echo "--equal-vol") \
    $([ -n "$DENSITY_WIDTHS" ] && echo "--density-widths $DENSITY_WIDTHS") \
    $([ "$SIMULATION_TYPE" = "lensing" ] && echo "--nz-shear $NZ_SHEAR --min-z $MIN_Z --max-z $MAX_Z --n-integrate $N_INTEGRATE") \
    --h 0.6774 \
    --output-dir "$OUTPUT_DIR" \
    $([ "$ENABLE_X64" = "true" ] && echo "--enable-x64")
