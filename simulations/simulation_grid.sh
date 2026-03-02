#!/bin/bash

# Configuration
ACCOUNT="XXX"
CONSTRAINT="h100"
GPUS_PER_NODE=4
CPUS_PER_NODE=16
TASKS_PER_NODE=$GPUS_PER_NODE
PDIMS="1 1"
NODES=1
OUTPUT_DIR="results/grid_runs"
TIME_LIMIT="24:00:00"   # fli-grid runs ALL combos in one job — set generously

CPUS_PER_TASK=$((CPUS_PER_NODE / TASKS_PER_NODE))
echo "CPUS_PER_TASK: $CPUS_PER_TASK"

mkdir -p "$OUTPUT_DIR"

if [ -z "$SLURM_SCRIPT" ]; then
    echo "Error: SLURM_SCRIPT environment variable is not set."
    exit 1
fi

# Fixed simulation parameters
SIMULATION_TYPE='nbody'   # lpt | nbody | lensing
NZ_SHEAR="s3"             # only used when SIMULATION_TYPE=lensing
LENSING_TYPE="born"       # born | raytrace | both
NB_SHELLS=10
NB_STEPS=18               # fixed, not part of grid
T0=0.1
T1=1.0
LPT_ORDER=2
HALO_FRACTION=8
INTERP="none"
DRIFT_ON_LC="--drift-on-lightcone"
EQUAL_VOL=false
MIN_WIDTH=50.0
# Space-separated scalar density widths (each becomes a separate grid dimension).
# Leave empty for default behaviour.
DENSITY_WIDTHS=""

# Grid parameters
# MESH_SIZES: each element is "MX MY MZ"; all are passed as a flat list to fli-grid.
# BOX_SIZES, OMEGA_C, SIGMA_8, SEED support two styles (can be mixed):
#   Explicit list:  OMEGA_C=(0.2589 0.3 0.4)
#   Range notation: OMEGA_C=("0.25:0.45:0.05")   → 0.25 0.30 0.35 0.40 0.45 (stop inclusive)
#   Seed range:     SEED=("0:9:1")                → seeds 0..9
MESH_SIZES=(
    "64 64 64"
    "128 128 128"
    "256 256 256"
    "512 512 512"
)
BOX_SIZES=(
    "200.0 200.0 200.0"
    "400.0 400.0 400.0"
    "1000.0 1000.0 1000.0"
)
OMEGA_C=(0.2:0.5:0.1)
SIGMA_8=(0.8:0.85:0.01)
SEED=(0:9:1)
NSIDE=(512)

read -r PX PY <<< "$PDIMS"

BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT --gres=gpu:$GPUS_PER_NODE --cpus-per-task=$CPUS_PER_TASK --nodes=$NODES --tasks-per-node=$TASKS_PER_NODE --qos=qos_gpu_h100-t3"

echo "Submitting single fli-grid job, time limit $TIME_LIMIT"

sbatch $BASE_SBATCH_ARGS \
    --time=$TIME_LIMIT \
    --job-name="fli_grid_${SIMULATION_TYPE}" \
    --output="DEL/LOGS/%x_%j.out" \
    --error="DEL/LOGS/%x_%j.err" \
    $SLURM_SCRIPT LOGS fli-grid $SIMULATION_TYPE \
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
    $DRIFT_ON_LC \
    --min-width $MIN_WIDTH \
    $([ "$EQUAL_VOL" = "true" ] && echo "--equal-vol") \
    $([ -n "$DENSITY_WIDTHS" ] && echo "--density-widths $DENSITY_WIDTHS") \
    $([ "$SIMULATION_TYPE" = "lensing" ] && echo "--nz-shear $NZ_SHEAR --lensing $LENSING_TYPE") \
    --h 0.6774 \
    --output-dir "$OUTPUT_DIR"
