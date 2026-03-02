#!/bin/bash

# Configuration
ACCOUNT="XXX"
CONSTRAINT="h100"
GPUS_PER_NODE=4
CPUS_PER_NODE=16
TASKS_PER_NODE=$GPUS_PER_NODE
PDIMS="2 1"
NODES=1
OUTPUT_DIR="results/grid_runs"
TIME_LIMIT="24:00:00"   # fli-grid runs ALL combos in one job — set generously

mkdir -p "$OUTPUT_DIR"

if [ -z "$SLURM_SCRIPT" ]; then
    echo "Error: SLURM_SCRIPT environment variable is not set."
    exit 1
fi

# Fixed simulation parameters
SIMULATION_TYPE='nbody'   # lpt | nbody | lensing
NZ_SHEAR="s3"             # only used when SIMULATION_TYPE=lensing
LENSING_TYPE="born"       # born | raytrace | both
NSIDE=512
NB_SHELLS=10
T0=0.1
DT0=""          # leave empty to use NB_STEPS
NB_STEPS=18     # ignored if DT0 is set; dt0 = (t1 - t0) / (nb_steps - 1)
T1=1.0
INTERP="none"
DRIFT_ON_LC="--drift-on-lightcone"
EQUAL_VOL=false
MIN_WIDTH=50.0
HALO_SIZE="0 0"    # single value (fli-grid does not support per-mesh halo sizes)

# Grid parameters
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
OMEGA_C=(0.2589 0.3 0.4)
SIGMA_8=(0.8159 0.812 0.8)
SEED=(0 1 2)

read -r PX PY <<< "$PDIMS"

BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT \
  --gres=gpu:$GPUS_PER_NODE --cpus-per-task=$((CPUS_PER_NODE / TASKS_PER_NODE)) \
  --gpus-per-task=1 --nodes=$NODES --tasks-per-node=$TASKS_PER_NODE --exclusive"

TOTAL=$(( ${#MESH_SIZES[@]} * ${#BOX_SIZES[@]} * ${#OMEGA_C[@]} * ${#SIGMA_8[@]} * ${#SEED[@]} ))
echo "Submitting fli-grid job: $TOTAL combinations, time limit $TIME_LIMIT"

sbatch $BASE_SBATCH_ARGS \
    --time=$TIME_LIMIT \
    --job-name="fli_grid_${SIMULATION_TYPE}" \
    $SLURM_SCRIPT LOGS fli-grid $SIMULATION_TYPE \
    --mesh-size ${MESH_SIZES[*]} \
    --box-size ${BOX_SIZES[*]} \
    --Omega-c ${OMEGA_C[*]} \
    --sigma8 ${SIGMA_8[*]} \
    --seed ${SEED[*]} \
    --nb-shells $NB_SHELLS \
    $([ -n "$DT0" ] && echo "--dt0 $DT0" || echo "--nb-steps $NB_STEPS") \
    --t0 $T0 \
    --t1 $T1 \
    --nside $NSIDE \
    --pdim $PX $PY \
    --nodes $NODES \
    --halo-size $HALO_SIZE \
    --interp $INTERP \
    $DRIFT_ON_LC \
    --min-width $MIN_WIDTH \
    $([ "$EQUAL_VOL" = "true" ] && echo "--equal-vol") \
    $([ "$SIMULATION_TYPE" = "lensing" ] && echo "--nz-shear $NZ_SHEAR --lensing $LENSING_TYPE") \
    --h 0.6774 \
    --output-dir "$OUTPUT_DIR"
