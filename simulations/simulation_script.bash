#!/bin/bash

# Configuration
ACCOUNT="XXX"
CONSTRAINT="h100"
GPUS_PER_NODE=4
CPUS_PER_NODE=16
TASKS_PER_NODE=$GPUS_PER_NODE
PDIMS="2 1"
NODES=1
OUTPUT_DIR="results/cosmology_runs"

CPUS_PER_TASK=$((CPUS_PER_NODE / TASKS_PER_NODE))
echo "CPUS_PER_TASK: $CPUS_PER_TASK"
mkdir -p "$OUTPUT_DIR"

# Check for SLURM_SCRIPT environment variable
if [ -z "$SLURM_SCRIPT" ]; then
    echo "Error: SLURM_SCRIPT environment variable is not set."
    exit 1
fi

# Simulation fixed parameters
NZ_SHEAR="s3"
SIMULATION_TYPE='nbody' # can also be lpt or lensing
LENSING_METHOD='raytrace' # born, raytrace, or both (only used when SIMULATION_TYPE=lensing)
NSIDE=1024
NB_SHELLS=10
T0=0.1
T1=1.0
NUM_STEPS=40
DT0=$(echo "scale=4; ($T1 - $T0) / ($NUM_STEPS - 1)" | bc)
INTERP="none"
DRIFT_ON_LC="--drift-on-lightcone"
EQUAL_VOL=false    # set to "true" to enable equal-volume shells
MIN_WIDTH=50.0     # minimum shell width in Mpc/h (used when EQUAL_VOL=true)

# Simulation-grid parameters
MESH_SIZES=(
    "512 512 512"
    "1024 1024 1024"
    "1536 1536 1536"
    "2048 2048 2048"
    "3072 3072 3072"
    "4096 4096 4096"
)
# 1/8
HALO_SIZES=(
    "64 64"
    "128 128"
    "192 192"
    "256 256"
    "384 384"
    "512 512"
)
BOX_SIZES=(
    "6000.0 6000.0 6000.0"
)
OMEGA_C=(0.2589)
SIGMA_8=(0.8159)
SEED=(0)

# Common SBATCH arguments base
BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT --time=00:30:00 --gres=gpu:$GPUS_PER_NODE --cpus-per-task=$CPUS_PER_TASK --nodes=$NODES --tasks-per-node=$TASKS_PER_NODE --qos=qos_gpu_h100-t3"

# Extract Pdim_X and Pdim_Y
read -r PX PY <<< "$PDIMS"

# Function to submit simulations
run_simulations() {
    echo "Launching cosmology simulations..."

    # Loop 1: Box Sizes
    for BOX in "${BOX_SIZES[@]}"; do
        BOX_NAME=${BOX// /x}
        BOX_NAME=${BOX_NAME//.0/}

        # Loop 2: Mesh and Halo Sizes (Iterated together using array index)
        for i in "${!MESH_SIZES[@]}"; do
            MESH="${MESH_SIZES[$i]}"
            HALO="${HALO_SIZES[$i]}"

            MESH_NAME=${MESH// /x}

            # Loop 3: Omega_c
            for OC in "${OMEGA_C[@]}"; do

                # Loop 4: sigma8
                for S8 in "${SIGMA_8[@]}"; do

                    # Loop 5: Seed
                    for SD in "${SEED[@]}"; do

                        JOB_NAME="${CONSTRAINT}_cosmo_M${MESH_NAME}_B${BOX_NAME}_STEPS${NUM_STEPS}_H${HALO}_c${OC}_S8${S8}_s${SD}"

                        echo "Submitting $JOB_NAME"
                        echo "  -> Box: $BOX | Mesh: $MESH | Halo: $HALO | NUM_STEPS: $NUM_STEPS | Oc: $OC | S8: $S8 | Seed: $SD"

                        OUT_PARQUET_FILE="$OUTPUT_DIR/${JOB_NAME}.parquet"

                        # Notice: removed the duplicate sbatch args since they are in BASE_SBATCH_ARGS
                        sbatch $BASE_SBATCH_ARGS \
                            --job-name="$JOB_NAME" \
                            --output="DEL/LOGS/%x_%j.out" \
                            --error="DEL/LOGS/%x_%j.err" \
                            $SLURM_SCRIPT LOGS fli-simulate $SIMULATION_TYPE \
                            --mesh-size $MESH \
                            --box-size $BOX \
                            --pdim $PX $PY \
                            --nodes $NODES \
                            --halo-size $HALO \
                            --nside $NSIDE \
                            --nb-shells $NB_SHELLS \
                            --t0 $T0 \
                            --dt0 $DT0 \
                            --t1 $T1 \
                            --interp $INTERP \
                            $DRIFT_ON_LC \
                            --nz-shear $NZ_SHEAR \
                            --lensing $LENSING_METHOD \
                            --Omega-c $OC \
                            --sigma8 $S8 \
                            --seed $SD \
                            --h 0.6774 \
                            --output "$OUT_PARQUET_FILE" \
                            --perf \
                            --iterations 3


                    done
                done
            done
        done
    done
}

# Execute the grid
run_simulations
