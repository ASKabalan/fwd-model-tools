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

mkdir -p "$OUTPUT_DIR"

# Check for SLURM_SCRIPT environment variable
if [ -z "$SLURM_SCRIPT" ]; then
    echo "Error: SLURM_SCRIPT environment variable is not set."
    exit 1
fi

# Simulation fixed parameters
NZ_SHEAR="s3"
SIMULATION_TYPE='nbody' # can also be lpt or lensing
NSIDE=512
NB_SHELLS=10
T0=0.1
DT0=0.05
T1=1.0
INTERP="none"
DRIFT_ON_LC="--drift-on-lightcone"

# Simulation-grid parameters
MESH_SIZES=(
    "64 64 64"
    "128 128 128"
    "256 256 256"
    "512 512 512"
)
HALO_SIZES=(
    "8 8"
    "16 16"
    "32 32"
    "64 64"
)
BOX_SIZES=(
    "200.0 200.0 200.0"
    "400.0 400.0 400.0"
    "1000.0 1000.0 1000.0"
)
OMEGA_C=(0.2589 0.3 0.4)
SIGMA_8=(0.8159 0.812 0.8)
SEED=(0 1 2)

# Common SBATCH arguments base
BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT --time=01:00:00 --gres=gpu:$GPUS_PER_NODE --cpus-per-task=$((CPUS_PER_NODE / TASKS_PER_NODE)) --gpus-per-task=1 --nodes=$NODES --tasks-per-node=$TASKS_PER_NODE --exclusive"

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

                        JOB_NAME="cosmo_M${MESH_NAME}_B${BOX_NAME}_Oc${OC}_S8${S8}_s${SD}"

                        echo "Submitting $JOB_NAME"
                        echo "  -> Box: $BOX | Mesh: $MESH | Halo: $HALO | Oc: $OC | S8: $S8 | Seed: $SD"

                        OUT_PARQUET_FILE="$OUTPUT_DIR/${JOB_NAME}.parquet"

                        # Notice: removed the duplicate sbatch args since they are in BASE_SBATCH_ARGS
                        sbatch $BASE_SBATCH_ARGS \
                            --job-name="$JOB_NAME" \
                            --output="$OUTPUT_DIR/%x_%j.out" \
                            $SLURM_SCRIPT LOGS ffi-simulate $SIMULATION_TYPE \
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
                            --Omega-c $OC \
                            --sigma8 $S8 \
                            --seed $SD \
                            --h 0.6774 \
                            --output "$OUT_PARQUET_FILE"


                    done
                done
            done
        done
    done
}

# Execute the grid
run_simulations
