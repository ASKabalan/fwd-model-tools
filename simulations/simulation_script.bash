#!/bin/bash
# Submits independent cosmological N-body simulations over a mesh/box/cosmology/seed
# grid via SLURM using `fli-simulate`.

# --- SLURM / Cluster configuration ---
RUN_LOCALLY=false # (true, false, or dryrun)
# If set to false then it is launched with sbatch, if set to true then it is launched locally, if set to dryrun then it prints the sbatch command without executing it.
ACCOUNT="XXX"
CONSTRAINT="h100"
GPUS_PER_NODE=4
CPUS_PER_NODE=16
TASKS_PER_NODE=$GPUS_PER_NODE
PDIMS="2 1"
NODES=1
QOS="qos_gpu_h100-t3"
TIME_LIMIT="00:30:00"

# --- I/O paths ---
OUTPUT_DIR="results/cosmology_runs"

# --- Simulation parameters ---
SIMULATION_TYPE='nbody' # can also be lpt or lensing
NSIDE=1024
LPT_ORDER=2
INTERP="none"

# --- Integration parameters ---
T0=0.1
T1=1.0
NB_STEPS=40
DRIFT_ON_LC="--drift-on-lightcone"
EQUAL_VOL=false    # set to "true" to enable equal-volume shells
MIN_WIDTH=50.0     # minimum shell width in Mpc/h (used when EQUAL_VOL=true)

# --- Shell / Lightcone parameters ---
NB_SHELLS=10
HALO_FRACTION=8
OBSERVER_POSITION="0.5 0.5 0.5"

# --- Lensing parameters ---
NZ_SHEAR="s3"
LENSING_METHOD='raytrace' # born, raytrace, or both (only used when SIMULATION_TYPE=lensing)

# --- Grid parameters ---
MESH_SIZES=(
    "512 512 512"
    "1024 1024 1024"
    "1536 1536 1536"
    "2048 2048 2048"
    "3072 3072 3072"
    "4096 4096 4096"
)
BOX_SIZES=(
    "6000.0 6000.0 6000.0"
)
OMEGA_C=(0.2589)
SIGMA_8=(0.8159)
SEED=(0)

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

# Common SBATCH arguments base
BASE_SBATCH_ARGS="--account=$ACCOUNT -C $CONSTRAINT --time=$TIME_LIMIT --gres=gpu:$GPUS_PER_NODE --cpus-per-task=$CPUS_PER_TASK --nodes=$NODES --tasks-per-node=$TASKS_PER_NODE --qos=$QOS"

# Extract Pdim_X and Pdim_Y
read -r PX PY <<< "$PDIMS"

# Function to submit simulations
run_simulations() {
    echo "Launching cosmology simulations..."

    # Loop 1: Box Sizes
    for BOX in "${BOX_SIZES[@]}"; do
        BOX_NAME=${BOX// /x}
        BOX_NAME=${BOX_NAME//.0/}

        # Loop 2: Mesh Sizes
        for MESH in "${MESH_SIZES[@]}"; do
            MESH_NAME=${MESH// /x}

            # Loop 3: Omega_c
            for OC in "${OMEGA_C[@]}"; do

                # Loop 4: sigma8
                for S8 in "${SIGMA_8[@]}"; do

                    # Loop 5: Seed
                    for SD in "${SEED[@]}"; do

                        JOB_NAME="${CONSTRAINT}_cosmo_M${MESH_NAME}_B${BOX_NAME}_STEPS${NB_STEPS}_c${OC}_S8${S8}_s${SD}"

                        echo "Submitting $JOB_NAME"
                        echo "  -> Box: $BOX | Mesh: $MESH | NB_STEPS: $NB_STEPS | Oc: $OC | S8: $S8 | Seed: $SD"

                        OUT_PARQUET_FILE="$OUTPUT_DIR/${JOB_NAME}.parquet"

                        if [ "$RUN_LOCALLY" = true ]; then
                            if [ "$TOTAL_GPUS" -eq 1 ]; then
                                SBATCH_CMD=""
                            else
                                SBATCH_CMD="mpirun -n $TOTAL_GPUS --oversubscribe"
                            fi    
                        elif [ "$RUN_LOCALLY" = dryrun ]; then
                            SBATCH_CMD=dry_run_submit
                        else
                            SBATCH_CMD="sbatch $BASE_SBATCH_ARGS --job-name=$JOB_NAME --output=SLURM_LOGS/%x_%j.out --error=SLURM_LOGS/%x_%j.err $SLURM_SCRIPT FLI_SIMULATION"
                        fi

                        $SBATCH_CMD fli-simulate $SIMULATION_TYPE \
                            --mesh-size $MESH \
                            --box-size $BOX \
                            --pdim $PX $PY \
                            --nodes $NODES \
                            --halo-fraction $HALO_FRACTION \
                            --observer-position $OBSERVER_POSITION \
                            --nside $NSIDE \
                            --nb-shells $NB_SHELLS \
                            --t0 $T0 \
                            --nb-steps $NB_STEPS \
                            --t1 $T1 \
                            --lpt-order $LPT_ORDER \
                            --interp $INTERP \
                            $DRIFT_ON_LC \
                            $([ "$EQUAL_VOL" = "true" ] && echo "--equal-vol") \
                            --min-width $MIN_WIDTH \
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
