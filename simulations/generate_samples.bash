#!/bin/bash

# Simulation parameters
MODEL="mock"
MESH_SIZE="64 64 64"
BOX_SIZE="250 250 250"
NSIDE=4
LPT_ORDER=2
T0=0.01
T1=1.0
NB_STEPS=100
NB_SHELLS=8
HALO_FRACTION=8
OBSERVER_POSITION="0.5 0.5 0.5"
NZ_SHEAR="s3"
LENSING="born"
INTERP="none"
NUM_SAMPLES=10
OUTPUT_DIR="test_fli_samples"

CHAINS=(0)
# Make 20 batches
BATCHES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

for chain in "${CHAINS[@]}"; do
    for batch in "${BATCHES[@]}"; do
        echo "Generating samples for chain $chain, batch $batch"
        fli-samples \
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
