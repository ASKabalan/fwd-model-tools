#!/bin/bash



CHAINS=(0)
# Make 10 batches
BATCHES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

for chain in "${CHAINS[@]}"; do
    for batch in "${BATCHES[@]}"; do
        echo "Generating samples for chain $chain, batch $batch"
        ffi-samples \
            --model mock \
            --mesh-size 64 64 64 \
            --box-size 250 250 250 \
            --nside 4 \
            --num-samples 10 \
            --seed $batch \
            --path test_ffi_samples/chain_$chain \
            --batch-id $batch
    done
done
