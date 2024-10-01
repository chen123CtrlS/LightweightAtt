#!/bin/bash

# Define learning rates for groups A and B, and seeds
lrs_B=(1e-4 2e-4 4e-4 8e-4 1e-3 2e-3)
lrs_A=(2e-5 5e-5 1e-4 2e-4 4e-4 8e-4)
seeds=(0)

# Initial GPU device ID to use for training
device=0
# Counter for the total number of tasks started
count=0
# Total number of GPUs available
num_gpus=8  # Adjust based on your actual setup

# Iterate over learning rates and seeds
for lr_B in "${lrs_B[@]}"; do
    for lr_A in "${lrs_A[@]}"; do
        for seed in "${seeds[@]}"; do
            echo "Running lr_B=${lr_B} lr_A=${lr_A} seed=${seed} on device=${device}"
            
            /.../LightweightAtt/sh/exp2_pre_roberta.sh "$lr_B" "$lr_A" "$seed" "$device" &
            
            # Increment the count and calculate the next device ID
            count=$((count + 1))
            device=$((count % num_gpus))

            # Wait for all background tasks to complete if we've filled up all GPUs
            if (( count % num_gpus == 0 )); then
                echo "Waiting for batch to finish"
                wait
            fi
        done
    done
done

# After all loops, wait for the last set of background tasks to complete
echo "$count tasks ran, waiting for the last batch to finish..."
wait