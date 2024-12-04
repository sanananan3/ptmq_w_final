#!/bin/bash

# Parameters
models=("timm/deit_small_patch16_224.fb_in1k" "timm/deit_base_patch16_224.fb_in1k")
w_values=(4 5 6 7 8)
al=6
am=7
ah=8
lr=4e-4
iterations=5000

# Iterate over models and w values
for model in "${models[@]}"; do
  for w in "${w_values[@]}"; do
    echo "Running: python run_ptmq.py -m $model -w $w -a $al -al $al -am $am -ah $ah -lr $lr -i $iterations"
    python3 run_ptmq.py -m "$model" -w "$w" -a "$al" -al "$al" -am "$am" -ah "$ah" -lr "$lr" -i "$iterations"
  done
done