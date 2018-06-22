#!/bin/bash

Dim=(10 20 50 100 200)
Nd=(1310720 655360 262144 131072 65536)
idx=(3 4 5 6 7)

echo "Starting Performance test..."

for i in $(seq 0 4); do
    rm -rf data/gpu_performance/
    mkdir -p data/gpu_performance/

    echo "Dataset with Nd: ${Nd[$i]} and Dim: ${Dim[$i]}"
    python scr/data_scr/synthetic_generation.py -split 5000 -sz ${Nd[$i]} -dim ${Dim[$i]} -dir gpu_performance
    echo "Dataset completed"
    
    echo "Starting test for current dataset..."
    bin/gpu_performance -d ${idx[$i]} -sz ${Nd[$i]} -dim ${Dim[$i]} -samp 30000 -burn 5000 -lag 500 -tune 0 \
                        -maxThreads 128 -maxBlocks 64 -kernel 2 -cpuThresh 32

done
