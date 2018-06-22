#!/bin/bash

Dim=(10 20 50 100 200)
idx=(7 6 5 4 3)
Nd=(500 1000 2000 5000 10000 20000 50000 100000 200000 300000 400000 500000)

echo "Starting Performance test..."

for i in $(seq 0 4); do
    rm -rf data/compare_gpu_cpu/
    mkdir -p data/compare_gpu_cpu/

    echo "Dataset with Nd: 500000 and Dim: ${Dim[$i]}"
    python scr/data_scr/synthetic_generation.py -split 5000 -sz 500000 -dim ${Dim[$i]} -dir compare_gpu_cpu
    echo "Dataset completed"
    for j in $(seq 0 11); do
	    echo "Starting test for dataset Nd: ${Nd[j]} Dim: ${Dim[i]}..."
	    bin/compare_gpu_cpu -d ${idx[10#$i]} -sz ${Nd[10#$j]} -dim ${Dim[10#$i]} -samp 30000 -burn 5000 -lag 500 -tune 1 \
	                        -maxThreads 256 -maxBlocks 64 -kernel 2 -cpuThresh 32
	done
done
