#!/bin/bash

Dim=(5 12 30 70 100)
idx=(12 11 10 9 8)
Nd=(500 1000 2000 5000 7000 10000 12000)

echo "Starting Performance test..."

for i in $(seq 0 4); do
    rm -rf data/compare_gpu_sp_mnist/
    mkdir -p data/compare_gpu_sp_mnist/

    echo "Dataset with Nd: 500000 and Dim: ${Dim[$i]}"
    python scr/data_scr/mnist_generation.py -sz 12000 -dim ${Dim[$i]} -dir compare_gpu_sp_mnist

    echo "Dataset completed"
    for j in $(seq 0 6); do
	    echo "Starting test for dataset Nd: ${Nd[j]} Dim: ${Dim[i]}..."
	    bin/compare_gpu_sp -d ${idx[10#$i]} -sz ${Nd[10#$j]} -dim ${Dim[10#$i]} -samp 30000 -burn 5000 -lag 500 -tune 1 \
	                        -maxThreads 256 -maxBlocks 64 -kernel 2 -cpuThresh 32
	done
done
