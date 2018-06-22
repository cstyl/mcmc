#!/bin/bash

Dim=(3 10 20 40 60 80 100)

echo "Starting comparison test for synthetic..."

for i in $(seq 0 6); do
    rm -rf data/compare_tuners/
    mkdir -p data/compare_tuners/

    echo "Dataset with Nd: 10000 and Dim: ${Dim[$i]}"
    python scr/data_scr/synthetic_generation.py -split 2000 -sz 10000 -dim ${Dim[$i]} -dir compare_tuners
    echo "Dataset completed"

	 # echo "Starting test for dataset Nd: ${Nd[j]} Dim: ${Dim[i]}..."
	# bin/compare_tuners -d 1 -sz 10000 -dim ${Dim[10#$i]} -samp 30000 -burn 5000 -lag 500 -tune 0 \
	#  					-maxThreads 128 -maxBlocks 64 -kernel 2 -cpuThresh 32
    bin/compare_tuners -d 1 -sz 10000 -dim ${Dim[10#$i]} -samp 1 -burn 1 -lag 500 -tune 0 \
                        -maxThreads 128 -maxBlocks 64 -kernel 2 -cpuThresh 32
done


echo "Starting comparison test for mnist..."

for i in $(seq 0 6); do
    rm -rf data/compare_tuners/
    mkdir -p data/compare_tuners/

    echo "Dataset with Nd: 12000 and Dim: ${Dim[$i]}"
    python scr/data_scr/mnist_generation.py -sz 12000 -dim ${Dim[$i]} -dir compare_tuners
    echo "Dataset completed"

	 # echo "Starting test for dataset Nd: ${Nd[j]} Dim: ${Dim[i]}..."
	# bin/compare_tuners -d 2 -sz 12000 -dim ${Dim[10#$i]} -samp 30000 -burn 5000 -lag 500 -tune 0 \
	#  					-maxThreads 128 -maxBlocks 64 -kernel 2 -cpuThresh 32
    bin/compare_tuners -d 2 -sz 12000 -dim ${Dim[10#$i]} -samp 1 -burn 1 -lag 500 -tune 0 \
                        -maxThreads 128 -maxBlocks 64 -kernel 2 -cpuThresh 32
done


