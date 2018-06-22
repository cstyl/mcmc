#!/bin/bash

Dim=(3 10 20 50 100 200 500)
idx=(9 8 7 6 5 4 3)

echo "Starting Performance test..."

for i in $(seq 0 6); do
    rm -rf data/large_data_cpu/
    mkdir -p data/large_data_cpu/

    echo "Dataset with Nd: 500000 and Dim: ${Dim[$i]}"
    python scr/data_scr/synthetic_generation.py -split 5000 -sz 500000 -dim ${Dim[$i]} -dir large_data_cpu
    echo "Dataset completed"
    
    echo "Starting test for current dataset..."
    bin/large_data_cpu -d ${idx[$i]} -sz 500000 -dim ${Dim[$i]} -samp 10000 -burn 2000 -lag 500 -tune 0 -first 1
done
