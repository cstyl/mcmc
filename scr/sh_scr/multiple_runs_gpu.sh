#!/bin/bash

for i in $(seq 0 2999); do
	echo "Starting $i"
	GSL_RNG_SEED=$i bin/mul_runs_gpu -d 1 -sz 500 -dim 3 -samp 30000 -burn 10000 -lag 500 -tune 1 -maxThreads 256 -maxBlocks 64 -kernel 2 -cpuThresh 32	
done