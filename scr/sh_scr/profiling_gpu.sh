#!/bin/bash
	
rm -rf data/host
mkdir -p data/host/
python scr/data_scr/synthetic_generation.py -sz 1000 -dim 3 -split 10 -dir host
nvprof --analysis-metrics -o out/prof/small_data_sp.nvprof bin/mcmc_sp -d 1 -sz 1000 -dim 3 -samp 100 -burn 1 -lag 500 -tune 0 -maxThreads 256 -maxBlocks 64 -kernel 2 -cpuThresh 32

rm -rf data/host
mkdir -p data/host/
python scr/data_scr/synthetic_generation.py -sz 20000 -dim 60 -split 10 -dir host
nvprof --analysis-metrics -o out/prof/med_data_sp.nvprof bin/mcmc_sp -d 1 -sz 20000 -dim 60 -samp 100 -burn 1 -lag 500 -tune 0 -maxThreads 256 -maxBlocks 64 -kernel 2 -cpuThresh 32

rm -rf data/host
mkdir -p data/host/
python scr/data_scr/synthetic_generation.py -sz 100000 -dim 200 -split 10 -dir host
nvprof --analysis-metrics -o out/prof/large_data_sp.nvprof bin/mcmc_sp -d 1 -sz 100000 -dim 200 -samp 100 -burn 1 -lag 500 -tune 0 -maxThreads 256 -maxBlocks 64 -kernel 2 -cpuThresh 32

