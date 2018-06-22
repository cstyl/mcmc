GCC = gcc
# add the directory of the nvcc compiler
NVCC = /usr/local/cuda-9.0/bin/nvcc

# add the directories of gsl library and cuda
GSL = /usr/local
CUDA = /usr/local/cuda-9.0

GCCFLAGS = -O3 -g
GCCFLAGS += -Wall
GCCFLAGS += -Wno-unused-function

NVCCFLAGS = -O3
NVCCFLAGS += --compiler-options -Wall
NVCCFLAGS += --compiler-options -Wno-unused-function
NVCCFLAGS += -rdc=true

GENCODE_FLAGS += -gencode arch=compute_61,code=compute_61

GCCLIBS += -L$(GSL)/lib
GCCLIBS += -lm -lgsl -lgslcblas

NVCCLIBS += -L$(CUDA)/lib64
NVCCLIBS += -lcudart -lcuda -lcudadevrt
NVCCLIBS += -lcublas -lcublas_device -lcurand

LIBRARIES = $(GCCLIBS) $(NVCCLIBS)

ifeq ($(OS),Windows_NT)
LDLIBS += -lws2_32
else
LDLIBS += -lrt
endif

SRC = src
OBJ = obj
BIN = bin
LIB = lib
OUT = out
SCR = scr
DATA = data
TESTS = tests
RES = res

CSVPARSER = $(LIB)/ccsvparser/src
UTIL = $(SRC)/util
MCMC_CPU = $(SRC)/mcmc_cpu
MCMC_GPU = $(SRC)/mcmc_gpu
MCMC_SP = $(SRC)/mcmc_gpu_sp
MCMC_MP = $(SRC)/mcmc_mp
RED = $(SRC)/reduction

DATA_SCRIPTS = $(SCR)/data_scr
PLOT_SCRIPTS = $(SCR)/plot_scr
SH_SCRIPTS =  $(SCR)/sh_scr

VPATH = $(CSVPARSER) $(UTIL) $(RED)\
		$(MCMC_CPU) $(MCMC_GPU) $(MCMC_MP) $(MCMC_SP)\
 		$(TESTS) $(DATA)\
		$(DATA_SCRIPTS) $(PLOT_SCRIPTS) $(SH_SCRIPTS)

INCLUDES += -I$(GSL)/include -I$(CUDA)/include -Iinclude\
			-I$(MCMC_CPU) -I$(MCMC_GPU) -I$(MCMC_MP) -I$(MCMC_SP)\
			-I$(CSVPARSER) -I$(UTIL) -I$(RED)\
			-I$(TESTS)

LIB_OBJ = $(OBJ)/csvparser.o
UTIL_OBJ = $(OBJ)/alloc_util.o $(OBJ)/data_util.o $(OBJ)/processing_util.o \
			$(OBJ)/io.o
RED_OBJ = $(OBJ)/reduction_kernel.o
ALL_OBJ = $(LIB_OBJ) $(UTIL_OBJ) $(RED_OBJ)

CPU_OBJ = $(OBJ)/cpu_host.o $(OBJ)/mcmc_cpu.o
GPU_OBJ =  $(OBJ)/mcmc_gpu.o $(OBJ)/gpu_host.o
SP_OBJ = $(OBJ)/mcmc_gpu_sp.o $(OBJ)/gpu_sp_host.o
MP_OBJ = $(OBJ)/mcmc_mp.o $(OBJ)/mp_host.o

RUNS_CPU_OBJ = $(OBJ)/mcmc_cpu.o $(OBJ)/multiple_runs_cpu.o
RUNS_GPU_OBJ = $(OBJ)/mcmc_gpu.o $(OBJ)/multiple_runs_gpu.o
RUNS_SP_OBJ = $(OBJ)/mcmc_gpu_sp.o $(OBJ)/multiple_runs_sp.o
TUNE_CPU_OBJ = $(OBJ)/mcmc_cpu.o $(OBJ)/tunning_cpu.o
LARGE_DATA_CPU_OBJ = $(OBJ)/mcmc_cpu.o $(OBJ)/large_data_cpu.o
GPU_PERFORMANCE_OBJ = $(OBJ)/mcmc_gpu.o $(OBJ)/gpu_perf.o
GPU_CPU_COMP_OBJ = $(OBJ)/mcmc_gpu.o $(OBJ)/mcmc_cpu.o $(OBJ)/compare_cpu_gpu.o
GPU_SP_COMP_OBJ = $(OBJ)/mcmc_gpu.o $(OBJ)/mcmc_gpu_sp.o $(OBJ)/compare_gpu_sp.o
TUNERS_CPU_OBJ = $(OBJ)/mcmc_gpu.o $(OBJ)/compare_tuners.o

all: clean samplers tests generateMnist

# samplers: $(BIN)/mcmc_cpu $(BIN)/mcmc_gpu $(BIN)/mcmc_sp $(BIN)/mcmc_mp
samplers: $(BIN)/mcmc_cpu $(BIN)/mcmc_gpu $(BIN)/mcmc_sp

tests: mulRuns tuneCpu perf comp

mulRuns: $(BIN)/mul_runs_cpu $(BIN)/mul_runs_gpu $(BIN)/mul_runs_sp

tuneCpu: $(BIN)/tune_cpu

perf: $(BIN)/large_data_cpu $(BIN)/gpu_performance

comp: $(BIN)/compare_gpu_cpu $(BIN)/compare_gpu_sp $(BIN)/compare_tuners


runSamplers: allSyn allMn

# runSepSamplers: runCpu runGpu runGpuSp runGpuMp
runSepSamplers: runCpu runGpu runGpuSp

runCpu: cpuSyn cpuMn

runGpu: gpuSyn gpuMn

runGpuSp: spSyn spMn

runGpuMp: spSyn spMn

runTests: runMulRuns runPerf runTune runComp

runMulRuns: mulRunsSynthetic mulRunsMnist

runPerf: runLargeCpu runGpuPerf

runTune: runTuneCpu

runComp: runCompSyn runCompMn runCompTun

runCompSyn: runGpuCpuCompSyn runGpuSpCompSyn

runCompMn: runGpuCpuCompMn runGpuSpCompMn

runCompTun: runCompTuners


hostSynthetic:
	rm -rf $(DATA)/host
	mkdir -p $(DATA)/host/
	python $(DATA_SCRIPTS)/synthetic_generation.py -sz 1000 -dim 3 -split 1 -dir host

generateMnist:
	rm -rf $(DATA)
	mkdir -p $(DATA)
	python $(DATA_SCRIPTS)/mnist.py

allSyn: $(BIN)/mcmc_cpu $(BIN)/mcmc_gpu $(BIN)/mcmc_sp
	rm -rf $(DATA)/host
	mkdir -p $(DATA)/host/ $(OUT)/host/synthetic
	python $(DATA_SCRIPTS)/synthetic_generation.py -sz 1000 -dim 3 -split 1 -dir host
	$(BIN)/mcmc_cpu -d 1 -sz 1000 -dim 3 -samp 30000 -burn 10000 -lag 500 -tune 1
	$(BIN)/mcmc_gpu -d 1 -sz 1000 -dim 3 -samp 30000 -burn 10000 -lag 500 -tune 1 -maxThreads 128 -maxBlocks 64 -kernel 2 -cpuThresh 32
	$(BIN)/mcmc_sp -d 1 -sz 1000 -dim 3 -samp 30000 -burn 10000 -lag 500 -tune 1 -maxThreads 128 -maxBlocks 64 -kernel 2 -cpuThresh 32

allMn: $(BIN)/mcmc_cpu $(BIN)/mcmc_gpu $(BIN)/mcmc_sp
	rm -rf $(DATA)/host
	mkdir -p $(DATA)/host/ $(OUT)/host/mnist
	python $(DATA_SCRIPTS)/mnist_generation.py -sz 1000 -dim 3 -dir host
	$(BIN)/mcmc_cpu -d 2 -sz 1000 -dim 3 -samp 30000 -burn 10000 -lag 500 -tune 1
	$(BIN)/mcmc_gpu -d 2 -sz 1000 -dim 3 -samp 30000 -burn 10000 -lag 500 -tune 1 -maxThreads 128 -maxBlocks 64 -kernel 2 -cpuThresh 32
	$(BIN)/mcmc_sp -d 2 -sz 1000 -dim 3 -samp 30000 -burn 10000 -lag 500 -tune 1 -maxThreads 128 -maxBlocks 64 -kernel 2 -cpuThresh 32

cpuSyn: $(BIN)/mcmc_cpu
	rm -rf $(DATA)/host
	mkdir -p $(DATA)/host/ $(OUT)/host/synthetic
	python $(DATA_SCRIPTS)/synthetic_generation.py -sz 1000 -dim 3 -split 1 -dir host

	$(BIN)/mcmc_cpu -d 1 -sz 1000 -dim 3 -samp 30000 -burn 10000 -lag 500 -tune 1

cpuMn: $(BIN)/mcmc_cpu
	rm -rf $(DATA)/host
	mkdir -p $(DATA)/host/ $(OUT)/host/mnist
	python $(DATA_SCRIPTS)/mnist_generation.py -sz 1000  -dim 3 -dir host

	$(BIN)/mcmc_cpu -d 2 -sz 1000 -dim 3 -samp 30000 -burn 10000 -lag 500 -tune 1

gpuSyn: $(BIN)/mcmc_gpu
	rm -rf $(DATA)/host
	mkdir -p $(DATA)/host/ $(OUT)/host/synthetic
	python $(DATA_SCRIPTS)/synthetic_generation.py -sz 1000 -dim 3 -split 1 -dir host

	$(BIN)/mcmc_gpu -d 1 -sz 1000 -dim 3 -samp 30000 -burn 10000 -lag 500 -tune 1 -maxThreads 128 -maxBlocks 64 -kernel 2 -cpuThresh 32

gpuMn: $(BIN)/mcmc_gpu
	rm -rf $(DATA)/host
	mkdir -p $(DATA)/host/ $(OUT)/host/mnist
	python $(DATA_SCRIPTS)/mnist_generation.py -sz 1000  -dim 3 -dir host

	$(BIN)/mcmc_gpu -d 2 -sz 1000 -dim 3 -samp 30000 -burn 10000 -lag 500 -tune 1 -maxThreads 128 -maxBlocks 64 -kernel 2 -cpuThresh 32

mpSyn: $(BIN)/mcmc_mp
	rm -rf $(DATA)/host
	mkdir -p $(DATA)/host/ $(OUT)/host/synthetic
	python $(DATA_SCRIPTS)/synthetic_generation.py -sz 1024 -dim 3 -split 1 -dir host

	$(BIN)/mcmc_mp -d 1 -sz 1024 -dim 3 -samp 1 -burn 1 -lag 500 -tune 0 -maxThreads 128 -maxBlocks 64 -kernel 2 -cpuThresh 32

mpMn: $(BIN)/mcmc_mp
	rm -rf $(DATA)/host
	mkdir -p $(DATA)/host/ $(OUT)/host/mnist
	python $(DATA_SCRIPTS)/mnist_generation.py -sz 1000  -dim 3 -dir host

	$(BIN)/mcmc_mp -d 1 -sz 1000 -dim 3 -samp 1000 -burn 5000 -lag 500 -tune 0 -maxThreads 128 -maxBlocks 64 -kernel 2 -cpuThresh 32

spSyn: $(BIN)/mcmc_sp
	rm -rf $(DATA)/host
	mkdir -p $(DATA)/host/ $(OUT)/host/synthetic
	python $(DATA_SCRIPTS)/synthetic_generation.py -sz 1000 -dim 3 -split 1 -dir host

	$(BIN)/mcmc_sp -d 1 -sz 1000 -dim 3 -samp 30000 -burn 10000 -lag 500 -tune 1 -maxThreads 128 -maxBlocks 64 -kernel 2 -cpuThresh 32

spMn: $(BIN)/mcmc_sp
	rm -rf $(DATA)/host
	mkdir -p $(DATA)/host/ $(OUT)/host/mnist
	python $(DATA_SCRIPTS)/mnist_generation.py -sz 1000  -dim 3 -dir host

	$(BIN)/mcmc_sp -d 2 -sz 1000 -dim 3 -samp 30000 -burn 10000 -lag 500 -tune 1 -maxThreads 128 -maxBlocks 64 -kernel 2 -cpuThresh 32

gpuProf: $(BIN)/mcmc_gpu
	rm -rf $(OUT)/prof
	mkdir -p $(OUT)/prof $(RES)/prof

	./$(SH_SCRIPTS)/profiling_gpu.sh

	cp -a $(OUT)/prof/. $(RES)/prof


mulRunsSynthetic: $(BIN)/mul_runs_cpu $(BIN)/mul_runs_gpu $(BIN)/mul_runs_sp
	rm -rf $(DATA)/runs $(OUT)/runs
	mkdir -p $(DATA)/runs $(RES)/runs/out $(OUT)/runs

	python $(DATA_SCRIPTS)/synthetic_generation_3d.py -sz 500 -dim 3 -split 1 -dir runs
	$(BIN)/mul_runs_cpu -d 1 -sz 500 -dim 3 -samp 30000 -burn 10000 -lag 500 -tune 1		
	./$(SH_SCRIPTS)/multiple_runs_gpu.sh
	./$(SH_SCRIPTS)/multiple_runs_sp.sh	
	cp -a $(OUT)/runs/. $(RES)/runs/out
	python $(PLOT_SCRIPTS)/hist_plot_runs.py -d 1

mulRunsMnist: $(BIN)/mul_runs_cpu $(BIN)/mul_runs_gpu $(BIN)/mul_runs_sp
	rm -rf $(DATA)/runs $(OUT)/runs
	mkdir -p $(DATA)/runs $(RES)/runs_mnist/out $(OUT)/runs_mnist

	python $(DATA_SCRIPTS)/mnist_generation.py -sz 500 -dim 3 -dir runs
	$(BIN)/mul_runs_cpu -d 2 -sz 500 -dim 3 -samp 30000 -burn 10000 -lag 500 -tune 1	
	./$(SH_SCRIPTS)/multiple_runs_gpu_mnist.sh
	./$(SH_SCRIPTS)/multiple_runs_sp_mnist.sh
	cp -a $(OUT)/runs_mnist/. $(RES)/runs_mnist/out	
	python $(PLOT_SCRIPTS)/hist_plot_runs.py -d 2


runTuneCpu: $(BIN)/tune_cpu
	rm -rf $(DATA)/tunning_cpu
	mkdir -p $(DATA)/tunning_cpu $(OUT)/tunning_cpu $(RES)/tunning_cpu/out

	python $(DATA_SCRIPTS)/synthetic_generation_3d.py -sz 500 -dim 3 -split 1 -dir tunning_cpu
	$(BIN)/tune_cpu -d 1 -sz 500 -dim 3 -samp 20000 -burn 5000 -lag 500 -tune 1		

	python $(DATA_SCRIPTS)/mnist_generation.py -sz 500 -dim 3 -dir tunning_cpu	
	$(BIN)/tune_cpu -d 2 -sz 500 -dim 3 -samp 20000 -burn 5000 -lag 500 -tune 1
	
	cp -a $(OUT)/tunning_cpu/. $(RES)/tunning_cpu/out
	python $(PLOT_SCRIPTS)/tunning_cpu.py -d 1
	python $(PLOT_SCRIPTS)/tunning_cpu.py -d 2


runLargeCpu: $(BIN)/large_data_cpu
	rm -rf $(OUT)/large_data_cpu/
	mkdir -p $(OUT)/large_data_cpu/ $(RES)/large_data_cpu/out

	./$(SH_SCRIPTS)/large_data_cpu.sh
	cp -a $(OUT)/large_data_cpu/. $(RES)/large_data_cpu/out
	python $(PLOT_SCRIPTS)/large_data_cpu.py
	rm -rf $(DATA)/large_data_cpu	


runGpuPerf: $(BIN)/gpu_performance
	rm -rf $(OUT)/gpu_performance/
	mkdir -p $(OUT)/gpu_performance/ $(RES)/gpu_performance/out

	./$(SH_SCRIPTS)/gpu_performance.sh
	cp -a $(OUT)/gpu_performance/. $(RES)/gpu_performance/out
	python $(PLOT_SCRIPTS)/gpu_performance.py
	rm -rf $(DATA)/gpu_performance


runGpuCpuCompSyn: $(BIN)/compare_gpu_cpu
	rm -rf $(OUT)/compare_gpu_cpu/
	mkdir -p $(OUT)/compare_gpu_cpu/ res/compare_gpu_cpu/out

	./$(SH_SCRIPTS)/compare_gpu_cpu.sh
	cp -a $(OUT)/compare_gpu_cpu/. $(RES)/compare_gpu_cpu/out

	python $(PLOT_SCRIPTS)/compare_implementations.py -v 1 -d 1
	rm -rf $(DATA)/compare_gpu_cpu


runGpuSpCompSyn: $(BIN)/compare_gpu_sp
	rm -rf $(OUT)/compare_gpu_sp/
	mkdir -p $(OUT)/compare_gpu_sp/ $(RES)/compare_gpu_sp/out

	./$(SH_SCRIPTS)/compare_gpu_sp.sh
	cp -a $(OUT)/compare_gpu_sp/. $(RES)/compare_gpu_sp/out

	python $(PLOT_SCRIPTS)/compare_implementations.py -v 2 -d 1
	rm -rf $(DATA)/compare_gpu_sp


runGpuCpuCompMn: $(BIN)/compare_gpu_cpu
	rm -rf $(OUT)/compare_gpu_cpu_mnist/
	mkdir -p $(OUT)/compare_gpu_cpu_mnist/ $(RES)/compare_gpu_cpu_mnist/out

	./$(SH_SCRIPTS)/compare_gpu_cpu_mnist.sh
	cp -a $(OUT)/compare_gpu_cpu_mnist/. $(RES)/compare_gpu_cpu_mnist/out

	python $(PLOT_SCRIPTS)/compare_implementations.py -v 1 -d 2
	rm -rf $(DATA)/compare_gpu_cpu


runGpuSpCompMn: $(BIN)/compare_gpu_sp
	rm -rf $(OUT)/compare_gpu_sp_mnist/
	mkdir -p $(OUT)/compare_gpu_sp_mnist/ $(RES)/compare_gpu_sp_mnist/out

	./$(SH_SCRIPTS)/compare_gpu_sp_mnist.sh
	cp -a $(OUT)/compare_gpu_sp_mnist/. $(RES)/compare_gpu_sp_mnist/out

	python $(PLOT_SCRIPTS)/compare_implementations.py -v 2 -d 2
	rm -rf $(DATA)/compare_gpu_sp


runCompTuners: $(BIN)/compare_tuners
	rm -rf $(OUT)/compare_tuners/
	mkdir -p $(OUT)/compare_tuners/ $(RES)/compare_tuners/out

	./$(SH_SCRIPTS)/compare_tuners.sh
	cp -a $(OUT)/compare_tuners/. $(RES)/compare_tuners/out

	python scr/plot_scr/compare_tuners.py
	rm -rf $(DATA)/compare_tuners

mpProf: $(BIN)/mcmc_mp
	mkdir -p $(OUT)/prof
	nvprof --analysis-metrics -o $(OUT)/prof/small_data_mp.nvprof bin/mcmc_mp -d 1 -sz 1024 -dim 3 -samp 1 -burn 1 -lag 500 -tune 0 -maxThreads 64 -maxBlocks 64 -kernel 2 -cpuThresh 32

clean:
	rm -rf $(BIN)
	rm -rf $(OBJ)
	rm -rf $(OUT)
	rm -rf $(DATA)

	mkdir -p $(OBJ) $(BIN) $(DATA)


$(BIN)/mcmc_cpu: $(ALL_OBJ) $(CPU_OBJ)
	$(NVCC) $^ -o $@ $(LIBRARIES)

$(BIN)/mcmc_gpu: $(ALL_OBJ) $(GPU_OBJ)
	$(NVCC) $^ -o $@ $(LIBRARIES)

$(BIN)/mcmc_sp: $(ALL_OBJ) $(SP_OBJ)
	$(NVCC) $^ -o $@ $(LIBRARIES)

$(BIN)/mcmc_mp: $(ALL_OBJ) $(MP_OBJ)
	$(NVCC) $^ -o $@ $(LIBRARIES)

$(OBJ)/%.o: %.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(OBJ)/%.o: %.c
	$(GCC) $(INCLUDES) $(GCCFLAGS) -o $@ -c $<


$(BIN)/mul_runs_cpu: $(ALL_OBJ) $(RUNS_CPU_OBJ)
	$(NVCC) $^ -o $@ $(LIBRARIES)	

$(BIN)/mul_runs_gpu: $(ALL_OBJ) $(RUNS_GPU_OBJ)
	$(NVCC) $^ -o $@ $(LIBRARIES)	

$(BIN)/mul_runs_sp: $(ALL_OBJ) $(RUNS_SP_OBJ)
	$(NVCC) $^ -o $@ $(LIBRARIES)	

$(BIN)/tune_cpu: $(ALL_OBJ) $(TUNE_CPU_OBJ)
	$(NVCC) $^ -o $@ $(LIBRARIES)

$(BIN)/large_data_cpu: $(ALL_OBJ) $(LARGE_DATA_CPU_OBJ)
	$(NVCC) $^ -o $@ $(LIBRARIES)	

$(BIN)/gpu_performance: $(ALL_OBJ) $(GPU_PERFORMANCE_OBJ)
	$(NVCC) $^ -o $@ $(LIBRARIES)	

$(BIN)/compare_gpu_cpu: $(ALL_OBJ) $(GPU_CPU_COMP_OBJ)
	$(NVCC) $^ -o $@ $(LIBRARIES)

$(BIN)/compare_gpu_sp: $(ALL_OBJ) $(GPU_SP_COMP_OBJ)
	$(NVCC) $^ -o $@ $(LIBRARIES)

$(BIN)/compare_tuners: $(ALL_OBJ) $(TUNERS_CPU_OBJ)
	$(NVCC) $^ -o $@ $(LIBRARIES)	