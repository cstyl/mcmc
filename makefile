SERVER = 1

ifeq ($(SERVER),1)
GCC = gcc
NVCC = /mnt/applications/nvidia/cuda-9.0/bin/nvcc

GSL = /mnt/applications/gsl/2.4
CUDA = /mnt/applications/nvidia/cuda-9.0


GCCFLAGS += -O3
GCCFLAGS += -g -W -Wall
GCCFLAGS += -Wno-unused-function
GCCFLAGS += -ffast-math
GCCFLAGS += -funroll-loops

ARCHFLAGS += -arch=sm_30
ARCHFLAGS += -gencode=arch=compute_30,code=sm_30 

NVCCFLAGS += -O3
NVCCFLAGS += --compiler-options -Wall
NVCCFLAGS += --compiler-options -g
NVCCFLAGS += --compiler-options -ffast-math
NVCCFLAGS += --compiler-options -Wno-unused-function
NVCCFLAGS += --compiler-options -funroll-loops
NVCCFLAGS += -use_fast_math
NVCCFLAGS += $(ARCHFLAGS)

GCCLIBS += -L$(GSL)/lib
GCCLIBS += -lm -lgsl -lgslcblas

NVCCLIBS += -L$(CUDA)/lib64
NVCCLIBS += -lcudart 
NVCCLIBS += -lcublas

LIBRARIES = $(GCCLIBS) $(NVCCLIBS)
else
GCC = gcc
NVCC = 

GSL = /usr/local
CUDA = 

GCCFLAGS += -O3
GCCFLAGS += -g -W -Wall
GCCFLAGS += -Wno-unused-function
GCCFLAGS += -ffast-math
GCCFLAGS += -funroll-loops

ARCHFLAGS = 
NVCCFLAGS = 

GCCLIBS += -L$(GSL)/lib
GCCLIBS += -lm -lgsl -lgslcblas

NVCCLIBS +=
LIBRARIES = $(GCCLIBS) $(NVCCLIBS)
endif

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

CSVPARSER = $(LIB)/ccsvparser/src
UTIL = $(SRC)/util
MCMC_CPU = $(SRC)/mcmc_cpu
MCMC_GPU = $(SRC)/mcmc_gpu

TEST = $(SRC)/tests

DATA_SCRIPTS = $(SCR)/data_scr
PLOT_SCRIPTS = $(SCR)/plot_scr
SH_SCRIPTS =  $(SCR)/sh_scr

VPATH = $(CSVPARSER) $(UTIL) $(MCMC_CPU) $(MCMC_GPU)\
 		$(TEST) $(DATA)\
		$(DATA_SCRIPTS) $(PLOT_SCRIPTS) $(SH_SCRIPTS)

INCLUDES += -I$(GSL)/include -I$(CUDA)/include -Iinclude \
			-I$(MCMC_CPU) -I$(MCMC_GPU) -I$(CSVPARSER) -I$(UTIL)

LIB_OBJ = $(OBJ)/csvparser.o
UTIL_OBJ = $(OBJ)/alloc_util.o $(OBJ)/data_util.o $(OBJ)/processing_util.o \
			$(OBJ)/io.o
ALL_OBJ = $(LIB_OBJ) $(UTIL_OBJ)
CPU_OBJ = $(OBJ)/cpu_host.o $(OBJ)/mcmc_cpu.o
GPU_OBJ =  $(OBJ)/mcmc_gpu.o $(OBJ)/mcmc_gpu_kernel.o $(OBJ)/gpu_util.o $(OBJ)/gpu_host.o

PERF_TEST_OBJ = $(OBJ)/performanceTest.o $(OBJ)/mcmc_cpu.o $(OBJ)/mcmc_gpu.o \
				$(OBJ)/mcmc_gpu_kernel.o $(OBJ)/gpu_util.o

all: clean $(BIN)/mcmc_cpu $(BIN)/mcmc_gpu $(BIN)/performance

both: cpu gpu

cpu: $(BIN)/mcmc_cpu
	$(BIN)/mcmc_cpu -d 1 -sz 5000 -dim 5 -samp 20000 -burn 5000 -lag 500 -tune 0

gpu: $(BIN)/mcmc_gpu
	$(BIN)/mcmc_gpu -d 1 -sz 5000 -dim 5 -samp 20000 -burn 5000 -lag 500 -tune 0 \
					-maxThreads 128 -maxBlocks 64 -kernel 0 -cpuThresh 32

hostSynthetic: $(DATA_SCRIPTS)
	rm -rf $(DATA)/host
	mkdir -p $(DATA)/host/
	python $(DATA_SCRIPTS)/synthetic_generation.py -sz 5000 -dim 5 -split 10 -dir host

performanceTest: $(DATA_SCRIPTS) $(SH_SCRIPTS) $(BIN)/performance
	rm -rf $(OUT)/performance/ $(DATA)/performance/
	mkdir -p $(OUT)/performance/synthetic/ $(OUT)/performance/mnist/
	mkdir -p $(DATA)/performance/
	./$(SH_SCRIPTS)/PerformanceTest.sh

clean:
	rm -rf $(BIN)
	rm -rf $(OBJ)
	rm -rf $(OUT)

	mkdir -p $(OBJ) $(BIN) 
	mkdir -p $(OUT)/host/synthetic $(OUT)/host/mnist
	mkdir -p $(OUT)/performance/synthetic $(OUT)/performance/mnist

$(BIN)/performance: $(ALL_OBJ) $(PERF_TEST_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIBRARIES)	

$(BIN)/mcmc_cpu: $(ALL_OBJ) $(CPU_OBJ)
	$(GCC) $(GCCFLAGS) $^ -o $@ $(LIBRARIES)

$(BIN)/mcmc_gpu: $(ALL_OBJ) $(GPU_OBJ)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIBRARIES)

$(OBJ)/%.o: %.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $< $(LIBRARIES)

$(OBJ)/%.o: %.c
	$(GCC) $(INCLUDES) $(GCCFLAGS) -o $@ -c $<



