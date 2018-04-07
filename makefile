GCC = gcc
NVCC = /mnt/applications/nvidia/cuda-9.0/bin/nvcc

GSL = /mnt/applications/gsl/2.4
CUDA = /mnt/applications/nvidia/cuda-9.0

GCCFLAGS += -g -W -Wall \
			-Wno-unused-function \
			-O3

NVCCFLAGS = --compiler-options -Wall \
			--compiler-options -g \
			--compiler-options -Wno-unused-function \
			-arch=sm_30 \
			-O3

GCCLIBS += -L$(GSL)/lib
NVCCLIBS += -L$(CUDA)/lib64

GCCLIBS += -lm -lgsl -lgslcblas
NVCCLIBS += -lcudart -lcuda 
# NVCCLIBS += -lcublas

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

CSVPARSER = $(LIB)/ccsvparser/src
UTIL = $(SRC)/util
MCMC_HOST = $(SRC)
MCMC_CPU = $(SRC)/mcmc_cpu
MCMC_GPU = $(SRC)/mcmc_gpu

DATA_SCRIPTS = $(SCR)/data_scr
PLOT_SCRIPTS = $(SCR)/plot_scr
SH_SCRIPTS =  $(SCR)/sh_scr

VPATH = $(MCMC_HOST) $(CSVPARSER) $(MCMC_CPU) $(MCMC_GPU) $(UTIL) \
		$(DATA) $(DATA_SCRIPTS) $(PLOT_SCRIPTS) $(SH_SCRIPTS)

INCLUDES += -I$(GSL)/include -I$(CUDA)/include -Iinclude \
			-I$(MCMC_CPU) -I$(MCMC_GPU) -I$(CSVPARSER) -I$(MCMC_HOST) -I$(UTIL)

LIB_OBJ = $(OBJ)/csvparser.o
UTIL_OBJ = $(OBJ)/alloc_util.o $(OBJ)/data_util.o $(OBJ)/processing_util.o
ALL_OBJ = $(LIB_OBJ) $(UTIL_OBJ)
CPU_OBJ = $(OBJ)/cpu_host.o $(OBJ)/mcmc_cpu.o
GPU_OBJ = $(OBJ)/gpu_host.o $(OBJ)/mcmc_gpu_v1.o


all: clean $(BIN)/mcmc_cpu $(BIN)/mcmc_gpu

synthetic: $(DATA_SCRIPTS)
	python $(DATA_SCRIPTS)/synthetic_generation.py -sz 500 -dim 3

$(BIN)/mcmc_cpu: $(ALL_OBJ) $(CPU_OBJ)
	$(GCC) $(GCCFLAGS) $^ -o $@ $(LIBRARIES)


$(BIN)/mcmc_gpu: $(GPU_OBJ) $(ALL_OBJ)
	mkdir -p $(OBJ) $(BIN) $(OUT)/gpu/synthetic $(OUT)/gpu/mnist
	$(GCC) $(GCCFLAGS) $^ -o $@ $(LIBRARIES)

clean:
	rm -rf $(BIN)
	rm -rf $(OBJ)
	rm -rf $(OUT)

	mkdir -p $(OBJ) $(BIN) 
	mkdir -p $(OUT)/cpu/synthetic $(OUT)/cpu/mnist
	mkdir -p $(OUT)/gpu/synthetic $(OUT)/gpu/mnist

$(OBJ)/%.o: %.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) -o $@ -c $<

$(OBJ)/%.o: %.c
	$(GCC) $(INCLUDES) $(GCCFLAGS) -o $@ -c $<

cpu: $(BIN)/mcmc_cpu
	$(BIN)/mcmc_cpu -d 1 -sz 500 -dim 3 -samp 20000 -burn 5000 -sd 1.8 -lag 500 -autoc 3 -norm 1 -out 3 -aout 3
