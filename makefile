SERVER = 0

ifeq ($(SERVER),1)
GCC = gcc
NVCC = /mnt/applications/nvidia/cuda-9.0/bin/nvcc

GSL = /mnt/applications/gsl/2.4
CUDA = /mnt/applications/nvidia/cuda-9.0

GCCFLAGS += -g -W -Wall \
			-Wno-unused-function \
			-O3

ARCHFLAGS = -arch=sm_30 \
 			-gencode=arch=compute_30,code=sm_30 \
 			-gencode=arch=compute_50,code=sm_50 \
 			-gencode=arch=compute_52,code=sm_52 \
 			-gencode=arch=compute_60,code=sm_60 \
 			-gencode=arch=compute_61,code=sm_61 \
 			-gencode=arch=compute_62,code=sm_62 \
 			-gencode=arch=compute_70,code=sm_70 \
 			-gencode=arch=compute_70,code=compute_70

NVCCFLAGS = --compiler-options -Wall \
			--compiler-options -g \
			--compiler-options -Wno-unused-function \
			-O3 -cubin \
			$(ARCHFLAGS)

GCCLIBS += -L$(GSL)/lib
GCCLIBS += -lm -lgsl -lgslcblas

NVCCLIBS += -L$(CUDA)/lib64
NVCCLIBS += -lcudart -lcuda 
# NVCCLIBS += -lcublas

LIBRARIES = $(GCCLIBS) $(NVCCLIBS)
else
GCC = gcc
NVCC = 

GSL = /usr/local
CUDA = 

GCCFLAGS += -g -W -Wall \
			-Wno-unused-function \
			-O3

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

DATA_SCRIPTS = $(SCR)/data_scr
PLOT_SCRIPTS = $(SCR)/plot_scr
SH_SCRIPTS =  $(SCR)/sh_scr

VPATH = $(CSVPARSER) $(UTIL) $(MCMC_CPU) \
 		$(MCMC_GPU) \
		$(DATA) $(DATA_SCRIPTS) $(PLOT_SCRIPTS) $(SH_SCRIPTS)

INCLUDES += -I$(GSL)/include -I$(CUDA)/include -Iinclude \
			-I$(MCMC_CPU) -I$(MCMC_GPU) -I$(CSVPARSER) -I$(UTIL)

LIB_OBJ = $(OBJ)/csvparser.o
UTIL_OBJ = $(OBJ)/alloc_util.o $(OBJ)/data_util.o $(OBJ)/processing_util.o \
			$(OBJ)/io.o
ALL_OBJ = $(LIB_OBJ) $(UTIL_OBJ)
CPU_OBJ = $(OBJ)/cpu_host.o $(OBJ)/mcmc_cpu.o
GPU_V1_OBJ = $(OBJ)/gpu_host.o $(OBJ)/mcmc_gpu_v1.o
# GPU_V2_OBJ = $(OBJ)/gpu_host_v2.o $(OBJ)/mcmc_gpu_v2.o

# all: clean $(BIN)/mcmc_cpu $(BIN)/mcmc_gpu_v1
all: clean $(BIN)/mcmc_cpu
	
synthetic: $(DATA_SCRIPTS)
	python $(DATA_SCRIPTS)/synthetic_generation.py -sz 1024 -dim 256

$(BIN)/mcmc_cpu: $(ALL_OBJ) $(CPU_OBJ)
	$(GCC) $(GCCFLAGS) $^ -o $@ $(LIBRARIES)

$(BIN)/mcmc_gpu_v1: $(ALL_OBJ) $(GPU_V1_OBJ)
	# $(GCC) $(GCCFLAGS) $^ -o $@ $(LIBRARIES)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIBRARIES)


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
	$(BIN)/mcmc_cpu -d 1 -sz 150000 -dim 128 -samp 500000 -burn 50000 \
					-lag 500 -autoc 3 -norm 1 -out 3 -aout 3

gpu_v1: $(BIN)/mcmc_gpu_v1
	$(BIN)/mcmc_gpu_v1 -d 1 -sz 1024 -dim 256 -samp 20000 -burn 5000 \
						-lag 500 -autoc 3 -norm 1 -out 3 -aout 3