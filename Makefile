# Build file for cuda_canny.
#
#   make            build bin/cuda_canny
#   make run        build and run the full demo (see run.sh)
#   make data       download the SIPI images into data/input
#   make lint       check Google C++ style with cpplint
#   make clean      remove build products
#
# Override the CUDA location or target GPUs if needed, e.g.
#   make CUDA_PATH=/usr/local/cuda-12.6 GENCODE="-arch=sm_75"

CUDA_PATH ?= /usr/local/cuda
NVCC      ?= $(CUDA_PATH)/bin/nvcc

# Real code for common architectures plus PTX for forward compatibility.
GENCODE ?= -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_70,code=sm_70 \
           -gencode arch=compute_75,code=sm_75 \
           -gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_86,code=compute_86

# -fmad=false / -ffp-contract=off stop the compilers from fusing multiply-add
# operations, so the GPU and CPU evaluate floating point identically and the
# --verify check can demand a bit-exact match.
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -fmad=false $(GENCODE)
HOSTFLAGS ?= -O3 -Wall -Wextra -ffp-contract=off
INCLUDES  := -I.
LDFLAGS   := -lpthread

BIN_DIR   := bin
BUILD_DIR := build
TARGET    := $(BIN_DIR)/cuda_canny

CU_SRCS := src/canny_gpu.cu
CC_SRCS := src/main.cc src/canny_cpu.cc src/canny_types.cc \
           src/cli_options.cc src/image_io.cc
OBJS    := $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS)) \
           $(patsubst src/%.cc,$(BUILD_DIR)/%.o,$(CC_SRCS))
HEADERS := $(wildcard src/*.h)

.PHONY: all run data lint clean

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: src/%.cu $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -Xcompiler "$(HOSTFLAGS)" -c $< -o $@

$(BUILD_DIR)/%.o: src/%.cc $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -Xcompiler "$(HOSTFLAGS)" -c $< -o $@

# stb_image is third-party code with its own warnings; build it quietly.
$(BUILD_DIR)/image_io.o: src/image_io.cc src/image_io.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -Xcompiler "-O3 -w" -c $< -o $@

$(BIN_DIR) $(BUILD_DIR):
	mkdir -p $@

run: $(TARGET)
	./run.sh

data:
	./scripts/fetch_data.sh

lint:
	cpplint --recursive --extensions=cc,cu,h src

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
