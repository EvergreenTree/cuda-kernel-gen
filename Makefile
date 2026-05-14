NVCC ?= $(shell command -v nvcc 2>/dev/null || printf /usr/local/cuda/bin/nvcc)
COMPUTE_SANITIZER ?= $(shell command -v compute-sanitizer 2>/dev/null || printf /usr/local/cuda/bin/compute-sanitizer)

OPT_SRC := src/cuda_prog.cu
PROBLEM_SRC := problem/cuda_prog_unoptimized.cu

COMMON_FLAGS := -O3 -lineinfo -Xcompiler -fopenmp
ARCH_FLAGS ?= -arch=native
FAST_FLAGS := --use_fast_math
TUNE_FLAGS ?=
FATBIN_FLAGS := \
	-gencode arch=compute_89,code=sm_89 \
	-gencode arch=compute_90,code=sm_90 \
	-gencode arch=compute_100,code=sm_100 \
	-gencode arch=compute_120,code=sm_120 \
	-gencode arch=compute_120,code=compute_120

.PHONY: all check baseline-check fatbin-check sanitize clean

all: optimized.x

optimized.x: $(OPT_SRC)
	$(NVCC) $(COMMON_FLAGS) $(ARCH_FLAGS) $(FAST_FLAGS) $(TUNE_FLAGS) $< -o $@

baseline.x: $(PROBLEM_SRC)
	$(NVCC) $(COMMON_FLAGS) $< -o $@

fatbin.x: $(OPT_SRC)
	$(NVCC) $(COMMON_FLAGS) $(FAST_FLAGS) $(FATBIN_FLAGS) $(TUNE_FLAGS) $< -o $@

check: optimized.x
	./optimized.x

baseline-check: baseline.x
	./baseline.x

fatbin-check: fatbin.x
	./fatbin.x

sanitize: optimized.x
	$(COMPUTE_SANITIZER) --tool memcheck --kernel-name regex=kernel_vector4_fast --launch-count 1 --error-exitcode 99 ./optimized.x

clean:
	rm -f *.x
