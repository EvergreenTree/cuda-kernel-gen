NVCC ?= $(shell command -v nvcc 2>/dev/null || printf /usr/local/cuda/bin/nvcc)
COMPUTE_SANITIZER ?= $(shell command -v compute-sanitizer 2>/dev/null || printf /usr/local/cuda/bin/compute-sanitizer)
PYTHON ?= python3
REPORT_DIR ?= reports/latest
BENCH_NREPS ?= 100
BENCH_RUNS ?= 3
PROFILE_DIM ?= 8192
NCU_PREFIX ?=
PROFILE_FLAGS ?=

OPT_SRC := src/cuda_prog.cu
PROBLEM_SRC := problem/cuda_prog_unoptimized.cu

COMMON_FLAGS := -O3 -lineinfo -Xcompiler -fopenmp
ARCH_FLAGS ?= -arch=native
FAST_FLAGS := --use_fast_math
TUNE_FLAGS ?=
SPECIALIZED_FLAGS ?= -DUSE_POLY_APPROX_DEFAULT=1
BF16_FLAGS ?= -DENABLE_BF16_OUTPUT_EXPERIMENT=1
FATBIN_FLAGS := \
	-gencode arch=compute_89,code=sm_89 \
	-gencode arch=compute_90,code=sm_90 \
	-gencode arch=compute_100,code=sm_100 \
	-gencode arch=compute_120,code=sm_120 \
	-gencode arch=compute_120,code=compute_120

.PHONY: all check baseline-check specialized-check bf16-experiment fatbin-check sanitize bench profile-time profile-space profile report profile-quick fit-poly clean

all: optimized.x

optimized.x: $(OPT_SRC)
	$(NVCC) $(COMMON_FLAGS) $(ARCH_FLAGS) $(FAST_FLAGS) $(TUNE_FLAGS) $< -o $@

specialized.x: $(OPT_SRC)
	$(NVCC) $(COMMON_FLAGS) $(ARCH_FLAGS) $(FAST_FLAGS) $(SPECIALIZED_FLAGS) $(TUNE_FLAGS) $< -o $@

bf16.x: $(OPT_SRC)
	$(NVCC) $(COMMON_FLAGS) $(ARCH_FLAGS) $(FAST_FLAGS) $(BF16_FLAGS) $(TUNE_FLAGS) $< -o $@

baseline.x: $(PROBLEM_SRC)
	$(NVCC) $(COMMON_FLAGS) $< -o $@

fatbin.x: $(OPT_SRC)
	$(NVCC) $(COMMON_FLAGS) $(FAST_FLAGS) $(FATBIN_FLAGS) $(TUNE_FLAGS) $< -o $@

check: optimized.x
	./optimized.x

baseline-check: baseline.x
	./baseline.x

specialized-check: specialized.x
	./specialized.x

bf16-experiment: bf16.x
	./bf16.x

fatbin-check: fatbin.x
	./fatbin.x

sanitize: optimized.x
	$(COMPUTE_SANITIZER) --tool memcheck --kernel-name regex=kernel_vector4_fast --launch-count 1 --error-exitcode 99 ./optimized.x

bench:
	$(PYTHON) tools/bench.py --build --runs $(BENCH_RUNS) --nreps $(BENCH_NREPS) --dimx $(PROFILE_DIM) --dimy $(PROFILE_DIM) --output-dir $(REPORT_DIR)

profile-time: bench

profile-space: optimized.x
	$(PYTHON) tools/profile.py --dimx $(PROFILE_DIM) --dimy $(PROFILE_DIM) --ncu-prefix "$(NCU_PREFIX)" --output-dir $(REPORT_DIR) $(PROFILE_FLAGS)

report:
	$(PYTHON) tools/render_report.py --output-dir $(REPORT_DIR)

profile: profile-time profile-space report

profile-quick:
	$(PYTHON) tools/bench.py --build --runs 1 --nreps 10 --dimx 1024 --dimy 1024 --output-dir $(REPORT_DIR)/quick
	$(PYTHON) tools/profile.py --dimx 1024 --dimy 1024 --skip-ncu --output-dir $(REPORT_DIR)/quick
	$(PYTHON) tools/render_report.py --output-dir $(REPORT_DIR)/quick

fit-poly:
	$(PYTHON) tools/fit_poly.py --output $(REPORT_DIR)/poly_fits.json

clean:
	rm -f *.x
