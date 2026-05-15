NVCC ?= $(shell command -v nvcc 2>/dev/null || printf /usr/local/cuda/bin/nvcc)
COMPUTE_SANITIZER ?= $(shell command -v compute-sanitizer 2>/dev/null || printf /usr/local/cuda/bin/compute-sanitizer)
PYTHON ?= python3
REPORT_DIR ?= reports/latest
CLIENT_REPORT_DIR ?= reports/client-$(shell hostname)-$(shell date -u +%Y%m%dT%H%M%SZ)
PAGES_DIR ?= docs
BENCH_NREPS ?= 100
BENCH_RUNS ?= 3
PROFILE_DIM ?= 8192
NCU_PREFIX ?=
PROFILE_FLAGS ?=
PROFILE_TUNE_FLAGS ?= $(LAYOUT_FLAGS)
BENCH_TUNE_FLAGS ?=
CLIENT_PROFILE_FLAGS ?= --memory-details

OPT_SRC := src/cuda_prog.cu
PROBLEM_SRC := problem/cuda_prog_unoptimized.cu
MULTI_GPU_SRC := src/multi_gpu_row_shard.cu

COMMON_FLAGS := -O3 -lineinfo -Xcompiler -fopenmp
ARCH_FLAGS ?= -arch=native
FAST_FLAGS := --use_fast_math
TUNE_FLAGS ?=
SPECIALIZED_FLAGS ?= -DUSE_POLY_APPROX_DEFAULT=1
BF16_FLAGS ?= -DENABLE_BF16_OUTPUT_EXPERIMENT=1
LAYOUT_FLAGS ?= -DENABLE_LAYOUT_SETUP_EXPERIMENT=1
GRAPH_FLAGS ?= -DENABLE_CUDA_GRAPH_EXPERIMENT=1 -DDIMX=64 -DDIMY=64 -DNREPS=5000
ERROR_FLAGS ?= -DENABLE_ERROR_STATS=1 -DENABLE_BF16_OUTPUT_EXPERIMENT=1 -DENABLE_LAYOUT_SETUP_EXPERIMENT=1
FATBIN_FLAGS := \
	-gencode arch=compute_89,code=sm_89 \
	-gencode arch=compute_90,code=sm_90 \
	-gencode arch=compute_100,code=sm_100 \
	-gencode arch=compute_120,code=sm_120 \
	-gencode arch=compute_120,code=compute_120

.PHONY: all check baseline-check specialized-check bf16-experiment layout-experiment graph-experiment error-report multi-gpu-check multi-gpu-report fatbin-check sanitize bench baseline-report profile-time profile-space hardware-report profile report client-summary client-report publish-report profile-quick fit-poly clean

all: optimized.x

optimized.x: $(OPT_SRC)
	$(NVCC) $(COMMON_FLAGS) $(ARCH_FLAGS) $(FAST_FLAGS) $(TUNE_FLAGS) $< -o $@

specialized.x: $(OPT_SRC)
	$(NVCC) $(COMMON_FLAGS) $(ARCH_FLAGS) $(FAST_FLAGS) $(SPECIALIZED_FLAGS) $(TUNE_FLAGS) $< -o $@

bf16.x: $(OPT_SRC)
	$(NVCC) $(COMMON_FLAGS) $(ARCH_FLAGS) $(FAST_FLAGS) $(BF16_FLAGS) $(TUNE_FLAGS) $< -o $@

layout.x: $(OPT_SRC)
	$(NVCC) $(COMMON_FLAGS) $(ARCH_FLAGS) $(FAST_FLAGS) $(LAYOUT_FLAGS) $(TUNE_FLAGS) $< -o $@

graph.x: $(OPT_SRC)
	$(NVCC) $(COMMON_FLAGS) $(ARCH_FLAGS) $(FAST_FLAGS) $(GRAPH_FLAGS) $(TUNE_FLAGS) $< -o $@

error.x: $(OPT_SRC)
	$(NVCC) $(COMMON_FLAGS) $(ARCH_FLAGS) $(FAST_FLAGS) $(ERROR_FLAGS) $(TUNE_FLAGS) $< -o $@

baseline.x: $(PROBLEM_SRC)
	$(NVCC) $(COMMON_FLAGS) $< -o $@

multi-gpu.x: $(MULTI_GPU_SRC)
	$(NVCC) $(COMMON_FLAGS) $(ARCH_FLAGS) $(FAST_FLAGS) $(TUNE_FLAGS) $< -o $@

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

layout-experiment: layout.x
	./layout.x

graph-experiment: graph.x
	./graph.x

error-report:
	$(PYTHON) tools/error_report.py --build --runs $(BENCH_RUNS) --nreps $(BENCH_NREPS) --dimx $(PROFILE_DIM) --dimy $(PROFILE_DIM) --output-dir $(REPORT_DIR)

multi-gpu-check: multi-gpu.x
	./multi-gpu.x

fatbin-check: fatbin.x
	./fatbin.x

sanitize: optimized.x
	$(COMPUTE_SANITIZER) --tool memcheck --kernel-name regex=kernel_vector4_fast --launch-count 1 --error-exitcode 99 ./optimized.x

bench:
	$(PYTHON) tools/bench.py --build --runs $(BENCH_RUNS) --nreps $(BENCH_NREPS) --dimx $(PROFILE_DIM) --dimy $(PROFILE_DIM) --tune-flags="$(BENCH_TUNE_FLAGS)" --output-dir $(REPORT_DIR)

baseline-report: baseline.x
	$(PYTHON) tools/baseline.py --output-dir $(REPORT_DIR)

profile-time: BENCH_TUNE_FLAGS=$(PROFILE_TUNE_FLAGS)
profile-time: bench

profile-space: optimized.x
	$(PYTHON) tools/profile.py --dimx $(PROFILE_DIM) --dimy $(PROFILE_DIM) --ncu-prefix "$(NCU_PREFIX)" --output-dir $(REPORT_DIR) $(PROFILE_FLAGS)

hardware-report:
	$(PYTHON) tools/hardware_report.py --output-dir $(REPORT_DIR)

multi-gpu-report:
	$(PYTHON) tools/multi_gpu_report.py --build --runs $(BENCH_RUNS) --nreps $(BENCH_NREPS) --dimx $(PROFILE_DIM) --dimy $(PROFILE_DIM) --output-dir $(REPORT_DIR)

report:
	$(PYTHON) tools/render_report.py --output-dir $(REPORT_DIR)

profile: profile-time profile-space hardware-report report

client-summary:
	$(PYTHON) tools/export_client_summary.py --output-dir $(REPORT_DIR)

client-report:
	$(MAKE) profile REPORT_DIR="$(CLIENT_REPORT_DIR)" PROFILE_FLAGS="$(CLIENT_PROFILE_FLAGS)" NCU_PREFIX="$(NCU_PREFIX)" PROFILE_TUNE_FLAGS="$(PROFILE_TUNE_FLAGS)"
	$(PYTHON) tools/baseline.py --build --output-dir "$(CLIENT_REPORT_DIR)"
	$(PYTHON) tools/render_report.py --output-dir "$(CLIENT_REPORT_DIR)"
	$(PYTHON) tools/export_client_summary.py --output-dir "$(CLIENT_REPORT_DIR)"

publish-report:
	$(PYTHON) tools/render_report.py --output-dir "$(REPORT_DIR)" --publish-dir "$(PAGES_DIR)"

profile-quick:
	$(PYTHON) tools/bench.py --build --runs 1 --nreps 10 --dimx 1024 --dimy 1024 --output-dir $(REPORT_DIR)/quick
	$(PYTHON) tools/profile.py --dimx 1024 --dimy 1024 --skip-ncu --output-dir $(REPORT_DIR)/quick
	$(PYTHON) tools/hardware_report.py --output-dir $(REPORT_DIR)/quick
	$(PYTHON) tools/render_report.py --output-dir $(REPORT_DIR)/quick

fit-poly:
	$(PYTHON) tools/fit_poly.py --output $(REPORT_DIR)/poly_fits.json

clean:
	rm -f *.x
