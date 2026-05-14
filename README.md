# CUDA Kernel Gen

## Project Promises

- Keep the original benchmark available as the problem definition.
- Maintain an optimized implementation that passes the assignment tolerance.
- Preserve ablation paths so performance claims can be measured, not guessed.
- Build cleanly on the local Ada L4 system and carry a CUDA fatbin path for
  Hopper and Blackwell follow-up work.
- Favor practical kernel performance: memory coalescing, occupancy, register
  pressure, fast math, launch geometry, and profiler evidence.

## Scope

This project optimizes a CUDA benchmark over a fixed `8192 x 8192` float grid.
Each element performs five dependent iterations of one operation selected by
`ix % 4`:

- `log`
- `cos`
- `sin`
- `tan`

The correctness contract is the benchmark's existing `1e-3` relative tolerance.
The optimized path is allowed to use CUDA fast math when it remains inside that
tolerance.

## Project Structure

```text
.
├── Makefile
├── README.md
├── problem/
│   └── cuda_prog_unoptimized.cu
└── src/
    └── cuda_prog.cu
```

- `problem/cuda_prog_unoptimized.cu` is the original benchmark/problem
  definition.
- `src/cuda_prog.cu` is the optimized implementation and self-contained
  ablation harness.
- Generated binaries, profiler reports, and build scratch files are ignored by
  `.gitignore`.

## Build And Run

```bash
make check
```

Builds `optimized.x` from `src/cuda_prog.cu` with `-arch=native` and
`--use_fast_math`, then runs the benchmark and correctness checks.

```bash
make baseline-check
```

Builds and runs the original problem definition from
`problem/cuda_prog_unoptimized.cu`.

```bash
make fatbin-check
```

Builds the optimized implementation for `sm_89`, `sm_90`, `sm_100`, `sm_120`,
and forward-compatible `compute_120` PTX.

```bash
make sanitize
```

Runs a focused `compute-sanitizer` memcheck pass on the optimized vector kernel.

```bash
make profile-quick
```

Runs a small `1024 x 1024` timing/profile smoke test and writes an HTML report
under `reports/latest/quick/`.

```bash
make profile NCU_PREFIX=sudo
```

Runs full-size timing, Nsight Compute space/resource collection, and report
rendering under `reports/latest/`. If performance counter access is enabled for
the current user, omit `NCU_PREFIX=sudo`.

```bash
make specialized-check
```

Builds the explicit fixed-range polynomial binary. This is benchmark-specific:
it assumes inputs in `[1.0, 1.01]` and `niterations == 5`.

## Current Optimization

The original kernel launched one warp per SM and had each warp access one
column across many rows. That pattern underused memory transactions and left
little parallelism to hide transcendental latency.

The optimized default maps each thread to one contiguous `float4`. The four
lanes naturally correspond to the repeating `ix % 4` operation pattern, so the
kernel avoids the original row-stride access pattern and removes the hot
warp-level branch chain from the main vector path.

The benchmark harness now resets device input before each timed launch and
times only the kernel body with CUDA events. This keeps benchmark-specialized
variants inside their valid input domain without counting host-to-device reset
copies as kernel time.

Measured on the local NVIDIA L4 with CUDA 12.8:

| Build / variant | Correct | Time per launch |
| --- | --- | ---: |
| Original problem definition | yes | 34.88 ms |
| Optimized build, original row-stride ablation | yes | 15.03 ms |
| Optimized build, scalar coalesced ablation | yes | 2.53 ms |
| Optimized build, vectorized default | yes | 2.31 ms |

The current default is about `15.1x` faster than the original strict build on
the local L4.

## Blackwell Results

Measured on the local NVIDIA RTX PRO 6000 Blackwell Server Edition
(`sm_120`, 188 SMs) with CUDA 13.0:

| Build / variant | Correct | Time per launch |
| --- | --- | ---: |
| Original problem definition | yes | 13.21 ms |
| Optimized build, original row-stride ablation | yes | 5.67 ms |
| Optimized build, scalar coalesced ablation | yes | 0.51 ms |
| Optimized build, vectorized default | yes | 0.31 ms |
| Optimized build, ILP `float4` variant | yes | 0.34 ms |
| Experimental guarded fixed-range polynomial variant | yes | 0.31 ms |
| Experimental unchecked fixed-range polynomial variant | yes | 0.31 ms |

The tuned default is about `43x` faster than the original strict build on this
Blackwell system. Nsight Compute on the default vector path reports about `91%`
DRAM throughput, `48%` SM throughput, `26` registers per thread, and `86%`
achieved occupancy. The workload is effectively memory-throughput limited after
coalescing and fast math; extra ILP and fixed-range polynomial approximations
do not materially improve the steady-state time.

The unchecked polynomial variant was evaluated as the benchmark-specialized
path. Its median time was effectively tied with the general vector kernel
(`0.3095 ms` vs. `0.3099 ms` in the full profile run), so it remains an
explicit experimental target instead of becoming the default.

The fatbin path was also checked on the same machine:

```bash
make -B fatbin.x
./fatbin.x
CUDA_FORCE_PTX_JIT=1 ./fatbin.x
```

Native `sm_120` SASS and forced PTX JIT both measured about `0.31 ms` for the
default vectorized variant.

## Profiling Outputs

The profiling scripts write generated artifacts under ignored `reports/`
directories:

- `timings.csv`: per-run variant timing, speedup, and effective bandwidth.
- `summary.json`: median/min/max timing and speedup summary.
- `space.json`: ptxas register/spill counts, binary sizes, logical memory
  traffic, and selected Nsight Compute metrics.
- `index.html`: concise visual report with speedup, memory handling, occupancy,
  and resource-footprint charts.

## Next Moves

1. Re-sweep launch geometry if the benchmark dimensions, GPU clocks, or CUDA
   version change. The best measured Blackwell settings were close to:

   ```bash
   make -B optimized.x TUNE_FLAGS="-DTHREADS_PER_BLOCK=512 -DBLOCKS_PER_SM=32"
   ```

2. Capture deeper Nsight Compute metrics for the vector kernel if further work
   is needed:
   - SM/SFU utilization
   - eligible warps per scheduler
   - global load/store sector efficiency
   - instruction mix
3. If benchmark rules allow changing adjacent setup work, test fusing data
   generation with the kernel or otherwise removing one global read. The current
   optimized kernels are mostly constrained by global memory traffic.
4. Keep Tensor Cores, WGMMA, TMA, and shared-memory tiling out of the default
   path unless profiling shows a new reason; this workload is scalar
   transcendental math with almost no data reuse.

## References

- CUDA Programming Guide, coalesced global memory access:
  https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/02-basics/writing-cuda-kernels.html
- CUDA Programming Guide, mathematical functions and fast math:
  https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/05-appendices/mathematical-functions.html
- CUDA Blackwell Compatibility Guide:
  https://docs.nvidia.com/cuda/archive/12.8.2/blackwell-compatibility-guide/index.html
