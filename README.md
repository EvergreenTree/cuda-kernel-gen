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

## Current Optimization

The original kernel launched one warp per SM and had each warp access one
column across many rows. That pattern underused memory transactions and left
little parallelism to hide transcendental latency.

The optimized default maps each thread to one contiguous `float4`. The four
lanes naturally correspond to the repeating `ix % 4` operation pattern, so the
kernel avoids the original row-stride access pattern and removes the hot
warp-level branch chain from the main vector path.

Measured on the local NVIDIA L4 with CUDA 12.8:

| Build / variant | Correct | Time per launch |
| --- | --- | ---: |
| Original problem definition | yes | 34.88 ms |
| Optimized build, original row-stride ablation | yes | 15.03 ms |
| Optimized build, scalar coalesced ablation | yes | 2.53 ms |
| Optimized build, vectorized default | yes | 2.31 ms |

The current default is about `15.1x` faster than the original strict build on
the local L4.

## Blackwell Next Steps

1. Re-run `make fatbin-check` on the Blackwell machine and save the full
   variant table.
2. Sweep launch geometry with `TUNE_FLAGS`, for example:

   ```bash
   make -B optimized.x TUNE_FLAGS="-DTHREADS_PER_BLOCK=512 -DBLOCKS_PER_SM=16"
   ```

3. Capture Nsight Compute metrics for the vector kernel:
   - SM/SFU utilization
   - eligible warps per scheduler
   - achieved occupancy
   - global load/store sector efficiency
   - register count and instruction mix
4. Compare native `sm_120` SASS against JITed `compute_120` PTX.
5. Test whether range-specialized approximations for the fixed input
   distribution and `niterations == 5` beat CUDA intrinsics while preserving
   the `1e-3` tolerance.
6. Keep Tensor Cores, WGMMA, TMA, and shared-memory tiling out of the default
   path unless profiling shows a new reason; this workload is scalar
   transcendental math with almost no data reuse.

## References

- CUDA Programming Guide, coalesced global memory access:
  https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/02-basics/writing-cuda-kernels.html
- CUDA Programming Guide, mathematical functions and fast math:
  https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/05-appendices/mathematical-functions.html
- CUDA Blackwell Compatibility Guide:
  https://docs.nvidia.com/cuda/archive/12.8.2/blackwell-compatibility-guide/index.html
