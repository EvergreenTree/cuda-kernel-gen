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
make profile-space NCU_PREFIX=sudo PROFILE_FLAGS="--memory-details --kernel-regex kernel_vector4_affine_half_output_sparse"
```

Runs the optional Nsight Compute memory-detail pass for a selected kernel. This
adds DRAM byte counters and L1/L2 sector counters to `space.json` and the HTML
report.

```bash
make specialized-check
```

Builds the explicit fixed-range polynomial binary. This is benchmark-specific:
it assumes inputs in `[1.0, 1.01]` and `niterations == 5`.

```bash
make fit-poly
```

Re-runs the offline least-squares fit for the fixed input interval and writes
`reports/latest/poly_fits.json`. This documents which polynomial degrees are
needed to stay within the benchmark's `1e-3` tolerance.

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
| Experimental sparse affine fixed-range variant | yes | 0.31 ms |
| Experimental FP16-output affine variant | yes | 0.257 ms |

The tuned default is about `43x` faster than the original strict build on this
Blackwell system while preserving the float input/output ABI. The FP16-output
experiment changes the output ABI and reaches about `51x` speedup versus the
original strict build, or about `1.20x` over the best float-output variants.

Nsight Compute on the default vector path reports about `91%` DRAM throughput,
`48%` SM throughput, `26` registers per thread, and `86%` achieved occupancy.
The workload is effectively memory-throughput limited after coalescing and fast
math; extra ILP and fixed-range polynomial approximations do not materially
improve the steady-state time.

The unchecked polynomial variant was evaluated as the benchmark-specialized
path. Its median time was effectively tied with the general vector kernel
(`0.3095 ms` vs. `0.3099 ms` in the full profile run), so it remains an
explicit experimental target instead of becoming the default.

The offline fit shows an even cheaper fixed-range approximation is valid:
`log` and `tan` only need affine fits, while the `cos` and `sin` lanes can be
constants. That sparse affine variant drops to `22` registers per thread, has no
spills, and still ties the default at about `0.31 ms`; the remaining bottleneck
is the required global-memory writeback and transaction granularity, not SFU
math.

The FP16-output experiment keeps float input, computes the fixed-range affine
map in FP32, and stores four half values per `float4` input group. Full-size
median timings over three runs with `NREPS=100` were:

| Variant | Correct | Median time | Nominal logical traffic |
| --- | --- | ---: | ---: |
| Float-output vector default | yes | 0.3099 ms | 512 MiB |
| Float-output affine loaded | yes | 0.3096 ms | 512 MiB |
| FP16-output affine loaded | yes | 0.2576 ms | 384 MiB |
| FP16-output affine sparse | yes | 0.2572 ms | 256 MiB |

Nsight Compute on `kernel_vector4_affine_half_output_sparse` reports `235 us`,
`93.84%` DRAM throughput, `5.37%` SM throughput, `30` registers per thread, and
no spills. The memory-detail pass explains why sparse and loaded FP16 output
tie: sparse has lower nominal input bytes, but it still requests `16,777,216`
L1 global-load sectors and about `268 MB` of DRAM reads because the two scalar
loads per group are 16 bytes apart across warp lanes. The win comes from
shrinking output traffic, not from skipping the unused input lanes.

A follow-up packed-store variant writes the four half results as one 64-bit
word per group. It passed correctness and measured `0.2573 ms` in one
full-size `NREPS=100` sample, which is the same performance tier as the two
`half2` stores. Store instruction count is therefore not a clear standalone
wall.

The fatbin path was also checked on the same machine:

```bash
make -B fatbin.x
./fatbin.x
CUDA_FORCE_PTX_JIT=1 ./fatbin.x
```

Native `sm_120` SASS and forced PTX JIT both measured about `0.31 ms` for the
default vectorized variant.

## Experiment Ledger

Use this ledger before starting a new optimization pass. It records the
hypothesis, the measured result on Blackwell, the profiler or SASS mechanism,
and the practical takeaway.

| Hypothesis / variant | Result number | Nsight or SASS mechanism | Takeaway |
| --- | ---: | --- | --- |
| Original row-stride launch is the headline bug | `13.21 ms` strict baseline, `5.67 ms` row-stride fast-math ablation | Warp lanes were separated by `dimx * sizeof(float)` and the launch exposed too few resident warps | Do not revisit small math tweaks until memory coalescing and launch geometry stay fixed |
| Scalar coalescing plus fast math should dominate early wins | `0.51 ms` | Contiguous global access and enough blocks to fill the GPU hide SFU latency much better | This is the main semantic-preserving structural fix |
| `float4` branch fusion should improve memory handling and divergence | `0.31 ms` | Default vector kernel emits `LDG.E.128` and `STG.E.128`; one thread owns the four `ix % 4` lanes | Keep this as the default shape while the ABI is float input and float output |
| More ILP per thread might hide SFU latency | `0.34 ms` | ILP path used `37` registers/thread with no spills, but reduced scheduling freedom enough to lose | Do not make ILP the default for this problem size |
| Blackwell launch geometry needs retuning | Best measured setting remained near `THREADS_PER_BLOCK=512`, `BLOCKS_PER_SM=32`; nearby sweeps landed around `0.319-0.327 ms` | Nsight reported about `86%` achieved occupancy and `41` active warps/SM on the default | Re-sweep only after toolkit, clocks, dimensions, or default kernel shape change |
| Register caps / `__launch_bounds__` can lift occupancy | `-maxrregcount=16/20/24/28/32` did not improve the default | ptxas reports no spills and the default uses about `26` registers/thread | Not a current limiter; use caps only if a future variant inflates registers |
| Native SASS may differ from PTX JIT on Blackwell | Both native `sm_120` and `CUDA_FORCE_PTX_JIT=1` measured about `0.31 ms` | Driver JIT did not produce a materially faster path than offline ptxas | Keep fatbin/PTX for compatibility, not as a speed lever right now |
| Fixed-range quadratic polynomial can remove transcendental calls | `0.309-0.310 ms`, correctness passes | SFU pressure disappears, but Nsight still shows about `91%` DRAM throughput and only about `48%` SM throughput | Math is no longer the wall; global writeback dominates |
| Sparse polynomial / sparse affine can exploit the narrow input interval | `0.309-0.310 ms`, correctness passes | Offline fit shows `cos`/`sin` can be constants and `log`/`tan` can be affine; sparse affine uses about `22` registers/thread | Good documentation of the benchmark-specialized bound, but still tied with default |
| FP16 output can reduce the writeback wall if the ABI can change | `0.2572-0.2576 ms`, correctness passes; packed 64-bit store sampled at `0.2573 ms` | Nsight on the sparse FP16 kernel reports `93.84%` DRAM throughput, `5.37%` SM throughput, and the same L1 load-sector footprint as the loaded variant | This is the first post-`float4` speedup; it is real but ABI-changing |
| SASS should confirm what `tan` actually costs | Default vector SASS contains `MUFU.SIN`, `MUFU.COS`, and `MUFU.RCP` in the tangent lane | `__tanf` lowers to sin/cos/reciprocal-like work, so explicit `sincos` sharing is not free across independent lanes | Worth revisiting only if the iterative scalar path becomes the target again |
| CUDA Graph replay can amortize launch overhead | Not expected to move the full-size timed kernel | The measured kernel body is already about `0.31 ms`; launch overhead is outside the CUDA-event timing loop | Useful for many small launches or end-to-end host overhead, not this main timing |
| Tensor Cores / MMA for polynomial evaluation might use idle units | Not implemented as default; expected to lose at the current fitted degree | The valid approximation is degree `0-1` per lane, so building or storing a Vandermonde-like matrix would add scalar work and memory traffic for a tiny GEMM | Revisit only for high-degree fits, many output functions per input, or a batched layout that amortizes basis construction |
| TMA, `cp.async`, and shared-memory tiling could overlap memory | Not applicable to the current pointwise path | There is one global read and one global write with no tile reuse | Save these for a problem shape with reuse or producer-consumer tiling |

## Profiling Outputs

The profiling scripts write generated artifacts under ignored `reports/`
directories:

- `timings.csv`: per-run variant timing, speedup, and effective bandwidth.
- `summary.json`: median/min/max timing and speedup summary.
- `space.json`: ptxas register/spill counts, binary sizes, logical memory
  traffic, and selected Nsight Compute metrics.
- optional memory-detail Nsight counters: DRAM read/write bytes, L1 global
  load/store sectors, and L2 read/write sectors when `--memory-details` is set.
- `index.html`: concise visual report with speedup, memory handling, occupancy,
  and resource-footprint charts.
- `poly_fits.json`: fixed-range polynomial search results and validation error.

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
3. If the output ABI can change, FP16 output is the current best experimental
   path. Packing the four half results into one 64-bit store ties the two
   `half2` stores, so the next ABI-side tests are whether downstream code can
   consume half natively and whether bfloat16 is too coarse for the `1e-3`
   tolerance.
4. If benchmark rules allow changing adjacent setup work, test fusing data
   generation with the kernel or otherwise removing one global read. The current
   optimized kernels are mostly constrained by global memory traffic.
5. Use CUDA Graph replay only for many small launches or end-to-end host
   overhead studies; it should not move the full-size kernel much because the
   measured body is already hundreds of microseconds.
6. Multi-GPU row partitioning is the clean scale-out path for larger arrays.
   It is orthogonal to the per-GPU kernel and mostly needs host orchestration.
7. Keep Tensor Cores, WGMMA, TMA, and shared-memory tiling out of the default
   path unless profiling shows a new reason; this workload is scalar
   transcendental math with almost no data reuse.

## References

- CUDA Programming Guide, coalesced global memory access:
  https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/02-basics/writing-cuda-kernels.html
- CUDA Programming Guide, mathematical functions and fast math:
  https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/05-appendices/mathematical-functions.html
- CUDA Blackwell Compatibility Guide:
  https://docs.nvidia.com/cuda/archive/12.8.2/blackwell-compatibility-guide/index.html
