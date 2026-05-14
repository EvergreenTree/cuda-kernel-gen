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
make bf16-experiment
```

Builds and runs the BF16-output precision boundary test. This is expected to
fail the `1e-3` tolerance and still exit successfully, because BF16 is tracked
as an explicit expected-fail experiment rather than a default correctness path.

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

| Build / variant | ABI / role | Correct | Time per launch |
| --- | --- | --- | ---: |
| Original problem definition | float in/out | yes | 13.21 ms |
| Original row-stride ablation | float in/out | yes | 5.67 ms |
| Scalar coalesced ablation | float in/out | yes | 0.51 ms |
| Vectorized default | float in/out | yes | 0.31 ms |
| Fixed-range polynomial / affine family | float in/out | yes | 0.309-0.310 ms |
| FP16-output affine family | float in, half out | yes | 0.257 ms |
| BF16-output affine boundary | float in, BF16 out | expected no | 0.257 ms |

The durable hypotheses, profiler mechanisms, and stop/revisit decisions live in
the Experiment Ledger below; this section is intentionally just the scoreboard.

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
| BF16 output might be cheaper enough while staying inside tolerance | `0.2570 ms`, expected failure; first checked element had `rdiff 0.001955` | BF16 has the same output byte count as FP16 here but too few mantissa bits for the benchmark tolerance | Do not use BF16 unless the tolerance relaxes or output error is judged differently downstream |
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

- [x] Re-sweep launch geometry on Blackwell. Best setting stayed near
  `THREADS_PER_BLOCK=512`, `BLOCKS_PER_SM=32`.
- [x] Capture Nsight basic metrics for the default vector kernel.
- [x] Add an optional Nsight memory-detail pass for DRAM bytes and L1/L2 sectors.
- [x] Fit benchmark-specialized polynomials and reduce them to sparse affine
  where tolerance allows.
- [x] Test FP16 output. It is the current best ABI-changing speed path.
- [x] Test packed four-half output stores. It ties the two-`half2` path.
- [x] Test BF16 output. It is too coarse for the `1e-3` tolerance.
- [ ] Test setup/layout changes that reduce actual input sectors, not just
  nominal input bytes.
- [ ] Use CUDA Graph replay only for many small launches or end-to-end host
  overhead studies.
- [ ] Add multi-GPU row partitioning only when arrays exceed one GPU or scaling
  throughput matters more than single-GPU kernel time.
- [ ] Revisit Tensor Cores/MMA only if a future formulation has high-degree
  basis work or many output functions per input.

## Iteration Tips

- Keep each hypothesis in the Experiment Ledger before or immediately after
  running it. The table should answer: what changed, what number moved, what
  mechanism explains it, and whether to revisit.
- Separate semantic-preserving paths from ABI-changing paths. FP16 output is a
  real win, but it should not silently replace the float-output default.
- Prefer one winning full sweep and cheap single-sample probes for likely
  non-winners. The CPU checker and input reset copies dominate wall time.
- Track actual memory sectors with Nsight when a sparse idea looks better on
  paper. Nominal bytes have already misled us once.
- Keep commits atomic by axis: launch geometry, approximation, ABI/storage,
  profiling/reporting, and documentation.

## References

- CUDA Programming Guide, coalesced global memory access:
  https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/02-basics/writing-cuda-kernels.html
- CUDA Programming Guide, mathematical functions and fast math:
  https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/05-appendices/mathematical-functions.html
- CUDA Blackwell Compatibility Guide:
  https://docs.nvidia.com/cuda/archive/12.8.2/blackwell-compatibility-guide/index.html
