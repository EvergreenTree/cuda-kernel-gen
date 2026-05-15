# CUDA Kernel Gen

## Client Report

Start with the published client-facing performance report:
https://evergreentree.github.io/cuda-kernel-gen/

This README is the engineering runbook for reproducing, profiling, and extending
the benchmark results.

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

The default deliverable is a semantic-preserving float in/out kernel. Additional
specialized variants are included to quantify what becomes possible when a
client can change storage format or downstream consumption.

## Project Structure

```text
.
├── Makefile
├── README.md
├── problem/
│   └── cuda_prog_unoptimized.cu
├── src/
│   └── cuda_prog.cu
└── tools/
    ├── bench.py
    ├── export_client_summary.py
    ├── hardware_report.py
    ├── profile.py
    └── render_report.py
```

- `problem/cuda_prog_unoptimized.cu` is the original benchmark/problem
  definition.
- `src/cuda_prog.cu` is the optimized implementation and self-contained
  ablation harness.
- `tools/` contains the automated timing, profiling, hardware, visualization,
  and client-summary exporters.
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

`make profile` and `make client-report` include layout/setup variants by
default through `PROFILE_TUNE_FLAGS`. Set `PROFILE_TUNE_FLAGS=` for a leaner
core-kernel report.

```bash
make client-report NCU_PREFIX=sudo
```

Runs the productized target-machine workflow into a timestamped
`reports/client-.../` directory. It enables layout/setup variants, captures
Nsight memory details, renders `index.html`, and writes `client_summary.md` for
handoff. Override `CLIENT_PROFILE_FLAGS` to change which Nsight metrics or
kernel regex are collected.

```bash
make client-summary
```

Regenerates `client_summary.md` for the current `REPORT_DIR` after an existing
profile run.

```bash
make profile-space NCU_PREFIX=sudo PROFILE_FLAGS="--memory-details --kernel-regex kernel_vector4_affine_half_output_sparse"
```

Runs the optional Nsight Compute memory-detail pass for a selected kernel. This
adds DRAM byte counters and L1/L2 sector counters to `space.json` and the HTML
report.

```bash
make hardware-report
```

Writes `hardware.json` under `REPORT_DIR`. It records device/toolchain facts,
adds a lightweight bottleneck hint from the timing and Nsight summaries, and
models memory headroom plus row partitioning for future multi-GPU runs.

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
make layout-experiment
```

Builds and runs the optional compact-input setup test. This measures both the
upper-bound compact `x/w` consumers and the end-to-end GPU pack plus compact
consumer paths, including a compact downstream projection that avoids decoding
the full output grid back to floats.

```bash
make graph-experiment
```

Builds a small `64 x 64` replay test with `NREPS=5000` to compare ordinary
stream submission against CUDA Graph replay for many tiny H2D-copy-plus-kernel
launches.

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

Measured on the local NVIDIA L4 with CUDA 12.8. Speedup is relative to the
strict original problem definition on the same GPU:

| Build / variant | Correct | Time per launch | Speedup |
| --- | --- | ---: | ---: |
| Original problem definition | yes | 34.88 ms | 1.0x |
| Optimized build, original row-stride ablation | yes | 15.03 ms | 2.3x |
| Optimized build, scalar coalesced ablation | yes | 2.53 ms | 13.8x |
| Optimized build, vectorized default | yes | 2.31 ms | 15.1x |

The current default is about `15.1x` faster than the original strict build on
the local L4.

## Blackwell Results

Measured on the local NVIDIA RTX PRO 6000 Blackwell Server Edition
(`sm_120`, 188 SMs) with CUDA 13.0. Speedup is relative to the strict original
problem definition on the same GPU; ABI-changing rows are not drop-in
replacements for the float in/out default. Expected-fail rows are boundary
probes, not acceptable winners.

| Build / variant | ABI / role | Correct | Time per launch | Speedup |
| --- | --- | --- | ---: | ---: |
| Original problem definition | float in/out | yes | 13.21 ms | 1.0x |
| Original row-stride ablation | float in/out | yes | 5.67 ms | 2.3x |
| Scalar coalesced ablation | float in/out | yes | 0.51 ms | 25.9x |
| Vectorized default | float in/out | yes | 0.31 ms | 42.6x |
| Fixed-range polynomial / affine family | float in/out | yes | 0.309-0.310 ms | 42.6x |
| FP16-output affine family | float in, half out | yes | 0.257 ms | 51.4x |
| Compact `x/w` input + FP16 output | compact float2 in, half out | yes | 0.169 ms | 78.2x |
| Compact U16 `x/w` input + FP16 output | compact ushort2 in, half out | yes | 0.124 ms | 106.5x |
| Compact U8 `x/w` input + FP16 output | compact uchar2 in, half out | yes | 0.105 ms | 125.8x |
| Compact U8 `x/w` input + U8 `x/w` output | custom compact in/out | yes | 0.025 ms | 528.4x |
| Decode compact U8 output to float | custom U8 in, float out | yes | 0.197 ms | 67.1x |
| Downstream projection from float output | float4 in, score out | yes | 0.213 ms | 62.0x |
| Downstream projection from compact U8 output | custom U8 in, score out | yes | 0.027 ms | 489.3x |
| GPU pack + compact `x/w` pipeline | float in, half out | yes | 0.445 ms | 29.7x |
| GPU pack + compact U8 `x/w` pipeline | float in, half out | yes | 0.310 ms | 42.6x |
| GPU pack + compact U8 in/out pipeline | float in, custom U8 out | yes | 0.250 ms | 52.8x |
| GPU pack + compact U8 in/out + float decode | float in/out via custom path | yes | 0.436 ms | 30.3x |
| Float output + downstream projection pipeline | float in/out + score | yes | 0.579 ms | 22.8x |
| GPU pack + compact U8 in/out + compact projection | float in, custom U8 + score | yes | 0.287 ms | 46.0x |
| Compact FP16 `x/w` input boundary | compact half2 in, half out | expected no | 0.121 ms | 109.2x |
| Compact U4 `x/w` input boundary | packed nibbles in, half out | expected no | 0.100 ms | 132.1x |
| BF16-output affine boundary | float in, BF16 out | expected no | 0.257 ms | 51.4x |

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
| Compact input layout can reduce actual input sectors | `0.1687 ms` median over 3 full-size runs | Nsight reports `137.952 us`, `91.8%` DRAM throughput, `134 MB` DRAM reads, and `4,194,304` L1 load sectors versus `16,777,216` for sparse AoS | Strong setup/layout-changing step; count packing cost unless a producer can emit compact `x/w` directly |
| Compact FP16 input might cut compact `x/w` traffic again | `0.1210 ms`, expected failure; first checked miss had `rdiff 0.001544` | Same logical traffic as U16 fixed-point, but FP16 quantization near `1.0` is too coarse for the tangent-sensitive lane | Do not use raw FP16 input under the current tolerance |
| Compact U16 fixed-point input can keep 16-bit storage and tolerance | `0.1243 ms` median over 3 full-size runs, correctness passes | Nsight reports `93.312 us`, `88.62%` DRAM throughput, `13.42%` SM throughput, `67 MB` DRAM reads, `63 MB` DRAM writes, and `2,097,152` L1 load sectors | Strong 16-bit storage result, but U8 supersedes it as the fastest kernel-side variant |
| Compact U8 fixed-point input can cut input sectors again | `0.1051 ms` median over 3 full-size runs, correctness passes | Nsight reports `76.896 us`, `78.11%` DRAM throughput, `17.16%` SM throughput, `34 MB` DRAM reads, `63 MB` DRAM writes, and `1,048,576` L1 load sectors | Fastest FP16-output kernel-side result; output writes then become the dominant traffic |
| Packed U4 fixed-point input is below the tolerance floor | `0.0998 ms`, expected failure; first checked miss had `rdiff 0.001280` | Nsight reports `68.096 us`, `69.67%` DRAM throughput, `20.41%` SM throughput, `17 MB` DRAM reads, `60 MB` DRAM writes, and `524,288` L1 load sectors | U4 proves there is one more small speed step, but the tangent lane exceeds tolerance; keep U8 as the lowest valid input encoding |
| Custom U8 x/w output can attack the final write wall | `0.0247 ms` median over 3 full-size runs, correctness passes | Nsight reports `40.160 us`, `53.7%` DRAM throughput, `37.9%` SM throughput, `34 MB` DRAM reads, `1,048,576` L1 load sectors, and `1,048,576` L1 store sectors | Fastest benchmark-specialized path; very ABI-changing because y/z are implicit constants and x/w require custom decode |
| GPU packing from original AoS can feed compact input | Pack alone `0.2565 ms`; pack plus compact consumer `0.4448 ms` | Pack kernel still reads `268 MB`, writes about `81 MB`, and requests `16,777,216` L1 load sectors | Not an end-to-end win when starting from the original float grid; compact layout must come from upstream or amortization |
| GPU packing from original AoS can feed compact U16 input | Pack alone `0.2115 ms`; pack plus U16 compact consumer `0.3535 ms` | Nsight on the U16 pack reports `210.272 us`, `92.81%` DRAM throughput, `268 MB` reads, `43 MB` writes, `16,777,216` L1 load sectors, and `2,097,152` L1 store sectors | Better than FP32 compact packing, but still slower than the `0.31 ms` default when setup is paid every launch |
| GPU packing from original AoS can feed compact U8 input | Pack alone `0.1826 ms`; pack plus U8 compact consumer `0.3097 ms` median over 3 full-size runs | Nsight on the U8 pack reports `196.320 us`, `92.78%` DRAM throughput, `268 MB` reads, `23 MB` writes, `16,777,216` L1 load sectors, and `1,048,576` L1 store sectors | This reaches parity with the default when setup is paid every launch; it becomes a win only if packing is fused, reused, or provided upstream |
| GPU packing plus custom U8 output can win end-to-end | `0.2500 ms` median over 3 full-size runs from original AoS input | Reuses the measured U8 pack and custom U8-output consumer; logical traffic drops to `320 MiB` for pack input/write plus compact output path | First setup-paid compact win, but it requires the strongest ABI specialization: U8 x/w input, U8 x/w output, and implicit y/z constants |
| Decoding custom U8 output back to float can erase the win | Decode-only `0.1972 ms`; pack plus custom U8 output plus float decode `0.4356 ms` median over 3 full-size runs | Nsight on decode reports `177.344 us`, `82.98%` DRAM throughput, `34 MB` reads, `199 MB` writes, `1,048,576` L1 load sectors, and `8,388,608` L1 store sectors | Custom output is only attractive if downstream consumes compact form or decode is fused with useful work |
| A realistic compact downstream consumer can preserve the custom-output win | Float-output projection `0.2125 ms`; compact-U8 projection `0.0274 ms`; full float pipeline plus projection `0.5789 ms`; setup-paid compact pipeline plus projection `0.2866 ms` | Nsight reports the float consumer at `211.168 us`, `92.61%` DRAM throughput, `268 MB` reads, and `8,388,608` L1 load sectors; compact consumer at `36.640 us`, `80.3%` DRAM throughput, `34 MB` reads, and `1,048,576` L1 load sectors | Custom U8 output is viable only when the next stage consumes compact x/w directly; this is the current best measured end-to-end specialized path |
| BF16 output might be cheaper enough while staying inside tolerance | `0.2570 ms`, expected failure; first checked element had `rdiff 0.001955` | BF16 has the same output byte count as FP16 here but too few mantissa bits for the benchmark tolerance | Do not use BF16 unless the tolerance relaxes or output error is judged differently downstream |
| SASS should confirm what `tan` actually costs | Default vector SASS contains `MUFU.SIN`, `MUFU.COS`, and `MUFU.RCP` in the tangent lane | `__tanf` lowers to sin/cos/reciprocal-like work, so explicit `sincos` sharing is not free across independent lanes | Worth revisiting only if the iterative scalar path becomes the target again |
| CUDA Graph replay can amortize launch overhead | On `64 x 64`, stream H2D+kernel replay measured `0.014572 ms`; graph replay measured `0.013954 ms` | Graph replay trims host submission overhead, but the tested end-to-end replay still includes the H2D copy and tiny kernel work | Useful only for many small launches; it is not a lever for the full-size event-timed kernel |
| Hardware scale can flip the bottleneck | Report now records GPU count, compute capability, memory size, max clocks, driver, NVCC, and a bottleneck hint | On this Blackwell run, high DRAM pressure plus low SM pressure marks the tuned kernels as memory-throughput bound | Treat every new GPU or problem size as a new measurement point; rerun the profile instead of carrying Blackwell conclusions blindly |
| Profiling must cover both time and space | `make client-report` now wraps timing, Nsight, hardware capture, HTML rendering, and Markdown summary export | `summary.json` stores medians/speedups, `space.json` stores ptxas plus Nsight facts, `hardware.json` stores device/scaling facts, `index.html` visualizes the practical deltas, and `client_summary.md` provides the handoff readout | Rerun the report bundle for every serious result; do not rely on stopwatch-only comparisons |
| Multi-GPU row partitioning should be gated by capacity or throughput need | Current host has one GPU; the current harness model is `784 MiB` device memory for `8192 x 8192` | `hardware.json` now records memory models, 85% headroom checks, and contiguous row-shard ranges from `nvidia-smi` | Do not implement a multi-GPU runner on this box; use the planner to decide when a future host justifies it |
| Tensor Cores / MMA for polynomial evaluation might use idle units | Not implemented as default; expected to lose at the current fitted degree | The valid approximation is degree `0-1` per lane, so building or storing a Vandermonde-like matrix would add scalar work and memory traffic for a tiny GEMM | Revisit only for high-degree fits, many output functions per input, or a batched layout that amortizes basis construction |
| TMA, `cp.async`, and shared-memory tiling could overlap memory | Not applicable to the current pointwise path | There is one global read and one global write with no tile reuse | Save these for a problem shape with reuse or producer-consumer tiling |

## Profiling Outputs

The profiling scripts write generated artifacts under ignored `reports/`
directories:

- `timings.csv`: per-run variant timing, speedup, and effective bandwidth.
- `summary.json`: median/min/max timing and speedup summary.
- `space.json`: ptxas register/spill counts, binary sizes, logical memory
  traffic, and selected Nsight Compute metrics.
- `hardware.json`: GPU/toolchain facts, bottleneck hint, hardware-specific
  adaptation notes, memory models, and row-shard feasibility for multi-GPU
  portability.
- optional memory-detail Nsight counters: DRAM read/write bytes, L1 global
  load/store sectors, and L2 read/write sectors when `--memory-details` is set.
- `index.html`: concise visual report with speedup, memory handling, occupancy,
  and resource-footprint charts.
- `client_summary.md`: concise handoff summary with hardware facts, key variant
  timings, profiler status, and recommendations.
- `poly_fits.json`: fixed-range polynomial search results and validation error.

## Glossary

| Term | Meaning in this repo |
| --- | --- |
| ABI | Application Binary Interface. Here it mostly means the kernel's data contract: float in/out, FP16 output, compact U8 x/w output, and so on. |
| AoS | Array of Structures. The original float grid is a flat array, but the optimized variants treat repeated four-lane groups as an AoS-like record. |
| BF16 | Bfloat16, a 16-bit float format with fewer mantissa bits than FP16. It is fast and compact, but missed this benchmark's `1e-3` tolerance. |
| CC | Compute Capability, NVIDIA's GPU architecture version. Blackwell here reports `12.0`. |
| CUDA Graph | Captured launch graph that can reduce CPU submission overhead for many small repeated launches. |
| DRAM | Device global memory. High DRAM throughput with low SM throughput means byte traffic, not arithmetic, is the likely wall. |
| FP16 | IEEE half precision. The FP16-output path passes tolerance and reduces write traffic. |
| H2D | Host-to-device copy. The timing harness resets device input before each measured launch, but CUDA event timing excludes that copy. Pipeline rows include GPU-side setup such as packing. |
| ILP | Instruction-Level Parallelism. More independent work per thread can hide latency, but the tested ILP path increased register pressure and lost on this problem. |
| L1/L2 sectors | Nsight memory transaction counters. They are more reliable than nominal byte counts when checking coalescing and sparse loads. |
| NCU | Nsight Compute, NVIDIA's kernel profiler. `make profile-space NCU_PREFIX=sudo PROFILE_FLAGS="--memory-details"` collects the important memory counters. |
| PTX | NVIDIA's virtual GPU instruction format. Driver JIT can compile PTX to native code for the installed GPU. |
| SASS | Native GPU machine code emitted by ptxas or the driver JIT. Use it to confirm actual load/store width and transcendental instructions. |
| Score | The synthetic downstream projection used to test whether compact U8 output can be consumed directly without expanding the whole grid back to float. |
| SFU | Special Function Unit. It handles operations like sine, cosine, reciprocal, and transcendental math. |
| SM | Streaming Multiprocessor. Blackwell timing here was on a 188-SM device. |
| Tensor Core | Matrix-multiply hardware. It is idle in the current pointwise kernel and only becomes relevant if the problem is reformulated as enough matrix-shaped work. |
| TMA | Tensor Memory Accelerator. Useful for tiled producer-consumer memory movement, not for this one-load/one-store pointwise kernel. |
| U4/U8/U16 | Unsigned 4-, 8-, or 16-bit fixed-point encodings used for compact benchmark-specific x/w storage. |
| WGMMA | Warp-Group Matrix Multiply-Accumulate. Tensor Core path for matrix work; not useful here unless a future formulation creates enough batched polynomial basis work. |
| x/w | The first and fourth lanes in each four-float group. For this input range, the second and third lanes collapse to constants after approximation, so x/w carry the useful varying data. |

## Universal Machine Workflow

The completed Blackwell checklist has been retired because its facts now live in
the scoreboard or Experiment Ledger above. Treat this workflow as the product
path for any target machine: workstation, server, cloud GPU, or future
architecture.

1. Generate the target-machine report.
   Run `make client-report NCU_PREFIX=sudo`. Preserve the generated
   `client_summary.md`, `index.html`, `summary.json`, `space.json`, and
   `hardware.json` before changing code.

2. Classify the bottleneck from evidence.
   Use `client_summary.md` for the client-facing readout and `hardware.json` /
   `space.json` for details. If the target is memory-throughput bound, focus on
   byte/sector reduction. If it becomes compute-, SFU-, occupancy-, or
   launch-sensitive, reopen only the matching ledger rows.

3. Keep ABI decisions explicit.
   Treat the float in/out vector kernel as the semantic-preserving baseline.
   Treat FP16 output, compact U8 input/output, and compact downstream consumers
   as separate ABI tracks with their own result rows.

4. Re-test the compact consumer path early.
   On Blackwell, custom U8 output only paid off when the next stage consumed
   compact `x/w` directly. Re-measure the float projection, compact projection,
   float pipeline, and setup-paid compact pipeline before recommending that ABI
   on any target.

5. Add one target-machine results section.
   Do not overwrite the Blackwell table. Add a new concise scoreboard plus any
   new ledger rows, and record whether each old conclusion held, flipped, or was
   not applicable.

6. Automate before repeating manual analysis.
   If a step will be reused on multiple machines, add or extend a Make target or
   `tools/` script before documenting it as a manual runbook step.

7. Gate larger engineering work on evidence.
   Implement multi-GPU row partitioning only on a multi-GPU host with capacity
   pressure or throughput goals. Revisit Tensor Cores, TMA, or `cp.async` only
   if the problem formulation changes enough to create matrix-shaped work or
   reusable tiles.

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
- Label upper-bound layout experiments clearly. Compact `x/w` is excellent
  kernel-side, but not equivalent to a free end-to-end win.
- For hardware portability, compare mechanisms, not just winners. A smaller GPU
  may become launch/latency sensitive; a higher-bandwidth GPU may expose math or
  occupancy again.
- Treat `hardware.json` as part of every serious result. It captures the device
  count, memory headroom, and row-shard plan needed to explain whether a win is
  single-GPU-specific or likely to scale.
- Keep commits atomic by axis: launch geometry, approximation, ABI/storage,
  profiling/reporting, and documentation.

## References

- CUDA Programming Guide, coalesced global memory access:
  https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/02-basics/writing-cuda-kernels.html
- CUDA Programming Guide, mathematical functions and fast math:
  https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/05-appendices/mathematical-functions.html
- CUDA Blackwell Compatibility Guide:
  https://docs.nvidia.com/cuda/archive/12.8.2/blackwell-compatibility-guide/index.html
