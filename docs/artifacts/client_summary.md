# CUDA Kernel Performance Summary

## Hardware

| Item | Value |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Compute capability | 12.0 |
| Device count | 2 |
| Driver | 580.126.09 |
| NVCC | Cuda compilation tools, release 12.8, V12.8.93 |
| Problem size | 8192 x 8192 |
| Bottleneck hint | memory-throughput |
| Scaling status | throughput-scaling-only |

## Baseline Contract

| Item | Value |
| --- | --- |
| Strict original baseline | 13.0800 ms |
| Drop-in vectorized default | 0.3099 ms |
| Drop-in speedup vs strict baseline | 42.21x |

## Key Variants

| Variant | Correct | Median time | Speedup vs row-stride | Logical traffic |
| --- | --- | --- | --- | --- |
| Original row-stride ablation | yes | 6.5958 ms | 1.00x | 512.0 MiB |
| Vectorized float default | yes | 0.3099 ms | 21.28x | 512.0 MiB |
| Affine float default | yes | 0.3095 ms | 21.31x | 512.0 MiB |
| FP16 output | yes | 0.2571 ms | 25.65x | 256.0 MiB |
| Compact U8 input + FP16 output | yes | 0.1046 ms | 63.06x | 160.0 MiB |
| Compact U8 input + U8 output | yes | 0.0241 ms | 273.68x | 64.0 MiB |
| Decode U8 output to float | yes | 0.1966 ms | 33.55x | 288.0 MiB |
| Float output + score pipeline | yes | 0.5786 ms | 11.40x | 832.0 MiB |
| Compact U8 output + score pipeline | yes | 0.2868 ms | 23.00x | 448.0 MiB |

## Profiler Status

| Artifact | Status |
| --- | --- |
| summary.json | present |
| space.json | present |
| hardware.json | present |
| index.html | present |
| Nsight Compute | ok |

Nsight command: `sudo -n /usr/local/cuda/bin/ncu --set basic --page raw --csv --kernel-name regex:kernel_vector4_fast --launch-count 1 ./optimized.x`

## Multi-GPU

| Item | Value |
| --- | --- |
| Topology status | ok |
| GPU-to-GPU paths | PHB |
| NVLink detected | no |
| Multi-GPU readout | The problem fits one GPU with headroom; multi-GPU work should be gated on throughput goals and measured transfer/reduction overhead. The detected GPU-to-GPU path does not include NVLink, so multi-GPU runs are most meaningful when data is already sharded by GPU, when the working set requires capacity, or when throughput matters more than a gather-heavy single-result benchmark. |

## Recommendations

- Nsight shows high DRAM pressure with modest SM utilization; prioritize byte/sector reductions over extra arithmetic work.
- Polynomial/affine math removal does not materially beat the default float-output vector kernel on this run.
- FP16 output improves runtime, so output traffic is a useful ABI-changing optimization axis on this device.
- Compact x/w input gives a large kernel-side win; verify producer layout or packing amortization before treating it as end-to-end.
- U16 fixed-point x/w input improves over FP32 compact x/w on this memory-bound run while preserving the benchmark tolerance.
- U8 fixed-point x/w input improves over U16 compact x/w on this memory-bound run while staying inside tolerance.
- Custom U8 x/w output improves over FP16 output by attacking the remaining write traffic; this is a strongly ABI-changing path.
- A downstream consumer that stays in compact U8 x/w form is much cheaper than consuming expanded float4 output.
- Even when U8 input packing is paid every launch, the compact-output pipeline wins if downstream consumes compact form directly.
- The problem fits one GPU with headroom; multi-GPU work should be gated on throughput goals and measured transfer/reduction overhead. The detected GPU-to-GPU path does not include NVLink, so multi-GPU runs are most meaningful when data is already sharded by GPU, when the working set requires capacity, or when throughput matters more than a gather-heavy single-result benchmark.

## Handoff

- Use `index.html` for visual review.
- Use `summary.json`, `space.json`, and `hardware.json` for exact machine-readable facts.
- Add any new target-machine conclusion to the README scoreboard or Experiment Ledger before applying it elsewhere.
