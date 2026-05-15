# CUDA Kernel Performance Summary

## Hardware

| Item | Value |
| --- | --- |
| GPU | NVIDIA B300 SXM6 AC |
| Compute capability | 10.3 |
| Device count | 1 |
| Driver | 580.126.09 |
| NVCC | Cuda compilation tools, release 13.0, V13.0.88 |
| Problem size | 8192 x 8192 |
| Bottleneck hint | compute-throughput |
| Scaling status | single-gpu |

## Baseline Contract

| Item | Value |
| --- | --- |
| Strict original baseline | 14.8900 ms |
| Drop-in vectorized default | 0.1848 ms |
| Drop-in speedup vs strict baseline | 80.57x |

## Key Variants

| Variant | Correct | Median time | Speedup vs kernel baseline | Data moved |
| --- | --- | --- | --- | --- |
| Original row-stride kernel | yes | 6.4807 ms | 1.00x | 512.0 MiB |
| Vectorized float default | yes | 0.1848 ms | 35.07x | 512.0 MiB |
| Affine float default | yes | 0.0846 ms | 76.60x | 512.0 MiB |
| FP16 output | yes | 0.0629 ms | 103.03x | 256.0 MiB |
| Compact U8 input + FP16 output | yes | 0.0398 ms | 162.83x | 160.0 MiB |
| Compact U8 input + U8 output | yes | 0.0321 ms | 201.89x | 64.0 MiB |
| Decode U8 output to float | yes | 0.0530 ms | 122.28x | 288.0 MiB |
| Float output + score pipeline | yes | 0.1399 ms | 46.32x | 832.0 MiB |
| Compact U8 output + score pipeline | yes | 0.1162 ms | 55.77x | 448.0 MiB |
| L2-persisting U8 in/out kernel | yes | 0.0276 ms | 540.5x | 64.0 MiB |
| uint4-packed U8 in/out kernel | yes | 0.0236 ms | 632.2x | 64.0 MiB |
| uint4 + L2 U8 in/out kernel | yes | 0.0233 ms | 638.7x | 64.0 MiB |
| L2-resident U8 in/out pipeline | yes | 0.0625 ms | 238.2x | 160.0 MiB |
| uint4 + L2 compact pipeline | yes | 0.0458 ms | 325.0x | 160.0 MiB |
| Fused compact U8 input to score | yes | 0.0316 ms | 470.6x | 96.0 MiB |
| Fused compact U8 input to score + L2 | yes | 0.0250 ms | 595.9x | 96.0 MiB |

## Extreme L2-Resident Option

| Item | Value |
| --- | --- |
| Best U8 in/out kernel | 0.0233 ms |
| Kernel-only speedup vs strict baseline | 638.7x |
| Scalar input L2 lift vs warm producer | 1.16x |
| Scalar input L2 lift vs thrashed producer | 1.55x |
| uint4 lift over scalar L2 producer | 1.18x |
| L2 lift on uint4 producer | 1.01x |
| Best producer + consumer | 0.0458 ms |
| Pipeline speedup vs strict baseline | 325.0x |
| Warm consumer vs thrashed consumer | 1.3x |
| Persisting total lift | 1.07x |
| Fused score result | 0.0250 ms |
| Fused vs best adjacent | 1.83x |
| Fused speedup vs strict baseline | 595.9x |
| Compact output footprint | 32.0 MiB |
| L2 budget on this host | 79.1 MiB |

The kernel-only L2 row combines uint4-packed U8 input/output with persisting L2 on compact input. The adjacent pipeline row uses the uint4 producer plus persisting L2 on compact output for the downstream consumer. Both are custom-ABI options, remain memory-path limited rather than compute-bound, and should be remeasured on B200-class systems where HBM3e bandwidth narrows the cache advantage.

| GPU family | Usable cache estimate | 256 MiB working set | 512 MiB working set |
| --- | --- | --- | --- |
| H100 / H200 | ~35 MiB | 8 GPUs | 15+ GPUs |
| RTX 6000 Ada / RTX 5090 | ~67 MiB | 4 GPUs | 8 GPUs |
| RTX Pro 6000 Blackwell | ~90 MiB | 3 GPUs | 6 GPUs |
| B200 | ~180 MiB logical | 2 GPUs | 3 GPUs |
| B300 SXM6 AC (measured) | ~89 MiB | 3 GPUs | 6 GPUs |

## Profiler Status

| Artifact | Status |
| --- | --- |
| summary.json | present |
| space.json | present |
| hardware.json | present |
| index.html | present |
| Nsight Compute | ok |

Nsight command: `/usr/local/cuda/bin/ncu --set basic --page raw --csv --kernel-name regex:kernel_vector4_fast --launch-count 1 ./optimized.x`

## Multi-GPU

| Item | Value |
| --- | --- |
| Topology status | ok |
| GPU-to-GPU paths | n/a |
| NVLink detected | no |
| Multi-GPU readout | This host has one GPU, so no multi-GPU speedup can be measured here. Keep row partitioning as a capacity/throughput option for a future multi-GPU host. |

## Recommendations

- SM throughput is higher than DRAM pressure; math reduction, ILP, or occupancy tuning may matter on this device.
- FP16 output improves runtime, so output data movement is a useful ABI-changing optimization axis on this device.
- Compact two-value input gives a large kernel-side win; verify producer layout or packing amortization before treating it as end-to-end.
- U16 fixed-point two-value input improves over FP32 compact two-value input while preserving the benchmark tolerance.
- U8 fixed-point two-value input improves over U16 compact two-value input while staying inside tolerance.
- Custom U8 two-value output improves over FP16 output by reducing the remaining write volume; this is a strongly ABI-changing path.
- A downstream consumer that stays in compact U8 two-value form is much cheaper than consuming expanded float4 output.
- Even when U8 input packing is paid every launch, the compact-output pipeline wins if downstream consumes compact form directly.
- This host has one GPU, so no multi-GPU speedup can be measured here. Keep row partitioning as a capacity/throughput option for a future multi-GPU host.

## Handoff

- Use `index.html` for visual review.
- Use `summary.json`, `space.json`, and `hardware.json` for exact machine-readable facts.
- Add any new target-machine conclusion to the README scoreboard or Experiment Ledger before applying it elsewhere.
