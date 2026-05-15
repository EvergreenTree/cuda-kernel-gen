#!/usr/bin/env python3
"""Summarize hardware-sensitive benchmark behavior."""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIMX = 8 * 1024
DEFAULT_DIMY = 8 * 1024
HEADROOM_FRACTION = 0.85

SMI_FIELDS = (
    "name",
    "compute_cap",
    "memory.total",
    "clocks.max.memory",
    "clocks.max.sm",
    "driver_version",
    "power.limit",
)


def run(cmd):
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode, proc.stdout.strip()


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def read_problem_size(summary):
    dimx = summary.get("dimx") or DEFAULT_DIMX
    dimy = summary.get("dimy") or DEFAULT_DIMY
    return int(dimx), int(dimy)


def parse_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def query_nvidia_smi():
    if shutil.which("nvidia-smi") is None:
        return {"status": "missing"}

    fields = ",".join(SMI_FIELDS)
    code, text = run(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0:
        return {"status": "failed", "error": text}

    devices = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != len(SMI_FIELDS):
            continue
        raw = dict(zip(SMI_FIELDS, parts))
        devices.append(
            {
                "name": raw["name"],
                "compute_capability": raw["compute_cap"],
                "memory_total_mib": parse_number(raw["memory.total"]),
                "max_memory_clock_mhz": parse_number(raw["clocks.max.memory"]),
                "max_sm_clock_mhz": parse_number(raw["clocks.max.sm"]),
                "driver_version": raw["driver_version"],
                "power_limit_w": parse_number(raw["power.limit"]),
            }
        )
    return {"status": "ok", "devices": devices, "device_count": len(devices)}


def query_topology():
    if shutil.which("nvidia-smi") is None:
        return {"status": "missing"}

    code, text = run(["nvidia-smi", "topo", "-m"])
    if code != 0:
        return {"status": "failed", "error": text}

    clean = re.sub(r"\x1b\[[0-9;]*m", "", text)
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    matrix_lines = [line for line in lines if line.startswith("GPU")]
    gpu_names = (
        [token for token in matrix_lines[0].split() if re.fullmatch(r"GPU\d+", token)]
        if matrix_lines
        else []
    )
    interconnects = []

    for line in matrix_lines[1:]:
        parts = line.split()
        if not parts:
            continue
        src = parts[0]
        for index, value in enumerate(parts[1 : 1 + len(gpu_names)]):
            if index >= len(gpu_names):
                continue
            dst = gpu_names[index]
            if src == dst:
                continue
            interconnects.append({"source": src, "target": dst, "path": value})

    has_nvlink = any(link["path"].startswith("NV") for link in interconnects)
    paths = sorted({link["path"] for link in interconnects})
    return {
        "status": "ok",
        "has_nvlink": has_nvlink,
        "paths": paths,
        "interconnects": interconnects,
        "raw": clean,
    }


def query_nvcc():
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not Path(nvcc).exists():
        return {"status": "missing"}
    code, text = run([nvcc, "--version"])
    if code != 0:
        return {"status": "failed", "error": text}
    version_line = next(
        (line.strip() for line in text.splitlines() if "release" in line), text
    )
    return {"status": "ok", "path": nvcc, "version": version_line}


def get_variant(summary, name):
    return summary.get("variants", {}).get(name, {})


def speedup(a, b):
    a_ms = a.get("median_ms")
    b_ms = b.get("median_ms")
    if not a_ms or not b_ms:
        return None
    return a_ms / b_ms


def mib(value):
    return value / (1024 * 1024)


def split_rows(dimy, device_count):
    ranges = []
    base_rows = dimy // device_count
    extra_rows = dimy % device_count
    row_start = 0
    for index in range(device_count):
        rows = base_rows + (1 if index < extra_rows else 0)
        row_end = row_start + rows
        ranges.append({"row_start": row_start, "row_end": row_end, "rows": rows})
        row_start = row_end
    return ranges


def build_memory_models(dimx, dimy):
    elements = dimx * dimy
    return [
        {
            "name": "current_harness_device_allocations",
            "bytes_per_element": 12.25,
            "total_bytes": int(elements * 12.25),
            "note": "Current benchmark allocates float input/output plus FP16 output, U8 output, downstream score output, and float, FP16, u16, u8, and u4 compact x/w scratch buffers.",
        },
        {
            "name": "float_inplace_default",
            "bytes_per_element": 4.0,
            "total_bytes": elements * 4,
            "note": "Production minimum for the float in-place default kernel.",
        },
        {
            "name": "float_input_fp16_output",
            "bytes_per_element": 6.0,
            "total_bytes": elements * 6,
            "note": "ABI-changing path with float input retained and FP16 output.",
        },
        {
            "name": "compact_xw_input_fp16_output",
            "bytes_per_element": 4.0,
            "total_bytes": elements * 4,
            "note": "Upper-bound compact consumer: compact x/w input plus FP16 output.",
        },
        {
            "name": "compact_half_or_u16_xw_input_fp16_output",
            "bytes_per_element": 3.0,
            "total_bytes": elements * 3,
            "note": "Quantized compact consumer: two 16-bit x/w inputs per four outputs plus FP16 output.",
        },
        {
            "name": "compact_u8_xw_input_fp16_output",
            "bytes_per_element": 2.5,
            "total_bytes": int(elements * 2.5),
            "note": "Aggressively quantized compact consumer: two 8-bit x/w inputs per four outputs plus FP16 output.",
        },
        {
            "name": "compact_u8_xw_input_u8_xw_output",
            "bytes_per_element": 1.0,
            "total_bytes": elements,
            "note": "Fully benchmark-specialized ABI: U8 x/w input plus U8 x/w output with y/z as implicit constants.",
        },
        {
            "name": "compact_u8_xw_output_score",
            "bytes_per_element": 1.5,
            "total_bytes": int(elements * 1.5),
            "note": "Compact U8 x/w output consumed directly into one float score per four elements.",
        },
        {
            "name": "compact_u4_xw_input_fp16_output_expected_fail",
            "bytes_per_element": 2.25,
            "total_bytes": int(elements * 2.25),
            "note": "Boundary-only compact consumer: two 4-bit x/w inputs per four outputs plus FP16 output; expected to miss tolerance.",
        },
    ]


def scaling_plan(nvidia_smi, dimx, dimy, topology=None):
    devices = nvidia_smi.get("devices", []) if nvidia_smi.get("status") == "ok" else []
    device_count = len(devices)
    elements = dimx * dimy
    memory_models = build_memory_models(dimx, dimy)
    harness_model = memory_models[0]

    result = {
        "dimx": dimx,
        "dimy": dimy,
        "elements": elements,
        "headroom_fraction": HEADROOM_FRACTION,
        "memory_models": memory_models,
        "partitions": [],
    }

    if device_count == 0:
        result.update(
            {
                "status": "unknown",
                "recommendation": "No GPU memory data was available; rerun on the target host before making scale decisions.",
            }
        )
        return result

    row_ranges = split_rows(dimy, device_count)
    for device, row_range in zip(devices, row_ranges):
        shard_elements = dimx * row_range["rows"]
        memory_total_mib = device.get("memory_total_mib")
        usable_bytes = (
            int(memory_total_mib * 1024 * 1024 * HEADROOM_FRACTION)
            if memory_total_mib
            else None
        )
        harness_bytes = int(shard_elements * harness_model["bytes_per_element"])
        result["partitions"].append(
            {
                "gpu_index": len(result["partitions"]),
                "gpu_name": device.get("name"),
                "row_start": row_range["row_start"],
                "row_end": row_range["row_end"],
                "rows": row_range["rows"],
                "harness_bytes": harness_bytes,
                "harness_mib": mib(harness_bytes),
                "usable_memory_mib": mib(usable_bytes) if usable_bytes else None,
                "fits_harness_with_headroom": (
                    harness_bytes <= usable_bytes if usable_bytes else None
                ),
            }
        )

    single_harness_bytes = harness_model["total_bytes"]
    single_device = devices[0]
    single_usable = (
        int(single_device["memory_total_mib"] * 1024 * 1024 * HEADROOM_FRACTION)
        if single_device.get("memory_total_mib")
        else None
    )
    fits_single = single_harness_bytes <= single_usable if single_usable else None
    fits_partitioned = all(
        part["fits_harness_with_headroom"] is not False
        for part in result["partitions"]
    )

    no_nvlink = (
        device_count > 1
        and topology
        and topology.get("status") == "ok"
        and not topology.get("has_nvlink")
    )
    interconnect_note = (
        " The detected GPU-to-GPU path does not include NVLink, so multi-GPU "
        "runs are most meaningful when data is already sharded by GPU, when "
        "the working set requires capacity, or when throughput matters more "
        "than a gather-heavy single-result benchmark."
        if no_nvlink
        else ""
    )

    if device_count == 1:
        status = "single-gpu"
        recommendation = (
            "This host has one GPU, so no multi-GPU speedup can be measured here. "
            "Keep row partitioning as a capacity/throughput option for a future multi-GPU host."
        )
    elif not fits_single and fits_partitioned:
        status = "capacity-scaling-candidate"
        recommendation = (
            "A single GPU may not fit the current harness with headroom, while row shards do; "
            "multi-GPU partitioning is justified for capacity."
        )
    else:
        status = "throughput-scaling-only"
        recommendation = (
            "The problem fits one GPU with headroom; multi-GPU work should be gated on "
            "throughput goals and measured transfer/reduction overhead."
            + interconnect_note
        )

    result.update(
        {
            "status": status,
            "single_gpu_harness_mib": mib(single_harness_bytes),
            "fits_single_gpu_harness_with_headroom": fits_single,
            "fits_partitioned_harness_with_headroom": fits_partitioned,
            "recommendation": recommendation,
            "interconnect_note": interconnect_note.strip() or None,
        }
    )
    return result


def classify(summary, space):
    ncu = space.get("ncu", {}).get("metrics", {})
    dram_pct = ncu.get("dram_throughput_pct")
    sm_pct = ncu.get("sm_throughput_pct")
    occupancy_pct = ncu.get("achieved_occupancy_pct")
    variants = summary.get("variants", {})

    default = variants.get("vector4_coalesced_fast", {})
    poly = (
        variants.get("vector4_affine_loaded_fixed_range_experimental")
        or variants.get("vector4_poly5_unchecked_fixed_range_experimental", {})
    )
    half = variants.get("vector4_affine_half_output_sparse_experimental", {})
    compact = variants.get("compact_xw_affine_half_output_experimental", {})
    compact_u16 = variants.get("compact_u16_xw_affine_half_output_experimental", {})
    compact_u8 = variants.get("compact_u8_xw_affine_half_output_experimental", {})
    compact_u8_output = variants.get(
        "compact_u8_xw_affine_u8_xw_output_experimental", {}
    )
    float_consumer = variants.get("consume_float_output_experimental", {})
    compact_consumer = variants.get("consume_u8_xw_output_experimental", {})
    float_consumer_pipeline = variants.get(
        "vector4_affine_loaded_float_consumer_pipeline_experimental", {}
    )
    compact_consumer_pipeline = variants.get(
        "compact_u8_xw_u8_output_consumer_with_gpu_pack_pipeline_experimental",
        {},
    )

    notes = []
    bottleneck = "unknown"
    if dram_pct is not None and sm_pct is not None:
        if dram_pct >= 80 and sm_pct <= 60:
            bottleneck = "memory-throughput"
            notes.append(
                "Nsight shows high DRAM pressure with modest SM utilization; "
                "prioritize byte/sector reductions over extra arithmetic work."
            )
        elif sm_pct >= 80 and dram_pct <= 70:
            bottleneck = "compute-throughput"
            notes.append(
                "SM throughput is higher than DRAM pressure; math reduction, "
                "ILP, or occupancy tuning may matter on this device."
            )
        elif sm_pct >= 70 and dram_pct >= 70:
            bottleneck = "balanced"
            notes.append(
                "Both SM and DRAM pressure are high; tune one axis at a time "
                "and expect smaller gains."
            )
        else:
            bottleneck = "latency-or-launch-sensitive"
            notes.append(
                "Neither SM nor DRAM throughput is saturated in the profiled "
                "kernel; inspect launch overhead, eligible warps, and problem size."
            )

    poly_speedup = speedup(default, poly)
    half_speedup = speedup(default, half)
    compact_speedup = speedup(half or default, compact)
    compact_u16_speedup = speedup(compact, compact_u16)
    compact_u8_speedup = speedup(compact_u16, compact_u8)
    compact_u8_output_speedup = speedup(compact_u8, compact_u8_output)
    compact_consumer_speedup = speedup(float_consumer, compact_consumer)
    compact_consumer_pipeline_speedup = speedup(
        float_consumer_pipeline, compact_consumer_pipeline
    )

    if poly_speedup and poly_speedup < 1.05:
        notes.append(
            "Polynomial/affine math removal does not materially beat the "
            "default float-output vector kernel on this run."
        )
    if half_speedup and half_speedup >= 1.05:
        notes.append(
            "FP16 output improves runtime, so output traffic is a useful "
            "ABI-changing optimization axis on this device."
        )
    if compact_speedup and compact_speedup >= 1.10:
        notes.append(
            "Compact x/w input gives a large kernel-side win; verify producer "
            "layout or packing amortization before treating it as end-to-end."
        )
    if compact_u16_speedup and compact_u16_speedup >= 1.05:
        notes.append(
            "U16 fixed-point x/w input improves over FP32 compact x/w on this "
            "memory-bound run while preserving the benchmark tolerance."
        )
    if compact_u8_speedup and compact_u8_speedup >= 1.03:
        notes.append(
            "U8 fixed-point x/w input improves over U16 compact x/w on this "
            "memory-bound run while staying inside tolerance."
        )
    if compact_u8_output_speedup and compact_u8_output_speedup >= 1.05:
        notes.append(
            "Custom U8 x/w output improves over FP16 output by attacking the "
            "remaining write traffic; this is a strongly ABI-changing path."
        )
    if compact_consumer_speedup and compact_consumer_speedup >= 1.10:
        notes.append(
            "A downstream consumer that stays in compact U8 x/w form is much "
            "cheaper than consuming expanded float4 output."
        )
    if compact_consumer_pipeline_speedup and compact_consumer_pipeline_speedup >= 1.10:
        notes.append(
            "Even when U8 input packing is paid every launch, the compact-output "
            "pipeline wins if downstream consumes compact form directly."
        )
    if occupancy_pct is not None and occupancy_pct < 50:
        notes.append(
            "Achieved occupancy is low; launch geometry and register pressure "
            "should be re-swept on this hardware."
        )

    return {
        "bottleneck_hint": bottleneck,
        "ncu": {
            "dram_throughput_pct": dram_pct,
            "sm_throughput_pct": sm_pct,
            "achieved_occupancy_pct": occupancy_pct,
        },
        "variant_speedups": {
            "default_vs_polynomial_or_affine": poly_speedup,
            "default_vs_fp16_output": half_speedup,
            "fp16_output_vs_compact_xw": compact_speedup,
            "compact_xw_vs_compact_u16_xw": compact_u16_speedup,
            "compact_u16_xw_vs_compact_u8_xw": compact_u8_speedup,
            "compact_u8_xw_half_output_vs_u8_xw_output": compact_u8_output_speedup,
            "float_consumer_vs_compact_u8_consumer": compact_consumer_speedup,
            "float_pipeline_vs_compact_u8_consumer_pipeline": compact_consumer_pipeline_speedup,
        },
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/latest")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = read_json(output_dir / "summary.json", {})
    space = read_json(output_dir / "space.json", {})

    result = {
        "nvidia_smi": query_nvidia_smi(),
        "topology": query_topology(),
        "nvcc": query_nvcc(),
    }
    dimx, dimy = read_problem_size(summary)
    result["classification"] = classify(summary, space)
    result["scaling"] = scaling_plan(result["nvidia_smi"], dimx, dimy, result["topology"])
    (output_dir / "hardware.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {output_dir / 'hardware.json'}")


if __name__ == "__main__":
    main()
