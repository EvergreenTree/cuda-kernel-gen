#!/usr/bin/env python3
"""Summarize hardware-sensitive benchmark behavior."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

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
        "nvcc": query_nvcc(),
        "classification": classify(summary, space),
    }
    (output_dir / "hardware.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {output_dir / 'hardware.json'}")


if __name__ == "__main__":
    main()
