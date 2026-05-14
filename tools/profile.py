#!/usr/bin/env python3
"""Collect compact time/space profiler facts for the CUDA benchmark."""

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


KERNEL_VARIANTS = (
    ("kernel_original", "original_row_stride"),
    ("kernel_scalar_coalescedILi5E", "scalar_coalesced_fast"),
    ("kernel_vector4_fastILi5E", "vector4_coalesced_fast"),
    ("kernel_vector4_ilp_fast", "vector4_ilp_fast"),
    ("kernel_vector4_poly5_fixed", "vector4_poly5_fixed_range_experimental"),
    (
        "kernel_vector4_poly5_unchecked",
        "vector4_poly5_unchecked_fixed_range_experimental",
    ),
    ("kernel_vector4_fast_dynamic", "vector4_dynamic_fallback"),
    ("kernel_scalar_coalesced_dynamic", "scalar_dynamic_fallback"),
)


NCU_METRICS = {
    "duration_ns": "gpu__time_duration.avg",
    "dram_throughput_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "memory_throughput_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_throughput_pct": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "l2_throughput_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "active_warps_per_sm": "sm__warps_active.avg.per_cycle_active",
    "registers_per_thread": "launch__registers_per_thread",
    "block_size": "launch__block_size",
    "grid_size": "launch__grid_size",
    "waves_per_sm": "launch__waves_per_multiprocessor",
    "thread_count": "launch__thread_count",
    "sm_count": "launch__sm_count",
}


def run(cmd):
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode, proc.stdout


def variant_for_entry(entry):
    for token, variant in KERNEL_VARIANTS:
        if token in entry:
            return variant
    return entry


def parse_ptxas(text):
    variants = {}
    current = None
    for line in text.splitlines():
        compile_match = re.search(r"Compiling entry function '([^']+)'", line)
        if compile_match:
            current = variant_for_entry(compile_match.group(1))
            variants.setdefault(current, {})
            continue

        if current is None:
            continue

        spill_match = re.search(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
            line,
        )
        if spill_match:
            variants[current]["stack_frame_bytes"] = int(spill_match.group(1))
            variants[current]["spill_stores_bytes"] = int(spill_match.group(2))
            variants[current]["spill_loads_bytes"] = int(spill_match.group(3))
            continue

        reg_match = re.search(r"Used (\d+) registers", line)
        if reg_match:
            variants[current]["registers"] = int(reg_match.group(1))
    return variants


def parse_ncu_csv(text):
    csv_lines = [line for line in text.splitlines() if line.startswith('"')]
    if len(csv_lines) < 3:
        raise ValueError("Nsight Compute CSV did not contain a raw metric table")

    reader = list(csv.reader(csv_lines))
    header = reader[0]
    data = reader[2] if len(reader) > 2 else reader[1]
    row = dict(zip(header, data))
    metrics = {}
    for name, key in NCU_METRICS.items():
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            if "." in value:
                metrics[name] = float(value)
            else:
                metrics[name] = int(value)
        except ValueError:
            metrics[name] = value
    kernel = row.get("Kernel Name") or row.get("launch__kernel_name")
    if kernel:
        metrics["kernel"] = kernel
    return metrics


def file_size(path):
    full = ROOT / path
    return full.stat().st_size if full.exists() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/latest")
    parser.add_argument("--binary", default="./optimized.x")
    parser.add_argument("--dimx", type=int, default=8192)
    parser.add_argument("--dimy", type=int, default=8192)
    parser.add_argument("--kernel-regex", default="kernel_vector4_fast")
    parser.add_argument("--ncu", default="/opt/nvidia/nsight-compute/2025.3.0/ncu")
    parser.add_argument("--ncu-prefix", default=os.environ.get("NCU_PREFIX", ""))
    parser.add_argument("--skip-ncu", action="store_true")
    parser.add_argument("--skip-ptxas", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "dimx": args.dimx,
        "dimy": args.dimy,
        "logical_bytes_per_launch": args.dimx * args.dimy * 2 * 4,
        "binary_sizes": {
            "optimized_x_bytes": file_size("optimized.x"),
            "fatbin_x_bytes": file_size("fatbin.x"),
        },
        "ptxas": {"status": "skipped", "variants": {}},
        "ncu": {"status": "skipped", "metrics": {}},
    }

    if not args.skip_ptxas:
        tune_flags = f"-Xptxas=-v -DDIMX={args.dimx} -DDIMY={args.dimy}"
        code, text = run(["make", "-B", "optimized.x", f"TUNE_FLAGS={tune_flags}"])
        (output_dir / "ptxas.log").write_text(text)
        result["ptxas"] = {
            "status": "ok" if code == 0 else "failed",
            "variants": parse_ptxas(text),
            "log": "ptxas.log",
        }
        result["binary_sizes"]["optimized_x_bytes"] = file_size("optimized.x")

    if not args.skip_ncu:
        ncu_cmd = [
            args.ncu,
            "--set",
            "basic",
            "--page",
            "raw",
            "--csv",
            "--kernel-name",
            f"regex:{args.kernel_regex}",
            "--launch-count",
            "1",
            args.binary,
        ]
        cmd = shlex.split(args.ncu_prefix) + ncu_cmd
        code, text = run(cmd)
        (output_dir / "ncu_raw.csv").write_text(text)
        result["ncu"]["command"] = " ".join(shlex.quote(part) for part in cmd)
        if code == 0:
            try:
                result["ncu"] = {
                    "status": "ok",
                    "metrics": parse_ncu_csv(text),
                    "raw": "ncu_raw.csv",
                    "command": result["ncu"]["command"],
                }
            except ValueError as exc:
                result["ncu"]["status"] = "parse_failed"
                result["ncu"]["error"] = str(exc)
        else:
            result["ncu"]["status"] = "failed"
            result["ncu"]["error"] = "Nsight Compute failed; inspect ncu_raw.csv"
            if "ERR_NVGPUCTRPERM" in text and not args.ncu_prefix:
                result["ncu"]["rerun_hint"] = (
                    "make profile-space NCU_PREFIX=sudo or "
                    f"sudo {' '.join(shlex.quote(part) for part in ncu_cmd)}"
                )

    (output_dir / "space.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {output_dir / 'space.json'}")
    if result["ncu"].get("rerun_hint"):
        print(result["ncu"]["rerun_hint"])


if __name__ == "__main__":
    main()
