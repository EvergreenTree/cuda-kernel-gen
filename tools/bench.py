#!/usr/bin/env python3
"""Run the CUDA benchmark and save timing summaries."""

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd, *, cwd=ROOT):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout, end="")
        raise SystemExit(proc.returncode)
    return proc.stdout


def parse_benchmark_output(text):
    gpu = {}
    rows = []
    gpu_match = re.search(
        r"GPU: (?P<name>.*), compute capability (?P<cc>\d+\.\d+), SMs (?P<sms>\d+)",
        text,
    )
    if gpu_match:
        gpu = {
            "name": gpu_match.group("name"),
            "compute_capability": gpu_match.group("cc"),
            "sms": int(gpu_match.group("sms")),
        }

    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or parts[0] in ("variant", "CUDA"):
            continue
        try:
            time_ms = float(parts[2])
        except ValueError:
            continue
        row = {"variant": parts[0], "correct": parts[1], "time_ms": time_ms}
        if len(parts) >= 4:
            try:
                row["logical_bytes"] = int(parts[3])
            except ValueError:
                pass
        rows.append(row)

    if not rows:
        raise ValueError("benchmark output did not contain variant timing rows")
    return gpu, rows


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, dimx, dimy, nreps, gpu):
    variants = sorted({row["variant"] for row in rows})
    default_logical_bytes = dimx * dimy * 2 * 4
    by_variant = {}
    baseline = None

    for variant in variants:
        times = [row["time_ms"] for row in rows if row["variant"] == variant]
        correct = all(
            row["correct"] == "yes" for row in rows if row["variant"] == variant
        )
        median_ms = statistics.median(times)
        logical_bytes = max(
            row.get("logical_bytes", default_logical_bytes)
            for row in rows
            if row["variant"] == variant
        )
        if variant == "original_row_stride":
            baseline = median_ms
        by_variant[variant] = {
            "correct": correct,
            "median_ms": median_ms,
            "min_ms": min(times),
            "max_ms": max(times),
            "samples": len(times),
            "logical_bytes_per_launch": logical_bytes,
            "effective_bandwidth_gbps": logical_bytes
            / (median_ms * 1.0e-3)
            / 1.0e9,
        }

    if baseline:
        for stats in by_variant.values():
            stats["speedup_vs_original_row_stride"] = baseline / stats["median_ms"]

    return {
        "gpu": gpu,
        "dimx": dimx,
        "dimy": dimy,
        "nreps": nreps,
        "logical_bytes_per_launch": default_logical_bytes,
        "variants": by_variant,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./optimized.x")
    parser.add_argument("--output-dir", default="reports/latest")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--nreps", type=int, default=100)
    parser.add_argument("--dimx", type=int, default=8192)
    parser.add_argument("--dimy", type=int, default=8192)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--arch-flags", default="-arch=native")
    parser.add_argument("--tune-flags", default="")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.build:
        tune_flags = " ".join(
            flag
            for flag in (
                args.tune_flags,
                f"-DNREPS={args.nreps}",
                f"-DDIMX={args.dimx}",
                f"-DDIMY={args.dimy}",
            )
            if flag
        )
        run(["make", "-B", "optimized.x", f"ARCH_FLAGS={args.arch_flags}", f"TUNE_FLAGS={tune_flags}"])

    samples = []
    gpu = {}
    for run_id in range(args.runs):
        text = run([args.binary])
        (gpu, rows) = parse_benchmark_output(text)
        original = next(
            (row["time_ms"] for row in rows if row["variant"] == "original_row_stride"),
            None,
        )
        for row in rows:
            logical_bytes = row.get("logical_bytes", args.dimx * args.dimy * 2 * 4)
            enriched = {
                "run": run_id,
                "variant": row["variant"],
                "correct": row["correct"],
                "time_ms": f"{row['time_ms']:.6f}",
                "dimx": args.dimx,
                "dimy": args.dimy,
                "nreps": args.nreps,
                "logical_bytes": logical_bytes,
            }
            if original:
                enriched["speedup_vs_original_row_stride"] = f"{original / row['time_ms']:.6f}"
            enriched["effective_bandwidth_gbps"] = (
                f"{logical_bytes / (row['time_ms'] * 1.0e-3) / 1.0e9:.3f}"
            )
            samples.append(enriched)

    numeric_rows = [
        {
            "variant": row["variant"],
            "correct": row["correct"],
            "time_ms": float(row["time_ms"]),
            "logical_bytes": int(row["logical_bytes"]),
        }
        for row in samples
    ]
    summary = summarize(numeric_rows, args.dimx, args.dimy, args.nreps, gpu)

    write_csv(output_dir / "timings.csv", samples)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {output_dir / 'timings.csv'}")
    print(f"Wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
