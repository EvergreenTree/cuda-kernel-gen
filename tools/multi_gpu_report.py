#!/usr/bin/env python3
"""Run the row-sharded multi-GPU benchmark and save structured artifacts."""

import argparse
import csv
import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stdout, end="")
        raise SystemExit(proc.returncode)
    return proc.stdout


def parse_output(text):
    rows = []
    partitions = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or parts[0] in ("variant", "CUDA"):
            continue
        if parts[0] == "partition" and len(parts) == 5 and parts[1] != "gpu":
            partitions.append(
                {
                    "gpu": int(parts[1]),
                    "row_start": int(parts[2]),
                    "row_end": int(parts[3]),
                    "rows": int(parts[4]),
                }
            )
            continue
        if len(parts) != 6:
            continue
        try:
            rows.append(
                {
                    "variant": parts[0],
                    "gpu_count": int(parts[1]),
                    "correct": parts[2],
                    "kernel_or_total_ms": float(parts[3]),
                    "host_wall_ms": float(parts[4]),
                    "logical_bytes": int(parts[5]),
                }
            )
        except ValueError:
            continue
    if not rows:
        raise ValueError("multi-gpu benchmark output did not contain rows")
    return rows, partitions


def query_topology():
    if shutil.which("nvidia-smi") is None:
        return {"status": "missing"}
    proc = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        return {"status": "failed", "error": proc.stdout.strip()}

    clean = re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout)
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
            if src != dst:
                interconnects.append({"source": src, "target": dst, "path": value})

    return {
        "status": "ok",
        "paths": sorted({link["path"] for link in interconnects}),
        "has_nvlink": any(link["path"].startswith("NV") for link in interconnects),
        "interconnects": interconnects,
        "raw": clean,
    }


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./multi-gpu.x")
    parser.add_argument("--output-dir", default="reports/latest")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--nreps", type=int, default=100)
    parser.add_argument("--dimx", type=int, default=8192)
    parser.add_argument("--dimy", type=int, default=8192)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--arch-flags", default="-arch=native")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.build:
        tune_flags = " ".join(
            (
                f"-DNREPS={args.nreps}",
                f"-DDIMX={args.dimx}",
                f"-DDIMY={args.dimy}",
            )
        )
        run(
            [
                "make",
                "-B",
                "multi-gpu.x",
                f"ARCH_FLAGS={args.arch_flags}",
                f"TUNE_FLAGS={tune_flags}",
            ]
        )

    samples = []
    partitions = []
    raw_outputs = []
    for run_id in range(args.runs):
        text = run([args.binary])
        raw_outputs.append(text)
        rows, partitions = parse_output(text)
        for row in rows:
            enriched = dict(row)
            enriched["run"] = run_id
            samples.append(enriched)

    grouped = {}
    for row in samples:
        grouped.setdefault(row["variant"], []).append(row)

    summary_rows = {}
    for variant, variant_rows in grouped.items():
        times = sorted(row["kernel_or_total_ms"] for row in variant_rows)
        walls = sorted(row["host_wall_ms"] for row in variant_rows)
        summary_rows[variant] = {
            "gpu_count": max(row["gpu_count"] for row in variant_rows),
            "correct": all(row["correct"] == "yes" for row in variant_rows),
            "median_ms": times[len(times) // 2],
            "min_ms": min(times),
            "max_ms": max(times),
            "median_host_wall_ms": walls[len(walls) // 2],
            "logical_bytes_per_launch": max(row["logical_bytes"] for row in variant_rows),
            "samples": len(variant_rows),
        }

    single = summary_rows.get("single_gpu_kernel", {})
    single_ms = single.get("median_ms")
    if single_ms:
        for stats in summary_rows.values():
            stats["speedup_vs_single_gpu_kernel"] = single_ms / stats["median_ms"]

    report = {
        "dimx": args.dimx,
        "dimy": args.dimy,
        "nreps": args.nreps,
        "runs": args.runs,
        "partitions": partitions,
        "topology": query_topology(),
        "variants": summary_rows,
        "raw_output": "\n--- run ---\n".join(raw_outputs),
    }

    write_csv(output_dir / "multi_gpu.csv", samples)
    (output_dir / "multi_gpu.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output_dir / 'multi_gpu.csv'}")
    print(f"Wrote {output_dir / 'multi_gpu.json'}")


if __name__ == "__main__":
    main()
