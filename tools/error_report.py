#!/usr/bin/env python3
"""Capture precision/error distributions for ABI-changing CUDA variants."""

import argparse
import csv
import json
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


def parse_error_stats(text):
    rows = []
    for line in text.splitlines():
        if not line.startswith("error_stats "):
            continue
        row = {}
        for token in line.split()[1:]:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            if key == "variant":
                row[key] = value
            elif key in ("elements", "misses", "le_1e_5", "le_1e_4", "le_1e_3", "gt_1e_3"):
                row[key] = int(value)
            else:
                row[key] = float(value)
        if row:
            rows.append(row)
    if not rows:
        raise ValueError("benchmark output did not contain error_stats lines")
    return rows


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "variant",
        "elements",
        "misses",
        "miss_rate",
        "max_rel",
        "mean_rel",
        "rms_rel",
        "max_abs",
        "mean_abs",
        "le_1e_5",
        "le_1e_4",
        "le_1e_3",
        "gt_1e_3",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./error.x")
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
                "error.x",
                f"ARCH_FLAGS={args.arch_flags}",
                f"TUNE_FLAGS={tune_flags}",
            ]
        )

    samples = []
    raw_outputs = []
    for run_id in range(args.runs):
        text = run([args.binary])
        raw_outputs.append(text)
        for row in parse_error_stats(text):
            enriched = {"run": run_id}
            enriched.update(row)
            samples.append(enriched)

    by_variant = {}
    for row in samples:
        by_variant.setdefault(row["variant"], []).append(row)

    summary = {}
    for variant, rows in by_variant.items():
        miss_rates = sorted(row["miss_rate"] for row in rows)
        max_rels = sorted(row["max_rel"] for row in rows)
        rms_rels = sorted(row["rms_rel"] for row in rows)
        mid = len(rows) // 2
        summary[variant] = {
            "samples": len(rows),
            "elements": rows[0]["elements"],
            "median_miss_rate": miss_rates[mid],
            "median_max_rel": max_rels[mid],
            "median_rms_rel": rms_rels[mid],
            "max_misses": max(row["misses"] for row in rows),
            "histogram": {
                "le_1e_5": max(row["le_1e_5"] for row in rows),
                "le_1e_4": max(row["le_1e_4"] for row in rows),
                "le_1e_3": max(row["le_1e_3"] for row in rows),
                "gt_1e_3": max(row["gt_1e_3"] for row in rows),
            },
        }

    report = {
        "dimx": args.dimx,
        "dimy": args.dimy,
        "nreps": args.nreps,
        "runs": args.runs,
        "variants": summary,
        "raw_output": "\n--- run ---\n".join(raw_outputs),
    }

    write_csv(output_dir / "error_stats.csv", samples)
    (output_dir / "error_stats.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output_dir / 'error_stats.csv'}")
    print(f"Wrote {output_dir / 'error_stats.json'}")


if __name__ == "__main__":
    main()
