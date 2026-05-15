#!/usr/bin/env python3
"""Run a focused launch-geometry/register-cap sweep for the CUDA benchmark."""

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIGS = (
    {"threads": 256, "blocks_per_sm": 32, "items": 1, "maxrregcount": None},
    {"threads": 512, "blocks_per_sm": 16, "items": 1, "maxrregcount": None},
    {"threads": 512, "blocks_per_sm": 32, "items": 1, "maxrregcount": None},
    {"threads": 512, "blocks_per_sm": 48, "items": 1, "maxrregcount": None},
    {"threads": 1024, "blocks_per_sm": 16, "items": 1, "maxrregcount": None},
    {"threads": 512, "blocks_per_sm": 32, "items": 2, "maxrregcount": None},
    {"threads": 512, "blocks_per_sm": 32, "items": 4, "maxrregcount": None},
    {"threads": 512, "blocks_per_sm": 32, "items": 1, "maxrregcount": 24},
    {"threads": 512, "blocks_per_sm": 32, "items": 1, "maxrregcount": 32},
)

TRACKED_VARIANTS = (
    "vector4_coalesced_fast",
    "vector4_ilp_fast",
    "scalar_coalesced_fast",
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
    if proc.returncode != 0:
        print(proc.stdout, end="")
        raise SystemExit(proc.returncode)
    return proc.stdout


def tune_flags(config):
    flags = [
        f"-DTHREADS_PER_BLOCK={config['threads']}",
        f"-DBLOCKS_PER_SM={config['blocks_per_sm']}",
        f"-DITEMS_PER_THREAD={config['items']}",
    ]
    if config["maxrregcount"]:
        flags.append(f"-maxrregcount={config['maxrregcount']}")
    return " ".join(flags)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config_id",
        "variant",
        "correct",
        "median_ms",
        "speedup_vs_best_default",
        "threads_per_block",
        "blocks_per_sm",
        "items_per_thread",
        "maxrregcount",
        "effective_bandwidth_gbps",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/latest")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--nreps", type=int, default=100)
    parser.add_argument("--dimx", type=int, default=8192)
    parser.add_argument("--dimy", type=int, default=8192)
    parser.add_argument("--arch-flags", default="-arch=native")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    sweep_dir = output_dir / "launch_sweep_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    config_summaries = []
    for config_id, config in enumerate(DEFAULT_CONFIGS):
        run_dir = sweep_dir / f"config_{config_id:02d}"
        flags = tune_flags(config)
        run(
            [
                "python3",
                "tools/bench.py",
                "--build",
                "--runs",
                str(args.runs),
                "--nreps",
                str(args.nreps),
                "--dimx",
                str(args.dimx),
                "--dimy",
                str(args.dimy),
                f"--arch-flags={args.arch_flags}",
                f"--tune-flags={flags}",
                "--output-dir",
                str(run_dir.relative_to(ROOT)),
            ]
        )
        summary = json.loads((run_dir / "summary.json").read_text())
        config_summaries.append(
            {
                "config_id": config_id,
                "config": config,
                "summary_path": str((run_dir / "summary.json").relative_to(ROOT)),
            }
        )
        for variant in TRACKED_VARIANTS:
            stats = summary.get("variants", {}).get(variant)
            if not stats:
                continue
            rows.append(
                {
                    "config_id": config_id,
                    "variant": variant,
                    "correct": bool(stats.get("correct")),
                    "median_ms": stats.get("median_ms"),
                    "speedup_vs_best_default": None,
                    "threads_per_block": config["threads"],
                    "blocks_per_sm": config["blocks_per_sm"],
                    "items_per_thread": config["items"],
                    "maxrregcount": config["maxrregcount"] or "",
                    "effective_bandwidth_gbps": stats.get("effective_bandwidth_gbps"),
                }
            )

    default_times = [
        row["median_ms"]
        for row in rows
        if row["variant"] == "vector4_coalesced_fast" and row["correct"]
    ]
    best_default = min(default_times) if default_times else None
    if best_default:
        for row in rows:
            row["speedup_vs_best_default"] = best_default / row["median_ms"]

    best_row = min(
        (
            row
            for row in rows
            if row["variant"] == "vector4_coalesced_fast" and row["correct"]
        ),
        key=lambda row: row["median_ms"],
        default=None,
    )
    report = {
        "dimx": args.dimx,
        "dimy": args.dimy,
        "nreps": args.nreps,
        "runs": args.runs,
        "best_default": best_row,
        "configs": config_summaries,
        "rows": rows,
    }

    write_csv(output_dir / "launch_sweep.csv", rows)
    (output_dir / "launch_sweep.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output_dir / 'launch_sweep.csv'}")
    print(f"Wrote {output_dir / 'launch_sweep.json'}")
    if best_row:
        print(
            "Best default: config {config_id} {median_ms:.6f} ms".format(
                **best_row
            )
        )


if __name__ == "__main__":
    main()
