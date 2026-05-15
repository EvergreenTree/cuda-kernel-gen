#!/usr/bin/env python3
"""Summarize compact-layout pipeline tradeoffs from benchmark timings."""

import argparse
import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PIPELINE_VARIANTS = (
    ("Drop-in float default", "vector4_coalesced_fast"),
    ("FP16 output", "vector4_affine_half_output_sparse_experimental"),
    ("Compact U8 input + FP16 output", "compact_u8_xw_affine_half_output_experimental"),
    ("Pack U8 input setup", "pack_u8_xw_setup_experimental"),
    (
        "Pack U8 input + FP16 output",
        "compact_u8_xw_with_gpu_pack_pipeline_experimental",
    ),
    (
        "Pack U8 input + compact U8 output",
        "compact_u8_xw_u8_output_with_gpu_pack_pipeline_experimental",
    ),
    (
        "Pack U8 input + compact U8 output + float decode",
        "compact_u8_xw_u8_output_decode_with_gpu_pack_pipeline_experimental",
    ),
    (
        "Float output + score consumer",
        "vector4_affine_loaded_float_consumer_pipeline_experimental",
    ),
    (
        "Compact U8 output + score consumer",
        "compact_u8_xw_u8_output_consumer_with_gpu_pack_pipeline_experimental",
    ),
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


def read_json(path):
    return json.loads(path.read_text())


def fmt_ms(value):
    return "n/a" if value is None else f"{value:.4f} ms"


def fmt_speedup(value):
    return "n/a" if value is None else f"{value:.2f}x"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "variant",
        "correct",
        "median_ms",
        "speedup_vs_float_default",
        "logical_bytes_per_launch",
        "effective_bandwidth_gbps",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, verdict):
    lines = [
        "# Compact Pipeline Summary",
        "",
        verdict,
        "",
        "| Path | Correct | Median time | Speedup vs float default | Data moved |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        mib = row["logical_bytes_per_launch"] / (1024 * 1024)
        lines.append(
            "| {label} | {correct} | {time} | {speedup} | {mib:.1f} MiB |".format(
                label=row["label"],
                correct="yes" if row["correct"] else "no",
                time=fmt_ms(row["median_ms"]),
                speedup=fmt_speedup(row["speedup_vs_float_default"]),
                mib=mib,
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
                "--tune-flags=-DENABLE_LAYOUT_SETUP_EXPERIMENT=1",
                "--output-dir",
                str(output_dir.relative_to(ROOT)),
            ]
        )

    summary = read_json(output_dir / "summary.json")
    variants = summary.get("variants", {})
    default = variants.get("vector4_coalesced_fast", {})
    default_ms = default.get("median_ms")
    rows = []

    for label, name in PIPELINE_VARIANTS:
        stats = variants.get(name)
        if not stats:
            continue
        median_ms = stats.get("median_ms")
        rows.append(
            {
                "label": label,
                "variant": name,
                "correct": bool(stats.get("correct")),
                "median_ms": median_ms,
                "speedup_vs_float_default": (
                    default_ms / median_ms if default_ms and median_ms else None
                ),
                "logical_bytes_per_launch": stats.get("logical_bytes_per_launch", 0),
                "effective_bandwidth_gbps": stats.get("effective_bandwidth_gbps"),
            }
        )

    compact_consumer = variants.get(
        "compact_u8_xw_u8_output_consumer_with_gpu_pack_pipeline_experimental", {}
    )
    float_consumer = variants.get(
        "vector4_affine_loaded_float_consumer_pipeline_experimental", {}
    )
    if compact_consumer and float_consumer:
        compact_ms = compact_consumer.get("median_ms")
        float_ms = float_consumer.get("median_ms")
        if compact_ms and float_ms and compact_ms < float_ms:
            verdict = (
                "Compact U8 output remains useful when the downstream consumer "
                "stays compact: it beats the float-output consumer path on this "
                "run."
            )
        else:
            verdict = (
                "This run captured the compact-vs-float consumer comparison, but "
                "compact output did not beat the float-output path at this size. "
                "Use full-size runs before making the ABI call."
            )
    else:
        verdict = (
            "Compact pipeline rows were incomplete; rerun with layout setup "
            "enabled before making an ABI recommendation."
        )

    report = {
        "dimx": args.dimx,
        "dimy": args.dimy,
        "nreps": args.nreps,
        "runs": args.runs,
        "verdict": verdict,
        "rows": rows,
    }
    write_csv(output_dir / "pipeline_summary.csv", rows)
    (output_dir / "pipeline_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(output_dir / "pipeline_summary.md", rows, verdict)
    print(f"Wrote {output_dir / 'pipeline_summary.csv'}")
    print(f"Wrote {output_dir / 'pipeline_summary.json'}")
    print(f"Wrote {output_dir / 'pipeline_summary.md'}")


if __name__ == "__main__":
    main()
