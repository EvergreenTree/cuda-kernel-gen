#!/usr/bin/env python3
"""Run the compact U8 L2 residency experiment and save artifacts."""

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


def parse_output(text):
    rows = []
    config = {}
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or parts[0] in ("variant", "CUDA"):
            continue
        if parts[0] == "l2_config" and len(parts) == 3:
            try:
                config[parts[1]] = int(parts[2])
            except ValueError:
                config[parts[1]] = parts[2]
            continue
        if len(parts) != 5:
            continue
        try:
            rows.append(
                {
                    "variant": parts[0],
                    "correct": parts[1],
                    "time_ms": float(parts[2]),
                    "logical_bytes": int(parts[3]),
                    "persisting_l2": parts[4],
                }
            )
        except ValueError:
            continue
    if not rows:
        raise ValueError("L2 experiment output did not contain timing rows")
    return rows, config


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./l2.x")
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
                "l2.x",
                f"ARCH_FLAGS={args.arch_flags}",
                f"TUNE_FLAGS={tune_flags}",
            ]
        )

    samples = []
    config = {}
    raw_outputs = []
    for run_id in range(args.runs):
        text = run([args.binary])
        raw_outputs.append(text)
        rows, config = parse_output(text)
        for row in rows:
            enriched = dict(row)
            enriched["run"] = run_id
            samples.append(enriched)

    by_variant = {}
    for row in samples:
        by_variant.setdefault(row["variant"], []).append(row)

    variants = {}
    for variant, rows in by_variant.items():
        times = sorted(row["time_ms"] for row in rows)
        mid = len(times) // 2
        logical_bytes = max(row["logical_bytes"] for row in rows)
        median_ms = times[mid]
        variants[variant] = {
            "correct": all(row["correct"] == "yes" for row in rows),
            "median_ms": median_ms,
            "min_ms": min(times),
            "max_ms": max(times),
            "samples": len(rows),
            "logical_bytes_per_launch": logical_bytes,
            "effective_bandwidth_gbps": logical_bytes / (median_ms * 1.0e-3) / 1.0e9,
            "persisting_l2": any(row["persisting_l2"] == "yes" for row in rows),
        }

    warm = variants.get("consume_u8_after_producer_warm_l2_experimental", {})
    cold = variants.get("consume_u8_after_l2_thrash_experimental", {})
    persist = variants.get("consume_u8_after_persisting_l2_experimental", {})
    persist_thrash = variants.get(
        "consume_u8_after_persisting_l2_thrash_experimental", {}
    )
    producer_warm = variants.get("compact_u8_producer_warm_l2_experimental", {})
    producer_cold = variants.get("compact_u8_producer_after_l2_thrash_experimental", {})
    producer_persist = variants.get(
        "compact_u8_producer_persisting_input_experimental", {}
    )
    producer_persist_thrash = variants.get(
        "compact_u8_producer_persisting_input_after_l2_thrash_experimental", {}
    )
    total_uint4_persist = variants.get(
        "compact_u8_producer_uint4_consumer_persisting_l2_total_experimental", {}
    )
    fused_direct = variants.get("compact_u8_fused_score_direct_experimental", {})
    fused_persist = variants.get(
        "compact_u8_fused_score_direct_persisting_input_experimental", {}
    )
    if warm.get("median_ms") and cold.get("median_ms"):
        cold["speedup_of_warm_l2_vs_thrash"] = cold["median_ms"] / warm["median_ms"]
    if persist.get("median_ms") and warm.get("median_ms"):
        persist["speedup_vs_warm_l2"] = warm["median_ms"] / persist["median_ms"]
    if persist_thrash.get("median_ms") and cold.get("median_ms"):
        persist_thrash["speedup_vs_thrash_without_persisting"] = (
            cold["median_ms"] / persist_thrash["median_ms"]
        )
    if producer_warm.get("median_ms") and producer_cold.get("median_ms"):
        producer_cold["speedup_of_warm_l2_vs_thrash"] = (
            producer_cold["median_ms"] / producer_warm["median_ms"]
        )
    if producer_persist.get("median_ms") and producer_warm.get("median_ms"):
        producer_persist["speedup_vs_warm_l2"] = (
            producer_warm["median_ms"] / producer_persist["median_ms"]
        )
    if producer_persist_thrash.get("median_ms") and producer_cold.get("median_ms"):
        producer_persist_thrash["speedup_vs_thrash_without_persisting"] = (
            producer_cold["median_ms"] / producer_persist_thrash["median_ms"]
        )
    if fused_persist.get("median_ms") and fused_direct.get("median_ms"):
        fused_persist["speedup_vs_direct_without_persisting"] = (
            fused_direct["median_ms"] / fused_persist["median_ms"]
        )
    if fused_persist.get("median_ms") and total_uint4_persist.get("median_ms"):
        fused_persist["speedup_vs_best_adjacent_pipeline"] = (
            total_uint4_persist["median_ms"] / fused_persist["median_ms"]
        )

    report = {
        "dimx": args.dimx,
        "dimy": args.dimy,
        "nreps": args.nreps,
        "runs": args.runs,
        "config": config,
        "variants": variants,
        "raw_output": "\n--- run ---\n".join(raw_outputs),
    }

    write_csv(output_dir / "l2_cache.csv", samples)
    (output_dir / "l2_cache.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {output_dir / 'l2_cache.csv'}")
    print(f"Wrote {output_dir / 'l2_cache.json'}")
    if warm.get("median_ms") and cold.get("median_ms"):
        print(
            "Warm L2 consumer vs thrashed consumer: {:.2f}x".format(
                cold["median_ms"] / warm["median_ms"]
            )
        )
    if fused_persist.get("median_ms") and total_uint4_persist.get("median_ms"):
        print(
            "Fused score + L2 vs best adjacent pipeline: {:.2f}x".format(
                total_uint4_persist["median_ms"] / fused_persist["median_ms"]
            )
        )


if __name__ == "__main__":
    main()
