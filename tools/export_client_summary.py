#!/usr/bin/env python3
"""Export a concise Markdown summary for a generated performance report."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


KEY_VARIANTS = (
    ("Original row-stride kernel", "original_row_stride"),
    ("Vectorized float default", "vector4_coalesced_fast"),
    ("Affine float default", "vector4_affine_loaded_fixed_range_experimental"),
    ("FP16 output", "vector4_affine_half_output_sparse_experimental"),
    (
        "Compact U8 input + FP16 output",
        "compact_u8_xw_affine_half_output_experimental",
    ),
    (
        "Compact U8 input + U8 output",
        "compact_u8_xw_affine_u8_xw_output_experimental",
    ),
    (
        "L2-persisting U8 in/out kernel",
        "compact_u8_producer_persisting_input_experimental",
    ),
    (
        "uint4-packed U8 in/out kernel",
        "compact_u8_producer_uint4_experimental",
    ),
    (
        "uint4 + L2 U8 in/out kernel",
        "compact_u8_producer_uint4_persisting_input_experimental",
    ),
    (
        "L2-resident U8 in/out pipeline",
        "compact_u8_producer_consumer_persisting_l2_total_experimental",
    ),
    (
        "uint4 + L2 compact pipeline",
        "compact_u8_producer_uint4_consumer_persisting_l2_total_experimental",
    ),
    (
        "Fused compact U8 input to score",
        "compact_u8_fused_score_direct_experimental",
    ),
    (
        "Fused compact U8 input to score + L2",
        "compact_u8_fused_score_direct_persisting_input_experimental",
    ),
    ("Decode U8 output to float", "decode_u8_xw_output_to_float_experimental"),
    (
        "Float output + score pipeline",
        "vector4_affine_loaded_float_consumer_pipeline_experimental",
    ),
    (
        "Compact U8 output + score pipeline",
        "compact_u8_xw_u8_output_consumer_with_gpu_pack_pipeline_experimental",
    ),
)

CACHE_PLANNING_ROWS = (
    ("H100 / H200", "~35 MiB", "8 GPUs", "15+ GPUs"),
    ("RTX 6000 Ada / RTX 5090", "~67 MiB", "4 GPUs", "8 GPUs"),
    ("RTX Pro 6000 Blackwell", "~90 MiB", "3 GPUs", "6 GPUs"),
    ("B200", "~180 MiB logical", "2 GPUs", "3 GPUs"),
    ("B300 SXM6 AC (measured)", "~89 MiB", "3 GPUs", "6 GPUs"),
)


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def fmt(value, digits=3, suffix=""):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}{suffix}"
    return str(value)


def fmt_mib(value):
    if value is None:
        return "n/a"
    return f"{value / (1024 * 1024):.1f} MiB"


def table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def variant_rows(summary):
    variants = summary.get("variants", {})
    rows = []
    for label, name in KEY_VARIANTS:
        stats = variants.get(name)
        if not stats:
            continue
        rows.append(
            [
                label,
                "yes" if stats.get("correct") else "no",
                f"{stats.get('median_ms', 0.0):.4f} ms",
                fmt(stats.get("speedup_vs_original_row_stride"), 2, "x"),
                fmt_mib(stats.get("logical_bytes_per_launch")),
            ]
        )
    return rows


def l2_variant_rows(l2_cache, baseline):
    variants = l2_cache.get("variants", {})
    strict_ms = baseline.get("time_ms")
    rows = []
    for label, name in KEY_VARIANTS:
        stats = variants.get(name)
        if not stats:
            continue
        strict_gain = (
            strict_ms / stats.get("median_ms")
            if strict_ms and stats.get("median_ms")
            else None
        )
        rows.append(
            [
                label,
                "yes" if stats.get("correct") else "no",
                f"{stats.get('median_ms', 0.0):.4f} ms",
                fmt(strict_gain, 1, "x"),
                fmt_mib(stats.get("logical_bytes_per_launch")),
            ]
        )
    return rows


def l2_summary_rows(l2_cache, baseline):
    variants = l2_cache.get("variants", {})
    total = variants.get("compact_u8_producer_consumer_persisting_l2_total_experimental")
    total_uint4 = variants.get(
        "compact_u8_producer_uint4_consumer_persisting_l2_total_experimental", {}
    )
    producer = variants.get("compact_u8_producer_persisting_input_experimental", {})
    producer_uint4 = variants.get(
        "compact_u8_producer_uint4_persisting_input_experimental", {}
    )
    producer_uint4_warm = variants.get("compact_u8_producer_uint4_experimental", {})
    producer_warm = variants.get("compact_u8_producer_warm_l2_experimental", {})
    producer_thrashed = variants.get("compact_u8_producer_after_l2_thrash_experimental", {})
    warm = variants.get("consume_u8_after_producer_warm_l2_experimental", {})
    thrashed = variants.get("consume_u8_after_l2_thrash_experimental", {})
    no_persist_total = variants.get("compact_u8_producer_consumer_total_experimental", {})
    fused = variants.get(
        "compact_u8_fused_score_direct_persisting_input_experimental", {}
    ) or variants.get("compact_u8_fused_score_direct_experimental", {})
    if not total and not producer and not producer_uint4:
        return []

    strict_ms = baseline.get("time_ms")
    best_producer = producer_uint4 or producer
    best_total = total_uint4 or total or {}
    producer_gain = (
        producer_warm.get("median_ms") / producer.get("median_ms")
        if producer_warm.get("median_ms") and producer.get("median_ms")
        else None
    )
    producer_thrash_gain = (
        producer_thrashed.get("median_ms") / producer.get("median_ms")
        if producer_thrashed.get("median_ms") and producer.get("median_ms")
        else None
    )
    producer_strict_gain = (
        strict_ms / best_producer.get("median_ms")
        if strict_ms and best_producer.get("median_ms")
        else None
    )
    uint4_gain = (
        producer.get("median_ms") / producer_uint4.get("median_ms")
        if producer.get("median_ms") and producer_uint4.get("median_ms")
        else None
    )
    uint4_l2_gain = (
        producer_uint4_warm.get("median_ms") / producer_uint4.get("median_ms")
        if producer_uint4_warm.get("median_ms") and producer_uint4.get("median_ms")
        else None
    )
    warm_gain = (
        thrashed.get("median_ms") / warm.get("median_ms")
        if thrashed.get("median_ms") and warm.get("median_ms")
        else None
    )
    persist_gain = (
        no_persist_total.get("median_ms") / total.get("median_ms")
        if no_persist_total.get("median_ms") and total.get("median_ms")
        else None
    )
    strict_gain = (
        strict_ms / best_total.get("median_ms")
        if strict_ms and best_total.get("median_ms")
        else None
    )
    fused_gain = (
        best_total.get("median_ms") / fused.get("median_ms")
        if best_total.get("median_ms") and fused.get("median_ms")
        else None
    )
    fused_strict_gain = (
        strict_ms / fused.get("median_ms")
        if strict_ms and fused.get("median_ms")
        else None
    )
    return [
        ["Best U8 in/out kernel", fmt(best_producer.get("median_ms"), 4, " ms")],
        ["Kernel-only speedup vs strict baseline", fmt(producer_strict_gain, 1, "x")],
        ["Scalar input L2 lift vs warm producer", fmt(producer_gain, 2, "x")],
        [
            "Scalar input L2 lift vs thrashed producer",
            fmt(producer_thrash_gain, 2, "x"),
        ],
        ["uint4 lift over scalar L2 producer", fmt(uint4_gain, 2, "x")],
        ["L2 lift on uint4 producer", fmt(uint4_l2_gain, 2, "x")],
        ["Best producer + consumer", fmt(best_total.get("median_ms"), 4, " ms")],
        ["Pipeline speedup vs strict baseline", fmt(strict_gain, 1, "x")],
        ["Warm consumer vs thrashed consumer", fmt(warm_gain, 1, "x")],
        ["Persisting total lift", fmt(persist_gain, 2, "x")],
        ["Fused score result", fmt(fused.get("median_ms"), 4, " ms")],
        ["Fused vs best adjacent", fmt(fused_gain, 2, "x")],
        ["Fused speedup vs strict baseline", fmt(fused_strict_gain, 1, "x")],
        ["Compact output footprint", fmt_mib(l2_cache.get("config", {}).get("compact_output_bytes"))],
        ["L2 budget on this host", fmt_mib(l2_cache.get("config", {}).get("persisting_l2_max_bytes"))],
    ]


def hardware_rows(summary, hardware):
    smi = hardware.get("nvidia_smi", {})
    devices = smi.get("devices", [])
    device = devices[0] if devices else {}
    nvcc = hardware.get("nvcc", {})
    classification = hardware.get("classification", {})
    scaling = hardware.get("scaling", {})
    return [
        ["GPU", device.get("name") or summary.get("gpu", {}).get("name", "n/a")],
        [
            "Compute capability",
            device.get("compute_capability")
            or summary.get("gpu", {}).get("compute_capability", "n/a"),
        ],
        ["Device count", str(smi.get("device_count", "n/a"))],
        ["Driver", str(device.get("driver_version", "n/a"))],
        ["NVCC", nvcc.get("version", "n/a")],
        [
            "Problem size",
            f"{summary.get('dimx', 'n/a')} x {summary.get('dimy', 'n/a')}",
        ],
        ["Bottleneck hint", classification.get("bottleneck_hint", "n/a")],
        ["Scaling status", scaling.get("status", "n/a")],
    ]


def baseline_rows(baseline, summary):
    default = summary.get("variants", {}).get("vector4_coalesced_fast", {})
    strict_ms = baseline.get("time_ms")
    default_ms = default.get("median_ms")
    if not strict_ms:
        return [["Strict original baseline", "not captured"]]
    speedup = strict_ms / default_ms if default_ms else None
    return [
        ["Strict original baseline", f"{strict_ms:.4f} ms"],
        ["Drop-in vectorized default", f"{default_ms:.4f} ms" if default_ms else "n/a"],
        [
            "Drop-in speedup vs strict baseline",
            f"{speedup:.2f}x" if speedup else "n/a",
        ],
    ]


def topology_rows(hardware):
    topology = hardware.get("topology", {})
    scaling = hardware.get("scaling", {})
    paths = ", ".join(topology.get("paths", [])) or "n/a"
    return [
        ["Topology status", topology.get("status", "n/a")],
        ["GPU-to-GPU paths", paths],
        ["NVLink detected", "yes" if topology.get("has_nvlink") else "no"],
        ["Multi-GPU readout", scaling.get("recommendation", "n/a")],
    ]


def recommendation_notes(hardware):
    notes = hardware.get("classification", {}).get("notes", [])
    scaling_note = hardware.get("scaling", {}).get("recommendation")
    result = list(notes)
    if scaling_note:
        result.append(scaling_note)
    return result or ["No hardware classification notes were captured."]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/latest")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    summary = read_json(output_dir / "summary.json", {})
    hardware = read_json(output_dir / "hardware.json", {})
    space = read_json(output_dir / "space.json", {})
    baseline = read_json(output_dir / "baseline.json", {})
    l2_cache = read_json(output_dir / "l2_cache.json", {})

    ncu_status = space.get("ncu", {}).get("status", "missing")
    ncu_command = space.get("ncu", {}).get("command")
    rows = variant_rows(summary) + l2_variant_rows(l2_cache, baseline)
    notes = recommendation_notes(hardware)

    sections = [
        "# CUDA Kernel Performance Summary",
        "",
        "## Hardware",
        "",
        table(["Item", "Value"], hardware_rows(summary, hardware)),
        "",
        "## Baseline Contract",
        "",
        table(["Item", "Value"], baseline_rows(baseline, summary)),
        "",
        "## Key Variants",
        "",
        table(
            [
                "Variant",
                "Correct",
                "Median time",
                "Speedup vs kernel baseline",
                "Data moved",
            ],
            rows,
        )
        if rows
        else "No timing rows were found. Run `make profile` first.",
        "",
        "## Extreme L2-Resident Option",
        "",
        (
            table(["Item", "Value"], l2_summary_rows(l2_cache, baseline))
            if l2_summary_rows(l2_cache, baseline)
            else "No L2 residency result was found. Run `make l2-report` for this report directory."
        ),
        "",
        "The kernel-only L2 row combines uint4-packed U8 input/output with "
        "persisting L2 on compact input. The adjacent pipeline row uses the "
        "uint4 producer plus persisting L2 on compact output for the downstream "
        "consumer. Both are custom-ABI options, remain memory-path limited "
        "rather than compute-bound, and should be remeasured on B200-class "
        "systems where HBM3e bandwidth narrows the cache advantage.",
        "",
        table(
            [
                "GPU family",
                "Usable cache estimate",
                "256 MiB working set",
                "512 MiB working set",
            ],
            [list(row) for row in CACHE_PLANNING_ROWS],
        ),
        "",
        "## Profiler Status",
        "",
        table(
            ["Artifact", "Status"],
            [
                [
                    "summary.json",
                    "present" if (output_dir / "summary.json").exists() else "missing",
                ],
                [
                    "space.json",
                    "present" if (output_dir / "space.json").exists() else "missing",
                ],
                [
                    "hardware.json",
                    "present" if (output_dir / "hardware.json").exists() else "missing",
                ],
                [
                    "index.html",
                    "present" if (output_dir / "index.html").exists() else "missing",
                ],
                ["Nsight Compute", ncu_status],
            ],
        ),
    ]
    if ncu_command:
        sections.extend(["", f"Nsight command: `{ncu_command}`"])

    sections.extend(
        [
            "",
            "## Multi-GPU",
            "",
            table(["Item", "Value"], topology_rows(hardware)),
            "",
            "## Recommendations",
            "",
            "\n".join(f"- {note}" for note in notes),
            "",
            "## Handoff",
            "",
            "- Use `index.html` for visual review.",
            "- Use `summary.json`, `space.json`, and `hardware.json` for exact "
            "machine-readable facts.",
            "- Add any new target-machine conclusion to the README scoreboard or "
            "Experiment Ledger before applying it elsewhere.",
            "",
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "client_summary.md"
    path.write_text("\n".join(sections))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
