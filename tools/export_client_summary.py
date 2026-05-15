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

    ncu_status = space.get("ncu", {}).get("status", "missing")
    ncu_command = space.get("ncu", {}).get("command")
    rows = variant_rows(summary)
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
