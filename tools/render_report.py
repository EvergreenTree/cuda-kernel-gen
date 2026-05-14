#!/usr/bin/env python3
"""Render a compact HTML profiling report with inline SVG charts."""

import argparse
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def fmt(value, digits=2):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return html.escape(value)
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def bar_chart(title, items, unit="", width=760, row_height=30, digits=2):
    items = [(label, value) for label, value in items if value is not None]
    if not items:
        return f"<section><h2>{html.escape(title)}</h2><p>No data.</p></section>"
    max_value = max(value for _, value in items) or 1.0
    height = 44 + row_height * len(items)
    rows = []
    for idx, (label, value) in enumerate(items):
        y = 32 + idx * row_height
        bar_w = max(1, int((width - 260) * value / max_value))
        rows.append(
            f'<text x="0" y="{y + 16}" class="label">{html.escape(label)}</text>'
            f'<rect x="220" y="{y}" width="{bar_w}" height="18" rx="3"></rect>'
            f'<text x="{230 + bar_w}" y="{y + 14}" class="value">{fmt(value, digits)}{html.escape(unit)}</text>'
        )
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">'
        + "".join(rows)
        + "</svg></section>"
    )


def render_table(headers, rows):
    if not rows:
        return "<p>No data.</p>"
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/latest")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    summary = read_json(output_dir / "summary.json", {})
    space = read_json(output_dir / "space.json", {})
    variants = summary.get("variants", {})
    logical_mib = summary.get("logical_bytes_per_launch", 0) / (1024 * 1024)

    time_items = [
        (variant, stats.get("median_ms")) for variant, stats in variants.items()
    ]
    speed_items = [
        (variant, stats.get("speedup_vs_original_row_stride"))
        for variant, stats in variants.items()
        if variant != "original_row_stride"
    ]
    bandwidth_items = [
        (variant, stats.get("effective_bandwidth_gbps"))
        for variant, stats in variants.items()
    ]

    ptxas = space.get("ptxas", {}).get("variants", {})
    register_items = [
        (variant, stats.get("registers")) for variant, stats in ptxas.items()
    ]
    spill_rows = []
    for variant, stats in ptxas.items():
        spill_rows.append(
            [
                html.escape(variant),
                fmt(stats.get("registers"), 0),
                fmt(stats.get("stack_frame_bytes"), 0),
                fmt(stats.get("spill_stores_bytes"), 0),
                fmt(stats.get("spill_loads_bytes"), 0),
            ]
        )

    ncu_metrics = space.get("ncu", {}).get("metrics", {})
    ncu_rows = [
        ["Kernel", fmt(ncu_metrics.get("kernel"))],
        [
            "Duration",
            "n/a"
            if ncu_metrics.get("duration_ns") is None
            else f"{fmt(ncu_metrics.get('duration_ns') / 1000.0)} us",
        ],
        ["DRAM Throughput", f"{fmt(ncu_metrics.get('dram_throughput_pct'))}%"],
        ["Memory Throughput", f"{fmt(ncu_metrics.get('memory_throughput_pct'))}%"],
        ["SM Throughput", f"{fmt(ncu_metrics.get('sm_throughput_pct'))}%"],
        ["Achieved Occupancy", f"{fmt(ncu_metrics.get('achieved_occupancy_pct'))}%"],
        ["Active Warps / SM", fmt(ncu_metrics.get("active_warps_per_sm"))],
        ["Block x Grid", f"{fmt(ncu_metrics.get('block_size'), 0)} x {fmt(ncu_metrics.get('grid_size'), 0)}"],
    ]

    summary_rows = []
    for variant, stats in variants.items():
        summary_rows.append(
            [
                html.escape(variant),
                "yes" if stats.get("correct") else "no",
                f"{fmt(stats.get('median_ms'), 4)} ms",
                f"{fmt(stats.get('speedup_vs_original_row_stride'))}x",
                f"{fmt(stats.get('effective_bandwidth_gbps'))} GB/s",
            ]
        )

    binary_sizes = space.get("binary_sizes", {})
    binary_rows = [
        ["optimized.x", f"{fmt((binary_sizes.get('optimized_x_bytes') or 0) / 1024.0)} KiB"],
        ["fatbin.x", f"{fmt((binary_sizes.get('fatbin_x_bytes') or 0) / 1024.0)} KiB"],
        ["Logical traffic / launch", f"{logical_mib:.2f} MiB"],
    ]

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CUDA Kernel Profile</title>
<style>
body {{
  color: #172026;
  font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 32px auto;
  max-width: 980px;
  padding: 0 20px;
}}
h1 {{ font-size: 28px; margin-bottom: 6px; }}
h2 {{ font-size: 18px; margin: 30px 0 12px; }}
.meta {{ color: #58656f; margin-top: 0; }}
section {{ border-top: 1px solid #d8e0e6; padding-top: 10px; }}
svg {{ height: auto; max-width: 100%; }}
rect {{ fill: #147d64; }}
.label {{ fill: #26343d; font-size: 13px; }}
.value {{ fill: #3c4b54; font-size: 13px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #d8e0e6; padding: 8px 10px; text-align: left; }}
th {{ color: #52616b; font-weight: 650; }}
code {{ background: #edf2f5; border-radius: 4px; padding: 1px 4px; }}
</style>
</head>
<body>
<h1>CUDA Kernel Profile</h1>
<p class="meta">{html.escape(summary.get('gpu', {}).get('name', 'Unknown GPU'))} · {summary.get('dimx', 'n/a')} x {summary.get('dimy', 'n/a')} · nreps {summary.get('nreps', 'n/a')}</p>
<section>
<h2>Benchmark Summary</h2>
{render_table(["Variant", "Correct", "Median Time", "Speedup", "Effective Bandwidth"], summary_rows)}
</section>
{bar_chart("Time Per Launch", time_items, " ms", digits=4)}
{bar_chart("Speedup Vs Row-Stride", speed_items, "x")}
{bar_chart("Effective Memory Bandwidth", bandwidth_items, " GB/s", digits=1)}
<section>
<h2>Nsight Compute Snapshot</h2>
{render_table(["Metric", "Value"], ncu_rows)}
</section>
{bar_chart("Registers Per Thread", register_items, "", digits=0)}
<section>
<h2>Kernel Resource Footprint</h2>
{render_table(["Variant", "Registers", "Stack Bytes", "Spill Stores", "Spill Loads"], spill_rows)}
</section>
<section>
<h2>Space Summary</h2>
{render_table(["Item", "Value"], binary_rows)}
</section>
</body>
</html>
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(html_doc)
    print(f"Wrote {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
