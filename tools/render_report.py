#!/usr/bin/env python3
"""Render a client-readable HTML performance report."""

import argparse
import html
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

KEY_VARIANTS = (
    ("Original row-stride ablation", "Drop-in reference", "original_row_stride"),
    ("Vectorized float default", "Drop-in recommendation", "vector4_coalesced_fast"),
    (
        "Affine float default",
        "Benchmark-specialized float path",
        "vector4_affine_loaded_fixed_range_experimental",
    ),
    (
        "FP16 output",
        "ABI-changing output reduction",
        "vector4_affine_half_output_sparse_experimental",
    ),
    (
        "Compact U8 input + FP16 output",
        "ABI-changing input reduction",
        "compact_u8_xw_affine_half_output_experimental",
    ),
    (
        "Compact U8 input + U8 output",
        "Strongest specialized ABI",
        "compact_u8_xw_affine_u8_xw_output_experimental",
    ),
    (
        "Float output + score pipeline",
        "Expanded downstream consumer",
        "vector4_affine_loaded_float_consumer_pipeline_experimental",
    ),
    (
        "Compact U8 output + score pipeline",
        "Compact downstream consumer",
        "compact_u8_xw_u8_output_consumer_with_gpu_pack_pipeline_experimental",
    ),
)

ARTIFACTS = (
    "client_summary.md",
    "summary.json",
    "space.json",
    "hardware.json",
    "timings.csv",
    "baseline.json",
    "ncu_raw.csv",
    "ncu_memory_raw.csv",
    "ptxas.log",
)


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def esc(value):
    return html.escape(str(value))


def fmt(value, digits=2):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return esc(value)


def fmt_ms(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f} ms"


def fmt_mib(value, digits=1):
    return "n/a" if value is None else f"{value / (1024 * 1024):.{digits}f} MiB"


def fmt_speedup(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}x"


def variant(summary, name):
    return summary.get("variants", {}).get(name, {})


def speedup(reference_ms, candidate_ms):
    if not reference_ms or not candidate_ms:
        return None
    return reference_ms / candidate_ms


def render_table(headers, rows, class_name=""):
    if not rows:
        return '<p class="muted">No data captured.</p>'
    head = "".join(f"<th>{esc(header)}</th>" for header in headers)
    body = []
    for row in rows:
        cells = []
        for header, cell in zip(headers, row):
            cells.append(f'<td data-label="{esc(header)}">{cell}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        f'<div class="table-wrap {esc(class_name)}"><table><thead><tr>{head}</tr>'
        f"</thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def stat_card(label, value, note=""):
    note_html = f'<span class="stat-note">{esc(note)}</span>' if note else ""
    return (
        '<div class="stat">'
        f'<span class="stat-label">{esc(label)}</span>'
        f'<strong>{esc(value)}</strong>'
        f"{note_html}</div>"
    )


def bar_list(title, rows, unit="", digits=2):
    rows = [(label, value) for label, value in rows if value is not None]
    if not rows:
        return ""
    max_value = max(value for _, value in rows) or 1.0
    items = []
    for label, value in rows:
        width = max(1.0, min(100.0, value / max_value * 100.0))
        items.append(
            '<div class="bar-row">'
            f'<span class="bar-label">{esc(label)}</span>'
            '<span class="bar-track">'
            f'<span class="bar-fill" style="width: {width:.2f}%"></span>'
            "</span>"
            f'<strong>{fmt(value, digits)}{esc(unit)}</strong>'
            "</div>"
        )
    return f'<section><h2>{esc(title)}</h2><div class="bar-list">{"".join(items)}</div></section>'


def artifact_links(output_dir, artifact_prefix):
    rows = []
    for name in ARTIFACTS:
        if (output_dir / name).exists():
            href = esc(f"{artifact_prefix}{name}")
            rows.append([f'<a href="{href}">{esc(name)}</a>', "present"])
    return rows


def topology_text(hardware):
    topology = hardware.get("topology", {})
    smi = hardware.get("nvidia_smi", {})
    device_count = smi.get("device_count")
    paths = ", ".join(topology.get("paths", [])) or "unknown"
    if topology.get("status") != "ok":
        return "GPU topology was not captured."
    if not device_count or device_count < 2:
        return "Only one GPU was detected, so no multi-GPU throughput result applies."
    if topology.get("has_nvlink"):
        return (
            f"{device_count} GPUs were detected with NVLink-class paths ({paths}). "
            "A row-sharded multi-GPU run would be a meaningful next throughput benchmark."
        )
    return (
        f"{device_count} GPUs were detected, but the GPU-to-GPU path is {paths}, not NVLink. "
        "For this pointwise kernel, multi-GPU is still meaningful for capacity or for a "
        "production pipeline that already shards rows per GPU. It is less meaningful as a "
        "single-job speedup if every launch must gather outputs through host/PCIe."
    )


def build_key_rows(summary, baseline):
    strict_ms = baseline.get("time_ms")
    rows = []
    for label, role, name in KEY_VARIANTS:
        stats = variant(summary, name)
        if not stats:
            continue
        median = stats.get("median_ms")
        rows.append(
            [
                esc(label),
                esc(role),
                "yes" if stats.get("correct") else "no",
                fmt_ms(median),
                fmt_speedup(speedup(strict_ms, median)),
                fmt_speedup(stats.get("speedup_vs_original_row_stride")),
                fmt_mib(stats.get("logical_bytes_per_launch")),
            ]
        )
    return rows


def build_all_rows(summary, baseline):
    strict_ms = baseline.get("time_ms")
    rows = []
    for name, stats in summary.get("variants", {}).items():
        median = stats.get("median_ms")
        rows.append(
            [
                f"<code>{esc(name)}</code>",
                "yes" if stats.get("correct") else "no",
                fmt_ms(median),
                fmt_speedup(speedup(strict_ms, median)),
                fmt_speedup(stats.get("speedup_vs_original_row_stride")),
                fmt_mib(stats.get("logical_bytes_per_launch")),
                f"{fmt(stats.get('effective_bandwidth_gbps'), 1)} GB/s",
            ]
        )
    return rows


def build_profiler_rows(space):
    ncu = space.get("ncu", {})
    metrics = ncu.get("metrics", {})
    memory = ncu.get("memory_details", {}).get("metrics", {})
    rows = [
        ["Nsight status", esc(ncu.get("status", "missing"))],
        ["Kernel", esc(metrics.get("kernel", "n/a"))],
        [
            "NCU duration",
            "n/a"
            if metrics.get("duration_ns") is None
            else f"{fmt(metrics.get('duration_ns') / 1000.0)} us",
        ],
        ["DRAM throughput", f"{fmt(metrics.get('dram_throughput_pct'))}%"],
        ["SM throughput", f"{fmt(metrics.get('sm_throughput_pct'))}%"],
        ["Achieved occupancy", f"{fmt(metrics.get('achieved_occupancy_pct'))}%"],
        ["Registers/thread", fmt(metrics.get("registers_per_thread"), 0)],
    ]
    if memory.get("dram_read_bytes") is not None:
        rows.append(["DRAM read bytes", fmt_mib(memory.get("dram_read_bytes"))])
    if memory.get("dram_write_bytes") is not None:
        rows.append(["DRAM write bytes", fmt_mib(memory.get("dram_write_bytes"))])
    rows.extend(
        [
            ["L1 global load sectors", fmt(memory.get("l1_global_load_sectors"), 0)],
            ["L1 global store sectors", fmt(memory.get("l1_global_store_sectors"), 0)],
            ["L2 read sectors", fmt(memory.get("l2_read_sectors"), 0)],
            ["L2 write sectors", fmt(memory.get("l2_write_sectors"), 0)],
        ]
    )
    if ncu.get("error"):
        rows.append(["Profiler note", esc(ncu.get("error"))])
    return rows


def build_resource_rows(space):
    rows = []
    for name, stats in space.get("ptxas", {}).get("variants", {}).items():
        rows.append(
            [
                f"<code>{esc(name)}</code>",
                fmt(stats.get("registers"), 0),
                fmt(stats.get("stack_frame_bytes"), 0),
                fmt(stats.get("spill_stores_bytes"), 0),
                fmt(stats.get("spill_loads_bytes"), 0),
            ]
        )
    return rows


def build_scaling_rows(hardware):
    rows = []
    for part in hardware.get("scaling", {}).get("partitions", []):
        fits = part.get("fits_harness_with_headroom")
        rows.append(
            [
                fmt(part.get("gpu_index"), 0),
                esc(part.get("gpu_name", "")),
                f"{fmt(part.get('row_start'), 0)}-{fmt(part.get('row_end'), 0)}",
                fmt(part.get("rows"), 0),
                f"{fmt(part.get('harness_mib'))} MiB",
                f"{fmt(part.get('usable_memory_mib'))} MiB",
                "yes" if fits else "no" if fits is False else "n/a",
            ]
        )
    return rows


def build_memory_model_rows(hardware):
    rows = []
    for model in hardware.get("scaling", {}).get("memory_models", []):
        rows.append(
            [
                esc(model.get("name", "")),
                fmt(model.get("bytes_per_element")),
                fmt_mib(model.get("total_bytes")),
                esc(model.get("note", "")),
            ]
        )
    return rows


def build_html(output_dir, artifact_prefix=""):
    summary = read_json(output_dir / "summary.json", {})
    space = read_json(output_dir / "space.json", {})
    hardware = read_json(output_dir / "hardware.json", {})
    baseline = read_json(output_dir / "baseline.json", {})

    gpu = summary.get("gpu", {})
    smi = hardware.get("nvidia_smi", {})
    devices = smi.get("devices", [])
    device = devices[0] if devices else {}
    nvcc = hardware.get("nvcc", {})
    classification = hardware.get("classification", {})
    scaling = hardware.get("scaling", {})

    strict_ms = baseline.get("time_ms")
    row_stride = variant(summary, "original_row_stride")
    default = variant(summary, "vector4_coalesced_fast")
    fp16 = variant(summary, "vector4_affine_half_output_sparse_experimental")
    compact = variant(summary, "compact_u8_xw_affine_u8_xw_output_experimental")
    compact_pipeline = variant(
        summary, "compact_u8_xw_u8_output_consumer_with_gpu_pack_pipeline_experimental"
    )

    baseline_note = (
        f"The strict original problem definition measured {fmt_ms(strict_ms)} on this host."
        if strict_ms
        else "The strict original problem definition was not captured in this report run."
    )
    default_vs_strict = speedup(strict_ms, default.get("median_ms"))
    compact_vs_strict = speedup(strict_ms, compact.get("median_ms"))

    notes = classification.get("notes", [])
    if scaling.get("recommendation"):
        notes.append(scaling["recommendation"])
    recommendation_items = "".join(f"<li>{esc(note)}</li>" for note in notes)

    chart_rows = []
    for label, _, name in KEY_VARIANTS[:6]:
        stats = variant(summary, name)
        if stats:
            chart_rows.append((label, stats.get("median_ms")))

    speed_rows = []
    for label, _, name in KEY_VARIANTS[1:6]:
        stats = variant(summary, name)
        if stats:
            speed_rows.append((label, speedup(strict_ms, stats.get("median_ms"))))

    hardware_rows = [
        ["GPU", esc(device.get("name") or gpu.get("name", "n/a"))],
        [
            "Compute capability",
            esc(device.get("compute_capability") or gpu.get("compute_capability", "n/a")),
        ],
        ["Device count", fmt(smi.get("device_count"), 0)],
        ["Driver", esc(device.get("driver_version", "n/a"))],
        ["CUDA compiler", esc(nvcc.get("version", "n/a"))],
        ["Problem size", f"{fmt(summary.get('dimx'), 0)} x {fmt(summary.get('dimy'), 0)}"],
        ["Tolerance contract", "1e-3 relative tolerance"],
        ["Scaling status", esc(scaling.get("status", "n/a"))],
    ]

    artifact_rows = artifact_links(output_dir, artifact_prefix)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CUDA Kernel Gen Performance Report</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f6f8f8;
  --surface: #ffffff;
  --ink: #172026;
  --muted: #60707a;
  --line: #d9e1e4;
  --green: #147d64;
  --blue: #245f9f;
  --gold: #9b6a10;
  --soft-green: #e7f4ef;
  --soft-blue: #e8f0fa;
  --soft-gold: #f7efd9;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg);
  color: var(--ink);
  font: 16px/1.55 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0;
}}
main {{
  margin: 0 auto;
  max-width: 1120px;
  padding: 28px 18px 48px;
}}
.hero {{
  background: var(--surface);
  border-bottom: 4px solid var(--green);
  padding: 34px 0 26px;
}}
.hero-inner {{
  margin: 0 auto;
  max-width: 1120px;
  padding: 0 18px;
}}
.eyebrow {{
  color: var(--green);
  font-size: 0.78rem;
  font-weight: 750;
  letter-spacing: 0;
  margin: 0 0 8px;
  text-transform: uppercase;
}}
h1 {{
  font-size: clamp(2rem, 4vw, 3.7rem);
  line-height: 1.02;
  margin: 0;
  max-width: 900px;
}}
.lede {{
  color: var(--muted);
  font-size: 1.08rem;
  max-width: 820px;
}}
section {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin-top: 18px;
  padding: 20px;
}}
h2 {{
  font-size: 1.25rem;
  line-height: 1.2;
  margin: 0 0 12px;
}}
h3 {{
  font-size: 1rem;
  margin: 18px 0 8px;
}}
p {{ margin: 0 0 12px; }}
.muted {{ color: var(--muted); }}
.stats {{
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin-top: 22px;
}}
.stat {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  min-width: 0;
  padding: 14px;
}}
.stat:nth-child(2) {{ background: var(--soft-green); }}
.stat:nth-child(3) {{ background: var(--soft-blue); }}
.stat:nth-child(4) {{ background: var(--soft-gold); }}
.stat-label, .stat-note {{
  color: var(--muted);
  display: block;
  font-size: 0.82rem;
}}
.stat strong {{
  display: block;
  font-size: 1.55rem;
  line-height: 1.1;
  margin: 6px 0;
  overflow-wrap: anywhere;
}}
.two-col {{
  display: grid;
  gap: 18px;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
}}
.table-wrap {{
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}}
table {{
  border-collapse: collapse;
  min-width: 720px;
  width: 100%;
}}
th, td {{
  border-bottom: 1px solid var(--line);
  padding: 10px 9px;
  text-align: left;
  vertical-align: top;
}}
th {{
  color: #4b5961;
  font-size: 0.82rem;
  font-weight: 750;
  text-transform: uppercase;
}}
td {{ font-size: 0.94rem; }}
code {{
  background: #edf2f3;
  border-radius: 4px;
  padding: 1px 4px;
}}
a {{ color: var(--blue); }}
ul {{
  margin: 0;
  padding-left: 20px;
}}
li {{ margin: 8px 0; }}
.callout {{
  background: #f8fbfb;
  border-left: 4px solid var(--blue);
  padding: 14px 16px;
}}
.bar-list {{
  display: grid;
  gap: 11px;
}}
.bar-row {{
  align-items: center;
  display: grid;
  gap: 10px;
  grid-template-columns: minmax(180px, 280px) minmax(120px, 1fr) 90px;
}}
.bar-label {{
  color: #27343a;
  font-size: 0.94rem;
}}
.bar-track {{
  background: #e5ecef;
  border-radius: 999px;
  display: block;
  height: 14px;
  overflow: hidden;
}}
.bar-fill {{
  background: linear-gradient(90deg, var(--green), var(--blue));
  display: block;
  height: 100%;
}}
details summary {{
  cursor: pointer;
  font-weight: 750;
}}
.footer {{
  color: var(--muted);
  margin: 24px 0 0;
}}
@media (max-width: 820px) {{
  .hero {{ padding-top: 26px; }}
  .stats, .two-col {{ grid-template-columns: 1fr; }}
  section {{ padding: 16px; }}
  .bar-row {{
    align-items: start;
    grid-template-columns: 1fr;
    gap: 6px;
  }}
  .bar-row strong {{ font-size: 0.95rem; }}
  table {{ min-width: 640px; }}
}}
</style>
</head>
<body>
<header class="hero">
  <div class="hero-inner">
    <p class="eyebrow">CUDA Kernel Gen report</p>
    <h1>Baseline-preserving CUDA speedup with explicit ABI tradeoffs</h1>
    <p class="lede">{esc(baseline_note)} The safe drop-in path keeps float input/output semantics; the faster compact paths are separate choices for clients that can change storage or downstream consumption.</p>
    <div class="stats">
      {stat_card("Strict original baseline", fmt_ms(strict_ms), "problem definition")}
      {stat_card("Drop-in float default", fmt_ms(default.get("median_ms")), fmt_speedup(default_vs_strict))}
      {stat_card("FP16 output path", fmt_ms(fp16.get("median_ms")), "ABI-changing")}
      {stat_card("Compact U8 in/out", fmt_ms(compact.get("median_ms")), fmt_speedup(compact_vs_strict))}
    </div>
  </div>
</header>
<main>
  <section>
    <h2>Executive Readout</h2>
    <div class="two-col">
      <div>
        <p>The original benchmark computes a fixed {fmt(summary.get('dimx'), 0)} x {fmt(summary.get('dimy'), 0)} float grid. Each element performs five dependent iterations of log, cos, sin, or tan selected by <code>ix % 4</code>, and correctness is checked against the existing 1e-3 relative tolerance.</p>
        <p>The drop-in recommendation is <code>vector4_coalesced_fast</code>: it preserves float in/out behavior while replacing the row-stride access pattern with contiguous vectorized work.</p>
      </div>
      <div class="callout">
        <p><strong>Client decision:</strong> use the float default when the ABI must stay unchanged. Consider FP16 output, compact U8 input, or compact U8 output only when the surrounding system can own that data contract.</p>
        <p class="muted">The report separates speedups against the strict original baseline from speedups against the in-harness row-stride ablation.</p>
      </div>
    </div>
  </section>

  <section>
    <h2>Key Results</h2>
    {render_table(["Variant", "Role", "Correct", "Median time", "Speedup vs strict baseline", "Speedup vs row-stride", "Logical traffic"], build_key_rows(summary, baseline))}
  </section>

  <section>
    <h2>Recommendations</h2>
    <ul>{recommendation_items}</ul>
  </section>

  <section>
    <h2>Multi-GPU Readout</h2>
    <p>{esc(topology_text(hardware))}</p>
    {render_table(["GPU", "Name", "Rows", "Row count", "Harness shard", "Usable memory", "Fits"], build_scaling_rows(hardware))}
    <details>
      <summary>Memory models used for capacity planning</summary>
      {render_table(["Model", "Bytes/element", "Total", "Meaning"], build_memory_model_rows(hardware))}
    </details>
  </section>

  <section>
    <h2>Hardware</h2>
    {render_table(["Item", "Value"], hardware_rows)}
  </section>

  {bar_list("Time Per Launch", chart_rows, " ms", digits=4)}
  {bar_list("Speedup Vs Strict Original", speed_rows, "x", digits=1)}

  <section>
    <h2>Profiler Evidence</h2>
    {render_table(["Metric", "Value"], build_profiler_rows(space))}
  </section>

  <section>
    <h2>Kernel Resource Footprint</h2>
    {render_table(["Variant", "Registers", "Stack bytes", "Spill stores", "Spill loads"], build_resource_rows(space))}
  </section>

  <section>
    <h2>All Timed Variants</h2>
    <details open>
      <summary>Show benchmark table</summary>
      {render_table(["Variant", "Correct", "Median time", "Speedup vs strict baseline", "Speedup vs row-stride", "Logical traffic", "Effective bandwidth"], build_all_rows(summary, baseline))}
    </details>
  </section>

  <section>
    <h2>Artifacts</h2>
    {render_table(["File", "Status"], artifact_rows)}
  </section>

  <p class="footer">Generated from machine-readable benchmark artifacts in <code>{esc(output_dir.relative_to(ROOT) if output_dir.is_relative_to(ROOT) else output_dir)}</code>.</p>
</main>
</body>
</html>
"""


def copy_artifacts(output_dir, publish_dir):
    artifact_dir = publish_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACTS:
        source = output_dir / name
        if source.exists():
            shutil.copy2(source, artifact_dir / name)


def write_report(output_dir, publish_dir=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    html_doc = build_html(output_dir)
    report_path = output_dir / "index.html"
    report_path.write_text(html_doc)
    print(f"Wrote {report_path}")

    if publish_dir:
        publish_dir.mkdir(parents=True, exist_ok=True)
        copy_artifacts(output_dir, publish_dir)
        published = build_html(output_dir, artifact_prefix="artifacts/")
        publish_path = publish_dir / "index.html"
        publish_path.write_text(published)
        print(f"Wrote {publish_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reports/latest")
    parser.add_argument("--publish-dir")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    publish_dir = ROOT / args.publish_dir if args.publish_dir else None
    write_report(output_dir, publish_dir)


if __name__ == "__main__":
    main()
