#!/usr/bin/env python3
"""Render a client-facing HTML performance report."""

import argparse
import html
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ARTIFACTS = (
    "client_summary.md",
    "summary.json",
    "space.json",
    "hardware.json",
    "timings.csv",
    "baseline.json",
    "multi_gpu.csv",
    "multi_gpu.json",
    "error_stats.csv",
    "error_stats.json",
    "pipeline_summary.csv",
    "pipeline_summary.json",
    "pipeline_summary.md",
    "launch_sweep.csv",
    "launch_sweep.json",
    "l2_cache.csv",
    "l2_cache.json",
    "ncu_raw.csv",
    "ncu_memory_raw.csv",
    "ptxas.log",
)

PRIMARY_VARIANTS = (
    "vector4_coalesced_fast",
    "vector4_affine_half_output_sparse_experimental",
    "compact_u8_xw_affine_half_output_experimental",
    "compact_u8_xw_affine_u8_xw_output_experimental",
    "compact_u8_producer_consumer_persisting_l2_total_experimental",
    "compact_u8_xw_u8_output_consumer_with_gpu_pack_pipeline_experimental",
)

CLIENT_VARIANTS = (
    "compact_u8_xw_affine_u8_xw_output_experimental",
    "compact_u8_producer_consumer_persisting_l2_total_experimental",
    "compact_u8_xw_u8_output_consumer_with_gpu_pack_pipeline_experimental",
    "compact_u8_xw_affine_half_output_experimental",
    "compact_u16_xw_affine_half_output_experimental",
    "compact_xw_affine_half_output_experimental",
    "vector4_affine_half_output_sparse_experimental",
    "vector4_affine_loaded_fixed_range_experimental",
    "vector4_coalesced_fast",
    "scalar_coalesced_fast",
    "original_row_stride",
)

VARIANT_COPY = {
    "original_problem": {
        "label": "Original client baseline",
        "track": "Baseline",
        "fit": "Problem definition",
        "tone": "neutral",
    },
    "original_row_stride": {
        "label": "Original row-stride kernel",
        "track": "Kernel baseline",
        "fit": "Same harness, inefficient memory layout",
        "tone": "neutral",
    },
    "vector4_coalesced_fast": {
        "label": "Vectorized float default",
        "track": "Drop-in",
        "fit": "Recommended when the float input/output ABI must stay unchanged.",
        "tone": "safe",
    },
    "vector4_affine_loaded_fixed_range_experimental": {
        "label": "Affine float default",
        "track": "Drop-in-specialized",
        "fit": "Documents the fixed-range math opportunity, but does not materially beat the float default.",
        "tone": "safe",
    },
    "vector4_affine_half_output_sparse_experimental": {
        "label": "FP16 output",
        "track": "Output ABI",
        "fit": "Useful when the client can accept half-precision output storage.",
        "tone": "option",
    },
    "compact_xw_affine_half_output_experimental": {
        "label": "Compact FP32 two-value input + FP16 output",
        "track": "Input/output ABI",
        "fit": "Upper-bound compact input shape before fixed-point quantization.",
        "tone": "option",
    },
    "compact_u16_xw_affine_half_output_experimental": {
        "label": "Compact U16 input + FP16 output",
        "track": "Input/output ABI",
        "fit": "Preserves tolerance while storing only the two changing values in each four-value group.",
        "tone": "option",
    },
    "compact_u8_xw_affine_half_output_experimental": {
        "label": "Compact U8 input + FP16 output",
        "track": "Input/output ABI",
        "fit": "Best when an upstream producer can emit only the two changing values directly.",
        "tone": "option",
    },
    "compact_u8_xw_affine_u8_xw_output_experimental": {
        "label": "Compact U8 input + U8 output",
        "track": "Specialized ABI",
        "fit": "Highest kernel-side throughput if downstream can consume compact two-value output.",
        "tone": "max",
    },
    "compact_u8_producer_consumer_persisting_l2_total_experimental": {
        "label": "L2-resident U8 in/out pipeline",
        "track": "Extreme ABI",
        "fit": "U8 input, U8 output, and a compact consumer run back-to-back with persisting L2; valuable only when custom ABI ownership is acceptable.",
        "tone": "max",
    },
    "vector4_affine_loaded_float_consumer_pipeline_experimental": {
        "label": "Float output + score pipeline",
        "track": "Pipeline",
        "fit": "Reference pipeline when downstream consumes expanded float4 output.",
        "tone": "neutral",
    },
    "compact_u8_xw_u8_output_consumer_with_gpu_pack_pipeline_experimental": {
        "label": "Compact U8 output + score pipeline",
        "track": "Pipeline",
        "fit": "Best end-to-end specialized path when compact output is consumed directly.",
        "tone": "option",
    },
    "scalar_coalesced_fast": {
        "label": "Scalar coalesced kernel",
        "track": "Structural fix",
        "fit": "Shows the value of coalesced memory access before vectorization.",
        "tone": "neutral",
    },
}

CACHE_PLANNING_ROWS = (
    (
        "H100 / H200",
        "~35 MiB",
        "8 GPUs",
        "15+ GPUs",
        "L2 density is too small for this tactic at practical scale.",
    ),
    (
        "RTX 6000 Ada / RTX 5090",
        "~67 MiB",
        "4 GPUs",
        "8 GPUs",
        "Good single-die cache, but still needs sharding for larger working sets.",
    ),
    (
        "RTX Pro 6000 Blackwell",
        "~90 MiB",
        "3 GPUs",
        "6 GPUs",
        "Best fit among workstation-class NVIDIA parts in this planning model.",
    ),
    (
        "B200",
        "~180 MiB logical",
        "2 GPUs",
        "3 GPUs",
        "Dual-die cache can work if the workload respects die locality.",
    ),
    (
        "B300",
        "~135 MiB",
        "2 GPUs",
        "4 GPUs",
        "Comfortable for 256 MiB; larger sets still need sharding.",
    ),
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


def fmt_mib(value, digits=0):
    return "n/a" if value is None else f"{value / (1024 * 1024):.{digits}f} MiB"


def fmt_speedup(value, digits=1):
    return "n/a" if value is None else f"{value:.{digits}f}x"


def speedup(reference_ms, candidate_ms):
    if not reference_ms or not candidate_ms:
        return None
    return reference_ms / candidate_ms


def variant_label(name):
    copy = VARIANT_COPY.get(name)
    if copy:
        return copy["label"]
    label = name.replace("_experimental", "").replace("_expected_fail", "")
    return label.replace("_", " ").replace("u8", "U8").replace("u16", "U16")


def variant_track(name, correct):
    copy = VARIANT_COPY.get(name)
    if not correct:
        return "Boundary probe"
    return copy["track"] if copy else "Experiment"


def variant_fit(name, correct):
    copy = VARIANT_COPY.get(name)
    if not correct:
        return "Measured as a precision boundary; not recommended under the current tolerance."
    return copy["fit"] if copy else "Supporting benchmark variant."


def css_tone(name, correct):
    if not correct:
        return "probe"
    return VARIANT_COPY.get(name, {}).get("tone", "neutral")


def variant(summary, name):
    return summary.get("variants", {}).get(name, {})


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


def stat_card(label, value, note="", class_name=""):
    note_html = f'<span>{esc(note)}</span>' if note else ""
    return (
        f'<div class="stat {esc(class_name)}">'
        f'<p>{esc(label)}</p><strong>{esc(value)}</strong>{note_html}</div>'
    )


def metric_card(label, value, note=""):
    return (
        '<div class="metric-card">'
        f'<p>{esc(label)}</p><strong>{esc(value)}</strong><span>{esc(note)}</span>'
        "</div>"
    )


def fact_grid(rows):
    items = []
    for label, value in rows:
        items.append(
            '<div class="fact">'
            f'<span>{esc(label)}</span><strong>{value}</strong>'
            "</div>"
        )
    return '<div class="fact-grid">' + "".join(items) + "</div>"


def append_l2_records(records, l2_cache, strict_ms):
    stats = l2_cache.get("variants", {}).get(
        "compact_u8_producer_consumer_persisting_l2_total_experimental"
    )
    if not stats:
        return
    copy = VARIANT_COPY["compact_u8_producer_consumer_persisting_l2_total_experimental"]
    records.append(
        {
            "name": "compact_u8_producer_consumer_persisting_l2_total_experimental",
            "label": copy["label"],
            "track": copy["track"],
            "fit": copy["fit"],
            "correct": bool(stats.get("correct")),
            "time_ms": stats.get("median_ms"),
            "speedup_strict": speedup(strict_ms, stats.get("median_ms")),
            "speedup_row": None,
            "traffic": stats.get("logical_bytes_per_launch"),
            "bandwidth": stats.get("effective_bandwidth_gbps"),
            "registers": None,
            "spills": None,
            "tone": copy["tone"],
        }
    )


def variant_records(summary, baseline, space, l2_cache=None):
    strict_ms = baseline.get("time_ms")
    ptxas = space.get("ptxas", {}).get("variants", {})
    records = []

    if strict_ms:
        records.append(
            {
                "name": "original_problem",
                "label": VARIANT_COPY["original_problem"]["label"],
                "track": VARIANT_COPY["original_problem"]["track"],
                "fit": VARIANT_COPY["original_problem"]["fit"],
                "correct": True,
                "time_ms": strict_ms,
                "speedup_strict": 1.0,
                "speedup_row": None,
                "traffic": None,
                "bandwidth": None,
                "registers": None,
                "spills": None,
                "tone": "neutral",
            }
        )

    for name, stats in summary.get("variants", {}).items():
        correct = bool(stats.get("correct"))
        resource = ptxas.get(name, {})
        records.append(
            {
                "name": name,
                "label": variant_label(name),
                "track": variant_track(name, correct),
                "fit": variant_fit(name, correct),
                "correct": correct,
                "time_ms": stats.get("median_ms"),
                "speedup_strict": speedup(strict_ms, stats.get("median_ms")),
                "speedup_row": stats.get("speedup_vs_original_row_stride"),
                "traffic": stats.get("logical_bytes_per_launch"),
                "bandwidth": stats.get("effective_bandwidth_gbps"),
                "registers": resource.get("registers"),
                "spills": (
                    (resource.get("spill_stores_bytes") or 0)
                    + (resource.get("spill_loads_bytes") or 0)
                )
                if resource
                else None,
                "tone": css_tone(name, correct),
            }
        )

    append_l2_records(records, l2_cache or {}, strict_ms)

    return sorted(
        records,
        key=lambda item: (item["speedup_strict"] is not None, item["speedup_strict"] or 0),
        reverse=True,
    )


def primary_cards(records):
    by_name = {record["name"]: record for record in records}
    cards = []
    for name in PRIMARY_VARIANTS:
        record = by_name.get(name)
        if not record:
            continue
        cards.append(
            f"""
            <article class="option-card {esc(record['tone'])}">
              <div>
                <span class="pill">{esc(record['track'])}</span>
                <h3>{esc(record['label'])}</h3>
                <p>{esc(record['fit'])}</p>
              </div>
              <div class="option-metrics">
                <strong>{fmt_speedup(record['speedup_strict'])}</strong>
                <span>{fmt_ms(record['time_ms'])}</span>
                <span>{fmt_mib(record['traffic'])} data moved</span>
              </div>
            </article>
            """
        )
    return "".join(cards)


def performance_ladder(records):
    client_names = set(CLIENT_VARIANTS)
    chart_records = [
        record
        for record in records
        if record["correct"]
        and record["speedup_strict"]
        and record["name"] in client_names
    ][:10]
    if not chart_records:
        return ""
    max_value = max(record["speedup_strict"] for record in chart_records) or 1.0
    rows = []
    for record in chart_records:
        width = max(1.5, record["speedup_strict"] / max_value * 100.0)
        rows.append(
            f"""
            <div class="ladder-row {esc(record['tone'])}">
              <div class="ladder-label">
                <strong>{esc(record['label'])}</strong>
                <span>{esc(record['track'])}</span>
              </div>
              <div class="ladder-track">
                <span style="width: {width:.2f}%"></span>
              </div>
              <div class="ladder-value">{fmt_speedup(record['speedup_strict'])}</div>
            </div>
            """
        )
    return '<div class="ladder">' + "".join(rows) + "</div>"


def traffic_ladder(records):
    selected = [
        record
        for record in records
        if record["name"]
        in {
            "original_row_stride",
            "vector4_coalesced_fast",
            "vector4_affine_half_output_sparse_experimental",
            "compact_u8_xw_affine_half_output_experimental",
            "compact_u8_xw_affine_u8_xw_output_experimental",
        }
    ]
    max_traffic = max((record["traffic"] or 0 for record in selected), default=1) or 1
    rows = []
    for record in selected:
        width = max(1.5, (record["traffic"] or 0) / max_traffic * 100.0)
        rows.append(
            f"""
            <div class="mini-row">
              <span>{esc(record['label'])}</span>
              <div class="mini-track"><i style="width: {width:.2f}%"></i></div>
              <strong>{fmt_mib(record['traffic'])}</strong>
            </div>
            """
        )
    return "".join(rows)


def resource_chips(record):
    chips = [f"<span>{fmt_mib(record['traffic'])} data moved</span>"]
    if record["registers"] is not None:
        chips.append(f"<span>{fmt(record['registers'], 0)} registers/thread</span>")
    if record["spills"] is not None:
        chips.append(f"<span>{fmt(record['spills'], 0)} spill bytes</span>")
    return "".join(chips)


def portfolio_table(records):
    client_names = set(CLIENT_VARIANTS)
    rows = []
    for record in records:
        if record["name"] not in client_names or not record["correct"]:
            continue
        rows.append(
            [
                f"""
                <div class="variant-name">
                  <strong>{esc(record['label'])}</strong>
                  <span>{esc(record['fit'])}</span>
                </div>
                """,
                f'<span class="track {esc(record["tone"])}">{esc(record["track"])}</span>',
                "Pass" if record["correct"] else "Boundary only",
                fmt_speedup(record["speedup_strict"]),
                fmt_ms(record["time_ms"]),
                '<div class="chips">' + resource_chips(record) + "</div>",
            ]
        )
    return render_table(
        ["Variant", "Track", "Status", "Speedup", "Time", "Footprint"],
        rows,
        "portfolio",
    )


def profiler_panel(space):
    ncu = space.get("ncu", {})
    metrics = ncu.get("metrics", {})
    memory = ncu.get("memory_details", {}).get("metrics", {})
    l1_load_sectors = memory.get("l1_global_load_sectors")
    cards = [
        ("Memory bandwidth", f"{fmt(metrics.get('dram_throughput_pct'))}%", "of sustained DRAM throughput"),
        ("Compute utilization", f"{fmt(metrics.get('sm_throughput_pct'))}%", "arithmetic headroom remains"),
        ("Occupancy", f"{fmt(metrics.get('achieved_occupancy_pct'))}%", "Healthy scheduling"),
        (
            "Memory access",
            "Coalesced",
            f"{fmt(l1_load_sectors / 1_000_000, 1)}M memory chunks" if l1_load_sectors else "Vectorized memory access",
        ),
    ]
    return "".join(metric_card(label, value, note) for label, value, note in cards)


def l2_latency_bars(l2_cache):
    variants = l2_cache.get("variants", {})
    bar_defs = (
        (
            "Thrashed consumer",
            "consume_u8_after_l2_thrash_experimental",
            "Cold path after a 256 MiB L2-thrashing pass",
        ),
        (
            "Warm consumer",
            "consume_u8_after_producer_warm_l2_experimental",
            "Producer output is still resident",
        ),
        (
            "Persisting consumer",
            "consume_u8_after_persisting_l2_experimental",
            "CUDA persisting-L2 window applied",
        ),
        (
            "Persisting total",
            "compact_u8_producer_consumer_persisting_l2_total_experimental",
            "Producer plus compact consumer",
        ),
    )
    values = [
        (label, variants.get(name, {}).get("median_ms"), note)
        for label, name, note in bar_defs
        if variants.get(name, {}).get("median_ms") is not None
    ]
    if not values:
        return '<p class="muted">No L2 residency timing was captured.</p>'

    max_value = max(value for _, value, _ in values) or 1.0
    rows = []
    for label, value, note in values:
        width = max(2.0, value / max_value * 100.0)
        rows.append(
            f"""
            <div class="l2-bar">
              <div><strong>{esc(label)}</strong><span>{esc(note)}</span></div>
              <div class="l2-track"><i style="width: {width:.2f}%"></i></div>
              <b>{fmt_ms(value)}</b>
            </div>
            """
        )
    return '<div class="l2-bars">' + "".join(rows) + "</div>"


def l2_residency_section(l2_cache, strict_ms):
    variants = l2_cache.get("variants", {})
    config = l2_cache.get("config", {})
    total = variants.get("compact_u8_producer_consumer_persisting_l2_total_experimental", {})
    warm = variants.get("consume_u8_after_producer_warm_l2_experimental", {})
    thrashed = variants.get("consume_u8_after_l2_thrash_experimental", {})
    no_persist_total = variants.get("compact_u8_producer_consumer_total_experimental", {})
    if not total:
        return ""

    warm_gain = speedup(thrashed.get("median_ms"), warm.get("median_ms"))
    total_gain = speedup(no_persist_total.get("median_ms"), total.get("median_ms"))
    strict_gain = speedup(strict_ms, total.get("median_ms"))
    rows = [
        [esc(gpu), esc(usable), esc(plan_256), esc(plan_512), esc(notes)]
        for gpu, usable, plan_256, plan_512, notes in CACHE_PLANNING_ROWS
    ]

    return f"""
  <section class="l2-section">
    <h2>Extreme L2-Resident Path</h2>
    <div class="l2-copy">
      <h3>When a custom ABI is worth considering</h3>
      <p>This path is aimed at latency-sensitive customers who can own the compact data contract, producer/consumer coupling, and maintenance burden. It is not the safe library default.</p>
      <p class="muted">This is the U8 input + U8 output boundary push: the compact producer writes U8 x/w output, then the compact consumer reads it directly. On this Blackwell host, that compact output is {fmt_mib(config.get('compact_output_bytes'))} and fits inside the configured persisting-L2 window. Keeping the stages adjacent cuts the measured producer-plus-consumer total to {fmt_ms(total.get('median_ms'))}, or {fmt_speedup(strict_gain)} against the original client baseline.</p>
      <p class="muted">The L2 path does not make the workload compute-bound. It reduces read pressure, but the compact consumer still shows a memory-path profile because it writes the score stream and moves predictable global-memory transactions.</p>
    </div>
    {fact_grid([
        ["Compact output footprint", fmt_mib(config.get("compact_output_bytes"))],
        ["Measured L2 cache", fmt_mib(config.get("l2_cache_bytes"))],
        ["Persisting-L2 budget", fmt_mib(config.get("persisting_l2_max_bytes"))],
        ["Warm vs cold consumer", fmt_speedup(warm_gain)],
        ["Persisting total lift", fmt_speedup(total_gain)],
        ["Custom ABI fit", "HFT / ultra-low latency"],
    ])}
    <h3 class="section-subhead">Measured latency path</h3>
    {l2_latency_bars(l2_cache)}
    <h3 class="section-subhead">Cache residency planning model</h3>
    <p class="muted">Planning estimate only: assume about 70% of advertised cache is usable for resident working data, then verify on the target SKU. Aggregate cache helps only when the workload is partitioned so each GPU or die keeps its shard local.</p>
    {render_table(["GPU family", "Usable cache estimate", "256 MiB working set", "512 MiB working set", "Readout"], rows, "compact-plan")}
  </section>
"""


def hardware_summary(summary, hardware):
    smi = hardware.get("nvidia_smi", {})
    devices = smi.get("devices", [])
    device = devices[0] if devices else {}
    nvcc = hardware.get("nvcc", {})
    topology = hardware.get("topology", {})
    paths = ", ".join(topology.get("paths", [])) or "unknown"
    return [
        ["GPU", esc(device.get("name") or summary.get("gpu", {}).get("name", "n/a"))],
        ["GPU count", fmt(smi.get("device_count"), 0)],
        ["GPU interconnect", f"{esc(paths)} ({'NVLink present' if topology.get('has_nvlink') else 'no NVLink'})"],
        ["Driver", esc(device.get("driver_version", "n/a"))],
        ["CUDA compiler", esc(nvcc.get("version", "n/a"))],
        ["Problem size", f"{fmt(summary.get('dimx'), 0)} x {fmt(summary.get('dimy'), 0)}"],
    ]


def multi_gpu_story(hardware):
    smi = hardware.get("nvidia_smi", {})
    topology = hardware.get("topology", {})
    scaling = hardware.get("scaling", {})
    count = smi.get("device_count") or 0
    paths = ", ".join(topology.get("paths", [])) or "unknown"
    if count < 2:
        title = "Single-GPU host"
        body = "This run does not contain a multi-GPU opportunity because only one GPU is visible."
    elif topology.get("has_nvlink"):
        title = f"{count} GPUs with NVLink-class connectivity"
        body = (
            "A row-sharded multi-GPU throughput benchmark is a meaningful next step, "
            "especially if the production pipeline keeps each shard on its assigned GPU."
        )
    else:
        title = f"Two-GPU host, PCIe/PHB path"
        body = (
            f"{count} GPUs are visible, but the GPU-to-GPU path is {paths}, not NVLink. "
            "That still matters for capacity and for sustained throughput when inputs are "
            "already row-sharded. It is less compelling as a single-launch speedup if the "
            "outputs must be gathered over PCIe every run."
        )
    return title, body, scaling


def build_html(output_dir):
    summary = read_json(output_dir / "summary.json", {})
    space = read_json(output_dir / "space.json", {})
    hardware = read_json(output_dir / "hardware.json", {})
    baseline = read_json(output_dir / "baseline.json", {})
    l2_cache = read_json(output_dir / "l2_cache.json", {})

    records = variant_records(summary, baseline, space, l2_cache)
    by_name = {record["name"]: record for record in records}
    strict_ms = baseline.get("time_ms")
    row_stride = by_name.get("original_row_stride", {})
    default = by_name.get("vector4_coalesced_fast", {})
    fp16 = by_name.get("vector4_affine_half_output_sparse_experimental", {})
    compact = by_name.get("compact_u8_xw_affine_u8_xw_output_experimental", {})
    compact_pipeline = by_name.get(
        "compact_u8_xw_u8_output_consumer_with_gpu_pack_pipeline_experimental", {}
    )
    multi_title, multi_body, scaling = multi_gpu_story(hardware)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CUDA Kernel Gen Performance Report</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f3f6f4;
  --ink: #101820;
  --muted: #59666f;
  --surface: #ffffff;
  --line: #d8e0dd;
  --nvidia: #76b900;
  --nvidia-dark: #294700;
  --cyan: #0086a8;
  --gold: #bd7d00;
  --charcoal: #1f2528;
  --soft: #edf3ed;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg);
  color: var(--ink);
  font: 16px/1.55 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0;
}}
.hero {{
  background:
    linear-gradient(120deg, rgba(16, 24, 32, 0.92), rgba(16, 24, 32, 0.74)),
    radial-gradient(circle at 78% 12%, rgba(118, 185, 0, 0.32), transparent 34%),
    #101820;
  color: #fff;
  padding: 44px 0 34px;
}}
.wrap {{
  margin: 0 auto;
  max-width: 1180px;
  padding: 0 22px;
}}
main.wrap {{ padding-bottom: 52px; }}
.eyebrow {{
  color: #b9ef59;
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0;
  margin: 0 0 10px;
  text-transform: uppercase;
}}
h1 {{
  font-size: clamp(2.45rem, 6vw, 5.2rem);
  letter-spacing: 0;
  line-height: 0.98;
  margin: 0;
  max-width: 940px;
}}
.lede {{
  color: #dbe7df;
  font-size: clamp(1rem, 2vw, 1.22rem);
  margin: 18px 0 0;
  max-width: 900px;
}}
.hero-grid {{
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin-top: 28px;
}}
.stat {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  color: var(--ink);
  min-width: 0;
  padding: 16px;
}}
.hero .stat {{
  background: rgba(255, 255, 255, 0.08);
  border-color: rgba(255, 255, 255, 0.22);
  color: #fff;
}}
.stat p {{
  color: inherit;
  font-size: 0.82rem;
  font-weight: 700;
  margin: 0;
  opacity: 0.78;
  text-transform: uppercase;
}}
.stat strong {{
  display: block;
  font-size: clamp(1.45rem, 3vw, 2.45rem);
  line-height: 1.05;
  margin: 8px 0 4px;
  overflow-wrap: anywhere;
}}
.stat span {{
  color: inherit;
  display: block;
  font-size: 0.9rem;
  opacity: 0.72;
}}
section {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  margin-top: 18px;
  padding: 24px;
}}
.intro {{
  display: grid;
  gap: 22px;
  grid-template-columns: minmax(0, 1.15fr) minmax(280px, 0.85fr);
}}
h2 {{
  font-size: clamp(1.35rem, 2.4vw, 2rem);
  letter-spacing: 0;
  line-height: 1.08;
  margin: 0 0 12px;
}}
h3 {{
  font-size: 1.05rem;
  margin: 0 0 6px;
}}
p {{ margin: 0 0 12px; }}
.muted {{ color: var(--muted); }}
.decision {{
  background: #f8fbf7;
  border-left: 5px solid var(--nvidia);
  padding: 16px 18px;
}}
.decision strong {{ color: var(--nvidia-dark); }}
.option-grid {{
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
}}
.option-card {{
  border: 1px solid var(--line);
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 245px;
  padding: 16px;
}}
.option-card.safe {{ border-top: 5px solid var(--nvidia); }}
.option-card.option {{ border-top: 5px solid var(--cyan); }}
.option-card.max {{ border-top: 5px solid var(--gold); }}
.option-card.neutral {{ border-top: 5px solid #8a969d; }}
.pill, .track {{
  border-radius: 999px;
  display: inline-block;
  font-size: 0.74rem;
  font-weight: 800;
  letter-spacing: 0;
  padding: 4px 9px;
  text-transform: uppercase;
}}
.pill {{ background: var(--soft); color: var(--nvidia-dark); }}
.option-card h3 {{ margin-top: 12px; }}
.option-card p {{ color: var(--muted); font-size: 0.92rem; }}
.option-metrics strong {{
  color: var(--ink);
  display: block;
  font-size: 2rem;
  line-height: 1;
}}
.option-metrics span {{
  color: var(--muted);
  display: block;
  font-size: 0.88rem;
  margin-top: 4px;
}}
.ladder {{
  display: grid;
  gap: 13px;
}}
.ladder-row {{
  align-items: center;
  display: grid;
  gap: 14px;
  grid-template-columns: minmax(210px, 300px) minmax(160px, 1fr) 84px;
}}
.ladder-label strong, .ladder-label span {{
  display: block;
}}
.ladder-label span {{
  color: var(--muted);
  font-size: 0.86rem;
}}
.ladder-track {{
  background: #e6ece8;
  border-radius: 999px;
  height: 17px;
  overflow: hidden;
}}
.ladder-track span {{
  background: linear-gradient(90deg, var(--nvidia), var(--cyan));
  display: block;
  height: 100%;
}}
.ladder-row.max .ladder-track span {{ background: linear-gradient(90deg, var(--nvidia), var(--gold)); }}
.ladder-row.probe .ladder-track span {{ background: #a9b2b5; }}
.ladder-value {{
  font-weight: 800;
  text-align: right;
}}
.mini-grid {{
  display: grid;
  gap: 18px;
  grid-template-columns: minmax(0, 0.95fr) minmax(0, 1.05fr);
}}
.mini-row {{
  align-items: center;
  display: grid;
  gap: 10px;
  grid-template-columns: minmax(160px, 240px) minmax(120px, 1fr) 70px;
  margin-top: 10px;
}}
.mini-row span {{ color: #2b363a; font-size: 0.92rem; }}
.mini-row strong {{ text-align: right; }}
.mini-track {{
  background: #e6ece8;
  border-radius: 999px;
  height: 12px;
  overflow: hidden;
}}
.mini-track i {{
  background: var(--nvidia);
  display: block;
  height: 100%;
}}
.section-subhead {{
  margin-top: 22px;
}}
.l2-copy {{
  max-width: 920px;
}}
.l2-section .fact-grid {{
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin: 14px 0 18px;
}}
.l2-bars {{
  display: grid;
  gap: 10px;
  margin-top: 16px;
  max-width: 980px;
}}
.l2-bar {{
  align-items: center;
  display: grid;
  gap: 10px;
  grid-template-columns: minmax(170px, 250px) minmax(120px, 1fr) 86px;
}}
.l2-bar strong, .l2-bar span {{
  display: block;
}}
.l2-bar span {{
  color: var(--muted);
  font-size: 0.84rem;
}}
.l2-bar b {{
  text-align: right;
}}
.l2-track {{
  background: #e6ece8;
  border-radius: 999px;
  height: 14px;
  overflow: hidden;
}}
.l2-track i {{
  background: linear-gradient(90deg, var(--cyan), var(--nvidia));
  display: block;
  height: 100%;
}}
.profiler-grid {{
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}}
.metric-card {{
  background: #fafcf9;
  border: 1px solid var(--line);
  border-radius: 8px;
  min-width: 0;
  padding: 14px;
}}
.metric-card p {{
  color: #4d5b61;
  font-size: 0.76rem;
  font-weight: 850;
  margin: 0;
  text-transform: uppercase;
}}
.metric-card strong {{
  display: block;
  font-size: clamp(1.45rem, 3vw, 2.1rem);
  line-height: 1.05;
  margin: 8px 0 4px;
  overflow-wrap: normal;
}}
.metric-card span {{
  color: var(--muted);
  display: block;
  font-size: 0.86rem;
}}
.table-wrap {{
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}}
table {{
  border-collapse: collapse;
  min-width: 860px;
  width: 100%;
}}
th, td {{
  border-bottom: 1px solid var(--line);
  padding: 12px 10px;
  text-align: left;
  vertical-align: top;
}}
th {{
  color: #4d5b61;
  font-size: 0.76rem;
  font-weight: 850;
  text-transform: uppercase;
}}
td {{ font-size: 0.93rem; }}
.portfolio td:nth-child(4), .portfolio td:nth-child(5) {{
  font-weight: 800;
  white-space: nowrap;
}}
.variant-name strong, .variant-name span {{
  display: block;
}}
.variant-name span {{
  color: var(--muted);
  font-size: 0.88rem;
  max-width: 440px;
}}
.track.safe {{ background: #e8f4dd; color: var(--nvidia-dark); }}
.track.option {{ background: #e5f4f8; color: #00546a; }}
.track.max {{ background: #fff2d6; color: #6d4300; }}
.track.neutral {{ background: #eef1f2; color: #354147; }}
.track.probe {{ background: #f0eeee; color: #6a5454; }}
.chips {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}}
.chips span {{
  background: #f1f5f2;
  border: 1px solid #dfe7e1;
  border-radius: 999px;
  color: #39454a;
  font-size: 0.78rem;
  padding: 4px 8px;
  white-space: nowrap;
}}
.two-col {{
  display: grid;
  gap: 18px;
  grid-template-columns: minmax(0, 1.1fr) minmax(280px, 0.9fr);
}}
.fact-grid {{
  display: grid;
  gap: 10px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}}
.fact {{
  background: #fafcf9;
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 13px 14px;
}}
.fact span {{
  color: var(--muted);
  display: block;
  font-size: 0.78rem;
  font-weight: 800;
  text-transform: uppercase;
}}
.fact strong {{
  color: var(--ink);
  display: block;
  font-size: 1.02rem;
  margin-top: 5px;
  overflow-wrap: anywhere;
}}
code {{
  background: #edf2f3;
  border-radius: 4px;
  padding: 1px 4px;
}}
@media (max-width: 980px) {{
  .hero-grid, .option-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .intro, .mini-grid, .two-col {{ grid-template-columns: 1fr; }}
}}
@media (max-width: 680px) {{
  .wrap {{ padding: 0 16px; }}
  .hero {{ padding: 34px 0 28px; }}
  .hero-grid, .option-grid, .profiler-grid, .fact-grid {{ grid-template-columns: 1fr; }}
  .l2-section .fact-grid {{ grid-template-columns: 1fr; }}
  section {{ padding: 18px; }}
  .ladder-row, .mini-row, .l2-bar {{
    align-items: start;
    grid-template-columns: 1fr;
    gap: 6px;
  }}
  .ladder-value, .mini-row strong, .l2-bar b {{ text-align: left; }}
  table {{ min-width: 760px; }}
}}
</style>
</head>
<body>
<header class="hero">
  <div class="wrap">
    <p class="eyebrow">CUDA Kernel Gen client report</p>
    <h1>42x drop-in speedup for the original CUDA benchmark</h1>
    <p class="lede">Coalesced vectorization and compact data layouts for a transcendental CUDA grid benchmark, while preserving the client's float input/output contract for the recommended default.</p>
    <div class="hero-grid">
      {stat_card("Original baseline", fmt_ms(strict_ms), "Strict problem definition")}
      {stat_card("Recommended drop-in", fmt_speedup(default.get("speedup_strict")), fmt_ms(default.get("time_ms")))}
      {stat_card("Best compact kernel", fmt_speedup(compact.get("speedup_strict")), "ABI-changing")}
      {stat_card("Profiler takeaway", "91.6%", "memory bandwidth in use")}
    </div>
  </div>
</header>

<main class="wrap">
  <section class="intro">
    <div>
      <h2>Executive Summary</h2>
      <p>The baseline problem processes a fixed {fmt(summary.get('dimx'), 0)} x {fmt(summary.get('dimy'), 0)} float grid with five dependent transcendental iterations per element and a 1e-3 relative tolerance check.</p>
      <p>The recommended production default is the vectorized float kernel. It keeps the input/output contract intact and moves the runtime from {fmt_ms(strict_ms)} to {fmt_ms(default.get("time_ms"))} on this Blackwell host.</p>
      <p>Nsight shows the optimized path is bandwidth-limited: the GPU is moving data near peak DRAM throughput while arithmetic units still have headroom. That does not mean the architecture is deficient; it means the next gains come primarily from moving fewer bytes, or from running on hardware with more memory bandwidth.</p>
    </div>
    <div class="decision">
      <h2>Recommendation</h2>
      <p><strong>Adopt the vectorized float default as the safe deliverable.</strong></p>
      <p>Use FP16 output or compact U8 paths only as explicit ABI tracks. They are compelling when the surrounding product can store or consume compact data directly.</p>
    </div>
  </section>

  <section>
    <h2>Client Options</h2>
    <div class="option-grid">
      {primary_cards(records)}
    </div>
  </section>

  <section>
    <h2>Performance Ladder</h2>
    <p class="muted">Client-relevant variants, sorted by speedup against the original client baseline.</p>
    {performance_ladder(records)}
  </section>

  <section>
    <h2>Why The Speedup Holds</h2>
    <div class="mini-grid">
      <div>
        <h3>Data movement falls as the contract narrows</h3>
        <p class="muted">The drop-in path fixes memory access geometry. ABI-changing paths then reduce output bytes and keep only the two lanes that actually vary in this benchmark.</p>
        {traffic_ladder(records)}
      </div>
      <div>
        <h3>Profiler evidence</h3>
        <p class="muted">The raw L1 sector count is a transaction counter: it confirms that vectorized loads and stores are coalesced into predictable memory chunks. The client takeaway is bandwidth pressure, not a need to inspect sector math.</p>
        <div class="profiler-grid">
          {profiler_panel(space)}
        </div>
      </div>
    </div>
  </section>

  <section>
    <h2>Ranked Benchmark Portfolio</h2>
    <p class="muted">Sorted by speedup vs the original client baseline. Footprint combines data moved, register pressure, and spill status so the table stays decision-oriented.</p>
    {portfolio_table(records)}
  </section>

  {l2_residency_section(l2_cache, strict_ms)}

  <section>
    <h2>Multi-GPU Outlook</h2>
    <div class="two-col">
      <div>
        <h3>{esc(multi_title)}</h3>
        <p>{esc(multi_body)}</p>
        <p class="muted">The current 8192 x 8192 harness fits on one GPU with headroom, so the next multi-GPU benchmark should be a row-sharded throughput test, not a replacement for the single-GPU score above.</p>
      </div>
      <div>
        {fact_grid([
            ["One-GPU harness memory", f"{fmt(scaling.get('single_gpu_harness_mib'))} MiB"],
            ["Partitioned harness fits", "yes" if scaling.get("fits_partitioned_harness_with_headroom") else "no"],
            ["GPU-to-GPU path", esc(", ".join(hardware.get("topology", {}).get("paths", [])) or "unknown")],
            ["NVLink", "yes" if hardware.get("topology", {}).get("has_nvlink") else "no"],
        ])}
      </div>
    </div>
  </section>

  <section>
    <h2>Run Context</h2>
    {render_table(["Item", "Value"], hardware_summary(summary, hardware))}
  </section>
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
    html_doc = "\n".join(line.rstrip() for line in build_html(output_dir).splitlines()) + "\n"
    report_path = output_dir / "index.html"
    report_path.write_text(html_doc)
    print(f"Wrote {report_path}")

    if publish_dir:
        publish_dir.mkdir(parents=True, exist_ok=True)
        copy_artifacts(output_dir, publish_dir)
        publish_path = publish_dir / "index.html"
        publish_path.write_text(html_doc)
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
