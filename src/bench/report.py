"""Generate a self-contained HTML report from benchmark results."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from bench.store import SessionResult


def generate_report(session: SessionResult, output_path: str | Path) -> Path:
    """Generate a self-contained HTML report and return its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = _build_html(session)
    output_path.write_text(doc)
    return output_path


def _v(d: dict, key: str, default: Any = None) -> Any:
    """Safe nested dict access."""
    return d.get(key, default)


def _fmt(val: Any, fmt: str = ".1f", suffix: str = "") -> str:
    if val is None:
        return "—"
    try:
        return f"{val:{fmt}}{suffix}"
    except (ValueError, TypeError):
        return str(val)


def _build_html(session: SessionResult) -> str:
    sys = session.system_info
    cfg = session.config_snapshot
    agg = session.aggregated
    tc = session.tool_calling
    qual = session.quality
    pwr = session.power
    families = cfg.get("model_families", [])

    # Build variant lookup: key -> {family_name, size, kind, quant, repo}
    variant_info: dict[str, dict] = {}
    family_variants: dict[str, list[str]] = defaultdict(list)
    for fam in families:
        for var in fam.get("variants", []):
            key = f"{var['repo']}|{var['quant']}"
            variant_info[key] = {
                "family": fam["name"],
                "size": fam["size"],
                "kind": fam["kind"],
                "quant": var["quant"],
                "repo": var["repo"],
                "reference": fam["reference"],
            }
            family_variants[fam["name"]].append(key)

    # Collect per-prompt data
    prompt_data = _collect_prompt_data(session.runs)

    # --- Build HTML ---
    parts: list[str] = []
    parts.append(_HEADER)

    # Timestamp
    ts = session.timestamp
    if ts:
        # Format "20260320_150000" -> "2026-03-20 15:00:00 UTC"
        try:
            from datetime import datetime
            dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
            ts_display = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except ValueError:
            ts_display = ts
        duration = session.duration_s
        if duration > 0:
            mins, secs = divmod(int(duration), 60)
            hours, mins = divmod(mins, 60)
            if hours > 0:
                dur_str = f"{hours}h {mins}m {secs}s"
            elif mins > 0:
                dur_str = f"{mins}m {secs}s"
            else:
                dur_str = f"{secs}s"
            ts_display += f" ({dur_str})"
        parts.append(f'<p class="timestamp">Run: {html.escape(ts_display)}</p>')

    # System info
    parts.append('<section class="system-info">')
    parts.append("<h2>System</h2>")
    parts.append('<div class="info-grid">')
    parts.append(_info_card("Chip", sys.get("chip", "Unknown")))
    parts.append(_info_card("Memory", f"{sys.get('memory_gb', '?')} GB"))
    parts.append(_info_card("OS", sys.get("os_version", "Unknown")))
    parts.append(_info_card("Python", sys.get("python_version", "?")))
    parts.append(_info_card("MLX", sys.get("mlx_version", "?") or "—"))
    parts.append(_info_card("mlx_lm", sys.get("mlx_lm_version", "?") or "—"))
    parts.append("</div>")

    parts.append('<div class="info-grid">')
    parts.append(_info_card("Warmup Runs", str(cfg.get("warmup_runs", "?"))))
    parts.append(_info_card("Measured Runs", str(cfg.get("measured_runs", "?"))))
    parts.append(_info_card("Max Tokens", str(cfg.get("max_tokens", "?"))))
    parts.append(_info_card("Temperature", str(cfg.get("temperature", "?"))))
    parts.append("</div>")
    parts.append("</section>")

    # Summary table
    parts.append('<section>')
    parts.append("<h2>Summary</h2>")
    parts.append('<p class="section-desc">Overall performance and quality for each model variant. '
                 '<b>TTFT</b> is how long you wait before the first word appears. '
                 '<b>Prefill</b> is how fast the model reads your prompt. '
                 '<b>Decode</b> is how fast it writes the response — this is the speed you feel during streaming. '
                 '<b>tok/W</b> shows energy efficiency — higher means more output per watt of power. '
                 '<b>Perplexity</b> measures language understanding (lower = smarter). '
                 '<b>MMLU</b> is accuracy on knowledge questions (higher = smarter).</p>')
    parts.append(_summary_table(agg, qual, pwr, tc, variant_info))
    parts.append("</section>")

    # Per-family comparison
    parts.append('<section>')
    parts.append("<h2>Quantization Comparison by Family</h2>")
    parts.append('<p class="section-desc">Compares quantization levels (8-bit vs 4-bit) of the same model. '
                 'Quantization shrinks the model to use less memory and run faster, but at the cost of quality. '
                 'The <b>reference</b> variant (usually 8-bit) is the quality baseline. '
                 '<b>vs ref</b> shows the speed and memory tradeoff — e.g., "1.5x" means 50% faster. '
                 '<b>PPL delta</b> shows quality loss — under 2% is negligible, over 5% is significant. '
                 '<b>Output Similarity</b> measures how close the actual text output is to the reference (1.0 = identical).</p>')
    for fam_name, keys in family_variants.items():
        valid_keys = [k for k in keys if k in agg]
        if len(valid_keys) < 1:
            continue
        parts.append(f"<h3>{html.escape(fam_name)}</h3>")
        parts.append(_family_comparison(fam_name, valid_keys, agg, qual, pwr, variant_info))
        if len(valid_keys) >= 2:
            parts.append(_family_charts(fam_name, valid_keys, agg, qual, variant_info))
    parts.append("</section>")

    # Per-prompt breakdown
    parts.append('<section>')
    parts.append("<h2>Per-Prompt Breakdown</h2>")
    parts.append('<p class="section-desc">Performance broken down by prompt type. '
                 'Short prompts (like "What is the capital of France?") test latency — how fast the model starts responding. '
                 'Long prompts (like essay writing or code generation) test sustained throughput. '
                 '<b>CV%</b> shows measurement consistency — if it\'s over 10%, something may have interfered (background processes, thermal throttling). '
                 'The <b>95% CI</b> is the confidence interval for decode speed.</p>')
    parts.append(_prompt_table(prompt_data, variant_info))
    parts.append("</section>")

    # Power details
    if pwr:
        parts.append('<section>')
        parts.append("<h2>Power</h2>")
        parts.append('<p class="section-desc">Energy consumption during inference, measured via Apple Silicon power counters. '
                     '<b>Avg W</b> is the total power draw during the benchmark window. '
                     '<b>CPU/GPU</b> shows where the power is going — GPU dominates during inference, '
                     'DRAM is significant for larger models that need more memory bandwidth. '
                     'Lower watts for the same throughput means better efficiency and less battery drain.</p>')
        parts.append(_power_table(pwr, variant_info))
        parts.append("</section>")

    # Tool calling
    if tc:
        parts.append('<section>')
        parts.append("<h2>Tool Calling</h2>")
        parts.append('<p class="section-desc">Tests whether the model can act as an AI agent by calling the right tools with the right parameters. '
                     'This matters for apps like OpenClaw, where the model needs to invoke functions (search the web, send emails, manage files) '
                     'based on natural language instructions. '
                     '<b>Function accuracy</b> — did it pick the right tool? '
                     '<b>Param accuracy</b> — did it extract the correct arguments from the user\'s message? '
                     '<b>JSON valid</b> — did it produce parseable structured output? '
                     '<b>Refusal</b> — did it correctly avoid calling a tool when none was needed?</p>')
        parts.append(_tool_calling_table(tc, variant_info))
        parts.append("</section>")

    # Embed raw JSON for reference
    parts.append('<section>')
    parts.append('<details><summary>Raw JSON</summary>')
    parts.append('<pre class="json-dump">')
    raw = {
        "timestamp": session.timestamp,
        "system_info": sys,
        "config_snapshot": cfg,
        "aggregated": agg,
        "quality": qual,
        "power": pwr,
    }
    parts.append(html.escape(json.dumps(raw, indent=2, default=str)))
    parts.append("</pre></details>")
    parts.append("</section>")

    parts.append(_FOOTER)
    return "\n".join(parts)


def _info_card(label: str, value: str) -> str:
    return (
        f'<div class="info-card">'
        f'<span class="info-label">{html.escape(label)}</span>'
        f'<span class="info-value">{html.escape(value)}</span>'
        f"</div>"
    )


# Column help text for tooltips
_HELP = {
    "JSON Valid": "Percentage of tool-calling scenarios where the model produced parseable structured output (JSON function call). 100% means the model always formats tool calls correctly.",
    "Function Acc.": "Percentage of scenarios where the model selected the correct tool/function from the available options.",
    "Param Acc.": "Percentage of scenarios where the model extracted the correct parameters from the user's natural language message.",
    "Refusal Acc.": "Percentage of scenarios where the model correctly chose NOT to call a tool when the request didn't need one (e.g., casual conversation).",
    "Overall": "Combined accuracy across all tool calling scenarios — function selection + correct refusals.",
    "Tools": "Tool calling accuracy — how often the model correctly selects the right function and extracts the right parameters from natural language. Tests single-tool, multi-choice, parameter extraction, multi-step, and refusal scenarios. Critical for AI agent use cases like OpenClaw. (higher is better)",
    "Prefill": "Prompt processing speed in tokens/sec — how fast the model reads the input (higher is better). Scales with input length.",
    "Decode": "Token generation speed in tokens/sec — how fast the model produces output (higher is better). This is the sustained throughput.",
    "Model": "HuggingFace repo or local model directory name",
    "Quant": "Quantization level — bf16/fp16 is full precision, 8bit and 4bit reduce memory at the cost of quality",
    "TTFT (ms)": "Time To First Token — latency from prompt submission to first generated token (lower is better)",
    "tok/s": "Tokens per second — generation throughput after first token (higher is better)",
    "tok/W": "Tokens per watt — energy efficiency, computed as median tok/s divided by average power draw (higher is better)",
    "Mem (MB)": "Peak GPU memory usage during generation in megabytes (lower means the model fits on smaller devices)",
    "Watts": "Average combined power draw (CPU + GPU + DRAM) during inference, measured via zeus-ml",
    "Perplexity": "Intrinsic language model quality on WikiText-2 sample — measures how well the model predicts text (lower is better)",
    "MMLU": "Accuracy on a 100-question multiple-choice knowledge benchmark spanning STEM, humanities, social science, and general knowledge (higher is better)",
    "vs ref": "Performance relative to the reference (highest precision) variant — e.g., 1.5x means 50% faster or 50% more memory",
    "PPL delta": "Perplexity change vs reference variant — positive means worse quality; <2% is negligible, >5% is significant",
    "Output Sim.": "Token-level F1 similarity between this variant's output and the reference variant's output (1.0 = identical, measures quantization drift)",
    "Prompt": "The prompt category used for this measurement",
    "95% CI": "95% confidence interval for the median tok/s — narrower intervals mean more reliable measurements",
    "CV%": "Coefficient of Variation — standard deviation as a percentage of the mean; >10% is flagged as unreliable (may indicate thermal throttling or background load)",
    "N": "Number of measured runs (excluding warmup) used to compute statistics",
    "Avg W": "Average combined power draw in watts across the full measurement window for this variant",
    "Total J": "Total energy consumed in joules during the measurement window",
    "Duration (s)": "Wall-clock time of the measurement window in seconds",
    "CPU (W)": "Average CPU power draw in watts during inference",
    "GPU (W)": "Average GPU power draw in watts during inference",
    "Family": "Model family name — variants within a family differ only by quantization level",
}


def _th(label: str, cls: str = "") -> str:
    """Render a <th> with a CSS tooltip from _HELP."""
    tip = _HELP.get(label, "")
    cls_attr = f' class="{cls}"' if cls else ""
    tip_attr = f' data-tip="{html.escape(tip)}"' if tip else ""
    return f"<th{cls_attr}{tip_attr}>{html.escape(label)}</th>"


def _short_name(key: str, variant_info: dict[str, dict]) -> str:
    info = variant_info.get(key, {})
    repo = info.get("repo", key.split("|")[0])
    return repo.split("/")[-1]


def _summary_table(
    agg: dict, qual: dict, pwr: dict, tc: dict, variant_info: dict[str, dict]
) -> str:
    rows: list[str] = []
    for key, metrics in agg.items():
        info = variant_info.get(key, {})
        name = _short_name(key, variant_info)
        quant = info.get("quant", key.split("|")[-1])

        ttft = _v(metrics.get("ttft_ms", {}), "median")
        prefill = _v(metrics.get("prefill_tps", {}), "median")
        decode = _v(metrics.get("decode_tps", {}), "median")
        tpw = _v(metrics.get("tokens_per_watt", {}), "median")
        mem = _v(metrics.get("peak_memory_bytes", {}), "median")
        mem_mb = mem / (1024 ** 2) if mem else None

        q = qual.get(key, {})
        ppl = q.get("perplexity")
        mmlu = q.get("mmlu_accuracy")

        p = pwr.get(key, {})
        watts = p.get("avg_watts")

        t = tc.get(key, {}) if tc else {}
        tool_acc = t.get("overall_accuracy")

        rows.append(
            f"<tr>"
            f"<td>{html.escape(name)}</td>"
            f'<td class="quant-badge"><span class="badge badge-{quant}">{html.escape(quant)}</span></td>'
            f'<td class="num">{_fmt(ttft, ".1f")}</td>'
            f'<td class="num">{_fmt(prefill, ".1f")}</td>'
            f'<td class="num">{_fmt(decode, ".1f")}</td>'
            f'<td class="num">{_fmt(tpw, ".2f")}</td>'
            f'<td class="num">{_fmt(mem_mb, ".0f")}</td>'
            f'<td class="num">{_fmt(watts, ".1f")}</td>'
            f'<td class="num">{_fmt(ppl, ".2f")}</td>'
            f'<td class="num">{_fmt(mmlu * 100 if mmlu is not None else None, ".1f", "%")}</td>'
            f'<td class="num">{_fmt(tool_acc * 100 if tool_acc is not None else None, ".1f", "%")}</td>'
            f"</tr>"
        )

    return (
        '<table class="data-table sortable">'
        "<thead><tr>"
        f'{_th("Model")}{_th("Quant")}'
        f'{_th("TTFT (ms)", "num")}'
        f'{_th("Prefill", "num")}'
        f'{_th("Decode", "num")}'
        f'{_th("tok/W", "num")}'
        f'{_th("Mem (MB)", "num")}'
        f'{_th("Watts", "num")}'
        f'{_th("Perplexity", "num")}'
        f'{_th("MMLU", "num")}'
        f'{_th("Tools", "num")}'
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _family_comparison(
    fam_name: str,
    keys: list[str],
    agg: dict,
    qual: dict,
    pwr: dict,
    variant_info: dict[str, dict],
) -> str:
    ref_quant = variant_info.get(keys[0], {}).get("reference", "")
    ref_key = next((k for k in keys if variant_info.get(k, {}).get("quant") == ref_quant), None)

    ref_tps = _v(agg.get(ref_key, {}).get("tokens_per_sec", {}), "median") if ref_key else None
    ref_mem = _v(agg.get(ref_key, {}).get("peak_memory_bytes", {}), "median") if ref_key else None
    ref_ppl = qual.get(ref_key, {}).get("perplexity") if ref_key else None

    rows: list[str] = []
    for key in keys:
        info = variant_info.get(key, {})
        quant = info.get("quant", "?")
        metrics = agg.get(key, {})
        q = qual.get(key, {})

        tps = _v(metrics.get("tokens_per_sec", {}), "median")
        mem = _v(metrics.get("peak_memory_bytes", {}), "median")
        ppl = q.get("perplexity")

        speed_vs = f"{tps / ref_tps:.2f}x" if tps and ref_tps else "—"
        mem_vs = f"{mem / ref_mem:.2f}x" if mem and ref_mem else "—"
        ppl_delta = ""
        if ppl and ref_ppl and ref_ppl > 0:
            change = ((ppl - ref_ppl) / ref_ppl) * 100
            sign = "+" if change >= 0 else ""
            cls = "delta-bad" if change > 5 else "delta-ok" if change < 2 else "delta-mid"
            ppl_delta = f'<span class="{cls}">{sign}{change:.1f}%</span>'
        elif ppl:
            ppl_delta = "ref"

        sim = q.get("output_similarity", {})
        avg_sim = sum(sim.values()) / len(sim) if sim else None

        is_ref = quant == ref_quant
        ref_marker = ' <span class="ref-tag">ref</span>' if is_ref else ""

        rows.append(
            f"<tr{'  class=\"ref-row\"' if is_ref else ''}>"
            f'<td><span class="badge badge-{quant}">{html.escape(quant)}</span>{ref_marker}</td>'
            f'<td class="num">{_fmt(tps, ".1f")}</td>'
            f'<td class="num">{speed_vs}</td>'
            f'<td class="num">{_fmt(mem and mem / 1024**2, ".0f")}</td>'
            f'<td class="num">{mem_vs}</td>'
            f'<td class="num">{_fmt(ppl, ".2f")}</td>'
            f'<td class="num">{ppl_delta or "—"}</td>'
            f'<td class="num">{_fmt(avg_sim, ".3f")}</td>'
            f"</tr>"
        )

    return (
        '<table class="data-table">'
        "<thead><tr>"
        f'{_th("Quant")}'
        f'{_th("tok/s", "num")}'
        f'{_th("vs ref", "num")}'
        f'{_th("Mem (MB)", "num")}'
        f'{_th("vs ref", "num")}'
        f'{_th("Perplexity", "num")}'
        f'{_th("PPL delta", "num")}'
        f'{_th("Output Sim.", "num")}'
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _family_charts(
    fam_name: str,
    keys: list[str],
    agg: dict,
    qual: dict,
    variant_info: dict[str, dict],
) -> str:
    """Render inline SVG bar charts for a family."""
    quants = []
    tps_vals = []
    ppl_vals = []
    mem_vals = []
    for key in keys:
        info = variant_info.get(key, {})
        quants.append(info.get("quant", "?"))
        tps_vals.append(_v(agg.get(key, {}).get("tokens_per_sec", {}), "median") or 0)
        ppl_vals.append(qual.get(key, {}).get("perplexity") or 0)
        mem_vals.append((_v(agg.get(key, {}).get("peak_memory_bytes", {}), "median") or 0) / (1024**2))

    charts: list[str] = ['<div class="chart-row">']

    charts.append(_svg_bar_chart(
        f"Throughput (tok/s)", quants, tps_vals, "#4f8cff", "{:.1f}"
    ))
    if any(v > 0 for v in ppl_vals):
        charts.append(_svg_bar_chart(
            f"Perplexity (lower is better)", quants, ppl_vals, "#e8734a", "{:.1f}"
        ))
    if any(v > 0 for v in mem_vals):
        charts.append(_svg_bar_chart(
            f"Memory (MB)", quants, mem_vals, "#7c4dff", "{:.0f}"
        ))

    charts.append("</div>")
    return "\n".join(charts)


def _svg_bar_chart(
    title: str,
    labels: list[str],
    values: list[float],
    color: str,
    fmt: str,
) -> str:
    n = len(labels)
    if n == 0:
        return ""
    max_val = max(values) if max(values) > 0 else 1

    w = 260
    bar_area_h = 130
    top_pad = 30
    bottom_pad = 30
    h = top_pad + bar_area_h + bottom_pad
    bar_w = min(50, (w - 40) // n - 10)
    spacing = (w - 20) / n
    start_x = 10 + spacing / 2

    bars: list[str] = []
    for i, (label, val) in enumerate(zip(labels, values)):
        bh = (val / max_val) * (bar_area_h - 20) if max_val > 0 else 0
        x = start_x + i * spacing - bar_w / 2
        y = top_pad + bar_area_h - bh
        bars.append(
            f'<rect x="{x:.0f}" y="{y:.0f}" width="{bar_w}" height="{bh:.0f}" '
            f'fill="{color}" rx="3" opacity="0.85"/>'
        )
        bars.append(
            f'<text x="{x + bar_w/2:.0f}" y="{y - 4:.0f}" '
            f'text-anchor="middle" class="bar-val">{fmt.format(val)}</text>'
        )
        bars.append(
            f'<text x="{x + bar_w/2:.0f}" y="{h - 8:.0f}" '
            f'text-anchor="middle" class="bar-label">{html.escape(label)}</text>'
        )

    return (
        f'<div class="chart-card">'
        f'<svg viewBox="0 0 {w} {h}" class="bar-chart">'
        f'<text x="{w/2}" y="16" text-anchor="middle" class="chart-title">{html.escape(title)}</text>'
        f'{"".join(bars)}'
        f"</svg></div>"
    )


def _collect_prompt_data(
    runs: list[dict],
) -> dict[tuple[str, str, str], list[dict]]:
    """Group non-warmup runs by (repo, quant, prompt_id)."""
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for run in runs:
        if not run.get("is_warmup", False):
            key = (run["variant_repo"], run["variant_quant"], run["prompt_id"])
            grouped[key].append(run)
    return grouped


def _prompt_table(
    prompt_data: dict[tuple[str, str, str], list[dict]],
    variant_info: dict[str, dict],
) -> str:
    from bench.stats import aggregate

    rows: list[str] = []
    for (repo, quant, prompt_id), runs in prompt_data.items():
        name = repo.split("/")[-1]
        ttft_values = [r["ttft_ms"] for r in runs]
        prefill_values = [r.get("prefill_tps", 0) for r in runs if r.get("prefill_tps", 0) > 0]
        decode_values = [r.get("decode_tps", 0) for r in runs if r.get("decode_tps", 0) > 0]
        ttft_agg = aggregate(ttft_values)
        prefill_agg = aggregate(prefill_values) if prefill_values else None
        decode_agg = aggregate(decode_values) if decode_values else None

        cv_cls = "warn" if decode_agg and decode_agg.unreliable else ""

        rows.append(
            f"<tr>"
            f"<td>{html.escape(name)}</td>"
            f'<td><span class="badge badge-{quant}">{html.escape(quant)}</span></td>'
            f"<td>{html.escape(prompt_id)}</td>"
            f'<td class="num">{ttft_agg.median:.1f}</td>'
            f'<td class="num">{_fmt(prefill_agg.median if prefill_agg else None, ".1f")}</td>'
            f'<td class="num">{_fmt(decode_agg.median if decode_agg else None, ".1f")}</td>'
            f'<td class="num">{decode_agg.ci_str if decode_agg else "—"}</td>'
            f'<td class="num {cv_cls}">{decode_agg.cv_percent:.1f}%' + (' !' if decode_agg and decode_agg.unreliable else '') + '</td>' if decode_agg else f'<td class="num">—</td>'
            f'<td class="num">{len(runs)}</td>'
            f"</tr>"
        )

    return (
        '<table class="data-table">'
        "<thead><tr>"
        f'{_th("Model")}{_th("Quant")}{_th("Prompt")}'
        f'{_th("TTFT (ms)", "num")}'
        f'{_th("Prefill", "num")}'
        f'{_th("Decode", "num")}'
        f'{_th("95% CI", "num")}'
        f'{_th("CV%", "num")}'
        f'{_th("N", "num")}'
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _tool_calling_table(tc: dict, variant_info: dict[str, dict]) -> str:
    rows: list[str] = []
    for key, data in tc.items():
        name = _short_name(key, variant_info)
        info = variant_info.get(key, {})
        quant = info.get("quant", key.split("|")[-1])

        rows.append(
            f"<tr>"
            f"<td>{html.escape(name)}</td>"
            f'<td><span class="badge badge-{quant}">{html.escape(quant)}</span></td>'
            f'<td class="num">{_fmt(data.get("json_valid_rate", 0) * 100, ".1f", "%")}</td>'
            f'<td class="num">{_fmt(data.get("function_accuracy", 0) * 100, ".1f", "%")}</td>'
            f'<td class="num">{_fmt(data.get("param_accuracy", 0) * 100, ".1f", "%")}</td>'
            f'<td class="num">{_fmt(data.get("refusal_accuracy", 0) * 100, ".1f", "%")}</td>'
            f'<td class="num">{_fmt(data.get("overall_accuracy", 0) * 100, ".1f", "%")}</td>'
            f'<td class="num">{data.get("total", 0)}</td>'
            f"</tr>"
        )

    return (
        '<table class="data-table sortable">'
        "<thead><tr>"
        f'{_th("Model")}{_th("Quant")}'
        f'{_th("JSON Valid", "num")}'
        f'{_th("Function Acc.", "num")}'
        f'{_th("Param Acc.", "num")}'
        f'{_th("Refusal Acc.", "num")}'
        f'{_th("Overall", "num")}'
        f'{_th("N", "num")}'
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _power_table(pwr: dict, variant_info: dict[str, dict]) -> str:
    rows: list[str] = []
    for key, data in pwr.items():
        name = _short_name(key, variant_info)
        info = variant_info.get(key, {})
        quant = info.get("quant", key.split("|")[-1])
        comps = data.get("components", {})

        rows.append(
            f"<tr>"
            f"<td>{html.escape(name)}</td>"
            f'<td><span class="badge badge-{quant}">{html.escape(quant)}</span></td>'
            f'<td class="num">{_fmt(data.get("avg_watts"), ".1f")}</td>'
            f'<td class="num">{_fmt(data.get("total_joules"), ".1f")}</td>'
            f'<td class="num">{_fmt(data.get("duration_s"), ".1f")}</td>'
            f'<td class="num">{_fmt(comps.get("cpu"), ".1f")}</td>'
            f'<td class="num">{_fmt(comps.get("gpu"), ".1f")}</td>'
            f"</tr>"
        )

    return (
        '<table class="data-table">'
        "<thead><tr>"
        f'{_th("Model")}{_th("Quant")}'
        f'{_th("Avg W", "num")}'
        f'{_th("Total J", "num")}'
        f'{_th("Duration (s)", "num")}'
        f'{_th("CPU (W)", "num")}'
        f'{_th("GPU (W)", "num")}'
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HEADER = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MacOS-MLX-Benchmark — MLX Inference Report</title>
<style>
:root {
  --bg: #0d1117;
  --surface: #161b22;
  --surface2: #1c2128;
  --border: #30363d;
  --text: #e6edf3;
  --text2: #8b949e;
  --accent: #4f8cff;
  --green: #3fb950;
  --orange: #e8734a;
  --purple: #7c4dff;
  --red: #f85149;
  --yellow: #d29922;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', system-ui, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
}
h1 {
  font-size: 1.75rem;
  font-weight: 600;
  margin-bottom: 0.25rem;
}
h1 small {
  font-size: 0.875rem;
  color: var(--text2);
  font-weight: 400;
}
h2 {
  font-size: 1.25rem;
  font-weight: 600;
  margin: 2rem 0 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}
h3 {
  font-size: 1.05rem;
  font-weight: 600;
  margin: 1.5rem 0 0.75rem;
  color: var(--accent);
}
section { margin-bottom: 1.5rem; }
.timestamp { color: var(--text2); font-size: 0.85rem; margin: 0.25rem 0 1rem; }
.section-desc { color: var(--text2); font-size: 0.85rem; line-height: 1.5; margin-bottom: 1rem; max-width: 900px; }
.section-desc b { color: var(--text); }

/* Info grid */
.info-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-bottom: 0.75rem;
}
.info-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.6rem 1rem;
  display: flex;
  flex-direction: column;
  min-width: 120px;
}
.info-label { font-size: 0.75rem; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; }
.info-value { font-size: 0.95rem; font-weight: 600; }

/* Tables */
.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
  margin-bottom: 1rem;
  overflow: visible;
}
.data-table th, .data-table td {
  padding: 0.5rem 0.75rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
}
.data-table th {
  background: var(--surface);
  color: var(--text2);
  font-weight: 600;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.data-table thead { position: relative; z-index: 10; }
.data-table th.num, .data-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
.data-table tbody tr:hover { background: var(--surface2); }
.data-table tr.ref-row { background: var(--surface); }

/* Badges */
.badge {
  display: inline-block;
  font-size: 0.75rem;
  font-weight: 600;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  letter-spacing: 0.02em;
}
.badge-bf16, .badge-fp16 { background: #1a3a2a; color: var(--green); }
.badge-8bit { background: #1a2640; color: var(--accent); }
.badge-4bit { background: #2a1a35; color: var(--purple); }
.ref-tag {
  font-size: 0.65rem;
  color: var(--text2);
  border: 1px solid var(--border);
  border-radius: 3px;
  padding: 0 0.3rem;
  margin-left: 0.4rem;
  vertical-align: middle;
}

/* Deltas */
.delta-ok { color: var(--green); }
.delta-mid { color: var(--yellow); }
.delta-bad { color: var(--red); }
.warn { color: var(--yellow); }

/* Charts */
.chart-row {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  margin: 1rem 0;
}
.chart-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.75rem;
  flex: 1;
  min-width: 220px;
  max-width: 320px;
}
.bar-chart { width: 100%; height: auto; }
.bar-chart .chart-title { fill: var(--text2); font-size: 11px; font-weight: 600; }
.bar-chart .bar-val { fill: var(--text); font-size: 10px; font-weight: 600; }
.bar-chart .bar-label { fill: var(--text2); font-size: 11px; font-weight: 600; }

/* JSON dump */
.json-dump {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  font-size: 0.75rem;
  overflow-x: auto;
  max-height: 400px;
  overflow-y: auto;
  color: var(--text2);
}
details summary {
  cursor: pointer;
  color: var(--text2);
  font-size: 0.85rem;
}
details summary:hover { color: var(--text); }

/* Tooltips on headers */
th[data-tip] {
  cursor: help;
  text-decoration: underline dotted var(--text2);
  text-underline-offset: 3px;
}
th[data-tip]:hover { color: var(--text); }
#tooltip {
  position: fixed;
  background: #000000ee;
  color: #f0f0f0;
  font-size: 0.75rem;
  font-weight: 400;
  padding: 0.5rem 0.75rem;
  border-radius: 6px;
  max-width: 280px;
  line-height: 1.4;
  z-index: 9999;
  pointer-events: none;
  box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  display: none;
}

/* Sortable table header */
.sortable th { cursor: pointer; user-select: none; }
.sortable th:hover { color: var(--text); }
.sortable th::after { content: ' ↕'; font-size: 0.7em; opacity: 0.4; }

@media (max-width: 768px) {
  body { padding: 1rem; }
  .data-table { font-size: 0.8rem; }
  .data-table th, .data-table td { padding: 0.35rem 0.5rem; }
}
</style>
</head>
<body>
<h1>MacOS-MLX-Benchmark <small>MLX Inference Report</small></h1>
<div id="tooltip"></div>
"""

_FOOTER = """\
<script>
// Tooltips on th[data-tip]
const tip = document.getElementById('tooltip');
document.querySelectorAll('th[data-tip]').forEach(th => {
  th.addEventListener('mouseenter', e => {
    tip.textContent = th.dataset.tip;
    tip.style.display = 'block';
    const r = th.getBoundingClientRect();
    tip.style.left = Math.min(r.left, window.innerWidth - 300) + 'px';
    tip.style.top = (r.bottom + 6) + 'px';
  });
  th.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
});

// Minimal sortable table
document.querySelectorAll('.sortable th').forEach(th => {
  th.addEventListener('click', () => {
    const table = th.closest('table');
    const idx = Array.from(th.parentNode.children).indexOf(th);
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const asc = th.dataset.sort !== 'asc';
    th.parentNode.querySelectorAll('th').forEach(t => delete t.dataset.sort);
    th.dataset.sort = asc ? 'asc' : 'desc';
    rows.sort((a, b) => {
      let av = a.children[idx]?.textContent.replace(/[^0-9.\\-]/g, '') || '';
      let bv = b.children[idx]?.textContent.replace(/[^0-9.\\-]/g, '') || '';
      let an = parseFloat(av), bn = parseFloat(bv);
      if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
      return asc ? av.localeCompare(bv) : bv.localeCompare(av);
    });
    rows.forEach(r => tbody.appendChild(r));
  });
});
</script>
</body>
</html>
"""
