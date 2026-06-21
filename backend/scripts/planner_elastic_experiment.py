"""Elastic-workforce planner benchmark (bd-anv, follow-up to bd-cjx).

Renders the 2D sweep ``planner_type × utility.labor_coeff`` into a
matplotlib-free SVG: two panels (welfare, tax revenue) vs labor_coeff,
one line per planner. The welfare panel carries p10-p90 bands; the
question it answers visually is "do the four planner lines fan apart as
the workforce gets more labor-elastic?" (They don't — that's the
finding.) The revenue panel shows the one separation that does exist:
RL's line sits far above the other three at every elasticity level.

Usage::

    uv run python -m scripts.sweep_gtb \
        worlds/gather_trade_build/scenarios/ai_economist_saez.yaml \
        --n-seeds 100 --epochs 30 \
        --sweep 'planner.planner_type=heuristic,bandit,saez,rl' \
        --sweep 'utility.labor_coeff=0.15,0.3,0.5,0.8' \
        --output runs/planner_elastic

    uv run python -m scripts.planner_elastic_experiment --run-dir runs/planner_elastic

Reads ``aggregate_final.csv`` and writes ``elastic.svg`` next to it.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

PLANNERS = ("heuristic", "bandit", "saez", "rl")
COLORS = {
    "heuristic": "#7ab8ff",
    "bandit": "#9b8cff",
    "saez": "#3aa55c",
    "rl": "#e8833a",
}
PANELS = (("welfare", "welfare (eq_times_prod)", True),
          ("total_tax_revenue", "tax revenue", False))


def _load(run_dir: Path):
    """Return (sorted labor_coeff list, {planner: {lc: {metric_stat: val}}})."""
    rows = list(csv.DictReader((run_dir / "aggregate_final.csv").open()))
    lcs = sorted({float(r["override_utility.labor_coeff"]) for r in rows})
    data: Dict[str, Dict[float, dict]] = {p: {} for p in PLANNERS}
    for r in rows:
        p = r["override_planner.planner_type"]
        lc = float(r["override_utility.labor_coeff"])
        if p in data:
            data[p][lc] = r
    return lcs, data


def _panel(field, title, with_band, lcs, data, x0, y0, w, h) -> List[str]:
    series: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    for p in PLANNERS:
        means = [float(data[p][lc][f"{field}_mean"]) for lc in lcs]
        p10 = [float(data[p][lc][f"{field}_p10"]) for lc in lcs]
        p90 = [float(data[p][lc][f"{field}_p90"]) for lc in lcs]
        series[p] = (means, p10, p90)

    hi = max(max(s[2]) for s in series.values()) if with_band else max(
        max(s[0]) for s in series.values())
    vmax = hi * 1.10 or 1.0
    vmin = 0.0
    n = len(lcs)

    def sx(i: int) -> float:
        return x0 + (w * (i / (n - 1)) if n > 1 else w / 2)

    def sy(v: float) -> float:
        return y0 + h - (v - vmin) / (vmax - vmin) * h

    parts: List[str] = [
        f'<text class="ptitle" x="{x0}" y="{y0 - 8:.1f}">{title}</text>',
        f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + h}" />',
        f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}" />',
    ]
    # y ticks
    for f in (0.0, 0.5, 1.0):
        v = vmin + f * (vmax - vmin)
        yy = sy(v)
        parts.append(f'<text class="tick" x="{x0 - 6:.1f}" y="{yy + 3:.1f}" '
                     f'text-anchor="end">{v:.0f}</text>')
        parts.append(f'<line class="grid" x1="{x0}" y1="{yy:.1f}" '
                     f'x2="{x0 + w}" y2="{yy:.1f}" />')
    # x ticks
    for i, lc in enumerate(lcs):
        parts.append(f'<text class="tick" x="{sx(i):.1f}" y="{y0 + h + 14:.1f}" '
                     f'text-anchor="middle">{lc:g}</text>')
    # optional p10-p90 bands (welfare panel)
    if with_band:
        for p in PLANNERS:
            _, p10, p90 = series[p]
            top = " ".join(f"{sx(i):.1f},{sy(p90[i]):.1f}" for i in range(n))
            bot = " ".join(f"{sx(i):.1f},{sy(p10[i]):.1f}"
                           for i in reversed(range(n)))
            parts.append(f'<polygon points="{top} {bot}" fill="{COLORS[p]}" '
                         f'fill-opacity="0.08" stroke="none" />')
    # lines + markers
    for p in PLANNERS:
        means = series[p][0]
        poly = " ".join(f"{sx(i):.1f},{sy(means[i]):.1f}" for i in range(n))
        parts.append(f'<polyline points="{poly}" fill="none" '
                     f'stroke="{COLORS[p]}" stroke-width="2" />')
        for i in range(n):
            parts.append(f'<circle cx="{sx(i):.1f}" cy="{sy(means[i]):.1f}" '
                         f'r="2.5" fill="{COLORS[p]}" />')
    parts.append(f'<text class="alabel" x="{x0 + w / 2:.1f}" '
                 f'y="{y0 + h + 30:.1f}" text-anchor="middle">'
                 f'utility.labor_coeff (worker tax-elasticity →)</text>')
    return parts


def render_svg(lcs, data, out: Path) -> None:
    width, height = 720, 560
    pad_l, pad_t, pad_r = 60, 60, 110
    panel_h, gap = 180, 70
    plot_w = width - pad_l - pad_r
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="system-ui,sans-serif">',
        "<style>"
        ".title{font-size:15px;font-weight:600;fill:#111}"
        ".sub{font-size:11px;fill:#666}"
        ".ptitle{font-size:13px;font-weight:600;fill:#222}"
        ".alabel{font-size:11px;fill:#555}"
        ".tick{font-size:10px;fill:#666}"
        ".axis{stroke:#999;stroke-width:1}.grid{stroke:#eee;stroke-width:1}"
        ".leg{font-size:11px;fill:#222}"
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#fff" />',
        f'<text class="title" x="{pad_l}" y="22">Elastic-workforce planner '
        f'benchmark (N=100, 30 epochs)</text>',
        f'<text class="sub" x="{pad_l}" y="38">welfare lines coincide at every '
        f'elasticity; only RL revenue separates · welfare band = p10–p90</text>',
    ]
    # legend
    lx, ly = width - pad_r + 16, pad_t + 6
    for i, p in enumerate(PLANNERS):
        parts.append(f'<line x1="{lx}" y1="{ly + i*18}" x2="{lx+18}" '
                     f'y2="{ly + i*18}" stroke="{COLORS[p]}" stroke-width="3" />')
        parts.append(f'<text class="leg" x="{lx+24}" y="{ly + i*18 + 4}">{p}</text>')
    y = pad_t + 10
    for field, title, band in PANELS:
        parts.extend(_panel(field, title, band, lcs, data, pad_l, y, plot_w, panel_h))
        y += panel_h + gap
    parts.append("</svg>")
    out.write_text("\n".join(parts))
    print(f"wrote {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=Path("runs/planner_elastic"))
    args = ap.parse_args(argv)
    lcs, data = _load(args.run_dir)
    render_svg(lcs, data, args.run_dir / "elastic.svg")


if __name__ == "__main__":
    main()
