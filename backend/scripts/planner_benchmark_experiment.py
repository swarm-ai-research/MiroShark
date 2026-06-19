"""Planner benchmark: Saez vs RL vs heuristic vs bandit (bd-cjx).

Renders the N=100 planner-type sweep emitted by ``sweep_gtb`` into a
matplotlib-free SVG (three stacked panels: welfare, Gini, tax revenue),
each a row of one bar per planner with a p10-p90 whisker. Categorical
axis, so bars-with-whiskers rather than the continuous band chart used
by ``audit_sweep_experiment``.

Usage::

    # 1. Produce the data (see FINDINGS.md / CLAUDE.md for the canonical cmd)
    uv run python -m scripts.sweep_gtb \
        worlds/gather_trade_build/scenarios/ai_economist_saez.yaml \
        --n-seeds 100 --epochs 30 \
        --sweep 'planner.planner_type=heuristic,bandit,saez,rl' \
        --output runs/planner_benchmark

    # 2. Render the chart from the aggregate CSV
    uv run python -m scripts.planner_benchmark_experiment \
        --run-dir runs/planner_benchmark

Reads ``aggregate_final.csv`` (one row per planner cell, final-epoch
mean/median/p10/p90) and writes ``benchmark.svg`` next to it.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

# Display order + palette, keyed by planner_type.
PLANNERS = ("heuristic", "bandit", "saez", "rl")
COLORS = {
    "heuristic": "#7ab8ff",
    "bandit": "#9b8cff",
    "saez": "#3aa55c",
    "rl": "#e8833a",
}
PANELS = (
    ("welfare", "welfare (eq_times_prod)"),
    ("gini_coefficient", "Gini (wealth)"),
    ("total_tax_revenue", "tax revenue"),
)


def _load(run_dir: Path) -> Dict[str, dict]:
    """Map planner_type -> its aggregate_final row."""
    path = run_dir / "aggregate_final.csv"
    rows = list(csv.DictReader(path.open()))
    by_planner: Dict[str, dict] = {}
    for r in rows:
        # The sweep prefixes the swept key with ``override_``.
        pt = r.get("override_planner.planner_type") or r.get("planner.planner_type")
        if pt is not None:
            by_planner[pt] = r
    return by_planner


def _panel(
    field: str,
    title: str,
    by_planner: Dict[str, dict],
    x0: float,
    y0: float,
    w: float,
    h: float,
) -> List[str]:
    """One metric panel: a row of mean bars with p10-p90 whiskers."""
    means = [float(by_planner[p][f"{field}_mean"]) for p in PLANNERS]
    p10s = [float(by_planner[p][f"{field}_p10"]) for p in PLANNERS]
    p90s = [float(by_planner[p][f"{field}_p90"]) for p in PLANNERS]

    vmax = max(p90s) * 1.08 or 1.0
    vmin = min(0.0, min(p10s))

    def sy(v: float) -> float:
        return y0 + h - (v - vmin) / (vmax - vmin) * h

    parts: List[str] = [
        f'<text class="ptitle" x="{x0}" y="{y0 - 8:.1f}">{title}</text>',
        f'<line class="axis" x1="{x0}" y1="{sy(vmin):.1f}" '
        f'x2="{x0 + w}" y2="{sy(vmin):.1f}" />',
    ]
    # Zero/baseline gridline if the range crosses it usefully.
    n = len(PLANNERS)
    slot = w / n
    bar_w = slot * 0.5
    for i, p in enumerate(PLANNERS):
        cx = x0 + slot * (i + 0.5)
        bx = cx - bar_w / 2
        top = sy(means[i])
        base = sy(vmin)
        parts.append(
            f'<rect x="{bx:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
            f'height="{base - top:.1f}" fill="{COLORS[p]}" fill-opacity="0.85" />'
        )
        # p10-p90 whisker.
        parts.append(
            f'<line x1="{cx:.1f}" y1="{sy(p10s[i]):.1f}" x2="{cx:.1f}" '
            f'y2="{sy(p90s[i]):.1f}" stroke="#333" stroke-width="1.5" />'
        )
        for vv in (p10s[i], p90s[i]):
            parts.append(
                f'<line x1="{cx - 5:.1f}" y1="{sy(vv):.1f}" x2="{cx + 5:.1f}" '
                f'y2="{sy(vv):.1f}" stroke="#333" stroke-width="1.5" />'
            )
        parts.append(
            f'<text class="val" x="{cx:.1f}" y="{top - 6:.1f}" '
            f'text-anchor="middle">{means[i]:.1f}</text>'
        )
        parts.append(
            f'<text class="plabel" x="{cx:.1f}" y="{base + 14:.1f}" '
            f'text-anchor="middle">{p}</text>'
        )
    return parts


def render_svg(by_planner: Dict[str, dict], out: Path) -> None:
    missing = [p for p in PLANNERS if p not in by_planner]
    if missing:
        raise SystemExit(f"aggregate_final.csv missing planner cells: {missing}")

    width, height = 760, 720
    pad_l, pad_t, pad_r = 70, 50, 30
    panel_h = 170
    panel_gap = 50
    plot_w = width - pad_l - pad_r

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="system-ui,sans-serif">',
        "<style>"
        ".title{font-size:16px;font-weight:600;fill:#111}"
        ".sub{font-size:11px;fill:#666}"
        ".ptitle{font-size:13px;font-weight:600;fill:#222}"
        ".plabel{font-size:11px;fill:#444}"
        ".val{font-size:10px;fill:#222}"
        ".axis{stroke:#999;stroke-width:1}"
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#fff" />',
        f'<text class="title" x="{pad_l}" y="24">Planner benchmark — '
        f'welfare / equity / revenue (N=100, 30 epochs)</text>',
        f'<text class="sub" x="{pad_l}" y="40">ai_economist_saez · '
        f'whiskers = p10–p90 across 100 seeds · bar = mean</text>',
    ]
    y = pad_t + 24
    for field, title in PANELS:
        parts.extend(_panel(field, title, by_planner, pad_l, y, plot_w, panel_h))
        y += panel_h + panel_gap
    parts.append("</svg>")
    out.write_text("\n".join(parts))
    print(f"wrote {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=Path("runs/planner_benchmark"))
    args = ap.parse_args(argv)
    by_planner = _load(args.run_dir)
    render_svg(by_planner, args.run_dir / "benchmark.svg")


if __name__ == "__main__":
    main()
