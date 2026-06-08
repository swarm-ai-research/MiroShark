#!/usr/bin/env python3
"""Audit-rate sweep across the 0.5 EV-breakeven threshold.

Closes bd-an2. Builds on bd-cec (sweep_gtb harness) to map the
welfare / tax_revenue / catch-rate frontier across 11 audit_probability
cells × 100 seeds × 30 epochs. Writes:

    runs/audit_sweep/aggregate.csv             # raw sweep output
    runs/audit_sweep/aggregate_final.csv       # ditto
    runs/audit_sweep/FINDINGS.md               # narrative + table
    runs/audit_sweep/welfare_tax_frontier.svg  # 2-line chart

Findings (with default fine_multiplier=2.0, underreport=0.5):

* The naive expected-value calculation says evasion is +EV up to
  audit_probability=0.5 (fine_multiplier × catch_prob == 1.0 only
  there). The actual sim shows catches collapse toward zero by
  audit_probability=0.2, because the EvasiveWorkerPolicy ALSO
  reasons about risk — once selection is high enough, it stops
  reporting under, and the planner can't catch anyone.
* Welfare is monotonically DECREASING in audit_probability, despite
  enforcement intuitively being "good". With aggressive auditing,
  workers spend more time being audited and less time producing;
  total tax revenue also falls.
* The sweet spot under default parameters is around
  audit_probability=0.05-0.10 — low enough to keep workers free to
  produce, high enough to deter the largest evasions.
"""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


_BACKEND = Path(__file__).resolve().parent.parent
_SCENARIO = (
    _BACKEND / "worlds" / "gather_trade_build" / "scenarios" / "ai_economist_full.yaml"
)
_DEFAULT_OUTPUT = _BACKEND / "runs" / "audit_sweep"

logger = logging.getLogger("audit_sweep_experiment")


_AUDIT_PROBS = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.65, 0.80, 0.95)


def _run_sweep(output: Path, n_seeds: int, n_epochs: int, steps: int) -> None:
    cmd = [
        sys.executable, "-m", "scripts.sweep_gtb",
        str(_SCENARIO),
        "--n-seeds", str(n_seeds),
        "--epochs", str(n_epochs),
        "--steps", str(steps),
        "--sweep", f"misreporting.audit_probability={','.join(str(p) for p in _AUDIT_PROBS)}",
        "--output", str(output),
    ]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=_BACKEND, check=True)


def _read_finals(csv_path: Path) -> List[dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _frontier_table(rows: List[dict]) -> str:
    """Render the markdown table for FINDINGS.md."""
    out = []
    out.append("| audit_prob | audits | catches | tax_revenue | welfare | gini |")
    out.append("|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        ap = float(r["override_misreporting.audit_probability"])
        au = float(r["total_audits_mean"])
        ca = float(r["total_catches_mean"])
        tr = float(r["total_tax_revenue_mean"])
        we = float(r["welfare_mean"])
        gi = float(r["gini_coefficient_mean"])
        out.append(f"| {ap:.3f} | {au:6.1f} | {ca:6.2f} | {tr:7.2f} | {we:6.2f} | {gi:5.3f} |")
    return "\n".join(out)


def _svg_frontier(rows: List[dict], path: Path) -> None:
    """Hand-rolled SVG line chart of welfare + tax_revenue vs audit_prob."""
    width, height = 720, 360
    pad_l, pad_r, pad_t, pad_b = 60, 60, 40, 50
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    probs = [float(r["override_misreporting.audit_probability"]) for r in rows]
    welfare = [float(r["welfare_mean"]) for r in rows]
    welfare_lo = [float(r["welfare_p10"]) for r in rows]
    welfare_hi = [float(r["welfare_p90"]) for r in rows]
    tax = [float(r["total_tax_revenue_mean"]) for r in rows]
    tax_lo = [float(r["total_tax_revenue_p10"]) for r in rows]
    tax_hi = [float(r["total_tax_revenue_p90"]) for r in rows]

    x_min, x_max = min(probs), max(probs)
    # Two y-axes: welfare on left, tax on right.
    w_min, w_max = min(welfare_lo), max(welfare_hi)
    t_min, t_max = min(tax_lo), max(tax_hi)
    # Pad y-ranges by 10%.
    w_min -= (w_max - w_min) * 0.1
    w_max += (w_max - w_min) * 0.05
    t_min -= (t_max - t_min) * 0.1
    t_max += (t_max - t_min) * 0.05

    def xs(p: float) -> float:
        return pad_l + (p - x_min) / (x_max - x_min) * plot_w if x_max > x_min else pad_l

    def ys_w(v: float) -> float:
        return pad_t + plot_h - (v - w_min) / (w_max - w_min) * plot_h if w_max > w_min else pad_t

    def ys_t(v: float) -> float:
        return pad_t + plot_h - (v - t_min) / (t_max - t_min) * plot_h if t_max > t_min else pad_t

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-monospace, monospace">',
        '<style>'
        '.axis{stroke:#888;stroke-width:1;fill:none}'
        '.grid{stroke:#eee;stroke-width:1;fill:none}'
        '.label{font-size:12px;fill:#444}'
        '.title{font-size:13px;fill:#222;font-weight:600}'
        '.welfare-line{stroke:#3aa55c;stroke-width:2;fill:none}'
        '.welfare-band{fill:#3aa55c;fill-opacity:0.12;stroke:none}'
        '.tax-line{stroke:#7ab8ff;stroke-width:2;fill:none}'
        '.tax-band{fill:#7ab8ff;fill-opacity:0.12;stroke:none}'
        '.tick{font-size:10px;fill:#777}'
        '</style>',
        f'<text class="title" x="{pad_l}" y="20">'
        'GTB audit-rate frontier — welfare + tax revenue vs audit_probability '
        f'(N={rows[0]["n_seeds"]} seeds)'
        '</text>',
    ]

    # Welfare confidence band.
    band_top = " ".join(f"{xs(p):.1f},{ys_w(v):.1f}" for p, v in zip(probs, welfare_hi))
    band_bot = " ".join(f"{xs(p):.1f},{ys_w(v):.1f}" for p, v in zip(reversed(probs), reversed(welfare_lo)))
    parts.append(f'<polygon class="welfare-band" points="{band_top} {band_bot}" />')
    # Tax confidence band.
    band_top = " ".join(f"{xs(p):.1f},{ys_t(v):.1f}" for p, v in zip(probs, tax_hi))
    band_bot = " ".join(f"{xs(p):.1f},{ys_t(v):.1f}" for p, v in zip(reversed(probs), reversed(tax_lo)))
    parts.append(f'<polygon class="tax-band" points="{band_top} {band_bot}" />')

    # Lines.
    welfare_poly = " ".join(f"{xs(p):.1f},{ys_w(v):.1f}" for p, v in zip(probs, welfare))
    tax_poly = " ".join(f"{xs(p):.1f},{ys_t(v):.1f}" for p, v in zip(probs, tax))
    parts.append(f'<polyline class="welfare-line" points="{welfare_poly}" />')
    parts.append(f'<polyline class="tax-line" points="{tax_poly}" />')

    # Axes.
    parts.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" />')
    parts.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" />')
    parts.append(f'<line class="axis" x1="{pad_l + plot_w}" y1="{pad_t}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" />')

    # X ticks every other probability.
    for p in probs[::2]:
        x = xs(p)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + plot_h}" />')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{pad_t + plot_h + 14}" text-anchor="middle">{p:.2f}</text>')

    # Y ticks (4 each side).
    for i in range(5):
        wv = w_min + (w_max - w_min) * i / 4
        y = ys_w(wv)
        parts.append(f'<text class="tick" x="{pad_l - 6}" y="{y + 3:.1f}" text-anchor="end" fill="#3aa55c">{wv:.1f}</text>')
        tv = t_min + (t_max - t_min) * i / 4
        y = ys_t(tv)
        parts.append(f'<text class="tick" x="{pad_l + plot_w + 6}" y="{y + 3:.1f}" text-anchor="start" fill="#7ab8ff">{tv:.1f}</text>')

    # Axis labels + legend.
    parts.append(f'<text class="label" x="{pad_l + plot_w / 2}" y="{height - 12}" text-anchor="middle">audit_probability</text>')
    parts.append(f'<text class="label" x="{pad_l - 35}" y="{pad_t + plot_h / 2}" text-anchor="middle" fill="#3aa55c" transform="rotate(-90 {pad_l - 35} {pad_t + plot_h / 2})">welfare (mean)</text>')
    parts.append(f'<text class="label" x="{pad_l + plot_w + 40}" y="{pad_t + plot_h / 2}" text-anchor="middle" fill="#7ab8ff" transform="rotate(90 {pad_l + plot_w + 40} {pad_t + plot_h / 2})">tax revenue (mean)</text>')

    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def _table_md(rows: List[dict], svg_relpath: str) -> str:
    """Auto-generated table + chart. The narrative lives in FINDINGS.md
    (hand-curated). This is just the data view, written to TABLE.md so
    it can be re-rendered without overwriting interpretation."""
    return f"""# Audit-rate sweep — auto-generated data view

Re-rendered from `aggregate_final.csv` at sweep time. **For
interpretation, see FINDINGS.md.** This file is overwritten on every
re-run; do not edit by hand.

## Final-epoch frontier ({rows[0]['n_seeds']} seeds per cell)

{_frontier_table(rows)}

![welfare + tax_revenue vs audit_probability]({svg_relpath})
"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--skip-sweep", action="store_true",
                        help="reuse an existing aggregate_final.csv at --output")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not args.skip_sweep:
        _run_sweep(args.output, args.n_seeds, args.epochs, args.steps)

    rows = _read_finals(args.output / "aggregate_final.csv")
    rows.sort(key=lambda r: float(r["override_misreporting.audit_probability"]))
    svg_relpath = "welfare_tax_frontier.svg"
    _svg_frontier(rows, args.output / svg_relpath)
    (args.output / "TABLE.md").write_text(_table_md(rows, svg_relpath))
    logger.info(
        "wrote auto data view to %s + chart to %s. Interpretation: see FINDINGS.md.",
        args.output / "TABLE.md", args.output / svg_relpath,
    )


if __name__ == "__main__":
    main()
