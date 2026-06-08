#!/usr/bin/env python3
"""Planner reactivity over long horizons.

Closes bd-coq. Builds on bd-cec's sweep harness.

Hypothesis: at 30-epoch horizons:
  - heuristic planner converges in ~15 epochs to a stable bracket set
  - bandit planner takes ~30
  - rl planner is a stub (no-op); welfare = baseline forever

Method: run 20 seeds × {heuristic, bandit, rl} × 40 epochs × 10 steps.
For each planner type, measure:
  - per-epoch welfare trajectory (mean, p10-p90 across seeds)
  - per-epoch tax_revenue trajectory
  - epoch at which the welfare slope falls below 0.05 / epoch
    averaged over a 5-epoch window (proxy for convergence)
  - total welfare gain over the first 5 epochs (a measure of how
    quickly the planner reacts)

Writes:
  runs/planner_reactivity/aggregate.csv
  runs/planner_reactivity/welfare_trajectory.csv
  runs/planner_reactivity/planner_comparison.svg
  runs/planner_reactivity/FINDINGS.md   (hand-curated)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


_BACKEND = Path(__file__).resolve().parent.parent
_SCENARIO = _BACKEND / "worlds/gather_trade_build/scenarios/ai_economist_full.yaml"
_DEFAULT_OUTPUT = _BACKEND / "runs/planner_reactivity"

logger = logging.getLogger("planner_reactivity")


def _run_sweep(output: Path, n_seeds: int, n_epochs: int, steps: int) -> None:
    cmd = [
        sys.executable, "-m", "scripts.sweep_gtb", str(_SCENARIO),
        "--n-seeds", str(n_seeds), "--epochs", str(n_epochs), "--steps", str(steps),
        "--sweep", "planner.planner_type=heuristic,bandit,rl",
        "--output", str(output),
    ]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=_BACKEND, check=True)


def _percentile(vs: List[float], p: float) -> float:
    if not vs:
        return float("nan")
    if len(vs) == 1:
        return vs[0]
    qs = statistics.quantiles(vs, n=100, method="inclusive")
    return qs[max(0, min(98, int(round(p * 100)) - 1))]


def _read_cells(output: Path) -> List[dict]:
    with open(output / "sweep.json") as f:
        return json.load(f)["cells"]


def _trajectory_by_planner(output: Path) -> Dict[str, List[dict]]:
    cells = _read_cells(output)
    by_planner: Dict[str, List[dict]] = {}
    for cell in cells:
        planner_type = cell["overrides"]["planner.planner_type"]
        per_epoch_seeds: Dict[int, dict] = {}
        seed_dirs = sorted((output / "cells" / f"cell_{cell['index']:03d}").glob("seed_*"))
        for sd in seed_dirs:
            m_csv = sd / "csv" / "metrics.csv"
            if not m_csv.exists():
                continue
            with open(m_csv) as f:
                for row in csv.DictReader(f):
                    epoch = int(row["epoch"])
                    bucket = per_epoch_seeds.setdefault(epoch, {"welfare": [], "tax": []})
                    bucket["welfare"].append(float(row["welfare"]))
                    bucket["tax"].append(float(row["total_tax_revenue"]))
        rows = []
        for epoch in sorted(per_epoch_seeds):
            wv = per_epoch_seeds[epoch]["welfare"]
            tv = per_epoch_seeds[epoch]["tax"]
            rows.append({
                "epoch": epoch,
                "planner_type": planner_type,
                "n_seeds": len(wv),
                "welfare_mean": statistics.mean(wv),
                "welfare_p10": _percentile(wv, 0.10),
                "welfare_p90": _percentile(wv, 0.90),
                "tax_revenue_mean": statistics.mean(tv),
            })
        by_planner[planner_type] = rows
    return by_planner


def _convergence_epoch(rows: List[dict], window: int = 5, threshold: float = 0.05) -> int:
    """First epoch e such that the slope (welfare[e+window] - welfare[e]) / window <= threshold."""
    for i in range(len(rows) - window):
        slope = (rows[i + window]["welfare_mean"] - rows[i]["welfare_mean"]) / window
        if abs(slope) <= threshold:
            return rows[i]["epoch"]
    return -1


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _render_svg(by_planner: Dict[str, List[dict]], path: Path) -> None:
    width, height = 720, 360
    pad_l, pad_r, pad_t, pad_b = 60, 40, 40, 60
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    colors = {"heuristic": "#3aa55c", "bandit": "#7ab8ff", "rl": "#ff9a66"}

    all_epochs = sorted({r["epoch"] for rows in by_planner.values() for r in rows})
    all_welfare = [r["welfare_mean"] for rows in by_planner.values() for r in rows]
    x_min, x_max = all_epochs[0], all_epochs[-1]
    w_min, w_max = min(all_welfare), max(all_welfare)
    w_min -= (w_max - w_min) * 0.05
    w_max += (w_max - w_min) * 0.05

    def xs(e):
        return pad_l + (e - x_min) / max(1, x_max - x_min) * plot_w

    def ys(v):
        return pad_t + plot_h - (v - w_min) / max(1e-9, w_max - w_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-monospace, monospace">',
        '<style>'
        '.axis{stroke:#888;stroke-width:1;fill:none}'
        '.grid{stroke:#eee;stroke-width:1;fill:none}'
        '.title{font-size:13px;fill:#222;font-weight:600}'
        '.label{font-size:12px;fill:#444}'
        '.tick{font-size:10px;fill:#777}'
        '</style>',
        f'<text class="title" x="{pad_l}" y="20">'
        'Welfare trajectory by planner type — bd-coq</text>',
    ]
    # axes
    parts.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" />')
    parts.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" />')
    # bands + lines per planner
    legend_y = pad_t + 20
    for pt in ("heuristic", "bandit", "rl"):
        rows = by_planner.get(pt, [])
        if not rows:
            continue
        color = colors.get(pt, "#888")
        band_top = " ".join(f"{xs(r['epoch']):.1f},{ys(r['welfare_p90']):.1f}" for r in rows)
        band_bot = " ".join(f"{xs(r['epoch']):.1f},{ys(r['welfare_p10']):.1f}" for r in reversed(rows))
        parts.append(f'<polygon points="{band_top} {band_bot}" fill="{color}" fill-opacity="0.12" />')
        line = " ".join(f"{xs(r['epoch']):.1f},{ys(r['welfare_mean']):.1f}" for r in rows)
        parts.append(f'<polyline points="{line}" stroke="{color}" stroke-width="2" fill="none" />')
        parts.append(f'<rect x="{pad_l + 10}" y="{legend_y - 8}" width="10" height="10" fill="{color}" />')
        parts.append(f'<text class="label" x="{pad_l + 25}" y="{legend_y}" fill="{color}">{pt}</text>')
        legend_y += 16

    # ticks
    step_e = max(1, (x_max - x_min) // 6)
    for e in range(int(x_min), int(x_max) + 1, step_e):
        x = xs(e)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + plot_h}" />')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{pad_t + plot_h + 14}" text-anchor="middle">{e}</text>')
    for i in range(5):
        wv = w_min + (w_max - w_min) * i / 4
        y = ys(wv)
        parts.append(f'<text class="tick" x="{pad_l - 6}" y="{y + 3:.1f}" text-anchor="end">{wv:.1f}</text>')

    parts.append(f'<text class="label" x="{pad_l + plot_w / 2}" y="{height - 20}" text-anchor="middle">epoch</text>')
    parts.append(f'<text class="label" x="20" y="{pad_t + plot_h / 2}" text-anchor="middle" transform="rotate(-90 20 {pad_t + plot_h / 2})">welfare (mean ± p10–p90)</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not args.skip_sweep:
        _run_sweep(args.output, args.n_seeds, args.epochs, args.steps)

    by_planner = _trajectory_by_planner(args.output)
    flat = []
    for pt, rows in by_planner.items():
        flat.extend(rows)
    _write_csv(args.output / "welfare_trajectory.csv", flat)
    _render_svg(by_planner, args.output / "planner_comparison.svg")

    for pt, rows in by_planner.items():
        conv = _convergence_epoch(rows)
        first5_gain = (rows[5]["welfare_mean"] - rows[0]["welfare_mean"]) if len(rows) > 5 else 0
        final_welfare = rows[-1]["welfare_mean"] if rows else 0
        logger.info(
            "%-10s: convergence_epoch=%s first_5_epoch_gain=%.2f final_welfare=%.2f",
            pt, conv if conv >= 0 else "never", first5_gain, final_welfare,
        )


if __name__ == "__main__":
    main()
