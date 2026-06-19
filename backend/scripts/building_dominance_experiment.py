#!/usr/bin/env python3
"""Does house-rent dominate gross income as the run progresses?

Closes bd-vel. Builds on bd-cec's sweep harness.

Hypothesis: under honest-only agents at the default scenario,
house-rent income comes to dominate the per-epoch gross income within
~10 epochs. The other gather/build/trade work tapers as workers stop
producing and live off rent.

Method:
  Run 20 seeds × 30 epochs × 10 steps with an honest-only override of
  the default scenario. For each epoch e:
    house_income[e]  ≈ total_houses_built[e] × income_per_house × steps
    gather_income[e] ≈ max(0, total_production[e] - house_income[e])
    rent_fraction[e] = house_income[e] / total_production[e]
  Aggregate rent_fraction per epoch across seeds (mean / p10 / p90).

Writes:
  runs/building_dominance/aggregate.csv             # raw sweep output
  runs/building_dominance/rent_share_trajectory.csv # rent_fraction[e]
  runs/building_dominance/FINDINGS.md               # hand-curated
  runs/building_dominance/rent_share.svg           # auto chart
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
from typing import List


_BACKEND = Path(__file__).resolve().parent.parent
_SCENARIO = _BACKEND / "worlds/gather_trade_build/scenarios/ai_economist_full.yaml"
_DEFAULT_OUTPUT = _BACKEND / "runs/building_dominance"

logger = logging.getLogger("building_dominance")


def _write_honest_only_scenario(scratch: Path) -> Path:
    """Materialize a honest-only variant of the default scenario."""
    import yaml
    with open(_SCENARIO) as f:
        data = yaml.safe_load(f)
    data["agents"] = [{"policy": "honest", "count": 8, "skill_gather": 1.0, "skill_build": 1.0}]
    out = scratch / "honest_only.yaml"
    scratch.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump(data, f)
    return out


def _run_sweep(scenario: Path, output: Path, n_seeds: int, n_epochs: int, steps: int) -> None:
    cmd = [
        sys.executable, "-m", "scripts.sweep_gtb", str(scenario),
        "--n-seeds", str(n_seeds), "--epochs", str(n_epochs), "--steps", str(steps),
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
    idx = max(0, min(98, int(round(p * 100)) - 1))
    return qs[idx]


def _compute_rent_trajectory(output: Path, steps: int, income_per_house: float = 1.0) -> List[dict]:
    """Walk every per-(cell, seed) metrics.csv, compute rent_fraction by
    epoch, aggregate across seeds."""
    seed_dirs = sorted((output / "cells" / "cell_000").glob("seed_*"))
    if not seed_dirs:
        raise SystemExit(f"no per-seed runs found under {output}")

    per_epoch_seeds: dict = {}  # epoch -> list of rent_fractions
    for sd in seed_dirs:
        m_csv = sd / "csv" / "metrics.csv"
        if not m_csv.exists():
            continue
        with open(m_csv) as f:
            for row in csv.DictReader(f):
                epoch = int(row["epoch"])
                total_prod = float(row["total_production"])
                total_houses = int(row["total_houses_built"])
                house_income = total_houses * income_per_house * steps
                # Cap at total to avoid floating-point > 1.0 noise.
                rent_fraction = min(1.0, house_income / total_prod) if total_prod > 0 else 0.0
                per_epoch_seeds.setdefault(epoch, []).append(rent_fraction)

    rows = []
    for epoch in sorted(per_epoch_seeds):
        vs = per_epoch_seeds[epoch]
        rows.append({
            "epoch": epoch,
            "n_seeds": len(vs),
            "rent_fraction_mean": statistics.mean(vs),
            "rent_fraction_median": statistics.median(vs),
            "rent_fraction_p10": _percentile(vs, 0.10),
            "rent_fraction_p90": _percentile(vs, 0.90),
        })
    return rows


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _render_svg(rows: List[dict], path: Path) -> None:
    width, height = 720, 320
    pad_l, pad_r, pad_t, pad_b = 60, 30, 40, 50
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    epochs = [r["epoch"] for r in rows]
    mean_v = [r["rent_fraction_mean"] for r in rows]
    p10 = [r["rent_fraction_p10"] for r in rows]
    p90 = [r["rent_fraction_p90"] for r in rows]
    x_min, x_max = min(epochs), max(epochs)

    def xs(e):
        return pad_l + (e - x_min) / max(1, x_max - x_min) * plot_w

    def ys(v):
        return pad_t + plot_h - v * plot_h  # 0..1 → bottom..top

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-monospace, monospace">',
        '<style>'
        '.axis{stroke:#888;stroke-width:1;fill:none}'
        '.grid{stroke:#eee;stroke-width:1;fill:none}'
        '.line{stroke:#3aa55c;stroke-width:2.5;fill:none}'
        '.band{fill:#3aa55c;fill-opacity:0.18;stroke:none}'
        '.ref80{stroke:#ff9a66;stroke-width:1;stroke-dasharray:4 4;fill:none}'
        '.ref95{stroke:#ff5577;stroke-width:1;stroke-dasharray:4 4;fill:none}'
        '.title{font-size:13px;fill:#222;font-weight:600}'
        '.label{font-size:12px;fill:#444}'
        '.tick{font-size:10px;fill:#777}'
        '</style>',
        f'<text class="title" x="{pad_l}" y="20">'
        f'Rent-share of gross income vs epoch (n={rows[0]["n_seeds"]} seeds, honest-only)'
        '</text>',
    ]
    # p10-p90 band
    band_top = " ".join(f"{xs(e):.1f},{ys(v):.1f}" for e, v in zip(epochs, p90))
    band_bot = " ".join(f"{xs(e):.1f},{ys(v):.1f}" for e, v in zip(reversed(epochs), reversed(p10)))
    parts.append(f'<polygon class="band" points="{band_top} {band_bot}" />')
    # mean line
    mean_poly = " ".join(f"{xs(e):.1f},{ys(v):.1f}" for e, v in zip(epochs, mean_v))
    parts.append(f'<polyline class="line" points="{mean_poly}" />')
    # 80% / 95% reference lines
    parts.append(f'<line class="ref80" x1="{pad_l}" y1="{ys(0.8):.1f}" x2="{pad_l + plot_w}" y2="{ys(0.8):.1f}" />')
    parts.append(f'<text class="tick" x="{pad_l + plot_w + 4}" y="{ys(0.8) + 3:.1f}" fill="#ff9a66">80%</text>')
    parts.append(f'<line class="ref95" x1="{pad_l}" y1="{ys(0.95):.1f}" x2="{pad_l + plot_w}" y2="{ys(0.95):.1f}" />')
    parts.append(f'<text class="tick" x="{pad_l + plot_w + 4}" y="{ys(0.95) + 3:.1f}" fill="#ff5577">95%</text>')
    # axes
    parts.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" />')
    parts.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" />')
    # ticks
    step_e = max(1, (x_max - x_min) // 6)
    for e in range(int(x_min), int(x_max) + 1, step_e):
        x = xs(e)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + plot_h}" />')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{pad_t + plot_h + 14}" text-anchor="middle">{e}</text>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = ys(frac)
        parts.append(f'<text class="tick" x="{pad_l - 6}" y="{y + 3:.1f}" text-anchor="end">{int(frac * 100)}%</text>')
    parts.append(f'<text class="label" x="{pad_l + plot_w / 2}" y="{height - 14}" text-anchor="middle">epoch</text>')
    parts.append(f'<text class="label" x="20" y="{pad_t + plot_h / 2}" text-anchor="middle" transform="rotate(-90 20 {pad_t + plot_h / 2})">rent share of gross income</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not args.skip_sweep:
        scratch = args.output / "_scratch"
        scenario = _write_honest_only_scenario(scratch)
        _run_sweep(scenario, args.output, args.n_seeds, args.epochs, args.steps)

    rows = _compute_rent_trajectory(args.output, steps=args.steps)
    _write_csv(args.output / "rent_share_trajectory.csv", rows)
    _render_svg(rows, args.output / "rent_share.svg")
    # Quick console summary.
    over80 = next((r for r in rows if r["rent_fraction_mean"] >= 0.8), None)
    over95 = next((r for r in rows if r["rent_fraction_mean"] >= 0.95), None)
    logger.info(
        "rent_share crosses 80%% at epoch %s, 95%% at epoch %s",
        over80["epoch"] if over80 else "never",
        over95["epoch"] if over95 else "never",
    )


if __name__ == "__main__":
    main()
