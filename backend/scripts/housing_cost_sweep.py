#!/usr/bin/env python3
"""Map the housing-cost parameter region where rent doesn't dominate.

Closes bd-yy1. Builds on bd-vel's finding that the default scenario
collapses to "build N houses then idle" within ~3 epochs.

Sweep: `build.wood_cost` ∈ {3, 6, 12} × `build.stone_cost` ∈ {3, 6, 12}
× `build.income_per_house_per_step` ∈ {0.25, 0.5, 1.0}. 3 × 3 × 3 = 27
cells. 20 seeds each, 30 epochs × 10 steps. ~540 runs total — should
finish in under 15 seconds on the bd-cec harness.

For each cell we compute:
  - final-epoch rent share (estimator from bd-vel)
  - final-epoch welfare + Gini
  - "rent_dominates" = rent_share >= 0.80 in the last 5 epochs (the
    bd-vel threshold for the dominant-strategy attractor)

The deliverable is `FINDINGS.md` identifying the (wood_cost,
stone_cost, income_per_step) cells where labor stays competitive.
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
_DEFAULT_OUTPUT = _BACKEND / "runs/housing_cost_sweep"

logger = logging.getLogger("housing_cost_sweep")


def _run_sweep(output: Path, n_seeds: int, n_epochs: int, steps: int) -> None:
    cmd = [
        sys.executable, "-m", "scripts.sweep_gtb", str(_SCENARIO),
        "--n-seeds", str(n_seeds), "--epochs", str(n_epochs), "--steps", str(steps),
        "--sweep", "build.wood_cost=3,6,12",
        "--sweep", "build.stone_cost=3,6,12",
        "--sweep", "build.income_per_house_per_step=0.25,0.5,1.0",
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


def _read_cell_descriptors(output: Path) -> List[dict]:
    with open(output / "sweep.json") as f:
        return json.load(f)["cells"]


def _compute_rent_share_by_cell(output: Path, steps: int) -> List[dict]:
    """For each cell, walk every seed run, compute rent_share trajectory,
    and pull out (a) rent share at last 5 epochs (mean), (b) final welfare
    and Gini."""
    cells = _read_cell_descriptors(output)
    rows = []
    for cell in cells:
        income_per_house = float(cell["overrides"]["build.income_per_house_per_step"])
        wood_cost = cell["overrides"]["build.wood_cost"]
        stone_cost = cell["overrides"]["build.stone_cost"]
        cell_dir = output / "cells" / f"cell_{cell['index']:03d}"
        seed_dirs = sorted(cell_dir.glob("seed_*"))
        if not seed_dirs:
            continue
        late_rent_shares: List[float] = []
        final_welfare: List[float] = []
        final_gini: List[float] = []
        for sd in seed_dirs:
            m_csv = sd / "csv" / "metrics.csv"
            if not m_csv.exists():
                continue
            with open(m_csv) as f:
                run_rows = list(csv.DictReader(f))
            if not run_rows:
                continue
            for row in run_rows[-5:]:
                tp = float(row["total_production"])
                th = int(row["total_houses_built"])
                house_income = th * income_per_house * steps
                late_rent_shares.append(min(1.0, house_income / tp) if tp > 0 else 0.0)
            final_welfare.append(float(run_rows[-1]["welfare"]))
            final_gini.append(float(run_rows[-1]["gini_coefficient"]))
        if not late_rent_shares:
            continue
        rent_mean = statistics.mean(late_rent_shares)
        rows.append({
            "cell_index": cell["index"],
            "wood_cost": wood_cost,
            "stone_cost": stone_cost,
            "income_per_step": income_per_house,
            "n_seeds": len(seed_dirs),
            "late_rent_share_mean": rent_mean,
            "late_rent_share_p10": _percentile(late_rent_shares, 0.10),
            "late_rent_share_p90": _percentile(late_rent_shares, 0.90),
            "rent_dominates": rent_mean >= 0.80,
            "final_welfare_mean": statistics.mean(final_welfare),
            "final_gini_mean": statistics.mean(final_gini),
        })
    # Sort by rent_share ascending so the labor-competitive cells lead.
    rows.sort(key=lambda r: r["late_rent_share_mean"])
    return rows


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


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
        _run_sweep(args.output, args.n_seeds, args.epochs, args.steps)

    rows = _compute_rent_share_by_cell(args.output, steps=args.steps)
    _write_csv(args.output / "rent_share_by_cell.csv", rows)
    n_labor = sum(1 for r in rows if not r["rent_dominates"])
    logger.info(
        "wrote %d cell rows. %d/%d cells keep labor competitive (late rent share < 80%%).",
        len(rows), n_labor, len(rows),
    )
    # Print top 5 labor-competitive cells to console for the FINDINGS doc.
    for r in rows[:5]:
        logger.info(
            "  cell %03d wood=%s stone=%s inc/step=%s -> rent_share=%.2f welfare=%.1f gini=%.3f",
            r["cell_index"], r["wood_cost"], r["stone_cost"], r["income_per_step"],
            r["late_rent_share_mean"], r["final_welfare_mean"], r["final_gini_mean"],
        )


if __name__ == "__main__":
    main()
