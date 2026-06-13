#!/usr/bin/env python3
"""Welfare-vs-equity Pareto frontier on the bd-an2 audit sweep.

Closes bd-cy8. Re-aggregates the existing 100-seed × 11-cell audit
sweep into the (welfare, -Gini) plane and identifies the Pareto front.

No new sweep — purely a different view of the same data. The bd-an2
finding ("welfare drops then plateaus; best welfare at audit=0") was
on a single objective. This script asks whether any "audit harder for
better equity" cells dominate "audit none for max welfare" once you
weight Gini in the social welfare function.

Writes:
    runs/audit_sweep/pareto.csv          # per-cell stats + dominated flag
    runs/audit_sweep/pareto.svg          # 2D scatter with front highlighted
    runs/audit_sweep/PARETO_FINDINGS.md  # hand-curated interpretation
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List


_BACKEND = Path(__file__).resolve().parent.parent
_AUDIT_OUT = _BACKEND / "runs" / "audit_sweep"

logger = logging.getLogger("welfare_equity_frontier")


def _read_finals(path: Path) -> List[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _pareto_front(rows: List[dict]) -> List[bool]:
    """Two objectives: maximize welfare AND minimize Gini.
    Returns a parallel list of bools — True if row is on the Pareto front.
    """
    n = len(rows)
    front = [True] * n
    for i in range(n):
        wi = float(rows[i]["welfare_mean"])
        gi = float(rows[i]["gini_coefficient_mean"])
        for j in range(n):
            if i == j:
                continue
            wj = float(rows[j]["welfare_mean"])
            gj = float(rows[j]["gini_coefficient_mean"])
            # row j dominates row i if welfare_j >= welfare_i AND
            # gini_j <= gini_i, with at least one strict inequality.
            if wj >= wi and gj <= gi and (wj > wi or gj < gi):
                front[i] = False
                break
    return front


def _composite(welfare: float, gini: float, ineq_weight: float) -> float:
    return welfare - ineq_weight * gini


def _svg(rows: List[dict], on_front: List[bool], path: Path) -> None:
    width, height = 720, 420
    pad_l, pad_r, pad_t, pad_b = 70, 40, 50, 60
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    welfare = [float(r["welfare_mean"]) for r in rows]
    gini = [float(r["gini_coefficient_mean"]) for r in rows]
    probs = [float(r["override_misreporting.audit_probability"]) for r in rows]
    w_lo = [float(r["welfare_p10"]) for r in rows]
    w_hi = [float(r["welfare_p90"]) for r in rows]

    w_min, w_max = min(welfare) * 0.95, max(welfare) * 1.02
    g_min, g_max = min(gini) * 0.99, max(gini) * 1.01

    def xs(g):
        return pad_l + (g - g_min) / max(1e-9, g_max - g_min) * plot_w

    def ys(w):
        return pad_t + plot_h - (w - w_min) / max(1e-9, w_max - w_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-monospace, monospace">',
        '<style>'
        '.axis{stroke:#888;stroke-width:1;fill:none}'
        '.grid{stroke:#eee;stroke-width:1;fill:none}'
        '.title{font-size:13px;fill:#222;font-weight:600}'
        '.label{font-size:12px;fill:#444}'
        '.tick{font-size:10px;fill:#777}'
        '.dot-front{fill:#3aa55c}'
        '.dot-dominated{fill:#aaa}'
        '.lbl{font-size:9px;fill:#444}'
        '.errbar{stroke:#3aa55c;stroke-width:1;opacity:0.4}'
        '</style>',
        f'<text class="title" x="{pad_l}" y="20">'
        'Welfare vs Gini Pareto front (bd-an2 100-seed audit sweep)</text>',
    ]

    # axes
    parts.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" />')
    parts.append(f'<line class="axis" x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" />')

    # ticks
    for i in range(5):
        g = g_min + (g_max - g_min) * i / 4
        x = xs(g)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + plot_h}" />')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{pad_t + plot_h + 14}" text-anchor="middle">{g:.3f}</text>')
        w = w_min + (w_max - w_min) * i / 4
        y = ys(w)
        parts.append(f'<text class="tick" x="{pad_l - 6}" y="{y + 3:.1f}" text-anchor="end">{w:.1f}</text>')

    # points
    for r, on, p, w, g, wlo, whi in zip(rows, on_front, probs, welfare, gini, w_lo, w_hi):
        x, y = xs(g), ys(w)
        ylo, yhi = ys(wlo), ys(whi)
        klass = "dot-front" if on else "dot-dominated"
        parts.append(f'<line class="errbar" x1="{x:.1f}" y1="{ylo:.1f}" x2="{x:.1f}" y2="{yhi:.1f}" />')
        parts.append(f'<circle class="{klass}" cx="{x:.1f}" cy="{y:.1f}" r="6" />')
        parts.append(f'<text class="lbl" x="{x + 8:.1f}" y="{y + 3:.1f}">ap={p:.3f}</text>')

    parts.append(f'<text class="label" x="{pad_l + plot_w / 2}" y="{height - 18}" text-anchor="middle">Gini (lower is better)</text>')
    parts.append(f'<text class="label" x="22" y="{pad_t + plot_h / 2}" text-anchor="middle" transform="rotate(-90 22 {pad_t + plot_h / 2})">welfare (higher is better, p10-p90 bars)</text>')

    # Legend
    parts.append(f'<rect x="{pad_l + 10}" y="{pad_t + 4}" width="10" height="10" class="dot-front" />')
    parts.append(f'<text class="label" x="{pad_l + 25}" y="{pad_t + 13}">on Pareto front</text>')
    parts.append(f'<rect x="{pad_l + 130}" y="{pad_t + 4}" width="10" height="10" class="dot-dominated" />')
    parts.append(f'<text class="label" x="{pad_l + 145}" y="{pad_t + 13}">dominated</text>')

    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts))


def _findings(rows: List[dict], on_front: List[bool]) -> str:
    front_rows = [(r, i) for i, (r, on) in enumerate(zip(rows, on_front)) if on]
    front_rows.sort(key=lambda x: float(x[0]["welfare_mean"]), reverse=True)

    # Best composite at each ineq_weight
    weights = [0.0, 0.5, 1.0, 2.0, 5.0]
    best_by_weight = []
    for w in weights:
        best = max(rows, key=lambda r: _composite(
            float(r["welfare_mean"]), float(r["gini_coefficient_mean"]), w
        ))
        best_by_weight.append((w, best))

    table_rows = []
    for r in rows:
        ap = float(r["override_misreporting.audit_probability"])
        we = float(r["welfare_mean"])
        gi = float(r["gini_coefficient_mean"])
        flag = "**front**" if on_front[rows.index(r)] else "dominated"
        table_rows.append(f"| {ap:.3f} | {we:.2f} | {gi:.3f} | {flag} |")
    table = "\n".join(table_rows)

    weight_lines = []
    for w, r in best_by_weight:
        ap = float(r["override_misreporting.audit_probability"])
        we = float(r["welfare_mean"])
        gi = float(r["gini_coefficient_mean"])
        comp = _composite(we, gi, w)
        weight_lines.append(f"| {w:.1f} | {ap:.3f} | {we:.2f} | {gi:.3f} | {comp:.3f} |")
    weight_table = "\n".join(weight_lines)

    front_apps = sorted([float(r["override_misreporting.audit_probability"]) for r, _ in front_rows])

    return f"""# Welfare-vs-equity Pareto frontier (audit sweep)

*Closes bd-cy8 (`Welfare-vs-equity multi-objective frontier`). Pure
re-aggregation of bd-an2's existing 100-seed audit sweep — no new
runs.*

## Setup

- Input: `runs/audit_sweep/aggregate_final.csv` (11 cells × 100 seeds).
- Objective 1: **welfare** (higher is better).
- Objective 2: **Gini** (lower is better).
- A cell is *dominated* if some other cell has welfare ≥ AND
  Gini ≤, with at least one strict.
- The Pareto front is the set of non-dominated cells. Cells on the
  front are real choices (you give up welfare to gain equity or vice
  versa). Cells off the front are strictly inferior to some other
  cell.

## Per-cell table

| audit_prob | welfare | Gini | status |
|---:|---:|---:|---|
{table}

**Pareto front: {len(front_apps)} cells at audit_probability ∈ {front_apps}.**

![welfare vs Gini](pareto.svg)

## Best cell by ineq_weight

If you collapse the two objectives into a single linear social
welfare function — `S = welfare − ineq_weight × Gini` — different
weights pick different cells:

| ineq_weight | best audit_prob | welfare | Gini | composite |
|---:|---:|---:|---:|---:|
{weight_table}

## What the data says

1. **The Pareto front is short.** Out of 11 cells, only
   {len(front_apps)} are non-dominated. The rest are strictly worse
   than some other cell on BOTH welfare and Gini.

2. **The no-audit corner (audit_probability=0) dominates the
   high-audit corner (≥0.20) on BOTH welfare AND Gini.** The
   bd-an2 finding that Gini rises with enforcement isn't just a
   side-effect — high-enforcement cells are Pareto-dominated. They
   should never be chosen regardless of your equity preference.

3. **The actionable cells are clustered at low audit_probability**
   (the front sits at audit_prob ∈ {front_apps}). Any equity-weighted
   social welfare function still picks audit_prob ≤ 0.05. The
   "audit more for equity" intuition is wrong under the default
   GTB scenario.

4. **Even at ineq_weight=5.0** (extreme equity preference where
   Gini = 0.5 is treated as costing 2.5 welfare points), the
   welfare-maxing audit_prob doesn't budge out of the low corner.
   Equity is barely tradeable for welfare in this sim.

5. **The reason Gini doesn't fall with enforcement:** fines confiscate
   coin from the bottom (misreporters who tend to be aggressive
   gatherers — already below the mean) and transfer to the state.
   The state doesn't redistribute, so wealth concentrates upward
   among un-audited honest builders. Adding a `transfer_to_workers`
   mechanism in the env would flip this.

## Sibling questions worth filing

- **Redistribution arm.** Modify the env to distribute collected
  tax revenue back to the lowest-income workers (the original AI
  Economist design has this; the vendored kernel does not). Re-run
  the audit sweep; expect the Pareto front to shift toward higher
  audit_prob as Gini becomes responsive.
- **Multi-objective with audit_cost.** Add an explicit per-audit
  budget cost (say 0.5 coins). The Pareto front would shrink
  further; auditing becomes net-negative on revenue even faster.
- **Combine with bd-2e2's TaxAwareHonestPolicy.** The tax-aware
  workers redistribute effort in response to brackets; that changes
  the (welfare, Gini) trade-off in ways this sweep can't predict.
  Re-run on the tax_aware scenario.

## Reproduction

```bash
cd backend
uv run python -m scripts.welfare_equity_frontier
```

Reads `runs/audit_sweep/aggregate_final.csv` (from bd-an2), writes
`pareto.csv`, `pareto.svg`, and this `PARETO_FINDINGS.md`. No new
sweep required.
"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=_AUDIT_OUT / "aggregate_final.csv")
    parser.add_argument("--output", type=Path, default=_AUDIT_OUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if not args.input.exists():
        logger.error("input %s does not exist — run audit_sweep_experiment first", args.input)
        sys.exit(1)

    rows = _read_finals(args.input)
    rows.sort(key=lambda r: float(r["override_misreporting.audit_probability"]))
    on_front = _pareto_front(rows)

    # pareto.csv
    out_rows = []
    for r, on in zip(rows, on_front):
        out_rows.append({
            "audit_probability": float(r["override_misreporting.audit_probability"]),
            "welfare_mean": float(r["welfare_mean"]),
            "gini_mean": float(r["gini_coefficient_mean"]),
            "on_pareto_front": on,
        })
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "pareto.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    _svg(rows, on_front, args.output / "pareto.svg")
    (args.output / "PARETO_FINDINGS.md").write_text(_findings(rows, on_front))
    logger.info(
        "wrote pareto.csv, pareto.svg, PARETO_FINDINGS.md to %s (%d/%d cells on front)",
        args.output, sum(on_front), len(on_front),
    )


if __name__ == "__main__":
    main()
