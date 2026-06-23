"""Map the elasticity band where welfare-weighted Saez beats the heuristic (bd-brb).

bd-5gz found a small but significant paired welfare win for Saez at
labor_coeff=0.3, nothing at 0.15 or >=0.5. This sweeps labor_coeff
finely across 0.15-0.50 and, at each point, runs saez and heuristic on
the SAME seeds so the per-seed welfare difference can be paired-tested.
The marginal aggregate CSVs can't show this — a t=3.58 effect hides
inside fully overlapping p10-p90 bands — so this experiment owns its own
paired stats rather than reading sweep_gtb output.

Usage::

    uv run python -m scripts.saez_band_experiment \
        --n-seeds 100 --epochs 30 --output runs/saez_band

Writes ``paired.csv`` (one row per labor_coeff) and ``band.svg`` (paired
mean welfare diff vs labor_coeff with a +/-1.96 se band and a zero line).
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import math
import statistics as st
from pathlib import Path
from typing import Dict, List

import yaml

# Quiet the per-epoch sim logging; this script runs thousands of sims.
logging.disable(logging.CRITICAL)

from worlds.gather_trade_build.config import GTBConfig  # noqa: E402
from worlds.gather_trade_build.runner import GTBScenarioRunner  # noqa: E402

SCENARIO = "worlds/gather_trade_build/scenarios/ai_economist_saez.yaml"
DEFAULT_BAND = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)


def _final_welfare(scenario: dict, planner_type: str, labor_coeff: float,
                   seed: int, epochs: int, steps: int) -> float:
    s = copy.deepcopy(scenario)
    dom = s.setdefault("domain", {})
    dom.setdefault("planner", {})["planner_type"] = planner_type
    dom.setdefault("utility", {})["labor_coeff"] = labor_coeff
    cfg = GTBConfig.from_dict(dom)
    cfg.seed = seed
    runner = GTBScenarioRunner(
        config=cfg, agent_specs=s["agents"],
        n_epochs=epochs, steps_per_epoch=steps, seed=seed,
    )
    last = runner.run()[-1]
    d = last.to_dict() if hasattr(last, "to_dict") else last
    return float(d["welfare"])


def _paired_row(scenario, labor_coeff, n_seeds, epochs, steps) -> Dict[str, float]:
    """saez - heuristic welfare on identical seeds, with a paired t-stat."""
    diffs: List[float] = []
    for seed in range(n_seeds):
        sa = _final_welfare(scenario, "saez", labor_coeff, seed, epochs, steps)
        he = _final_welfare(scenario, "heuristic", labor_coeff, seed, epochs, steps)
        diffs.append(sa - he)
    mean = st.mean(diffs)
    sd = st.pstdev(diffs)
    se = sd / math.sqrt(len(diffs)) if sd > 0 else 0.0
    wins = sum(1 for d in diffs if d > 1e-9)
    losses = sum(1 for d in diffs if d < -1e-9)
    ties = len(diffs) - wins - losses
    return {
        "labor_coeff": labor_coeff,
        "n": len(diffs),
        "mean_diff": mean,
        "se": se,
        "t": (mean / se) if se > 0 else 0.0,
        "median_diff": st.median(diffs),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def _svg(rows: List[Dict[str, float]], out: Path) -> None:
    width, height = 720, 380
    pad_l, pad_t, pad_r, pad_b = 64, 50, 24, 56
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b
    xs = [r["labor_coeff"] for r in rows]
    means = [r["mean_diff"] for r in rows]
    los = [r["mean_diff"] - 1.96 * r["se"] for r in rows]
    his = [r["mean_diff"] + 1.96 * r["se"] for r in rows]
    xmin, xmax = min(xs), max(xs)
    ymin = min(0.0, min(los)) * 1.1
    ymax = max(0.0, max(his)) * 1.1 or 1.0

    def sx(v): return pad_l + (v - xmin) / (xmax - xmin) * pw if xmax > xmin else pad_l + pw / 2
    def sy(v): return pad_t + ph - (v - ymin) / (ymax - ymin) * ph

    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui,sans-serif">',
        "<style>.t{font-size:15px;font-weight:600;fill:#111}.s{font-size:11px;fill:#666}"
        ".ax{stroke:#999;stroke-width:1}.gr{stroke:#eee}.tk{font-size:10px;fill:#666}"
        ".lab{font-size:11px;fill:#555}</style>",
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
        f'<text class="t" x="{pad_l}" y="22">Where welfare-weighted Saez beats the '
        f'heuristic (paired welfare diff)</text>',
        f'<text class="s" x="{pad_l}" y="38">band = mean ± 1.96·se · N per point in '
        f'paired.csv · above the zero line = Saez wins</text>',
    ]
    # zero line
    yz = sy(0.0)
    p.append(f'<line class="ax" x1="{pad_l}" y1="{yz:.1f}" x2="{pad_l+pw}" y2="{yz:.1f}" '
             f'stroke="#bbb" stroke-dasharray="4 3"/>')
    # axes
    p.append(f'<line class="ax" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+ph}"/>')
    # y ticks
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = ymin + f * (ymax - ymin); yy = sy(v)
        p.append(f'<text class="tk" x="{pad_l-6:.1f}" y="{yy+3:.1f}" text-anchor="end">{v:+.2f}</text>')
        p.append(f'<line class="gr" x1="{pad_l}" y1="{yy:.1f}" x2="{pad_l+pw}" y2="{yy:.1f}"/>')
    # x ticks
    for r in rows:
        p.append(f'<text class="tk" x="{sx(r["labor_coeff"]):.1f}" y="{pad_t+ph+16:.1f}" '
                 f'text-anchor="middle">{r["labor_coeff"]:g}</text>')
    # se band
    top = " ".join(f"{sx(x):.1f},{sy(h):.1f}" for x, h in zip(xs, his))
    bot = " ".join(f"{sx(x):.1f},{sy(l):.1f}" for x, l in zip(reversed(xs), reversed(los)))
    p.append(f'<polygon points="{top} {bot}" fill="#3aa55c" fill-opacity="0.15" stroke="none"/>')
    # mean line + markers (filled where significant: |t|>=1.96)
    poly = " ".join(f"{sx(x):.1f},{sy(m):.1f}" for x, m in zip(xs, means))
    p.append(f'<polyline points="{poly}" fill="none" stroke="#3aa55c" stroke-width="2"/>')
    for r in rows:
        sig = abs(r["t"]) >= 1.96
        fill = "#3aa55c" if sig else "#fff"
        p.append(f'<circle cx="{sx(r["labor_coeff"]):.1f}" cy="{sy(r["mean_diff"]):.1f}" '
                 f'r="3.5" fill="{fill}" stroke="#3aa55c" stroke-width="1.5"/>')
    p.append(f'<text class="lab" x="{pad_l+pw/2:.1f}" y="{height-12}" text-anchor="middle">'
             f'utility.labor_coeff (worker tax-elasticity →) · filled dot = significant (|t|≥1.96)</text>')
    p.append("</svg>")
    out.write_text("\n".join(p))
    print(f"wrote {out}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--band", type=str, default=None,
                    help="Comma-separated labor_coeff values (default: 0.15..0.50 by 0.05)")
    ap.add_argument("--output", type=Path, default=Path("runs/saez_band"))
    args = ap.parse_args(argv)

    scenario = yaml.safe_load(open(SCENARIO))
    band = ([float(v) for v in args.band.split(",")] if args.band else list(DEFAULT_BAND))

    rows: List[Dict[str, float]] = []
    for lc in band:
        row = _paired_row(scenario, lc, args.n_seeds, args.epochs, args.steps)
        rows.append(row)
        sig = "*" if abs(row["t"]) >= 1.96 else " "
        print(f"  lc={lc:<5} mean={row['mean_diff']:+.3f} t={row['t']:+.2f}{sig} "
              f"wins={row['wins']:>3} ties={row['ties']:>3} losses={row['losses']:>3}")

    args.output.mkdir(parents=True, exist_ok=True)
    cols = ["labor_coeff", "n", "mean_diff", "se", "t", "median_diff", "wins", "ties", "losses"]
    with (args.output / "paired.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.output/'paired.csv'}")
    _svg(rows, args.output / "band.svg")


if __name__ == "__main__":
    main()
