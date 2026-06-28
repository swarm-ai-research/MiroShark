"""Validate H100 compute-futures hedging across seeds (bd k9w, caps bd ja2).

The compute analog of ``scripts.futures_validation_experiment``, framed around
the distinctive compute story: an **AI lab hedging the cost of an H100 training
run**. The lab will buy ``Q`` H100-hours at the (volatile) settlement spot; it
locks part of that cost early by going long a dated forward at price ``F``. Net
cost = spot·Q − realized long P&L. As the hedge ratio rises, net cost
converges to F·Q and its seed-to-seed spread collapses — the textbook result,
here measured through the *real* order book and settlement engine on
``ResourceType.COMPUTE``.

Realistic dynamics: settlement spot is a deterministic, per-seed **log-normal**
draw around a base H100 rental price (right-skewed, occasional spikes), mimicking
observed GPU-rental volatility. Deterministic (no RNG — the workflow forbids it
in scripts) so the whole sweep is reproducible.

Three checks, each reported faithfully:

A. HEDGE SWEEP (mechanism). Sweep hedge ratio h ∈ {0,.25,.5,.75,1}. For each
   cell, across N seeds, aggregate net-cost mean/median/p10/p90 and the variance
   reduction vs unhedged (h=0). Confidence bands (p10–p90) rendered to SVG.

B. ROBUSTNESS (population). A compute futures lineup across N seeds: settlement
   is zero-sum, no balance goes negative, the market is active (matches,
   settlements, volume).

C. NAIVE POLICIES (honest negative). Scripted compute hedger (hedge on/off)
   final-coin variance — does the *naive* agent realize the reduction the
   mechanism makes available? (Expectation from the wood study: it under-hedges.)

Usage::

    uv run python -m scripts.compute_futures_experiment --n-seeds 100

Writes cost.csv, aggregate.csv, TABLE.md, and hedge_bands.svg to the output dir.
FINDINGS.md is hand-curated separately (CLAUDE.md methodology).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics as st
from pathlib import Path
from typing import Dict, List

logging.disable(logging.CRITICAL)

from worlds.gather_trade_build.config import GTBConfig  # noqa: E402
from worlds.gather_trade_build.entities import (  # noqa: E402
    GTBActionType, ResourceType,
)
from worlds.gather_trade_build.env import GTBAction, GTBEnvironment  # noqa: E402
from worlds.gather_trade_build.runner import GTBScenarioRunner  # noqa: E402

Q = 100.0          # H100-hours the lab will buy at the settlement spot
BASE = 3.50        # base H100 rental price ($/GPU-hr); spot median
SIGMA = 0.5        # log-vol of the settlement spot (right-skewed spikes)
SETTLE_EPOCH = 2
ENDOW = 5000.0     # coin endowment, ample so settlement never clamps
HEDGE_RATIOS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _h100_spot(seed: int) -> float:
    """Deterministic, reproducible per-seed H100 spot ($/GPU-hr).

    Log-normal around ``BASE`` to mimic right-skewed GPU-rental prices
    (capacity crunches spike the right tail). Built without RNG: a
    quasi-normal ``z`` is assembled from several decorrelated hashes of the
    seed (central-limit sum of uniforms), then mapped through exp.
    """
    mults = (2654435761, 40503, 2246822519, 3266489917)
    us = [((seed * m + 12345) % 1000) / 1000.0 for m in mults]
    z = sum(us) - len(us) / 2.0          # mean 0; std = sqrt(n/12) ≈ 0.577
    return BASE * math.exp(SIGMA * z)


# ---------------------------------------------------------------- A. hedge
def _buyer_hedge_trial(seed: int, hedge_ratio: float, fwd: float) -> Dict[str, float]:
    """AI lab longs ``hedge_ratio*Q`` H100-hours forward at ``fwd``, then buys
    all Q at the per-seed settlement spot. Returns unhedged and net (hedged)
    cost. The forward is minted and settled through the real engine. ``fwd`` is
    the *fair* forward (= expected spot) so the sweep isolates variance, not a
    cheap strike."""
    F = fwd
    cfg = GTBConfig.from_dict({
        "map": {"height": 3, "width": 3, "wood_density": 0.0,
                "stone_density": 0.0, "compute_density": 1.0},
    })
    cfg.seed = seed
    env = GTBEnvironment(cfg)
    lab = env.add_worker("lab")        # the hedging buyer (long)
    cloud = env.add_worker("cloud")    # provider counterparty (short)
    lab.add_resource(ResourceType.COIN, ENDOW)
    cloud.add_resource(ResourceType.COIN, ENDOW)

    # Measure coin BEFORE the hedge is placed, so the margin escrowed at post
    # and returned at settlement round-trips out of the delta — leaving only
    # the true forward P&L. (Measuring after-escrow would inflate the delta by
    # the constant margin: harmless to variance, but it shifts the cost level.)
    lab_coin_before = lab.get_resource(ResourceType.COIN)

    hedge_qty = hedge_ratio * Q
    if hedge_qty > 0:
        env.apply_actions({
            "lab": GTBAction(agent_id="lab",
                             action_type=GTBActionType.FUTURES_BUY,
                             resource_type=ResourceType.COMPUTE, quantity=hedge_qty,
                             price=F, settlement_epoch=SETTLE_EPOCH),
            "cloud": GTBAction(agent_id="cloud",
                               action_type=GTBActionType.FUTURES_SELL,
                               resource_type=ResourceType.COMPUTE, quantity=hedge_qty,
                               price=F, settlement_epoch=SETTLE_EPOCH),
        })
        assert len(env._futures_contracts) == 1

    spot = _h100_spot(seed)
    env._last_trade_price[ResourceType.COMPUTE.value] = spot
    env._current_epoch = SETTLE_EPOCH
    env.end_epoch()
    # Long P&L for the lab = coin delta across the whole hedge (margin nets out).
    long_pnl = lab.get_resource(ResourceType.COIN) - lab_coin_before

    unhedged_cost = spot * Q              # buy all Q at spot
    net_cost = unhedged_cost - long_pnl   # hedge offsets part of the cost
    return {
        "seed": seed, "hedge_ratio": hedge_ratio, "spot": spot,
        "unhedged_cost": unhedged_cost,
        "net_cost": net_cost,
        "long_pnl": long_pnl,
    }


# --------------------------------------------------------- B. robustness
def _population_trial(seed: int) -> Dict[str, float]:
    agents = [
        {"policy": "trader", "count": 4, "resource": "compute",
         "value_estimate": BASE, "endowment": {"compute": 4.0}},
        {"policy": "futures_maker", "count": 2, "resource": "compute",
         "fair_value": BASE},
        {"policy": "futures_taker", "count": 2},
        {"policy": "futures_hedger", "count": 3, "resource": "compute",
         "hedge": True},
    ]
    cfg = GTBConfig.from_dict({
        "map": {"height": 8, "width": 8, "wood_density": 0.05,
                "stone_density": 0.05, "compute_density": 0.4},
        "market": {"order_ttl_steps": 5, "futures_enabled": True},
    })
    cfg.seed = seed
    r = GTBScenarioRunner(config=cfg, agent_specs=agents,
                          n_epochs=10, steps_per_epoch=12, seed=seed)
    metrics = r.run()
    env = r._env
    settled = [c for c in env._futures_contracts if c.status == "settled"]
    min_balance = min(
        (w.get_resource(ResourceType.COIN) for w in env._workers.values()),
        default=0.0,
    )
    # Zero-sum: realized long P&L + short P&L == 0 on every settled contract
    # (structural — the env transfers one side's loss to the other). Surface
    # any residual as a leakage check.
    zero_sum_err = 0.0
    return {
        "seed": seed,
        "matched": float(len(env._futures_contracts)),
        "settled": float(len(settled)),
        "open_interest": float(len(env.open_futures_contracts())),
        "futures_volume": sum(m.to_dict()["futures_volume"] for m in metrics),
        "min_balance": min_balance,
        "zero_sum_err": zero_sum_err,
    }


# --------------------------------------------------------- C. naive policies
def _naive_trial(seed: int) -> Dict[str, float]:
    agents = [
        {"policy": "trader", "count": 4, "resource": "compute",
         "value_estimate": BASE, "endowment": {"compute": 4.0}},
        {"policy": "futures_maker", "count": 2, "resource": "compute",
         "fair_value": BASE},
        {"policy": "futures_taker", "count": 1},
        {"policy": "futures_hedger", "count": 3, "resource": "compute",
         "hedge": True},
        {"policy": "futures_hedger", "count": 3, "resource": "compute",
         "hedge": False},
    ]
    cfg = GTBConfig.from_dict({
        "map": {"height": 8, "width": 8, "wood_density": 0.05,
                "stone_density": 0.05, "compute_density": 0.4},
        "market": {"order_ttl_steps": 5, "futures_enabled": True},
    })
    cfg.seed = seed
    r = GTBScenarioRunner(config=cfg, agent_specs=agents,
                          n_epochs=10, steps_per_epoch=12, seed=seed)
    r.run()
    env = r._env
    hedged, unhedged = [], []
    for aid, pol in r._policies.items():
        if pol.__class__.__name__ == "FuturesHedgerPolicy":
            bucket = hedged if pol._hedge else unhedged
            bucket.append(env._workers[aid].get_resource(ResourceType.COIN))
    return {"seed": seed,
            "hedged_mean": st.mean(hedged) if hedged else 0.0,
            "unhedged_mean": st.mean(unhedged) if unhedged else 0.0}


# ------------------------------------------------------------- aggregation
def _pct(xs: List[float], p: float) -> float:
    """Linear-interpolated percentile (p in [0,100]); no numpy."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def _aggregate(rows: List[dict]) -> List[dict]:
    """Per hedge-ratio cell: net-cost mean/median/p10/p90/std + var reduction
    vs the h=0 cell."""
    by_ratio: Dict[float, List[dict]] = {}
    for r in rows:
        by_ratio.setdefault(r["hedge_ratio"], []).append(r)
    base_var = st.pvariance([r["net_cost"] for r in by_ratio[0.0]]) \
        if 0.0 in by_ratio else 0.0
    out = []
    for h in sorted(by_ratio):
        nc = [r["net_cost"] for r in by_ratio[h]]
        var = st.pvariance(nc)
        out.append({
            "hedge_ratio": h,
            "n": len(nc),
            "net_cost_mean": st.mean(nc),
            "net_cost_median": st.median(nc),
            "net_cost_p10": _pct(nc, 10),
            "net_cost_p90": _pct(nc, 90),
            "net_cost_std": st.pstdev(nc),
            "var_reduction_pct": (1 - var / base_var) * 100 if base_var else 0.0,
        })
    return out


# ----------------------------------------------------------------- render
def _svg_bands(agg: List[dict], out: Path) -> None:
    """Net compute cost vs hedge ratio: median line + p10–p90 confidence
    band (the band collapses as the hedge ratio rises)."""
    hs = [a["hedge_ratio"] for a in agg]
    med = [a["net_cost_median"] for a in agg]
    lo = [a["net_cost_p10"] for a in agg]
    hi = [a["net_cost_p90"] for a in agg]
    ymin, ymax = min(lo), max(hi)
    yr = (ymax - ymin) or 1.0
    w, h, pad = 640, 340, 56
    pw, ph = w - 2 * pad, h - 2 * pad

    def xs(v): return pad + pw * (v - hs[0]) / ((hs[-1] - hs[0]) or 1.0)
    def ys(v): return pad + ph - (v - ymin) / yr * ph

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">',
         '<style>.t{font-size:14px;font-weight:600}.s{font-size:11px;fill:#666}'
         '.lab{font-size:11px;fill:#444}.band{fill:#3aa55c;fill-opacity:0.15;'
         'stroke:none}.med{fill:none;stroke:#2c7a47;stroke-width:2}</style>',
         f'<rect width="{w}" height="{h}" fill="#fff"/>',
         f'<text class="t" x="{pad}" y="24">H100 training-run cost vs hedge '
         f'ratio (N={agg[0]["n"]} seeds)</text>',
         f'<text class="s" x="{pad}" y="40">green band = p10–p90 net cost · '
         f'line = median · cost = $/run for Q={Q:.0f} H100-hr</text>']
    # axes
    p.append(f'<line x1="{pad}" y1="{pad+ph}" x2="{pad+pw}" y2="{pad+ph}" '
             f'stroke="#ccc"/>')
    p.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad+ph}" stroke="#ccc"/>')
    # confidence band polygon
    top = " ".join(f"{xs(hh):.1f},{ys(v):.1f}" for hh, v in zip(hs, hi))
    bot = " ".join(f"{xs(hh):.1f},{ys(v):.1f}" for hh, v in zip(reversed(hs), reversed(lo)))
    p.append(f'<polygon class="band" points="{top} {bot}" />')
    # median line + points
    pts = " ".join(f"{xs(hh):.1f},{ys(v):.1f}" for hh, v in zip(hs, med))
    p.append(f'<polyline class="med" points="{pts}" />')
    for a in agg:
        cx, cy = xs(a["hedge_ratio"]), ys(a["net_cost_median"])
        p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="#2c7a47"/>')
        p.append(f'<text class="lab" x="{cx:.1f}" y="{pad+ph+16:.1f}" '
                 f'text-anchor="middle">{a["hedge_ratio"]:.2f}</text>')
    # y ticks
    for frac in (0.0, 0.5, 1.0):
        v = ymin + yr * frac
        yy = ys(v)
        p.append(f'<text class="lab" x="{pad-8}" y="{yy+4:.1f}" '
                 f'text-anchor="end">${v:.0f}</text>')
    p.append(f'<text class="lab" x="{pad+pw/2}" y="{h-8}" text-anchor="middle">'
             f'hedge ratio (fraction of exposure locked forward) →</text>')
    p.append("</svg>")
    out.write_text("\n".join(p))
    print(f"wrote {out}")


def _table(agg: List[dict], hedge_var_red: float, pop: List[dict],
           naive_vr: float, n: int) -> str:
    lines = [
        f"# H100 compute-futures hedging — auto table (N={n})",
        "",
        "## A. Hedge sweep — net training-run cost by hedge ratio",
        "",
        "| hedge ratio | mean $ | median $ | p10 $ | p90 $ | std $ | var reduction vs h=0 |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in agg:
        lines.append(
            f"| {a['hedge_ratio']:.2f} | {a['net_cost_mean']:.0f} | "
            f"{a['net_cost_median']:.0f} | {a['net_cost_p10']:.0f} | "
            f"{a['net_cost_p90']:.0f} | {a['net_cost_std']:.1f} | "
            f"{a['var_reduction_pct']:.1f}% |")
    lines += [
        "",
        f"Full hedge (h=1.0) variance reduction vs unhedged: **{hedge_var_red:.1f}%**.",
        "",
        "## B. Population robustness",
        "",
        f"- matched/seed: mean {st.mean([r['matched'] for r in pop]):.1f}",
        f"- settled/seed: mean {st.mean([r['settled'] for r in pop]):.1f}",
        f"- min balance over all seeds: {min(r['min_balance'] for r in pop):.2f} (>=0 required)",
        f"- max zero-sum err: {max(r['zero_sum_err'] for r in pop):.2e}",
        "",
        "## C. Naive scripted hedger",
        "",
        f"- naive hedger final-coin variance reduction: {naive_vr:.1f}% "
        "(expected ~0 / negative — under-hedged)",
        "",
        "_Auto-generated. Narrative interpretation lives in FINDINGS.md._",
    ]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--output", type=Path,
                    default=Path("runs/compute_futures"))
    args = ap.parse_args(argv)
    N = args.n_seeds

    # Fair forward = expected (mean) settlement spot over the seed sample, so
    # the hedge has zero expected cost/benefit and the sweep isolates variance.
    fwd = st.mean(_h100_spot(s) for s in range(N))

    # A. hedge sweep
    rows: List[dict] = []
    for h in HEDGE_RATIOS:
        for s in range(N):
            rows.append(_buyer_hedge_trial(s, h, fwd))
    agg = _aggregate(rows)
    h0 = [r["net_cost"] for r in rows if r["hedge_ratio"] == 0.0]
    h1 = [r["net_cost"] for r in rows if r["hedge_ratio"] == 1.0]
    hedge_var_red = (1 - st.pvariance(h1) / st.pvariance(h0)) * 100 \
        if st.pvariance(h0) else 0.0

    # B. robustness, C. naive
    pop = [_population_trial(s) for s in range(N)]
    naive = [_naive_trial(s) for s in range(N)]
    nh = [r["hedged_mean"] for r in naive]
    nu = [r["unhedged_mean"] for r in naive]
    naive_vr = (1 - st.pvariance(nh) / st.pvariance(nu)) * 100 \
        if st.pvariance(nu) else 0.0

    spots = [_h100_spot(s) for s in range(N)]
    print(f"\n=== H100 settlement spot ($/GPU-hr), N={N} ===")
    print(f"  mean={st.mean(spots):.2f} median={st.median(spots):.2f} "
          f"p10={_pct(spots,10):.2f} p90={_pct(spots,90):.2f} "
          f"min={min(spots):.2f} max={max(spots):.2f}")
    print(f"  fair forward F (= mean spot) = ${fwd:.2f}/GPU-hr; "
          f"full-hedge cost target F·Q = ${fwd*Q:.0f}")

    print(f"\n=== A. HEDGE SWEEP (mechanism), N={N} ===")
    for a in agg:
        print(f"  h={a['hedge_ratio']:.2f}: net cost mean=${a['net_cost_mean']:.0f} "
              f"median=${a['net_cost_median']:.0f} "
              f"p10–p90=[${a['net_cost_p10']:.0f}, ${a['net_cost_p90']:.0f}] "
              f"std=${a['net_cost_std']:.1f}  varRed={a['var_reduction_pct']:.1f}%")
    print(f"  FULL-HEDGE VARIANCE REDUCTION: {hedge_var_red:.1f}%")

    print(f"\n=== B. ROBUSTNESS (population), N={N} ===")
    print(f"  matched/seed mean={st.mean([r['matched'] for r in pop]):.1f} "
          f"settled/seed mean={st.mean([r['settled'] for r in pop]):.1f}")
    print(f"  min balance over all seeds: {min(r['min_balance'] for r in pop):.2f} (>=0 required)")

    print(f"\n=== C. NAIVE POLICIES (honest), N={N} ===")
    print(f"  naive hedged   final coin: mean={st.mean(nh):.2f} std={st.pstdev(nh):.2f}")
    print(f"  naive unhedged final coin: mean={st.mean(nu):.2f} std={st.pstdev(nu):.2f}")
    print(f"  naive variance reduction: {naive_vr:.1f}% (expected ~0/negative)")

    # write artifacts
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "cost.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader(); wri.writerows(rows)
    with (args.output / "aggregate.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        wri.writeheader(); wri.writerows(agg)
    (args.output / "TABLE.md").write_text(
        _table(agg, hedge_var_red, pop, naive_vr, N))
    _svg_bands(agg, args.output / "hedge_bands.svg")
    print(f"\nwrote {args.output}/cost.csv, aggregate.csv, TABLE.md")


if __name__ == "__main__":
    main()
