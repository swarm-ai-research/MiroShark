"""H100 hedging *through the order book*, with slippage (bd 5ij, caps k9w).

bd k9w left an A<->C gap: check A proved the compute-futures *mechanism* is a
perfect linear hedge (up to 100% variance reduction), but it minted one matched
forward in isolation — no execution friction. This experiment closes the gap by
hedging the *same* H100 cost exposure against a **finite-depth resting forward
book**, through the real matching and settlement engine, and measures how much
of the 100% ceiling survives once you must pay the spread and walk the book.

Setup (controlled, deterministic — no RNG, fully reproducible):

- A producer faces a known exposure: it will sell Q=100 H100-hr at the per-seed
  settlement spot ``P(seed)`` — the same right-skewed log-normal price process
  as k9w (median $3.50/GPU-hr).
- A forward dealer rests a *bid ladder*: ``n_levels`` price levels descending
  from the best bid (fair forward F minus a half-spread), total depth =
  ``liquidity * Q`` units. Shallow book => the producer can only hedge part of
  Q; deep book => it walks down into worse levels (depth slippage).
- The producer runs the **MatchedHedgerPolicy rule**: hit the best resting bid,
  sized to its exposure, top-down. Every fill goes through the real
  ``_match_futures_orders`` (fills at the resting bid) and the real
  ``end_epoch`` settlement at ``P(seed)``.

We sweep ``liquidity`` and report, across N seeds: realized hedged-revenue
variance reduction vs the unhedged control, the fraction of exposure actually
hedged, and the slippage cost (gap to the ideal full hedge at fair F). The
unhedged control has the *same* genuine exposure (P·Q), fixing the degenerate
near-inert control of k9w check C.

Usage::

    uv run python -m scripts.compute_hedge_execution_experiment --n-seeds 100

Writes revenue.csv, aggregate.csv, TABLE.md, execution_bands.svg to the output
dir. FINDINGS.md is hand-curated (CLAUDE.md methodology).
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics as st
from pathlib import Path
from typing import Dict, List, Tuple

logging.disable(logging.CRITICAL)

from worlds.gather_trade_build.config import GTBConfig  # noqa: E402
from worlds.gather_trade_build.entities import (  # noqa: E402
    GTBActionType, ResourceType,
)
from worlds.gather_trade_build.env import GTBAction, GTBEnvironment  # noqa: E402

# Reuse k9w's exact price process + percentile helper so the two studies are
# directly comparable (same H100 spot distribution).
from scripts.compute_futures_experiment import _h100_spot, _pct  # noqa: E402

Q = 100.0           # H100-hours the producer will sell at settlement spot
SIGMA_NOTE = 0.5    # (log-vol lives in _h100_spot; here for the docstring only)
SETTLE_EPOCH = 2
ENDOW = 100_000.0   # ample coin so margin/settlement never clamps
N_LEVELS = 5        # forward bid-ladder price levels
HALF_SPREAD = 0.02  # best bid sits at F*(1-HALF_SPREAD)
LEVEL_DROP = 0.02   # each deeper level is this much lower (depth slippage)
LIQUIDITY = [0.25, 0.5, 1.0, 2.0, 4.0]   # forward depth as a multiple of Q


def _ladder(fwd: float, depth: float) -> List[Tuple[float, float]]:
    """Resting forward bid ladder, best-first: ``N_LEVELS`` levels, each priced
    a notch below the last, splitting ``depth`` units evenly."""
    lot = depth / N_LEVELS
    return [(fwd * (1.0 - HALF_SPREAD - k * LEVEL_DROP), lot)
            for k in range(N_LEVELS)]


def _exec_trial(seed: int, liquidity: float, fwd: float) -> Dict[str, float]:
    """Hedge Q against a finite forward book through the real engine, settle at
    the per-seed spot, return unhedged vs realized-hedged revenue."""
    cfg = GTBConfig.from_dict({
        "map": {"height": 3, "width": 3, "wood_density": 0.0,
                "stone_density": 0.0, "compute_density": 1.0},
    })
    cfg.seed = seed
    env = GTBEnvironment(cfg)
    prod = env.add_worker("prod")     # the hedging producer (short)
    dealer = env.add_worker("dealer")  # forward dealer (long) absorbing the hedge
    prod.add_resource(ResourceType.COIN, ENDOW)
    dealer.add_resource(ResourceType.COIN, ENDOW)

    depth = liquidity * Q
    levels = _ladder(fwd, depth)
    # Dealer rests the whole bid ladder (no crossing sells yet -> all rest).
    for price, size in levels:
        env.apply_actions({"dealer": GTBAction(
            agent_id="dealer", action_type=GTBActionType.FUTURES_BUY,
            resource_type=ResourceType.COMPUTE, quantity=size,
            price=price, settlement_epoch=SETTLE_EPOCH)})

    # Measure coin BEFORE the producer hedges, so escrowed margin round-trips
    # out of the delta (the k9w level-bug fix) — leaving only true forward P&L.
    prod_coin_before = prod.get_resource(ResourceType.COIN)

    # Producer crosses the book top-down (MatchedHedgerPolicy rule: hit the best
    # remaining bid, sized to remaining exposure), until Q hedged or book dry.
    remaining = Q
    for price, size in levels:
        if remaining <= 1e-9:
            break
        q = min(remaining, size)
        env.apply_actions({"prod": GTBAction(
            agent_id="prod", action_type=GTBActionType.FUTURES_SELL,
            resource_type=ResourceType.COMPUTE, quantity=q,
            price=price, settlement_epoch=SETTLE_EPOCH)})  # crosses this level
        remaining -= q
    hedged_qty = Q - remaining

    spot = _h100_spot(seed)
    env._last_trade_price[ResourceType.COMPUTE.value] = spot
    env._current_epoch = SETTLE_EPOCH
    env.end_epoch()
    futures_pnl = prod.get_resource(ResourceType.COIN) - prod_coin_before

    unhedged_rev = spot * Q                 # sell all Q at spot, no hedge
    hedged_rev = unhedged_rev + futures_pnl  # spot sale + realized hedge P&L
    return {
        "seed": seed, "liquidity": liquidity, "spot": spot,
        "hedged_qty": hedged_qty,
        "unhedged_rev": unhedged_rev,
        "hedged_rev": hedged_rev,
    }


def _aggregate(rows: List[dict], fwd: float) -> List[dict]:
    """Per liquidity cell: realized hedged-revenue mean/median/p10/p90/std,
    variance reduction vs the unhedged control, coverage, slippage."""
    unh = [r["unhedged_rev"] for r in rows]
    base_var = st.pvariance(unh)
    ideal = fwd * Q                      # full hedge at fair F, no slippage
    by_liq: Dict[float, List[dict]] = {}
    for r in rows:
        by_liq.setdefault(r["liquidity"], []).append(r)
    out = []
    for liq in sorted(by_liq):
        cell = by_liq[liq]
        hr = [r["hedged_rev"] for r in cell]
        var = st.pvariance(hr)
        out.append({
            "liquidity": liq,
            "n": len(hr),
            "depth_units": liq * Q,
            "hedged_qty_mean": st.mean(r["hedged_qty"] for r in cell),
            "coverage_pct": 100.0 * st.mean(r["hedged_qty"] for r in cell) / Q,
            "hedged_rev_mean": st.mean(hr),
            "hedged_rev_median": st.median(hr),
            "hedged_rev_p10": _pct(hr, 10),
            "hedged_rev_p90": _pct(hr, 90),
            "hedged_rev_std": st.pstdev(hr),
            "var_reduction_pct": (1 - var / base_var) * 100 if base_var else 0.0,
            "slippage_cost_mean": ideal - st.mean(hr),
            "slippage_pct_of_ideal": 100.0 * (ideal - st.mean(hr)) / ideal,
        })
    return out


def _svg_bands(agg: List[dict], unh_p10: float, unh_p90: float, unh_med: float,
               ideal: float, out: Path) -> None:
    """Realized hedged revenue vs forward-book liquidity: median + p10–p90
    band (collapses as depth rises), with the unhedged p10–p90 as a gray
    reference and the ideal-full-hedge line."""
    xs_vals = [a["liquidity"] for a in agg]
    med = [a["hedged_rev_median"] for a in agg]
    lo = [a["hedged_rev_p10"] for a in agg]
    hi = [a["hedged_rev_p90"] for a in agg]
    ymin = min(min(lo), unh_p10) * 0.98
    ymax = max(max(hi), unh_p90) * 1.02
    yr = (ymax - ymin) or 1.0
    w, h, pad = 660, 360, 60
    pw, ph = w - 2 * pad, h - 2 * pad
    xlogs = [math.log(v) for v in xs_vals]
    x0, x1 = min(xlogs), max(xlogs)

    def xs(v): return pad + pw * (math.log(v) - x0) / ((x1 - x0) or 1.0)
    def ys(v): return pad + ph - (v - ymin) / yr * ph

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">',
         '<style>.t{font-size:14px;font-weight:600}.s{font-size:11px;fill:#666}'
         '.lab{font-size:11px;fill:#444}.band{fill:#3aa55c;fill-opacity:0.16;'
         'stroke:none}.unh{fill:#999;fill-opacity:0.13;stroke:none}'
         '.med{fill:none;stroke:#2c7a47;stroke-width:2}'
         '.ideal{stroke:#888;stroke-width:1;stroke-dasharray:4 3}</style>',
         f'<rect width="{w}" height="{h}" fill="#fff"/>',
         f'<text class="t" x="{pad}" y="24">H100 hedge through the book: '
         f'realized revenue vs forward depth (N={agg[0]["n"]})</text>',
         f'<text class="s" x="{pad}" y="40">green = hedged p10–p90 · gray = '
         f'unhedged p10–p90 · dashed = ideal full hedge (F·Q=${ideal:.0f})</text>']
    # unhedged reference band (flat, spans the plot)
    p.append(f'<rect class="unh" x="{pad}" y="{ys(unh_p90):.1f}" width="{pw}" '
             f'height="{(ys(unh_p10)-ys(unh_p90)):.1f}"/>')
    p.append(f'<line x1="{pad}" y1="{ys(unh_med):.1f}" x2="{pad+pw}" '
             f'y2="{ys(unh_med):.1f}" stroke="#999" stroke-dasharray="2 2"/>')
    # ideal full-hedge line
    p.append(f'<line class="ideal" x1="{pad}" y1="{ys(ideal):.1f}" '
             f'x2="{pad+pw}" y2="{ys(ideal):.1f}"/>')
    # hedged confidence band
    top = " ".join(f"{xs(a['liquidity']):.1f},{ys(v):.1f}" for a, v in zip(agg, hi))
    bot = " ".join(f"{xs(a['liquidity']):.1f},{ys(v):.1f}"
                   for a, v in zip(reversed(agg), reversed(lo)))
    p.append(f'<polygon class="band" points="{top} {bot}" />')
    pts = " ".join(f"{xs(a['liquidity']):.1f},{ys(v):.1f}" for a, v in zip(agg, med))
    p.append(f'<polyline class="med" points="{pts}" />')
    for a in agg:
        cx, cy = xs(a["liquidity"]), ys(a["hedged_rev_median"])
        p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="#2c7a47"/>')
        p.append(f'<text class="lab" x="{cx:.1f}" y="{pad+ph+16:.1f}" '
                 f'text-anchor="middle">{a["liquidity"]:.2g}×</text>')
    for frac in (0.0, 0.5, 1.0):
        v = ymin + yr * frac
        p.append(f'<text class="lab" x="{pad-8}" y="{ys(v)+4:.1f}" '
                 f'text-anchor="end">${v:.0f}</text>')
    p.append(f'<text class="lab" x="{pad+pw/2}" y="{h-10}" text-anchor="middle">'
             f'forward-book depth (× exposure Q) →</text>')
    p.append("</svg>")
    out.write_text("\n".join(p))
    print(f"wrote {out}")


def _table(agg: List[dict], unh_std: float, ideal: float, n: int) -> str:
    lines = [
        f"# H100 hedge through the order book — auto table (N={n})",
        "",
        f"Unhedged revenue std (baseline): ${unh_std:.1f}. "
        f"Ideal full hedge at fair F: ${ideal:.0f}.",
        "",
        "| fwd depth (×Q) | coverage | hedged mean $ | p10–p90 $ | std $ "
        "| var reduction | slippage $ | slippage % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in agg:
        lines.append(
            f"| {a['liquidity']:.2g} | {a['coverage_pct']:.0f}% | "
            f"{a['hedged_rev_mean']:.0f} | "
            f"{a['hedged_rev_p10']:.0f}–{a['hedged_rev_p90']:.0f} | "
            f"{a['hedged_rev_std']:.1f} | {a['var_reduction_pct']:.1f}% | "
            f"{a['slippage_cost_mean']:.1f} | {a['slippage_pct_of_ideal']:.2f}% |")
    lines += ["", "_Auto-generated. Narrative interpretation lives in FINDINGS.md._"]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--output", type=Path,
                    default=Path("runs/compute_hedge_execution"))
    args = ap.parse_args(argv)
    N = args.n_seeds

    fwd = st.mean(_h100_spot(s) for s in range(N))   # fair forward = mean spot
    ideal = fwd * Q

    rows: List[dict] = []
    for liq in LIQUIDITY:
        for s in range(N):
            rows.append(_exec_trial(s, liq, fwd))
    agg = _aggregate(rows, fwd)

    unh = [r["unhedged_rev"] for r in rows if r["liquidity"] == LIQUIDITY[0]]
    unh_std = st.pstdev(unh)
    unh_med = st.median(unh)
    unh_p10, unh_p90 = _pct(unh, 10), _pct(unh, 90)

    print(f"\n=== H100 hedge through the book, N={N} ===")
    print(f"  fair forward F=${fwd:.2f}/GPU-hr; ideal full hedge F·Q=${ideal:.0f}")
    print(f"  unhedged revenue: mean=${st.mean(unh):.0f} std=${unh_std:.1f} "
          f"p10–p90=[${unh_p10:.0f}, ${unh_p90:.0f}]")
    print()
    for a in agg:
        print(f"  depth={a['liquidity']:.2g}×Q: coverage={a['coverage_pct']:.0f}% "
              f"hedged mean=${a['hedged_rev_mean']:.0f} "
              f"p10–p90=[${a['hedged_rev_p10']:.0f}, ${a['hedged_rev_p90']:.0f}] "
              f"std=${a['hedged_rev_std']:.1f}  varRed={a['var_reduction_pct']:.1f}% "
              f"slippage=${a['slippage_cost_mean']:.1f} "
              f"({a['slippage_pct_of_ideal']:.2f}%)")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "revenue.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader(); wri.writerows(rows)
    with (args.output / "aggregate.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        wri.writeheader(); wri.writerows(agg)
    (args.output / "TABLE.md").write_text(_table(agg, unh_std, ideal, N))
    _svg_bands(agg, unh_p10, unh_p90, unh_med, ideal,
               args.output / "execution_bands.svg")
    print(f"\nwrote {args.output}/revenue.csv, aggregate.csv, TABLE.md")


if __name__ == "__main__":
    main()
