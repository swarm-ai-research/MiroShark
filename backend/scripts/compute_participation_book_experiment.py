"""Emergent participation through the real order book (bd 6qu, ← umm).

Phase-1..4 measured dealer-intermediation *participation* in a standalone
Monte-Carlo. This reproduces it **emergently through the vendored order book**,
now that the machinery exists: a ``RiskAwareDealerPolicy`` (bd rzt) makes a
two-sided SKU market whose half-spread widens with the inventory it warehouses,
and a heterogeneous two-sided principal population crosses those quotes **only
when the spread is within its reservation**. Volume is therefore not imposed — it
emerges from real quoting, matching, and the dealer's inventory feedback.

Mechanism (the phase-1 story, now microstructural):

- The dealer quotes bid/ask on SKU "a" at ``anchor·(1 ∓ spread)`` with
  ``spread = base + risk_factor·|net inventory|``. ``risk_factor`` proxies the
  basis residual (∝ 1 − r_i): a worse basis ⇒ the dealer widens faster per unit
  of warehoused inventory.
- Principal-longs lift the ask, principal-shorts hit the bid — each only if the
  dealer's half-spread ≤ its private reservation ``s_max``. Shorts (neoclouds)
  withdraw with probability ``reluctance``.
- Matched longs vs shorts **net** in the dealer's book; only the net imbalance
  grows its inventory (and thus its spread). So balanced flow keeps the spread
  tight (high participation); one-sided flow lets inventory — and the spread —
  run, pricing principals out.

Sweep ``risk_factor`` (basis proxy) × ``reluctance``; measure emergent volume
share and realized spread. Expect the phase-1 result to reappear: balanced flow
stays liquid across basis, the collapse lives in the one-sided regime.

Usage::

    uv run python -m scripts.compute_participation_book_experiment --n-seeds 50

Writes trials.csv, aggregate.csv, TABLE.md, participation_book.svg. FINDINGS curated.
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

from worlds.gather_trade_build.agents import RiskAwareDealerPolicy  # noqa: E402
from worlds.gather_trade_build.config import GTBConfig  # noqa: E402
from worlds.gather_trade_build.entities import (  # noqa: E402
    GTBActionType, ResourceType,
)
from worlds.gather_trade_build.env import GTBAction, GTBEnvironment  # noqa: E402
from scripts.compute_basis_experiment import _u, _z  # noqa: E402

F = 3.50            # anchor spot / fair forward
N_PRINCIPALS = 12
T_STEPS = 48
HORIZON = 3
BASE_SPREAD = 0.04
RISK_FACTORS = [0.0, 0.02, 0.05, 0.10, 0.20]   # basis proxy (∝ 1−r_i)
RELUCTANCE = [0.0, 1.0]                          # balanced vs one-sided
SETTLE = HORIZON                                 # epoch stays 0; settle = horizon
SKU = "a"
BIG = 10_000_000.0


def _principals(seed: int):
    """Heterogeneous two-sided principals: side, reservation half-spread."""
    ps = []
    for j in range(N_PRINCIPALS):
        side = 1 if _u(seed, j, 11) < 0.5 else -1
        s_max = 0.06 + 0.30 * _u(seed, j, 22)       # reservation ∈ [0.06, 0.36]
        ps.append({"id": f"p{j}", "side": side, "s_max": s_max,
                   "withdraw_u": _u(seed, j, 60), "crossed": False,
                   "faced": None})
    return ps


def _trial(seed: int, risk_factor: float, reluctance: float) -> Dict[str, float]:
    cfg = GTBConfig.from_dict({
        "map": {"height": 3, "width": 3, "wood_density": 0.0,
                "stone_density": 0.0, "compute_density": 1.0}})
    cfg.seed = seed
    env = GTBEnvironment(cfg)
    env.add_worker("dealer"); env._workers["dealer"].add_resource(ResourceType.COIN, BIG)
    env.add_worker("cp"); env._workers["cp"].add_resource(ResourceType.COIN, BIG)
    ps = _principals(seed)
    for p in ps:
        env.add_worker(p["id"])
        env._workers[p["id"]].add_resource(ResourceType.COIN, BIG)
    env._last_trade_price[ResourceType.COMPUTE.value] = F

    dealer = RiskAwareDealerPolicy(
        "dealer", seed=seed, sku=SKU, horizon=HORIZON, base_spread=BASE_SPREAD,
        risk_factor=risk_factor, hedge_ratio=1.0, qty=1.0, fair_value=F)

    sku_key = f"compute/{SKU}@{SETTLE}"
    for step in range(T_STEPS):
        env._current_step = step
        env.apply_actions({"dealer": dealer.decide(env.obs("dealer"))})
        # Principals attempt IN ORDER, re-reading the book before each so a
        # consumed one-lot quote isn't double-lifted: each incremental crosser
        # faces the dealer's inventory-widened spread (one cross per lot).
        for p in ps:
            if p["crossed"]:
                continue
            willing = p["side"] == 1 or p["withdraw_u"] >= reluctance
            if not willing:
                continue
            q = env._futures_curve().get(sku_key) or {}
            if p["side"] == 1:
                ba = q.get("best_ask")
                if ba is None:
                    continue
                p["faced"] = ba / F - 1.0                   # half-spread faced now
                if ba <= F * (1.0 + p["s_max"]):
                    env.apply_actions({p["id"]: GTBAction(agent_id=p["id"],
                        action_type=GTBActionType.FUTURES_BUY,
                        resource_type=ResourceType.COMPUTE, quantity=1.0,
                        price=ba, settlement_epoch=SETTLE, sku=SKU)})
                    p["crossed"] = True
            else:
                bb = q.get("best_bid")
                if bb is None:
                    continue
                p["faced"] = 1.0 - bb / F
                if bb >= F * (1.0 - p["s_max"]):
                    env.apply_actions({p["id"]: GTBAction(agent_id=p["id"],
                        action_type=GTBActionType.FUTURES_SELL,
                        resource_type=ResourceType.COMPUTE, quantity=1.0,
                        price=bb, settlement_epoch=SETTLE, sku=SKU)})
                    p["crossed"] = True
        # cp takes the other side of the dealer's basket hedge (deep-pocketed).
        bq = (env._futures_curve().get(f"compute@{SETTLE}") or {})
        if bq.get("best_bid") is not None:                 # dealer long-basket bid
            env.apply_actions({"cp": GTBAction(agent_id="cp",
                action_type=GTBActionType.FUTURES_SELL,
                resource_type=ResourceType.COMPUTE, quantity=1.0,
                price=bq["best_bid"], settlement_epoch=SETTLE)})
        if bq.get("best_ask") is not None:                 # dealer short-basket ask
            env.apply_actions({"cp": GTBAction(agent_id="cp",
                action_type=GTBActionType.FUTURES_BUY,
                resource_type=ResourceType.COMPUTE, quantity=1.0,
                price=bq["best_ask"], settlement_epoch=SETTLE)})

    crossed = sum(1 for p in ps if p["crossed"])
    willing_n = sum(1 for p in ps
                    if p["side"] == 1 or p["withdraw_u"] >= reluctance)
    faced = [p["faced"] for p in ps if p["faced"] is not None]
    dealer_pos = env.obs("dealer")["own_futures_position"]
    net_sku = dealer_pos.get(f"compute/{SKU}@{SETTLE}", 0.0)
    net_basket = dealer_pos.get(f"compute@{SETTLE}", 0.0)
    return {
        "seed": seed, "risk_factor": risk_factor, "reluctance": reluctance,
        "volume_share": crossed / N_PRINCIPALS,
        "willing_clear_frac": crossed / willing_n if willing_n else 0.0,
        "mean_faced_spread": st.mean(faced) if faced else 0.0,
        "abs_net_sku": abs(net_sku),
        "hedge_coverage": (abs(net_basket) / abs(net_sku)) if abs(net_sku) > 1e-9 else 1.0,
    }


def _aggregate(rows: List[dict]) -> List[dict]:
    cells: Dict[tuple, List[dict]] = {}
    for r in rows:
        cells.setdefault((r["reluctance"], r["risk_factor"]), []).append(r)
    out = []
    for (rel, rf) in sorted(cells):
        c = cells[(rel, rf)]
        vs = [r["volume_share"] for r in c]
        out.append({
            "reluctance": rel, "risk_factor": rf, "n": len(c),
            "volume_share_median": st.median(vs),
            "volume_share_p10": _pct(vs, 10), "volume_share_p90": _pct(vs, 90),
            "willing_clear_frac_mean": st.mean(r["willing_clear_frac"] for r in c),
            "mean_faced_spread": st.mean(r["mean_faced_spread"] for r in c),
            "abs_net_sku_mean": st.mean(r["abs_net_sku"] for r in c),
            "hedge_coverage_mean": st.mean(r["hedge_coverage"] for r in c),
        })
    return out


def _pct(xs, q):
    ys = sorted(xs)
    if not ys:
        return 0.0
    i = (q / 100.0) * (len(ys) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return ys[lo] + (ys[hi] - ys[lo]) * (i - lo)


_REL_COLOR = {0.0: "#2c7a47", 1.0: "#b3341f"}


def _svg(agg: List[dict], out: Path, n: int) -> None:
    finite = RISK_FACTORS
    w, h, pad = 680, 400, 66
    pw, ph = w - 2 * pad, h - 2 * pad
    x0, x1 = min(finite), max(finite)

    def xs(r): return pad + pw * (r - x0) / ((x1 - x0) or 1.0)
    def ys(v): return pad + ph - v * ph

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">',
         '<style>.t{font-size:14px;font-weight:600}.s{font-size:11px;fill:#666}'
         '.lab{font-size:11px;fill:#444}.med{fill:none;stroke-width:2}</style>',
         f'<rect width="{w}" height="{h}" fill="#fff"/>',
         f'<text class="t" x="{pad}" y="26">Emergent participation through the '
         f'book: volume share vs basis proxy (N={n})</text>',
         f'<text class="s" x="{pad}" y="43">band = p10–p90 · risk_factor ∝ '
         f'(1−r_i) · balanced stays liquid, one-sided collapses</text>']
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = ys(frac)
        p.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{pad+pw}" y2="{y:.1f}" '
                 f'stroke="#eee"/>')
        p.append(f'<text class="lab" x="{pad-8}" y="{y+4:.1f}" '
                 f'text-anchor="end">{frac:.0%}</text>')
    for r in finite:
        p.append(f'<text class="lab" x="{xs(r):.1f}" y="{pad+ph+18:.1f}" '
                 f'text-anchor="middle">{r:g}</text>')
    by_rel: Dict[float, List[dict]] = {}
    for a in agg:
        by_rel.setdefault(a["reluctance"], []).append(a)
    for rel, series in sorted(by_rel.items()):
        series.sort(key=lambda a: a["risk_factor"])
        color = _REL_COLOR.get(rel, "#555")
        top = " ".join(f"{xs(a['risk_factor']):.1f},{ys(a['volume_share_p90']):.1f}"
                       for a in series)
        bot = " ".join(f"{xs(a['risk_factor']):.1f},{ys(a['volume_share_p10']):.1f}"
                       for a in reversed(series))
        p.append(f'<polygon points="{top} {bot}" fill="{color}" '
                 f'fill-opacity="0.13" stroke="none"/>')
        line = " ".join(f"{xs(a['risk_factor']):.1f},{ys(a['volume_share_median']):.1f}"
                        for a in series)
        p.append(f'<polyline class="med" points="{line}" stroke="{color}"/>')
        for a in series:
            p.append(f'<circle cx="{xs(a["risk_factor"]):.1f}" '
                     f'cy="{ys(a["volume_share_median"]):.1f}" r="3" fill="{color}"/>')
    ly = pad + 14
    for rel in sorted(by_rel):
        color = _REL_COLOR.get(rel, "#555")
        lbl = "balanced (reluctance 0)" if rel == 0.0 else "one-sided (reluctance 1)"
        p.append(f'<rect x="{pad+pw-186}" y="{ly-9}" width="12" height="12" '
                 f'fill="{color}" fill-opacity="0.6"/>')
        p.append(f'<text class="lab" x="{pad+pw-170}" y="{ly+1}">{lbl}</text>')
        ly += 18
    p.append(f'<text class="lab" x="{pad+pw/2}" y="{h-12}" text-anchor="middle">'
             f'risk_factor (basis proxy ∝ 1−r_i) →</text>')
    p.append("</svg>")
    out.write_text("\n".join(p))
    print(f"wrote {out}")


def _table(agg: List[dict], n: int) -> str:
    lines = [
        f"# Emergent participation through the book — auto table (N={n})", "",
        "Dealer + two-sided principals through the real order book. risk_factor "
        "= spread widening per unit inventory (∝ 1−r_i basis proxy). "
        "reluctance = short-side withdrawal.", "",
        "| reluctance | risk_factor | volume share (p10–p90) | willing clear | "
        "faced spread | dealer |net SKU| | hedge cov |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in sorted(agg, key=lambda a: (a["reluctance"], a["risk_factor"])):
        lines.append(
            f"| {a['reluctance']:g} | {a['risk_factor']:g} | "
            f"{a['volume_share_median']:.2f} "
            f"({a['volume_share_p10']:.2f}–{a['volume_share_p90']:.2f}) | "
            f"{a['willing_clear_frac_mean']:.2f} | "
            f"{a['mean_faced_spread']:.3f} | {a['abs_net_sku_mean']:.2f} | "
            f"{a['hedge_coverage_mean']:.2f} |")
    lines += ["", "_Auto-generated. Narrative interpretation lives in FINDINGS.md._"]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--output", type=Path,
                    default=Path("runs/compute_participation_book"))
    args = ap.parse_args(argv)
    N = args.n_seeds

    rows: List[dict] = []
    for rel in RELUCTANCE:
        for rf in RISK_FACTORS:
            for s in range(N):
                rows.append(_trial(s, rf, rel))
    agg = _aggregate(rows)

    print(f"\n=== Emergent participation through the book, N={N} ===")
    for rel in RELUCTANCE:
        print(f"\n  reluctance={rel:g}:  risk_factor -> volume share")
        for a in sorted((a for a in agg if a["reluctance"] == rel),
                        key=lambda a: a["risk_factor"]):
            print(f"    rf={a['risk_factor']:g}: vol={a['volume_share_median']:.2f} "
                  f"[{a['volume_share_p10']:.2f},{a['volume_share_p90']:.2f}] "
                  f"faced_spread={a['mean_faced_spread']:.3f} "
                  f"|net|={a['abs_net_sku_mean']:.2f} "
                  f"hedge_cov={a['hedge_coverage_mean']:.2f}")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trials.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader(); wri.writerows(rows)
    with (args.output / "aggregate.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        wri.writeheader(); wri.writerows(agg)
    (args.output / "TABLE.md").write_text(_table(agg, N))
    _svg(agg, args.output / "participation_book.svg", N)
    print(f"\nwrote {args.output}/trials.csv, aggregate.csv, TABLE.md")


if __name__ == "__main__":
    main()
