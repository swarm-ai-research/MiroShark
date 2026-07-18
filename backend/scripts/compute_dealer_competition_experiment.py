"""Do competing dealers restore balanced-flow basis-invariance? (bd hka, ← 6qu/4tw).

The through-the-book participation study (bd 6qu) found that a *single* sequential
dealer breaks the Monte-Carlo's "balanced flow is basis-invariant" result: real-
time netting is imperfect, the dealer's inventory random-walks, its spread spikes
in transient-imbalance windows, and low-reservation principals get priced out —
so balanced volume falls with the basis proxy. Open question: does **dealer
competition** restore the MC upper bound?

Setup: ``n_dealers`` RiskAwareDealerPolicy makers quote the *same* SKU. The book
aggregates their quotes, so a principal always crosses the **best** ask/bid
across dealers. Each dealer widens its *own* spread on its *own* inventory
(``own_futures_position``), so flow **load-balances** to the least-filled dealer:
per-dealer |net| shrinks ≈ 1/n_dealers, the aggregate best quote stays tighter,
and fewer principals are priced out.

Sweep ``n_dealers`` × ``risk_factor`` × ``reluctance``. Expect: more dealers ⇒
higher volume and balanced-flow basis-sensitivity shrinking toward the MC's
invariance; one-sided flow still collapses (every dealer accumulates the same
direction) but less steeply.

Usage::

    uv run python -m scripts.compute_dealer_competition_experiment --n-seeds 50

Writes trials.csv, aggregate.csv, TABLE.md, competition_curve.svg. FINDINGS curated.
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
from scripts.compute_participation_book_experiment import (  # noqa: E402
    F, N_PRINCIPALS, T_STEPS, HORIZON, BASE_SPREAD, SETTLE, SKU, BIG, _principals,
)
from scripts.compute_basis_experiment import _u  # noqa: E402

N_DEALERS = [1, 2, 4]
RISK_FACTORS = [0.0, 0.05, 0.10, 0.20]
RELUCTANCE = [0.0, 1.0]
SKU_KEY = f"compute/{SKU}@{SETTLE}"
BASKET_KEY = f"compute@{SETTLE}"


def _trial(seed: int, n_dealers: int, risk_factor: float,
           reluctance: float) -> Dict[str, float]:
    cfg = GTBConfig.from_dict({
        "map": {"height": 3, "width": 3, "wood_density": 0.0,
                "stone_density": 0.0, "compute_density": 1.0}})
    cfg.seed = seed
    env = GTBEnvironment(cfg)
    dealers = []
    for k in range(n_dealers):
        did = f"dealer{k}"
        env.add_worker(did)
        env._workers[did].add_resource(ResourceType.COIN, BIG)
        dealers.append(RiskAwareDealerPolicy(
            did, seed=seed * 100 + k, sku=SKU, horizon=HORIZON,
            base_spread=BASE_SPREAD, risk_factor=risk_factor, hedge_ratio=1.0,
            qty=1.0, fair_value=F))
    env.add_worker("cp"); env._workers["cp"].add_resource(ResourceType.COIN, BIG)
    ps = _principals(seed)
    for p in ps:
        env.add_worker(p["id"])
        env._workers[p["id"]].add_resource(ResourceType.COIN, BIG)
    env._last_trade_price[ResourceType.COMPUTE.value] = F

    for step in range(T_STEPS):
        env._current_step = step
        # All dealers quote (staggered by seed so they don't post identically).
        env.apply_actions({d.agent_id: d.decide(env.obs(d.agent_id))
                           for d in dealers})
        for p in ps:
            if p["crossed"]:
                continue
            willing = p["side"] == 1 or p["withdraw_u"] >= reluctance
            if not willing:
                continue
            q = env._futures_curve().get(SKU_KEY) or {}
            if p["side"] == 1:
                ba = q.get("best_ask")           # best across all dealers
                if ba is None:
                    continue
                p["faced"] = ba / F - 1.0
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
        # cp provides basket liquidity for every dealer's hedge.
        bq = env._futures_curve().get(BASKET_KEY) or {}
        if bq.get("best_bid") is not None:
            env.apply_actions({"cp": GTBAction(agent_id="cp",
                action_type=GTBActionType.FUTURES_SELL,
                resource_type=ResourceType.COMPUTE, quantity=1.0,
                price=bq["best_bid"], settlement_epoch=SETTLE)})
        if bq.get("best_ask") is not None:
            env.apply_actions({"cp": GTBAction(agent_id="cp",
                action_type=GTBActionType.FUTURES_BUY,
                resource_type=ResourceType.COMPUTE, quantity=1.0,
                price=bq["best_ask"], settlement_epoch=SETTLE)})

    crossed = sum(1 for p in ps if p["crossed"])
    # Per-dealer net SKU inventory (dispersion shows load-balancing).
    nets = []
    for d in dealers:
        pos = env.obs(d.agent_id)["own_futures_position"]
        nets.append(abs(pos.get(SKU_KEY, 0.0)))
    faced = [p["faced"] for p in ps if p["faced"] is not None]
    return {"seed": seed, "n_dealers": n_dealers, "risk_factor": risk_factor,
            "reluctance": reluctance,
            "volume_share": crossed / N_PRINCIPALS,
            "mean_faced_spread": st.mean(faced) if faced else 0.0,
            "max_dealer_net": max(nets) if nets else 0.0,
            "mean_dealer_net": st.mean(nets) if nets else 0.0}


def _aggregate(rows: List[dict]) -> List[dict]:
    cells: Dict[tuple, List[dict]] = {}
    for r in rows:
        cells.setdefault((r["reluctance"], r["n_dealers"], r["risk_factor"]), []).append(r)
    out = []
    for key in sorted(cells):
        c = cells[key]
        vs = [r["volume_share"] for r in c]
        out.append({
            "reluctance": key[0], "n_dealers": key[1], "risk_factor": key[2],
            "n": len(c),
            "volume_share_median": st.median(vs),
            "volume_share_p10": _pct(vs, 10), "volume_share_p90": _pct(vs, 90),
            "mean_faced_spread": st.mean(r["mean_faced_spread"] for r in c),
            "max_dealer_net_mean": st.mean(r["max_dealer_net"] for r in c),
        })
    return out


def _pct(xs, q):
    ys = sorted(xs)
    if not ys:
        return 0.0
    i = (q / 100.0) * (len(ys) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return ys[lo] + (ys[hi] - ys[lo]) * (i - lo)


_ND_COLOR = {1: "#b3341f", 2: "#c9860a", 4: "#2c7a47"}


def _svg(agg: List[dict], out: Path, n: int, reluctance: float) -> None:
    sub = [a for a in agg if a["reluctance"] == reluctance]
    w, h, pad = 680, 400, 66
    pw, ph = w - 2 * pad, h - 2 * pad
    x0, x1 = min(RISK_FACTORS), max(RISK_FACTORS)

    def xs(r): return pad + pw * (r - x0) / ((x1 - x0) or 1.0)
    def ys(v): return pad + ph - v * ph

    lab = "balanced" if reluctance == 0.0 else "one-sided"
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">',
         '<style>.t{font-size:14px;font-weight:600}.s{font-size:11px;fill:#666}'
         '.lab{font-size:11px;fill:#444}.med{fill:none;stroke-width:2}</style>',
         f'<rect width="{w}" height="{h}" fill="#fff"/>',
         f'<text class="t" x="{pad}" y="26">Dealer competition vs basis — volume '
         f'share, {lab} flow (N={n})</text>',
         f'<text class="s" x="{pad}" y="43">band = p10–p90 · more dealers = '
         f'inventory load-balances = tighter aggregate quote</text>']
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = ys(frac)
        p.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{pad+pw}" y2="{y:.1f}" '
                 f'stroke="#eee"/>')
        p.append(f'<text class="lab" x="{pad-8}" y="{y+4:.1f}" '
                 f'text-anchor="end">{frac:.0%}</text>')
    for r in RISK_FACTORS:
        p.append(f'<text class="lab" x="{xs(r):.1f}" y="{pad+ph+18:.1f}" '
                 f'text-anchor="middle">{r:g}</text>')
    by_nd: Dict[int, List[dict]] = {}
    for a in sub:
        by_nd.setdefault(a["n_dealers"], []).append(a)
    for nd, series in sorted(by_nd.items()):
        series.sort(key=lambda a: a["risk_factor"])
        color = _ND_COLOR.get(nd, "#555")
        top = " ".join(f"{xs(a['risk_factor']):.1f},{ys(a['volume_share_p90']):.1f}"
                       for a in series)
        bot = " ".join(f"{xs(a['risk_factor']):.1f},{ys(a['volume_share_p10']):.1f}"
                       for a in reversed(series))
        p.append(f'<polygon points="{top} {bot}" fill="{color}" '
                 f'fill-opacity="0.10" stroke="none"/>')
        line = " ".join(f"{xs(a['risk_factor']):.1f},{ys(a['volume_share_median']):.1f}"
                        for a in series)
        p.append(f'<polyline class="med" points="{line}" stroke="{color}"/>')
        for a in series:
            p.append(f'<circle cx="{xs(a["risk_factor"]):.1f}" '
                     f'cy="{ys(a["volume_share_median"]):.1f}" r="3" fill="{color}"/>')
    ly = pad + 14
    for nd in sorted(by_nd):
        color = _ND_COLOR.get(nd, "#555")
        p.append(f'<rect x="{pad+pw-150}" y="{ly-9}" width="12" height="12" '
                 f'fill="{color}" fill-opacity="0.7"/>')
        p.append(f'<text class="lab" x="{pad+pw-134}" y="{ly+1}">{nd} dealer'
                 f'{"s" if nd > 1 else ""}</text>')
        ly += 18
    p.append(f'<text class="lab" x="{pad+pw/2}" y="{h-12}" text-anchor="middle">'
             f'risk_factor (basis proxy ∝ 1−r_i) →</text>')
    p.append("</svg>")
    out.write_text("\n".join(p))
    print(f"wrote {out}")


def _table(agg: List[dict], n: int) -> str:
    lines = [
        f"# Dealer competition vs basis — auto table (N={n})", "",
        "Volume share vs basis proxy, by number of competing dealers. More "
        "dealers load-balance inventory (max per-dealer |net| falls) -> tighter "
        "aggregate quote -> more volume.", "",
        "| reluctance | dealers | risk_factor | volume (p10–p90) | faced spread "
        "| max dealer |net| |",
        "|---|---|---|---|---|---|",
    ]
    for a in sorted(agg, key=lambda a: (a["reluctance"], a["n_dealers"], a["risk_factor"])):
        lines.append(
            f"| {a['reluctance']:g} | {a['n_dealers']} | {a['risk_factor']:g} | "
            f"{a['volume_share_median']:.2f} "
            f"({a['volume_share_p10']:.2f}–{a['volume_share_p90']:.2f}) | "
            f"{a['mean_faced_spread']:.3f} | {a['max_dealer_net_mean']:.2f} |")
    lines += ["", "_Auto-generated. Narrative interpretation lives in FINDINGS.md._"]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--output", type=Path,
                    default=Path("runs/compute_dealer_competition"))
    args = ap.parse_args(argv)
    N = args.n_seeds

    rows: List[dict] = []
    for rel in RELUCTANCE:
        for nd in N_DEALERS:
            for rf in RISK_FACTORS:
                for s in range(N):
                    rows.append(_trial(s, nd, rf, rel))
    agg = _aggregate(rows)

    print(f"\n=== Dealer competition vs basis, N={N} ===")
    for rel in RELUCTANCE:
        print(f"\n  reluctance={rel:g}:  volume share by (dealers, risk_factor)")
        for nd in N_DEALERS:
            row = "  ".join(
                f"rf{a['risk_factor']:g}={a['volume_share_median']:.2f}"
                for a in sorted((a for a in agg if a["reluctance"] == rel
                                 and a["n_dealers"] == nd),
                                key=lambda a: a["risk_factor"]))
            print(f"    {nd} dealer(s): {row}")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trials.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader(); wri.writerows(rows)
    with (args.output / "aggregate.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        wri.writeheader(); wri.writerows(agg)
    (args.output / "TABLE.md").write_text(_table(agg, N))
    _svg(agg, args.output / "competition_curve.svg", N, reluctance=0.0)
    print(f"\nwrote {args.output}/trials.csv, aggregate.csv, TABLE.md")


if __name__ == "__main__":
    main()
