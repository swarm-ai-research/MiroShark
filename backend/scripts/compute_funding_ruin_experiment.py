"""Dealer funding-ruin under asymmetric margining (bd b2s, ← umm/1z3).

The bd-1z3 daily-MTM kernel makes concrete the risk Pirrong (Trafigura) argues is
the real binding constraint on commodity dealers: **variation-margin cash flow**,
not basis variance. A dealer can be hedged in *value* yet fail on *cash*.

Setup (the asymmetric-margining mechanism):

- The dealer has sold a SKU forward to a principal — an **OTC** short, ``Q`` at
  fair ``F``, that settles only at expiry (no daily margin). Modelled
  analytically: OTC P&L = ``(F − S_T)·Q``.
- The dealer hedges by going **long β·Q of the basket** through the *real*
  MTM'd futures book (``futures_daily_mtm=True``), against a deep-pocketed
  counterparty. This leg is **marked every epoch** — it draws or pays variation
  margin from the dealer's finite free coin.

Each seed is a basket price **path** ``I_0…I_T``. When the basket falls, the
dealer's long hedge posts variation margin daily, but the offsetting SKU gain is
**locked until expiry** — a funding drain. If the drain exceeds the dealer's coin
buffer, the hedge is **liquidated** (kernel), leaving the dealer unhedged for the
rest of the path. At β=1 the position is *economically flat* at expiry, so any
loss is **pure funding risk with zero terminal price risk** — the sharpest form
of Pirrong's point.

We sweep the coin buffer ``B`` (as a fraction of hedge notional) × β and report
ruin frequency and terminal P&L (survivors vs ruined).

Usage::

    uv run python -m scripts.compute_funding_ruin_experiment --n-seeds 200

Writes trials.csv, aggregate.csv, TABLE.md, ruin_curve.svg. FINDINGS curated.
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
from scripts.compute_basis_experiment import _u, _z  # noqa: E402

F = 3.50            # fair forward / path start ($/GPU-hr)
Q = 10.0            # principal's SKU exposure the dealer is short (OTC)
T = 6               # path length (epochs); contract settles at epoch T
PATH_LOGVOL = 0.16  # per-epoch basket log-return vol
BETAS = [1.0, 0.8]
BUFFER_FRACS = [0.05, 0.10, 0.20, 0.40, 0.80, 1.60]  # coin buffer ÷ hedge notional
BIG = 10_000_000.0  # counterparty coin (deep-pocketed / CCP)


def _basket_path(seed: int) -> List[float]:
    """Deterministic per-seed basket spot path I_0..I_T (GBM-ish random walk)."""
    path = [F]
    for t in range(1, T + 1):
        r = PATH_LOGVOL * _z(seed, t, 300)          # quasi-normal log-return
        path.append(path[-1] * math.exp(r))
    return path


def _sku_terminal(seed: int, beta: float, I_T: float) -> float:
    """Terminal SKU spot via the factor model on the terminal basket level."""
    idio = F * math.exp(PATH_LOGVOL * math.sqrt(T) * _z(seed, -7, 909))
    return F + beta * (I_T - F) + math.sqrt(1.0 - beta * beta) * (idio - F)


def _trial(seed: int, beta: float, buffer_frac: float) -> Dict[str, float]:
    hedge = beta * Q
    notional = max(hedge * F, 1e-9)
    buffer = buffer_frac * notional

    cfg = GTBConfig.from_dict({
        "map": {"height": 3, "width": 3, "wood_density": 0.0,
                "stone_density": 0.0, "compute_density": 1.0},
        "market": {"futures_daily_mtm": True},
    })
    cfg.seed = seed
    env = GTBEnvironment(cfg)
    env.add_worker("dealer")
    env.add_worker("cp")
    # Dealer holds exactly the buffer as free coin; cp is deep-pocketed.
    env._workers["dealer"].add_resource(ResourceType.COIN, buffer)
    env._workers["cp"].add_resource(ResourceType.COIN, BIG)
    dealer_start = env._workers["dealer"].get_resource(ResourceType.COIN)

    path = _basket_path(seed)
    contract = None
    if hedge > 1e-12:
        # Dealer long β·Q basket, cp short — matched at F, MTM'd each epoch.
        for agent, atype in (("dealer", GTBActionType.FUTURES_BUY),
                             ("cp", GTBActionType.FUTURES_SELL)):
            env.apply_actions({agent: GTBAction(
                agent_id=agent, action_type=atype,
                resource_type=ResourceType.COMPUTE, quantity=hedge, price=F,
                settlement_epoch=T)})
        contract = env._futures_contracts[0] if env._futures_contracts else None

    # Walk the path; MTM runs at each epoch boundary.
    ruin_epoch = 0
    for t in range(1, T + 1):
        env._last_trade_price[ResourceType.COMPUTE.value] = path[t]
        env._current_epoch = t
        env.end_epoch()
        if contract is not None and contract.status == "liquidated" and not ruin_epoch:
            ruin_epoch = t

    ruined = 1.0 if (contract is not None and contract.status == "liquidated") else 0.0
    # Dealer coin delta = realized basket-hedge P&L (incl. any liquidation loss).
    basket_pnl = env._workers["dealer"].get_resource(ResourceType.COIN) - dealer_start
    # OTC SKU short settles at expiry (analytic, no daily margin).
    S_T = _sku_terminal(seed, beta, path[-1])
    otc_pnl = (F - S_T) * Q
    net_pnl = basket_pnl + otc_pnl

    return {"seed": seed, "beta": beta, "buffer_frac": buffer_frac,
            "ruined": ruined, "ruin_epoch": ruin_epoch,
            "basket_pnl": basket_pnl, "otc_pnl": otc_pnl, "net_pnl": net_pnl,
            "min_basket": min(path), "max_basket": max(path)}


def _aggregate(rows: List[dict]) -> List[dict]:
    cells: Dict[tuple, List[dict]] = {}
    for r in rows:
        cells.setdefault((r["beta"], r["buffer_frac"]), []).append(r)
    out = []
    for (beta, bf) in sorted(cells):
        c = cells[(beta, bf)]
        net = [r["net_pnl"] for r in c]
        ruined = [r for r in c if r["ruined"] > 0.5]
        survived = [r for r in c if r["ruined"] < 0.5]
        out.append({
            "beta": beta, "buffer_frac": bf, "n": len(c),
            "ruin_freq": st.mean(r["ruined"] for r in c),
            "net_pnl_mean": st.mean(net),
            "net_pnl_std": st.pstdev(net),
            "net_pnl_p10": _pct(net, 10),
            "net_pnl_ruined_mean": st.mean(r["net_pnl"] for r in ruined)
            if ruined else 0.0,
            "net_pnl_survived_std": st.pstdev(r["net_pnl"] for r in survived)
            if len(survived) > 1 else 0.0,
        })
    return out


def _pct(xs, q):
    ys = sorted(xs)
    if not ys:
        return 0.0
    i = (q / 100.0) * (len(ys) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return ys[lo] + (ys[hi] - ys[lo]) * (i - lo)


_BETA_COLOR = {1.0: "#b3341f", 0.8: "#c9860a"}


def _svg(agg: List[dict], out: Path, n: int) -> None:
    """Ruin frequency vs coin buffer (÷ hedge notional), one series per β."""
    w, h, pad = 680, 400, 66
    pw, ph = w - 2 * pad, h - 2 * pad
    xlogs = [math.log(b) for b in BUFFER_FRACS]
    x0, x1 = min(xlogs), max(xlogs)

    def xs(b): return pad + pw * (math.log(b) - x0) / ((x1 - x0) or 1.0)
    def ys(v): return pad + ph - v * ph

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">',
         '<style>.t{font-size:14px;font-weight:600}.s{font-size:11px;fill:#666}'
         '.lab{font-size:11px;fill:#444}.med{fill:none;stroke-width:2}</style>',
         f'<rect width="{w}" height="{h}" fill="#fff"/>',
         f'<text class="t" x="{pad}" y="26">Dealer funding-ruin vs coin buffer '
         f'(N={n})</text>',
         f'<text class="s" x="{pad}" y="43">β=1 is economically flat at expiry — '
         f'ruin there is PURE funding risk (zero terminal price risk)</text>']
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = ys(frac)
        p.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{pad+pw}" y2="{y:.1f}" '
                 f'stroke="#eee"/>')
        p.append(f'<text class="lab" x="{pad-8}" y="{y+4:.1f}" '
                 f'text-anchor="end">{frac:.0%}</text>')
    for b in BUFFER_FRACS:
        p.append(f'<text class="lab" x="{xs(b):.1f}" y="{pad+ph+18:.1f}" '
                 f'text-anchor="middle">{b:g}×</text>')
    by_beta: Dict[float, List[dict]] = {}
    for a in agg:
        by_beta.setdefault(a["beta"], []).append(a)
    for beta, series in sorted(by_beta.items()):
        series.sort(key=lambda a: a["buffer_frac"])
        color = _BETA_COLOR.get(beta, "#555")
        line = " ".join(f"{xs(a['buffer_frac']):.1f},{ys(a['ruin_freq']):.1f}"
                        for a in series)
        p.append(f'<polyline class="med" points="{line}" stroke="{color}"/>')
        for a in series:
            p.append(f'<circle cx="{xs(a["buffer_frac"]):.1f}" '
                     f'cy="{ys(a["ruin_freq"]):.1f}" r="3" fill="{color}"/>')
    ly = pad + 14
    for beta in sorted(by_beta):
        color = _BETA_COLOR.get(beta, "#555")
        p.append(f'<rect x="{pad+pw-150}" y="{ly-9}" width="12" height="12" '
                 f'fill="{color}" fill-opacity="0.7"/>')
        lbl = "β=1.0 (flat at expiry)" if beta == 1.0 else f"β={beta:g}"
        p.append(f'<text class="lab" x="{pad+pw-134}" y="{ly+1}">{lbl}</text>')
        ly += 18
    p.append(f'<text class="lab" x="{pad+pw/2}" y="{h-12}" text-anchor="middle">'
             f'dealer coin buffer ÷ hedge notional  (log) →</text>')
    p.append("</svg>")
    out.write_text("\n".join(p))
    print(f"wrote {out}")


def _table(agg: List[dict], n: int) -> str:
    lines = [
        f"# Dealer funding-ruin under asymmetric margining — auto table (N={n})",
        "", "Buffer = dealer free coin ÷ hedge notional (β·Q·F). Ruin = the "
        "MTM'd basket hedge is liquidated mid-path. β=1 is economically flat at "
        "expiry (pure funding risk).", "",
        "| β | buffer | ruin freq | net P&L mean | net P&L std | net P&L p10 | "
        "ruined-only mean | survivors std |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in sorted(agg, key=lambda a: (a["beta"], a["buffer_frac"])):
        lines.append(
            f"| {a['beta']:g} | {a['buffer_frac']:g}× | {a['ruin_freq']:.0%} | "
            f"{a['net_pnl_mean']:.2f} | {a['net_pnl_std']:.2f} | "
            f"{a['net_pnl_p10']:.2f} | {a['net_pnl_ruined_mean']:.2f} | "
            f"{a['net_pnl_survived_std']:.2f} |")
    lines += ["", "_Auto-generated. Narrative interpretation lives in FINDINGS.md._"]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=200)
    ap.add_argument("--output", type=Path,
                    default=Path("runs/compute_funding_ruin"))
    args = ap.parse_args(argv)
    N = args.n_seeds

    rows: List[dict] = []
    for beta in BETAS:
        for bf in BUFFER_FRACS:
            for s in range(N):
                rows.append(_trial(s, beta, bf))
    agg = _aggregate(rows)

    print(f"\n=== Dealer funding-ruin under asymmetric margining, N={N} ===")
    print(f"  Q={Q} hedge=β·Q  F=${F}  path T={T} epochs  logvol={PATH_LOGVOL}")
    for beta in BETAS:
        print(f"\n  β={beta:g} ({'flat at expiry' if beta == 1.0 else 'has basis'}):")
        for a in sorted((a for a in agg if a["beta"] == beta),
                        key=lambda a: a["buffer_frac"]):
            print(f"    buffer={a['buffer_frac']:g}×: ruin={a['ruin_freq']:.0%}  "
                  f"net P&L mean={a['net_pnl_mean']:.2f} std={a['net_pnl_std']:.2f} "
                  f"p10={a['net_pnl_p10']:.2f}  ruined-mean={a['net_pnl_ruined_mean']:.2f}")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trials.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader(); wri.writerows(rows)
    with (args.output / "aggregate.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        wri.writeheader(); wri.writerows(agg)
    (args.output / "TABLE.md").write_text(_table(agg, N))
    _svg(agg, args.output / "ruin_curve.svg", N)
    print(f"\nwrote {args.output}/trials.csv, aggregate.csv, TABLE.md")


if __name__ == "__main__":
    main()
