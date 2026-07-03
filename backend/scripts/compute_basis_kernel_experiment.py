"""Basis-hedge through the REAL order book (bd er8, ← umm §7 increment 2).

The phase-1..4 studies (`compute_basis_experiment.py` &c.) modelled the dealer's
basis residual analytically. This runs the *same* mechanic through the vendored
futures engine — SKU forwards on the new SKU-keyed book (bd bfl / increment 1)
plus a basket forward on the generic book — and checks the dealer's realised
residual matches the analytic ``Q·√((1−r_i)·V)`` and collapses to 0 at β=1.

Per seed, per β:

- **SKU forward**: the principal goes long Q on SKU 'a' at fair forward F, the
  dealer takes the short (crossing on the SKU book).
- **Basket hedge**: the dealer goes long β·Q on the *generic* book at fair G=F,
  a basket counterparty takes the short. β·Q is the variance-minimising hedge.
- At settlement the synthetic factor spots are published — basket ``I`` to
  ``_last_trade_price['compute']`` and SKU ``S = F + σ(β·iI + √(1−β²)·eN)`` to
  ``_last_trade_price['compute/a']`` (the per-SKU settlement path from
  increment 1) — and ``end_epoch`` cash-settles both contracts.

The dealer's realised net P&L is ``(F−S)·Q + (I−F)·β·Q = −Q·√(1−β²)·σ·eN``, so
across seeds its **std = Q·√((1−r_i)·V)** and its mean ≈ 0 — the exact analytic
residual, now produced by real matching + settlement. Reuses the phase-1
orthonormal innovations so the basis is noise-free (corr(S,I)=β exactly).

Usage::

    uv run python -m scripts.compute_basis_kernel_experiment --n-seeds 100

Writes trials.csv, aggregate.csv, TABLE.md, kernel_residual.svg. FINDINGS curated.
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
from scripts.compute_basis_experiment import _innovations  # noqa: E402

BETAS = [0.20, 0.45, 0.63, 0.80, 0.95, 1.0]
Q = 10.0            # principal's SKU exposure (units)
SETTLE_EPOCH = 2
ENDOW = 1_000_000.0  # ample coin so settlement never clamps
SKU = "a"


def _trial(seed: int, beta: float, F: float, sd_i: float,
           iI_s: float, eN_s: float) -> Dict[str, float]:
    cfg = GTBConfig.from_dict({
        "map": {"height": 3, "width": 3, "wood_density": 0.0,
                "stone_density": 0.0, "compute_density": 1.0},
    })
    cfg.seed = seed
    env = GTBEnvironment(cfg)
    for name in ("principal", "dealer", "basket_cp"):
        env.add_worker(name)
        env._workers[name].add_resource(ResourceType.COIN, ENDOW)

    # Measure dealer coin BEFORE posting, so escrowed margins round-trip out of
    # the delta (k9w discipline) — leaving only realised forward P&L.
    dealer_before = env._workers["dealer"].get_resource(ResourceType.COIN)

    def act(agent, atype, qty, sku):
        env.apply_actions({agent: GTBAction(
            agent_id=agent, action_type=atype,
            resource_type=ResourceType.COMPUTE, quantity=qty, price=F,
            settlement_epoch=SETTLE_EPOCH, sku=sku)})

    # SKU forward: principal long, dealer short (crosses on the SKU book).
    act("principal", GTBActionType.FUTURES_BUY, Q, SKU)
    act("dealer", GTBActionType.FUTURES_SELL, Q, SKU)
    # Basket hedge: dealer long β·Q on the generic book, counterparty short.
    hedge = beta * Q
    if hedge > 1e-12:
        act("dealer", GTBActionType.FUTURES_BUY, hedge, "")
        act("basket_cp", GTBActionType.FUTURES_SELL, hedge, "")

    n_contracts = len(env._futures_contracts)

    # Publish synthetic factor spots and settle both contracts.
    I = F + sd_i * iI_s
    S = F + sd_i * (beta * iI_s + math.sqrt(1.0 - beta * beta) * eN_s)
    env._last_trade_price[ResourceType.COMPUTE.value] = I
    env._last_trade_price[f"{ResourceType.COMPUTE.value}/{SKU}"] = S
    env._current_epoch = SETTLE_EPOCH
    env.end_epoch()

    dealer_pnl = env._workers["dealer"].get_resource(ResourceType.COIN) - dealer_before
    settled = sum(1 for c in env._futures_contracts if c.status == "settled")
    return {"seed": seed, "beta": beta, "r_i": beta * beta,
            "dealer_pnl": dealer_pnl, "spot_I": I, "spot_S": S,
            "n_contracts": n_contracts, "n_settled": settled}


def _aggregate(rows: List[dict], V: float) -> List[dict]:
    cells: Dict[float, List[dict]] = {}
    for r in rows:
        cells.setdefault(r["beta"], []).append(r)
    out = []
    for beta in sorted(cells):
        c = cells[beta]
        pnl = [r["dealer_pnl"] for r in c]
        realized = st.pstdev(pnl)
        analytic = Q * math.sqrt((1.0 - beta * beta) * V)
        corr = _corr([r["spot_I"] for r in c], [r["spot_S"] for r in c])
        out.append({
            "beta": beta, "r_i": beta * beta, "n": len(c), "corr_S_I": corr,
            "dealer_pnl_mean": st.mean(pnl),
            "residual_realized": realized,
            "residual_analytic": analytic,
            "rel_err_pct": 100.0 * abs(realized - analytic) / analytic
            if analytic > 1e-9 else 0.0,
            "n_settled_mean": st.mean(r["n_settled"] for r in c),
        })
    return out


def _corr(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx, sy = st.pstdev(xs), st.pstdev(ys)
    return cov / (sx * sy) if sx and sy else 0.0


def _svg(agg: List[dict], out: Path, n: int) -> None:
    """Dealer residual (std of realised P&L) vs β: real-book points over the
    analytic Q·√((1−r_i)·V) curve."""
    w, h, pad = 660, 380, 62
    pw, ph = w - 2 * pad, h - 2 * pad
    ymax = max(a["residual_analytic"] for a in agg) * 1.1 or 1.0

    def xs(b): return pad + pw * b
    def ys(v): return pad + ph - v / ymax * ph

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">',
         '<style>.t{font-size:14px;font-weight:600}.s{font-size:11px;fill:#666}'
         '.lab{font-size:11px;fill:#444}</style>',
         f'<rect width="{w}" height="{h}" fill="#fff"/>',
         f'<text class="t" x="{pad}" y="26">Dealer basis residual through the '
         f'real order book vs analytic (N={n})</text>',
         f'<text class="s" x="{pad}" y="43">dashed = Q·√((1−r_i)·V) · dots = '
         f'std of realised dealer P&L (real matching + settlement)</text>']
    for frac in (0.0, 0.5, 1.0):
        v = ymax * frac
        p.append(f'<line x1="{pad}" y1="{ys(v):.1f}" x2="{pad+pw}" '
                 f'y2="{ys(v):.1f}" stroke="#eee"/>')
        p.append(f'<text class="lab" x="{pad-8}" y="{ys(v)+4:.1f}" '
                 f'text-anchor="end">{v:.1f}</text>')
    for b in BETAS:
        p.append(f'<text class="lab" x="{xs(b):.1f}" y="{pad+ph+18:.1f}" '
                 f'text-anchor="middle">{b:.2g}</text>')
    ana = " ".join(f"{xs(a['beta']):.1f},{ys(a['residual_analytic']):.1f}"
                   for a in agg)
    p.append(f'<polyline points="{ana}" fill="none" stroke="#888" '
             f'stroke-width="1.3" stroke-dasharray="5 3"/>')
    for a in agg:
        p.append(f'<circle cx="{xs(a["beta"]):.1f}" '
                 f'cy="{ys(a["residual_realized"]):.1f}" r="4" fill="#2c7a47"/>')
    p.append(f'<text class="lab" x="{pad+pw/2}" y="{h-12}" '
             f'text-anchor="middle">basis correlation β  (r_i = β²) →</text>')
    p.append("</svg>")
    out.write_text("\n".join(p))
    print(f"wrote {out}")


def _table(agg: List[dict], n: int) -> str:
    lines = [
        f"# Basis-hedge through the real order book — auto table (N={n})", "",
        "Dealer residual = std of realised net P&L across seeds, from real "
        "matching + settlement. Analytic = Q·√((1−r_i)·V).", "",
        "| β | r_i | corr(S,I) | settled/seed | dealer P&L μ | residual "
        "realized | residual analytic | rel err |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in agg:
        lines.append(
            f"| {a['beta']:.2g} | {a['r_i']:.2f} | {a['corr_S_I']:.2f} | "
            f"{a['n_settled_mean']:.1f} | {a['dealer_pnl_mean']:.3f} | "
            f"{a['residual_realized']:.3f} | {a['residual_analytic']:.3f} | "
            f"{a['rel_err_pct']:.1f}% |")
    lines += ["", "_Auto-generated. Narrative interpretation lives in FINDINGS.md._"]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--output", type=Path, default=Path("runs/compute_basis_kernel"))
    args = ap.parse_args(argv)
    N = args.n_seeds

    F, sd_i, iI, eN = _innovations(N)
    V = sd_i * sd_i

    rows: List[dict] = []
    for beta in BETAS:
        for s in range(N):
            rows.append(_trial(s, beta, F, sd_i, iI[s], eN[s]))
    agg = _aggregate(rows, V)

    print(f"\n=== Basis-hedge through the real order book, N={N} ===")
    print(f"  fair forward F=${F:.2f}  Var(I)={V:.2f}  Q={Q}")
    for a in agg:
        print(f"  β={a['beta']:.2g} (r_i={a['r_i']:.2f}): residual realized="
              f"{a['residual_realized']:.3f} analytic={a['residual_analytic']:.3f} "
              f"(rel err {a['rel_err_pct']:.1f}%)  settled/seed={a['n_settled_mean']:.1f} "
              f"corr(S,I)={a['corr_S_I']:.2f}")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trials.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader(); wri.writerows(rows)
    with (args.output / "aggregate.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        wri.writeheader(); wri.writerows(agg)
    (args.output / "TABLE.md").write_text(_table(agg, N))
    _svg(agg, args.output / "kernel_residual.svg", N)
    print(f"\nwrote {args.output}/trials.csv, aggregate.csv, TABLE.md")


if __name__ == "__main__":
    main()
