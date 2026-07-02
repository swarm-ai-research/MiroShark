"""Compute basis — SKU dispersion: diversification vs netting (bd z45, ← umm).

Phase 2 of the basis-risk study (`docs/plans/2026-06-30-compute-basis-dealer-
experiment.md`, factor ``sku_dispersion``). The headline sweep
(``compute_basis_experiment.py``) found that with *two-sided* flow a dealer NETS
longs against shorts, so a central exchange forms across the whole basis range;
the basis-gated phase transition only appears in the *one-sided* limit
(reluctance→1). This asks a second question:

    Can SKU *dispersion* substitute for netting? If one-sided demand is spread
    over K SKUs with **independent** idiosyncratic components, the dealer holds K
    independent basis residuals, and diversification (Var ~ 1/K) should shrink
    the book it must warehouse — forming the market even when it cannot net.

Model (extends the headline; K=1 reduces to it exactly):

- One common basket index ``I`` (k9w log-normal) and **K orthonormal** idio
  innovations ``eN_k`` (Gram-Schmidt: each ⟂ I and ⟂ each other) → every SKU
  shares the same basis loading β (same r_i=β²) but has *independent* idio, so
  ``S_k = F + σ·(β·iI + √(1−β²)·eN_k)``.
- Each principal is assigned one SKU. Within a SKU longs/shorts net; the dealer's
  book is the vector of per-SKU net imbalances ``D_k``. It hedges ``β·Σ_k D_k`` on
  the basket and warehouses the residual, whose variance is ``Σ_k D_k²·V·(1−r)``
  (independent idio → variances add, no cross terms). One uniform clearing spread
  ``s = ½·γ_D·(Σ_k D_k²)·V·(1−r)/gross`` prices the whole book.

**Independent idio across SKUs is the optimistic (best-case) diversification** —
real SKUs (H200 us-east vs us-west) co-move. A cross-SKU correlation knob is the
obvious next refinement; this measures the ceiling of the diversification effect.

Usage::

    uv run python -m scripts.compute_basis_sku_experiment --n-seeds 100

Writes trials.csv, aggregate.csv, TABLE.md, sku_scurve.svg. FINDINGS hand-curated.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import statistics as st
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

logging.disable(logging.CRITICAL)

from scripts.compute_futures_experiment import BASE, SIGMA, _h100_spot  # noqa: E402
from scripts.compute_basis_experiment import _u, _z, _corr  # noqa: E402

BETAS = [0.20, 0.45, 0.63, 0.80, 0.95, 1.0]
SKU_DISPERSION = [1, 4, 12]
RELUCTANCE = [0.0, 1.0]        # balanced vs one-sided (where diversification bites)
N_PRINCIPALS = 20
GAMMA_P = 1.0
GAMMA_D = 1.0
SIZE_LOGVOL = 0.4
GAMMA_LOGVOL = 0.3


@dataclass
class Principal:
    side: int
    q: float
    gamma: float
    withdraw_u: float
    sku: int

    def s_max(self) -> float:
        return 0.5 * self.gamma * self.q


def _population(seed: int, n_p: int, gamma_p: float, K: int) -> List[Principal]:
    pop = []
    for j in range(n_p):
        side = 1 if _u(seed, j, 11) < 0.5 else -1
        q = math.exp(SIZE_LOGVOL * _z(seed, j, 20))
        gamma = gamma_p * math.exp(GAMMA_LOGVOL * _z(seed, j, 40))
        sku = int(_u(seed, j, 90) * K) % K
        pop.append(Principal(side, q, gamma, _u(seed, j, 60), sku))
    return pop


def _innovations_k(n: int, K: int) -> Tuple[float, float, List[float],
                                            List[List[float]]]:
    """Fair forward F, σ(I), standardised basket innovation ``iI``, and K
    idiosyncratic innovations orthonormalised against iI *and each other*
    (Gram-Schmidt) — each Var=1, mutually ⟂, in-sample. Makes every SKU's basis
    exactly β and the K idio components exactly independent."""
    Is = [_h100_spot(s) for s in range(n)]
    F = st.mean(Is)
    sd_i = st.pstdev(Is)
    iI = [(x - F) / sd_i for x in Is]

    basis = [iI]                                  # orthonormal set, start with iI
    eNs: List[List[float]] = []
    for k in range(K):
        raw = [BASE * math.exp(SIGMA * _z(s, -1 - k, 700 + 100 * k))
               for s in range(n)]
        me = st.mean(raw)
        v = [x - me for x in raw]
        for u in basis:                            # subtract projections
            dot = sum(a * b for a, b in zip(v, u)) / sum(b * b for b in u)
            v = [a - dot * b for a, b in zip(v, u)]
        sv = st.pstdev(v)
        e = [a / sv for a in v]
        eNs.append(e)
        basis.append(e)
    return F, sd_i, iI, eNs


def _clear(pop: List[Principal], beta: float, reluctance: float,
           gamma_d: float, K: int) -> Tuple[List[Principal], float, float]:
    """Uniform-spread clearing over the whole (multi-SKU) book. Residual is the
    sum of per-SKU squared net imbalances (independent idio). Returns
    (participants, Σ_k D_k², clearing spread ÷V)."""
    one_minus_r = 1.0 - beta * beta
    willing = [p for p in pop if p.side == 1 or p.withdraw_u >= reluctance]
    willing.sort(key=lambda p: p.s_max(), reverse=True)

    best_k, best_R, best_s = 0, 0.0, 0.0
    for k in range(1, len(willing) + 1):
        part = willing[:k]
        gross = sum(p.q for p in part)
        Dk = [0.0] * K
        for p in part:
            Dk[p.sku] += p.side * p.q
        R = sum(d * d for d in Dk)                 # Σ_k D_k²
        s_req = 0.5 * gamma_d * R * one_minus_r / gross if gross else 0.0
        if s_req <= part[-1].s_max() + 1e-12:
            best_k, best_R, best_s = k, R, s_req
    return willing[:best_k], best_R, best_s


def _trial(seed: int, beta: float, reluctance: float, gamma_d: float, n_p: int,
           gamma_p: float, K: int, F: float, sd_i: float,
           iI_s: float, eN_s: List[float]) -> Dict[str, float]:
    V = sd_i * sd_i
    pop = _population(seed, n_p, gamma_p, K)
    part, R, s_over_V = _clear(pop, beta, reluctance, gamma_d, K)
    part_set = set(id(p) for p in part)

    total_gross = sum(p.q for p in pop)
    willing_gross = sum(p.q for p in pop
                        if p.side == 1 or p.withdraw_u >= reluctance)
    gross_part = sum(p.q for p in part)

    root = math.sqrt(1.0 - beta * beta)
    I = F + sd_i * iI_s
    # Per-SKU net imbalance among participants → realised residual P&L.
    Dk = [0.0] * K
    for p in part:
        Dk[p.sku] += p.side * p.q
    dealer_pnl_ex_spread = -root * sd_i * sum(Dk[k] * eN_s[k] for k in range(K))
    spread_dollar = s_over_V * V
    dealer_pnl = dealer_pnl_ex_spread + spread_dollar * gross_part

    # Realised SKU spot for a reference SKU (0) — for the corr sanity check.
    S0 = F + sd_i * (beta * iI_s + root * eN_s[0])

    return {
        "seed": seed, "beta": beta, "reluctance": reluctance, "K": K,
        "r_i": beta * beta,
        "volume_share": gross_part / total_gross if total_gross else 0.0,
        "willing_clear_frac": gross_part / willing_gross if willing_gross else 0.0,
        "residual_risk": math.sqrt(V * (1.0 - beta * beta) * R),
        "sum_Dk2": R,
        "clearing_spread_pct": 100.0 * spread_dollar / F if F else 0.0,
        "dealer_pnl": dealer_pnl,
        "spot_I": I, "spot_S0": S0,
    }


def _aggregate(rows: List[dict]) -> List[dict]:
    cells: Dict[Tuple[float, float, int], List[dict]] = {}
    for r in rows:
        cells.setdefault((r["beta"], r["reluctance"], r["K"]), []).append(r)
    out = []
    for key in sorted(cells):
        c = cells[key]
        vs = [r["volume_share"] for r in c]
        wc = [r["willing_clear_frac"] for r in c]
        corr = _corr([r["spot_I"] for r in c], [r["spot_S0"] for r in c])
        out.append({
            "beta": key[0], "reluctance": key[1], "K": key[2],
            "r_i": key[0] ** 2, "n": len(c), "corr_S_I": corr,
            "volume_share_median": st.median(vs),
            "volume_share_p10": _p(vs, 10), "volume_share_p90": _p(vs, 90),
            "willing_clear_frac_mean": st.mean(wc),
            "willing_clear_frac_p10": _p(wc, 10),
            "willing_clear_frac_p90": _p(wc, 90),
            "residual_risk_mean": st.mean(r["residual_risk"] for r in c),
            "clearing_spread_pct_mean": st.mean(r["clearing_spread_pct"] for r in c),
            "dealer_pnl_mean": st.mean(r["dealer_pnl"] for r in c),
        })
    return out


def _p(xs: List[float], q: float) -> float:
    ys = sorted(xs)
    if not ys:
        return 0.0
    i = (q / 100.0) * (len(ys) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return ys[lo] + (ys[hi] - ys[lo]) * (i - lo)


_K_COLOR = {1: "#b3341f", 4: "#c9860a", 12: "#2c7a47"}


def _svg(agg: List[dict], out: Path, n: int, reluctance: float) -> None:
    """Willing-clear fraction vs β at fixed reluctance, one series per K. In the
    one-sided panel this shows diversification pulling the transition left."""
    sub = [a for a in agg if a["reluctance"] == reluctance]
    w, h, pad = 680, 400, 64
    pw, ph = w - 2 * pad, h - 2 * pad

    def xs(b): return pad + pw * b
    def ys(v): return pad + ph - v * ph

    lab = ("one-sided (reluctance 1.0)" if reluctance >= 1.0
           else f"reluctance {reluctance:g}")
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">',
         '<style>.t{font-size:14px;font-weight:600}.s{font-size:11px;fill:#666}'
         '.lab{font-size:11px;fill:#444}.med{fill:none;stroke-width:2}</style>',
         f'<rect width="{w}" height="{h}" fill="#fff"/>',
         f'<text class="t" x="{pad}" y="26">SKU dispersion vs basis — willing-'
         f'clear fraction, {lab} (N={n})</text>',
         f'<text class="s" x="{pad}" y="43">band = p10–p90 · more SKUs = more '
         f'independent-idio diversification of the dealer book</text>']
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = ys(frac)
        p.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{pad+pw}" y2="{y:.1f}" '
                 f'stroke="#eee"/>')
        p.append(f'<text class="lab" x="{pad-8}" y="{y+4:.1f}" '
                 f'text-anchor="end">{frac:.0%}</text>')
    for b in BETAS:
        p.append(f'<text class="lab" x="{xs(b):.1f}" y="{pad+ph+18:.1f}" '
                 f'text-anchor="middle">{b:.2g}</text>')

    by_k: Dict[int, List[dict]] = {}
    for a in sub:
        by_k.setdefault(a["K"], []).append(a)
    for K, series in sorted(by_k.items()):
        series.sort(key=lambda a: a["beta"])
        color = _K_COLOR.get(K, "#555")
        top = " ".join(f"{xs(a['beta']):.1f},{ys(a['willing_clear_frac_p90']):.1f}"
                       for a in series)
        bot = " ".join(f"{xs(a['beta']):.1f},{ys(a['willing_clear_frac_p10']):.1f}"
                       for a in reversed(series))
        p.append(f'<polygon points="{top} {bot}" fill="{color}" '
                 f'fill-opacity="0.12" stroke="none"/>')
        line = " ".join(f"{xs(a['beta']):.1f},{ys(a['willing_clear_frac_mean']):.1f}"
                        for a in series)
        p.append(f'<polyline class="med" points="{line}" stroke="{color}"/>')
        for a in series:
            p.append(f'<circle cx="{xs(a["beta"]):.1f}" '
                     f'cy="{ys(a["willing_clear_frac_mean"]):.1f}" r="3" '
                     f'fill="{color}"/>')
    ly = pad + 14
    for K in sorted(by_k):
        color = _K_COLOR.get(K, "#555")
        p.append(f'<rect x="{pad+pw-150}" y="{ly-9}" width="12" height="12" '
                 f'fill="{color}" fill-opacity="0.6"/>')
        p.append(f'<text class="lab" x="{pad+pw-134}" y="{ly+1}">K={K} '
                 f'SKU{"s" if K > 1 else ""}</text>')
        ly += 18
    p.append(f'<text class="lab" x="{pad+pw/2}" y="{h-12}" '
             f'text-anchor="middle">basis correlation β  (r_i = β²) →</text>')
    p.append("</svg>")
    out.write_text("\n".join(p))
    print(f"wrote {out}")


def _table(agg: List[dict], n: int) -> str:
    lines = [
        f"# Compute basis — SKU dispersion vs netting — auto table (N={n})", "",
        "`K` = number of distinct SKUs (independent idio). Order parameter: "
        "willing-clear fraction (dealer's intermediation of *willing* demand — "
        "isolates the basis/diversification effect).", "",
        "| reluctance | K | β | r_i | corr(S,I) | willing clear (p10–p90) | "
        "volume share | residual risk | spread % | dealer P&L |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in sorted(agg, key=lambda a: (a["reluctance"], a["K"], a["beta"])):
        lines.append(
            f"| {a['reluctance']:.1f} | {a['K']} | {a['beta']:.2g} | "
            f"{a['r_i']:.2f} | {a['corr_S_I']:.2f} | "
            f"{a['willing_clear_frac_mean']:.2f} "
            f"({a['willing_clear_frac_p10']:.2f}–{a['willing_clear_frac_p90']:.2f}) | "
            f"{a['volume_share_median']:.2f} | {a['residual_risk_mean']:.2f} | "
            f"{a['clearing_spread_pct_mean']:.2f}% | {a['dealer_pnl_mean']:.2f} |")
    lines += ["", "_Auto-generated. Narrative interpretation lives in FINDINGS.md._"]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--n-principals", type=int, default=N_PRINCIPALS)
    ap.add_argument("--gamma-dealer", type=float, default=GAMMA_D)
    ap.add_argument("--gamma-principal", type=float, default=GAMMA_P)
    ap.add_argument("--output", type=Path, default=Path("runs/compute_basis_sku"))
    args = ap.parse_args(argv)
    N = args.n_seeds

    rows: List[dict] = []
    for K in SKU_DISPERSION:
        F, sd_i, iI, eNs = _innovations_k(N, K)
        for beta in BETAS:
            for rel in RELUCTANCE:
                for s in range(N):
                    eN_s = [eNs[k][s] for k in range(K)]
                    rows.append(_trial(s, beta, rel, args.gamma_dealer,
                                       args.n_principals, args.gamma_principal,
                                       K, F, sd_i, iI[s], eN_s))
    agg = _aggregate(rows)

    print(f"\n=== Compute basis / SKU dispersion, N={N} ===")
    for rel in RELUCTANCE:
        print(f"\n  reluctance={rel:.1f} — willing-clear fraction vs β:")
        for K in SKU_DISPERSION:
            cells = sorted((a for a in agg
                            if a["reluctance"] == rel and a["K"] == K),
                           key=lambda a: a["beta"])
            row = "  ".join(f"β{a['beta']:.2g}={a['willing_clear_frac_mean']:.2f}"
                            for a in cells)
            print(f"    K={K:>2}: {row}")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trials.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader(); wri.writerows(rows)
    with (args.output / "aggregate.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        wri.writeheader(); wri.writerows(agg)
    (args.output / "TABLE.md").write_text(_table(agg, N))
    _svg(agg, args.output / "sku_scurve.svg", N, reluctance=1.0)
    print(f"\nwrote {args.output}/trials.csv, aggregate.csv, TABLE.md")


if __name__ == "__main__":
    main()
