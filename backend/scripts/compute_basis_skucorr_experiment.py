"""Compute basis — cross-SKU idio correlation (bd 0h4, ← z45 ← umm).

Phase 3. Phase 2 (`compute_basis_sku_experiment.py`) found SKU dispersion
substitutes for netting — but assumed the K SKUs have *perfectly independent*
idiosyncratic components, the optimistic ceiling. Real SKUs co-move (H200
us-east vs us-west share GPU, vendor, demand cycle). This measures how much of
the diversification benefit survives a cross-SKU idio correlation ``rho_sku``.

Construction: a single common idio factor ``c`` (⟂ the basket) plus K independent
own factors ``o_k`` (⟂ basket, ⟂ c, ⟂ each other), all orthonormal in-sample.
Each SKU's idio innovation is::

    eN_k = √rho_sku · c + √(1−rho_sku) · o_k

so corr(eN_j, eN_k) = rho_sku (j≠k), Var(eN_k)=1, and eN_k ⟂ basket — every SKU
keeps basis exactly β. The dealer's warehoused residual variance then uses::

    R_eff = rho_sku·(Σ_k D_k)² + (1−rho_sku)·Σ_k D_k²

which interpolates the independent case (rho=0 → Σ_k D_k², phase 2) and the
fully-correlated case (rho=1 → (Σ_k D_k)², i.e. one effective SKU — no
diversification). Swept at reluctance=1 (one-sided), where diversification is the
only thing that can form the market.

Usage::

    uv run python -m scripts.compute_basis_skucorr_experiment --n-seeds 100

Writes trials.csv, aggregate.csv, TABLE.md, skucorr_scurve.svg. FINDINGS curated.
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

from scripts.compute_futures_experiment import BASE, SIGMA, _h100_spot  # noqa: E402
from scripts.compute_basis_experiment import _u, _z, _corr  # noqa: E402
from scripts.compute_basis_sku_experiment import Principal, _population, _p  # noqa: E402

BETAS = [0.20, 0.45, 0.63, 0.80, 0.95, 1.0]
RHO_SKU = [0.0, 0.5, 0.9, 1.0]     # cross-SKU idio correlation
K_VALUES = [4, 12]
RELUCTANCE = 1.0                   # one-sided: diversification is the only lever
N_PRINCIPALS = 20
GAMMA_P = 1.0
GAMMA_D = 1.0


def _orthonormal(raws: List[List[float]], n: int) -> List[List[float]]:
    """Gram-Schmidt a list of raw vectors into an orthonormal set (Var=1, ⟂)."""
    basis: List[List[float]] = []
    for raw in raws:
        me = st.mean(raw)
        v = [x - me for x in raw]
        for u in basis:
            dot = sum(a * b for a, b in zip(v, u)) / sum(b * b for b in u)
            v = [a - dot * b for a, b in zip(v, u)]
        sv = st.pstdev(v)
        basis.append([a / sv for a in v])
    return basis


def _innovations_corr(n: int, K: int):
    """F, σ(I), basket iI, common idio c, and K own idio o_k — all orthonormal."""
    Is = [_h100_spot(s) for s in range(n)]
    F = st.mean(Is)
    sd_i = st.pstdev(Is)
    raws = [Is,                                             # → iI
            [BASE * math.exp(SIGMA * _z(s, -99, 500)) for s in range(n)]]  # → c
    for k in range(K):                                     # → o_k
        raws.append([BASE * math.exp(SIGMA * _z(s, -1 - k, 700 + 100 * k))
                     for s in range(n)])
    onb = _orthonormal(raws, n)
    return F, sd_i, onb[0], onb[1], onb[2:]


def _clear(pop: List[Principal], beta: float, rho: float, gamma_d: float,
           K: int) -> Tuple[List[Principal], float, float]:
    """One-sided clearing with correlated-SKU residual R_eff."""
    one_minus_r = 1.0 - beta * beta
    willing = [p for p in pop if p.side == 1 or p.withdraw_u >= RELUCTANCE]
    willing.sort(key=lambda p: p.s_max(), reverse=True)
    best_k, best_R, best_s = 0, 0.0, 0.0
    for k in range(1, len(willing) + 1):
        part = willing[:k]
        gross = sum(p.q for p in part)
        Dk = [0.0] * K
        for p in part:
            Dk[p.sku] += p.side * p.q
        R_eff = rho * (sum(Dk)) ** 2 + (1.0 - rho) * sum(d * d for d in Dk)
        s_req = 0.5 * gamma_d * R_eff * one_minus_r / gross if gross else 0.0
        if s_req <= part[-1].s_max() + 1e-12:
            best_k, best_R, best_s = k, R_eff, s_req
    return willing[:best_k], best_R, best_s


def _trial(seed, beta, rho, gamma_d, n_p, gamma_p, K, F, sd_i,
           iI_s, c_s, o_s) -> Dict[str, float]:
    V = sd_i * sd_i
    pop = _population(seed, n_p, gamma_p, K)
    part, R_eff, s_over_V = _clear(pop, beta, rho, gamma_d, K)
    total_gross = sum(p.q for p in pop)
    willing_gross = sum(p.q for p in pop
                        if p.side == 1 or p.withdraw_u >= RELUCTANCE)
    gross_part = sum(p.q for p in part)

    root = math.sqrt(1.0 - beta * beta)
    I = F + sd_i * iI_s
    Dk = [0.0] * K
    for p in part:
        Dk[p.sku] += p.side * p.q
    # eN_k = √rho·c + √(1−rho)·o_k ; residual P&L = −√(1−r)·σ·Σ_k D_k·eN_k
    rq, orq = math.sqrt(rho), math.sqrt(1.0 - rho)
    idio = rq * c_s * sum(Dk) + orq * sum(Dk[k] * o_s[k] for k in range(K))
    dealer_pnl_ex_spread = -root * sd_i * idio
    spread_dollar = s_over_V * V
    return {
        "seed": seed, "beta": beta, "rho_sku": rho, "K": K, "r_i": beta * beta,
        "volume_share": gross_part / total_gross if total_gross else 0.0,
        "willing_clear_frac": gross_part / willing_gross if willing_gross else 0.0,
        "residual_risk": math.sqrt(V * (1.0 - beta * beta) * R_eff),
        "clearing_spread_pct": 100.0 * spread_dollar / F if F else 0.0,
        "dealer_pnl": dealer_pnl_ex_spread + spread_dollar * gross_part,
        "spot_I": I,
    }


def _aggregate(rows: List[dict]) -> List[dict]:
    cells: Dict[Tuple[int, float, float], List[dict]] = {}
    for r in rows:
        cells.setdefault((r["K"], r["rho_sku"], r["beta"]), []).append(r)
    out = []
    for key in sorted(cells):
        c = cells[key]
        wc = [r["willing_clear_frac"] for r in c]
        out.append({
            "K": key[0], "rho_sku": key[1], "beta": key[2], "r_i": key[2] ** 2,
            "n": len(c),
            "willing_clear_frac_mean": st.mean(wc),
            "willing_clear_frac_p10": _p(wc, 10),
            "willing_clear_frac_p90": _p(wc, 90),
            "volume_share_median": st.median(r["volume_share"] for r in c),
            "residual_risk_mean": st.mean(r["residual_risk"] for r in c),
            "clearing_spread_pct_mean": st.mean(r["clearing_spread_pct"] for r in c),
        })
    return out


def _crit_beta(series: List[dict]) -> float:
    """Interpolated β where willing-clear crosses 0.5 (∞ if it never does)."""
    s = sorted(series, key=lambda a: a["beta"])
    for a, b in zip(s, s[1:]):
        ya, yb = a["willing_clear_frac_mean"], b["willing_clear_frac_mean"]
        if ya < 0.5 <= yb:
            return a["beta"] + (b["beta"] - a["beta"]) * (0.5 - ya) / (yb - ya)
    return s[0]["beta"] if s[0]["willing_clear_frac_mean"] >= 0.5 else float("inf")


_RHO_COLOR = {0.0: "#2c7a47", 0.5: "#6a9a2f", 0.9: "#c9860a", 1.0: "#b3341f"}


def _svg(agg: List[dict], out: Path, n: int, K: int) -> None:
    sub = [a for a in agg if a["K"] == K]
    w, h, pad = 680, 400, 64
    pw, ph = w - 2 * pad, h - 2 * pad

    def xs(b): return pad + pw * b
    def ys(v): return pad + ph - v * ph

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">',
         '<style>.t{font-size:14px;font-weight:600}.s{font-size:11px;fill:#666}'
         '.lab{font-size:11px;fill:#444}.med{fill:none;stroke-width:2}</style>',
         f'<rect width="{w}" height="{h}" fill="#fff"/>',
         f'<text class="t" x="{pad}" y="26">Cross-SKU correlation erodes '
         f'diversification — willing-clear, K={K}, one-sided (N={n})</text>',
         f'<text class="s" x="{pad}" y="43">band = p10–p90 · ρ=0 independent '
         f'(best case) → ρ=1 fully correlated (≈ single SKU)</text>']
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = ys(frac)
        p.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{pad+pw}" y2="{y:.1f}" '
                 f'stroke="#eee"/>')
        p.append(f'<text class="lab" x="{pad-8}" y="{y+4:.1f}" '
                 f'text-anchor="end">{frac:.0%}</text>')
    p.append(f'<line x1="{pad}" y1="{ys(0.5):.1f}" x2="{pad+pw}" '
             f'y2="{ys(0.5):.1f}" stroke="#bbb" stroke-dasharray="4 3"/>')
    for b in BETAS:
        p.append(f'<text class="lab" x="{xs(b):.1f}" y="{pad+ph+18:.1f}" '
                 f'text-anchor="middle">{b:.2g}</text>')
    by_rho: Dict[float, List[dict]] = {}
    for a in sub:
        by_rho.setdefault(a["rho_sku"], []).append(a)
    for rho, series in sorted(by_rho.items()):
        series.sort(key=lambda a: a["beta"])
        color = _RHO_COLOR.get(rho, "#555")
        top = " ".join(f"{xs(a['beta']):.1f},{ys(a['willing_clear_frac_p90']):.1f}"
                       for a in series)
        bot = " ".join(f"{xs(a['beta']):.1f},{ys(a['willing_clear_frac_p10']):.1f}"
                       for a in reversed(series))
        p.append(f'<polygon points="{top} {bot}" fill="{color}" '
                 f'fill-opacity="0.10" stroke="none"/>')
        line = " ".join(f"{xs(a['beta']):.1f},{ys(a['willing_clear_frac_mean']):.1f}"
                        for a in series)
        p.append(f'<polyline class="med" points="{line}" stroke="{color}"/>')
        for a in series:
            p.append(f'<circle cx="{xs(a["beta"]):.1f}" '
                     f'cy="{ys(a["willing_clear_frac_mean"]):.1f}" r="3" '
                     f'fill="{color}"/>')
    ly = pad + 14
    for rho in sorted(by_rho):
        color = _RHO_COLOR.get(rho, "#555")
        p.append(f'<rect x="{pad+pw-150}" y="{ly-9}" width="12" height="12" '
                 f'fill="{color}" fill-opacity="0.6"/>')
        p.append(f'<text class="lab" x="{pad+pw-134}" y="{ly+1}">ρ_sku={rho:g}</text>')
        ly += 18
    p.append(f'<text class="lab" x="{pad+pw/2}" y="{h-12}" '
             f'text-anchor="middle">basis correlation β  (r_i = β²) →</text>')
    p.append("</svg>")
    out.write_text("\n".join(p))
    print(f"wrote {out}")


def _table(agg: List[dict], n: int) -> str:
    lines = [
        f"# Compute basis — cross-SKU idio correlation — auto table (N={n})", "",
        "`rho_sku` = cross-SKU idiosyncratic correlation (0 independent … 1 fully "
        "correlated ≈ single SKU). reluctance=1 (one-sided). Order parameter: "
        "willing-clear fraction.", "",
        "| K | ρ_sku | β | r_i | willing clear (p10–p90) | volume share | "
        "residual risk | spread % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in sorted(agg, key=lambda a: (a["K"], a["rho_sku"], a["beta"])):
        lines.append(
            f"| {a['K']} | {a['rho_sku']:g} | {a['beta']:.2g} | {a['r_i']:.2f} | "
            f"{a['willing_clear_frac_mean']:.2f} "
            f"({a['willing_clear_frac_p10']:.2f}–{a['willing_clear_frac_p90']:.2f}) | "
            f"{a['volume_share_median']:.2f} | {a['residual_risk_mean']:.2f} | "
            f"{a['clearing_spread_pct_mean']:.2f}% |")
    lines += ["", "_Auto-generated. Narrative interpretation lives in FINDINGS.md._"]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--n-principals", type=int, default=N_PRINCIPALS)
    ap.add_argument("--gamma-dealer", type=float, default=GAMMA_D)
    ap.add_argument("--gamma-principal", type=float, default=GAMMA_P)
    ap.add_argument("--output", type=Path,
                    default=Path("runs/compute_basis_skucorr"))
    args = ap.parse_args(argv)
    N = args.n_seeds

    rows: List[dict] = []
    for K in K_VALUES:
        F, sd_i, iI, c, os = _innovations_corr(N, K)
        for rho in RHO_SKU:
            for beta in BETAS:
                for s in range(N):
                    o_s = [os[k][s] for k in range(K)]
                    rows.append(_trial(s, beta, rho, args.gamma_dealer,
                                       args.n_principals, args.gamma_principal,
                                       K, F, sd_i, iI[s], c[s], o_s))
    agg = _aggregate(rows)

    print(f"\n=== Compute basis / cross-SKU correlation, N={N} (one-sided) ===")
    for K in K_VALUES:
        print(f"\n  K={K} — willing-clear vs β, and critical β (½-max):")
        for rho in RHO_SKU:
            series = [a for a in agg if a["K"] == K and a["rho_sku"] == rho]
            row = "  ".join(
                f"β{a['beta']:.2g}={a['willing_clear_frac_mean']:.2f}"
                for a in sorted(series, key=lambda a: a["beta"]))
            cb = _crit_beta(series)
            cb_s = f"{cb:.2f}" if cb != float("inf") else "none"
            print(f"    ρ={rho:g}: {row}   → crit β={cb_s}")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trials.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader(); wri.writerows(rows)
    with (args.output / "aggregate.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        wri.writeheader(); wri.writerows(agg)
    (args.output / "TABLE.md").write_text(_table(agg, N))
    _svg(agg, args.output / "skucorr_scurve.svg", N, K=12)
    print(f"\nwrote {args.output}/trials.csv, aggregate.csv, TABLE.md")


if __name__ == "__main__":
    main()
