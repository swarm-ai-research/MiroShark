"""Compute basis — critical-β sensitivity to γ-ratio and N_P (bd keq, ← umm).

Phase 4. Every prior phase carried the caveat that the *location* of the
one-sided phase transition (its critical basis β*) depends on the dealer/
principal risk-aversion ratio and the market size. This maps that surface and
checks it against an analytic prediction, turning the caveat into a result.

One-sided single-SKU regime (reluctance=1), where the transition is sharp. For a
uniform-spread dealer, principal ``k`` (of the willing, highest-WTP-first)
accepts iff ``½γ_D·k·q̄·(1−r) ≤ ½γ_P·q̄`` → ``k ≤ (γ_P/γ_D)/(1−r)``. With
willing longs ≈ N_P/2, the willing-clear fraction crosses ½ at

    (1−r*) = 4·(γ_P/γ_D)/N_P     ⇒     β* ≈ √(1 − 4·(γ_P/γ_D)/N_P)

i.e. the market forms at *worse* basis (lower β*) when the dealer is more
risk-tolerant than principals (high γ_P/γ_D) or the market is smaller (low N_P);
concentration of one-sided risk (high N_P) *raises* β*.

The willing-clear fraction depends only on the clearing game, not on realised
prices, so this needs no Monte-Carlo spot draws — just the population + clearing
across seeds. Reuses the headline ``_population``/``_clear``.

Usage::

    uv run python -m scripts.compute_basis_sensitivity_experiment --n-seeds 100

Writes aggregate.csv, TABLE.md, critbeta_sensitivity.svg. FINDINGS hand-curated.
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

from scripts.compute_basis_experiment import _population, _clear  # noqa: E402

GAMMA_RATIOS = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]   # γ_P / γ_D (γ_D fixed = 1)
NP_VALUES = [10, 20, 40]
BETA_GRID = [round(0.10 + 0.02 * i, 3) for i in range(46)]   # 0.10 … 1.00
RELUCTANCE = 1.0


def _willing_clear(seed: int, beta: float, n_p: int, gamma_p: float,
                   gamma_d: float) -> float:
    pop = _population(seed, n_p, gamma_p)
    part, _, _ = _clear(pop, beta, RELUCTANCE, gamma_d)
    willing_gross = sum(p.q for p in pop
                        if p.side == 1 or p.withdraw_u >= RELUCTANCE)
    return sum(p.q for p in part) / willing_gross if willing_gross else 0.0


def _crit_beta(betas: List[float], wc: List[float]) -> float:
    """Interpolated β where willing-clear first reaches ½ (∞ if never)."""
    for (ba, ya), (bb, yb) in zip(zip(betas, wc), zip(betas[1:], wc[1:])):
        if ya < 0.5 <= yb:
            return ba + (bb - ba) * (0.5 - ya) / (yb - ya)
    return betas[0] if wc[0] >= 0.5 else float("inf")


def _analytic_crit_beta(ratio: float, n_p: int) -> float:
    inside = 1.0 - 4.0 * ratio / n_p
    return math.sqrt(inside) if inside > 0 else 0.0


def _run(N: int, gamma_d: float) -> List[dict]:
    out = []
    for n_p in NP_VALUES:
        for ratio in GAMMA_RATIOS:
            gamma_p = ratio * gamma_d
            wc = [st.mean(_willing_clear(s, b, n_p, gamma_p, gamma_d)
                          for s in range(N)) for b in BETA_GRID]
            out.append({
                "n_p": n_p, "gamma_ratio": ratio,
                "crit_beta_empirical": _crit_beta(BETA_GRID, wc),
                "crit_beta_analytic": _analytic_crit_beta(ratio, n_p),
            })
    return out


_NP_COLOR = {10: "#2c7a47", 20: "#c9860a", 40: "#b3341f"}


def _fmt(b: float) -> str:
    return "—" if b == float("inf") else f"{b:.2f}"


def _svg(agg: List[dict], out: Path, n: int) -> None:
    """Critical β vs γ_P/γ_D, one series per N_P; dashed = analytic prediction."""
    w, h, pad = 680, 400, 66
    pw, ph = w - 2 * pad, h - 2 * pad
    xr = (min(GAMMA_RATIOS), max(GAMMA_RATIOS))

    def xs(r): return pad + pw * (r - xr[0]) / (xr[1] - xr[0])
    def ys(v): return pad + ph - max(0.0, min(1.0, v)) * ph

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">',
         '<style>.t{font-size:14px;font-weight:600}.s{font-size:11px;fill:#666}'
         '.lab{font-size:11px;fill:#444}.med{fill:none;stroke-width:2}'
         '.ana{fill:none;stroke-width:1.3;stroke-dasharray:5 3}</style>',
         f'<rect width="{w}" height="{h}" fill="#fff"/>',
         f'<text class="t" x="{pad}" y="26">Critical basis β* vs dealer risk '
         f'tolerance, by market size (one-sided, N={n})</text>',
         f'<text class="s" x="{pad}" y="43">solid = simulated · dashed = '
         f'β*≈√(1−4·(γ_P/γ_D)/N_P) · higher N_P (more concentrated one-sided '
         f'risk) ⇒ higher β*</text>']
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = ys(frac)
        p.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{pad+pw}" y2="{y:.1f}" '
                 f'stroke="#eee"/>')
        p.append(f'<text class="lab" x="{pad-8}" y="{y+4:.1f}" '
                 f'text-anchor="end">{frac:.2f}</text>')
    for r in GAMMA_RATIOS:
        p.append(f'<text class="lab" x="{xs(r):.1f}" y="{pad+ph+18:.1f}" '
                 f'text-anchor="middle">{r:g}</text>')
    by_np: Dict[int, List[dict]] = {}
    for a in agg:
        by_np.setdefault(a["n_p"], []).append(a)
    for n_p, series in sorted(by_np.items()):
        series.sort(key=lambda a: a["gamma_ratio"])
        color = _NP_COLOR.get(n_p, "#555")
        emp = [(a["gamma_ratio"], a["crit_beta_empirical"]) for a in series
               if a["crit_beta_empirical"] != float("inf")]
        ana = [(a["gamma_ratio"], a["crit_beta_analytic"]) for a in series]
        p.append('<polyline class="med" stroke="%s" points="%s"/>' % (
            color, " ".join(f"{xs(r):.1f},{ys(v):.1f}" for r, v in emp)))
        p.append('<polyline class="ana" stroke="%s" points="%s"/>' % (
            color, " ".join(f"{xs(r):.1f},{ys(v):.1f}" for r, v in ana)))
        for r, v in emp:
            p.append(f'<circle cx="{xs(r):.1f}" cy="{ys(v):.1f}" r="3" '
                     f'fill="{color}"/>')
    ly = pad + 14
    for n_p in sorted(by_np):
        color = _NP_COLOR.get(n_p, "#555")
        p.append(f'<rect x="{pad+pw-150}" y="{ly-9}" width="12" height="12" '
                 f'fill="{color}" fill-opacity="0.7"/>')
        p.append(f'<text class="lab" x="{pad+pw-134}" y="{ly+1}">N_P={n_p}</text>')
        ly += 18
    p.append(f'<text class="lab" x="{pad+pw/2}" y="{h-12}" text-anchor="middle">'
             f'principal/dealer risk-aversion ratio γ_P/γ_D →</text>')
    p.append("</svg>")
    out.write_text("\n".join(p))
    print(f"wrote {out}")


def _table(agg: List[dict], n: int) -> str:
    lines = [
        f"# Critical basis β* — γ-ratio × N_P sensitivity (one-sided, N={n})", "",
        "β* = basis where the one-sided willing-clear fraction crosses ½ "
        "(interpolated on a 0.02 β-grid). Analytic: β*≈√(1−4·(γ_P/γ_D)/N_P).", "",
        "| N_P | γ_P/γ_D | β* empirical | β* analytic |",
        "|---|---|---|---|",
    ]
    for a in sorted(agg, key=lambda a: (a["n_p"], a["gamma_ratio"])):
        lines.append(f"| {a['n_p']} | {a['gamma_ratio']:g} | "
                     f"{_fmt(a['crit_beta_empirical'])} | "
                     f"{_fmt(a['crit_beta_analytic'])} |")
    lines += ["", "_Auto-generated. Narrative interpretation lives in FINDINGS.md._"]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--gamma-dealer", type=float, default=1.0)
    ap.add_argument("--output", type=Path,
                    default=Path("runs/compute_basis_sensitivity"))
    args = ap.parse_args(argv)
    N = args.n_seeds

    agg = _run(N, args.gamma_dealer)
    print(f"\n=== Critical basis β* sensitivity, N={N} (one-sided) ===")
    for n_p in NP_VALUES:
        print(f"\n  N_P={n_p}:  γ_P/γ_D → β* (empirical | analytic)")
        for a in (x for x in agg if x["n_p"] == n_p):
            print(f"    ratio={a['gamma_ratio']:>4g}:  "
                  f"{_fmt(a['crit_beta_empirical'])}  |  "
                  f"{_fmt(a['crit_beta_analytic'])}")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "aggregate.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        wri.writeheader(); wri.writerows(agg)
    (args.output / "TABLE.md").write_text(_table(agg, N))
    _svg(agg, args.output / "critbeta_sensitivity.svg", N)
    print(f"\nwrote {args.output}/aggregate.csv, TABLE.md")


if __name__ == "__main__":
    main()
