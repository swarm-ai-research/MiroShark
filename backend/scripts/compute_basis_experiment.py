"""Compute basis-risk & dealer intermediation — Option A headline sweep (bd umm).

Design: ``docs/plans/2026-06-30-compute-basis-dealer-experiment.md``.

Thesis under test: principals never touch a generic compute exchange — they
trade bespoke SKU forwards with OTC dealers, and dealers hedge the *fungible*
component on a generic basket exchange, warehousing the residual basis. Whether
a liquid central exchange forms is a function of the dealer basis R²
(``r_i = beta²`` — how much of a SKU forward's variance a dealer can lay off on
the basket). We sweep ``beta`` and look for a phase transition in exchange
participation.

This is the **Option A** variant (design §8): a *risk-averse* dealer prices its
spread off the residual variance it must warehouse, so exchange volume is
*emergent* from the principal↔dealer participation game — not imposed. The
existing compute-futures work (bd k9w/5ij) is the ``beta=1`` endpoint (perfect
hedge, 100% variance reduction); this sweeps beta *down* from there.

Model (standalone Monte-Carlo — the vendored kernel has no SKU/basket split yet,
design §7, so the headline is MC as specified; execution slippage is the
separable, already-characterised bd 5ij friction and is *not* stacked here):

- **Basket index** spot ``I`` is the exact k9w log-normal (median $3.50/GPU-hr,
  log-vol 0.5), reused for direct comparability.
- **SKU spot** is the synthetic factor model (design §3), in *innovation* space
  so E[S_i]=E[I] and Var(S_i)=Var(I) exactly, giving corr(S_i,I)=beta exactly::

      S_i = F + beta·(I − F) + √(1−beta²)·(eps − F),   eps ⟂ I, same dist as I

- **Principals** (N_P) are each long (buy SKU) or short (sell SKU) with random
  size and risk-aversion. Shorts (neoclouds) *withdraw* from selling forward
  with probability ``reluctance`` — the design's short-side-reluctance factor,
  and the neocloud "short-compute-AND-short-a-tech-transition" reluctance.
- **Dealer** intermediates *both* sides and NETS them: matched long/short SKU
  cancels, so the dealer only warehouses the **net imbalance** ``D`` after
  hedging ``beta·D`` on the basket. Residual (unhedgeable) variance =
  ``D²·V·(1−r_i)`` (V=Var(I)) — the load-bearing quantity. A risk-averse dealer
  charges a uniform clearing spread ``s = ½·gamma_D·D²·V·(1−r_i)/gross`` and a
  principal participates iff ``s ≤ ½·gamma_P·q·V`` (its willingness to pay to
  shed its own variance). V cancels in the participation condition, so the
  transition location is scale-free in price volatility.

Because the dealer nets, ``reluctance=1.0`` reproduces the design §4 *one-sided*
model (no shorts → dealer warehouses gross → the predicted S-curve), while
``reluctance=0.0`` is fully netted (imbalance ~√N_P → basis barely bites). The
sweep spans both so the headline (reluctance=0) can be interpreted honestly.

Usage::

    uv run python -m scripts.compute_basis_experiment --n-seeds 50

Writes trials.csv, aggregate.csv, TABLE.md, basis_scurve.svg to the output dir.
FINDINGS.md is hand-curated separately (CLAUDE.md methodology).
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics as st
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

logging.disable(logging.CRITICAL)

# Reuse k9w's exact spot process + percentile helper for direct comparability.
from scripts.compute_futures_experiment import _h100_spot, _pct  # noqa: E402

BETAS = [0.20, 0.45, 0.63, 0.80, 0.95, 1.0]   # 1.0 = k9w perfect-hedge anchor
RELUCTANCE = [0.0, 0.5, 1.0]                   # 0=balanced … 1=one-sided (§4)
N_PRINCIPALS = 20
GAMMA_P = 1.0        # baseline principal risk-aversion (per-principal jitter ×)
GAMMA_D = 1.0        # dealer risk-aversion (symmetric baseline; CLI-overridable)
SIZE_LOGVOL = 0.4    # dispersion of principal sizes q (lognormal around ~1)
GAMMA_LOGVOL = 0.3   # dispersion of principal risk-aversion


# --------------------------------------------------------------- hashing
def _u(seed: int, idx: int, salt: int) -> float:
    """Deterministic uniform in [0,1) from (seed, principal idx, salt). No RNG
    (the workflow forbids it) — an integer avalanche hash, reproducible."""
    x = (seed * 73856093) ^ (idx * 19349663) ^ (salt * 83492791)
    x = (x * 2654435761) & 0xFFFFFFFF
    x ^= x >> 13
    x = (x * 1274126177) & 0xFFFFFFFF
    x ^= x >> 16
    return (x & 0xFFFF) / 65536.0


def _z(seed: int, idx: int, salt: int) -> float:
    """Quasi-normal (mean 0, std ≈ 0.577): centred sum of 4 salted uniforms."""
    return sum(_u(seed, idx, salt + k) for k in range(4)) - 2.0


def _sku_idio(seed: int) -> float:
    """Idiosyncratic SKU spot draw ``eps`` — same log-normal family as the basket
    index (so Var(eps)=Var(I)), but a decorrelated hash stream (idx=-1)."""
    from scripts.compute_futures_experiment import BASE, SIGMA
    import math
    return BASE * math.exp(SIGMA * _z(seed, -1, 700))


def _innovations(n: int) -> Tuple[float, float, List[float], List[float]]:
    """Fair forward F=E[I], σ(I), and standardised orthonormal innovations
    ``iI`` (basket) and ``eN`` (idiosyncratic), Gram-Schmidt'd so that
    ``eN ⟂ iI`` and both have unit variance *in-sample*. This makes the swept
    basis noise-free: with ``S_i = F + σ·(β·iI + √(1−β²)·eN)`` and
    ``I = F + σ·iI``, corr(S_i, I) = β and Var(S_i) = Var(I) exactly."""
    Is = [_h100_spot(s) for s in range(n)]
    F = st.mean(Is)
    sd_i = st.pstdev(Is)
    iI = [(x - F) / sd_i for x in Is]
    eps = [_sku_idio(s) for s in range(n)]
    me = st.mean(eps)
    ev = [x - me for x in eps]
    b = sum(e * z for e, z in zip(ev, iI)) / sum(z * z for z in iI)
    res = [e - b * z for e, z in zip(ev, iI)]     # ⟂ iI, mean 0
    sr = st.pstdev(res)
    eN = [x / sr for x in res]                     # unit variance
    return F, sd_i, iI, eN


# --------------------------------------------------------------- population
@dataclass
class Principal:
    side: int      # +1 long (buys SKU), −1 short (sells SKU)
    q: float       # size (SKU units)
    gamma: float   # risk-aversion
    withdraw_u: float  # short withdraws from forward if withdraw_u < reluctance

    def s_max(self) -> float:
        """Reservation spread per unit (÷V): willingness to pay to fully hedge
        its own variance. V-free so it cancels against the dealer's ask."""
        return 0.5 * self.gamma * self.q


def _population(seed: int, n_p: int, gamma_p: float) -> List[Principal]:
    """Draw the principal population for a seed (β/reluctance-independent)."""
    import math
    pop = []
    for j in range(n_p):
        side = 1 if _u(seed, j, 11) < 0.5 else -1
        q = math.exp(SIZE_LOGVOL * _z(seed, j, 20))
        gamma = gamma_p * math.exp(GAMMA_LOGVOL * _z(seed, j, 40))
        pop.append(Principal(side, q, gamma, _u(seed, j, 60)))
    return pop


# --------------------------------------------------------------- clearing
def _clear(pop: List[Principal], beta: float, reluctance: float,
           gamma_d: float) -> Tuple[List[Principal], float, float]:
    """Endogenous clearing: which willing principals the risk-averse dealer can
    profitably intermediate, and at what uniform spread (÷V). Returns
    (participants, net_imbalance D, clearing_spread_over_V)."""
    r_i = beta * beta
    one_minus_r = 1.0 - r_i
    # Longs always willing; shorts withdraw with prob=reluctance.
    willing = [p for p in pop if p.side == 1 or p.withdraw_u >= reluctance]
    # Serve highest willingness-to-pay first (dealer fills the keenest demand).
    willing.sort(key=lambda p: p.s_max(), reverse=True)

    best_k, best_D, best_s = 0, 0.0, 0.0
    for k in range(1, len(willing) + 1):
        part = willing[:k]
        gross = sum(p.q for p in part)
        D = sum(p.side * p.q for p in part)          # net imbalance
        # Dealer's required per-unit spread to warehouse the net residual.
        s_req = 0.5 * gamma_d * D * D * one_minus_r / gross if gross else 0.0
        # Feasible iff the marginal (lowest-WTP) participant still accepts.
        if s_req <= part[-1].s_max() + 1e-12:
            best_k, best_D, best_s = k, D, s_req      # largest feasible clearing
    return willing[:best_k], best_D, best_s


# --------------------------------------------------------------- one trial
def _trial(seed: int, beta: float, reluctance: float, gamma_d: float,
           n_p: int, gamma_p: float, F: float, sd_i: float,
           iI_s: float, eN_s: float) -> Dict[str, float]:
    """One market realisation: clear participation, realise (I, S_i), settle.

    ``iI_s`` and ``eN_s`` are the seed's *standardised, orthonormal* innovations
    (Var=1, iI ⟂ eN across the ensemble — Gram-Schmidt in ``main``), so
    ``I = F + sd·iI`` and ``S_i = F + sd·(β·iI + √(1−β²)·eN)`` give
    corr(S_i, I) = β and Var(S_i) = Var(I) *exactly* in-sample. The basis knob
    is thus noise-free: r_i = β² by construction."""
    import math
    V = sd_i * sd_i
    pop = _population(seed, n_p, gamma_p)
    part, D, s_over_V = _clear(pop, beta, reluctance, gamma_d)

    total_gross = sum(p.q for p in pop)
    willing_gross = sum(p.q for p in pop
                        if p.side == 1 or p.withdraw_u >= reluctance)
    gross_part = sum(p.q for p in part)

    root = math.sqrt(1.0 - beta * beta)
    I = F + sd_i * iI_s
    S_i = F + sd_i * (beta * iI_s + root * eN_s)

    spread_dollar = s_over_V * V                      # per-unit spread in $
    # Dealer P&L ex-spread = −D·√(1−r)·(idio innovation): mean 0, var = D²·(1−r)·V.
    dealer_pnl_ex_spread = -D * root * sd_i * eN_s
    dealer_pnl = dealer_pnl_ex_spread + spread_dollar * gross_part

    # Fraction of principal *risk* actually hedged. Every participant locks F,
    # shedding 100% of its own (q²·V) variance; non-participants keep theirs. All
    # principals share S_i, so demand-weighting by q² is the clean, price-noise-
    # free coverage of the principal-side risk (complements the Σq volume share).
    sum_q2_all = sum(p.q * p.q for p in pop)
    sum_q2_part = sum(p.q * p.q for p in part)

    residual_risk = abs(D) * math.sqrt(V * (1.0 - beta * beta))  # std units

    return {
        "seed": seed, "beta": beta, "reluctance": reluctance,
        "r_i": beta * beta,
        "volume_share": gross_part / total_gross if total_gross else 0.0,
        "willing_clear_frac": gross_part / willing_gross if willing_gross else 0.0,
        "risk_hedged_frac": sum_q2_part / sum_q2_all if sum_q2_all else 0.0,
        "net_imbalance": D,
        "gross_intermediated": gross_part,
        "clearing_spread_pct": 100.0 * spread_dollar / F if F else 0.0,
        "residual_risk": residual_risk,
        "dealer_pnl": dealer_pnl,
        "spot_I": I, "spot_S": S_i,
    }


# --------------------------------------------------------------- aggregate
def _aggregate(rows: List[dict]) -> List[dict]:
    """Per (beta, reluctance) cell: mean/median/p10/p90 of the order parameter
    and supporting metrics, plus principal-cashflow variance reduction and the
    realised corr(S_i, I) as a factor-model sanity check."""
    cells: Dict[Tuple[float, float], List[dict]] = {}
    for r in rows:
        cells.setdefault((r["beta"], r["reluctance"]), []).append(r)
    out = []
    for (beta, rel) in sorted(cells):
        c = cells[(beta, rel)]
        vs = [r["volume_share"] for r in c]
        # Realised corr(S_i, I) across seeds (should ≈ beta).
        Is = [r["spot_I"] for r in c]
        Ss = [r["spot_S"] for r in c]
        corr = _corr(Is, Ss)
        out.append({
            "beta": beta, "reluctance": rel, "r_i": beta * beta, "n": len(c),
            "corr_S_I": corr,
            "volume_share_mean": st.mean(vs),
            "volume_share_median": st.median(vs),
            "volume_share_p10": _pct(vs, 10),
            "volume_share_p90": _pct(vs, 90),
            "willing_clear_frac_mean": st.mean(r["willing_clear_frac"] for r in c),
            "risk_hedged_frac_mean": st.mean(r["risk_hedged_frac"] for r in c),
            "clearing_spread_pct_mean": st.mean(r["clearing_spread_pct"] for r in c),
            "net_imbalance_mean_abs": st.mean(abs(r["net_imbalance"]) for r in c),
            "residual_risk_mean": st.mean(r["residual_risk"] for r in c),
            "dealer_pnl_mean": st.mean(r["dealer_pnl"] for r in c),
            "dealer_pnl_std": st.pstdev(r["dealer_pnl"] for r in c),
        })
    return out


def _corr(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = st.mean(xs), st.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx, sy = st.pstdev(xs), st.pstdev(ys)
    return cov / (sx * sy) if sx and sy else 0.0


# --------------------------------------------------------------- SVG S-curve
_REL_STYLE = {0.0: ("#2c7a47", "balanced (reluctance 0)"),
              0.5: ("#c9860a", "half-withdrawn (0.5)"),
              1.0: ("#b3341f", "one-sided (1.0) — design §4")}


def _svg_scurve(agg: List[dict], out: Path, n: int) -> None:
    """Exchange volume share vs beta: median + p10–p90 band, one series per
    reluctance level. The order parameter of the phase transition."""
    w, h, pad = 680, 400, 64
    pw, ph = w - 2 * pad, h - 2 * pad

    def xs(b): return pad + pw * b                        # beta ∈ [0,1] linear
    def ys(v): return pad + ph - v * ph                   # share ∈ [0,1]

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">',
         '<style>.t{font-size:14px;font-weight:600}.s{font-size:11px;fill:#666}'
         '.lab{font-size:11px;fill:#444}.med{fill:none;stroke-width:2}</style>',
         f'<rect width="{w}" height="{h}" fill="#fff"/>',
         f'<text class="t" x="{pad}" y="26">Compute basis phase transition: '
         f'exchange volume share vs basis β (N={n})</text>',
         f'<text class="s" x="{pad}" y="43">band = p10–p90 across seeds · '
         f'β=1 is the k9w perfect-hedge anchor · r_i = β²</text>']
    # gridlines + axes
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = ys(frac)
        p.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{pad+pw}" y2="{y:.1f}" '
                 f'stroke="#eee"/>')
        p.append(f'<text class="lab" x="{pad-8}" y="{y+4:.1f}" '
                 f'text-anchor="end">{frac:.0%}</text>')
    for b in BETAS:
        p.append(f'<text class="lab" x="{xs(b):.1f}" y="{pad+ph+18:.1f}" '
                 f'text-anchor="middle">{b:.2g}</text>')

    by_rel: Dict[float, List[dict]] = {}
    for a in agg:
        by_rel.setdefault(a["reluctance"], []).append(a)
    for rel, series in sorted(by_rel.items()):
        series.sort(key=lambda a: a["beta"])
        color, _ = _REL_STYLE.get(rel, ("#555", str(rel)))
        top = " ".join(f"{xs(a['beta']):.1f},{ys(a['volume_share_p90']):.1f}"
                       for a in series)
        bot = " ".join(f"{xs(a['beta']):.1f},{ys(a['volume_share_p10']):.1f}"
                       for a in reversed(series))
        p.append(f'<polygon points="{top} {bot}" fill="{color}" '
                 f'fill-opacity="0.13" stroke="none"/>')
        line = " ".join(f"{xs(a['beta']):.1f},{ys(a['volume_share_median']):.1f}"
                        for a in series)
        p.append(f'<polyline class="med" points="{line}" stroke="{color}"/>')
        for a in series:
            p.append(f'<circle cx="{xs(a["beta"]):.1f}" '
                     f'cy="{ys(a["volume_share_median"]):.1f}" r="3" '
                     f'fill="{color}"/>')
    # legend
    ly = pad + 14
    for rel in sorted(by_rel):
        color, label = _REL_STYLE.get(rel, ("#555", str(rel)))
        p.append(f'<rect x="{pad+pw-186}" y="{ly-9}" width="12" height="12" '
                 f'fill="{color}" fill-opacity="0.5"/>')
        p.append(f'<text class="lab" x="{pad+pw-170}" y="{ly+1}">{label}</text>')
        ly += 18
    p.append(f'<text class="lab" x="{pad+pw/2}" y="{h-12}" '
             f'text-anchor="middle">basis correlation β  (r_i = β²) →</text>')
    p.append("</svg>")
    out.write_text("\n".join(p))
    print(f"wrote {out}")


# --------------------------------------------------------------- TABLE.md
def _table(agg: List[dict], n: int) -> str:
    lines = [
        f"# Compute basis-risk & dealer intermediation — auto table (N={n})",
        "",
        "Order parameter: **exchange volume share** (fraction of principal "
        "demand intermediated onto the basket exchange). `reluctance` = "
        "short-side (neocloud) withdrawal; 1.0 = the design §4 one-sided model.",
        "",
        "| β | r_i | reluctance | corr(S,I) | volume share (p10–p90) | willing "
        "clear | risk hedged | spread % | net imbal | dealer P&L μ±σ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in sorted(agg, key=lambda a: (a["reluctance"], a["beta"])):
        lines.append(
            f"| {a['beta']:.2g} | {a['r_i']:.2f} | {a['reluctance']:.1f} | "
            f"{a['corr_S_I']:.2f} | {a['volume_share_median']:.2f} "
            f"({a['volume_share_p10']:.2f}–{a['volume_share_p90']:.2f}) | "
            f"{a['willing_clear_frac_mean']:.2f} | "
            f"{a['risk_hedged_frac_mean']:.2f} | "
            f"{a['clearing_spread_pct_mean']:.2f}% | "
            f"{a['net_imbalance_mean_abs']:.2f} | "
            f"{a['dealer_pnl_mean']:.2f}±{a['dealer_pnl_std']:.2f} |")
    lines += ["", "_Auto-generated. Narrative interpretation lives in FINDINGS.md._"]
    return "\n".join(lines)


# --------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=50)
    ap.add_argument("--n-principals", type=int, default=N_PRINCIPALS)
    ap.add_argument("--gamma-dealer", type=float, default=GAMMA_D)
    ap.add_argument("--gamma-principal", type=float, default=GAMMA_P)
    ap.add_argument("--output", type=Path, default=Path("runs/compute_basis"))
    args = ap.parse_args(argv)
    N = args.n_seeds

    # Fair forwards F=G=E[spot] (no pricing edge — k9w discipline); noise-free
    # basis via orthonormal innovations (Gram-Schmidt in _innovations).
    F, sd_i, iI, eN = _innovations(N)

    rows: List[dict] = []
    for beta in BETAS:
        for rel in RELUCTANCE:
            for s in range(N):
                rows.append(_trial(s, beta, rel, args.gamma_dealer,
                                   args.n_principals, args.gamma_principal,
                                   F, sd_i, iI[s], eN[s]))
    agg = _aggregate(rows)

    print(f"\n=== Compute basis / dealer intermediation (Option A), N={N} ===")
    print(f"  basket F=${F:.2f}/GPU-hr  Var(I)={sd_i * sd_i:.2f}  "
          f"N_principals={args.n_principals}  γ_D={args.gamma_dealer} "
          f"γ_P={args.gamma_principal}")
    for rel in RELUCTANCE:
        print(f"\n  reluctance={rel:.1f}:")
        for a in sorted((a for a in agg if a["reluctance"] == rel),
                        key=lambda a: a["beta"]):
            print(f"    β={a['beta']:.2g} (r_i={a['r_i']:.2f}): "
                  f"vol_share={a['volume_share_median']:.2f} "
                  f"[{a['volume_share_p10']:.2f},{a['volume_share_p90']:.2f}] "
                  f"spread={a['clearing_spread_pct_mean']:.2f}% "
                  f"|D|={a['net_imbalance_mean_abs']:.1f} "
                  f"risk_hedged={a['risk_hedged_frac_mean']:.2f}  "
                  f"corr(S,I)={a['corr_S_I']:.2f}")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trials.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wri.writeheader(); wri.writerows(rows)
    with (args.output / "aggregate.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        wri.writeheader(); wri.writerows(agg)
    (args.output / "TABLE.md").write_text(_table(agg, N))
    _svg_scurve(agg, args.output / "basis_scurve.svg", N)
    print(f"\nwrote {args.output}/trials.csv, aggregate.csv, TABLE.md")


if __name__ == "__main__":
    main()
