"""Forward-contract default -> retreat to spot (bd 79l, ← umm; Pirrong lesson).

Pirrong's counterparty-performance risk: a party reneges on a forward when
performing costs more than the reneging penalty. In cotton 2010-11, sellers
defaulted when prices rose and buyers when they fell, so traders ended up
"dealing only on a spot basis." The forward's protection is therefore **capped**
at the enforcement level κ — and it is truncated *exactly in the tail*, where a
hedge matters most.

Model a defaultable SKU forward struck at the fair ``F``. The realised per-unit
hedge payoff is::

    h(κ) = clip(S − F, −κ, +κ)

(beyond ±κ the losing side walks away and pays only κ). The hedge's variance
reduction is then::

    E(κ) = 1 − Var((S−F) − h(κ)) / Var(S − F)

which we read as the **forward-market participation proxy**: under heterogeneous
hedgers (each hedges iff its risk-reduction benefit ∝ E(κ) beats a private cost),
participation is monotone in E(κ). As enforcement weakens (κ → 0) the forward
protects nothing and the market **retreats to spot** (E → 0). We sweep κ in units
of the price move's σ, across spot log-vols — showing that **fatter-tailed prices
need stronger enforcement** for a forward market to survive.

This is a default gate on exchange viability that is *orthogonal* to basis
hedgeability (phases 1-4) and dealer funding (bd b2s).

Usage::

    uv run python -m scripts.compute_forward_default_experiment --n-seeds 400

Writes aggregate.csv, TABLE.md, default_curve.svg. FINDINGS hand-curated.
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

from scripts.compute_basis_experiment import _u, _z  # noqa: E402

BASE = 3.50                                   # fair forward / spot median
KAPPA_OVER_SIGMA = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 1e9]   # 1e9 = full enforcement
SPOT_LOGVOLS = [0.3, 0.5, 0.8]                # SKU spot log-vol (tail heaviness)
N_BOOT = 300                                   # deterministic bootstrap resamples


def _spot(seed: int, sigma: float) -> float:
    """Deterministic per-seed SKU spot, log-normal around BASE at log-vol σ."""
    return BASE * math.exp(sigma * _z(seed, 5, 111))


def _effectiveness(devs: List[float], var_unh: float, k: float,
                   sigma_s: float) -> float:
    """Variance reduction of a forward whose protection is capped at ±k·σ_s."""
    kappa = k * sigma_s
    res = [d - max(-kappa, min(d, kappa)) for d in devs]   # tail beyond ±κ
    return 1.0 - st.pvariance(res) / var_unh if var_unh else 0.0


def _boot_indices(b: int, n: int) -> List[int]:
    """Deterministic bootstrap resample of [0,n) (no RNG — hashed picks)."""
    return [int(_u(b, i, 47) * n) % n for i in range(n)]


def _aggregate(N: int) -> List[dict]:
    out = []
    for sigma in SPOT_LOGVOLS:
        spots = [_spot(s, sigma) for s in range(N)]
        F = st.mean(spots)
        devs = [x - F for x in spots]
        var_unh = st.pvariance(devs)
        sigma_s = math.sqrt(var_unh)
        for k in KAPPA_OVER_SIGMA:
            point = _effectiveness(devs, var_unh, k, sigma_s)
            # Bootstrap CI over the price ensemble.
            boots = []
            for b in range(N_BOOT):
                idx = _boot_indices(b, N)
                d = [devs[i] for i in idx]
                vu = st.pvariance(d)
                boots.append(_effectiveness(d, vu, k, sigma_s) if vu else 0.0)
            boots.sort()
            out.append({
                "spot_logvol": sigma, "kappa_over_sigma": k,
                "effectiveness": point,
                "eff_p10": _pct(boots, 10), "eff_p90": _pct(boots, 90),
            })
    return out


def _pct(ys: List[float], q: float) -> float:
    if not ys:
        return 0.0
    i = (q / 100.0) * (len(ys) - 1)
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return ys[lo] + (ys[hi] - ys[lo]) * (i - lo)


_VOL_COLOR = {0.3: "#2c7a47", 0.5: "#c9860a", 0.8: "#b3341f"}


def _svg(agg: List[dict], out: Path, n: int) -> None:
    """Forward hedge effectiveness (participation proxy) vs enforcement κ/σ,
    one series per spot log-vol. Finite κ shown on a log x-axis."""
    finite = [k for k in KAPPA_OVER_SIGMA if k < 1e8]
    w, h, pad = 680, 400, 66
    pw, ph = w - 2 * pad, h - 2 * pad
    xlogs = [math.log(k) for k in finite]
    x0, x1 = min(xlogs), max(xlogs)

    def xs(k): return pad + pw * (math.log(k) - x0) / ((x1 - x0) or 1.0)
    def ys(v): return pad + ph - max(0.0, min(1.0, v)) * ph

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif">',
         '<style>.t{font-size:14px;font-weight:600}.s{font-size:11px;fill:#666}'
         '.lab{font-size:11px;fill:#444}.med{fill:none;stroke-width:2}</style>',
         f'<rect width="{w}" height="{h}" fill="#fff"/>',
         f'<text class="t" x="{pad}" y="26">Forward default -> retreat to spot: '
         f'hedge effectiveness vs enforcement (N={n})</text>',
         f'<text class="s" x="{pad}" y="43">band = bootstrap p10–p90 · '
         f'participation ∝ effectiveness · fatter tails need larger κ</text>']
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = ys(frac)
        p.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{pad+pw}" y2="{y:.1f}" '
                 f'stroke="#eee"/>')
        p.append(f'<text class="lab" x="{pad-8}" y="{y+4:.1f}" '
                 f'text-anchor="end">{frac:.0%}</text>')
    for k in finite:
        p.append(f'<text class="lab" x="{xs(k):.1f}" y="{pad+ph+18:.1f}" '
                 f'text-anchor="middle">{k:g}σ</text>')
    by_vol: Dict[float, List[dict]] = {}
    for a in agg:
        if a["kappa_over_sigma"] < 1e8:
            by_vol.setdefault(a["spot_logvol"], []).append(a)
    for sigma, series in sorted(by_vol.items()):
        series.sort(key=lambda a: a["kappa_over_sigma"])
        color = _VOL_COLOR.get(sigma, "#555")
        top = " ".join(f"{xs(a['kappa_over_sigma']):.1f},{ys(a['eff_p90']):.1f}"
                       for a in series)
        bot = " ".join(f"{xs(a['kappa_over_sigma']):.1f},{ys(a['eff_p10']):.1f}"
                       for a in reversed(series))
        p.append(f'<polygon points="{top} {bot}" fill="{color}" '
                 f'fill-opacity="0.12" stroke="none"/>')
        line = " ".join(f"{xs(a['kappa_over_sigma']):.1f},{ys(a['effectiveness']):.1f}"
                        for a in series)
        p.append(f'<polyline class="med" points="{line}" stroke="{color}"/>')
        for a in series:
            p.append(f'<circle cx="{xs(a["kappa_over_sigma"]):.1f}" '
                     f'cy="{ys(a["effectiveness"]):.1f}" r="3" fill="{color}"/>')
    ly = pad + 14
    for sigma in sorted(by_vol):
        color = _VOL_COLOR.get(sigma, "#555")
        p.append(f'<rect x="{pad+pw-150}" y="{ly-9}" width="12" height="12" '
                 f'fill="{color}" fill-opacity="0.7"/>')
        p.append(f'<text class="lab" x="{pad+pw-134}" y="{ly+1}">spot σ={sigma:g}</text>')
        ly += 18
    p.append(f'<text class="lab" x="{pad+pw/2}" y="{h-12}" text-anchor="middle">'
             f'enforcement κ (÷ price-move σ, log) →</text>')
    p.append("</svg>")
    out.write_text("\n".join(p))
    print(f"wrote {out}")


def _table(agg: List[dict], n: int) -> str:
    lines = [
        f"# Forward default -> retreat to spot — auto table (N={n})", "",
        "Effectiveness = variance reduction of a forward whose protection is "
        "capped at ±κ (default beyond that). κ in units of the price-move σ. "
        "Participation ∝ effectiveness.", "",
        "| spot logvol | κ/σ | effectiveness | boot p10–p90 |",
        "|---|---|---|---|",
    ]
    for a in sorted(agg, key=lambda a: (a["spot_logvol"], a["kappa_over_sigma"])):
        k = a["kappa_over_sigma"]
        klabel = "∞" if k > 1e8 else f"{k:g}"
        lines.append(f"| {a['spot_logvol']:g} | {klabel} | "
                     f"{a['effectiveness']:.2f} | "
                     f"{a['eff_p10']:.2f}–{a['eff_p90']:.2f} |")
    lines += ["", "_Auto-generated. Narrative interpretation lives in FINDINGS.md._"]
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=400)
    ap.add_argument("--output", type=Path,
                    default=Path("runs/compute_forward_default"))
    args = ap.parse_args(argv)
    N = args.n_seeds

    agg = _aggregate(N)
    print(f"\n=== Forward default -> retreat to spot, N={N} ===")
    for sigma in SPOT_LOGVOLS:
        print(f"\n  spot logvol={sigma:g}: effectiveness vs κ/σ")
        for a in sorted((a for a in agg if a["spot_logvol"] == sigma),
                        key=lambda a: a["kappa_over_sigma"]):
            k = a["kappa_over_sigma"]
            klabel = "inf" if k > 1e8 else f"{k:g}"
            print(f"    κ/σ={klabel:>4}: E={a['effectiveness']:.2f} "
                  f"[{a['eff_p10']:.2f},{a['eff_p90']:.2f}]")

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "aggregate.csv").open("w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(agg[0].keys()))
        wri.writeheader(); wri.writerows(agg)
    (args.output / "TABLE.md").write_text(_table(agg, N))
    _svg(agg, args.output / "default_curve.svg", N)
    print(f"\nwrote {args.output}/aggregate.csv, TABLE.md")


if __name__ == "__main__":
    main()
