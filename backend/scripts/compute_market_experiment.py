"""Validate the bilateral compute marketplace vs a listed-price baseline (N seeds).

Ports arkhai ``simple-compute-market``'s discover→negotiate→settle microstructure
into GTB (``worlds.gather_trade_build.compute_market``) and asks the one question
that microstructure choice actually decides: **when a buyer and seller are drawn
with private valuations, does buyer-driven bisection negotiation clear more of
the feasible trades — and split the surplus more evenly — than a take-it-or-leave-
it listed price?**

Two mechanisms, same drawn population, per seed:

  A. **BISECTION** — ``compute_market.negotiate``: buyer-driven, bounded-round,
     each side bisects toward the peer's quote; deal lands inside the bargaining
     range ``[floor, reservation]`` when one exists.
  B. **LISTED-PRICE** — take-it-or-leave-it at the seller's advertised ask: the
     buyer accepts iff ``ask <= reservation``. (Upstream's ``listed_price``
     buyer policy with no counter.)

Per encounter, ``seller_floor`` and ``buyer_reservation`` are deterministic,
reproducible per-seed log-normal draws around a base H100 rate (right-skewed, à
la GPU-rental prices); the seller advertises ``floor * markup``. No RNG (repo
convention) — a quasi-normal ``z`` is assembled from decorrelated seed hashes, so
the whole sweep reproduces byte-for-byte.

Metrics per seed (over ``ENCOUNTERS`` draws): fill rate, mean realized rate, mean
buyer surplus share, and welfare capture (realized ÷ feasible gains-from-trade).
Aggregated across seeds with mean/median/p10/p90; confidence bands (p10–p90)
rendered to SVG. FINDINGS.md is hand-curated separately (CLAUDE.md methodology).

Usage::

    uv run python -m scripts.compute_market_experiment --n-seeds 100

Writes encounters.csv, aggregate.csv, TABLE.md, and surplus_bands.svg to the
output dir (default ``runs/compute_market``).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import math
import statistics as st
from pathlib import Path
from typing import Dict, List

logging.disable(logging.CRITICAL)

from worlds.gather_trade_build.compute_market import negotiate  # noqa: E402

BASE = 3.50        # base H100 rate (coin/compute-hour); valuation median
SIGMA = 0.45       # log-vol of buyer/seller valuations
MARKUP = 1.30      # seller advertises floor * MARKUP
OPEN_FRAC = 0.70   # buyer opens at OPEN_FRAC * reservation
ENCOUNTERS = 400   # buyer/seller draws per seed
MAX_ROUNDS = 5
GAP_RATIO = 1.5


def _uniform(seed: int, salt: int, k: int) -> float:
    """A single reproducible uniform in [0, 1) from a SHA-256 of the coordinates.

    Deterministic (no ``random`` — repo convention: reproducible byte-for-byte),
    but a real cryptographic hash, so the draws are well-spread rather than
    collapsing onto a tiny cyclic residue set the way a raw ``(k*m) % n`` LCG
    does."""
    h = hashlib.sha256(f"{seed}:{salt}:{k}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def _lognormal(seed: int, salt: int) -> float:
    """Deterministic, reproducible log-normal draw around ``BASE``.

    A quasi-normal ``z`` via Irwin–Hall (sum of 12 hash-derived uniforms minus
    6 ≈ N(0, 1)), mapped through exp for a right-skewed valuation (à la GPU-
    rental prices). ``salt`` decorrelates the buyer and seller draws within one
    encounter."""
    z = sum(_uniform(seed, salt, k) for k in range(12)) - 6.0
    return BASE * math.exp(SIGMA * z)


def _encounter(seed: int, i: int) -> Dict[str, float]:
    """One buyer/seller encounter under both mechanisms."""
    seller_floor = _lognormal(seed, 2 * i)
    buyer_res = _lognormal(seed, 2 * i + 1)
    ask = seller_floor * MARKUP
    bid0 = OPEN_FRAC * buyer_res
    feasible = seller_floor <= buyer_res
    surplus = max(0.0, buyer_res - seller_floor)  # gains from trade if cleared

    # A. bisection negotiation
    res = negotiate(seller_floor=seller_floor, buyer_reservation=buyer_res,
                    opening_ask=ask, opening_bid=bid0,
                    max_rounds=MAX_ROUNDS, gap_ratio=GAP_RATIO)
    if res.is_deal:
        r = res.agreed_rate
        b_neg = {"deal": 1.0, "rate": r,
                 "buyer_share": (buyer_res - r) / surplus if surplus > 1e-9 else 0.0,
                 "welfare": surplus}
    else:
        b_neg = {"deal": 0.0, "rate": 0.0, "buyer_share": 0.0, "welfare": 0.0}

    # B. listed-price take-it-or-leave-it at the advertised ask
    if ask <= buyer_res:
        b_list = {"deal": 1.0, "rate": ask,
                  "buyer_share": (buyer_res - ask) / surplus if surplus > 1e-9 else 0.0,
                  "welfare": surplus}
    else:
        b_list = {"deal": 0.0, "rate": 0.0, "buyer_share": 0.0, "welfare": 0.0}

    return {"seed": seed, "i": i, "feasible": 1.0 if feasible else 0.0,
            "surplus": surplus,
            "neg_deal": b_neg["deal"], "neg_rate": b_neg["rate"],
            "neg_share": b_neg["buyer_share"], "neg_welfare": b_neg["welfare"],
            "list_deal": b_list["deal"], "list_rate": b_list["rate"],
            "list_share": b_list["buyer_share"], "list_welfare": b_list["welfare"]}


def _seed_row(seed: int) -> Dict[str, float]:
    """Aggregate one seed's ENCOUNTERS into the per-seed metrics."""
    rows = [_encounter(seed, i) for i in range(ENCOUNTERS)]
    feasible_surplus = sum(r["surplus"] for r in rows if r["feasible"])
    n = len(rows)

    def _fill(mech):
        return sum(r[f"{mech}_deal"] for r in rows) / n

    def _mean_rate(mech):
        rates = [r[f"{mech}_rate"] for r in rows if r[f"{mech}_deal"]]
        return st.mean(rates) if rates else 0.0

    def _mean_share(mech):
        shares = [r[f"{mech}_share"] for r in rows if r[f"{mech}_deal"]]
        return st.mean(shares) if shares else 0.0

    def _welfare(mech):
        got = sum(r[f"{mech}_welfare"] for r in rows)
        return got / feasible_surplus if feasible_surplus > 1e-9 else 0.0

    return {
        "seed": seed,
        "neg_fill": _fill("neg"), "list_fill": _fill("list"),
        "neg_rate": _mean_rate("neg"), "list_rate": _mean_rate("list"),
        "neg_share": _mean_share("neg"), "list_share": _mean_share("list"),
        "neg_welfare": _welfare("neg"), "list_welfare": _welfare("list"),
    }


def _agg(rows: List[Dict[str, float]], key: str) -> Dict[str, float]:
    xs = sorted(r[key] for r in rows)
    p = st.quantiles(xs, n=10) if len(xs) >= 2 else [xs[0]] * 9
    return {"mean": st.mean(xs), "median": st.median(xs),
            "p10": p[0], "p90": p[8]}


# ------------------------------------------------------------------ SVG
def _svg_bands(agg: Dict[str, Dict[str, float]], path: Path) -> None:
    """Grouped bar chart: bisection vs listed-price on the four metrics, each
    bar annotated with its p10–p90 whisker (confidence band)."""
    metrics = [("fill", "fill rate"), ("share", "buyer surplus share"),
               ("welfare", "welfare capture")]
    width, height = 640, 360
    pad_l, pad_b, pad_t = 50, 60, 40
    plot_h = height - pad_b - pad_t
    plot_w = width - pad_l - 30
    group_w = plot_w / len(metrics)
    bar_w = group_w * 0.28

    def ymap(v):  # metrics all in [0, 1]
        return pad_t + plot_h * (1.0 - max(0.0, min(1.0, v)))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="sans-serif" font-size="11">',
        '<style>.neg{fill:#3aa55c}.list{fill:#7ab8ff}.wh{stroke:#333;stroke-width:1}'
        '.ax{stroke:#999;stroke-width:1}.lbl{fill:#333}</style>',
        f'<text x="{width/2:.0f}" y="20" text-anchor="middle" font-size="14">'
        'Bisection negotiation vs listed-price (p10–p90 whiskers, N seeds)</text>',
    ]
    # y gridlines
    for gv in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = ymap(gv)
        parts.append(f'<line class="ax" x1="{pad_l}" y1="{y:.1f}" x2="{width-30}" y2="{y:.1f}" opacity="0.3"/>')
        parts.append(f'<text class="lbl" x="{pad_l-6}" y="{y+3:.1f}" text-anchor="end">{gv:.2f}</text>')

    for gi, (mkey, mlabel) in enumerate(metrics):
        gx = pad_l + gi * group_w + group_w / 2
        for bi, mech in enumerate(("neg", "list")):
            a = agg[f"{mech}_{mkey}"]
            cx = gx + (bi - 0.5) * (bar_w + 6)
            y = ymap(a["mean"])
            parts.append(f'<rect class="{mech}" x="{cx-bar_w/2:.1f}" y="{y:.1f}" '
                         f'width="{bar_w:.1f}" height="{ymap(0.0)-y:.1f}"/>')
            # p10–p90 whisker
            y10, y90 = ymap(a["p10"]), ymap(a["p90"])
            parts.append(f'<line class="wh" x1="{cx:.1f}" y1="{y90:.1f}" x2="{cx:.1f}" y2="{y10:.1f}"/>')
            parts.append(f'<line class="wh" x1="{cx-4:.1f}" y1="{y90:.1f}" x2="{cx+4:.1f}" y2="{y90:.1f}"/>')
            parts.append(f'<line class="wh" x1="{cx-4:.1f}" y1="{y10:.1f}" x2="{cx+4:.1f}" y2="{y10:.1f}"/>')
        parts.append(f'<text class="lbl" x="{gx:.0f}" y="{height-pad_b+18:.0f}" '
                     f'text-anchor="middle">{mlabel}</text>')
    # legend
    parts.append(f'<rect class="neg" x="{pad_l}" y="{height-24}" width="12" height="12"/>')
    parts.append(f'<text class="lbl" x="{pad_l+16}" y="{height-14}">bisection</text>')
    parts.append(f'<rect class="list" x="{pad_l+90}" y="{height-24}" width="12" height="12"/>')
    parts.append(f'<text class="lbl" x="{pad_l+106}" y="{height-14}">listed-price</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts))


def _table_md(agg: Dict[str, Dict[str, float]], n_seeds: int, svg: str) -> str:
    def fmt(k):
        a = agg[k]
        return f"{a['mean']:.3f} | {a['median']:.3f} | {a['p10']:.3f} | {a['p90']:.3f}"
    lines = [
        f"# Compute marketplace: bisection vs listed-price (N={n_seeds} seeds, "
        f"{ENCOUNTERS} encounters/seed)",
        "",
        "Metric (mean \\| median \\| p10 \\| p90) | bisection | listed-price",
        "--- | --- | ---",
        f"fill rate | {fmt('neg_fill')} | {fmt('list_fill')}",
        f"mean realized rate (coin/hr) | {fmt('neg_rate')} | {fmt('list_rate')}",
        f"buyer surplus share | {fmt('neg_share')} | {fmt('list_share')}",
        f"welfare capture | {fmt('neg_welfare')} | {fmt('list_welfare')}",
        "",
        f"![bisection vs listed-price]({svg})",
        "",
        "_Auto-generated. Narrative interpretation lives in FINDINGS.md._",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=100)
    ap.add_argument("--output", type=Path, default=Path("runs/compute_market"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    seed_rows = [_seed_row(s) for s in range(args.n_seeds)]

    with (args.output / "encounters.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(seed_rows[0].keys()))
        w.writeheader()
        w.writerows(seed_rows)

    metrics = ["neg_fill", "list_fill", "neg_rate", "list_rate",
               "neg_share", "list_share", "neg_welfare", "list_welfare"]
    agg = {m: _agg(seed_rows, m) for m in metrics}

    with (args.output / "aggregate.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "median", "p10", "p90"])
        for m in metrics:
            a = agg[m]
            w.writerow([m, a["mean"], a["median"], a["p10"], a["p90"]])

    _svg_bands(agg, args.output / "surplus_bands.svg")
    (args.output / "TABLE.md").write_text(
        _table_md(agg, args.n_seeds, "surplus_bands.svg"))

    print(f"[compute_market] N={args.n_seeds} seeds x {ENCOUNTERS} encounters")
    for m in metrics:
        a = agg[m]
        print(f"  {m:14s} mean={a['mean']:.3f} median={a['median']:.3f} "
              f"p10={a['p10']:.3f} p90={a['p90']:.3f}")
    print(f"  wrote {args.output}/TABLE.md, aggregate.csv, surplus_bands.svg")


if __name__ == "__main__":
    main()
