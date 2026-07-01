"""Unit tests for the compute basis-risk / dealer-intermediation model (bd umm).

Pins the load-bearing invariants of ``scripts.compute_basis_experiment`` — the
Option A headline sweep behind ``docs/plans/2026-06-30-compute-basis-dealer-
experiment.md``:

  * the synthetic factor model realises corr(S_i, I) = β *exactly* in-sample
    (the Gram-Schmidt'd orthonormal innovations — the swept basis must be
    noise-free or the x-axis is contaminated),
  * β=1 recovers the bd k9w perfect-hedge endpoint (zero residual, zero spread,
    full participation),
  * the risk-averse dealer earns a non-negative expected spread (it is always
    compensated for the residual it warehouses),
  * short-side reluctance=1 withdraws every short (the design §4 one-sided limit),
  * the one-sided willing-clear fraction is monotone non-decreasing in β (the
    phase transition), and netting makes the balanced market strictly larger
    than the one-sided market at imperfect basis.

Bypass-Flask by design (CLAUDE.md): no app imports, runs on pyyaml+pytest.
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from scripts.compute_basis_experiment import (  # noqa: E402
    BETAS, RELUCTANCE, _clear, _corr, _innovations, _population, _trial,
)

_N = 200          # ensemble for the factor-model / dealer checks
_GD = _GP = 1.0
_NP = 20


def _cell(beta: float, reluctance: float):
    """All per-seed rows for one (β, reluctance) cell over the _N ensemble."""
    F, sd_i, iI, eN = _innovations(_N)
    return [_trial(s, beta, reluctance, _GD, _NP, _GP, F, sd_i, iI[s], eN[s])
            for s in range(_N)]


def test_factor_model_realises_exact_corr():
    """corr(S_i, I) == β to numerical tolerance for every swept β (in-sample
    orthonormalisation removes finite-sample basis noise)."""
    for beta in BETAS:
        rows = _cell(beta, 0.0)
        corr = _corr([r["spot_I"] for r in rows], [r["spot_S"] for r in rows])
        assert abs(corr - beta) < 1e-9, f"β={beta}: corr={corr}"


def test_beta_one_recovers_perfect_hedge_endpoint():
    """β=1 (r_i=1): the dealer's residual and spread vanish and everyone who is
    willing clears — the bd k9w/5ij endpoint this sweep descends from."""
    for reluctance in RELUCTANCE:
        rows = _cell(1.0, reluctance)
        assert all(r["residual_risk"] == 0.0 for r in rows)
        assert all(r["clearing_spread_pct"] == 0.0 for r in rows)
        assert all(abs(r["dealer_pnl"]) < 1e-9 for r in rows)
        assert st.mean(r["willing_clear_frac"] for r in rows) == 1.0


def test_dealer_earns_nonnegative_expected_spread():
    """Across seeds the risk-averse dealer's mean P&L ≥ 0 — it is compensated
    for warehousing the residual basis (never runs the book at a loss in
    expectation)."""
    for beta in BETAS:
        for reluctance in RELUCTANCE:
            rows = _cell(beta, reluctance)
            assert st.mean(r["dealer_pnl"] for r in rows) >= -1e-9


def test_reluctance_one_withdraws_all_shorts():
    """reluctance=1.0 is the one-sided limit: no short principal is intermediated
    (the dealer sees long-only flow, design §4)."""
    for seed in range(50):
        pop = _population(seed, _NP, _GP)
        part, _, _ = _clear(pop, 0.8, 1.0, _GD)
        assert all(p.side == 1 for p in part)


def test_one_sided_transition_is_monotone_in_beta():
    """In the one-sided limit the willing-clear fraction (pure basis effect) is
    non-decreasing in β — the phase transition has the right sign."""
    means = []
    for beta in BETAS:
        rows = _cell(beta, 1.0)
        means.append(st.mean(r["willing_clear_frac"] for r in rows))
    assert all(b >= a - 1e-9 for a, b in zip(means, means[1:])), means


def test_netting_beats_one_sided_at_imperfect_basis():
    """The core qualitative claim: with two-sided flow the dealer nets longs
    against shorts, so a balanced market intermediates strictly more than a
    one-sided one whenever basis is imperfect (β<1)."""
    for beta in (b for b in BETAS if b < 1.0):
        bal = st.mean(r["volume_share"] for r in _cell(beta, 0.0))
        one = st.mean(r["volume_share"] for r in _cell(beta, 1.0))
        assert bal > one, f"β={beta}: balanced={bal} !> one-sided={one}"
