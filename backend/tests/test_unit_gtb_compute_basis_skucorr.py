"""Unit tests for the cross-SKU correlation model (bd 0h4, ← z45 ← umm).

Pins the invariants of ``scripts.compute_basis_skucorr_experiment``:

  * the correlation limits are exact — ρ=0 clears identically to the phase-2
    independent-idio model, and ρ=1 clears identically to the single-SKU
    (K=1) model (the correlated residual R_eff interpolates them),
  * {iI, c, o_1..o_K} are orthonormal (basis + common + own idio factors),
  * more cross-SKU correlation monotonically shrinks the diversification
    benefit (willing-clear non-increasing in ρ at fixed imperfect basis),
  * β=1 recovers the perfect-hedge endpoint.

Bypass-Flask by design (CLAUDE.md): no app imports, runs on pyyaml+pytest.
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from scripts.compute_basis_sku_experiment import (  # noqa: E402
    _clear as _sku_clear, _population,
)
from scripts.compute_basis_skucorr_experiment import (  # noqa: E402
    BETAS, RHO_SKU, _clear, _innovations_corr, _trial,
)

_N = 100
_GD = _GP = 1.0
_NP = 20


def _gross(part):
    return sum(p.q for p in part)


def test_rho0_matches_independent_phase2():
    """ρ=0 clears identically to the phase-2 independent-idio model (R_eff=ΣD_k²)."""
    for seed in range(40):
        pop = _population(seed, _NP, _GP, 12)
        for beta in (0.2, 0.63, 0.95):
            a, _, _ = _clear(pop, beta, 0.0, _GD, 12)
            b, _, _ = _sku_clear(pop, beta, 1.0, _GD, 12)   # reluctance=1
            assert abs(_gross(a) - _gross(b)) < 1e-12


def test_rho1_matches_single_sku():
    """ρ=1 (fully correlated) clears identically to a single concentrated SKU —
    R_eff=(ΣD_k)² regardless of the SKU split."""
    for seed in range(40):
        pop12 = _population(seed, _NP, _GP, 12)
        pop1 = _population(seed, _NP, _GP, 1)               # same draws, K=1
        for beta in (0.2, 0.63, 0.95):
            a, _, _ = _clear(pop12, beta, 1.0, _GD, 12)
            b, _, _ = _sku_clear(pop1, beta, 1.0, _GD, 1)
            assert abs(_gross(a) - _gross(b)) < 1e-12


def test_innovation_factors_orthonormal():
    K = 8
    _, _, iI, c, os = _innovations_corr(_N, K)
    vecs = [iI, c] + os
    for i in range(len(vecs)):
        assert abs(st.pvariance(vecs[i]) - 1.0) < 1e-9
        for j in range(i + 1, len(vecs)):
            dot = sum(x * y for x, y in zip(vecs[i], vecs[j])) / _N
            assert abs(dot) < 1e-9, f"({i},{j}) dot={dot}"


def _cell_wc(beta, rho, K):
    F, sd, iI, c, os = _innovations_corr(_N, K)
    rows = [_trial(s, beta, rho, _GD, _NP, _GP, K, F, sd, iI[s], c[s],
                   [os[k][s] for k in range(K)]) for s in range(_N)]
    return st.mean(r["willing_clear_frac"] for r in rows)


def test_correlation_erodes_diversification_monotone():
    """At fixed imperfect basis, willing-clear is non-increasing in ρ — more
    correlation means less diversification, so a smaller one-sided market."""
    for beta in (0.45, 0.63, 0.8):
        wc = [_cell_wc(beta, rho, 12) for rho in RHO_SKU]
        assert all(a >= b - 1e-9 for a, b in zip(wc, wc[1:])), f"β={beta}: {wc}"


def test_beta_one_perfect_hedge():
    F, sd, iI, c, os = _innovations_corr(_N, 12)
    for rho in RHO_SKU:
        rows = [_trial(s, 1.0, rho, _GD, _NP, _GP, 12, F, sd, iI[s], c[s],
                       [os[k][s] for k in range(12)]) for s in range(_N)]
        assert all(r["residual_risk"] == 0.0 for r in rows)
        assert all(r["clearing_spread_pct"] == 0.0 for r in rows)
