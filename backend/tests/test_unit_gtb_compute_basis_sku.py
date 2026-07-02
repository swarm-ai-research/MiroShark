"""Unit tests for the multi-SKU basis / diversification model (bd z45, ← umm).

Pins the invariants of ``scripts.compute_basis_sku_experiment``:

  * K=1 reduces *exactly* to the single-SKU headline (same population, same
    clearing) — the multi-SKU generalisation must not perturb the base case,
  * the K idiosyncratic innovations are mutually orthonormal and ⟂ the basket
    (independent-idio assumption realised in-sample), corr(S_0, I)=β,
  * cross-SKU diversification is monotone: in the one-sided limit, more SKUs
    intermediate strictly more willing demand at imperfect basis (the phase-2
    finding — diversification substitutes for netting),
  * β=1 recovers the perfect-hedge endpoint (zero residual, zero spread).

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
    _innovations as _h_innov, _trial as _h_trial,
)
from scripts.compute_basis_sku_experiment import (  # noqa: E402
    BETAS, _innovations_k, _trial,
)

_N = 120
_GD = _GP = 1.0
_NP = 20


def _cell(beta, reluctance, K):
    F, sd, iI, eNs = _innovations_k(_N, K)
    return [_trial(s, beta, reluctance, _GD, _NP, _GP, K, F, sd, iI[s],
                   [eNs[k][s] for k in range(K)]) for s in range(_N)]


def test_k1_reduces_to_single_sku_headline():
    """K=1 must reproduce the headline experiment bit-for-bit (identical draws
    and clearing) — the generalisation is a strict superset."""
    F, sd, iI, eNs = _innovations_k(_N, 1)
    Fh, sdh, iIh, eNh = _h_innov(_N)
    assert F == Fh and sd == sdh
    for beta in (0.2, 0.63, 0.95):
        for rel in (0.0, 1.0):
            for s in range(_N):
                a = _trial(s, beta, rel, _GD, _NP, _GP, 1, F, sd, iI[s],
                           [eNs[0][s]])
                b = _h_trial(s, beta, rel, _GD, _NP, _GP, Fh, sdh, iIh[s],
                             eNh[s])
                assert abs(a["willing_clear_frac"]
                           - b["willing_clear_frac"]) < 1e-12
                assert abs(a["volume_share"] - b["volume_share"]) < 1e-12


def test_idio_innovations_orthonormal():
    """The K idio innovations are mutually orthonormal and ⟂ the basket (so SKU
    idio components are independent in-sample), and corr(S_0, I)=β."""
    K = 6
    _, _, iI, eNs = _innovations_k(_N, K)
    vecs = [iI] + eNs
    for a in range(len(vecs)):
        assert abs(st.pvariance(vecs[a]) - 1.0) < 1e-9
        for b in range(a + 1, len(vecs)):
            dot = sum(x * y for x, y in zip(vecs[a], vecs[b])) / _N
            assert abs(dot) < 1e-9, f"({a},{b}) dot={dot}"
    for beta in BETAS:
        rows = _cell(beta, 0.0, K)
        from scripts.compute_basis_experiment import _corr
        corr = _corr([r["spot_I"] for r in rows], [r["spot_S0"] for r in rows])
        assert abs(corr - beta) < 1e-9


def test_diversification_monotone_one_sided():
    """One-sided limit, imperfect basis: more SKUs → strictly more willing demand
    intermediated (diversification of independent residuals substitutes for
    netting). Checked where the effect is unambiguous (β ≤ 0.63)."""
    for beta in (0.2, 0.45, 0.63):
        wc = {K: st.mean(r["willing_clear_frac"] for r in _cell(beta, 1.0, K))
              for K in (1, 4, 12)}
        assert wc[1] < wc[4] < wc[12], f"β={beta}: {wc}"


def test_beta_one_perfect_hedge_multi_sku():
    for K in (1, 4, 12):
        rows = _cell(1.0, 1.0, K)
        assert all(r["residual_risk"] == 0.0 for r in rows)
        assert all(r["clearing_spread_pct"] == 0.0 for r in rows)
