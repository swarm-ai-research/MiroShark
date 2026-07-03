"""Integration test — basis-hedge through the real order book (bd er8, ← umm §7).

Confirms ``scripts.compute_basis_kernel_experiment`` reproduces the analytic
dealer basis residual through the *real* futures matching + settlement engine
(SKU-keyed book from increment 1 + per-SKU spot settlement):

  * each seed mints and settles exactly two contracts (a SKU forward + a basket
    hedge),
  * the std of the dealer's realised net P&L equals the analytic
    Q·√((1−r_i)·V) to tight tolerance,
  * at β=1 the hedge is exact — zero residual.

Bypass-Flask by design (CLAUDE.md): imports the kernel (not the app).
"""

from __future__ import annotations

import math
import statistics as st
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from scripts.compute_basis_experiment import _innovations  # noqa: E402
from scripts.compute_basis_kernel_experiment import Q, _trial  # noqa: E402

_N = 80


def _residual(beta):
    F, sd_i, iI, eN = _innovations(_N)
    rows = [_trial(s, beta, F, sd_i, iI[s], eN[s]) for s in range(_N)]
    realized = st.pstdev(r["dealer_pnl"] for r in rows)
    analytic = Q * math.sqrt((1.0 - beta * beta) * sd_i * sd_i)
    return realized, analytic, rows


def test_two_contracts_mint_and_settle_per_seed():
    _, _, rows = _residual(0.63)
    assert all(r["n_contracts"] == 2 for r in rows)      # SKU forward + basket hedge
    assert all(r["n_settled"] == 2 for r in rows)


def test_realized_residual_matches_analytic_through_book():
    """Real matching + settlement reproduce Q·√((1−r_i)·V) within 3%."""
    for beta in (0.20, 0.45, 0.63, 0.80, 0.95):
        realized, analytic, _ = _residual(beta)
        assert abs(realized - analytic) / analytic < 0.03, \
            f"β={beta}: realized={realized} analytic={analytic}"


def test_beta_one_perfect_hedge_zero_residual():
    realized, analytic, rows = _residual(1.0)
    assert analytic == 0.0
    assert realized < 1e-6                                # dealer fully hedged
    assert all(abs(r["dealer_pnl"]) < 1e-6 for r in rows)
