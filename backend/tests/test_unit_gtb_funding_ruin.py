"""Integration test — dealer funding-ruin under asymmetric margining (bd b2s).

Confirms ``scripts.compute_funding_ruin_experiment`` exhibits the Pirrong
funding-liquidity mechanism through the real bd-1z3 MTM engine:

  * at β=1 the dealer is economically flat at expiry, so with an ample coin
    buffer there is zero ruin and (essentially) zero P&L variance,
  * a thin buffer causes real ruin even at β=1 — pure funding risk, no terminal
    price risk — injecting P&L dispersion into a flat position,
  * ruin frequency is monotone non-increasing in the coin buffer.

Bypass-Flask by design (CLAUDE.md): imports the kernel, not the app.
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from scripts.compute_funding_ruin_experiment import BUFFER_FRACS, _trial  # noqa: E402

_N = 80


def _cell(beta, buffer_frac):
    rows = [_trial(s, beta, buffer_frac) for s in range(_N)]
    return (st.mean(r["ruined"] for r in rows),
            st.pstdev(r["net_pnl"] for r in rows))


def test_beta1_ample_buffer_is_flat_and_ruin_free():
    """β=1 with a fat buffer: economically flat → no ruin, ~zero P&L variance."""
    ruin, net_std = _cell(1.0, 1.60)
    assert ruin == 0.0
    assert net_std < 1e-6                    # perfectly hedged, no funding stress


def test_beta1_thin_buffer_causes_pure_funding_ruin():
    """β=1 (zero terminal price risk) with a thin buffer still ruins the dealer
    and injects P&L dispersion — funding risk with no price risk."""
    ruin, net_std = _cell(1.0, 0.05)
    assert ruin > 0.1                        # real ruin despite being hedged
    assert net_std > 1e-3                    # dispersion where a flat book has none


def test_ruin_monotone_in_buffer():
    """More coin buffer → weakly less ruin (β=1)."""
    ruins = [_cell(1.0, bf)[0] for bf in BUFFER_FRACS]
    assert all(a >= b - 1e-9 for a, b in zip(ruins, ruins[1:])), ruins
    assert ruins[0] > ruins[-1]             # thin buffer strictly riskier than fat
