"""Integration test — emergent participation through the real book (bd 6qu).

Confirms ``scripts.compute_participation_book_experiment`` reproduces the phase-1
dealer-intermediation structure through actual quoting + matching:

  * emergent volume share falls as the basis proxy (risk_factor) rises — the
    dealer's inventory-widened spread prices principals out,
  * one-sided flow (reluctance=1) clears no more than balanced flow at every
    basis (netting helps),
  * at perfect basis (risk_factor=0) balanced flow is (near) fully intermediated,
  * the dealer actually hedges its net SKU inventory onto the basket book.

Bypass-Flask by design (CLAUDE.md): imports the kernel, not the app.
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from scripts.compute_participation_book_experiment import (  # noqa: E402
    RISK_FACTORS, _trial,
)

_N = 24


def _vol(rf, rel):
    return st.mean(_trial(s, rf, rel)["volume_share"] for s in range(_N))


def test_volume_falls_with_basis_proxy_balanced():
    """Higher risk_factor (worse basis) -> the dealer widens -> less volume."""
    vols = [_vol(rf, 0.0) for rf in RISK_FACTORS]
    assert all(a >= b - 1e-9 for a, b in zip(vols, vols[1:])), vols
    assert vols[0] > vols[-1]                     # rf=0 strictly more than rf=0.2


def test_one_sided_clears_no_more_than_balanced():
    """Netting helps: at every basis, one-sided volume <= balanced volume."""
    for rf in RISK_FACTORS:
        assert _vol(rf, 1.0) <= _vol(rf, 0.0) + 1e-9


def test_perfect_basis_balanced_is_liquid():
    assert _vol(0.0, 0.0) > 0.9                   # rf=0 balanced ~ fully cleared


def test_dealer_hedges_net_onto_basket():
    """Across trials the dealer offsets most of its net SKU inventory on the
    basket book (hedge coverage high)."""
    covs = [_trial(s, 0.05, 1.0)["hedge_coverage"] for s in range(_N)]
    assert st.mean(covs) > 0.8
