"""Integration test — competing dealers vs basis (bd hka).

Confirms ``scripts.compute_dealer_competition_experiment``:

  * more competing dealers never reduce volume (competition load-balances
    inventory into a tighter aggregate quote),
  * with enough dealers, balanced flow recovers basis-invariance — volume at a
    poor basis rises toward the perfect-basis level (the MC upper bound the
    single-dealer book had lost),
  * one-sided flow still falls with basis even under competition — dealers can
    divide one-sided inventory but cannot net it.

Bypass-Flask by design (CLAUDE.md): imports the kernel, not the app.
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from scripts.compute_dealer_competition_experiment import _trial  # noqa: E402

_N = 24


def _vol(nd, rf, rel):
    return st.mean(_trial(s, nd, rf, rel)["volume_share"] for s in range(_N))


def test_more_dealers_never_reduce_volume():
    for rel in (0.0, 1.0):
        for rf in (0.05, 0.1, 0.2):
            v1, v2, v4 = _vol(1, rf, rel), _vol(2, rf, rel), _vol(4, rf, rel)
            assert v2 >= v1 - 1e-9 and v4 >= v2 - 1e-9, (rel, rf, v1, v2, v4)


def test_competition_restores_balanced_basis_invariance():
    """At a poor basis, balanced flow is far more intermediated with 4 dealers
    than 1, approaching the perfect-basis level."""
    v1 = _vol(1, 0.2, 0.0)
    v4 = _vol(4, 0.2, 0.0)
    assert v4 - v1 > 0.2                          # competition lifts it substantially
    assert v4 > 0.85                              # ~restored toward the MC (rf=0 ~1.0)


def test_one_sided_gate_survives_competition():
    """One-sided flow still degrades with basis even with 4 dealers — nothing to
    net, so competition can only divide the inventory, not cancel it."""
    hi = _vol(4, 0.0, 1.0)
    lo = _vol(4, 0.2, 1.0)
    assert lo < hi - 1e-9                          # basis still bites one-sided
