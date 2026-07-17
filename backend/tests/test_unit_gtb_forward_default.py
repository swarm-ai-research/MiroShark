"""Unit tests for forward-contract default -> retreat to spot (bd 79l).

Pins the mechanics of ``scripts.compute_forward_default_experiment``:

  * a forward whose protection is capped at ±κ has effectiveness monotone
    increasing in κ, → 1 as κ → ∞ (full enforcement = perfect hedge) and → 0 as
    κ → 0 (no enforcement = retreat to spot),
  * κ ≈ 1σ already recovers most of the hedge; κ ≲ 0.5σ loses more than half,
  * at a fixed κ/σ, fatter-tailed prices (higher spot log-vol) are hedged *less*
    effectively — default truncation bites the tail harder.

Bypass-Flask by design (CLAUDE.md): no app imports, runs on pyyaml+pytest.
"""

from __future__ import annotations

import math
import statistics as st
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from scripts.compute_forward_default_experiment import (  # noqa: E402
    _effectiveness, _spot,
)

_N = 400


def _E(sigma, k):
    spots = [_spot(s, sigma) for s in range(_N)]
    F = st.mean(spots)
    devs = [x - F for x in spots]
    vu = st.pvariance(devs)
    return _effectiveness(devs, vu, k, math.sqrt(vu))


def test_effectiveness_monotone_in_enforcement():
    for sigma in (0.3, 0.5, 0.8):
        es = [_E(sigma, k) for k in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)]
        assert all(a <= b + 1e-9 for a, b in zip(es, es[1:])), (sigma, es)


def test_full_enforcement_is_perfect_hedge():
    for sigma in (0.3, 0.5, 0.8):
        assert abs(_E(sigma, 1e9) - 1.0) < 1e-9       # κ=∞ → full variance kill


def test_weak_enforcement_retreats_toward_spot():
    """κ ≲ 0.5σ loses more than half the hedge; κ ≈ 0.1σ is near-worthless."""
    assert _E(0.5, 0.5) < 0.65
    assert _E(0.5, 0.1) < 0.25
    assert _E(0.5, 1.0) > 0.75                         # κ≈1σ still mostly works


def test_fatter_tails_need_more_enforcement():
    """At a fixed κ/σ in the binding region, higher spot log-vol → lower
    effectiveness (default truncation bites the fatter tail harder)."""
    for k in (0.5, 1.0, 2.0):
        es = [_E(sigma, k) for sigma in (0.3, 0.5, 0.8)]
        assert all(a >= b - 1e-9 for a, b in zip(es, es[1:])), (k, es)
        assert es[0] > es[-1] - 1e-9                    # 0.3 vol >= 0.8 vol
