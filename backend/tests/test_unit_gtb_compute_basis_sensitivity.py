"""Unit tests for the critical-β sensitivity law (bd keq, ← umm).

Pins ``scripts.compute_basis_sensitivity_experiment``:

  * the simulated critical basis tracks the analytic β*≈√(1−4·(γ_P/γ_D)/N_P)
    across the surface (validates the scaling that governs where the one-sided
    transition sits),
  * critical β rises with N_P (one-sided risk concentration) and falls with
    γ_P/γ_D (dealer risk tolerance) — the two comparative statics.

Bypass-Flask by design (CLAUDE.md): no app imports, runs on pyyaml+pytest.
"""

from __future__ import annotations

import statistics as st
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from scripts.compute_basis_sensitivity_experiment import (  # noqa: E402
    BETA_GRID, _analytic_crit_beta, _crit_beta, _willing_clear,
)

_N = 80
_GD = 1.0


def _crit(n_p: int, ratio: float) -> float:
    wc = [st.mean(_willing_clear(s, b, n_p, ratio * _GD, _GD) for s in range(_N))
          for b in BETA_GRID]
    return _crit_beta(BETA_GRID, wc)


def test_empirical_tracks_analytic():
    """Where the transition sits inside the β-grid (analytic β* in [0.3,0.92]),
    the simulated critical β matches the closed form within 0.06."""
    for n_p in (20, 40):
        for ratio in (1.0, 2.0, 3.0, 4.0):
            ana = _analytic_crit_beta(ratio, n_p)
            if not (0.3 <= ana <= 0.92):
                continue
            emp = _crit(n_p, ratio)
            assert abs(emp - ana) < 0.06, f"N_P={n_p} ρ={ratio}: {emp} vs {ana}"


def test_crit_beta_rises_with_market_size():
    """Fixed γ-ratio: larger N_P concentrates one-sided risk → higher critical β."""
    for ratio in (2.0, 3.0):
        crits = [_crit(n_p, ratio) for n_p in (10, 20, 40)]
        assert all(a <= b + 1e-9 for a, b in zip(crits, crits[1:])), crits


def test_crit_beta_falls_with_dealer_risk_tolerance():
    """Fixed N_P: higher γ_P/γ_D (more risk-tolerant dealer) → lower critical β."""
    for n_p in (20, 40):
        crits = [_crit(n_p, r) for r in (1.0, 2.0, 3.0, 4.0)]
        assert all(a >= b - 1e-9 for a, b in zip(crits, crits[1:])), crits
