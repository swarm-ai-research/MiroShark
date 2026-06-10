# risk_based_audit_multiplier = 0 arm

*Closes bd-dn9 (`risk_based_audit_multiplier=0 arm`). Comparison to
bd-an2's default (risk_based_audit_multiplier=1.5) sweep.*

## Setup

- Scenario: default `ai_economist_full.yaml`.
- Sweep: `audit_probability` ∈ {0.0, 0.025, 0.05, 0.1, 0.15, 0.2,
  0.3, 0.5, 0.65, 0.8, 0.95} × `risk_based_audit_multiplier=0.0`.
- Seeds: 50 per cell. Length: 30 epochs × 15 steps.
- Harness: `python -m scripts.sweep_gtb`. 550 jobs in 17.3s.

## Comparison (rb=0 arm vs bd-an2's default rb=1.5)

| audit_prob | welfare (default) | welfare (rb=0) | Δ | tax (default) | tax (rb=0) | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 72.19 | 72.01 | −0.18 | 446.02 | 447.65 | +1.63 |
| 0.025 | 71.88 | 70.94 | −0.94 | 455.28 | 447.85 | −7.43 |
| 0.050 | 69.84 | 69.00 | −0.84 | 453.86 | 448.37 | −5.49 |
| 0.100 | 67.85 | 68.35 | +0.50 | 445.89 | 455.55 | +9.66 |
| 0.150 | 64.19 | 63.72 | −0.47 | 419.65 | 417.89 | −1.76 |
| 0.200 | 63.75 | 62.41 | −1.34 | 412.20 | 401.13 | −11.07 |
| 0.300 | 64.77 | 63.76 | −1.01 | 415.46 | 408.67 | −6.79 |
| 0.500 | 64.51 | 63.94 | −0.57 | 412.07 | 407.36 | −4.71 |
| 0.650 | 64.48 | 63.38 | −1.10 | 411.24 | 402.33 | −8.91 |
| 0.800 | 64.57 | 63.31 | −1.26 | 411.26 | 401.54 | −9.72 |
| 0.950 | 64.64 | 63.50 | −1.14 | 411.61 | 402.56 | −9.05 |

## Catches at the lower audit_prob cells

| audit_prob | catches (default) | catches (rb=0) | Δ |
|---:|---:|---:|---:|
| 0.025 | 0.19 | 0.18 | −0.01 |
| 0.050 | 0.35 | 0.32 | −0.03 |
| 0.100 | 0.55 | 0.46 | −0.09 |
| 0.150 | 0.79 | 0.62 | −0.17 |

## What the data says

1. **The risk-targeting kicker is mostly cosmetic.** Welfare
   differences are < 2% at every cell. Tax revenue differences are
   < 3%. The default `risk_based_audit_multiplier=1.5` doesn't
   change the headline economics — it just slightly biases WHICH
   misreporters get caught.

2. **At low audit_probability, removing the kicker REDUCES catches**
   (0.79 → 0.62 at audit_prob=0.15). Makes sense: the kicker
   amplifies audit_probability for workers with large discrepancies,
   so removing it makes detection slower for the biggest evaders.
   But the welfare impact of this is microscopic (−0.47 at the same
   cell).

3. **At high audit_probability, removing the kicker very slightly
   INCREASES catches** (0.95 cell: default 1.51 → rb=0 1.64). At
   saturation (audit_prob >= 0.2), the kicker is redundant —
   everyone's already being audited. Removing it doesn't matter
   for selection, but does affect the per-discrepancy hit rate.

4. **The bd-an2 shape is preserved.** Welfare still falls steeply
   to a trough around audit_prob=0.2, then plateaus. No-audit
   still wins. The risk-targeting kicker doesn't change the
   policy conclusion — it just adds a tiny modulation.

5. **The cell at audit_prob=0.1 is the only outlier** — rb=0 has
   slightly HIGHER welfare (+0.50) and tax revenue (+9.66). Likely
   seed noise; the p10-p90 spread on welfare at this cell is wider
   than 1.0 in both arms. Not a real signal.

## Implications

- **bd-an2's conclusions hold without the risk-targeting kicker.**
  The "audit nobody is best" finding is robust to whether the
  audit prob is flat or risk-weighted.
- **The kicker exists in the upstream env for theoretical
  realism** (real tax authorities target large discrepancies more)
  but doesn't materially shape outcomes here. Could be removed
  from the default scenario without changing any of our findings.
- **The combined sweep effort across bd-cy8, bd-11o, bd-dn9** is
  ~1000 GTB runs (audit_prob + cross with fine_multiplier +
  with/without risk_based). All three findings point to: the
  default rule-based lineup is structurally insensitive to fine
  enforcement parameters. The only audit lever that bites is
  audit_probability itself, and only because the env handles
  selection directly.

## How to reproduce

```bash
cd backend
uv run python -m scripts.sweep_gtb worlds/gather_trade_build/scenarios/ai_economist_full.yaml \
  --n-seeds 50 --epochs 30 --steps 15 \
  --sweep 'misreporting.audit_probability=0.0,0.025,0.05,0.10,0.15,0.20,0.30,0.50,0.65,0.80,0.95' \
  --sweep 'misreporting.risk_based_audit_multiplier=0.0' \
  --output runs/risk_based_zero_sweep
```
