# fine_multiplier × audit_probability sweep

*Closes bd-11o (`fine_multiplier sweep`).*

## Setup

- Scenario: default `ai_economist_full.yaml`.
- Sweep: `misreporting.fine_multiplier` ∈ {0.5, 1.0, 2.0, 4.0} ×
  `misreporting.audit_probability` ∈ {0.05, 0.20, 0.50, 0.80}.
- Seeds: 50 per cell. Length: 30 epochs × 15 steps.
- Harness: `python -m scripts.sweep_gtb`. 800 jobs in 23.4s.

## Final-epoch results

| fine_multiplier | audit_prob | welfare | tax | catches | Gini |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.05 | 68.55 | 446.39 | 0.34 | 0.521 |
| 0.5 | 0.20 | 62.74 | 403.73 | 1.14 | 0.544 |
| 0.5 | 0.50 | 63.40 | 402.71 | 1.60 | 0.541 |
| 0.5 | 0.80 | 63.43 | 401.77 | 1.66 | 0.542 |
| 1.0 | 0.05 | 68.55 | 446.39 | 0.34 | 0.521 |
| 1.0 | 0.20 | 62.74 | 403.71 | 1.14 | 0.544 |
| 1.0 | 0.50 | 63.40 | 402.68 | 1.60 | 0.541 |
| 1.0 | 0.80 | 63.43 | 401.74 | 1.66 | 0.542 |
| 2.0 | 0.05 | 68.55 | 446.39 | 0.34 | 0.521 |
| 2.0 | 0.20 | 62.74 | 403.70 | 1.14 | 0.544 |
| 2.0 | 0.50 | 63.40 | 402.65 | 1.60 | 0.541 |
| 2.0 | 0.80 | 63.43 | 401.71 | 1.66 | 0.542 |
| 4.0 | 0.05 | 68.55 | 446.37 | 0.34 | 0.521 |
| 4.0 | 0.20 | 62.74 | 403.69 | 1.14 | 0.544 |
| 4.0 | 0.50 | 63.40 | 402.62 | 1.60 | 0.541 |
| 4.0 | 0.80 | 63.43 | 401.68 | 1.66 | 0.542 |

## What the data says

1. **`fine_multiplier` is vacuous under the default scenario.**
   Welfare, Gini, and catches are IDENTICAL across the 4× range. Tax
   revenue differs by 0.02-0.10 coins (rounding noise on 400+).
   Going from 0.5× the default fine to 4× — an 8× swing — produces
   no measurable effect on any outcome metric.

2. **The reason: rule-based workers don't read fine_multiplier.**
   `EvasiveWorkerPolicy` underreports a fixed `underreport_fraction`
   per epoch regardless of the fine that would be levied if caught.
   Heavier fines only affect *paid* fine amounts, and only on
   audit-and-catch paths.

3. **And fine payment is bounded by worker coin.** Evasive workers
   tend to have low coin reserves (they spend gather time
   misreporting, not gathering). When caught, the fine is capped at
   coin_balance — so a 4× fine on a worker with 1 coin still only
   takes 1 coin. The multiplier has no bite.

4. **This compounds with the bd-an2 + bd-cy8 findings.** The
   "enforcement extracts from workers without redistribution"
   mechanism that drives welfare loss in bd-an2 *doesn't even care
   about the fine size* — the welfare cost is the time/coin spent
   being audited, not the fine paid. Cutting `fine_multiplier` to
   0.5 doesn't recover any welfare.

5. **The audit_probability axis still moves things** (welfare 68.55
   at 0.05 → 62.74 at 0.20). That's the bd-an2 finding reproduced
   at every fine_multiplier level. Confirms the audit_prob effect
   is independent of fine_multiplier.

## Implications

- **`fine_multiplier` joins the list of vacuous policy levers
  under the default rule-based lineup**, alongside the bd-coq
  finding that planner type / learning rate don't matter. The
  pattern is consistent: the default rule-based workers don't read
  most policy signals.
- **bd-2e2's TaxAwareHonestPolicy fix probably applies here too.**
  An "AuditAwareEvasivePolicy" that reduces `underreport_fraction`
  when `fine_multiplier × audit_probability` is high would
  reactivate this lever. Likely the cheapest follow-up.
- **The compute spent on this sweep was 23.4s.** Confirming a
  vacuous lever is fast; the value is in knowing not to design
  experiments around it.

## Sibling questions worth filing

- **AuditAwareEvasivePolicy.** A rule-based evasive worker that
  scales `underreport_fraction` by `1 / max(1, audit_probability ×
  fine_multiplier)`. With this, fine_multiplier should reactivate as
  a behavioural lever. Mirrors the bd-2e2 → bd-coq pattern.
- **Adversarial bd-11o on LLM-driven workers.** Run the same sweep
  with `BALANCED_LLM_LINEUP` or `STAKING_LINEUP`. LLM evaders SHOULD
  read the fine schedule from obs and react. Tests whether LLM
  workers exhibit the rationality the rule-based ones lack.

## Reproduction

```bash
cd backend
uv run python -m scripts.sweep_gtb worlds/gather_trade_build/scenarios/ai_economist_full.yaml \
  --n-seeds 50 --epochs 30 --steps 15 \
  --sweep 'misreporting.fine_multiplier=0.5,1.0,2.0,4.0' \
  --sweep 'misreporting.audit_probability=0.05,0.20,0.50,0.80' \
  --output runs/fine_multiplier_sweep
```
