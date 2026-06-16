# GTB Economic Sim — High-Fidelity Plan

> **Execution status (2026-06-11):** Phases 0–5 are implemented, tested,
> and merged on this branch. Every behavioral change is behind a config
> flag with the legacy default preserved. Acceptance evidence lives in
> `backend/worlds/gather_trade_build/baselines/` (pre-Phase-1 legacy
> baseline, Phase 1 coherent-ledger acceptance, Phase 2 Saez acceptance)
> and in the per-phase test files `test_unit_gtb_{sweep,ledger,utility,
> market,enforcement}.py` plus additions to `test_unit_gtb_service.py`.
> Beads closed: cec (sweep harness), ih5 (audit_miss filter), 4jr (trade
> activation). Remaining: Phase 6's stylized-fact CI suite + GTB_MODEL.md,
> and the experiment beads the harness now unblocks (an2 audit-rate sweep,
> mit prediction-market calibration, vel/yy1 housing studies, rej
> CLAUDE.md caveat, coq long-horizon planner study — note the runner
> planner-stats fix in Phase 2 changes coq's baseline behavior).

**Date:** 2026-06-11
**Scope:** `backend/worlds/gather_trade_build/` (the GTB kernel), plus the
service layer in `backend/app/services/gtb_service.py` / `gtb_llm_agent.py`
where it shapes what agents can perceive and do.
**Goal:** make GTB a defensible AI-Economist-class economic simulation — one
where the headline metrics (welfare, Gini, tax revenue, bunching, evasion
rates) are driven by real economic mechanisms rather than accounting
artifacts, and where known results from public economics reproduce.

Reference model throughout: Zheng et al., *The AI Economist* (2020) — the
scenario YAML already declares `motif: ai_economist`.

---

## Guiding principle

Fidelity work is ordered by **what invalidates what**. There is no point
tuning the planner or sweeping audit rates while the underlying coin ledger
is incoherent, because every metric (income, tax base, Gini, welfare) is
downstream of it. Hence the phases below: ledger → preferences/welfare →
markets → enforcement → agents → validation. Each phase has acceptance
criteria that double as regression tests.

---

## Phase 0 — Measure before changing anything

You cannot claim higher fidelity without a baseline to compare against.
This phase also unblocks four open beads (`cec`, `an2`, `vel`, `yy1`).

**0.1 Seed-sweep harness** (bead `miroshark-gtb-cec`, P1)
- New `backend/worlds/gather_trade_build/sweep.py`: run a scenario across
  N seeds (and optionally a parameter grid), aggregate per-epoch
  `GTBMetrics` into mean ± 95% CI, write one CSV per sweep plus a manifest
  (config hash, git SHA, seeds).
- `GTBScenarioRunner` already takes `seed`; the harness is a thin loop over
  it. Parallelize with `multiprocessing` — the kernel is pure Python and
  single-seeded runs are independent.

**0.2 Baseline characterization report**
- Run `ai_economist_full.yaml` over ≥30 seeds. Record the distributions of
  every metric in `metrics.py`. This is the "before" snapshot every later
  phase is judged against.

**0.3 Ledger audit instrumentation**
- Add a per-epoch conservation check (debug-level): total coin in worker
  inventories at epoch end must equal starting coin + minted − burned, with
  every mint/burn source itemized (house income, trade fees, taxes, fines,
  stake escrow/payouts). Today this check would *fail loudly* — that is the
  point; it makes Phase 1's bugs visible and keeps them fixed forever.

**0.4 Hygiene**
- Drop the vestigial `audit_miss` filter in `metrics.py:191-193` (bead
  `ih5`): the env no longer emits `audit_miss`, so `undetected_evasion_rate`
  is structurally 0. Replace it with a real definition (see 4.3).

---

## Phase 1 — Make the coin ledger coherent (highest impact)

The current accounting breaks the link between "income", "coin", and
"tax", which silently corrupts every economic conclusion.

**1.1 Gathering creates taxable income but no coin** (`env.py:331-335`)
- `_handle_gather` increments `gross_income_this_epoch += gathered` but
  adds zero COIN. Workers are taxed at epoch end on income they never
  received as money, paid out of the 10-coin starting endowment.
- Fix (AI Economist semantics): income = coin actually received. Gathering
  yields *resources only*; income arises when resources are sold or when
  houses pay out. Alternative (if we want gather-income to stay): mint coin
  1:1 at gather time and treat the resource as a separate asset. Pick one —
  recommend the former; it makes trade load-bearing (helps bead `4jr`).

**1.2 Selling double-counts income** (`env.py:607-609`)
- Sale revenue is added to `gross_income_this_epoch` on top of the gather
  income already booked for the same units. Under 1.1's fix this resolves
  itself; otherwise sale income must be net of the gather-time cost basis.
- Buyer side: purchases are not an expense. Income should be net coin
  inflow per epoch, not gross receipts.

**1.3 Taxes are burned; no redistribution**
- `end_epoch` (`env.py:659-676`) removes coin and the revenue vanishes. In
  the AI Economist, revenue is redistributed lump-sum equally — that is the
  entire equity side of the planner's tradeoff. Without it, taxation is
  pure deadweight loss and the planner's `ineq_weight` has almost nothing
  to act on.
- Fix: collect into a treasury, redistribute equally (configurable:
  `redistribution: lump_sum | none | public_goods`) at epoch close, emit a
  `redistribution` event per worker.

**1.4 Houses mint unbounded coin** (`env.py:522-544`)
- `income_per_house_per_step` is printed from nothing, forever, with no
  upkeep and no demand side. Combined with 1.3's burn, the money supply is
  arbitrary. This is why "building dominates" (bead `vel`) is suspected.
- Fix options (in fidelity order): (a) houses pay utility, not coin, as in
  AI Economist (build reward enters the utility function); (b) house income
  is rent paid by other agents (transfers, not minting); (c) keep minting
  but add maintenance cost + decay so ROI is finite. Recommend (a) + small
  coin payout funded by the treasury if a coin channel is wanted.
- Also fix: `_handle_build` (`env.py:346-395`) never checks `cell.house` —
  houses stack on one cell, and on top of resource tiles. Require an empty,
  non-resource cell (this also matches what the LLM prompt already claims).

**1.5 Insolvency is silent legal evasion** (`env.py:661-664`, `761`)
- Tax/fine owed beyond coin balance just evaporates (`actual_tax =
  min(tax, coin)`); the shortfall is logged in the event but never carried.
- Fix: carry unpaid tax/fines as debt into following epochs (deducted
  before other spending), and surface `total_tax_shortfall` as a metric.

**Acceptance for Phase 1:** conservation check from 0.3 passes on every
epoch of a 30-seed sweep; `total_tax_revenue` equals coin actually moved to
the treasury; income per worker equals net coin inflow.

---

## Phase 2 — Real preferences and a real welfare objective

**2.1 Isoelastic utility with labor disutility** (`reward.py`)
- Current utility is linear in coin/resources/houses; `compute_epoch_reward`
  is unused by any policy. With linear utility and no labor cost there is
  no labor-supply response to taxation — the central mechanism the AI
  Economist exists to study cannot occur.
- Implement `u_i = crra(coin_i; eta) + house_utility * houses_i -
  labor_coeff * effort_i`, where effort is accumulated energy spent
  (already tracked implicitly; book it per epoch). Expose `eta`,
  `labor_coeff`, `house_utility` in a new `UtilityConfig`.
- Wire it: scripted policies get an effort throttle (stop working when
  marginal post-tax utility of the next action < its labor cost — a
  one-line comparison against the current marginal tax rate, which is
  already in the obs); the LLM prompt states the same tradeoff explicitly.

**2.2 Fix the welfare objective’s unit mismatch** (`planner.py:69-73`,
`metrics.py:177`)
- `welfare = prod_weight * mean_income - ineq_weight * gini` subtracts a
  [0,1] index from a quantity in the tens — inequality is decorative.
- Replace with the AI Economist's `welfare = productivity × equality`
  (equality = 1 − Gini), and additionally report a utilitarian SWF
  (sum of isoelastic utilities) once 2.1 lands. Keep the old scalar as
  `welfare_legacy` for one release so dashboards don't break.

**2.3 Gini on wealth, not per-epoch gross income**
- Per-epoch income resets every ~15 steps and is noisy; the AI Economist
  measures inequality on cumulative coin endowment. Report both
  `gini_income` and `gini_wealth` (coin + house replacement value);
  planner consumes `gini_wealth`.

**2.4 A planner worth the name** (`planner.py`)
- Heuristic planner moves *all* brackets by the same delta; the bandit is a
  one-step hill climber; `schedule_family: saez` is declared in config but
  unimplemented.
- Implement: (a) per-bracket adjustment in the heuristic (top brackets
  react to Gini, bottom to productivity); (b) a Saez-style planner that
  estimates the income-elasticity from observed responses across epochs and
  sets rates by the inverse-elasticity rule — this is the flagship
  "high-fidelity" feature and directly serves bead `coq` (planner
  reactivity over long horizons); (c) leave the RL stub but document it.
- Damping already exists in `TaxSchedule.update_brackets`; verify the
  planner-side `learning_rate` and schedule-side `damping` don't fight
  (today both are applied).

**Acceptance:** with 2.1–2.4 on, a sweep over top-bracket rate shows an
interior Laffer peak in tax revenue (it cannot today, since effort never
responds); equality enters welfare with real weight.

---

## Phase 3 — Market microstructure that can discover prices

**3.1 Persistent order book** (`env.py:550-631`)
- Orders are cleared at the end of every step (`_buy_orders.clear()`), so
  a bid posted one tick before an ask never trades. With scripted agents
  this is why trade never activates (bead `4jr`).
- Keep unmatched orders alive with a TTL (default: rest of epoch), cancel
  on epoch close, and let agents cancel/replace. Escrow coin (buyer) and
  resources (seller) at post time so resting orders can't be invalidated
  by later actions — today matching re-checks balances and silently skips,
  which lets agents spoof.

**3.2 Price information in observations**
- Publish last-trade price, best bid/ask, and rolling volume per resource
  in `obs()` and in the LLM prompt's observation JSON
  (`gtb_llm_agent.py:_render_obs`). No agent — scripted or LLM — can do
  price discovery today because no price ever appears in an observation.

**3.3 Trading baseline policy**
- Add a `trader` scripted policy (buy wood/stone below its build-value,
  sell above) so markets have organic two-sided flow without LLM agents.
  This also fixes the one-sided-liquidity caveat on the prediction-market
  envelope (bead `xc2` documents it; this addresses the cause).

**Acceptance:** ≥X% of epochs contain at least one trade in the default
scenario (X calibrated from sweep); price series mean-reverts to the
build-value fundamental; spoofing test (post-and-drain) is rejected by
escrow.

---

## Phase 4 — Enforcement and collusion that behave like the literature

**4.1 Consistent misreporting semantics** (`env.py:496-516`, `531-536`)
- `MISREPORT` rewrites `reported_income` once, then *gather* income accrues
  to reported at full value while *house* income applies the fraction —
  the hidden share depends on action ordering within the epoch.
- Fix: `misreport_fraction` is a per-epoch stance; reported income is
  derived at epoch close as `gross × (1 − fraction)` uniformly. One code
  path, no ordering effects.

**4.2 Audits with noise** (`env.py:721-810`)
- Today selection ⇒ certain conviction. Real audits have detection power
  < 1 and use third-party-verifiable signals. Add `detection_power`
  (probability a selected audit finds the discrepancy) and make selection
  risk-scored on *observables* (reported income vs. visible wealth, houses,
  trade volume — all of which the env knows) rather than on the true
  discrepancy, which the tax authority cannot see. The current code
  conditions selection on `gross − reported`, i.e. the auditor reads the
  agent's mind.
- Re-define `undetected_evasion_rate` properly: hidden income not caught ÷
  total hidden income (computable from worker state at epoch close).
- Then run the audit-rate EV sweep (bead `an2`) on the corrected model:
  Allingham–Sandmo predicts evasion falls as `audit_prob × fine_multiplier
  × detection_power` crosses the marginal-rate breakeven; verify.

**4.3 Collusion that actually pays, detection that can be wrong**
(`env.py:816-902`, `agents.py:242-302`)
- Today the "cartel" synchronizes action *strings* but gains nothing — no
  price-fixing, no profit sharing (the docstring claims both). And the
  detector matches action strings positionally, so honest gatherers
  spamming GATHER trip the 0.7 similarity threshold: false positives are
  baked in, which contaminates `governance_backfire_events`.
- Give collusion a real channel once 3.1 lands: coalition members trade
  with each other at off-market prices to shift income to low-bracket
  members, and coordinate sell prices (price-fixing). Detection then has
  something real to find: within-coalition trade share, price deviation
  from market, transfer asymmetry — measurable, evadeable at a cost, and
  with a tunable ROC. Report detector precision/recall as metrics instead
  of pretending detection is free.

**Acceptance:** Allingham–Sandmo comparative statics hold in sweep;
collusion ON vs OFF changes coalition members' post-tax utility by a
measurable margin; detector ROC curve is computable from event logs.

---

## Phase 5 — Agent fidelity (scripted and LLM)

**5.1 Truthful LLM grounding** (`gtb_llm_agent.py`)
- Prompt says houses are built "on an empty cell adjacent to you"; env
  builds on the current cell (and after 1.4, requires it empty). Align
  prompt to env, not vice versa.
- Add to the obs rendering: last epoch's tax paid and effective rate, own
  audit history, market prices (3.2), current treasury/redistribution and
  the utility parameters (2.1) so "maximize long-run utility" is computable
  rather than vibes.
- Add a compact per-agent memory line (last epoch outcome) — strategic
  behavior across epochs is impossible when every tick is amnesiac.
- Faster model for live-demo continuous play is bead `0t5`; orthogonal,
  but the obs slimming above reduces tokens either way.

**5.2 Stronger scripted baselines**
- After 2.1, scripted agents should respond to marginal rates (work less
  past a bracket), giving the planner a non-degenerate environment even
  without LLM workers. Keep current policies as `*_naive` variants so old
  experiments remain reproducible.

---

## Phase 6 — Validation suite: stylized facts as CI

Encode the field's known results as pytest-marked "fidelity smokes" run on
the seed-sweep harness (small N in CI, large N on demand):

1. **Bunching at kinks** (Saez 2010): with gaming agents enabled, excess
   mass below bracket thresholds; vanishes when `smoothing > 0`.
2. **Laffer curve**: revenue is single-peaked in the top rate (needs 2.1).
3. **Allingham–Sandmo**: evasion decreasing in audit prob, fine, detection
   power; increasing in marginal rate (needs 4.2).
4. **Redistribution reduces Gini** without destroying production at
   moderate rates (needs 1.3, 2.1).
5. **No-free-lunch on building** (bead `vel`/`yy1`): after 1.4, building
   ROI is finite and housing-cost sensitivity sweeps are meaningful.
6. **Conservation invariant** (0.3) as a hard test, every run.
7. **Prediction-market calibration** (bead `mit`): Brier score of market
   prices vs outcomes across seeds — the headline must beat the base rate.

Document the whole model honestly in `docs/GTB_MODEL.md`: mechanisms,
deliberate simplifications (no consumption good, no credit, single
currency, uniform regen), and the validated-claims list with sweep
evidence. The CLAUDE.md caveat about seed sweeps is bead `rej`.

---

## Sequencing and effort

| Phase | Depends on | Size | Beads closed/unblocked |
|---|---|---|---|
| 0 Harness + baseline | — | S–M | cec; unblocks an2, mit, vel, yy1; ih5 |
| 1 Coherent ledger | 0.3 | M | (prereq for vel/yy1 to be meaningful) |
| 2 Preferences + welfare + Saez planner | 1 | M–L | coq |
| 3 Market microstructure | 1 | M | 4jr, xc2 (root cause) |
| 4 Enforcement + collusion | 1, 3 | M | an2 |
| 5 Agent fidelity | 2, 3 | S–M | 0t5 (partial) |
| 6 Validation suite + docs | all | M | mit, vel, yy1, rej |

Phases 2 and 3 are independent after Phase 1 and can run in parallel.
Every behavioral change lands behind a config flag with the legacy default
preserved for one release, so existing scenario YAMLs and the live demo
keep working while sweeps quantify the difference.
