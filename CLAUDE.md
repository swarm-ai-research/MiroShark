# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

MiroShark with the **Gather-Trade-Build (GTB) economic world model** layered on top — the AI Economist gridworld from [swarm-ai-research/swarm](https://github.com/swarm-ai-research/swarm) vendored under `backend/worlds/gather_trade_build/`, wrapped in a Flask service at `/api/gtb/`, with LLM-driven worker personas and an auto-derived prediction-market layer that emits Polymarket-shaped envelopes.

See PR #1 description on the fork for the full architecture overview.

## Issue tracking

This repo uses **bd (beads)** for trackable work. See `AGENTS.md` for the bd quick-reference. The current research backlog lives in `.beads/issues.jsonl` (not the issues tab on GitHub).

## GTB research methodology — read before claiming a `-` research issue

Single-run smokes are **anecdote, not evidence.** Welfare / Gini / tax revenue swing by 50%+ across seeds for the same scenario; any policy claim from one seed will be reversed by another. The PR #1 smokes that motivated this caveat showed this explicitly — see the audit-rate sweep findings at `backend/runs/audit_sweep/FINDINGS.md` (100 seeds vs the 5-seed smoke that motivated it produced qualitatively different conclusions).

**Before claiming any GTB research issue (any `-` issue in the audit / planner / market / housing space):**

1. Use the seed-sweep harness at `backend/scripts/sweep_gtb.py`. Minimum N=50 seeds per cell for any claim, N=100 for anything you'd put in a findings doc.
2. Aggregate across seeds with mean/median/p10/p90, not point estimates. The harness emits these automatically in `aggregate.csv` and `aggregate_final.csv`.
3. If a claim only holds at one seed or one cell, say so loudly. Surprising single-run results are usually wrong.
4. Render charts with confidence bands (p10–p90), not just lines. See `backend/scripts/audit_sweep_experiment.py` for a working SVG-renderer pattern that doesn't require matplotlib.
5. Write the narrative interpretation in a hand-curated `FINDINGS.md` separate from the auto-generated `TABLE.md` + SVG. Auto-generated narrative carries claims it can't justify from the data — see the bd-an2 commit message for the specific case where this bit us.

### Reproduction commands

```bash
# Generic seed × parameter sweep
cd backend
uv run python -m scripts.sweep_gtb worlds/gather_trade_build/scenarios/ai_economist_full.yaml \
  --n-seeds 50 --epochs 20 \
  --sweep 'misreporting.audit_probability=0.05,0.10,0.20,0.50,0.80'

# Pre-built audit-rate experiment (re-run or skip-sweep to re-render)
uv run python -m scripts.audit_sweep_experiment --n-seeds 100 --epochs 30
uv run python -m scripts.audit_sweep_experiment --skip-sweep  # re-render only

# Headless smoke of a single scenario (no sweep, useful for debugging)
uv run python -m worlds.gather_trade_build.run_scenario \
  worlds/gather_trade_build/scenarios/ai_economist_full.yaml --epochs 5 --steps 5
```

### What the harness already proves about the world

These are findings that should be treated as background knowledge when proposing new experiments:

- **Welfare declines steeply in `audit_probability` up to ~0.2, then plateaus below the no-audit baseline** (`runs/audit_sweep/FINDINGS.md`). Every cell at `audit_probability ≥ 0.05` sits below baseline; the welfare cost is 10.5% peak-to-trough. Sweeps proposing "audit more" are starting from a false premise. (Not strictly monotone — the 0.20→0.30 uptick is within seed noise; PR #2 review caught this.)
- **Tax revenue is Laffer-shaped**, peaking near `audit_probability=0.025`, not at the naive 50% EV-breakeven.
- **Catches don't collapse with high enforcement** — `EvasiveWorkerPolicy` keeps trying periodically regardless of audit rate.
- **Audits saturate at the worker count** (~14) by `audit_probability ≈ 0.2`.
- **Bilateral compute negotiation clears more feasible trades than a posted price.** The arkhai `simple-compute-market` port (`backend/worlds/gather_trade_build/compute_market.py`: listings registry + buyer-driven bisection negotiation + escrow settlement) captures ~all feasible gains-from-trade (welfare capture 1.00 vs 0.92) and fills 0.50 vs 0.34 of drawn encounters against a take-it-or-leave-it ask at a 30% markup — N=100, **disjoint** p10–p90 bands (`backend/runs/compute_market/FINDINGS.md`). The realized-rate / surplus-share gaps between the two mechanisms are **selection artifacts** (they transact different subsets), not price effects — don't read them as "negotiation moves prices."
- **Persona-aligned LLM populations produce one-sided markets by default.** Without an explicit contrarian persona (see `BALANCED_LLM_LINEUP` in `backend/app/services/gtb_llm_personas.py`), every market pile-on goes the same direction and the `yes_probability` is a sentiment poll, not a forecast. The `confidence_source` field on each `/polymarket` envelope flags this.

## Faithful reporting

Report outcomes faithfully: if a sweep shows the opposite of what you predicted, say so. If you ran fewer seeds than the methodology requires, say that too. The point of the harness is to let claims be checked cheaply — abusing it to confirm priors wastes everyone's time.

## Safety / invariants (do not break)

- All YES/NO probabilities the `/polymarket` envelope emits must remain in `[0, 1]`.
- The vendored `backend/worlds/gather_trade_build/` kernel is upstream code; touch sparingly and document divergences (the audit-selection fix at `env.py:_run_audits` is one such; see PR #1).
- Tests in `backend/tests/test_unit_gtb_*.py` are bypass-Flask by design so they run with just `pyyaml` + `pytest`. Don't add Flask-app imports to those test files.

## Memory + planning

If you're claiming a substantial issue (audit sweep, market PD test, RL planner), open a hand-curated `FINDINGS.md` alongside the auto-generated table the moment you have data — don't wait until you "have something to say." The discipline of writing it down per-cell catches off-by-one indexing errors in aggregation that no test catches.
