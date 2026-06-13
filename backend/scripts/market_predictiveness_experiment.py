#!/usr/bin/env python3
"""Does the /polymarket headline yes_probability predict the outcome?

Closes bd-mit. The phase-9 polymarket envelope surfaces a `yes_probability`
for every open market, derived from the live stake book. The bd-xc2 fix
adds a `confidence_source` field flagging one-sided pools as sentiment,
not forecasts. This experiment quantifies the predictive accuracy:

  - For every market that resolves YES or NO, record the LATEST open
    polymarket envelope seen before resolution (the most-informed
    snapshot a consumer could act on). Capturing at creation instead
    is structurally pre-stake — a just-created market cannot have
    stakes yet — so every pair would land in `no_stakes` by
    construction (PR #3 review catch).
  - Compute the Brier score (yes_prob - resolved)^2 grouped by
    confidence_source.
  - Hypothesis (from PR #1 PR-description analysis):
      * one_sided / no_stakes envelopes: Brier ~ 0.25 (chance)
      * two_sided envelopes: meaningfully < 0.25

Runs with the BALANCED_LLM_LINEUP (3 LLM personas including
CONTRARIAN_BEAR) so two-sided pools form by construction. Uses the
gemini-2.5-flash-lite model (15x faster than 2.5-flash per bd-0t5).

This script drives GTBWorldService directly — no Flask boot required.
Cost: ~10 LLM calls per seed × 15 seeds × ~0.6s = ~90 seconds wall.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import types
from pathlib import Path
from typing import Dict, List, Tuple


_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# Stub the `app` package family so we can load gtb_service / gtb_llm_agent /
# gtb_llm_personas / gtb_polymarket directly without booting Flask.
for pkg, path in [
    ("app", "app"), ("app.services", "app/services"), ("app.utils", "app/utils"),
]:
    if pkg not in sys.modules:
        stub = types.ModuleType(pkg); stub.__path__ = [str(_BACKEND / path)]
        sys.modules[pkg] = stub


def _load(name: str, relpath: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, _BACKEND / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Real llm_client (needs flask-free shim already via stubs)
_ = _load("app.utils.llm_client", "app/utils/llm_client.py")
_ = _load("app.services.gtb_llm_agent", "app/services/gtb_llm_agent.py")
gtb_llm_personas = _load("app.services.gtb_llm_personas", "app/services/gtb_llm_personas.py")
gtb_polymarket = _load("app.services.gtb_polymarket", "app/services/gtb_polymarket.py")
gtb_service = _load("app.services.gtb_service", "app/services/gtb_service.py")


logger = logging.getLogger("market_predictiveness")
_DEFAULT_OUTPUT = _BACKEND / "runs/market_predictiveness"


def _drive_one_seed(seed: int, n_epochs: int, steps_per_epoch: int) -> List[Tuple[Dict, str]]:
    """Run one world from start to past every market's deadline.

    Returns a list of (envelope_at_creation, resolved_status) tuples.
    Resolved status is 'yes' / 'no' / 'expired'. Open markets at
    end-of-run are skipped.
    """
    overrides = {
        "simulation": {"steps_per_epoch": steps_per_epoch, "seed": seed},
        "agents": gtb_llm_personas.BALANCED_LLM_LINEUP,
    }
    reg = gtb_service.GTBWorldRegistry()
    sim_id = f"mit-{seed}"
    svc = reg.start(sim_id, overrides=overrides)

    # market_id -> LATEST open envelope (overwritten each tick). When
    # the market resolves we use the last open snapshot — the
    # most-informed yes_probability a consumer could have acted on
    # before settlement. Capturing at creation misses the whole
    # stake-accumulation arc: a market's first envelope predates any
    # chance to stake on it, so it is `no_stakes` by construction.
    latest_open_snapshot: Dict[str, Dict] = {}
    pending_markets = set()

    total_ticks = n_epochs * steps_per_epoch
    for tick in range(total_ticks):
        payload = gtb_polymarket.compute_gtb_polymarket(svc.state(), sim_id)
        if payload:
            for env in payload["markets"]:
                if env["status"] == "open":
                    latest_open_snapshot[env["market_id"]] = dict(env)
                    pending_markets.add(env["market_id"])
        svc.step()

    # Walk resolved markets to find outcomes.
    state = svc.state()
    pairs: List[Tuple[Dict, str]] = []
    for m in state["markets"]["resolved"]:
        mid = m["market_id"]
        snap = latest_open_snapshot.get(mid)
        if snap is None or m["status"] not in ("yes", "no"):
            continue
        pairs.append((snap, m["status"]))
    # Diagnostic: count stake_placed events directly from the stake
    # history so "nobody staked" is observed, not inferred from the
    # envelope snapshots.
    stake_events = [h for h in state.get("stakes", {}).get("history", [])
                    if h.get("event") == "placed"]
    logger.info("  seed %d placed %d stakes", seed, len(stake_events))
    reg.stop(sim_id)
    return pairs


def _brier_by_source(pairs: List[Tuple[Dict, str]]) -> Dict[str, Dict[str, float]]:
    """Group by confidence_source, compute Brier + sample sizes."""
    by_src: Dict[str, List[float]] = {}
    by_src_label: Dict[str, List[Tuple[float, str]]] = {}
    for env, outcome in pairs:
        src = env.get("confidence_source", "unknown")
        yes_prob = float(env["yes_probability"])
        truth = 1.0 if outcome == "yes" else 0.0
        by_src.setdefault(src, []).append((yes_prob - truth) ** 2)
        by_src_label.setdefault(src, []).append((yes_prob, outcome))

    out: Dict[str, Dict[str, float]] = {}
    for src, vals in by_src.items():
        labels = by_src_label[src]
        yes_rate = sum(1 for _, o in labels if o == "yes") / len(labels)
        mean_pred = statistics.mean(p for p, _ in labels)
        out[src] = {
            "n": float(len(vals)),
            "brier_score": statistics.mean(vals),
            "yes_rate": yes_rate,
            "mean_yes_prob": mean_pred,
        }
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    all_pairs: List[Tuple[Dict, str]] = []
    for seed in range(args.n_seeds):
        logger.info("seed %d/%d...", seed + 1, args.n_seeds)
        pairs = _drive_one_seed(seed, args.epochs, args.steps)
        logger.info("  -> %d resolved pairs", len(pairs))
        all_pairs.extend(pairs)

    # Write raw pairs.
    raw = [{**p[0], "_outcome": p[1]} for p in all_pairs]
    (args.output / "pairs.json").write_text(json.dumps(raw, indent=2))

    summary = _brier_by_source(all_pairs)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("=== Brier by confidence_source ===")
    logger.info("%-12s %4s %8s %8s %12s", "source", "n", "Brier", "yes_rate", "mean_yes_prob")
    for src in ("two_sided", "one_sided", "no_stakes"):
        s = summary.get(src)
        if s is None:
            logger.info("%-12s    -        -        -            -  (no observations)", src)
            continue
        logger.info(
            "%-12s %4d %8.4f %8.3f %12.3f",
            src, int(s["n"]), s["brier_score"], s["yes_rate"], s["mean_yes_prob"],
        )


if __name__ == "__main__":
    main()
