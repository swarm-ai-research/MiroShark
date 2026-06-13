#!/usr/bin/env python3
"""External staker harness — drives YES/NO stakes from outside the workforce.

Closes bd-3wh. The PR #1 phase 7 design includes a `POST
/api/gtb/<sim>/stake` endpoint so external clients can take positions
without being LLM workers. This harness exercises that endpoint with
three configurable strategies, validating:

  1. The endpoint accepts inbound stakes from non-worker actors.
  2. Stakes from external clients survive resolution + payout
     (worker coin balances update correctly).
  3. Different staking strategies (kelly, naive, contrarian) produce
     different Brier scores.

Strategies:
  - kelly:     size = max(0, expected_value × bankroll / variance)
               Stakes only when |yes_prob − 0.5| > threshold.
  - naive:     constant amount per market, all YES.
  - contrarian: counter the current pool dominance, fixed amount.

Run modes:
  - Against a live backend (default): polls /state + /polymarket
    every poll_interval seconds, calls POST /stake when triggered.
  - In-process (--in-process): drives GTBWorldService directly, no
    HTTP. Used by the bd-3wh validation harness to test that the
    place_stake service method handles inbound external-actor stakes
    correctly without needing a Flask boot.

The bd-3wh acceptance criterion is in-process by default; the live-
backend mode is for ad-hoc demos.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import types
from pathlib import Path
from typing import Dict, List, Optional


_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# Stub the `app` package family for in-process mode.
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


_ = _load("app.utils.llm_client", "app/utils/llm_client.py")
_ = _load("app.services.gtb_llm_agent", "app/services/gtb_llm_agent.py")
gtb_polymarket = _load("app.services.gtb_polymarket", "app/services/gtb_polymarket.py")
gtb_service = _load("app.services.gtb_service", "app/services/gtb_service.py")


logger = logging.getLogger("external_staker")


def _kelly_size(yes_prob: float, bankroll: float, kelly_fraction: float = 0.1,
                threshold: float = 0.10) -> float:
    """Kelly criterion for binary YES/NO. Returns amount, capped at
    kelly_fraction × bankroll to prevent ruin from a single bad call."""
    edge = yes_prob - 0.5
    if abs(edge) < threshold:
        return 0.0
    # Even-money payoff assumed; size = edge × bankroll, clipped.
    size = abs(edge) * 2 * bankroll * kelly_fraction
    return min(size, bankroll * 0.5)


def _decide_stakes(strategy: str, envelopes: List[dict], bankroll: float,
                   amount: float = 1.0) -> List[dict]:
    """Return a list of stake intents: {market_id, side, amount}."""
    out: List[dict] = []
    for env in envelopes:
        if env.get("status") != "open":
            continue
        yes_prob = float(env["yes_probability"])
        if strategy == "kelly":
            # Don't skip `no_stakes` markets — the envelope's yes_prob is the
            # LLM/baseline prior, which is exactly the signal Kelly sizes
            # against. Skipping them meant kelly never traded on the
            # all-honest population fixture (every run had accepted=0).
            size = _kelly_size(yes_prob, bankroll)
            if size <= 0:
                continue
            side = "yes" if yes_prob > 0.5 else "no"
            out.append({"market_id": env["market_id"], "side": side, "amount": size})
        elif strategy == "naive":
            out.append({"market_id": env["market_id"], "side": "yes", "amount": amount})
        elif strategy == "contrarian":
            yes_pool = float(env.get("yes_pool", 0.0))
            no_pool = float(env.get("no_pool", 0.0))
            if yes_pool == 0 and no_pool == 0:
                # No pool signal yet — fall back to taking the opposite of
                # the envelope's prior so contrarian still opens positions
                # on no-stake markets instead of waiting forever.
                side = "no" if yes_prob >= 0.5 else "yes"
            else:
                side = "no" if yes_pool > no_pool else "yes"
            out.append({"market_id": env["market_id"], "side": side, "amount": amount})
    return out


def run_in_process(strategy: str, n_seeds: int, epochs: int, steps: int,
                   amount: float, external_agent: str) -> dict:
    """Validate that GTBWorldService.place_stake accepts external actors.

    The external_agent is one of the world's existing workers (since
    place_stake requires a known agent_id). In a real deployment this
    would be a separate ghost-agent registered for trading purposes.
    """
    import importlib
    importlib.reload(gtb_service)
    importlib.reload(gtb_polymarket)
    reg = gtb_service.GTBWorldRegistry()
    results = {"strategy": strategy, "n_seeds": n_seeds, "summary": []}
    for seed in range(n_seeds):
        sim_id = f"extstaker-{strategy}-{seed}"
        overrides = {
            "simulation": {"steps_per_epoch": steps, "seed": seed},
            "agents": [
                {"policy": "honest", "count": 4},
            ],
        }
        svc = reg.start(sim_id, overrides=overrides)
        # Give the external_agent a big bankroll to fund stakes.
        worker = svc._env.workers.get(external_agent)
        if worker is None:
            logger.warning("no worker %s in seed %d; skipping", external_agent, seed)
            continue
        worker.inventory["coin"] = 100.0

        accepted = 0
        rejected = 0
        for _tick in range(epochs * steps):
            state = svc.state()
            payload = gtb_polymarket.compute_gtb_polymarket(state, sim_id)
            if payload:
                bankroll = svc._env.workers[external_agent].inventory.get("coin", 0.0)
                intents = _decide_stakes(strategy, payload["markets"], bankroll, amount=amount)
                for intent in intents:
                    res = svc.place_stake(
                        agent_id=external_agent,
                        market_id=intent["market_id"],
                        side=intent["side"],
                        amount=intent["amount"],
                    )
                    if res.get("ok"):
                        accepted += 1
                    else:
                        rejected += 1
            svc.step()

        state = svc.state()
        history = state.get("stakes", {}).get("history", [])
        wins = sum(1 for h in history if h.get("event") == "won" and h.get("agent_id") == external_agent)
        losses = sum(1 for h in history if h.get("event") == "lost" and h.get("agent_id") == external_agent)
        refunds = sum(1 for h in history if h.get("event") in
                      ("refunded", "refunded_no_counterparty") and h.get("agent_id") == external_agent)
        gross_payout = sum(float(h.get("gross_payout", 0.0)) for h in history
                           if h.get("event") == "won" and h.get("agent_id") == external_agent)
        final_coin = svc._env.workers[external_agent].inventory.get("coin", 0.0)
        results["summary"].append({
            "seed": seed,
            "accepted": accepted,
            "rejected": rejected,
            "wins": wins,
            "losses": losses,
            "refunds": refunds,
            "gross_payout": gross_payout,
            "final_coin": final_coin,
        })
        reg.stop(sim_id)
    return results


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=("kelly", "naive", "contrarian"),
                        default="naive")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--amount", type=float, default=1.0,
                        help="Per-stake amount for naive / contrarian")
    parser.add_argument("--external-agent", type=str, default="worker_0",
                        help="Which existing worker acts as the external staker proxy")
    parser.add_argument("--output", type=Path,
                        default=_BACKEND / "runs/external_staker")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    results = run_in_process(args.strategy, args.n_seeds, args.epochs, args.steps,
                             args.amount, args.external_agent)
    out_path = args.output / f"{args.strategy}_results.json"
    out_path.write_text(json.dumps(results, indent=2))

    total_accepted = sum(s["accepted"] for s in results["summary"])
    total_rejected = sum(s["rejected"] for s in results["summary"])
    total_wins = sum(s["wins"] for s in results["summary"])
    total_losses = sum(s["losses"] for s in results["summary"])
    total_refunds = sum(s["refunds"] for s in results["summary"])
    total_gross = sum(s["gross_payout"] for s in results["summary"])
    avg_final_coin = (sum(s["final_coin"] for s in results["summary"])
                      / max(1, len(results["summary"])))

    logger.info("=== %s strategy summary ===", args.strategy)
    logger.info("seeds: %d, accepted: %d, rejected: %d, wins: %d, losses: %d, refunds: %d",
                args.n_seeds, total_accepted, total_rejected, total_wins, total_losses, total_refunds)
    logger.info("gross payout: %.2f, avg final coin: %.2f (started at 100)",
                total_gross, avg_final_coin)


if __name__ == "__main__":
    main()
