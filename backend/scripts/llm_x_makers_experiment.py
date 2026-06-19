#!/usr/bin/env python3
"""LLM agents × bot makers — does LLM-rent income cross the maker spread?

Closes bd-hov (unblocked by bd-3wh). Tests whether LLM workers,
once they've accumulated rent income from houses, start crossing the
order book that the bot makers post.

Arms:
  - llm_only: BALANCED_LLM_LINEUP (no makers)
  - llm_plus_makers: BALANCED_LLM_LINEUP + 2 MakerWorkerPolicy bots

Per arm: 5 seeds × 8 epochs × 4 steps. Drives GTBWorldService
directly (no Flask boot). Counts orders placed + executed trades.

Metric: executed_rate per arm. If LLM agents cross the maker spreads,
executed_rate > 0 in the llm_plus_makers arm. If not, the bd-4jr
finding (no rule-based or LLM agent crosses the maker book) carries
through to LLM-with-rent populations.

Cost: ~45s wall × 2 arms = ~90s Gemini calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from pathlib import Path
from typing import Dict, List


_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

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
gtb_llm_personas = _load("app.services.gtb_llm_personas", "app/services/gtb_llm_personas.py")
gtb_service = _load("app.services.gtb_service", "app/services/gtb_service.py")

logger = logging.getLogger("llm_x_makers")
_DEFAULT_OUTPUT = _BACKEND / "runs/llm_x_makers"


def _run_seed(arm: str, seed: int, epochs: int, steps: int) -> dict:
    """Run one seed of the given arm. Returns event counts + welfare."""
    if arm == "llm_only":
        agents = gtb_llm_personas.BALANCED_LLM_LINEUP
    elif arm == "llm_plus_makers":
        agents = list(gtb_llm_personas.BALANCED_LLM_LINEUP) + [
            {"policy": "maker", "count": 2,
             "sell_markup": 0.2, "buy_discount": 0.2, "target_inventory": 5.0},
        ]
    else:
        raise ValueError(f"unknown arm {arm}")

    overrides = {
        "simulation": {"steps_per_epoch": steps, "seed": seed},
        "agents": agents,
    }
    reg = gtb_service.GTBWorldRegistry()
    sim_id = f"hov-{arm}-{seed}"
    svc = reg.start(sim_id, overrides=overrides)

    placed = 0
    executed = 0
    total = 0
    for _ in range(epochs * steps):
        tick = svc.step()
        for ev in tick["events"]:
            et = ev["event_type"]
            if et in ("move", "gather", "gather_fail", "build", "build_fail",
                      "order_placed", "noop", "shift_income", "misreport"):
                total += 1
            if et == "order_placed":
                placed += 1
            if et == "trade":
                executed += 1

    final = svc.state()
    welfare_history = [m["welfare"] for m in final.get("metrics_history", [])]
    reg.stop(sim_id)
    return {
        "arm": arm,
        "seed": seed,
        "total_actions": total,
        "orders_placed": placed,
        "trades_executed": executed,
        "executed_rate": executed / total if total > 0 else 0.0,
        "final_welfare": welfare_history[-1] if welfare_history else 0.0,
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args.output.mkdir(parents=True, exist_ok=True)

    all_results: List[dict] = []
    for arm in ("llm_only", "llm_plus_makers"):
        logger.info("=== %s ===", arm)
        for seed in range(args.n_seeds):
            r = _run_seed(arm, seed, args.epochs, args.steps)
            all_results.append(r)
            logger.info(
                "  seed %d: total=%d orders=%d executed=%d (rate=%.4f) welfare=%.2f",
                seed, r["total_actions"], r["orders_placed"],
                r["trades_executed"], r["executed_rate"], r["final_welfare"],
            )

    (args.output / "results.json").write_text(json.dumps(all_results, indent=2))

    # Aggregate per arm
    import statistics
    for arm in ("llm_only", "llm_plus_makers"):
        arm_results = [r for r in all_results if r["arm"] == arm]
        if not arm_results:
            continue
        logger.info(
            "%-15s mean executed_rate=%.4f, mean orders_placed=%.1f, mean welfare=%.2f",
            arm,
            statistics.mean(r["executed_rate"] for r in arm_results),
            statistics.mean(r["orders_placed"] for r in arm_results),
            statistics.mean(r["final_welfare"] for r in arm_results),
        )


if __name__ == "__main__":
    main()
