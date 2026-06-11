#!/usr/bin/env python3
"""LLM workforce × planner reactivity.

Closes bd-5t0. Was blocked on bd-2e2 (TaxAwareHonestPolicy) OR bd-1nq
(stake-forcing personas); both landed, so this can run.

Question: with an LLM-driven workforce that reads `tax_schedule` from
its obs (the phase-3 LLMWorkerPolicy already does this), does
planner_type matter? bd-coq found NO with the default rule-based
lineup. bd-2e2's TaxAwareHonestPolicy showed YES with a tax-aware
rule-based policy. This experiment tests the LLM-population case.

Setup: BALANCED_LLM_LINEUP (Ada / Bo / Cy / Dax) × 3 planner types
× 8 seeds × 8 epochs × 4 steps each. Wall-clock ~12 minutes of
gemini-2.5-flash-lite calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
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

logger = logging.getLogger("llm_planner")
_DEFAULT_OUTPUT = _BACKEND / "runs/llm_planner"


def _run_seed(planner_type: str, seed: int, epochs: int, steps: int) -> dict:
    overrides = {
        "simulation": {"steps_per_epoch": steps, "seed": seed},
        "agents": gtb_llm_personas.BALANCED_LLM_LINEUP,
        "domain": {
            "planner": {
                "planner_type": planner_type,
                "learning_rate": 0.05,
                "update_interval_epochs": 1,
            },
            "taxation": {"allow_non_monotone": True},
        },
    }
    reg = gtb_service.GTBWorldRegistry()
    sim_id = f"5t0-{planner_type}-{seed}"
    svc = reg.start(sim_id, overrides=overrides)

    for _ in range(epochs * steps):
        svc.step()
    state = svc.state()
    final_metrics = state.get("last_metrics") or {}
    metrics_history = state.get("metrics_history") or []
    reg.stop(sim_id)
    return {
        "planner_type": planner_type,
        "seed": seed,
        "welfare": final_metrics.get("welfare"),
        "tax_revenue": final_metrics.get("total_tax_revenue"),
        "gini": final_metrics.get("gini_coefficient"),
        "production": final_metrics.get("total_production"),
        "trajectory": [m.get("welfare") for m in metrics_history],
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

    results: List[dict] = []
    for planner_type in ("heuristic", "bandit", "rl"):
        logger.info("=== planner=%s ===", planner_type)
        for seed in range(args.n_seeds):
            r = _run_seed(planner_type, seed, args.epochs, args.steps)
            results.append(r)
            logger.info(
                "  seed %d: welfare=%s tax=%s gini=%s",
                seed,
                f"{r['welfare']:.2f}" if r['welfare'] is not None else "-",
                f"{r['tax_revenue']:.2f}" if r['tax_revenue'] is not None else "-",
                f"{r['gini']:.3f}" if r['gini'] is not None else "-",
            )
    (args.output / "results.json").write_text(json.dumps(results, indent=2))

    # Aggregate
    logger.info("=== summary ===")
    for planner_type in ("heuristic", "bandit", "rl"):
        rows = [r for r in results if r["planner_type"] == planner_type and r["welfare"] is not None]
        if not rows:
            continue
        logger.info(
            "%-10s mean welfare=%.2f tax=%.2f gini=%.3f",
            planner_type,
            statistics.mean(r["welfare"] for r in rows),
            statistics.mean(r["tax_revenue"] for r in rows),
            statistics.mean(r["gini"] for r in rows),
        )


if __name__ == "__main__":
    main()
