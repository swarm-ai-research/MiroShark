#!/usr/bin/env python3
"""Trade activation experiment.

Closes bd-4jr. Across every prior smoke (rule-based + LLM-driven) the
wood/stone market saw ~zero use. This experiment tests three arms to
see which (if any) activates trade:

1. baseline — default scenario, default agent lineup
2. heterogeneous skill — workers get skill_gather ∈ {0.5, 1.0, 1.5}
   drawn round-robin. Should make gather efficiency asymmetric and
   create gains-from-specialization
3. bot-makers — add 2 MakerWorkerPolicy agents that post passive
   limit orders (buy below mid, sell above mid). Provides counterparty
   liquidity that other agents can cross

Metric: trade_count / total_actions per epoch, plus final tax revenue
and welfare so we can see if activating trade changes the equilibrium.

Method: 3 arms × 20 seeds × 25 epochs × 10 steps. The arms are
constructed as separate scenario YAML variants in a scratch dir; the
sweep harness runs each at multiple seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


_BACKEND = Path(__file__).resolve().parent.parent
_SCENARIO = _BACKEND / "worlds/gather_trade_build/scenarios/ai_economist_full.yaml"
_DEFAULT_OUTPUT = _BACKEND / "runs/trade_activation"

logger = logging.getLogger("trade_activation")


def _write_arm_scenarios(scratch: Path) -> Dict[str, Path]:
    """Materialize the three arms as separate YAMLs."""
    with open(_SCENARIO) as f:
        base = yaml.safe_load(f)
    scratch.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}

    # Arm 1: baseline (just copy)
    baseline = json.loads(json.dumps(base))
    p = scratch / "baseline.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(baseline, f)
    out["baseline"] = p

    # Arm 2: heterogeneous skill (replace lineup with 3 skill cohorts)
    hetero = json.loads(json.dumps(base))
    hetero["agents"] = [
        {"policy": "honest", "count": 4, "skill_gather": 0.5},
        {"policy": "honest", "count": 4, "skill_gather": 1.0},
        {"policy": "honest", "count": 4, "skill_gather": 1.5},
    ]
    p = scratch / "hetero.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(hetero, f)
    out["hetero_skill"] = p

    # Arm 3: bot-makers (default lineup + 2 makers)
    makers = json.loads(json.dumps(base))
    makers["agents"] = list(base["agents"]) + [
        {"policy": "maker", "count": 2, "sell_markup": 0.2, "buy_discount": 0.2, "target_inventory": 5.0},
    ]
    p = scratch / "makers.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(makers, f)
    out["bot_makers"] = p

    # Arm 4 (bd-8dj): market_aware honest workers + 2 makers.
    # Replaces 4 of the default honest workers with market_aware variants
    # that read the order book and cross spreads when the price is right.
    aware = json.loads(json.dumps(base))
    aware_agents = []
    for spec in base["agents"]:
        if spec.get("policy") == "honest":
            count = spec.get("count", 1)
            aware_count = min(4, count)
            if aware_count > 0:
                aware_agents.append({**spec, "policy": "market_aware", "count": aware_count})
            if count - aware_count > 0:
                aware_agents.append({**spec, "count": count - aware_count})
        else:
            aware_agents.append(spec)
    aware_agents.append(
        {"policy": "maker", "count": 2, "sell_markup": 0.2, "buy_discount": 0.2, "target_inventory": 5.0},
    )
    aware["agents"] = aware_agents
    p = scratch / "market_aware.yaml"
    with open(p, "w") as f:
        yaml.safe_dump(aware, f)
    out["market_aware"] = p

    return out


def _run_arm(scenario: Path, output: Path, n_seeds: int, n_epochs: int, steps: int) -> None:
    cmd = [
        sys.executable, "-m", "scripts.sweep_gtb", str(scenario),
        "--n-seeds", str(n_seeds), "--epochs", str(n_epochs), "--steps", str(steps),
        "--output", str(output),
    ]
    logger.info("running arm: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=_BACKEND, check=True)


def _percentile(vs: List[float], p: float) -> float:
    if not vs:
        return float("nan")
    if len(vs) == 1:
        return vs[0]
    qs = statistics.quantiles(vs, n=100, method="inclusive")
    return qs[max(0, min(98, int(round(p * 100)) - 1))]


def _trade_rate_by_arm(arm_dirs: Dict[str, Path]) -> List[dict]:
    """Walk event_log.jsonl files. Separately count:
    - orders_placed: limit-order submissions (TRADE_BUY/TRADE_SELL action → emits order_placed event)
    - trades_executed: actual order-book crossings (env emits 'trade' event when two orders cross)
    The activation question is about EXECUTIONS; orders_placed shows
    'agents try to trade but no one fills'."""
    rows: List[dict] = []
    for arm_name, arm_dir in arm_dirs.items():
        per_seed_executed: List[float] = []
        per_seed_placed: List[float] = []
        per_seed_welfare: List[float] = []
        per_seed_tax: List[float] = []
        seed_dirs = sorted((arm_dir / "cells" / "cell_000").glob("seed_*"))
        for sd in seed_dirs:
            log = sd / "event_log.jsonl"
            if not log.exists():
                continue
            total_actions = 0
            placed = 0
            executed = 0
            with open(log) as f:
                for line in f:
                    if not line.strip():
                        continue
                    ev = json.loads(line)
                    et = ev.get("event_type", "")
                    # Worker actions (one per (worker, step)).
                    if et in ("move", "gather", "gather_fail", "build", "build_fail",
                              "order_placed", "noop", "shift_income", "misreport"):
                        total_actions += 1
                    if et == "order_placed":
                        placed += 1
                    if et == "trade":
                        executed += 1
            per_seed_placed.append(placed / total_actions if total_actions > 0 else 0.0)
            per_seed_executed.append(executed / total_actions if total_actions > 0 else 0.0)
            m_csv = sd / "csv" / "metrics.csv"
            if m_csv.exists():
                with open(m_csv) as f:
                    rs = list(csv.DictReader(f))
                if rs:
                    per_seed_welfare.append(float(rs[-1]["welfare"]))
                    per_seed_tax.append(float(rs[-1]["total_tax_revenue"]))
        rows.append({
            "arm": arm_name,
            "n_seeds": len(per_seed_executed),
            "order_rate_mean": statistics.mean(per_seed_placed) if per_seed_placed else 0.0,
            "executed_rate_mean": statistics.mean(per_seed_executed) if per_seed_executed else 0.0,
            "executed_rate_p10": _percentile(per_seed_executed, 0.10),
            "executed_rate_p90": _percentile(per_seed_executed, 0.90),
            "final_welfare_mean": statistics.mean(per_seed_welfare) if per_seed_welfare else 0.0,
            "final_tax_mean": statistics.mean(per_seed_tax) if per_seed_tax else 0.0,
        })
    return rows


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    arm_dirs: Dict[str, Path] = {}
    if not args.skip_sweep:
        scratch = args.output / "_scratch"
        arms = _write_arm_scenarios(scratch)
        for arm_name, scenario_path in arms.items():
            arm_out = args.output / arm_name
            _run_arm(scenario_path, arm_out, args.n_seeds, args.epochs, args.steps)
            arm_dirs[arm_name] = arm_out
    else:
        for name in ("baseline", "hetero_skill", "bot_makers", "market_aware"):
            arm_dirs[name] = args.output / name

    rows = _trade_rate_by_arm(arm_dirs)
    _write_csv(args.output / "trade_rate_by_arm.csv", rows)
    for r in rows:
        logger.info(
            "%-15s orders=%.4f executed=%.4f (p10=%.4f p90=%.4f) welfare=%.2f tax=%.2f",
            r["arm"], r["order_rate_mean"], r["executed_rate_mean"],
            r["executed_rate_p10"], r["executed_rate_p90"],
            r["final_welfare_mean"], r["final_tax_mean"],
        )


if __name__ == "__main__":
    main()
