"""Seed-sweep harness for GTB scenarios.

Runs a scenario across many seeds (optionally with config overrides),
aggregates per-epoch metrics into mean / std / 95% CI, and exports a
manifest + CSVs so results are reproducible and comparable across code
changes. Closes bead miroshark-gtb-cec; the audit-rate (an2),
prediction-market calibration (mit), and housing-cost (yy1) experiments
all run on top of this.

Usage:
    python -m worlds.gather_trade_build.sweep scenarios/ai_economist_full.yaml \
        --seeds 30 --output runs/sweep_baseline
    python -m worlds.gather_trade_build.sweep scenarios/ai_economist_full.yaml \
        --seeds 30 --set domain.misreporting.audit_probability=0.5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from worlds.gather_trade_build.config import GTBConfig
from worlds.gather_trade_build.runner import GTBScenarioRunner

logger = logging.getLogger(__name__)


def _set_dotted(data: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested dict value from a dotted path, creating dicts as needed."""
    keys = dotted_key.split(".")
    node = data
    for k in keys[:-1]:
        nxt = node.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            node[k] = nxt
        node = nxt
    node[keys[-1]] = value


def _git_sha() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip() or None
    except Exception:
        return None


def _config_hash(data: Dict[str, Any]) -> str:
    blob = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def run_one_seed(
    scenario_data: Dict[str, Any],
    seed: int,
    n_epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one seeded scenario. Module-level so ProcessPoolExecutor can pickle it.

    Returns per-epoch metric dicts plus the final coin-ledger report.
    """
    config = GTBConfig.from_dict(scenario_data.get("domain", {}))
    sim = scenario_data.get("simulation", {})
    config.seed = seed
    runner = GTBScenarioRunner(
        config=config,
        agent_specs=scenario_data.get("agents", [{"policy": "honest", "count": 5}]),
        n_epochs=n_epochs or sim.get("n_epochs", 10),
        steps_per_epoch=steps_per_epoch or sim.get("steps_per_epoch", 10),
        seed=seed,
    )
    metrics = runner.run()
    ledger = runner.env.coin_ledger()
    return {
        "seed": seed,
        "metrics": [m.to_dict() for m in metrics],
        "ledger": ledger,
        "coin_conserved": abs(ledger["discrepancy"]) <= 1e-6,
    }


# Wrapper used by the process pool (single-arg for executor.map).
def _run_one_packed(args: tuple) -> Dict[str, Any]:
    scenario_data, seed, n_epochs, steps = args
    return run_one_seed(scenario_data, seed, n_epochs, steps)


@dataclass
class SweepResult:
    """All per-seed runs plus the aggregated per-epoch summary."""

    scenario_id: str
    scenario_data: Dict[str, Any]
    seeds: List[int]
    runs: List[Dict[str, Any]] = field(default_factory=list)
    # summary[epoch][metric] = {"mean": .., "std": .., "ci95": .., "min": .., "max": ..}
    summary: Dict[int, Dict[str, Dict[str, float]]] = field(default_factory=dict)

    @property
    def all_conserved(self) -> bool:
        return all(r["coin_conserved"] for r in self.runs)

    def metric_series(self, metric: str, stat: str = "mean") -> List[float]:
        """Per-epoch series of an aggregated stat, e.g. mean welfare by epoch."""
        return [
            self.summary[epoch][metric][stat]
            for epoch in sorted(self.summary)
            if metric in self.summary[epoch]
        ]


def _aggregate(runs: List[Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Aggregate per-seed metric dicts into per-epoch mean/std/ci95/min/max."""
    by_epoch: Dict[int, Dict[str, List[float]]] = {}
    for run in runs:
        for m in run["metrics"]:
            epoch = int(m["epoch"])
            bucket = by_epoch.setdefault(epoch, {})
            for key, value in m.items():
                if key == "epoch":
                    continue
                # Booleans (coin_conserved) aggregate as 0/1 rates
                bucket.setdefault(key, []).append(float(value))

    summary: Dict[int, Dict[str, Dict[str, float]]] = {}
    for epoch, metrics in by_epoch.items():
        summary[epoch] = {}
        for key, values in metrics.items():
            n = len(values)
            mean = sum(values) / n
            var = sum((v - mean) ** 2 for v in values) / n
            std = math.sqrt(var)
            ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
            summary[epoch][key] = {
                "mean": mean, "std": std, "ci95": ci95,
                "min": min(values), "max": max(values),
            }
    return summary


def run_sweep(
    scenario_path: str | Path,
    seeds: List[int],
    overrides: Optional[Dict[str, Any]] = None,
    n_epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    processes: int = 1,
) -> SweepResult:
    """Run a scenario across seeds and aggregate the metrics.

    Args:
        scenario_path: Path to the scenario YAML.
        seeds: Explicit list of seeds to run.
        overrides: Dotted-key overrides applied to the scenario dict,
            e.g. {"domain.misreporting.audit_probability": 0.5}.
        n_epochs / steps_per_epoch: Optional simulation overrides.
        processes: Worker processes; 1 runs inline (deterministic ordering,
            simpler debugging, required under pytest on some platforms).
    """
    with open(scenario_path) as f:
        scenario_data = yaml.safe_load(f)
    for key, value in (overrides or {}).items():
        _set_dotted(scenario_data, key, value)

    jobs = [(scenario_data, s, n_epochs, steps_per_epoch) for s in seeds]
    if processes > 1:
        with ProcessPoolExecutor(max_workers=processes) as pool:
            runs = list(pool.map(_run_one_packed, jobs))
    else:
        runs = [_run_one_packed(j) for j in jobs]

    result = SweepResult(
        scenario_id=scenario_data.get("scenario_id", Path(scenario_path).stem),
        scenario_data=scenario_data,
        seeds=list(seeds),
        runs=runs,
        summary=_aggregate(runs),
    )
    if not result.all_conserved:
        bad = [r["seed"] for r in runs if not r["coin_conserved"]]
        logger.warning("Coin conservation violated for seeds: %s", bad)
    return result


def export_sweep(result: SweepResult, output_dir: str | Path) -> Path:
    """Write manifest.json, summary.csv, and per_seed_metrics.csv."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "scenario_id": result.scenario_id,
        "config_hash": _config_hash(result.scenario_data),
        "git_sha": _git_sha(),
        "seeds": result.seeds,
        "n_seeds": len(result.seeds),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "coin_conserved_all_runs": result.all_conserved,
        "scenario": result.scenario_data,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # Long-format per-seed metrics: one row per (seed, epoch)
    if result.runs and result.runs[0]["metrics"]:
        metric_keys = [k for k in result.runs[0]["metrics"][0] if k != "epoch"]
        with open(out / "per_seed_metrics.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "epoch", *metric_keys])
            for run in result.runs:
                for m in run["metrics"]:
                    writer.writerow(
                        [run["seed"], m["epoch"], *(m[k] for k in metric_keys)]
                    )

        # Wide-format summary: one row per epoch, columns <metric>_<stat>
        stats = ("mean", "std", "ci95", "min", "max")
        with open(out / "summary.csv", "w", newline="") as f:
            writer = csv.writer(f)
            header = ["epoch"]
            for key in metric_keys:
                header.extend(f"{key}_{s}" for s in stats)
            writer.writerow(header)
            for epoch in sorted(result.summary):
                row: List[Any] = [epoch]
                for key in metric_keys:
                    agg = result.summary[epoch].get(key, {})
                    row.extend(agg.get(s, "") for s in stats)
                writer.writerow(row)

    logger.info("Exported sweep (%d seeds) to %s", len(result.seeds), out)
    return out


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Seed-sweep a GTB scenario")
    parser.add_argument("scenario", type=Path, help="Path to scenario YAML")
    parser.add_argument("--seeds", type=int, default=30,
                        help="Number of seeds (base-seed..base-seed+N-1)")
    parser.add_argument("--base-seed", type=int, default=1)
    parser.add_argument("--seed-list", type=str, default=None,
                        help="Comma-separated explicit seeds (overrides --seeds)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Dotted config override, e.g. "
                             "domain.misreporting.audit_probability=0.5")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)

    if args.seed_list:
        seeds = [int(s) for s in args.seed_list.split(",")]
    else:
        seeds = list(range(args.base_seed, args.base_seed + args.seeds))

    overrides: Dict[str, Any] = {}
    for item in args.set:
        key, _, raw = item.partition("=")
        overrides[key] = yaml.safe_load(raw)  # parse scalars: 0.5, true, "x"

    result = run_sweep(
        args.scenario, seeds, overrides=overrides,
        n_epochs=args.epochs, steps_per_epoch=args.steps,
        processes=args.processes,
    )

    output = args.output or (
        f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        f"_sweep_{result.scenario_id}_{len(seeds)}seeds"
    )
    out = export_sweep(result, output)

    last_epoch = max(result.summary)
    final = result.summary[last_epoch]
    print(f"Sweep complete: {len(seeds)} seeds -> {out}")
    print(f"Coin conserved in all runs: {result.all_conserved}")
    for key in ("welfare", "gini_coefficient", "total_production",
                "total_tax_revenue", "income_coin_gap"):
        if key in final:
            agg = final[key]
            print(f"  final-epoch {key}: {agg['mean']:.3f} ± {agg['ci95']:.3f} (95% CI)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
