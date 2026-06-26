#!/usr/bin/env python3
"""Seed-sweep harness for GTB scenarios.

Closes bd-cec (the keystone for bd-an2 / bd-mit / bd-rej): without
aggregation across many seeds, any single-run claim is anecdote
(welfare/Gini/production swung 50%+ across seeds in the PR #1 smokes).

Usage:
    # 50 seeds × default scenario, 20 epochs each, output to runs/baseline/
    python -m scripts.sweep_gtb \
        worlds/gather_trade_build/scenarios/ai_economist_full.yaml \
        --n-seeds 50 --epochs 20 --output runs/baseline

    # Sweep audit_probability across 7 cells × 20 seeds (the bd-an2 study)
    python -m scripts.sweep_gtb \
        worlds/gather_trade_build/scenarios/ai_economist_full.yaml \
        --n-seeds 20 --epochs 20 \
        --sweep 'misreporting.audit_probability=0.05,0.10,0.20,0.35,0.50,0.65,0.80' \
        --output runs/audit_sweep

    # Multi-axis sweep — cartesian product across both keys
    python -m scripts.sweep_gtb scenarios/baseline.yaml \
        --n-seeds 10 \
        --sweep 'misreporting.audit_probability=0.2,0.5' \
        --sweep 'taxation.smoothing=0,0.5,1.0'

Outputs (per sweep run, under --output):
    sweep.json               # the parameter grid + per-cell descriptors
    cells/cell_<i>/<seed>/   # per-(cell,seed) run dir with metrics + events
    aggregate.csv            # one row per (cell, epoch) with mean/median/p10/p90
    aggregate_final.csv      # one row per cell (final-epoch metrics only)

The kernel is pure-Python and parallelizes cleanly via multiprocessing.
50 seeds × 7 cells × 20 epochs target: under 10 min on a laptop.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from statistics import mean, median, quantiles
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


# Make `worlds.*` importable when running as `python scripts/sweep_gtb.py`
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from worlds.gather_trade_build.config import GTBConfig  # noqa: E402
from worlds.gather_trade_build.metrics import GTBMetrics  # noqa: E402
from worlds.gather_trade_build.runner import GTBScenarioRunner  # noqa: E402


logger = logging.getLogger("sweep_gtb")


_METRIC_FIELDS = (
    "total_production",
    "gini_coefficient",
    "welfare",
    "total_tax_revenue",
    "total_audits",
    "total_catches",
    "bunching_intensity",
    "enforcement_cost",
    "futures_volume",
    "futures_open_interest",
    "futures_basis",
)


@dataclass
class SweepCell:
    """One point in the parameter grid (a specific override combo)."""
    index: int
    overrides: Dict[str, Any]  # dotted-key -> value

    def label(self) -> str:
        if not self.overrides:
            return "baseline"
        parts = [f"{k.split('.')[-1]}={v}" for k, v in self.overrides.items()]
        return "_".join(parts)


def _set_dotted(data: Dict[str, Any], path: str, value: Any) -> None:
    """Set ``data[path[0]][path[1]]...`` = value, creating dicts as needed."""
    keys = path.split(".")
    cur = data
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _parse_sweep_arg(arg: str) -> Tuple[str, List[Any]]:
    """Parse 'key.path=v1,v2,v3' into ('key.path', [v1, v2, v3]) with
    type coercion (int -> float -> str fallback)."""
    key, _, rhs = arg.partition("=")
    if not key or not rhs:
        raise argparse.ArgumentTypeError(
            f"--sweep expects KEY=V1,V2,...; got {arg!r}"
        )
    raw_values = [v.strip() for v in rhs.split(",") if v.strip()]
    values: List[Any] = []
    for v in raw_values:
        try:
            values.append(int(v))
            continue
        except ValueError:
            pass
        try:
            values.append(float(v))
            continue
        except ValueError:
            pass
        if v.lower() in ("true", "false"):
            values.append(v.lower() == "true")
            continue
        values.append(v)
    return key.strip(), values


def _build_cells(sweep_args: List[str]) -> List[SweepCell]:
    """Cartesian product over the parsed sweep args."""
    if not sweep_args:
        return [SweepCell(index=0, overrides={})]
    axes: List[Tuple[str, List[Any]]] = [_parse_sweep_arg(a) for a in sweep_args]
    keys = [a[0] for a in axes]
    value_lists = [a[1] for a in axes]
    cells: List[SweepCell] = []
    for i, combo in enumerate(itertools.product(*value_lists)):
        cells.append(SweepCell(index=i, overrides=dict(zip(keys, combo))))
    return cells


@dataclass
class CellRunSpec:
    """One (cell, seed) job for the worker pool."""
    scenario_data: Dict[str, Any]
    cell_index: int
    cell_label: str
    cell_overrides: Dict[str, Any]
    seed: int
    n_epochs: int
    steps_per_epoch: int
    output_root: str
    cells_mode: str = "targz"  # targz | dir | none


def _compress_seed_dir(out_dir: Path) -> None:
    """Replace a per-seed export directory with a single ``.tar.gz``.

    A large sweep writes thousands of tiny files (each seed dir holds an
    event_log.jsonl plus a few CSV/JSON files); left as loose files they
    burn disk blocks and inodes out of proportion to their content. The
    event log is highly repetitive JSON and compresses ~10-20x. The
    archive lands next to the dir as ``seed_<n>.tar.gz`` and the original
    tree is removed.
    """
    import tarfile

    archive = out_dir.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(out_dir, arcname=out_dir.name)
    shutil.rmtree(out_dir)


def _run_one(spec: CellRunSpec) -> Dict[str, Any]:
    """Run one (cell, seed); return summary dict for aggregation.

    Per-cell-seed metrics written to disk; only summary returned to
    parent so the IPC payload stays small.
    """
    data = json.loads(json.dumps(spec.scenario_data))  # cheap deep-copy
    for path, value in spec.cell_overrides.items():
        _set_dotted(data.setdefault("domain", {}), path, value)

    config = GTBConfig.from_dict(data.get("domain", {}))
    config.seed = spec.seed
    agent_specs = data.get("agents", [{"policy": "honest", "count": 5}])

    runner = GTBScenarioRunner(
        config=config,
        agent_specs=agent_specs,
        n_epochs=spec.n_epochs,
        steps_per_epoch=spec.steps_per_epoch,
        seed=spec.seed,
    )
    t0 = time.time()
    metrics_list = runner.run()
    elapsed = time.time() - t0

    # Export per-(cell, seed) run dir for downstream notebooks. The
    # aggregates are built from the returned summary below, not by
    # re-reading these dirs, so the dump is optional: 'none' skips it
    # entirely, 'targz' (default) compresses it to bound disk + inode use.
    if spec.cells_mode != "none":
        out_dir = Path(spec.output_root) / "cells" / f"cell_{spec.cell_index:03d}" / f"seed_{spec.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        runner.export(str(out_dir))
        if spec.cells_mode == "targz":
            _compress_seed_dir(out_dir)

    final = metrics_list[-1]
    return {
        "cell_index": spec.cell_index,
        "cell_label": spec.cell_label,
        "cell_overrides": spec.cell_overrides,
        "seed": spec.seed,
        "elapsed_s": elapsed,
        "per_epoch": [_metrics_to_dict(m) for m in metrics_list],
        "final": _metrics_to_dict(final),
    }


def _metrics_to_dict(m: GTBMetrics) -> Dict[str, float]:
    return {
        "epoch": m.epoch,
        **{f: float(getattr(m, f)) for f in _METRIC_FIELDS if hasattr(m, f)},
    }


def _percentile(vs: Iterable[float], p: float) -> float:
    """p in [0,1]. Returns NaN-safe percentile via quantiles()."""
    arr = [v for v in vs if v is not None]
    if not arr:
        return float("nan")
    if len(arr) == 1:
        return arr[0]
    # quantiles with n=100 gives percentiles
    qs = quantiles(arr, n=100, method="inclusive")
    # qs has 99 cutpoints between p1..p99
    idx = max(0, min(98, int(round(p * 100)) - 1))
    return qs[idx]


def _aggregate(results: List[Dict[str, Any]], cells: List[SweepCell]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (per_epoch_rows, final_rows).

    per_epoch_rows: one row per (cell, epoch) with mean/median/p10/p90
                   across seeds for each metric field.
    final_rows: one row per cell, same stats on the final-epoch metrics.
    """
    # Group by cell_index
    by_cell: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        by_cell.setdefault(r["cell_index"], []).append(r)

    per_epoch_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []

    for cell in cells:
        runs = by_cell.get(cell.index, [])
        if not runs:
            continue
        # Determine n_epochs from first run's per_epoch length
        n_epochs = len(runs[0]["per_epoch"])
        for epoch in range(n_epochs):
            base = {
                "cell_index": cell.index,
                "cell_label": cell.label(),
                "n_seeds": len(runs),
                "epoch": epoch,
            }
            base.update({f"override_{k}": v for k, v in cell.overrides.items()})
            for field in _METRIC_FIELDS:
                values = [r["per_epoch"][epoch].get(field) for r in runs]
                values = [v for v in values if v is not None]
                if not values:
                    continue
                base[f"{field}_mean"] = mean(values)
                base[f"{field}_median"] = median(values)
                base[f"{field}_p10"] = _percentile(values, 0.10)
                base[f"{field}_p90"] = _percentile(values, 0.90)
            per_epoch_rows.append(base)

        # Final-epoch summary row
        final_base = {
            "cell_index": cell.index,
            "cell_label": cell.label(),
            "n_seeds": len(runs),
        }
        final_base.update({f"override_{k}": v for k, v in cell.overrides.items()})
        for field in _METRIC_FIELDS:
            values = [r["final"].get(field) for r in runs]
            values = [v for v in values if v is not None]
            if not values:
                continue
            final_base[f"{field}_mean"] = mean(values)
            final_base[f"{field}_median"] = median(values)
            final_base[f"{field}_p10"] = _percentile(values, 0.10)
            final_base[f"{field}_p90"] = _percentile(values, 0.90)
        final_rows.append(final_base)

    return per_epoch_rows, final_rows


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    # Union of keys preserves order across rows; sort for deterministic output.
    all_keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a (seed × parameter) sweep of a GTB scenario and aggregate metrics.",
    )
    parser.add_argument("scenario", type=Path, help="Path to scenario YAML")
    parser.add_argument(
        "--n-seeds", type=int, default=10,
        help="Number of seeds per cell (default 10). Mutually exclusive with --seeds.",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated explicit seed list (overrides --n-seeds)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override n_epochs (default: from scenario YAML)",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override steps_per_epoch (default: from scenario YAML)",
    )
    parser.add_argument(
        "--sweep", action="append", default=[],
        help="Sweep axis: KEY.PATH=V1,V2,V3. Repeatable. Cartesian product across axes. "
             "All keys are interpreted under the scenario's `domain:` block.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/sweep"),
        help="Output directory root (default: runs/sweep)",
    )
    parser.add_argument(
        "--cells", choices=("targz", "dir", "none"), default="targz",
        help="Per-(cell, seed) dump format: 'targz' (default, one .tar.gz "
             "per seed — compact), 'dir' (loose files), 'none' (skip; keep "
             "only the aggregate CSVs).",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Parallel worker count (default: cpu_count - 1; pass 1 for serial debugging)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    with open(args.scenario) as f:
        scenario_data = yaml.safe_load(f)
    sim = scenario_data.get("simulation", {})
    n_epochs = args.epochs or sim.get("n_epochs", 10)
    steps_per_epoch = args.steps or sim.get("steps_per_epoch", 10)

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = list(range(args.n_seeds))

    cells = _build_cells(args.sweep)
    total = len(cells) * len(seeds)
    logger.info(
        "sweep plan: %d cell(s) × %d seed(s) = %d runs · %d epochs × %d steps each",
        len(cells), len(seeds), total, n_epochs, steps_per_epoch,
    )

    # Build the job list.
    jobs: List[CellRunSpec] = []
    for cell in cells:
        for seed in seeds:
            jobs.append(CellRunSpec(
                scenario_data=scenario_data,
                cell_index=cell.index,
                cell_label=cell.label(),
                cell_overrides=cell.overrides,
                seed=seed,
                n_epochs=n_epochs,
                steps_per_epoch=steps_per_epoch,
                output_root=str(args.output),
                cells_mode=args.cells,
            ))

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "sweep.json", "w") as f:
        json.dump({
            "scenario": str(args.scenario),
            "n_epochs": n_epochs,
            "steps_per_epoch": steps_per_epoch,
            "seeds": seeds,
            "cells": [
                {"index": c.index, "label": c.label(), "overrides": c.overrides}
                for c in cells
            ],
        }, f, indent=2)

    t0 = time.time()
    if args.workers == 1:
        # Serial mode for stack traces; otherwise multiprocessing.
        results = [_run_one(j) for j in jobs]
    else:
        from multiprocessing import cpu_count
        workers = args.workers or max(1, cpu_count() - 1)
        logger.info("running %d jobs across %d worker process(es)", len(jobs), workers)
        with Pool(processes=workers) as pool:
            results = pool.map(_run_one, jobs)
    elapsed = time.time() - t0
    logger.info("ran %d jobs in %.1fs (%.2fs/job)", len(jobs), elapsed, elapsed / max(1, len(jobs)))

    per_epoch_rows, final_rows = _aggregate(results, cells)
    _write_csv(args.output / "aggregate.csv", per_epoch_rows)
    _write_csv(args.output / "aggregate_final.csv", final_rows)
    logger.info(
        "wrote %d per-(cell, epoch) rows -> %s and %d final-epoch rows -> %s",
        len(per_epoch_rows), args.output / "aggregate.csv",
        len(final_rows), args.output / "aggregate_final.csv",
    )


if __name__ == "__main__":
    main()
