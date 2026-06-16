"""Planner agent for the bilevel Planner-Workers loop.

The planner observes aggregate epoch statistics and updates the tax schedule
on a configurable cadence. Supports heuristic, bandit, Saez, and (stub) RL
planners.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Dict, List, Optional

from worlds.gather_trade_build.config import PlannerConfig, TaxBracket
from worlds.gather_trade_build.tax_schedule import TaxSchedule

logger = logging.getLogger(__name__)


class PlannerAgent:
    """Bilevel planner that updates tax policy each epoch.

    Planner types:
      - heuristic: rule-based per-bracket adjustments targeting a welfare
        objective (top brackets react to inequality, bottom to productivity)
      - bandit: epsilon-greedy over rate perturbations
      - saez: top-rate inverse-elasticity rule with the taxable-income
        elasticity estimated online from observed responses
      - rl: placeholder for future RL training
    """

    def __init__(
        self,
        config: PlannerConfig,
        tax_schedule: TaxSchedule,
        seed: Optional[int] = None,
    ) -> None:
        self._config = config
        self._tax_schedule = tax_schedule
        self._rng = random.Random(seed)
        self._epoch_count = 0

        # Bandit state
        self._prev_welfare: Optional[float] = None
        self._prev_action: Optional[List[TaxBracket]] = None

        # Saez state: online elasticity estimate via EMA over observed
        # (top income, net-of-tax rate) changes
        self._elasticity = config.saez_elasticity_init
        self._prev_top_income: Optional[float] = None
        self._prev_net_of_tax: Optional[float] = None

    def should_update(self, epoch: int) -> bool:
        """Whether the planner should update this epoch."""
        return epoch > 0 and epoch % self._config.update_interval_epochs == 0

    def update(self, stats: Dict[str, float]) -> List[TaxBracket]:
        """Observe aggregate stats and return new tax brackets.

        Args:
            stats: Aggregate stats from GTBEnvironment (see
                   stats_from_snapshot). Expected keys: total_income,
                   mean_income, gini, gini_wealth, mean_utility,
                   total_tax_revenue, total_houses, n_workers,
                   top_threshold, top_mean_income.

        Returns:
            New list of TaxBracket to apply.
        """
        self._epoch_count += 1

        if self._config.planner_type == "heuristic":
            return self._heuristic_update(stats)
        elif self._config.planner_type == "bandit":
            return self._bandit_update(stats)
        elif self._config.planner_type == "saez":
            return self._saez_update(stats)
        else:
            # RL stub: no-op, keep current schedule
            logger.info("RL planner stub: keeping current schedule")
            return self._tax_schedule.brackets

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _gini(self, stats: Dict[str, float]) -> float:
        """The inequality measure the planner is configured to react to."""
        if self._config.inequality_measure == "wealth":
            return stats.get("gini_wealth", stats.get("gini", 0.0))
        return stats.get("gini", 0.0)

    def _compute_welfare(self, stats: Dict[str, float]) -> float:
        """Compute the planner's welfare objective from stats.

        Objectives:
          - welfare (legacy): prod_weight * mean_income - ineq_weight * gini.
            Kept for backward compatibility; note the unit mismatch (income
            in coins vs Gini in [0,1]) makes the inequality term weak.
          - eq_times_prod: mean_income * (1 - gini) — the AI Economist's
            productivity-times-equality objective.
          - utilitarian: mean isoelastic worker utility.
        """
        objective = self._config.objective
        if objective == "eq_times_prod":
            return stats.get("mean_income", 0.0) * (1.0 - self._gini(stats))
        if objective == "utilitarian":
            return stats.get("mean_utility", 0.0)
        prod = stats.get("mean_income", 0.0)
        return (
            self._config.prod_weight * prod
            - self._config.ineq_weight * self._gini(stats)
        )

    # ------------------------------------------------------------------
    # Planner types
    # ------------------------------------------------------------------

    def _heuristic_update(self, stats: Dict[str, float]) -> List[TaxBracket]:
        """Rule-based planner with per-bracket response.

        Top brackets respond to inequality (raise rates when Gini is
        high); bottom brackets respond to productivity (lower rates when
        production is weak). The legacy version moved every bracket by
        the same delta, which made the schedule's *shape* unplannable.
        """
        gini = self._gini(stats)
        mean_income = stats.get("mean_income", 0.0)
        lr = self._config.learning_rate

        current = self._tax_schedule.brackets
        n_b = len(current)
        new_brackets = []

        gini_signal = (gini - 0.3) * 2.0  # positive when gini > 0.3
        prod_signal = -(max(0, 5.0 - mean_income) / 5.0)  # negative when low prod

        for i, bracket in enumerate(current):
            top_weight = i / max(n_b - 1, 1)  # 0 at bottom bracket, 1 at top
            adjustment = lr * (
                gini_signal * top_weight + prod_signal * (1.0 - top_weight)
            )
            new_rate = max(0.0, min(1.0, bracket.rate + adjustment))
            new_brackets.append(TaxBracket(
                threshold=bracket.threshold, rate=new_rate,
            ))

        self._tax_schedule.update_brackets(new_brackets)
        return self._tax_schedule.brackets

    def _bandit_update(self, stats: Dict[str, float]) -> List[TaxBracket]:
        """Epsilon-greedy bandit planner over rate perturbations."""
        welfare = self._compute_welfare(stats)
        current = self._tax_schedule.brackets

        # If previous action improved welfare, keep direction; otherwise reverse
        if self._prev_welfare is not None and self._prev_action is not None:
            if welfare < self._prev_welfare:
                # Revert to previous
                current = self._prev_action

        self._prev_welfare = welfare
        self._prev_action = list(current)

        # Epsilon-greedy: with probability epsilon, try random perturbation
        if self._rng.random() < self._config.exploration_rate:
            new_brackets = []
            for bracket in current:
                delta = self._rng.gauss(0, self._config.learning_rate)
                new_rate = max(0.0, min(1.0, bracket.rate + delta))
                new_brackets.append(TaxBracket(
                    threshold=bracket.threshold, rate=new_rate,
                ))
        else:
            new_brackets = list(current)

        self._tax_schedule.update_brackets(new_brackets)
        return self._tax_schedule.brackets

    def _saez_update(self, stats: Dict[str, float]) -> List[TaxBracket]:
        """Saez top-rate planner: tau* = 1 / (1 + a * e).

        a is the Pareto parameter of the top tail, estimated as
        zm / (zm - z*) from the mean top income zm above the top
        threshold z*. The taxable-income elasticity e is estimated
        online: e = dlog(zm) / dlog(1 - tau) between updates, EMA-
        smoothed and clamped to a sane range. The top-rate move per
        update is capped (saez_rate_change_cap) so a noisy elasticity
        estimate cannot whipsaw the schedule.

        Lower brackets are left unchanged: the inverse-elasticity rule
        as implemented here is the standard top-bracket result, not a
        full Mirrlees schedule.
        """
        current = self._tax_schedule.brackets
        if not current:
            return current
        top = current[-1]
        z_star = stats.get("top_threshold", top.threshold)
        zm = stats.get("top_mean_income", 0.0)
        net_of_tax = max(1e-6, 1.0 - top.rate)

        # Online elasticity estimate from the observed response
        if (
            self._prev_top_income is not None
            and self._prev_net_of_tax is not None
            and self._prev_top_income > 1e-9
            and zm > 1e-9
        ):
            dlog_net = math.log(net_of_tax) - math.log(self._prev_net_of_tax)
            if abs(dlog_net) > 1e-6:
                e_obs = (
                    (math.log(zm) - math.log(self._prev_top_income)) / dlog_net
                )
                e_obs = max(0.05, min(2.0, e_obs))
                w = self._config.saez_elasticity_lr
                self._elasticity = (1.0 - w) * self._elasticity + w * e_obs
        self._prev_top_income = zm if zm > 0 else self._prev_top_income
        self._prev_net_of_tax = net_of_tax

        # Inverse-elasticity rule for the top rate
        if zm > z_star > 0:
            pareto_a = zm / (zm - z_star) if zm > z_star else 2.0
        else:
            pareto_a = 2.0  # thin/no top tail observed: conservative default
        tau_star = 1.0 / (1.0 + pareto_a * self._elasticity)

        cap = self._config.saez_rate_change_cap
        delta = max(-cap, min(cap, tau_star - top.rate))
        new_top_rate = max(0.0, min(1.0, top.rate + delta))

        new_brackets = [
            TaxBracket(threshold=b.threshold, rate=b.rate) for b in current[:-1]
        ]
        new_brackets.append(TaxBracket(threshold=top.threshold, rate=new_top_rate))

        # A capped move toward tau* can dip the top rate below the bracket
        # under it while non-monotone schedules are disallowed; floor it.
        if not self._tax_schedule.allow_non_monotone and len(new_brackets) > 1:
            floor = new_brackets[-2].rate
            if new_brackets[-1].rate < floor:
                new_brackets[-1] = TaxBracket(
                    threshold=new_brackets[-1].threshold, rate=floor,
                )

        self._tax_schedule.update_brackets(new_brackets)
        logger.debug(
            "Saez update: e=%.3f a=%.3f tau*=%.3f -> top rate %.3f",
            self._elasticity, pareto_a, tau_star, new_top_rate,
        )
        return self._tax_schedule.brackets

    @property
    def elasticity_estimate(self) -> float:
        return self._elasticity

    @property
    def tax_schedule(self) -> TaxSchedule:
        return self._tax_schedule
