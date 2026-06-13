"""Planner agent for the bilevel Planner-Workers loop.

The planner observes aggregate epoch statistics and updates the tax schedule
on a configurable cadence. Supports heuristic, bandit, and (stub) RL planners.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

from worlds.gather_trade_build.config import PlannerConfig, TaxBracket
from worlds.gather_trade_build.tax_schedule import TaxSchedule

logger = logging.getLogger(__name__)


class PlannerAgent:
    """Bilevel planner that updates tax policy each epoch.

    Planner types:
      - heuristic: rule-based adjustments targeting a welfare objective
      - bandit: epsilon-greedy over rate perturbations
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

        # RL state (REINFORCE-style online policy gradient).
        # Linear policy parameters: per-bracket bias + per-stat weights.
        # State features: [mean_income, gini, n_frozen, 1.0 bias]
        # Action: per-bracket mean rate, sampled Normal(mean, sigma).
        self._rl_features = ("mean_income", "gini", "n_frozen")
        n_brackets = len(self._tax_schedule.brackets)
        n_features = len(self._rl_features) + 1
        # Initialize to small noise around current rates.
        init_rates = [b.rate for b in self._tax_schedule.brackets]
        self._rl_weights: List[List[float]] = [
            [0.0] * n_features for _ in range(n_brackets)
        ]
        # Bias term in the last slot initialized to current rate so the
        # initial policy matches the configured schedule.
        for i in range(n_brackets):
            self._rl_weights[i][-1] = init_rates[i]
        self._rl_sigma: float = 0.05
        self._rl_baseline: float = 0.0
        self._rl_baseline_alpha: float = 0.1
        self._rl_lr: float = 0.02
        self._rl_prev_features: Optional[List[float]] = None
        self._rl_prev_sample: Optional[List[float]] = None

    def should_update(self, epoch: int) -> bool:
        """Whether the planner should update this epoch."""
        return epoch > 0 and epoch % self._config.update_interval_epochs == 0

    def update(self, stats: Dict[str, float]) -> List[TaxBracket]:
        """Observe aggregate stats and return new tax brackets.

        Args:
            stats: Aggregate stats from GTBEnvironment.get_aggregate_stats().
                   Expected keys: total_income, mean_income, gini,
                   total_tax_revenue, total_houses, n_workers.

        Returns:
            New list of TaxBracket to apply.
        """
        self._epoch_count += 1

        if self._config.planner_type == "heuristic":
            return self._heuristic_update(stats)
        elif self._config.planner_type == "bandit":
            return self._bandit_update(stats)
        elif self._config.planner_type == "rl":
            return self._rl_update(stats)
        else:
            logger.info("Unknown planner_type %s: keeping current schedule",
                        self._config.planner_type)
            return self._tax_schedule.brackets

    def _compute_welfare(self, stats: Dict[str, float]) -> float:
        """Compute welfare objective from stats."""
        prod = stats.get("mean_income", 0.0)
        gini = stats.get("gini", 0.0)
        return self._config.prod_weight * prod - self._config.ineq_weight * gini

    def _heuristic_update(self, stats: Dict[str, float]) -> List[TaxBracket]:
        """Rule-based planner: increase rates if inequality is high, decrease if productivity is low."""
        gini = stats.get("gini", 0.0)
        mean_income = stats.get("mean_income", 0.0)
        lr = self._config.learning_rate

        current = self._tax_schedule.brackets
        new_brackets = []

        for bracket in current:
            # If inequality is high, raise rates (especially upper brackets)
            # If productivity is low, lower rates
            gini_signal = (gini - 0.3) * 2.0  # positive when gini > 0.3
            prod_signal = -(max(0, 5.0 - mean_income) / 5.0)  # negative when low prod

            adjustment = lr * (gini_signal + prod_signal)
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

    def _rl_update(self, stats: Dict[str, float]) -> List[TaxBracket]:
        """Online REINFORCE planner.

        Closes bd-i8o. Treats bracket rates as a continuous action
        sampled per-epoch from a per-bracket Gaussian whose mean is a
        linear function of (mean_income, gini, n_frozen, 1.0). Rewards
        the prior epoch's action with the observed welfare and updates
        the weights via the policy-gradient theorem:
            ∇log π(a|s) × (R − baseline)
        where the baseline is an exponential moving average of recent
        welfare to reduce variance.

        Not a "trained" policy — the weights adapt online during the
        run. Over enough epochs the planner should converge toward a
        bracket set that maximizes welfare under the current workforce.
        """
        # Compute reward = welfare just realized, with baseline subtraction.
        welfare = self._compute_welfare(stats)
        reward = welfare - self._rl_baseline
        self._rl_baseline = (
            (1 - self._rl_baseline_alpha) * self._rl_baseline
            + self._rl_baseline_alpha * welfare
        )

        # If we have a stored (s, a) from the previous epoch, apply the
        # policy-gradient update before sampling the next action.
        if self._rl_prev_features is not None and self._rl_prev_sample is not None:
            s = self._rl_prev_features
            a_sample = self._rl_prev_sample
            # ∇log π(a|s) = (a - μ(s)) / σ² × s   for each bracket dimension
            for i, sample in enumerate(a_sample):
                mu_i = sum(self._rl_weights[i][k] * s[k] for k in range(len(s)))
                grad_factor = (sample - mu_i) / (self._rl_sigma ** 2)
                for k in range(len(s)):
                    self._rl_weights[i][k] += (
                        self._rl_lr * grad_factor * s[k] * reward
                    )

        # Build feature vector for current state.
        s_now = [stats.get(f, 0.0) for f in self._rl_features] + [1.0]

        # Sample new rates from per-bracket Gaussians.
        new_brackets = []
        new_sample = []
        for i, b in enumerate(self._tax_schedule.brackets):
            mu_i = sum(self._rl_weights[i][k] * s_now[k] for k in range(len(s_now)))
            mu_i = max(0.0, min(1.0, mu_i))
            sample = self._rng.gauss(mu_i, self._rl_sigma)
            sample = max(0.0, min(1.0, sample))
            new_sample.append(sample)
            new_brackets.append(TaxBracket(threshold=b.threshold, rate=sample))

        # Enforce monotonicity if required by the schedule.
        if not self._tax_schedule._config.allow_non_monotone:
            for i in range(1, len(new_brackets)):
                if new_brackets[i].rate < new_brackets[i - 1].rate:
                    new_brackets[i] = TaxBracket(
                        threshold=new_brackets[i].threshold,
                        rate=new_brackets[i - 1].rate,
                    )
                    new_sample[i] = new_sample[i - 1]

        self._tax_schedule.update_brackets(new_brackets)
        self._rl_prev_features = s_now
        self._rl_prev_sample = new_sample
        return self._tax_schedule.brackets

    @property
    def tax_schedule(self) -> TaxSchedule:
        return self._tax_schedule
