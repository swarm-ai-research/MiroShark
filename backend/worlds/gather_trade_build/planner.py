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
        # Honor the configured learning rate so the planner-reactivity
        # sweep's `planner.learning_rate` overrides take effect.
        self._rl_lr: float = self._config.learning_rate
        # Per-weight update magnitude cap. With sigma²=0.0025 and raw
        # feature/reward scales on the order of tens, an uncapped step
        # routinely pushes a weight by hundreds, collapsing the next
        # mu_i clip to 0 or 1 and stalling learning.
        self._rl_update_clip: float = 0.05
        self._rl_prev_features: Optional[List[float]] = None
        self._rl_prev_sample: Optional[List[float]] = None

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
        elif self._config.planner_type == "rl":
            return self._rl_update(stats)
        else:
            logger.info("Unknown planner_type %s: keeping current schedule",
                        self._config.planner_type)
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
        z_star = stats.get("top_threshold", current[-1].threshold)
        zm = stats.get("top_mean_income", 0.0)

        # Target the highest bracket the top tail actually occupies — the
        # bracket whose threshold sits at or below z* (the income cutoff
        # defining the top group). On economies whose configured top
        # bracket sits above the realized income scale that bracket is
        # empty, so optimizing its rate is welfare-inert; the live lever is
        # the highest *populated* bracket (bd-anv/bd-kk5). z*/zm are made
        # data-driven in env._aggregate_stats so this index is meaningful
        # even when no worker reaches the configured top bracket.
        top_idx = 0
        for idx in range(len(current)):
            if current[idx].threshold <= z_star:
                top_idx = idx
        top = current[top_idx]
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

        # Welfare-weighted inverse-elasticity rule for the top rate:
        #   tau* = (1 - g) / (1 - g + a*e)
        # g is the marginal social welfare weight on the top tail (see
        # env._aggregate_stats). g=0 recovers the revenue-maximizing
        # Saez rate 1/(1 + a*e); g>0 lowers the optimal rate so the
        # planner stops over-taxing for revenue the welfare objective
        # doesn't reward. Absent (e.g. unit tests injecting bare stats) ->
        # g=0, preserving the classic formula. (bd-5gz)
        g = max(0.0, min(0.99, stats.get("top_welfare_weight", 0.0)))
        if zm > z_star > 0:
            pareto_a = zm / (zm - z_star) if zm > z_star else 2.0
        else:
            pareto_a = 2.0  # thin/no top tail observed: conservative default
        tau_star = (1.0 - g) / ((1.0 - g) + pareto_a * self._elasticity)

        cap = self._config.saez_rate_change_cap
        delta = max(-cap, min(cap, tau_star - top.rate))
        new_top_rate = max(0.0, min(1.0, top.rate + delta))

        # Set the optimized rate on the targeted bracket; brackets below it
        # are untouched (standard top-rate result).
        new_brackets = [TaxBracket(threshold=b.threshold, rate=b.rate)
                        for b in current]
        new_brackets[top_idx] = TaxBracket(threshold=top.threshold,
                                           rate=new_top_rate)

        if not self._tax_schedule.allow_non_monotone:
            # Floor the targeted bracket against the one below it.
            if top_idx > 0 and new_brackets[top_idx].rate < new_brackets[top_idx - 1].rate:
                new_top_rate = new_brackets[top_idx - 1].rate
                new_brackets[top_idx] = TaxBracket(
                    threshold=top.threshold, rate=new_top_rate,
                )
            # Carry the top rate up through any (empty) higher brackets so
            # the schedule stays monotone — a single top rate above z*.
            for k in range(top_idx + 1, len(new_brackets)):
                if new_brackets[k].rate < new_brackets[k - 1].rate:
                    new_brackets[k] = TaxBracket(
                        threshold=new_brackets[k].threshold,
                        rate=new_brackets[k - 1].rate,
                    )

        self._tax_schedule.update_brackets(new_brackets)
        logger.debug(
            "Saez update: idx=%d e=%.3f a=%.3f tau*=%.3f -> rate %.3f",
            top_idx, self._elasticity, pareto_a, tau_star, new_top_rate,
        )
        return self._tax_schedule.brackets

    @property
    def elasticity_estimate(self) -> float:
        return self._elasticity

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
                    delta = self._rl_lr * grad_factor * s[k] * reward
                    if delta > self._rl_update_clip:
                        delta = self._rl_update_clip
                    elif delta < -self._rl_update_clip:
                        delta = -self._rl_update_clip
                    self._rl_weights[i][k] += delta

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
