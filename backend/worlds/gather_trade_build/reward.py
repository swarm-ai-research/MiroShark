"""Reward / utility computation for GTB workers."""

from __future__ import annotations

import math

from worlds.gather_trade_build.config import UtilityConfig
from worlds.gather_trade_build.entities import ResourceType, WorkerState


def compute_worker_utility(
    worker: WorkerState,
    *,
    coin_weight: float = 1.0,
    wood_weight: float = 0.5,
    stone_weight: float = 0.5,
    house_weight: float = 5.0,
) -> float:
    """Compute a worker's utility from current holdings.

    utility = coin_weight * coin + wood_weight * wood + stone_weight * stone
              + house_weight * houses_built
    """
    return (
        coin_weight * worker.get_resource(ResourceType.COIN)
        + wood_weight * worker.get_resource(ResourceType.WOOD)
        + stone_weight * worker.get_resource(ResourceType.STONE)
        + house_weight * worker.houses_built
    )


def crra(x: float, eta: float) -> float:
    """Isoelastic (CRRA) utility of consumption x.

    eta=0 is linear; eta -> 1 approaches log. Defined for x >= 0
    (clamped at a small epsilon to avoid singularities at zero).
    """
    x = max(x, 1e-9)
    if abs(eta - 1.0) < 1e-9:
        return math.log(x)
    return (x ** (1.0 - eta) - 1.0) / (1.0 - eta)


def crra_marginal(x: float, eta: float) -> float:
    """Marginal utility u'(x) = x^-eta for the CRRA function."""
    return max(x, 1e-9) ** (-eta)


def compute_isoelastic_utility(
    worker: WorkerState,
    cfg: UtilityConfig,
) -> float:
    """AI Economist-style worker utility: CRRA coin utility plus house
    utility minus labor disutility on cumulative effort.

    u = crra(coin; eta) + house_weight * houses - labor_coeff * effort
    """
    return (
        crra(worker.get_resource(ResourceType.COIN), cfg.eta)
        + cfg.house_weight * worker.houses_built
        - cfg.labor_coeff * worker.cumulative_effort
    )


def compute_epoch_reward(
    worker: WorkerState,
    tax_paid: float,
    fine_paid: float,
    *,
    coin_weight: float = 1.0,
    house_weight: float = 5.0,
) -> float:
    """Compute epoch-level reward signal for RL-style policies.

    reward = coin_weight * net_income + house_weight * houses - fine_paid
    """
    net = worker.gross_income_this_epoch - tax_paid - fine_paid
    return coin_weight * net + house_weight * worker.houses_built
