"""Auto-derived YES/NO prediction markets on GTB metrics.

Phase 4: forward-looking markets the world generates from its own state.
A market is a binary question — "Will <metric> <op> <threshold> by epoch N?"
— resolved when (a) the threshold is crossed before the deadline (YES /
NO depending on the op direction) or (b) the deadline passes without
crossing (resolves against the staked direction).

Pure derivation, no LLM. Deterministic given the metrics stream, so the
same run resolves the same markets the same way.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


Op = Literal[">", "<", ">=", "<="]
Status = Literal["open", "yes", "no", "expired"]

_OP_HUMAN = {">": "exceed", "<": "fall below", ">=": "reach", "<=": "drop to"}

# Metrics the generator knows how to read off a GTBMetrics dict.
_METRIC_KEYS = {
    "welfare": "social welfare",
    "gini_coefficient": "Gini coefficient",
    "total_production": "total production",
    "total_tax_revenue": "total tax revenue",
    "total_audits": "audit count",
    "total_catches": "tax-evader catches",
    "bunching_intensity": "income bunching intensity",
}


@dataclass
class Market:
    market_id: str
    question: str
    metric: str
    op: Op
    threshold: float
    deadline_epoch: int
    created_epoch: int
    status: Status = "open"
    resolved_epoch: Optional[int] = None
    resolved_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _check(value: float, op: Op, threshold: float) -> bool:
    if op == ">":
        return value > threshold
    if op == "<":
        return value < threshold
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    return False


def _phrase(metric: str, op: Op, threshold: float, deadline: int) -> str:
    human = _METRIC_KEYS.get(metric, metric)
    verb = _OP_HUMAN.get(op, op)
    return f"Will {human} {verb} {threshold:.3g} by end of epoch {deadline}?"


class GTBMarketGenerator:
    """Derive a set of binary markets from the current metrics snapshot.

    Strategy: for each known metric, propose a small set of thresholds
    around the most recent value — one upside, one downside — with a
    deadline a few epochs out. Deterministic per (metrics, horizon).
    """

    def __init__(self, horizon_epochs: int = 5) -> None:
        self._horizon = horizon_epochs

    def generate(
        self,
        current_epoch: int,
        latest_metrics: Optional[Dict[str, Any]],
        existing_questions: Optional[set] = None,
        start_index: int = 0,
    ) -> List[Market]:
        if not latest_metrics:
            return []
        existing = existing_questions or set()
        out: List[Market] = []
        idx = start_index
        deadline = current_epoch + self._horizon
        for metric in _METRIC_KEYS:
            val = latest_metrics.get(metric)
            if val is None:
                continue
            for op, threshold in _threshold_pair(metric, float(val)):
                q = _phrase(metric, op, threshold, deadline)
                if q in existing:
                    continue
                out.append(
                    Market(
                        market_id=f"gtb-{idx:04d}",
                        question=q,
                        metric=metric,
                        op=op,
                        threshold=threshold,
                        deadline_epoch=deadline,
                        created_epoch=current_epoch,
                    )
                )
                idx += 1
                existing.add(q)
        return out


def _threshold_pair(metric: str, value: float):
    """Pick (op, threshold) tuples for upside + downside coverage."""
    if metric == "gini_coefficient":
        # Bounded [0, 1]; +/- 0.1 step, clamped.
        up = min(0.95, round(value + 0.10, 2))
        down = max(0.05, round(value - 0.10, 2))
    elif metric == "bunching_intensity":
        up = min(0.95, round(value + 0.10, 2))
        down = max(0.0, round(value - 0.10, 2))
    elif metric in ("total_audits", "total_catches"):
        up = int(value + max(2, value * 0.25))
        down = max(0, int(value - max(1, value * 0.25)))
    else:
        # production / welfare / tax revenue: ±20% bands.
        up = round(value * 1.2, 2) if value > 0 else 1.0
        down = round(value * 0.8, 2) if value > 0 else 0.0
    return [(">", up), ("<", down)]


class GTBMarketBook:
    """Open + resolved markets for one world."""

    def __init__(self, generator: Optional[GTBMarketGenerator] = None) -> None:
        self._generator = generator or GTBMarketGenerator()
        self._open: List[Market] = []
        self._resolved: List[Market] = []
        self._next_idx: int = 0

    @property
    def open_markets(self) -> List[Market]:
        return list(self._open)

    @property
    def resolved_markets(self) -> List[Market]:
        return list(self._resolved)

    def generate(
        self,
        current_epoch: int,
        latest_metrics: Optional[Dict[str, Any]],
    ) -> List[Market]:
        existing = {m.question for m in self._open}
        new = self._generator.generate(
            current_epoch=current_epoch,
            latest_metrics=latest_metrics,
            existing_questions=existing,
            start_index=self._next_idx,
        )
        self._open.extend(new)
        self._next_idx += len(new)
        return new

    def on_epoch_close(
        self, epoch: int, latest_metrics: Dict[str, Any]
    ) -> List[Market]:
        """Resolve any open market whose threshold has been crossed or
        whose deadline has passed. Returns the newly resolved set."""
        resolved_now: List[Market] = []
        still_open: List[Market] = []
        for m in self._open:
            v = latest_metrics.get(m.metric)
            if v is not None and _check(float(v), m.op, m.threshold):
                m.status = "yes"
                m.resolved_epoch = epoch
                m.resolved_value = float(v)
                resolved_now.append(m)
                continue
            if epoch >= m.deadline_epoch:
                # Deadline passed without crossing.
                m.status = "no" if v is not None else "expired"
                m.resolved_epoch = epoch
                m.resolved_value = float(v) if v is not None else None
                resolved_now.append(m)
                continue
            still_open.append(m)
        self._open = still_open
        self._resolved.extend(resolved_now)
        if resolved_now:
            logger.info(
                "GTB markets resolved at epoch %d: %s",
                epoch,
                [(m.market_id, m.status) for m in resolved_now],
            )
        return resolved_now

    def to_dict(self) -> Dict[str, Any]:
        return {
            "open": [m.to_dict() for m in self._open],
            "resolved": [m.to_dict() for m in self._resolved],
        }

    def find_open(self, market_id: str) -> Optional[Market]:
        for m in self._open:
            if m.market_id == market_id:
                return m
        return None


@dataclass
class Stake:
    agent_id: str
    market_id: str
    side: str  # "yes" | "no"
    amount: float
    epoch: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GTBStakeBook:
    """YES/NO stakes against the market book.

    Coin is deducted from the worker's inventory at place() time; on
    market resolution, the losing side's pool is distributed pro-rata
    to the winning side. Expired markets refund all stakes.
    """

    def __init__(self) -> None:
        self._stakes: Dict[str, List[Stake]] = {}
        self._history: List[Dict[str, Any]] = []  # placed + resolved events

    def place(
        self,
        agent_id: str,
        market: Market,
        side: str,
        amount: float,
        worker_coin: float,
        epoch: int,
    ) -> Optional[Stake]:
        """Record a stake if the worker has enough coin. Returns the Stake
        on success, None if rejected. Caller is responsible for deducting
        the coin from the worker's inventory."""
        if market.status != "open":
            return None
        if side not in ("yes", "no"):
            return None
        if amount <= 0 or amount > worker_coin + 1e-9:
            return None
        stake = Stake(
            agent_id=agent_id,
            market_id=market.market_id,
            side=side,
            amount=float(amount),
            epoch=epoch,
        )
        self._stakes.setdefault(market.market_id, []).append(stake)
        self._history.append({
            "event": "placed",
            "epoch": epoch,
            **stake.to_dict(),
        })
        return stake

    def open_stakes(self) -> Dict[str, List[Stake]]:
        return {k: list(v) for k, v in self._stakes.items()}

    def distribute(self, market: Market) -> Dict[str, float]:
        """Resolve stakes for one closed market. Returns
        {agent_id: payout} (gross — includes the agent's own returned
        principal on the winning side). Stakes for this market are
        removed from the book."""
        stakes = self._stakes.pop(market.market_id, [])
        if not stakes:
            return {}

        # Expired / no resolution → refund principal.
        if market.status not in ("yes", "no"):
            payouts: Dict[str, float] = {}
            for s in stakes:
                payouts[s.agent_id] = payouts.get(s.agent_id, 0.0) + s.amount
                self._history.append({
                    "event": "refunded",
                    "market_id": market.market_id,
                    **s.to_dict(),
                })
            return payouts

        winning_side = market.status  # "yes" | "no"
        winners = [s for s in stakes if s.side == winning_side]
        losers = [s for s in stakes if s.side != winning_side]
        loser_pool = sum(s.amount for s in losers)
        winner_pool = sum(s.amount for s in winners)

        payouts: Dict[str, float] = {}
        if not winners:
            # No one took the winning side — refund losers.
            for s in losers:
                payouts[s.agent_id] = payouts.get(s.agent_id, 0.0) + s.amount
                self._history.append({
                    "event": "refunded_no_counterparty",
                    "market_id": market.market_id,
                    **s.to_dict(),
                })
            return payouts

        for s in winners:
            share = s.amount / winner_pool if winner_pool > 0 else 0.0
            gross = s.amount + share * loser_pool
            payouts[s.agent_id] = payouts.get(s.agent_id, 0.0) + gross
            self._history.append({
                "event": "won",
                "market_id": market.market_id,
                "winning_side": winning_side,
                "gross_payout": gross,
                **s.to_dict(),
            })
        for s in losers:
            self._history.append({
                "event": "lost",
                "market_id": market.market_id,
                "winning_side": winning_side,
                **s.to_dict(),
            })
        return payouts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "open_stakes": {
                mid: [s.to_dict() for s in stakes]
                for mid, stakes in self._stakes.items()
            },
            "history": list(self._history[-200:]),  # cap to recent
        }
