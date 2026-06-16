"""Configuration for the Gather-Trade-Build domain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaxBracket:
    """A single bracket in a piecewise tax schedule."""

    threshold: float = 0.0  # income above which this rate applies
    rate: float = 0.0  # marginal rate in this bracket


@dataclass
class TaxScheduleConfig:
    """Configuration for the tax schedule."""

    schedule_family: str = "piecewise"  # piecewise | flat | saez
    brackets: List[TaxBracket] = field(default_factory=lambda: [
        TaxBracket(threshold=0.0, rate=0.1),
        TaxBracket(threshold=10.0, rate=0.2),
        TaxBracket(threshold=25.0, rate=0.35),
    ])
    smoothing: float = 0.0  # sigmoid smoothing at bracket edges
    damping: float = 0.0  # rate-of-change damping for planner updates
    update_interval_epochs: int = 1  # how often planner updates
    allow_non_monotone: bool = False  # permit U-shaped schedules
    # Where collected taxes + fines go:
    #   none     — burned (legacy behavior)
    #   lump_sum — redistributed equally to all workers at epoch close
    redistribution: str = "none"
    # When True, unpaid tax/fines accrue as debt collected from future
    # coin instead of silently evaporating.
    debt_enabled: bool = False


@dataclass
class GamingConfig:
    """Configuration for strategic gaming mechanics."""

    income_shifting_enabled: bool = True
    max_shift_fraction_per_epoch: float = 0.3
    gaming_cost_fraction: float = 0.05  # friction on shifted amount
    bunching_detection_enabled: bool = True
    bunching_bin_width: float = 1.0  # histogram bin width near brackets


@dataclass
class MisreportingConfig:
    """Configuration for misreporting / evasion mechanics."""

    enabled: bool = True
    max_underreport_fraction: float = 0.5
    audit_probability: float = 0.2
    risk_based_audit_multiplier: float = 1.5
    fine_multiplier: float = 2.0  # fine = multiplier * evaded_tax
    reputation_penalty_per_catch: float = 0.1
    freeze_on_repeat: bool = True
    freeze_after_n_catches: int = 3
    freeze_duration_epochs: int = 2
    # Misreport semantics:
    #   event  — legacy: MISREPORT rewrites reported income at that
    #            instant; income earned afterwards re-fills reported at
    #            full value (gather) or fractionally (house), so the
    #            hidden share depends on action ordering
    #   stance — the declared fraction is a per-epoch stance; reported
    #            income is derived once at epoch close as
    #            gross * (1 - fraction), uniformly
    semantics: str = "event"
    # Audit selection:
    #   discrepancy — legacy: selection conditions on the TRUE gap
    #                 between gross and reported income (the auditor
    #                 reads minds), and selection implies conviction
    #   observable  — selection risk-scores only what a tax authority
    #                 can see (reported income vs coin actually received
    #                 from houses and market sales); anyone can be
    #                 selected, honest audits surface as false positives,
    #                 and conviction requires detection_power to fire
    selection_mode: str = "discrepancy"
    # P(an audit of a truly discrepant worker finds the discrepancy).
    # Only used in observable mode; legacy mode convicts with certainty.
    detection_power: float = 1.0


@dataclass
class CollusionConfig:
    """Configuration for collusion mechanics."""

    enabled: bool = True
    max_coalition_size: int = 4
    min_coalition_size: int = 2
    detection_window_steps: int = 20
    similarity_threshold: float = 0.7
    suspicion_score_threshold: float = 0.6
    response_audit_multiplier: float = 2.0
    response_stake_increase: float = 1.5
    response_trade_restriction_epochs: int = 1
    # Market-based detector: flags pairs that post >= min_asks sell
    # orders each in an epoch at near-identical prices (price-fixing
    # signature). Detection method recorded on the event so detector
    # precision/recall can be measured against true coalition labels.
    detect_price_fixing: bool = True
    price_fixing_min_asks: int = 2
    price_fixing_price_tolerance: float = 0.01  # relative mean-price gap


@dataclass
class MapConfig:
    """Configuration for the gridworld map."""

    height: int = 15
    width: int = 15
    wood_density: float = 0.2
    stone_density: float = 0.15
    resource_regen_rate: float = 0.1
    resource_max_amount: float = 5.0


@dataclass
class MarketConfig:
    """Configuration for the centralized market."""

    enabled: bool = True
    transaction_fee_rate: float = 0.02
    price_floor: float = 0.1
    price_ceiling: float = 100.0
    # 0 = legacy: order books are wiped at the end of every step, so a
    # bid can only ever match an ask posted in the same tick.
    # >0 = persistent book: orders rest for this many steps (cancelled
    # at epoch close), with coin/resource escrowed at post time so
    # resting orders cannot be invalidated or spoofed.
    order_ttl_steps: int = 0


@dataclass
class BuildConfig:
    """Configuration for house building."""

    wood_cost: float = 3.0
    stone_cost: float = 3.0
    income_per_house_per_step: float = 1.0
    max_houses_per_agent: int = 10
    # How house income is funded:
    #   mint     — coin printed from nothing each step (legacy behavior)
    #   treasury — paid from collected taxes/fines, pro-rated when the
    #              treasury can't cover full demand (fiscal closure)
    house_income_mode: str = "mint"


@dataclass
class UtilityConfig:
    """Isoelastic (CRRA) worker preferences with labor disutility.

    u = crra(coin; eta) + house_weight * houses - labor_coeff * effort
    where effort is energy actually spent on actions. eta=0 reduces to
    linear coin utility; eta -> 1 approaches log utility.
    """

    eta: float = 0.35  # coefficient of relative risk aversion
    labor_coeff: float = 0.15  # disutility per unit of energy spent
    house_weight: float = 5.0  # utility per house owned


@dataclass
class PlannerConfig:
    """Configuration for the planner agent."""

    planner_type: str = "heuristic"  # heuristic | bandit | saez | rl
    objective: str = "welfare"
    prod_weight: float = 1.0
    ineq_weight: float = 0.5
    learning_rate: float = 0.01
    exploration_rate: float = 0.1
    update_interval_epochs: int = 1
    # Which Gini the planner reacts to: income (legacy, noisy per-epoch
    # flow) or wealth (cumulative endowment, AI Economist semantics)
    inequality_measure: str = "income"
    # Saez planner: elasticity estimate bounds and smoothing
    saez_elasticity_init: float = 0.5
    saez_elasticity_lr: float = 0.3  # EMA weight for new elasticity estimates
    saez_rate_change_cap: float = 0.05  # max top-rate move per update


@dataclass
class GTBConfig:
    """Top-level configuration for the Gather-Trade-Build domain."""

    map: MapConfig = field(default_factory=MapConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    build: BuildConfig = field(default_factory=BuildConfig)
    taxation: TaxScheduleConfig = field(default_factory=TaxScheduleConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    gaming: GamingConfig = field(default_factory=GamingConfig)
    misreporting: MisreportingConfig = field(default_factory=MisreportingConfig)
    collusion: CollusionConfig = field(default_factory=CollusionConfig)
    energy_per_step: float = 10.0
    energy_cost_move: float = 1.0
    energy_cost_gather: float = 1.0
    energy_cost_trade: float = 0.5
    energy_cost_build: float = 2.0
    # Income accounting:
    #   legacy — gathering books taxable income with no coin behind it,
    #            sale proceeds book income again (double count), buyer
    #            spending is not an expense
    #   coin   — income == net coin flow: house income + sale proceeds
    #            - purchase outlays; gathering yields resources only
    ledger_mode: str = "legacy"
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GTBConfig":
        """Parse a GTBConfig from a YAML-sourced dict."""
        if not data:
            return cls()

        map_data = data.get("map", {})
        map_cfg = MapConfig(**{
            k: map_data[k] for k in (
                "height", "width", "wood_density", "stone_density",
                "resource_regen_rate", "resource_max_amount",
            ) if k in map_data
        })

        market_data = data.get("market", {})
        market_cfg = MarketConfig(**{
            k: market_data[k] for k in (
                "enabled", "transaction_fee_rate", "price_floor",
                "price_ceiling", "order_ttl_steps",
            ) if k in market_data
        })

        build_data = data.get("build", {})
        build_cfg = BuildConfig(**{
            k: build_data[k] for k in (
                "wood_cost", "stone_cost", "income_per_house_per_step",
                "max_houses_per_agent", "house_income_mode",
            ) if k in build_data
        })

        tax_data = data.get("taxation", {})
        brackets = []
        for b in tax_data.get("brackets", []):
            brackets.append(TaxBracket(
                threshold=b.get("threshold", 0.0),
                rate=b.get("rate", 0.0),
            ))
        tax_kwargs: Dict[str, Any] = {}
        if brackets:
            tax_kwargs["brackets"] = brackets
        for k in ("schedule_family", "smoothing", "damping",
                   "update_interval_epochs", "allow_non_monotone",
                   "redistribution", "debt_enabled"):
            if k in tax_data:
                tax_kwargs[k] = tax_data[k]
        tax_cfg = TaxScheduleConfig(**tax_kwargs)

        planner_data = data.get("planner", {})
        planner_cfg = PlannerConfig(**{
            k: planner_data[k] for k in (
                "planner_type", "objective", "prod_weight", "ineq_weight",
                "learning_rate", "exploration_rate", "update_interval_epochs",
                "inequality_measure", "saez_elasticity_init",
                "saez_elasticity_lr", "saez_rate_change_cap",
            ) if k in planner_data
        })

        utility_data = data.get("utility", {})
        utility_cfg = UtilityConfig(**{
            k: utility_data[k] for k in (
                "eta", "labor_coeff", "house_weight",
            ) if k in utility_data
        })

        gaming_data = data.get("gaming", {})
        gaming_cfg = GamingConfig(**{
            k: gaming_data[k] for k in (
                "income_shifting_enabled", "max_shift_fraction_per_epoch",
                "gaming_cost_fraction", "bunching_detection_enabled",
                "bunching_bin_width",
            ) if k in gaming_data
        })

        misreport_data = data.get("misreporting", {})
        misreport_cfg = MisreportingConfig(**{
            k: misreport_data[k] for k in (
                "enabled", "max_underreport_fraction", "audit_probability",
                "risk_based_audit_multiplier", "fine_multiplier",
                "reputation_penalty_per_catch", "freeze_on_repeat",
                "freeze_after_n_catches", "freeze_duration_epochs",
                "semantics", "selection_mode", "detection_power",
            ) if k in misreport_data
        })

        collusion_data = data.get("collusion", {})
        collusion_cfg = CollusionConfig(**{
            k: collusion_data[k] for k in (
                "enabled", "max_coalition_size", "min_coalition_size",
                "detection_window_steps", "similarity_threshold",
                "suspicion_score_threshold", "response_audit_multiplier",
                "response_stake_increase", "response_trade_restriction_epochs",
                "detect_price_fixing", "price_fixing_min_asks",
                "price_fixing_price_tolerance",
            ) if k in collusion_data
        })

        return cls(
            map=map_cfg,
            market=market_cfg,
            build=build_cfg,
            taxation=tax_cfg,
            planner=planner_cfg,
            utility=utility_cfg,
            gaming=gaming_cfg,
            misreporting=misreport_cfg,
            collusion=collusion_cfg,
            energy_per_step=data.get("energy_per_step", 10.0),
            energy_cost_move=data.get("energy_cost_move", 1.0),
            energy_cost_gather=data.get("energy_cost_gather", 1.0),
            energy_cost_trade=data.get("energy_cost_trade", 0.5),
            energy_cost_build=data.get("energy_cost_build", 2.0),
            ledger_mode=data.get("ledger_mode", "legacy"),
            seed=data.get("seed"),
        )
