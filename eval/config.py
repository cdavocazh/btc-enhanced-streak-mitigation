"""
BTC Strategy Evaluation Framework Configuration
================================================
Central configuration for walk-forward optimization and performance tracking.

Focused on strategies from:
- backtest_15min_new (Tiered Capital + Streak Mitigation)
- backtest_15min_new_streak_a (Adaptive Streak Reduction with ADX/ProgPos)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import timedelta


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward optimization windows."""

    # Window sizes (in 15-minute bars)
    # 96 bars = 1 day (96 * 15min = 1440min = 24hrs)
    training_window: int = 96 * 14  # 14 days for in-sample training
    testing_window: int = 96 * 7    # 7 days for out-of-sample testing
    step_size: int = 96 * 7         # Roll forward by 7 days

    # Minimum data requirements
    min_training_bars: int = 96 * 7  # At least 7 days for training
    min_testing_bars: int = 96 * 2   # At least 2 days for testing

    # Optimization settings
    optimization_metric: str = "sharpe_ratio"  # Primary metric to optimize
    secondary_metrics: List[str] = field(default_factory=lambda: [
        "total_return_pct", "max_drawdown_pct", "win_rate_pct", "profit_factor"
    ])


@dataclass
class PerformanceThresholds:
    """Thresholds for performance degradation alerts."""

    # Return degradation thresholds
    return_degradation_warning: float = 0.20  # 20% below expected
    return_degradation_critical: float = 0.40  # 40% below expected

    # Drawdown thresholds
    max_drawdown_warning: float = 0.25  # 25% drawdown warning
    max_drawdown_critical: float = 0.40  # 40% drawdown critical

    # Win rate thresholds
    win_rate_min: float = 0.35  # Minimum acceptable win rate

    # Sharpe ratio thresholds
    sharpe_min: float = 0.5  # Minimum acceptable Sharpe ratio

    # Consistency thresholds
    max_losing_streak: int = 7  # Alert after 7 consecutive losses
    out_of_sample_efficiency: float = 0.6  # OOS should be 60% of IS performance


@dataclass
class AdaptationConfig:
    """Configuration for strategy parameter adaptation."""

    # How much to adjust parameters
    parameter_adjustment_rate: float = 0.10  # 10% max adjustment per cycle

    # Minimum evaluations before adaptation
    min_evaluations_before_adapt: int = 4

    # Parameter bounds (relative to current values)
    max_parameter_deviation: float = 0.30  # Max 30% deviation from baseline

    # Regime detection
    regime_lookback_windows: int = 4  # Look at last 4 evaluation windows
    regime_change_threshold: float = 0.25  # 25% performance shift = regime change

    # Minimum confidence to apply adaptation
    min_confidence: float = 0.50


@dataclass
class EvaluationSchedule:
    """Schedule for running evaluations."""

    # How often to run full evaluation
    full_evaluation_interval: timedelta = timedelta(days=7)

    # Quick performance check interval
    quick_check_interval: timedelta = timedelta(hours=6)

    # How often to check for parameter adaptation
    adaptation_check_interval: timedelta = timedelta(days=14)


# ============================================================
# STRATEGY CONFIGURATIONS FROM backtest_15min_new
# ============================================================

# Tiered risk configuration (from run_tiered_streak_mitigation.py)
RISK_TIERS = [
    # (min_equity, max_equity, risk_amount)
    (0, 150000, 5000),              # 5% of $100k = $5k
    (150000, 225000, 10500),        # 7% of $150k = $10.5k
    (225000, 337500, 14625),        # 6.5% of $225k = $14.625k
    (337500, 507000, 20250),        # 6% of $337.5k = $20.25k
    (507000, 760000, 28000),        # 5.5% of $507k = $28k
    (760000, 1200000, 38000),       # Fixed $38k
    (1200000, float('inf'), 54000), # Fixed $54k
]

# Streak mitigation rules
STREAK_RULES = {
    3: {'reduction': 0.40, 'recovery': 'initial'},
    6: {'reduction': 0.30, 'recovery': 'initial'},
    9: {'reduction': 0.30, 'recovery': '5pct'},
}

# Positioning thresholds
TOP_TRADER_STRONG = 0.60
TOP_TRADER_MODERATE = 0.55

# Asian hours filter (GMT+8)
ASIAN_HOURS = set(range(0, 12))

# Technical indicator periods
ATR_PERIOD = 14
RSI_PERIOD = 56
SMA_PERIOD = 200
ADX_PERIOD = 14


# ============================================================
# STRATEGY PARAMETER RANGES FOR OPTIMIZATION
# ============================================================

STRATEGY_PARAMETER_RANGES: Dict[str, Dict[str, Any]] = {
    # Strategies from backtest_15min_new
    "Baseline": {
        "min_pos_long": {"min": 0.2, "max": 0.8, "step": 0.1, "default": 0.4},
        "rsi_long_min": {"min": 15, "max": 35, "step": 5, "default": 20},
        "rsi_long_max": {"min": 40, "max": 55, "step": 5, "default": 45},
        "pullback_min_pct": {"min": 0.3, "max": 1.0, "step": 0.1, "default": 0.5},
        "pullback_max_pct": {"min": 2.0, "max": 4.0, "step": 0.5, "default": 3.0},
        "atr_period": {"min": 10, "max": 28, "step": 2, "default": 14},
        "rsi_period": {"min": 14, "max": 70, "step": 7, "default": 56},
        "sma_period": {"min": 100, "max": 300, "step": 50, "default": 200},
    },
    "PosVol_Combined": {
        "pos_decline_be_threshold": {"min": 0.3, "max": 0.7, "step": 0.1, "default": 0.5},
        "vol_collapse_threshold": {"min": 0.4, "max": 0.7, "step": 0.1, "default": 0.5},
        "vol_collapse_sl_mult": {"min": 0.8, "max": 1.5, "step": 0.1, "default": 1.0},
    },
    "MultiTP_30": {
        "tp1_atr_mult": {"min": 1.5, "max": 3.0, "step": 0.25, "default": 2.0},
        "tp1_exit_pct": {"min": 0.2, "max": 0.5, "step": 0.1, "default": 0.3},
    },
    "VolFilter_Adaptive": {
        "vol_min_for_entry": {"min": 0.6, "max": 1.0, "step": 0.1, "default": 0.8},
        "vol_collapse_threshold": {"min": 0.4, "max": 0.6, "step": 0.05, "default": 0.5},
    },
    "Conservative": {
        "tp1_atr_mult": {"min": 1.5, "max": 2.5, "step": 0.25, "default": 2.0},
        "tp1_exit_pct": {"min": 0.3, "max": 0.5, "step": 0.1, "default": 0.4},
        "pos_decline_be_threshold": {"min": 0.3, "max": 0.5, "step": 0.1, "default": 0.4},
        "vol_min_for_entry": {"min": 0.6, "max": 0.9, "step": 0.1, "default": 0.7},
    },

    # Strategies from backtest_15min_new_streak_a (Adaptive)
    "Adaptive_Baseline": {
        "adx_trending_threshold": {"min": 15, "max": 30, "step": 5, "default": 20},
        "cooldown_bars": {"min": 48, "max": 144, "step": 24, "default": 96},
        "cooldown_after_losses": {"min": 4, "max": 7, "step": 1, "default": 5},
        "progressive_pos_0": {"min": 0.3, "max": 0.5, "step": 0.05, "default": 0.4},
        "progressive_pos_2": {"min": 0.5, "max": 0.7, "step": 0.05, "default": 0.6},
        "progressive_pos_3": {"min": 0.7, "max": 0.9, "step": 0.05, "default": 0.8},
        "progressive_pos_4": {"min": 0.9, "max": 1.2, "step": 0.1, "default": 1.0},
    },
    "Adaptive_Conservative": {
        "adx_trending_threshold": {"min": 20, "max": 35, "step": 5, "default": 25},
        "tp1_atr_mult": {"min": 1.5, "max": 2.5, "step": 0.25, "default": 2.0},
        "tp1_exit_pct": {"min": 0.3, "max": 0.5, "step": 0.1, "default": 0.4},
        "pos_decline_be_threshold": {"min": 0.3, "max": 0.5, "step": 0.1, "default": 0.4},
    },
    "Adaptive_ADX_Only": {
        "adx_trending_threshold": {"min": 15, "max": 30, "step": 5, "default": 20},
        "vol_min_trending": {"min": 0.5, "max": 0.9, "step": 0.1, "default": 0.7},
        "vol_min_ranging": {"min": 1.0, "max": 1.5, "step": 0.1, "default": 1.2},
    },
    "Adaptive_ProgPos_Only": {
        "progressive_pos_0": {"min": 0.3, "max": 0.5, "step": 0.05, "default": 0.4},
        "progressive_pos_1": {"min": 0.4, "max": 0.6, "step": 0.05, "default": 0.5},
        "progressive_pos_2": {"min": 0.5, "max": 0.7, "step": 0.05, "default": 0.6},
        "progressive_pos_3": {"min": 0.7, "max": 0.9, "step": 0.05, "default": 0.8},
        "progressive_pos_4": {"min": 0.9, "max": 1.2, "step": 0.1, "default": 1.0},
        "cooldown_bars": {"min": 48, "max": 144, "step": 24, "default": 96},
        "cooldown_after_losses": {"min": 4, "max": 7, "step": 1, "default": 5},
    },
}


# ============================================================
# STRATEGY CONFIGURATIONS (BASELINE VALUES)
# ============================================================

# From backtest_15min_new
TIERED_STRATEGY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Baseline": {
        "pos_exit_enabled": True,
        "pos_exit_threshold": 0.0,
        "pos_decline_be_enabled": False,
        "vol_collapse_enabled": False,
        "multi_tp_enabled": False,
    },
    "PosVol_Combined": {
        "pos_exit_enabled": True,
        "pos_exit_threshold": 0.0,
        "pos_decline_be_enabled": True,
        "pos_decline_be_threshold": 0.5,
        "vol_collapse_enabled": True,
        "vol_collapse_threshold": 0.5,
        "vol_collapse_sl_mult": 1.0,
        "multi_tp_enabled": False,
    },
    "MultiTP_30": {
        "pos_exit_enabled": True,
        "pos_exit_threshold": 0.0,
        "multi_tp_enabled": True,
        "tp1_atr_mult": 2.0,
        "tp1_exit_pct": 0.3,
        "pos_decline_be_enabled": False,
        "vol_collapse_enabled": False,
    },
    "VolFilter_Adaptive": {
        "vol_min_for_entry": 0.8,
        "pos_exit_enabled": True,
        "pos_exit_threshold": 0.0,
        "vol_collapse_enabled": True,
        "vol_collapse_threshold": 0.5,
        "vol_size_adjust": True,
        "pos_decline_be_enabled": False,
        "multi_tp_enabled": False,
    },
    "Conservative": {
        "vol_min_for_entry": 0.7,
        "pos_exit_enabled": True,
        "pos_exit_threshold": 0.0,
        "multi_tp_enabled": True,
        "tp1_atr_mult": 2.0,
        "tp1_exit_pct": 0.4,
        "pos_decline_be_enabled": True,
        "pos_decline_be_threshold": 0.4,
        "vol_collapse_enabled": False,
    },
}

# From backtest_15min_new_streak_a
ADAPTIVE_STRATEGY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Adaptive_Baseline": {
        "pos_exit_enabled": True,
        "pos_exit_threshold": 0.0,
        "adx_filter": True,
        "progressive_pos": True,
        "cooldown_enabled": True,
    },
    "Adaptive_Conservative": {
        "pos_exit_enabled": True,
        "pos_exit_threshold": 0.0,
        "adx_filter": True,
        "progressive_pos": True,
        "cooldown_enabled": True,
        "multi_tp_enabled": True,
        "tp1_atr_mult": 2.0,
        "tp1_exit_pct": 0.4,
        "pos_decline_be_enabled": True,
        "pos_decline_be_threshold": 0.4,
        "vol_size_adjust": True,
    },
    "Adaptive_ADX_Only": {
        "pos_exit_enabled": True,
        "pos_exit_threshold": 0.0,
        "adx_filter": True,
        "progressive_pos": False,
        "cooldown_enabled": False,
    },
    "Adaptive_ProgPos_Only": {
        "pos_exit_enabled": True,
        "pos_exit_threshold": 0.0,
        "adx_filter": False,
        "progressive_pos": True,
        "cooldown_enabled": True,
    },
}


# ============================================================
# ENTRY FILTER CONFIGURATIONS
# ============================================================

ENTRY_FILTER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "min_pos_long": 0.4,
        "rsi_long_range": (20, 45),
        "pullback_range": (0.5, 3.0),
        "min_pos_score": 0.15,
    },
    "relaxed": {
        "min_pos_long": 0.2,
        "rsi_long_range": (15, 50),
        "pullback_range": (0.3, 4.0),
        "min_pos_score": 0.10,
    },
    "aggressive": {
        "min_pos_long": 0.1,
        "rsi_long_range": (10, 55),
        "pullback_range": (0.2, 5.0),
        "min_pos_score": 0.05,
    },
    "conservative": {
        "min_pos_long": 0.6,
        "rsi_long_range": (25, 40),
        "pullback_range": (0.8, 2.5),
        "min_pos_score": 0.20,
    },
}

# Adaptive entry filter (for adaptive strategies)
ADAPTIVE_ENTRY_FILTER: Dict[str, Any] = {
    'min_pos_long_base': 0.4,
    'rsi_long_range': [20, 45],
    'pullback_range': [0.5, 3.0],
    'min_pos_score': 0.15,

    # Progressive positioning threshold
    'progressive_pos_threshold': {
        0: 0.4,
        1: 0.5,
        2: 0.6,
        3: 0.8,
        4: 1.0,
    },
    'cooldown_after_losses': 5,
    'cooldown_bars': 96,

    # ADX filter
    'adx_filter_enabled': True,
    'adx_trending_threshold': 20,
    'adx_strong_trend_threshold': 30,

    # Volume requirements
    'vol_min_trending': 0.7,
    'vol_min_ranging': 1.2,
}


# ============================================================
# DEFAULT CONFIGURATION INSTANCES
# ============================================================

DEFAULT_WF_CONFIG = WalkForwardConfig()
DEFAULT_THRESHOLDS = PerformanceThresholds()
DEFAULT_ADAPTATION = AdaptationConfig()
DEFAULT_SCHEDULE = EvaluationSchedule()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_all_strategies() -> List[str]:
    """Get list of all available strategies."""
    tiered = list(TIERED_STRATEGY_CONFIGS.keys())
    adaptive = list(ADAPTIVE_STRATEGY_CONFIGS.keys())
    return tiered + adaptive


def get_strategy_config(strategy_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a specific strategy."""
    if strategy_name in TIERED_STRATEGY_CONFIGS:
        return TIERED_STRATEGY_CONFIGS[strategy_name]
    elif strategy_name in ADAPTIVE_STRATEGY_CONFIGS:
        return ADAPTIVE_STRATEGY_CONFIGS[strategy_name]
    return None


def get_parameter_ranges(strategy_name: str) -> Optional[Dict[str, Any]]:
    """Get optimization parameter ranges for a strategy."""
    return STRATEGY_PARAMETER_RANGES.get(strategy_name)


def is_adaptive_strategy(strategy_name: str) -> bool:
    """Check if a strategy is an adaptive strategy."""
    return strategy_name in ADAPTIVE_STRATEGY_CONFIGS
