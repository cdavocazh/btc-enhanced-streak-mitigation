"""
Adjusted Strategy Configuration - 2026-02-04
=============================================
Parameter adjustments based on first evaluation findings.

Root Cause Analysis:
1. OOS Efficiency 0.45-0.48 indicates potential overfitting
2. Win rate 0% in evaluation suggests entry conditions too restrictive
3. ADX filter may be blocking valid trades in ranging markets
4. Progressive positioning threshold too high after losses

Adjustment Strategy:
- Relax entry filters to increase trade frequency
- Lower ADX threshold to allow more trades
- Widen RSI range for entry signals
- Reduce minimum positioning score
- Adjust progressive positioning thresholds
"""

from typing import Dict, Any, List

# ============================================================
# BASELINE PARAMETERS (From first evaluation)
# ============================================================

BASELINE_PARAMS = {
    "Baseline": {
        "min_pos_long": 0.4,
        "rsi_long_range": (20, 45),
        "pullback_range": (0.5, 3.0),
        "min_pos_score": 0.15,
        "stop_atr_mult": 1.8,
        "tp_atr_mult": 4.5,
    },
    "Adaptive_Baseline": {
        "min_pos_long": 0.4,
        "rsi_long_range": (20, 45),
        "pullback_range": (0.5, 3.0),
        "min_pos_score": 0.15,
        "adx_threshold": 20,
        "progressive_pos": {0: 0.4, 1: 0.5, 2: 0.6, 3: 0.8, 4: 1.0},
        "cooldown_bars": 96,
    },
    "Adaptive_ProgPos_Only": {
        "min_pos_long": 0.4,
        "rsi_long_range": (20, 45),
        "pullback_range": (0.5, 3.0),
        "min_pos_score": 0.15,
        "progressive_pos": {0: 0.4, 1: 0.5, 2: 0.6, 3: 0.8, 4: 1.0},
        "cooldown_bars": 96,
    },
}

# ============================================================
# ADJUSTMENT EXPERIMENTS
# ============================================================

ADJUSTMENT_EXPERIMENTS: List[Dict[str, Any]] = [
    # Experiment 1: Relax entry filters
    {
        "name": "Relaxed_Entry",
        "description": "Wider RSI range and lower min position score",
        "changes": {
            "min_pos_score": 0.10,  # Reduced from 0.15
            "rsi_long_range": (15, 50),  # Widened from (20, 45)
            "pullback_range": (0.3, 3.5),  # Widened from (0.5, 3.0)
        },
        "rationale": "Increase trade frequency by relaxing entry criteria",
    },

    # Experiment 2: Lower ADX threshold
    {
        "name": "Lower_ADX",
        "description": "Lower ADX threshold to capture more trades",
        "changes": {
            "adx_threshold": 15,  # Reduced from 20
            "min_pos_score": 0.12,  # Slightly reduced
        },
        "rationale": "Allow trades in weaker trends where positioning is strong",
    },

    # Experiment 3: Aggressive progressive positioning
    {
        "name": "Aggressive_ProgPos",
        "description": "Lower progressive positioning thresholds",
        "changes": {
            "progressive_pos": {0: 0.3, 1: 0.4, 2: 0.5, 3: 0.6, 4: 0.8},
            "cooldown_bars": 72,  # Reduced from 96
        },
        "rationale": "Re-enter market faster after losses with lower thresholds",
    },

    # Experiment 4: Wider stops and targets
    {
        "name": "Wider_SL_TP",
        "description": "Wider stop-loss and take-profit for more room",
        "changes": {
            "stop_atr_mult": 2.2,  # Increased from 1.8
            "tp_atr_mult": 5.0,  # Increased from 4.5
        },
        "rationale": "Give trades more room to develop, reduce early stops",
    },

    # Experiment 5: Combined relaxation
    {
        "name": "Combined_Relaxed",
        "description": "Combine multiple relaxation adjustments",
        "changes": {
            "min_pos_score": 0.10,
            "rsi_long_range": (15, 50),
            "pullback_range": (0.3, 3.5),
            "adx_threshold": 15,
            "progressive_pos": {0: 0.3, 1: 0.4, 2: 0.5, 3: 0.6, 4: 0.8},
            "cooldown_bars": 72,
        },
        "rationale": "Comprehensive relaxation to maximize trade opportunities",
    },

    # Experiment 6: Tighter but more selective
    {
        "name": "Selective_Tight",
        "description": "Tighter entry with better exit management",
        "changes": {
            "min_pos_long": 0.5,  # Increased from 0.4
            "min_pos_score": 0.20,  # Increased from 0.15
            "stop_atr_mult": 1.5,  # Tighter from 1.8
            "tp_atr_mult": 3.5,  # Tighter from 4.5
        },
        "rationale": "Higher quality entries with tighter risk management",
    },
]

# ============================================================
# STRATEGY VARIANTS TO TEST
# ============================================================

STRATEGY_VARIANTS = [
    "Baseline",
    "Adaptive_Baseline",
    "Adaptive_ProgPos_Only",
]

# Walk-forward configuration for experiments
WF_CONFIG = {
    "training_window_days": 30,  # Increased from 14
    "testing_window_days": 14,   # Increased from 7
    "step_size_days": 7,
}

# Asian hours filter
ASIAN_HOURS = set(range(0, 12))

# Technical parameters
ATR_PERIOD = 14
RSI_PERIOD = 56
SMA_PERIOD = 200
ADX_PERIOD = 14
