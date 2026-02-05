"""
Adjusted Strategy Configuration - Auto-generated
================================================
Parameter adjustments based on evaluation findings.
"""

from typing import Dict, Any, List

# Baseline parameters
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

# Adjustment experiments
ADJUSTMENT_EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "name": "Relaxed_Entry",
        "description": "Wider RSI range and lower min position score",
        "changes": {
            "min_pos_score": 0.10,
            "rsi_long_range": (15, 50),
            "pullback_range": (0.3, 3.5),
        },
        "rationale": "Increase trade frequency by relaxing entry criteria",
    },
    {
        "name": "Lower_ADX",
        "description": "Lower ADX threshold to capture more trades",
        "changes": {
            "adx_threshold": 15,
            "min_pos_score": 0.12,
        },
        "rationale": "Allow trades in weaker trends where positioning is strong",
    },
    {
        "name": "Aggressive_ProgPos",
        "description": "Lower progressive positioning thresholds",
        "changes": {
            "progressive_pos": {0: 0.3, 1: 0.4, 2: 0.5, 3: 0.6, 4: 0.8},
            "cooldown_bars": 72,
        },
        "rationale": "Re-enter market faster after losses",
    },
    {
        "name": "Wider_SL_TP",
        "description": "Wider stop-loss and take-profit",
        "changes": {
            "stop_atr_mult": 2.2,
            "tp_atr_mult": 5.0,
        },
        "rationale": "Give trades more room to develop",
    },
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
        "rationale": "Comprehensive relaxation",
    },
    {
        "name": "Selective_Tight",
        "description": "Tighter entry with better exit management",
        "changes": {
            "min_pos_long": 0.5,
            "min_pos_score": 0.20,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.5,
        },
        "rationale": "Higher quality entries with tighter risk",
    },
]

STRATEGY_VARIANTS = ["Baseline", "Adaptive_Baseline", "Adaptive_ProgPos_Only"]

WF_CONFIG = {
    "training_window_days": 30,
    "testing_window_days": 14,
    "step_size_days": 7,
}

ASIAN_HOURS = set(range(0, 12))
ATR_PERIOD = 14
RSI_PERIOD = 56
SMA_PERIOD = 200
ADX_PERIOD = 14
