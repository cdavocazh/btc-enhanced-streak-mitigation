"""
BTC Strategy Evaluation Framework
==================================
Walk-forward optimization and performance tracking for BTC trading strategies.

This package provides:
- WalkForwardEngine: Rolling-window backtesting with IS/OOS validation
- PerformanceTracker: SQLite-based performance history and alerts
- StrategyLearner: Adaptive parameter adjustment based on trends
- EvaluationRunner: Scheduled evaluation orchestrator

Quick Start:
    cd eval && python run_evaluation.py --quick     # Quick check
    cd eval && python run_evaluation.py --full      # Full WFO
    cd eval && python run_evaluation.py --adapt     # Learning cycle

Note: Run scripts from the eval/ directory to ensure proper imports.
"""

# Only import when running as module (not as script)
try:
    from .config import (
    WalkForwardConfig,
    PerformanceThresholds,
    AdaptationConfig,
    EvaluationSchedule,
    DEFAULT_WF_CONFIG,
    DEFAULT_THRESHOLDS,
    DEFAULT_ADAPTATION,
    DEFAULT_SCHEDULE,
)

from .walk_forward_engine import WalkForwardEngine, BacktestResult, WalkForwardResult
from .performance_tracker import PerformanceTracker, PerformanceSnapshot, Alert
    from .strategy_learner import StrategyLearner, PerformanceTrend, ParameterAdjustment, LearningReport
except ImportError:
    # Running as script, imports handled locally
    pass

__all__ = [
    # Config
    'WalkForwardConfig',
    'PerformanceThresholds',
    'AdaptationConfig',
    'EvaluationSchedule',
    'DEFAULT_WF_CONFIG',
    'DEFAULT_THRESHOLDS',
    'DEFAULT_ADAPTATION',
    'DEFAULT_SCHEDULE',
    # Walk-forward
    'WalkForwardEngine',
    'BacktestResult',
    'WalkForwardResult',
    # Performance tracking
    'PerformanceTracker',
    'PerformanceSnapshot',
    'Alert',
    # Learning
    'StrategyLearner',
    'PerformanceTrend',
    'ParameterAdjustment',
    'LearningReport',
]
