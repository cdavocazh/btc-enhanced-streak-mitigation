#!/usr/bin/env python3
"""
Strategy Learner for BTC Strategies
====================================
Adapts strategy parameters based on performance trends and market regimes.

Features:
- Analyze performance trends using linear regression
- Propose parameter adjustments
- Apply adjustments (with dry-run support)
- Generate learning reports
"""

import os
import sys
import json
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import (
    AdaptationConfig, DEFAULT_ADAPTATION,
    STRATEGY_PARAMETER_RANGES,
    get_strategy_config, is_adaptive_strategy
)
from performance_tracker import PerformanceTracker, log


@dataclass
class PerformanceTrend:
    """Analysis of performance trend."""
    strategy_name: str
    entry_filter: str
    trend_direction: str  # improving, declining, stable
    trend_slope: float
    trend_r_squared: float
    confidence: float
    data_points: int
    period_days: int


@dataclass
class ParameterAdjustment:
    """Proposed parameter adjustment."""
    strategy_name: str
    parameter_name: str
    current_value: float
    proposed_value: float
    change_pct: float
    reason: str
    confidence: float


@dataclass
class LearningReport:
    """Report from a learning cycle."""
    timestamp: str
    strategy_name: str
    entry_filter: str
    trend: PerformanceTrend
    regime_change: Optional[Dict[str, Any]]
    proposed_adjustments: List[ParameterAdjustment]
    applied: bool
    notes: str


class StrategyLearner:
    """
    Adapts strategy parameters based on performance analysis.
    """

    def __init__(
        self,
        tracker: PerformanceTracker = None,
        config: AdaptationConfig = None
    ):
        self.tracker = tracker or PerformanceTracker()
        self.config = config or DEFAULT_ADAPTATION
        self.learnings_dir = os.path.join(SCRIPT_DIR, 'learnings')
        os.makedirs(self.learnings_dir, exist_ok=True)

    def analyze_performance_trend(
        self,
        strategy_name: str,
        entry_filter: str,
        days: int = 30
    ) -> Optional[PerformanceTrend]:
        """
        Analyze performance trend using linear regression.

        Uses the Sharpe ratio as the primary metric for trend detection.
        """
        snapshots = self.tracker.get_historical_performance(
            strategy_name, entry_filter, days=days
        )

        if len(snapshots) < self.config.min_evaluations_before_adapt:
            log(f"Insufficient data for trend analysis: {len(snapshots)} snapshots")
            return None

        # Extract Sharpe ratios with timestamps
        times = []
        sharpes = []
        base_time = None

        for s in sorted(snapshots, key=lambda x: x['timestamp']):
            ts = datetime.fromisoformat(s['timestamp'].replace('Z', '+00:00'))
            if base_time is None:
                base_time = ts
            times.append((ts - base_time).total_seconds() / 86400)  # Days from start
            sharpes.append(s['sharpe_ratio'])

        if len(times) < 2:
            return None

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(times, sharpes)

        # Determine trend direction
        if slope > 0.01 and p_value < 0.1:
            direction = "improving"
        elif slope < -0.01 and p_value < 0.1:
            direction = "declining"
        else:
            direction = "stable"

        # Confidence based on R-squared and p-value
        confidence = min(r_value ** 2 * (1 - p_value), 1.0)

        return PerformanceTrend(
            strategy_name=strategy_name,
            entry_filter=entry_filter,
            trend_direction=direction,
            trend_slope=slope,
            trend_r_squared=r_value ** 2,
            confidence=confidence,
            data_points=len(snapshots),
            period_days=days
        )

    def propose_adjustments(
        self,
        trend: PerformanceTrend,
        regime_change: Optional[Dict[str, Any]] = None
    ) -> List[ParameterAdjustment]:
        """
        Propose parameter adjustments based on trend and regime.
        """
        adjustments = []

        # Only propose adjustments if declining and confident
        if trend.trend_direction != "declining":
            log(f"No adjustments needed: trend is {trend.trend_direction}")
            return adjustments

        if trend.confidence < self.config.min_confidence:
            log(f"Low confidence ({trend.confidence:.2f}): skipping adjustments")
            return adjustments

        # Get parameter ranges for this strategy
        param_ranges = STRATEGY_PARAMETER_RANGES.get(trend.strategy_name, {})

        if not param_ranges:
            log(f"No parameter ranges defined for {trend.strategy_name}")
            return adjustments

        # Propose tightening entry criteria when declining
        # The adjustment rate is proportional to confidence
        adjustment_rate = self.config.parameter_adjustment_rate * trend.confidence

        for param_name, param_config in param_ranges.items():
            current_value = param_config.get('default', param_config['min'])
            param_min = param_config['min']
            param_max = param_config['max']

            # Determine adjustment direction based on parameter type
            if 'threshold' in param_name.lower() or 'min' in param_name.lower():
                # For thresholds: increase (tighter) when declining
                change = adjustment_rate * (param_max - param_min)
                proposed = min(current_value + change, param_max)
            elif 'max' in param_name.lower():
                # For maximums: decrease (tighter) when declining
                change = adjustment_rate * (param_max - param_min)
                proposed = max(current_value - change, param_min)
            else:
                # Default: no change
                continue

            # Check if change is meaningful
            change_pct = abs(proposed - current_value) / current_value if current_value != 0 else 0
            if change_pct < 0.01:
                continue

            # Check max deviation from baseline
            baseline = param_config.get('default', current_value)
            deviation = abs(proposed - baseline) / baseline if baseline != 0 else 0
            if deviation > self.config.max_parameter_deviation:
                log(f"Skipping {param_name}: deviation {deviation:.2%} exceeds max")
                continue

            adjustments.append(ParameterAdjustment(
                strategy_name=trend.strategy_name,
                parameter_name=param_name,
                current_value=current_value,
                proposed_value=proposed,
                change_pct=change_pct * 100,
                reason=f"Declining trend (slope={trend.trend_slope:.4f})",
                confidence=trend.confidence
            ))

        return adjustments

    def apply_adjustments(
        self,
        adjustments: List[ParameterAdjustment],
        dry_run: bool = True
    ) -> bool:
        """
        Apply parameter adjustments.

        If dry_run=True, only logs changes without applying.
        """
        if not adjustments:
            log("No adjustments to apply")
            return False

        for adj in adjustments:
            if dry_run:
                log(f"[DRY-RUN] Would adjust {adj.strategy_name}.{adj.parameter_name}: "
                    f"{adj.current_value:.4f} -> {adj.proposed_value:.4f} ({adj.change_pct:+.1f}%)")
            else:
                # Record to parameter history
                self.tracker.record_parameter_change(
                    adj.strategy_name,
                    adj.parameter_name,
                    adj.proposed_value,
                    adj.reason
                )
                log(f"Applied: {adj.strategy_name}.{adj.parameter_name} = {adj.proposed_value:.4f}")

        return not dry_run

    def run_learning_cycle(
        self,
        strategy_name: str,
        entry_filter: str = "baseline",
        apply: bool = False
    ) -> LearningReport:
        """
        Run a complete learning cycle for a strategy.

        1. Analyze performance trends
        2. Detect regime changes
        3. Propose parameter adjustments
        4. Optionally apply changes
        5. Generate report
        """
        log(f"Running learning cycle for {strategy_name} ({entry_filter})")

        # Analyze trend
        trend = self.analyze_performance_trend(strategy_name, entry_filter)

        if trend is None:
            trend = PerformanceTrend(
                strategy_name=strategy_name,
                entry_filter=entry_filter,
                trend_direction="unknown",
                trend_slope=0,
                trend_r_squared=0,
                confidence=0,
                data_points=0,
                period_days=30
            )

        # Detect regime change
        regime_change = self.tracker.detect_regime_change(strategy_name, entry_filter)

        # Propose adjustments
        adjustments = self.propose_adjustments(trend, regime_change)

        # Apply if requested
        applied = False
        if adjustments and apply:
            applied = self.apply_adjustments(adjustments, dry_run=False)
        elif adjustments:
            self.apply_adjustments(adjustments, dry_run=True)

        # Generate notes
        notes = self._generate_notes(trend, regime_change, adjustments, applied)

        # Create report
        report = LearningReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy_name=strategy_name,
            entry_filter=entry_filter,
            trend=trend,
            regime_change=regime_change,
            proposed_adjustments=adjustments,
            applied=applied,
            notes=notes
        )

        # Save report
        self._save_report(report)

        return report

    def _generate_notes(
        self,
        trend: PerformanceTrend,
        regime_change: Optional[Dict[str, Any]],
        adjustments: List[ParameterAdjustment],
        applied: bool
    ) -> str:
        """Generate human-readable notes for the learning report."""
        notes = []

        # Trend analysis
        notes.append(f"Trend: {trend.trend_direction.upper()}")
        if trend.data_points > 0:
            notes.append(f"  Based on {trend.data_points} snapshots over {trend.period_days} days")
            notes.append(f"  Slope: {trend.trend_slope:.4f}, R²: {trend.trend_r_squared:.2f}")
            notes.append(f"  Confidence: {trend.confidence:.2%}")

        # Regime change
        if regime_change:
            notes.append(f"\nRegime Change Detected: {regime_change['regime'].upper()}")
            notes.append(f"  Sharpe change: {regime_change['sharpe_change']:+.2%}")
            notes.append(f"  Return change: {regime_change['return_change']:+.2%}")
            notes.append(f"  Confidence: {regime_change['confidence']:.2%}")

        # Adjustments
        if adjustments:
            notes.append(f"\nProposed Adjustments: {len(adjustments)}")
            for adj in adjustments:
                status = "APPLIED" if applied else "PROPOSED"
                notes.append(f"  [{status}] {adj.parameter_name}: {adj.current_value:.4f} -> {adj.proposed_value:.4f}")
        else:
            notes.append("\nNo adjustments proposed")

        return "\n".join(notes)

    def _save_report(self, report: LearningReport):
        """Save learning report to JSON."""
        filename = f"learning_{report.strategy_name}_{report.entry_filter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.learnings_dir, filename)

        # Convert dataclasses to dicts
        data = {
            'timestamp': report.timestamp,
            'strategy_name': report.strategy_name,
            'entry_filter': report.entry_filter,
            'trend': asdict(report.trend),
            'regime_change': report.regime_change,
            'proposed_adjustments': [asdict(a) for a in report.proposed_adjustments],
            'applied': report.applied,
            'notes': report.notes,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        log(f"Learning report saved: {filepath}")

    def get_learning_history(
        self,
        strategy_name: str = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get learning history from saved reports."""
        reports = []

        for filename in os.listdir(self.learnings_dir):
            if not filename.endswith('.json'):
                continue

            if strategy_name and strategy_name not in filename:
                continue

            filepath = os.path.join(self.learnings_dir, filename)
            with open(filepath, 'r') as f:
                report = json.load(f)

            # Check if within date range
            report_time = datetime.fromisoformat(report['timestamp'].replace('Z', '+00:00'))
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)

            if report_time >= cutoff:
                reports.append(report)

        return sorted(reports, key=lambda x: x['timestamp'], reverse=True)


def main():
    """Run learning cycles for all strategies."""
    learner = StrategyLearner()

    strategies = [
        ("Baseline", "baseline"),
        ("Adaptive_Baseline", "baseline"),
        ("Adaptive_ProgPos_Only", "baseline"),
        ("Conservative", "baseline"),
    ]

    for strategy_name, entry_filter in strategies:
        log(f"\n{'='*60}")
        log(f"Learning: {strategy_name}")
        log(f"{'='*60}")

        report = learner.run_learning_cycle(
            strategy_name,
            entry_filter,
            apply=False  # Dry run by default
        )

        print(f"\n{report.notes}")


if __name__ == "__main__":
    main()
