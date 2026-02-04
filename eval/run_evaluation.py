#!/usr/bin/env python3
"""
BTC Strategy Evaluation Runner
===============================
Main scheduler and orchestrator for running strategy evaluations.

Three Execution Modes:
1. Quick Check (--quick): Lightweight performance snapshot
2. Full Evaluation (--full): Complete walk-forward analysis
3. Adaptation Check (--adapt): Run learning cycles

Usage:
    python run_evaluation.py --quick
    python run_evaluation.py --full --training-days 14 --testing-days 7
    python run_evaluation.py --adapt --apply
    python run_evaluation.py --scheduled
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
from dataclasses import asdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'backtest_15min'))
sys.path.insert(0, os.path.join(REPO_DIR, 'backtest_15min_new'))
sys.path.insert(0, os.path.join(REPO_DIR, 'backtest_15min_new_streak_a'))

from config import (
    EvaluationSchedule, DEFAULT_SCHEDULE, WalkForwardConfig,
    TIERED_STRATEGY_CONFIGS, ADAPTIVE_STRATEGY_CONFIGS,
    ENTRY_FILTER_CONFIGS, get_all_strategies
)
from walk_forward_engine import WalkForwardEngine, WalkForwardResult
from performance_tracker import PerformanceTracker, PerformanceSnapshot
from strategy_learner import StrategyLearner


def log(msg: str):
    """Print timestamped log message."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}")


class EvaluationRunner:
    """
    Main evaluation runner and scheduler.
    """

    def __init__(self, schedule: EvaluationSchedule = None):
        self.schedule = schedule or DEFAULT_SCHEDULE
        self.last_run_file = os.path.join(SCRIPT_DIR, 'last_run.json')
        self.results_dir = os.path.join(SCRIPT_DIR, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

        self.engine = WalkForwardEngine()
        self.tracker = PerformanceTracker()
        self.learner = StrategyLearner(self.tracker)

    def get_last_run_times(self) -> Dict[str, Optional[str]]:
        """Get last run times for each evaluation type."""
        if os.path.exists(self.last_run_file):
            with open(self.last_run_file, 'r') as f:
                return json.load(f)
        return {'quick': None, 'full': None, 'adapt': None}

    def update_last_run(self, eval_type: str):
        """Update last run time for an evaluation type."""
        times = self.get_last_run_times()
        times[eval_type] = datetime.now(timezone.utc).isoformat()
        with open(self.last_run_file, 'w') as f:
            json.dump(times, f, indent=2)

    def should_run(self, eval_type: str) -> bool:
        """Check if an evaluation type should run based on schedule."""
        times = self.get_last_run_times()
        last_run = times.get(eval_type)

        if last_run is None:
            return True

        last_time = datetime.fromisoformat(last_run.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)

        if eval_type == 'quick':
            interval = self.schedule.quick_check_interval
        elif eval_type == 'full':
            interval = self.schedule.full_evaluation_interval
        elif eval_type == 'adapt':
            interval = self.schedule.adaptation_check_interval
        else:
            return True

        return (now - last_time) >= interval

    def run_quick_check(self) -> Dict[str, Any]:
        """
        Run a quick performance check.

        - Lightweight snapshot of current performance
        - Compare to historical baseline
        - Check for regime changes
        """
        log("=" * 60)
        log("QUICK PERFORMANCE CHECK")
        log("=" * 60)

        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'quick',
            'strategies': {}
        }

        # Strategies to check
        strategies = [
            ("Baseline", "baseline"),
            ("Adaptive_Baseline", "baseline"),
            ("Adaptive_ProgPos_Only", "baseline"),
            ("Conservative", "baseline"),
        ]

        for strategy_name, entry_filter in strategies:
            log(f"\nChecking: {strategy_name}")

            # Get rolling metrics
            metrics = self.tracker.get_rolling_metrics(strategy_name, entry_filter, window_days=7)

            if not metrics:
                log(f"  No historical data available")
                continue

            log(f"  7-day avg return: {metrics.get('avg_return', 0):.2f}%")
            log(f"  7-day avg Sharpe: {metrics.get('avg_sharpe', 0):.2f}")
            log(f"  Max drawdown: {metrics.get('max_drawdown', 0):.2f}%")
            log(f"  Total trades: {metrics.get('total_trades', 0)}")

            # Check for regime change
            regime = self.tracker.detect_regime_change(strategy_name, entry_filter)
            if regime:
                log(f"  REGIME CHANGE DETECTED: {regime['regime']} (confidence: {regime['confidence']:.2%})")

            results['strategies'][strategy_name] = {
                'metrics': metrics,
                'regime_change': regime
            }

        # Get recent alerts
        alerts = self.tracker.get_recent_alerts(days=1)
        results['recent_alerts'] = [
            {'strategy': a['strategy_name'], 'severity': a['severity'], 'message': a['message']}
            for a in alerts
        ]

        if alerts:
            log(f"\nRecent alerts: {len(alerts)}")
            for a in alerts[:5]:
                log(f"  [{a['severity']}] {a['strategy_name']}: {a['message']}")

        self.update_last_run('quick')

        # Save results
        self._save_results('quick', results)

        return results

    def run_full_evaluation(
        self,
        training_days: int = 14,
        testing_days: int = 7
    ) -> Dict[str, Any]:
        """
        Run a full walk-forward evaluation.

        - Complete WFO for all strategies
        - Record performance snapshots
        - Generate detailed reports
        """
        log("=" * 60)
        log("FULL WALK-FORWARD EVALUATION")
        log(f"Training: {training_days} days, Testing: {testing_days} days")
        log("=" * 60)

        # Update config
        config = WalkForwardConfig(
            training_window=96 * training_days,
            testing_window=96 * testing_days,
            step_size=96 * testing_days
        )
        self.engine.config = config

        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'full',
            'config': {
                'training_days': training_days,
                'testing_days': testing_days
            },
            'strategies': {}
        }

        # Load data once
        try:
            self.engine.load_data()
        except Exception as e:
            log(f"Error loading data: {e}")
            return results

        # Strategies to evaluate
        strategies = [
            ("Baseline", "baseline"),
            ("MultiTP_30", "baseline"),
            ("Conservative", "baseline"),
            ("Adaptive_Baseline", "baseline"),
            ("Adaptive_ProgPos_Only", "baseline"),
            ("Adaptive_Conservative", "baseline"),
        ]

        for strategy_name, entry_filter in strategies:
            log(f"\n{'='*50}")
            log(f"Evaluating: {strategy_name}")
            log(f"{'='*50}")

            wf_result = self.engine.run_walk_forward(strategy_name, entry_filter)

            if wf_result is None:
                log(f"  Walk-forward failed for {strategy_name}")
                continue

            # Record snapshot
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                strategy_name=strategy_name,
                entry_filter=entry_filter,
                total_return_pct=wf_result.overall_oos_return,
                max_drawdown_pct=np.mean([w.get('oos_max_dd', 0) for w in wf_result.windows]),
                sharpe_ratio=np.mean([w.get('oos_sharpe', 0) for w in wf_result.windows]),
                win_rate_pct=0,  # Not tracked per window
                profit_factor=0,  # Not tracked per window
                total_trades=sum(w.get('oos_trades', 0) for w in wf_result.windows),
                max_losing_streak=0,
                oos_return_pct=wf_result.overall_oos_return,
                oos_efficiency=wf_result.oos_efficiency,
                parameter_stability=wf_result.parameter_stability,
                data_bars=wf_result.data_bars,
                evaluation_type="full"
            )
            self.tracker.record_snapshot(snapshot)

            results['strategies'][strategy_name] = {
                'is_return': wf_result.overall_is_return,
                'oos_return': wf_result.overall_oos_return,
                'oos_efficiency': wf_result.oos_efficiency,
                'parameter_stability': wf_result.parameter_stability,
                'status': wf_result.status,
                'windows': len(wf_result.windows)
            }

            log(f"  IS Return: {wf_result.overall_is_return:.2f}%")
            log(f"  OOS Return: {wf_result.overall_oos_return:.2f}%")
            log(f"  OOS Efficiency: {wf_result.oos_efficiency:.2f}")
            log(f"  Status: {wf_result.status}")

        self.update_last_run('full')

        # Save results
        self._save_results('full', results)

        # Generate summary report
        self._generate_evaluation_report(results)

        return results

    def run_adaptation_check(self, apply: bool = False) -> Dict[str, Any]:
        """
        Run adaptation check for all strategies.

        - Analyze performance trends
        - Propose parameter adjustments
        - Optionally apply changes
        """
        log("=" * 60)
        log("ADAPTATION CHECK")
        log(f"Apply changes: {apply}")
        log("=" * 60)

        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'type': 'adapt',
            'apply': apply,
            'strategies': {}
        }

        strategies = [
            ("Baseline", "baseline"),
            ("Adaptive_Baseline", "baseline"),
            ("Adaptive_ProgPos_Only", "baseline"),
            ("Conservative", "baseline"),
        ]

        for strategy_name, entry_filter in strategies:
            log(f"\n{'='*50}")
            log(f"Learning: {strategy_name}")
            log(f"{'='*50}")

            report = self.learner.run_learning_cycle(
                strategy_name,
                entry_filter,
                apply=apply
            )

            results['strategies'][strategy_name] = {
                'trend_direction': report.trend.trend_direction,
                'trend_confidence': report.trend.confidence,
                'regime_change': report.regime_change,
                'adjustments_proposed': len(report.proposed_adjustments),
                'applied': report.applied
            }

            print(f"\n{report.notes}")

        self.update_last_run('adapt')

        # Save results
        self._save_results('adapt', results)

        return results

    def run_scheduled(self) -> Dict[str, Any]:
        """
        Run evaluations according to schedule.

        Checks what should run and executes accordingly.
        """
        log("=" * 60)
        log("SCHEDULED EVALUATION RUN")
        log("=" * 60)

        results = {'ran': []}

        if self.should_run('quick'):
            log("Running scheduled quick check...")
            self.run_quick_check()
            results['ran'].append('quick')
        else:
            log("Quick check not due yet")

        if self.should_run('full'):
            log("Running scheduled full evaluation...")
            self.run_full_evaluation()
            results['ran'].append('full')
        else:
            log("Full evaluation not due yet")

        if self.should_run('adapt'):
            log("Running scheduled adaptation check...")
            self.run_adaptation_check(apply=False)
            results['ran'].append('adapt')
        else:
            log("Adaptation check not due yet")

        return results

    def _save_results(self, eval_type: str, results: Dict[str, Any]):
        """Save evaluation results to JSON."""
        filename = f"{eval_type}_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        log(f"Results saved: {filepath}")

    def _generate_evaluation_report(self, results: Dict[str, Any]):
        """Generate a markdown evaluation report."""
        filename = f"EVALUATION_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(SCRIPT_DIR, filename)

        lines = [
            f"# BTC Strategy Evaluation Report",
            f"",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Type:** Full Walk-Forward Evaluation",
            f"**Training:** {results['config']['training_days']} days",
            f"**Testing:** {results['config']['testing_days']} days",
            f"",
            f"## Strategy Performance Summary",
            f"",
            f"| Strategy | IS Return | OOS Return | OOS Efficiency | Status |",
            f"|----------|-----------|------------|----------------|--------|",
        ]

        for name, data in results['strategies'].items():
            status_emoji = {"HEALTHY": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}.get(data.get('status', ''), '⚪')
            lines.append(
                f"| {name} | {data.get('is_return', 0):.2f}% | {data.get('oos_return', 0):.2f}% | "
                f"{data.get('oos_efficiency', 0):.2f} | {status_emoji} {data.get('status', 'N/A')} |"
            )

        lines.extend([
            f"",
            f"## Key Findings",
            f"",
        ])

        # Add findings based on results
        healthy = [n for n, d in results['strategies'].items() if d.get('status') == 'HEALTHY']
        warning = [n for n, d in results['strategies'].items() if d.get('status') == 'WARNING']
        critical = [n for n, d in results['strategies'].items() if d.get('status') == 'CRITICAL']

        if healthy:
            lines.append(f"- **Healthy strategies:** {', '.join(healthy)}")
        if warning:
            lines.append(f"- **Warning strategies:** {', '.join(warning)}")
        if critical:
            lines.append(f"- **Critical strategies:** {', '.join(critical)}")

        lines.extend([
            f"",
            f"## Next Steps",
            f"",
            f"1. Review any WARNING or CRITICAL strategies",
            f"2. Run `python eval/run_evaluation.py --adapt` to analyze trends",
            f"3. Consider parameter adjustments for declining strategies",
            f"",
            f"---",
            f"*Generated by BTC Strategy Evaluation Framework*",
        ])

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        log(f"Report generated: {filepath}")


# Need numpy for averaging
import numpy as np


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="BTC Strategy Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_evaluation.py --quick          # Quick performance check
    python run_evaluation.py --full           # Full walk-forward evaluation
    python run_evaluation.py --adapt          # Learning cycle (dry run)
    python run_evaluation.py --adapt --apply  # Learning cycle (apply changes)
    python run_evaluation.py --scheduled      # Run based on schedule
    python run_evaluation.py --summary        # Show current performance summary
        """
    )

    parser.add_argument('--quick', action='store_true', help='Run quick performance check')
    parser.add_argument('--full', action='store_true', help='Run full walk-forward evaluation')
    parser.add_argument('--adapt', action='store_true', help='Run adaptation/learning check')
    parser.add_argument('--apply', action='store_true', help='Apply parameter adjustments (with --adapt)')
    parser.add_argument('--scheduled', action='store_true', help='Run based on schedule')
    parser.add_argument('--summary', action='store_true', help='Show performance summary')
    parser.add_argument('--training-days', type=int, default=14, help='Training window in days (default: 14)')
    parser.add_argument('--testing-days', type=int, default=7, help='Testing window in days (default: 7)')

    args = parser.parse_args()

    runner = EvaluationRunner()

    if args.summary:
        summary = runner.tracker.get_performance_summary()
        log("Performance Summary:")
        for key, data in summary.items():
            log(f"  {key}: Return={data.get('total_return_pct', 0):.2f}%, "
                f"Sharpe={data.get('sharpe_ratio', 0):.2f}")

    elif args.quick:
        runner.run_quick_check()

    elif args.full:
        runner.run_full_evaluation(
            training_days=args.training_days,
            testing_days=args.testing_days
        )

    elif args.adapt:
        runner.run_adaptation_check(apply=args.apply)

    elif args.scheduled:
        runner.run_scheduled()

    else:
        # Default: show help
        parser.print_help()


if __name__ == "__main__":
    main()
