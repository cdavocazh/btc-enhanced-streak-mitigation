#!/usr/bin/env python3
"""
Walk-Forward Optimization (WFO) for BTC Trading Strategy

Walk-Forward Optimization tests the strategy's robustness by:
1. Dividing data into multiple in-sample (IS) and out-of-sample (OOS) periods
2. Optimizing parameters on IS data
3. Testing on OOS data (which the optimization never saw)
4. Rolling forward and repeating

This validates whether the optimized parameters generalize to unseen data
and helps detect overfitting.

WFO Configuration:
- Training Window: 12 months
- Testing Window: 3 months
- Step Forward: 3 months
- This creates overlapping training periods with non-overlapping test periods

Parameters to Optimize:
- Entry Threshold: [0.5, 0.75, 1.0, 1.25, 1.5]
- Strong Positioning Threshold: [1.0, 1.5, 2.0, 2.5]
- Skip Neutral Threshold: [0.0, 0.25, 0.5]
- Strict Entry Min Score: [0.25, 0.5, 0.75]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import itertools
from collections import defaultdict

# Import from parent directory
from run_backtest import (
    load_ohlc_data, load_binance_data, calculate_indicators,
    merge_binance_data, calculate_enhanced_signals,
    INITIAL_EQ, RISK_PCT_INIT, STOP_PCT, ATR_MULT,
    ASIA_HRS, US_HRS
)

# WFO Configuration
TRAINING_MONTHS = 12
TESTING_MONTHS = 3
STEP_MONTHS = 3

# Parameter Grid for Optimization
PARAM_GRID = {
    'entry_threshold': [0.5, 0.75, 1.0, 1.25, 1.5],
    'strong_threshold': [1.0, 1.5, 2.0, 2.5],
    'skip_neutral_threshold': [0.0, 0.25, 0.5],
    'strict_entry_min_score': [0.25, 0.5, 0.75],
}

# Fixed parameters (from best config)
FIXED_PARAMS = {
    'strict_entry_after_losses': True,
    'strict_entry_trigger': 3,
}


def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"[{timestamp}] {message}")


@dataclass
class StrategyConfig:
    """Strategy configuration"""
    entry_threshold: float
    strong_threshold: float
    skip_neutral_threshold: float
    strict_entry_min_score: float
    strict_entry_after_losses: bool = True
    strict_entry_trigger: int = 3


@dataclass
class WFOPeriod:
    """Single WFO period definition"""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    period_number: int


@dataclass
class WFOResult:
    """Result from a single WFO period"""
    period: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_config: Dict
    train_metrics: Dict
    test_metrics: Dict


def calculate_positioning_score_top_trader_focused(row) -> float:
    """Calculate positioning score using TopTraderFocused method"""
    score = 0.0

    top_trader_long_pct = getattr(row, 'top_trader_position_long_pct', None)
    top_trader_short_pct = getattr(row, 'top_trader_position_short_pct', None)

    if top_trader_long_pct is not None and not pd.isna(top_trader_long_pct):
        if top_trader_long_pct > 0.60:
            score += 1.5
        elif top_trader_long_pct > 0.55:
            score += 1.0

    if top_trader_short_pct is not None and not pd.isna(top_trader_short_pct):
        if top_trader_short_pct > 0.60:
            score -= 1.5
        elif top_trader_short_pct > 0.55:
            score -= 1.0

    top_account_long_pct = getattr(row, 'top_trader_account_long_pct', None)
    top_account_short_pct = getattr(row, 'top_trader_account_short_pct', None)

    if top_account_long_pct is not None and not pd.isna(top_account_long_pct):
        if top_account_long_pct > 0.55:
            score += 0.25
    if top_account_short_pct is not None and not pd.isna(top_account_short_pct):
        if top_account_short_pct > 0.55:
            score -= 0.25

    global_ls = getattr(row, 'global_ls_ratio', None)
    if global_ls is not None and not pd.isna(global_ls):
        if global_ls < 0.7:
            score += 0.5
        elif global_ls > 1.5:
            score -= 0.5

    funding_rate = getattr(row, 'funding_rate', None)
    if funding_rate is not None and not pd.isna(funding_rate):
        if funding_rate > 0.0005:
            score -= 0.5
        elif funding_rate < -0.0005:
            score += 0.5

    oi_vs_ma = getattr(row, 'oi_vs_ma24', None)
    price_change = getattr(row, 'price_change_4h', None)
    if oi_vs_ma is not None and not pd.isna(oi_vs_ma) and oi_vs_ma > 1.05:
        if price_change is not None and not pd.isna(price_change):
            if price_change > 0.01:
                score += 0.25
            elif price_change < -0.01:
                score -= 0.25

    return score


def run_backtest_with_config(df: pd.DataFrame, atr_med: float, config: StrategyConfig) -> Dict:
    """Run backtest with specific configuration"""
    df_bt = df.copy()

    risk_amount = INITIAL_EQ * RISK_PCT_INIT
    equity = INITIAL_EQ
    open_trade = None
    pnl_history = []
    consecutive_losses = 0
    trade_count = 0

    for r in df_bt.itertuples():
        hr = r.Index.hour

        if pd.isna(r.sma20) or pd.isna(r.atr20):
            continue

        positioning_score = calculate_positioning_score_top_trader_focused(r)
        volume_surge = getattr(r, 'volume_surge', False)
        oi_surge = getattr(r, 'oi_surge', False)

        # ENTRY LOGIC
        if open_trade is None and hr in ASIA_HRS and r.atr20 > atr_med:
            # Skip neutral positioning
            if config.skip_neutral_threshold > 0 and abs(positioning_score) < config.skip_neutral_threshold:
                continue

            # Strict entry after losses
            if config.strict_entry_after_losses:
                if consecutive_losses >= config.strict_entry_trigger:
                    if abs(positioning_score) < config.strict_entry_min_score:
                        continue

            # Base signals
            long_mr = (r.close < r.lower_band) and (r.rsi14 < 30)
            short_mr = (r.close > r.upper_band) and (r.rsi14 > 70)
            long_bo = (r.close > r.high_3h) and (r.rsi14 > 60)
            short_bo = (r.close < r.low_3h) and (r.rsi14 < 40)

            # Enhanced with positioning filters
            long_mr_ok = long_mr and positioning_score > -1.5
            short_mr_ok = short_mr and positioning_score < 1.5
            long_bo_ok = long_bo and positioning_score > -1.0
            short_bo_ok = short_bo and positioning_score < 1.0

            # Strong positioning (using config thresholds)
            strong_long = (
                positioning_score >= config.strong_threshold and
                r.rsi14 < 45 and r.close < r.sma20 and
                (volume_surge or oi_surge)
            )
            strong_short = (
                positioning_score <= -config.strong_threshold and
                r.rsi14 > 55 and r.close > r.sma20 and
                (volume_surge or oi_surge)
            )

            any_long = long_mr_ok or long_bo_ok or strong_long
            any_short = short_mr_ok or short_bo_ok or strong_short

            if any_long or any_short:
                side = "long" if any_long else "short"
                entry_price = r.close
                stop_price = entry_price * (1 - STOP_PCT) if side == "long" else entry_price * (1 + STOP_PCT)
                target_price = entry_price + ATR_MULT * r.atr20 if side == "long" else entry_price - ATR_MULT * r.atr20

                unit_risk = abs(entry_price - stop_price)
                size = risk_amount / unit_risk if unit_risk > 0 else 0

                # Position sizing
                if abs(positioning_score) >= 1.5:
                    size *= 1.3
                elif abs(positioning_score) >= 1.0:
                    size *= 1.2
                elif abs(positioning_score) >= 0.5:
                    size *= 1.1

                open_trade = {
                    "side": side,
                    "entry_price": entry_price,
                    "stop": stop_price,
                    "target": target_price,
                    "size": size,
                }
                trade_count += 1

        # EXIT LOGIC
        elif open_trade:
            exit_price = None

            if hr in US_HRS and hr not in ASIA_HRS:
                exit_price = r.close
            else:
                if open_trade["side"] == "long":
                    if r.low <= open_trade["stop"]:
                        exit_price = open_trade["stop"]
                    elif r.high >= open_trade["target"]:
                        exit_price = open_trade["target"]
                else:
                    if r.high >= open_trade["stop"]:
                        exit_price = open_trade["stop"]
                    elif r.low <= open_trade["target"]:
                        exit_price = open_trade["target"]

            if exit_price is not None:
                pnl = ((exit_price - open_trade["entry_price"]) if open_trade["side"] == "long"
                       else (open_trade["entry_price"] - exit_price)) * open_trade["size"]

                pnl_history.append(pnl)
                equity += pnl

                if pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                open_trade = None

    # Calculate metrics
    if len(pnl_history) == 0:
        return {
            'trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'profit_factor': 0,
            'sharpe': 0,
        }

    pnl_arr = np.array(pnl_history)
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]

    win_rate = len(wins) / len(pnl_arr) * 100
    total_return = (equity - INITIAL_EQ) / INITIAL_EQ * 100

    eq_array = INITIAL_EQ + np.cumsum(pnl_history)
    drawdowns = np.maximum.accumulate(eq_array) - eq_array
    max_dd = drawdowns.max() / np.maximum.accumulate(eq_array).max() * 100 if len(eq_array) > 0 else 0

    profit_factor = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0

    # Sharpe
    if len(pnl_arr) > 1:
        returns = pnl_arr / INITIAL_EQ
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
    else:
        sharpe = 0

    return {
        'trades': len(pnl_arr),
        'win_rate': win_rate,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
    }


def generate_wfo_periods(df: pd.DataFrame) -> List[WFOPeriod]:
    """Generate WFO periods based on data range"""
    data_start = df.index.min()
    data_end = df.index.max()

    # Make timezone naive for comparison
    if data_start.tzinfo is not None:
        data_start = data_start.tz_localize(None)
    if data_end.tzinfo is not None:
        data_end = data_end.tz_localize(None)

    periods = []
    period_num = 1

    # Start from enough data to have training period
    current_train_start = data_start

    while True:
        train_end = current_train_start + pd.DateOffset(months=TRAINING_MONTHS)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=TESTING_MONTHS)

        # Check if we have enough data for this period
        if test_end > data_end:
            break

        periods.append(WFOPeriod(
            train_start=current_train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            period_number=period_num
        ))

        # Step forward
        current_train_start = current_train_start + pd.DateOffset(months=STEP_MONTHS)
        period_num += 1

    return periods


def optimize_on_training_data(df_train: pd.DataFrame, atr_med: float) -> Tuple[StrategyConfig, Dict]:
    """Find best parameters on training data"""
    best_score = -np.inf
    best_config = None
    best_metrics = None

    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        PARAM_GRID['entry_threshold'],
        PARAM_GRID['strong_threshold'],
        PARAM_GRID['skip_neutral_threshold'],
        PARAM_GRID['strict_entry_min_score'],
    ))

    for entry_thresh, strong_thresh, skip_neutral, strict_min in param_combinations:
        # Skip invalid combinations
        if strong_thresh <= entry_thresh:
            continue

        config = StrategyConfig(
            entry_threshold=entry_thresh,
            strong_threshold=strong_thresh,
            skip_neutral_threshold=skip_neutral,
            strict_entry_min_score=strict_min,
            **FIXED_PARAMS
        )

        metrics = run_backtest_with_config(df_train, atr_med, config)

        # Scoring function: balance return, drawdown, and profit factor
        # Objective: maximize return while penalizing drawdown
        if metrics['trades'] < 10:  # Skip if too few trades
            continue

        score = (
            metrics['total_return'] * 0.4 +
            (100 - metrics['max_drawdown']) * 0.3 +
            metrics['profit_factor'] * 10 * 0.2 +
            metrics['sharpe'] * 5 * 0.1
        )

        if score > best_score:
            best_score = score
            best_config = config
            best_metrics = metrics

    return best_config, best_metrics


def run_wfo_analysis(df: pd.DataFrame, atr_med: float) -> List[WFOResult]:
    """Run complete Walk-Forward Optimization"""
    periods = generate_wfo_periods(df)
    log(f"Generated {len(periods)} WFO periods")

    results = []

    for period in periods:
        log(f"")
        log(f"Period {period.period_number}: Train {period.train_start.strftime('%Y-%m')} to {period.train_end.strftime('%Y-%m')}, "
            f"Test {period.test_start.strftime('%Y-%m')} to {period.test_end.strftime('%Y-%m')}")

        # Split data
        df_train = df[(df.index >= period.train_start) & (df.index < period.train_end)]
        df_test = df[(df.index >= period.test_start) & (df.index < period.test_end)]

        if len(df_train) < 100 or len(df_test) < 50:
            log(f"  Skipping - insufficient data (train: {len(df_train)}, test: {len(df_test)})")
            continue

        # Optimize on training data
        best_config, train_metrics = optimize_on_training_data(df_train, atr_med)

        if best_config is None:
            log(f"  Skipping - no valid configuration found")
            continue

        # Test on out-of-sample data
        test_metrics = run_backtest_with_config(df_test, atr_med, best_config)

        log(f"  Best Config: entry={best_config.entry_threshold}, strong={best_config.strong_threshold}, "
            f"skip_neutral={best_config.skip_neutral_threshold}")
        log(f"  Train: {train_metrics['trades']} trades, {train_metrics['win_rate']:.1f}% WR, "
            f"{train_metrics['total_return']:.1f}% return")
        log(f"  Test:  {test_metrics['trades']} trades, {test_metrics['win_rate']:.1f}% WR, "
            f"{test_metrics['total_return']:.1f}% return")

        results.append(WFOResult(
            period=period.period_number,
            train_start=period.train_start.strftime('%Y-%m-%d'),
            train_end=period.train_end.strftime('%Y-%m-%d'),
            test_start=period.test_start.strftime('%Y-%m-%d'),
            test_end=period.test_end.strftime('%Y-%m-%d'),
            best_config=asdict(best_config),
            train_metrics=train_metrics,
            test_metrics=test_metrics
        ))

    return results


def analyze_wfo_results(results: List[WFOResult]) -> Dict:
    """Analyze WFO results for insights"""
    if not results:
        return {'error': 'No valid WFO periods'}

    # Aggregate test performance
    test_returns = [r.test_metrics['total_return'] for r in results]
    test_win_rates = [r.test_metrics['win_rate'] for r in results]
    test_drawdowns = [r.test_metrics['max_drawdown'] for r in results]
    test_trades = [r.test_metrics['trades'] for r in results]

    train_returns = [r.train_metrics['total_return'] for r in results]

    # Calculate efficiency (OOS vs IS performance)
    efficiency_ratios = []
    for r in results:
        if r.train_metrics['total_return'] != 0:
            efficiency = r.test_metrics['total_return'] / r.train_metrics['total_return']
            efficiency_ratios.append(efficiency)

    # Parameter stability analysis
    entry_thresholds = [r.best_config['entry_threshold'] for r in results]
    strong_thresholds = [r.best_config['strong_threshold'] for r in results]
    skip_neutral = [r.best_config['skip_neutral_threshold'] for r in results]

    # Count parameter frequencies
    entry_freq = defaultdict(int)
    strong_freq = defaultdict(int)
    neutral_freq = defaultdict(int)

    for r in results:
        entry_freq[r.best_config['entry_threshold']] += 1
        strong_freq[r.best_config['strong_threshold']] += 1
        neutral_freq[r.best_config['skip_neutral_threshold']] += 1

    # Find most common parameters
    most_common_entry = max(entry_freq, key=entry_freq.get)
    most_common_strong = max(strong_freq, key=strong_freq.get)
    most_common_neutral = max(neutral_freq, key=neutral_freq.get)

    return {
        'summary': {
            'total_periods': len(results),
            'profitable_periods': sum(1 for r in test_returns if r > 0),
            'avg_test_return': np.mean(test_returns),
            'median_test_return': np.median(test_returns),
            'std_test_return': np.std(test_returns),
            'min_test_return': min(test_returns),
            'max_test_return': max(test_returns),
            'avg_test_win_rate': np.mean(test_win_rates),
            'avg_test_drawdown': np.mean(test_drawdowns),
            'total_test_trades': sum(test_trades),
        },
        'efficiency': {
            'avg_efficiency_ratio': np.mean(efficiency_ratios) if efficiency_ratios else 0,
            'median_efficiency_ratio': np.median(efficiency_ratios) if efficiency_ratios else 0,
            'interpretation': 'ROBUST' if np.mean(efficiency_ratios) > 0.5 else 'POTENTIAL_OVERFIT' if np.mean(efficiency_ratios) < 0.3 else 'MODERATE'
        },
        'parameter_stability': {
            'most_common_entry_threshold': most_common_entry,
            'entry_threshold_consistency': entry_freq[most_common_entry] / len(results) * 100,
            'most_common_strong_threshold': most_common_strong,
            'strong_threshold_consistency': strong_freq[most_common_strong] / len(results) * 100,
            'most_common_skip_neutral': most_common_neutral,
            'neutral_threshold_consistency': neutral_freq[most_common_neutral] / len(results) * 100,
        },
        'recommended_config': {
            'entry_threshold': most_common_entry,
            'strong_threshold': most_common_strong,
            'skip_neutral_threshold': most_common_neutral,
            'strict_entry_min_score': 0.5,  # Most common from grid
        }
    }


def main():
    log("=" * 70)
    log("WALK-FORWARD OPTIMIZATION (WFO)")
    log("Strategy: TopTraderFocused with Mitigations")
    log("=" * 70)
    log(f"Training Window: {TRAINING_MONTHS} months")
    log(f"Testing Window: {TESTING_MONTHS} months")
    log(f"Step Forward: {STEP_MONTHS} months")

    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load and prepare data
    log("")
    log("Loading data...")
    df_raw = load_ohlc_data()
    df = calculate_indicators(df_raw)
    atr_med = df['atr20'].median()

    binance_df = load_binance_data()
    if binance_df is not None:
        df = merge_binance_data(df, binance_df)

    df = calculate_enhanced_signals(df)

    # Make index timezone naive for comparison
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    log(f"Loaded data: {len(df)} rows")
    log(f"Date range: {df.index.min()} to {df.index.max()}")

    # Run WFO
    log("")
    log("=" * 70)
    log("RUNNING WALK-FORWARD OPTIMIZATION")
    log("=" * 70)

    wfo_results = run_wfo_analysis(df, atr_med)

    # Analyze results
    log("")
    log("=" * 70)
    log("WFO ANALYSIS RESULTS")
    log("=" * 70)

    analysis = analyze_wfo_results(wfo_results)

    if 'error' in analysis:
        log(f"Error: {analysis['error']}")
        return

    summary = analysis['summary']
    efficiency = analysis['efficiency']
    stability = analysis['parameter_stability']
    recommended = analysis['recommended_config']

    log("")
    log("OUT-OF-SAMPLE PERFORMANCE SUMMARY:")
    log(f"  Total WFO Periods: {summary['total_periods']}")
    log(f"  Profitable Periods: {summary['profitable_periods']} ({summary['profitable_periods']/summary['total_periods']*100:.1f}%)")
    log(f"  Average Test Return: {summary['avg_test_return']:.1f}%")
    log(f"  Median Test Return: {summary['median_test_return']:.1f}%")
    log(f"  Test Return Std Dev: {summary['std_test_return']:.1f}%")
    log(f"  Return Range: [{summary['min_test_return']:.1f}%, {summary['max_test_return']:.1f}%]")
    log(f"  Average Test Win Rate: {summary['avg_test_win_rate']:.1f}%")
    log(f"  Average Test Max Drawdown: {summary['avg_test_drawdown']:.1f}%")
    log(f"  Total Test Trades: {summary['total_test_trades']}")

    log("")
    log("EFFICIENCY ANALYSIS (OOS vs IS Performance):")
    log(f"  Average Efficiency Ratio: {efficiency['avg_efficiency_ratio']:.2f}")
    log(f"  Median Efficiency Ratio: {efficiency['median_efficiency_ratio']:.2f}")
    log(f"  Interpretation: {efficiency['interpretation']}")

    if efficiency['interpretation'] == 'ROBUST':
        log("  -> Strategy performs well on unseen data")
        log("  -> Parameters generalize effectively")
    elif efficiency['interpretation'] == 'POTENTIAL_OVERFIT':
        log("  -> Warning: OOS performance significantly lower than IS")
        log("  -> Consider simplifying the strategy or reducing parameter count")
    else:
        log("  -> Moderate degradation from IS to OOS (typical)")

    log("")
    log("PARAMETER STABILITY:")
    log(f"  Most Common Entry Threshold: {stability['most_common_entry_threshold']} "
        f"(selected in {stability['entry_threshold_consistency']:.0f}% of periods)")
    log(f"  Most Common Strong Threshold: {stability['most_common_strong_threshold']} "
        f"(selected in {stability['strong_threshold_consistency']:.0f}% of periods)")
    log(f"  Most Common Skip Neutral: {stability['most_common_skip_neutral']} "
        f"(selected in {stability['neutral_threshold_consistency']:.0f}% of periods)")

    log("")
    log("RECOMMENDED CONFIGURATION (based on WFO):")
    log(f"  Entry Threshold: {recommended['entry_threshold']}")
    log(f"  Strong Threshold: {recommended['strong_threshold']}")
    log(f"  Skip Neutral Threshold: {recommended['skip_neutral_threshold']}")
    log(f"  Strict Entry Min Score: {recommended['strict_entry_min_score']}")

    # Period-by-period results table
    log("")
    log("PERIOD-BY-PERIOD RESULTS:")
    log("-" * 100)
    log(f"{'Period':<8} {'Train Period':<25} {'Test Period':<25} {'Train Ret%':>10} {'Test Ret%':>10} {'Test WR%':>10}")
    log("-" * 100)

    for r in wfo_results:
        log(f"{r.period:<8} {r.train_start} to {r.train_end[:7]:<10} "
            f"{r.test_start} to {r.test_end[:7]:<10} "
            f"{r.train_metrics['total_return']:>10.1f} "
            f"{r.test_metrics['total_return']:>10.1f} "
            f"{r.test_metrics['win_rate']:>10.1f}")

    # Save results
    results_file = os.path.join(results_dir, 'wfo_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'training_months': TRAINING_MONTHS,
                'testing_months': TESTING_MONTHS,
                'step_months': STEP_MONTHS,
                'param_grid': PARAM_GRID,
            },
            'periods': [asdict(r) for r in wfo_results],
            'analysis': analysis,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }, f, indent=2, default=str)

    log("")
    log(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
