#!/usr/bin/env python3
"""
Monte Carlo Shuffle Validation for BTC Trading Strategy

This script performs Monte Carlo analysis by shuffling the order of trade returns
to assess whether the strategy's performance is statistically significant or
potentially due to the specific sequence of trades.

Approach:
1. Run the optimized strategy to get the actual trade sequence
2. Shuffle the PnL sequence many times (N iterations)
3. Calculate key metrics for each shuffled sequence
4. Compare actual results against the distribution of shuffled results
5. Compute p-values and confidence intervals

Key Metrics Analyzed:
- Total Return
- Maximum Drawdown
- Sharpe Ratio (if applicable)
- Maximum Consecutive Losses
- Win Rate (constant, for reference)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import itertools

# Import from parent directory
from run_backtest import (
    load_ohlc_data, load_binance_data, calculate_indicators,
    merge_binance_data, calculate_enhanced_signals,
    INITIAL_EQ, RISK_PCT_INIT, STOP_PCT, ATR_MULT,
    ASIA_HRS, US_HRS
)

# Optimized Strategy Parameters (from experiments)
OPTIMIZED_CONFIG = {
    'name': 'Optimized_TopTraderFocused',
    'entry_threshold': 1.5,
    'strong_threshold': 2.0,
    'skip_neutral_threshold': 0.25,
    'strict_entry_after_losses': True,
    'strict_entry_trigger': 3,
    'strict_entry_min_score': 0.5,
}

# Monte Carlo Parameters
NUM_SIMULATIONS = 1000
CONFIDENCE_LEVELS = [0.95, 0.99]


def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"[{timestamp}] {message}")


@dataclass
class TradeResult:
    """Single trade result"""
    pnl: float
    entry_time: str
    exit_time: str
    side: str
    entry_type: str
    exit_reason: str
    positioning_score: float


def calculate_positioning_score_top_trader_focused(row) -> float:
    """
    Calculate positioning score using TopTraderFocused method
    (Higher weight on top trader signals)
    """
    score = 0.0

    # Top Trader Position (strongest signal - increased weight)
    top_trader_long_pct = getattr(row, 'top_trader_position_long_pct', None)
    top_trader_short_pct = getattr(row, 'top_trader_position_short_pct', None)

    if top_trader_long_pct is not None and not pd.isna(top_trader_long_pct):
        if top_trader_long_pct > 0.60:  # Strong threshold
            score += 1.5
        elif top_trader_long_pct > 0.55:  # Moderate threshold
            score += 1.0

    if top_trader_short_pct is not None and not pd.isna(top_trader_short_pct):
        if top_trader_short_pct > 0.60:
            score -= 1.5
        elif top_trader_short_pct > 0.55:
            score -= 1.0

    # Top Account (reduced weight)
    top_account_long_pct = getattr(row, 'top_trader_account_long_pct', None)
    top_account_short_pct = getattr(row, 'top_trader_account_short_pct', None)

    if top_account_long_pct is not None and not pd.isna(top_account_long_pct):
        if top_account_long_pct > 0.55:
            score += 0.25
    if top_account_short_pct is not None and not pd.isna(top_account_short_pct):
        if top_account_short_pct > 0.55:
            score -= 0.25

    # Global L/S Ratio (contrarian)
    global_ls = getattr(row, 'global_ls_ratio', None)
    if global_ls is not None and not pd.isna(global_ls):
        if global_ls < 0.7:
            score += 0.5
        elif global_ls > 1.5:
            score -= 0.5

    # Funding Rate (reversal signal)
    funding_rate = getattr(row, 'funding_rate', None)
    if funding_rate is not None and not pd.isna(funding_rate):
        if funding_rate > 0.0005:
            score -= 0.5
        elif funding_rate < -0.0005:
            score += 0.5

    # OI confirmation
    oi_vs_ma = getattr(row, 'oi_vs_ma24', None)
    price_change = getattr(row, 'price_change_4h', None)
    if oi_vs_ma is not None and not pd.isna(oi_vs_ma) and oi_vs_ma > 1.05:
        if price_change is not None and not pd.isna(price_change):
            if price_change > 0.01:
                score += 0.25
            elif price_change < -0.01:
                score -= 0.25

    return score


def run_optimized_backtest(df: pd.DataFrame, atr_med: float) -> List[TradeResult]:
    """
    Run backtest with optimized strategy parameters:
    - TopTraderFocused positioning score
    - Entry=1.5, Strong=2.0 thresholds
    - StrictEntry_3loss mitigation
    - SkipNeutral (0.25) mitigation
    """
    df_bt = df.copy()
    config = OPTIMIZED_CONFIG

    risk_amount = INITIAL_EQ * RISK_PCT_INIT
    equity = INITIAL_EQ
    open_trade = None
    trades: List[TradeResult] = []
    consecutive_losses = 0
    last_exit_reason = None
    last_side = None

    for r in df_bt.itertuples():
        hr = r.Index.hour

        if pd.isna(r.sma20) or pd.isna(r.atr20):
            continue

        # Calculate positioning score using TopTraderFocused method
        positioning_score = calculate_positioning_score_top_trader_focused(r)

        volume_surge = getattr(r, 'volume_surge', False)
        oi_surge = getattr(r, 'oi_surge', False)

        # ENTRY LOGIC
        if open_trade is None and hr in ASIA_HRS and r.atr20 > atr_med:
            # Skip neutral positioning (mitigation)
            if abs(positioning_score) < config['skip_neutral_threshold']:
                continue

            # StrictEntry after consecutive losses (mitigation)
            if config['strict_entry_after_losses']:
                if consecutive_losses >= config['strict_entry_trigger']:
                    if abs(positioning_score) < config['strict_entry_min_score']:
                        continue

            # Base signals
            long_mr = (r.close < r.lower_band) and (r.rsi14 < 30)
            short_mr = (r.close > r.upper_band) and (r.rsi14 > 70)
            long_bo = (r.close > r.high_3h) and (r.rsi14 > 60)
            short_bo = (r.close < r.low_3h) and (r.rsi14 < 40)

            # Enhanced signals with positioning filters
            long_mr_ok = long_mr and positioning_score > -1.5
            short_mr_ok = short_mr and positioning_score < 1.5
            long_bo_ok = long_bo and positioning_score > -1.0
            short_bo_ok = short_bo and positioning_score < 1.0

            # Strong positioning signals
            strong_long = (
                positioning_score >= config['strong_threshold'] and
                r.rsi14 < 45 and r.close < r.sma20 and
                (volume_surge or oi_surge)
            )
            strong_short = (
                positioning_score <= -config['strong_threshold'] and
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

                # Position sizing adjustments
                if abs(positioning_score) >= 1.5:
                    size *= 1.3
                elif abs(positioning_score) >= 1.0:
                    size *= 1.2
                elif abs(positioning_score) >= 0.5:
                    size *= 1.1

                # Determine entry type
                if any_long:
                    if long_mr_ok:
                        entry_type = "mr_long"
                    elif long_bo_ok:
                        entry_type = "bo_long"
                    else:
                        entry_type = "strong_long"
                else:
                    if short_mr_ok:
                        entry_type = "mr_short"
                    elif short_bo_ok:
                        entry_type = "bo_short"
                    else:
                        entry_type = "strong_short"

                open_trade = {
                    "side": side,
                    "entry_time": r.Index,
                    "entry_price": entry_price,
                    "stop": stop_price,
                    "target": target_price,
                    "size": size,
                    "entry_type": entry_type,
                    "positioning_score": positioning_score,
                }

        # EXIT LOGIC
        elif open_trade:
            exit_price = None
            exit_reason = None

            if hr in US_HRS and hr not in ASIA_HRS:
                exit_price = r.close
                exit_reason = "us_session"
            else:
                if open_trade["side"] == "long":
                    if r.low <= open_trade["stop"]:
                        exit_price = open_trade["stop"]
                        exit_reason = "stop_loss"
                    elif r.high >= open_trade["target"]:
                        exit_price = open_trade["target"]
                        exit_reason = "take_profit"
                else:
                    if r.high >= open_trade["stop"]:
                        exit_price = open_trade["stop"]
                        exit_reason = "stop_loss"
                    elif r.low <= open_trade["target"]:
                        exit_price = open_trade["target"]
                        exit_reason = "take_profit"

            if exit_price is not None:
                pnl = ((exit_price - open_trade["entry_price"]) if open_trade["side"] == "long"
                       else (open_trade["entry_price"] - exit_price)) * open_trade["size"]

                trades.append(TradeResult(
                    pnl=pnl,
                    entry_time=str(open_trade["entry_time"]),
                    exit_time=str(r.Index),
                    side=open_trade["side"],
                    entry_type=open_trade["entry_type"],
                    exit_reason=exit_reason,
                    positioning_score=open_trade["positioning_score"]
                ))

                # Update consecutive losses
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                last_exit_reason = exit_reason
                last_side = open_trade["side"]
                open_trade = None

    return trades


def calculate_metrics_from_pnl(pnl_sequence: np.ndarray) -> Dict:
    """Calculate key metrics from a PnL sequence"""
    if len(pnl_sequence) == 0:
        return {
            'total_return': 0,
            'max_drawdown': 0,
            'max_consecutive_losses': 0,
            'sharpe_ratio': 0,
            'win_rate': 0
        }

    # Total return
    total_return = (INITIAL_EQ + pnl_sequence.sum() - INITIAL_EQ) / INITIAL_EQ * 100

    # Equity curve and drawdown
    equity_curve = INITIAL_EQ + np.cumsum(pnl_sequence)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max * 100
    max_drawdown = drawdowns.max()

    # Consecutive losses
    losses = pnl_sequence < 0
    consec_losses = [sum(1 for _ in grp) for k, grp in itertools.groupby(losses) if k]
    max_consecutive_losses = max(consec_losses) if consec_losses else 0

    # Win rate
    win_rate = (pnl_sequence > 0).sum() / len(pnl_sequence) * 100

    # Sharpe ratio (daily returns approximation)
    if len(pnl_sequence) > 1:
        returns = pnl_sequence / INITIAL_EQ
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
    else:
        sharpe = 0

    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'max_consecutive_losses': max_consecutive_losses,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate
    }


def run_monte_carlo_simulation(
    pnl_sequence: np.ndarray,
    n_simulations: int = NUM_SIMULATIONS,
    antithetic: bool = False,
) -> Dict:
    """
    Run Monte Carlo simulation by shuffling PnL sequence.

    When antithetic=True, uses antithetic variates for variance reduction:
    for each shuffled sequence Z, also evaluates the reversed sequence Z[::-1]
    and averages the pair's metrics. This halves estimator variance because
    reversed orderings produce negatively correlated metrics.

    Returns distribution statistics for each metric.
    """
    actual_metrics = calculate_metrics_from_pnl(pnl_sequence)

    # Storage for simulation results
    sim_returns = []
    sim_max_dd = []
    sim_max_consec_losses = []
    sim_sharpe = []

    if antithetic:
        n_pairs = n_simulations // 2
        log(f"Running {n_pairs} antithetic pairs ({n_simulations} effective simulations)...")

        for i in range(n_pairs):
            # Generate shuffle Z
            shuffled_pnl = np.random.permutation(pnl_sequence)
            metrics_z = calculate_metrics_from_pnl(shuffled_pnl)

            # Antithetic: reverse Z
            antithetic_pnl = shuffled_pnl[::-1]
            metrics_anti = calculate_metrics_from_pnl(antithetic_pnl)

            # Average the pair
            sim_returns.append((metrics_z['total_return'] + metrics_anti['total_return']) / 2)
            sim_max_dd.append((metrics_z['max_drawdown'] + metrics_anti['max_drawdown']) / 2)
            sim_max_consec_losses.append((metrics_z['max_consecutive_losses'] + metrics_anti['max_consecutive_losses']) / 2)
            sim_sharpe.append((metrics_z['sharpe_ratio'] + metrics_anti['sharpe_ratio']) / 2)

            if (i + 1) % 100 == 0:
                log(f"  Completed {i + 1}/{n_pairs} antithetic pairs...")
    else:
        log(f"Running {n_simulations} Monte Carlo simulations...")

        for i in range(n_simulations):
            # Shuffle the PnL sequence
            shuffled_pnl = np.random.permutation(pnl_sequence)

            # Calculate metrics for shuffled sequence
            metrics = calculate_metrics_from_pnl(shuffled_pnl)

            sim_returns.append(metrics['total_return'])
            sim_max_dd.append(metrics['max_drawdown'])
            sim_max_consec_losses.append(metrics['max_consecutive_losses'])
            sim_sharpe.append(metrics['sharpe_ratio'])

            if (i + 1) % 200 == 0:
                log(f"  Completed {i + 1}/{n_simulations} simulations...")

    # Convert to arrays
    sim_returns = np.array(sim_returns)
    sim_max_dd = np.array(sim_max_dd)
    sim_max_consec_losses = np.array(sim_max_consec_losses)
    sim_sharpe = np.array(sim_sharpe)

    # Calculate p-values (one-tailed)
    # For returns/sharpe: what % of simulations achieved >= actual
    p_value_return = (sim_returns >= actual_metrics['total_return']).mean()
    p_value_sharpe = (sim_sharpe >= actual_metrics['sharpe_ratio']).mean()

    # For max_dd/consec_losses: what % of simulations had <= actual (better)
    p_value_max_dd = (sim_max_dd <= actual_metrics['max_drawdown']).mean()
    p_value_consec_losses = (sim_max_consec_losses <= actual_metrics['max_consecutive_losses']).mean()

    # Calculate confidence intervals
    ci_95_return = (np.percentile(sim_returns, 2.5), np.percentile(sim_returns, 97.5))
    ci_95_max_dd = (np.percentile(sim_max_dd, 2.5), np.percentile(sim_max_dd, 97.5))
    ci_95_consec_losses = (np.percentile(sim_max_consec_losses, 2.5), np.percentile(sim_max_consec_losses, 97.5))

    # Compute variance reduction estimate when antithetic is used
    variance_reduction_pct = None
    if antithetic:
        # Compare paired estimator variance vs independent
        # For a fair comparison, also run a small batch without antithetic
        independent_returns = []
        n_check = min(100, n_simulations // 2)
        for _ in range(n_check):
            shuffled = np.random.permutation(pnl_sequence)
            independent_returns.append(calculate_metrics_from_pnl(shuffled)['total_return'])
        independent_std = np.std(independent_returns)
        paired_std = np.std(sim_returns)
        if independent_std > 0:
            variance_reduction_pct = float((1 - (paired_std / independent_std) ** 2) * 100)
        else:
            variance_reduction_pct = 0.0

    return {
        'actual_metrics': actual_metrics,
        'antithetic': antithetic,
        'variance_reduction_pct': variance_reduction_pct,
        'simulation_stats': {
            'total_return': {
                'mean': float(sim_returns.mean()),
                'std': float(sim_returns.std()),
                'min': float(sim_returns.min()),
                'max': float(sim_returns.max()),
                'p5': float(np.percentile(sim_returns, 5)),
                'p25': float(np.percentile(sim_returns, 25)),
                'p50': float(np.percentile(sim_returns, 50)),
                'p75': float(np.percentile(sim_returns, 75)),
                'p95': float(np.percentile(sim_returns, 95)),
                'ci_95': ci_95_return,
                'p_value': float(p_value_return),
            },
            'max_drawdown': {
                'mean': float(sim_max_dd.mean()),
                'std': float(sim_max_dd.std()),
                'min': float(sim_max_dd.min()),
                'max': float(sim_max_dd.max()),
                'p5': float(np.percentile(sim_max_dd, 5)),
                'p50': float(np.percentile(sim_max_dd, 50)),
                'p95': float(np.percentile(sim_max_dd, 95)),
                'ci_95': ci_95_max_dd,
                'p_value': float(p_value_max_dd),
            },
            'max_consecutive_losses': {
                'mean': float(sim_max_consec_losses.mean()),
                'std': float(sim_max_consec_losses.std()),
                'min': float(sim_max_consec_losses.min()),
                'max': float(sim_max_consec_losses.max()),
                'p5': float(np.percentile(sim_max_consec_losses, 5)),
                'p50': float(np.percentile(sim_max_consec_losses, 50)),
                'p95': float(np.percentile(sim_max_consec_losses, 95)),
                'ci_95': ci_95_consec_losses,
                'p_value': float(p_value_consec_losses),
            },
            'sharpe_ratio': {
                'mean': float(sim_sharpe.mean()),
                'std': float(sim_sharpe.std()),
                'p_value': float(p_value_sharpe),
            }
        },
        'n_simulations': n_simulations,
        'interpretation': {
            'return_significance': 'SIGNIFICANT' if p_value_return < 0.05 else 'NOT SIGNIFICANT',
            'drawdown_significance': 'FAVORABLE' if p_value_max_dd < 0.05 else 'TYPICAL',
            'sequence_dependency': 'HIGH' if abs(actual_metrics['total_return'] - sim_returns.mean()) > sim_returns.std() * 2 else 'LOW',
        }
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Monte Carlo Shuffle Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--simulations", "-n", type=int, default=NUM_SIMULATIONS,
        help=f"Number of simulations (default: {NUM_SIMULATIONS})",
    )
    parser.add_argument(
        "--antithetic", "-a", action="store_true",
        help="Use antithetic variates for variance reduction",
    )
    args = parser.parse_args()

    log("=" * 70)
    log("MONTE CARLO SHUFFLE VALIDATION")
    log("Strategy: Optimized TopTraderFocused + Mitigations")
    if args.antithetic:
        log("Mode: ANTITHETIC VARIATES (variance reduction)")
    log("=" * 70)

    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load and prepare data
    log("Loading data...")
    df_raw = load_ohlc_data()
    df = calculate_indicators(df_raw)
    atr_med = df['atr20'].median()

    binance_df = load_binance_data()
    if binance_df is not None:
        df = merge_binance_data(df, binance_df)

    df = calculate_enhanced_signals(df)
    log(f"Loaded data: {len(df)} rows")

    # Run optimized backtest
    log("")
    log("Running optimized backtest...")
    trades = run_optimized_backtest(df, atr_med)
    pnl_sequence = np.array([t.pnl for t in trades])

    log(f"  Total trades: {len(trades)}")
    log(f"  Total PnL: ${pnl_sequence.sum():,.0f}")
    log(f"  Win rate: {(pnl_sequence > 0).sum() / len(pnl_sequence) * 100:.1f}%")

    # Run Monte Carlo simulation
    log("")
    log("=" * 70)
    log("MONTE CARLO SIMULATION")
    log("=" * 70)

    mc_results = run_monte_carlo_simulation(pnl_sequence, args.simulations, antithetic=args.antithetic)

    # Print results
    log("")
    log("=" * 70)
    log("RESULTS")
    log("=" * 70)

    actual = mc_results['actual_metrics']
    sim_stats = mc_results['simulation_stats']

    log("")
    log("ACTUAL PERFORMANCE:")
    log(f"  Total Return: {actual['total_return']:.1f}%")
    log(f"  Max Drawdown: {actual['max_drawdown']:.1f}%")
    log(f"  Max Consecutive Losses: {actual['max_consecutive_losses']}")
    log(f"  Win Rate: {actual['win_rate']:.1f}%")

    log("")
    log("MONTE CARLO DISTRIBUTION (from shuffled sequences):")
    log("")
    log("Total Return:")
    log(f"  Actual: {actual['total_return']:.1f}%")
    log(f"  Simulated Mean: {sim_stats['total_return']['mean']:.1f}% (±{sim_stats['total_return']['std']:.1f}%)")
    log(f"  Simulated Range: [{sim_stats['total_return']['min']:.1f}%, {sim_stats['total_return']['max']:.1f}%]")
    log(f"  95% CI: [{sim_stats['total_return']['ci_95'][0]:.1f}%, {sim_stats['total_return']['ci_95'][1]:.1f}%]")
    log(f"  P-Value: {sim_stats['total_return']['p_value']:.4f} ({mc_results['interpretation']['return_significance']})")

    log("")
    log("Max Drawdown:")
    log(f"  Actual: {actual['max_drawdown']:.1f}%")
    log(f"  Simulated Mean: {sim_stats['max_drawdown']['mean']:.1f}%")
    log(f"  Simulated Range: [{sim_stats['max_drawdown']['min']:.1f}%, {sim_stats['max_drawdown']['max']:.1f}%]")
    log(f"  95% CI: [{sim_stats['max_drawdown']['ci_95'][0]:.1f}%, {sim_stats['max_drawdown']['ci_95'][1]:.1f}%]")
    log(f"  P-Value: {sim_stats['max_drawdown']['p_value']:.4f} ({mc_results['interpretation']['drawdown_significance']})")

    log("")
    log("Max Consecutive Losses:")
    log(f"  Actual: {actual['max_consecutive_losses']}")
    log(f"  Simulated Mean: {sim_stats['max_consecutive_losses']['mean']:.1f}")
    log(f"  Simulated Range: [{sim_stats['max_consecutive_losses']['min']}, {sim_stats['max_consecutive_losses']['max']}]")
    log(f"  95% CI: [{sim_stats['max_consecutive_losses']['ci_95'][0]:.0f}, {sim_stats['max_consecutive_losses']['ci_95'][1]:.0f}]")

    log("")
    log("INTERPRETATION:")
    log(f"  Sequence Dependency: {mc_results['interpretation']['sequence_dependency']}")
    if mc_results['interpretation']['sequence_dependency'] == 'HIGH':
        log("  -> The actual trade sequence significantly impacts results")
        log("  -> Strategy may be more/less robust than shuffled performance suggests")
    else:
        log("  -> Results are relatively independent of trade sequence")
        log("  -> Strategy performance is robust to timing variations")

    if mc_results.get('antithetic'):
        log("")
        log("ANTITHETIC VARIATES:")
        vr = mc_results.get('variance_reduction_pct', 0)
        log(f"  Variance Reduction: {vr:.1f}%")
        log(f"  Method: Paired shuffle Z + reverse(Z), averaged metrics")

    # Save results
    results_file = os.path.join(results_dir, 'monte_carlo_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'config': OPTIMIZED_CONFIG,
            'n_trades': len(trades),
            'n_simulations': args.simulations,
            'antithetic': args.antithetic,
            'variance_reduction_pct': mc_results.get('variance_reduction_pct'),
            'actual_metrics': actual,
            'simulation_stats': sim_stats,
            'interpretation': mc_results['interpretation'],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }, f, indent=2)

    log("")
    log(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
