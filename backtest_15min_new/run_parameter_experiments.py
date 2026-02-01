#!/usr/bin/env python3
"""
Parameter Experiment Runner
===========================
Tests different parameter combinations for:
1. Asian Hours version (0-11 UTC)
2. All Hours version (0-23 UTC)

Parameters to experiment:
- ATR period: [14, 28, 42, 56]
- RSI period: [14, 28, 42, 56]
- SMA period: [50, 100, 150, 200]
- Entry filters: various combinations
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import json
import itertools

# Add parent directory to path for data loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backtest_15min'))
from load_15min_data import merge_all_data_15min

# Configuration
INITIAL_CAPITAL = 100000
BASE_RISK = 6000

# Positioning thresholds (original v21c)
TOP_TRADER_STRONG = 0.60
TOP_TRADER_MODERATE = 0.55

# Parameter ranges to test
ATR_PERIODS = [14, 28, 42, 56]
RSI_PERIODS = [14, 28, 42, 56]
SMA_PERIODS = [50, 100, 150, 200]

# Entry filter variations
ENTRY_FILTER_CONFIGS = {
    'baseline': {
        'min_pos_long': 0.4,
        'rsi_long_range': [20, 45],
        'pullback_range': [0.5, 3.0],
        'min_pos_score': 0.15,
        'consec_loss_threshold': 3,
        'consec_loss_min_pos': 0.75,
    },
    'tight_rsi': {
        'min_pos_long': 0.4,
        'rsi_long_range': [25, 40],
        'pullback_range': [0.5, 3.0],
        'min_pos_score': 0.15,
        'consec_loss_threshold': 3,
        'consec_loss_min_pos': 0.75,
    },
    'wide_rsi': {
        'min_pos_long': 0.4,
        'rsi_long_range': [15, 50],
        'pullback_range': [0.5, 3.0],
        'min_pos_score': 0.15,
        'consec_loss_threshold': 3,
        'consec_loss_min_pos': 0.75,
    },
    'tight_pullback': {
        'min_pos_long': 0.4,
        'rsi_long_range': [20, 45],
        'pullback_range': [1.0, 2.5],
        'min_pos_score': 0.15,
        'consec_loss_threshold': 3,
        'consec_loss_min_pos': 0.75,
    },
    'wide_pullback': {
        'min_pos_long': 0.4,
        'rsi_long_range': [20, 45],
        'pullback_range': [0.3, 4.0],
        'min_pos_score': 0.15,
        'consec_loss_threshold': 3,
        'consec_loss_min_pos': 0.75,
    },
    'strict_positioning': {
        'min_pos_long': 0.6,
        'rsi_long_range': [20, 45],
        'pullback_range': [0.5, 3.0],
        'min_pos_score': 0.3,
        'consec_loss_threshold': 2,
        'consec_loss_min_pos': 1.0,
    },
    'relaxed_positioning': {
        'min_pos_long': 0.3,
        'rsi_long_range': [20, 45],
        'pullback_range': [0.5, 3.0],
        'min_pos_score': 0.1,
        'consec_loss_threshold': 4,
        'consec_loss_min_pos': 0.5,
    },
}

# Strategy exit configurations (from original v21c)
STRATEGY_CONFIGS = {
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


def compute_indicators(df, atr_period=56, rsi_period=56, sma_period=200):
    """Compute technical indicators with configurable periods"""
    df = df.copy()

    # BB period (fixed at 80)
    bb_period = 80
    volume_ma_period = 96

    # Price indicators
    df['sma20'] = df['close'].rolling(bb_period).mean()
    df['sma50'] = df['close'].rolling(sma_period).mean()
    df['std20'] = df['close'].rolling(bb_period).std()
    df['upper_band'] = df['sma20'] + 2 * df['std20']
    df['lower_band'] = df['sma20'] - 2 * df['std20']

    # ATR with configurable period
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(atr_period).mean()

    # RSI with configurable period
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    df['rsi'] = 100 - 100 / (1 + gain / loss)

    # Trend and pullback
    df['uptrend'] = df['close'] > df['sma50']
    df['pullback_pct'] = (df['high'].rolling(16).max() - df['close']) / df['close'] * 100

    # Volume indicators
    df['vol_ma'] = df['volume'].rolling(volume_ma_period).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    df['vol_ma_4h'] = df['volume'].rolling(16).mean()
    df['vol_trend'] = df['vol_ma_4h'] - df['vol_ma_4h'].shift(4)
    df['vol_increasing'] = df['vol_trend'] > 0

    # Volume-price relationship
    df['price_change'] = df['close'].pct_change()
    df['bullish_volume'] = (df['price_change'] > 0) & (df['vol_ratio'] > 1.0)

    return df


def calculate_positioning_score(row) -> float:
    """Calculate positioning score (original v21c logic)"""
    score = 0.0

    top_long = row.get('top_trader_position_long_pct', None)
    top_short = row.get('top_trader_position_short_pct', None)

    if top_long is not None and not pd.isna(top_long):
        if top_long > TOP_TRADER_STRONG:
            score += 1.5
        elif top_long > TOP_TRADER_MODERATE:
            score += 1.0

    if top_short is not None and not pd.isna(top_short):
        if top_short > TOP_TRADER_STRONG:
            score -= 1.5
        elif top_short > TOP_TRADER_MODERATE:
            score -= 1.0

    acct_long = row.get('top_trader_account_long_pct', None)
    acct_short = row.get('top_trader_account_short_pct', None)

    if acct_long is not None and not pd.isna(acct_long):
        if acct_long > TOP_TRADER_MODERATE:
            score += 0.25
    if acct_short is not None and not pd.isna(acct_short):
        if acct_short > TOP_TRADER_MODERATE:
            score -= 0.25

    global_ls = row.get('global_ls_ratio', None)
    if global_ls is not None and not pd.isna(global_ls):
        if global_ls < 0.7:
            score += 0.5
        elif global_ls > 1.5:
            score -= 0.5

    funding = row.get('funding_rate', None)
    if funding is not None and not pd.isna(funding):
        if funding > 0.0005:
            score -= 0.5
        elif funding < -0.0005:
            score += 0.5

    return score


def calculate_volume_score(row) -> float:
    """Calculate volume quality score (0-2 scale)"""
    score = 0.0

    vol_ratio = row.get('vol_ratio', 1.0)
    if not pd.isna(vol_ratio):
        if vol_ratio > 1.5:
            score += 1.0
        elif vol_ratio > 1.2:
            score += 0.7
        elif vol_ratio > 1.0:
            score += 0.5
        elif vol_ratio > 0.8:
            score += 0.3

    vol_increasing = row.get('vol_increasing', False)
    if vol_increasing:
        score += 0.5

    bullish_vol = row.get('bullish_volume', False)
    if bullish_vol:
        score += 0.5

    return min(score, 2.0)


def run_backtest(df, strategy_config, entry_filter, entry_hours=None, initial_capital=INITIAL_CAPITAL):
    """
    Run backtest with specified configuration.

    Args:
        df: DataFrame with OHLC and indicators
        strategy_config: Exit strategy configuration
        entry_filter: Entry filter configuration
        entry_hours: Set of hours to allow entries (None = all hours)
        initial_capital: Starting capital
    """
    params = {**strategy_config}
    trades = []
    equity_curve = []

    # Entry filter params
    min_pos_long = entry_filter.get('min_pos_long', 0.4)
    rsi_long_range = entry_filter.get('rsi_long_range', [20, 45])
    pullback_range = entry_filter.get('pullback_range', [0.5, 3.0])
    min_pos_score = entry_filter.get('min_pos_score', 0.15)
    consec_loss_threshold = entry_filter.get('consec_loss_threshold', 3)
    consec_loss_min_pos = entry_filter.get('consec_loss_min_pos', 0.75)
    vol_min_for_entry = params.get('vol_min_for_entry', 0.0)

    # Exit settings
    pos_exit_enabled = params.get('pos_exit_enabled', True)
    pos_exit_threshold = params.get('pos_exit_threshold', 0.0)
    pos_decline_be_enabled = params.get('pos_decline_be_enabled', False)
    pos_decline_be_threshold = params.get('pos_decline_be_threshold', 0.5)
    vol_collapse_enabled = params.get('vol_collapse_enabled', False)
    vol_collapse_threshold = params.get('vol_collapse_threshold', 0.5)
    vol_collapse_sl_mult = params.get('vol_collapse_sl_mult', 1.0)

    # Multi-TP settings
    multi_tp_enabled = params.get('multi_tp_enabled', False)
    tp1_atr_mult = params.get('tp1_atr_mult', 2.5)
    tp1_exit_pct = params.get('tp1_exit_pct', 0.5)

    # Volume-based sizing
    vol_size_adjust = params.get('vol_size_adjust', False)
    vol_high_size_mult = params.get('vol_high_size_mult', 1.25)
    vol_low_size_mult = params.get('vol_low_size_mult', 0.8)

    # Fixed params
    stop_atr_mult = 1.8
    tp_atr_mult_base = 4.5

    # State
    equity = initial_capital
    open_trades = []
    pnl_history = []
    consecutive_losses = 0

    for i, row in df.iterrows():
        hr = i.hour

        # Skip if missing critical data
        if pd.isna(row.get('sma20')) or pd.isna(row.get('atr')) or pd.isna(row.get('sma50')) or pd.isna(row.get('vol_ma')):
            continue

        positioning_score = calculate_positioning_score(row)
        volume_score = calculate_volume_score(row)
        vol_ratio = row.get('vol_ratio', 1.0)
        quality_score = positioning_score + volume_score

        # Record equity
        equity_curve.append({
            'timestamp': i,
            'equity': equity,
            'capital': equity,
            'in_position': len(open_trades) > 0,
            'btc_price': row['close'],
        })

        # Update open trades and check exits
        trades_to_close = []
        for idx, trade in enumerate(open_trades):
            trade['bars_held'] += 1

            if positioning_score < trade['min_pos_during']:
                trade['min_pos_during'] = positioning_score
            if positioning_score > trade['max_pos_during']:
                trade['max_pos_during'] = positioning_score

            if vol_ratio < trade.get('min_vol_during', 999):
                trade['min_vol_during'] = vol_ratio

            if trade['side'] == 'long':
                if row['high'] > trade['max_price']:
                    trade['max_price'] = row['high']

            pos_change = positioning_score - trade['entry_pos']

            exit_price = None
            exit_reason = None
            partial_exit = False
            partial_pct = 0

            # Positioning exit
            if pos_exit_enabled and positioning_score < pos_exit_threshold:
                exit_price = row['close']
                exit_reason = "pos_exit"

            # Positioning decline - move to breakeven
            if pos_decline_be_enabled and exit_price is None:
                if pos_change <= -pos_decline_be_threshold and not trade.get('moved_to_be', False):
                    new_stop = trade['entry_price'] + (row['atr'] * 0.1)
                    if trade['side'] == 'long' and new_stop > trade['stop']:
                        trade['stop'] = new_stop
                        trade['moved_to_be'] = True

            # Volume collapse - tighten SL
            if vol_collapse_enabled and exit_price is None:
                if vol_ratio < vol_collapse_threshold and trade['bars_held'] > 4:
                    if not trade.get('vol_tightened', False):
                        new_stop = row['close'] - (row['atr'] * vol_collapse_sl_mult)
                        if trade['side'] == 'long' and new_stop > trade['stop']:
                            trade['stop'] = new_stop
                            trade['vol_tightened'] = True

            # Multi-level TP
            if multi_tp_enabled and exit_price is None and not trade.get('tp1_hit', False):
                tp1_price = trade['entry_price'] + (row['atr'] * tp1_atr_mult)
                if row['high'] >= tp1_price:
                    partial_exit = True
                    partial_pct = tp1_exit_pct
                    exit_price = tp1_price
                    exit_reason = "tp1_partial"
                    trade['tp1_hit'] = True
                    trade['size'] *= (1 - tp1_exit_pct)

            # Regular stop/TP check
            if exit_price is None and trade["side"] == "long":
                if row['low'] <= trade["stop"]:
                    exit_price = trade["stop"]
                    exit_reason = "stop_loss"
                elif row['high'] >= trade["target"]:
                    exit_price = trade["target"]
                    exit_reason = "take_profit"

            if exit_price is not None:
                if partial_exit:
                    partial_size = trade['size'] / (1 - partial_pct) * partial_pct
                    pnl = (exit_price - trade["entry_price"]) * partial_size
                else:
                    pnl = (exit_price - trade["entry_price"]) * trade["size"]

                trades.append({
                    "trade_id": len([t for t in trades if not t.get('is_partial', False)]),
                    "entry_time": str(trade["entry_time"]),
                    "entry_price": trade["entry_price"],
                    "exit_time": str(i),
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "size": partial_size if partial_exit else trade["size"],
                    "pnl": pnl,
                    "pnl_pct": (exit_price - trade["entry_price"]) / trade["entry_price"] * 100,
                    "is_partial": partial_exit,
                    "entry_pos": trade["entry_pos"],
                    "exit_pos": positioning_score,
                    "bars_held": trade["bars_held"],
                    "capital_before": equity,
                    "capital_after": equity + pnl,
                    "stop_price": trade["stop"],
                    "tp_price": trade["target"],
                })

                pnl_history.append(pnl)
                equity += pnl

                if not partial_exit:
                    if pnl < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    trades_to_close.append(idx)

        for idx in sorted(trades_to_close, reverse=True):
            open_trades.pop(idx)

        # Entry logic
        hour_ok = entry_hours is None or hr in entry_hours
        if hour_ok and len(open_trades) == 0:
            # Skip if positioning too weak
            if abs(positioning_score) < min_pos_score:
                continue

            # Require stronger positioning after consecutive losses
            if consecutive_losses >= consec_loss_threshold:
                if abs(positioning_score) < consec_loss_min_pos:
                    continue

            # Volume filter
            if vol_ratio < vol_min_for_entry:
                continue

            entry_signal = False

            pullback = row.get('pullback_pct', 0)
            uptrend = row.get('uptrend', False)
            rsi = row.get('rsi', 50)

            if pd.isna(rsi) or pd.isna(pullback):
                continue

            if positioning_score >= min_pos_long:
                if uptrend and pullback_range[0] < pullback < pullback_range[1]:
                    if rsi_long_range[0] < rsi < rsi_long_range[1]:
                        entry_signal = True

            if entry_signal:
                entry_price = row['close']
                stop_distance = row['atr'] * stop_atr_mult

                tp_mult = tp_atr_mult_base
                if abs(positioning_score) >= 1.5:
                    tp_mult *= 1.3
                elif abs(positioning_score) >= 1.0:
                    tp_mult *= 1.15

                stop_price = entry_price - stop_distance
                target_price = entry_price + row['atr'] * tp_mult

                unit_risk = abs(entry_price - stop_price)
                size = BASE_RISK / unit_risk if unit_risk > 0 else 0

                if abs(positioning_score) >= 1.5:
                    size *= 1.2
                elif abs(positioning_score) >= 1.0:
                    size *= 1.1

                if vol_size_adjust:
                    if vol_ratio > 1.2:
                        size *= vol_high_size_mult
                    elif vol_ratio < 0.8:
                        size *= vol_low_size_mult

                if size > 0:
                    open_trades.append({
                        "side": "long",
                        "entry_time": i,
                        "entry_price": entry_price,
                        "stop": stop_price,
                        "original_stop": stop_price,
                        "target": target_price,
                        "size": size,
                        "entry_pos": positioning_score,
                        "min_pos_during": positioning_score,
                        "max_pos_during": positioning_score,
                        "entry_vol_ratio": vol_ratio,
                        "min_vol_during": vol_ratio,
                        "entry_quality": quality_score,
                        "entry_atr": row['atr'],
                        "bars_held": 0,
                        "max_price": entry_price,
                        "moved_to_be": False,
                        "vol_tightened": False,
                        "tp1_hit": False,
                    })

    # Close remaining trades
    if open_trades:
        last_row = df.iloc[-1]
        for trade in open_trades:
            pnl = (last_row['close'] - trade["entry_price"]) * trade["size"]
            pnl_history.append(pnl)
            equity += pnl

    return trades, equity_curve


def calculate_statistics(trades, equity_curve, initial_capital=INITIAL_CAPITAL):
    """Calculate comprehensive performance statistics"""
    if not trades:
        return None

    # Filter to complete trades only
    complete_trades = [t for t in trades if not t.get('is_partial', False)]

    if not complete_trades:
        return None

    total_trades = len(complete_trades)
    winning_trades = [t for t in complete_trades if t['pnl'] > 0]
    losing_trades = [t for t in complete_trades if t['pnl'] < 0]

    win_count = len(winning_trades)
    loss_count = len(losing_trades)

    total_pnl = sum(t['pnl'] for t in complete_trades)
    gross_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
    gross_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0

    avg_win = gross_profit / win_count if win_count > 0 else 0
    avg_loss = gross_loss / loss_count if loss_count > 0 else 0

    # Equity curve stats
    equity_df = pd.DataFrame(equity_curve)
    if len(equity_df) > 0:
        final_equity = equity_df['equity'].iloc[-1]
        peak_equity = equity_df['equity'].cummax()
        drawdown = (peak_equity - equity_df['equity']) / peak_equity * 100
        max_drawdown = drawdown.max()

        equity_df['returns'] = equity_df['equity'].pct_change()
        avg_return = equity_df['returns'].mean()
        std_return = equity_df['returns'].std()
        sharpe = (avg_return / std_return) * np.sqrt(35040) if std_return > 0 else 0
    else:
        final_equity = initial_capital
        max_drawdown = 0
        sharpe = 0

    # Streak analysis
    streaks = []
    current_streak = 0
    for t in complete_trades:
        if t['pnl'] > 0:
            if current_streak < 0:
                streaks.append(current_streak)
                current_streak = 1
            else:
                current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
                current_streak = -1
            else:
                current_streak -= 1
    if current_streak != 0:
        streaks.append(current_streak)

    max_winning_streak = max([s for s in streaks if s > 0], default=0)
    max_losing_streak = abs(min([s for s in streaks if s < 0], default=0))

    return {
        'initial_capital': initial_capital,
        'final_equity': final_equity,
        'total_return_pct': (final_equity - initial_capital) / initial_capital * 100,
        'total_pnl': total_pnl,
        'max_drawdown_pct': max_drawdown,
        'sharpe_ratio': sharpe,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        'total_trades': total_trades,
        'winning_trades': win_count,
        'losing_trades': loss_count,
        'win_rate_pct': win_count / total_trades * 100 if total_trades > 0 else 0,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': avg_win / avg_loss if avg_loss > 0 else float('inf'),
        'expectancy': (avg_win * win_count - avg_loss * loss_count) / total_trades if total_trades > 0 else 0,
        'max_winning_streak': max_winning_streak,
        'max_losing_streak': max_losing_streak,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
    }


def run_parameter_grid_search(df_base, entry_hours, version_name, output_dir):
    """
    Run grid search over indicator periods to find optimal parameters.
    """
    print(f"\n{'='*70}")
    print(f"PARAMETER GRID SEARCH: {version_name}")
    print(f"{'='*70}")

    results = []

    # Test key combinations (not full grid to save time)
    param_combos = [
        # Original v21c
        {'atr': 56, 'rsi': 56, 'sma': 200},
        # Shorter periods
        {'atr': 14, 'rsi': 14, 'sma': 50},
        {'atr': 14, 'rsi': 14, 'sma': 100},
        {'atr': 28, 'rsi': 28, 'sma': 100},
        {'atr': 28, 'rsi': 28, 'sma': 150},
        # Medium periods
        {'atr': 42, 'rsi': 42, 'sma': 150},
        {'atr': 42, 'rsi': 42, 'sma': 200},
        # Mixed
        {'atr': 14, 'rsi': 56, 'sma': 200},
        {'atr': 56, 'rsi': 14, 'sma': 100},
        {'atr': 28, 'rsi': 56, 'sma': 150},
    ]

    for params in param_combos:
        print(f"\nTesting ATR={params['atr']}, RSI={params['rsi']}, SMA={params['sma']}...")

        # Compute indicators with these parameters
        df = compute_indicators(df_base.copy(),
                               atr_period=params['atr'],
                               rsi_period=params['rsi'],
                               sma_period=params['sma'])

        # Test with baseline entry filter and baseline strategy
        trades, equity_curve = run_backtest(
            df,
            STRATEGY_CONFIGS['Baseline'],
            ENTRY_FILTER_CONFIGS['baseline'],
            entry_hours=entry_hours
        )

        stats = calculate_statistics(trades, equity_curve)

        if stats and stats['total_trades'] >= 10:
            results.append({
                'atr_period': params['atr'],
                'rsi_period': params['rsi'],
                'sma_period': params['sma'],
                **stats
            })
            print(f"  Trades: {stats['total_trades']}, Return: {stats['total_return_pct']:.1f}%, "
                  f"WR: {stats['win_rate_pct']:.1f}%, MaxDD: {stats['max_drawdown_pct']:.1f}%, "
                  f"MaxCL: {stats['max_losing_streak']}")
        else:
            print(f"  Insufficient trades")

    # Sort by return
    results_sorted = sorted(results, key=lambda x: x['total_return_pct'], reverse=True)

    # Save results
    results_file = os.path.join(output_dir, 'parameter_grid_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_sorted, f, indent=2, default=str)

    print(f"\nSaved parameter grid results to: {results_file}")

    if results_sorted:
        best = results_sorted[0]
        print(f"\nBest parameters: ATR={best['atr_period']}, RSI={best['rsi_period']}, SMA={best['sma_period']}")
        print(f"  Return: {best['total_return_pct']:.1f}%, MaxDD: {best['max_drawdown_pct']:.1f}%, "
              f"Trades: {best['total_trades']}, WR: {best['win_rate_pct']:.1f}%")
        return best

    return None


def run_entry_filter_experiments(df, entry_hours, version_name, output_dir):
    """
    Test different entry filter configurations.
    """
    print(f"\n{'='*70}")
    print(f"ENTRY FILTER EXPERIMENTS: {version_name}")
    print(f"{'='*70}")

    results = []

    for filter_name, filter_config in ENTRY_FILTER_CONFIGS.items():
        print(f"\nTesting filter: {filter_name}...")

        trades, equity_curve = run_backtest(
            df,
            STRATEGY_CONFIGS['Baseline'],
            filter_config,
            entry_hours=entry_hours
        )

        stats = calculate_statistics(trades, equity_curve)

        if stats and stats['total_trades'] >= 10:
            results.append({
                'filter_name': filter_name,
                'filter_config': filter_config,
                **stats
            })
            print(f"  Trades: {stats['total_trades']}, Return: {stats['total_return_pct']:.1f}%, "
                  f"WR: {stats['win_rate_pct']:.1f}%, MaxDD: {stats['max_drawdown_pct']:.1f}%, "
                  f"MaxCL: {stats['max_losing_streak']}")
        else:
            print(f"  Insufficient trades")

    # Sort by return
    results_sorted = sorted(results, key=lambda x: x['total_return_pct'], reverse=True)

    # Save results
    results_file = os.path.join(output_dir, 'entry_filter_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_sorted, f, indent=2, default=str)

    print(f"\nSaved entry filter results to: {results_file}")

    if results_sorted:
        best = results_sorted[0]
        print(f"\nBest entry filter: {best['filter_name']}")
        print(f"  Return: {best['total_return_pct']:.1f}%, MaxDD: {best['max_drawdown_pct']:.1f}%, "
              f"Trades: {best['total_trades']}, WR: {best['win_rate_pct']:.1f}%")
        return best

    return None


def run_full_backtest_suite(df, entry_hours, version_suffix, output_dir, best_params=None, best_filter=None):
    """
    Run full backtest suite with all strategy configurations.
    """
    print(f"\n{'='*70}")
    print(f"FULL BACKTEST SUITE: {version_suffix}")
    print(f"{'='*70}")

    # Use best parameters if provided, else defaults
    if best_params:
        atr_period = best_params.get('atr_period', 56)
        rsi_period = best_params.get('rsi_period', 56)
        sma_period = best_params.get('sma_period', 200)
    else:
        atr_period, rsi_period, sma_period = 56, 56, 200

    # Use best entry filter if provided
    entry_filter = ENTRY_FILTER_CONFIGS.get(
        best_filter.get('filter_name', 'baseline') if best_filter else 'baseline',
        ENTRY_FILTER_CONFIGS['baseline']
    )

    print(f"\nUsing parameters: ATR={atr_period}, RSI={rsi_period}, SMA={sma_period}")
    print(f"Using entry filter: {best_filter.get('filter_name', 'baseline') if best_filter else 'baseline'}")

    # Compute indicators
    df_indicators = compute_indicators(df.copy(), atr_period, rsi_period, sma_period)

    all_stats = {}
    all_equity = {}

    for strategy_name, strategy_config in STRATEGY_CONFIGS.items():
        full_name = f"{strategy_name}_{version_suffix}"
        print(f"\nRunning: {full_name}")

        trades, equity_curve = run_backtest(
            df_indicators,
            strategy_config,
            entry_filter,
            entry_hours=entry_hours
        )

        stats = calculate_statistics(trades, equity_curve)

        if stats:
            all_stats[full_name] = {
                'strategy': strategy_name,
                'version': version_suffix,
                'atr_period': atr_period,
                'rsi_period': rsi_period,
                'sma_period': sma_period,
                **stats
            }

            print(f"  Trades: {stats['total_trades']}, Return: {stats['total_return_pct']:.1f}%, "
                  f"WR: {stats['win_rate_pct']:.1f}%, MaxDD: {stats['max_drawdown_pct']:.1f}%, "
                  f"MaxCL: {stats['max_losing_streak']}")

            # Export trade log
            trades_df = pd.DataFrame(trades)
            trades_file = os.path.join(output_dir, f'trades_{full_name}.csv')
            trades_df.to_csv(trades_file, index=False)
            print(f"  Exported: trades_{full_name}.csv")

            # Export equity curve
            equity_df = pd.DataFrame(equity_curve)
            equity_file = os.path.join(output_dir, f'equity_{full_name}.csv')
            equity_df.to_csv(equity_file, index=False)
            print(f"  Exported: equity_{full_name}.csv")

            all_equity[full_name] = equity_df
        else:
            print(f"  No valid trades")

    # Save statistics
    stats_file = os.path.join(output_dir, 'strategy_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\nSaved statistics to: {stats_file}")

    return all_stats, all_equity


def main():
    print("=" * 70)
    print("BTC STRATEGY PARAMETER EXPERIMENTS")
    print("=" * 70)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"Base Risk: ${BASE_RISK:,}")
    print("=" * 70)

    # Load data
    print("\nLoading 15-minute data...")
    df_base = merge_all_data_15min()
    print(f"Loaded {len(df_base):,} candles from {df_base.index.min()} to {df_base.index.max()}")

    # Define entry hours
    asian_hours = set(range(0, 12))
    all_hours = None  # None means all hours

    # Output directories
    base_dir = os.path.dirname(__file__)
    asian_dir = os.path.join(base_dir, 'results_asian_hours')
    all_dir = os.path.join(base_dir, 'results_all_hours')

    os.makedirs(asian_dir, exist_ok=True)
    os.makedirs(all_dir, exist_ok=True)

    # Export price data
    price_df = df_base[['open', 'high', 'low', 'close', 'volume']].copy()
    price_df.to_csv(os.path.join(asian_dir, 'btc_price_15min.csv'))
    price_df.to_csv(os.path.join(all_dir, 'btc_price_15min.csv'))

    # =========================================
    # ASIAN HOURS VERSION
    # =========================================
    print("\n" + "=" * 70)
    print("ASIAN HOURS VERSION (0-11 UTC)")
    print("=" * 70)

    # Compute indicators with original v21c parameters for grid search
    df_asian = compute_indicators(df_base.copy(), atr_period=56, rsi_period=56, sma_period=200)

    # 1. Parameter grid search
    best_params_asian = run_parameter_grid_search(df_base, asian_hours, "Asian Hours", asian_dir)

    # 2. Entry filter experiments (using best params)
    if best_params_asian:
        df_asian = compute_indicators(df_base.copy(),
                                      atr_period=best_params_asian['atr_period'],
                                      rsi_period=best_params_asian['rsi_period'],
                                      sma_period=best_params_asian['sma_period'])

    best_filter_asian = run_entry_filter_experiments(df_asian, asian_hours, "Asian Hours", asian_dir)

    # 3. Full backtest suite
    asian_stats, asian_equity = run_full_backtest_suite(
        df_base, asian_hours, "AHours", asian_dir,
        best_params_asian, best_filter_asian
    )

    # =========================================
    # ALL HOURS VERSION
    # =========================================
    print("\n" + "=" * 70)
    print("ALL HOURS VERSION (0-23 UTC)")
    print("=" * 70)

    # 1. Parameter grid search
    best_params_all = run_parameter_grid_search(df_base, all_hours, "All Hours", all_dir)

    # 2. Entry filter experiments
    if best_params_all:
        df_all = compute_indicators(df_base.copy(),
                                   atr_period=best_params_all['atr_period'],
                                   rsi_period=best_params_all['rsi_period'],
                                   sma_period=best_params_all['sma_period'])
    else:
        df_all = compute_indicators(df_base.copy(), atr_period=56, rsi_period=56, sma_period=200)

    best_filter_all = run_entry_filter_experiments(df_all, all_hours, "All Hours", all_dir)

    # 3. Full backtest suite
    all_stats, all_equity = run_full_backtest_suite(
        df_base, all_hours, "AllHours", all_dir,
        best_params_all, best_filter_all
    )

    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    print("\n--- ASIAN HOURS RESULTS ---")
    print(f"{'Strategy':<35} {'Return%':>10} {'MaxDD%':>8} {'WR%':>6} {'Trades':>7} {'MaxCL':>6}")
    print("-" * 75)
    for name, stats in sorted(asian_stats.items(), key=lambda x: x[1]['total_return_pct'], reverse=True):
        print(f"{name:<35} {stats['total_return_pct']:>9.1f}% {stats['max_drawdown_pct']:>7.1f}% "
              f"{stats['win_rate_pct']:>5.1f}% {stats['total_trades']:>7} {stats['max_losing_streak']:>6}")

    print("\n--- ALL HOURS RESULTS ---")
    print(f"{'Strategy':<35} {'Return%':>10} {'MaxDD%':>8} {'WR%':>6} {'Trades':>7} {'MaxCL':>6}")
    print("-" * 75)
    for name, stats in sorted(all_stats.items(), key=lambda x: x[1]['total_return_pct'], reverse=True):
        print(f"{name:<35} {stats['total_return_pct']:>9.1f}% {stats['max_drawdown_pct']:>7.1f}% "
              f"{stats['win_rate_pct']:>5.1f}% {stats['total_trades']:>7} {stats['max_losing_streak']:>6}")

    print("\n" + "=" * 70)
    print(f"Asian Hours results saved to: {asian_dir}")
    print(f"All Hours results saved to: {all_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
