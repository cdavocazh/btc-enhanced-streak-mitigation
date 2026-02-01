#!/usr/bin/env python3
"""
Tiered Capital Backtest with Streak Mitigation
===============================================
Runs backtest with tiered position sizing AND losing streak risk reduction.

Tier Structure (capital risked per trade):
- Initial ($100k starting): 5% of $100k = $5k
- $150k - $225k: 7% of $150k = $10.5k
- $225k - $337.5k: 6.5% of $225k = $14.625k
- $337.5k - $507k: 6% of $337.5k = $20.25k
- $507k - $760k: 5.5% of $507k = $28k
- $760k - $1.2m: $38k
- $1.2m onwards: $54k

Streak Mitigation Rules:
- Losing streak = 3: Reduce next trade risk by 40%. After first win, go back to initial risk.
- Losing streak = 6: Reduce next trade risk by another 30% (total 58% reduction). After first win, go back to initial risk.
- Losing streak = 9: Reduce next trade risk by another 30% (total 70.6% reduction). After first win, go back to 5% capital risk.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import json

# Add parent directory to path for data loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backtest_15min'))
from load_15min_data import merge_all_data_15min

# Configuration
INITIAL_CAPITAL = 100000

# Tiered risk configuration (equity threshold -> risk amount)
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
    3: {'reduction': 0.40, 'recovery': 'initial'},    # 40% reduction, recover to initial
    6: {'reduction': 0.30, 'recovery': 'initial'},    # Additional 30% reduction, recover to initial
    9: {'reduction': 0.30, 'recovery': '5pct'},       # Additional 30% reduction, recover to 5% capital
}

# Positioning thresholds (original v21c)
TOP_TRADER_STRONG = 0.60
TOP_TRADER_MODERATE = 0.55

# Asian hours filter
ASIAN_HOURS = set(range(0, 12))

# Optimal parameters (from parameter experiments)
ATR_PERIOD = 14
RSI_PERIOD = 56
SMA_PERIOD = 200

# Entry filter (baseline)
ENTRY_FILTER = {
    'min_pos_long': 0.4,
    'rsi_long_range': [20, 45],
    'pullback_range': [0.5, 3.0],
    'min_pos_score': 0.15,
    'consec_loss_threshold': 3,
    'consec_loss_min_pos': 0.75,
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


def get_risk_for_equity(equity: float) -> float:
    """Get the risk amount for current equity level based on tier structure."""
    for min_eq, max_eq, risk in RISK_TIERS:
        if min_eq <= equity < max_eq:
            return risk
    # Fallback to highest tier
    return RISK_TIERS[-1][2]


def get_tier_name(equity: float) -> str:
    """Get human-readable tier name for equity level."""
    for i, (min_eq, max_eq, risk) in enumerate(RISK_TIERS):
        if min_eq <= equity < max_eq:
            if max_eq == float('inf'):
                return f"Tier {i+1}: ${min_eq/1000:.0f}k+ (${risk/1000:.1f}k risk)"
            else:
                return f"Tier {i+1}: ${min_eq/1000:.0f}k-${max_eq/1000:.0f}k (${risk/1000:.1f}k risk)"
    return "Unknown Tier"


def get_streak_adjusted_risk(base_risk: float, consecutive_losses: int, equity: float) -> tuple:
    """
    Apply streak mitigation rules to base risk.

    Returns:
        (adjusted_risk, reduction_applied, streak_level)
    """
    if consecutive_losses < 3:
        return base_risk, 0.0, 0

    # Calculate cumulative reduction
    reduction = 1.0
    streak_level = 0

    if consecutive_losses >= 3:
        reduction *= (1 - STREAK_RULES[3]['reduction'])  # 60% of original
        streak_level = 3

    if consecutive_losses >= 6:
        reduction *= (1 - STREAK_RULES[6]['reduction'])  # 42% of original
        streak_level = 6

    if consecutive_losses >= 9:
        reduction *= (1 - STREAK_RULES[9]['reduction'])  # 29.4% of original
        streak_level = 9

    adjusted_risk = base_risk * reduction
    reduction_pct = 1 - reduction

    return adjusted_risk, reduction_pct, streak_level


def get_recovery_risk(streak_level: int, equity: float) -> float:
    """
    Get the risk to recover to after a win following a losing streak.

    Args:
        streak_level: The streak level that was active (3, 6, or 9)
        equity: Current equity

    Returns:
        Risk amount to use for next trade
    """
    if streak_level >= 9:
        # After streak of 9+, recover to 5% of capital
        return equity * 0.05
    else:
        # After streak of 3-8, recover to initial tiered risk
        return get_risk_for_equity(equity)


def compute_indicators(df, atr_period=ATR_PERIOD, rsi_period=RSI_PERIOD, sma_period=SMA_PERIOD):
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


def run_tiered_streak_backtest(df, strategy_config, entry_filter, entry_hours=None, initial_capital=INITIAL_CAPITAL):
    """
    Run backtest with tiered capital AND streak mitigation.

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
    losing_streaks = []  # Track all losing streaks >= 3

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
    current_tier = get_tier_name(equity)

    # Streak tracking
    current_streak_trades = []  # Trades in current losing streak
    streak_start_price = None
    streak_high = None
    streak_low = None
    max_streak_level = 0  # Track highest streak level reached in current streak
    recovery_mode = False  # True after a win following streak >= 3
    recovery_streak_level = 0  # The streak level we're recovering from

    for i, row in df.iterrows():
        hr = i.hour

        # Skip if missing critical data
        if pd.isna(row.get('sma20')) or pd.isna(row.get('atr')) or pd.isna(row.get('sma50')) or pd.isna(row.get('vol_ma')):
            continue

        positioning_score = calculate_positioning_score(row)
        volume_score = calculate_volume_score(row)
        vol_ratio = row.get('vol_ratio', 1.0)
        quality_score = positioning_score + volume_score

        # Get base tiered risk
        base_risk = get_risk_for_equity(equity)

        # Apply streak mitigation
        adjusted_risk, reduction_pct, streak_level = get_streak_adjusted_risk(base_risk, consecutive_losses, equity)

        if streak_level > max_streak_level:
            max_streak_level = streak_level

        # Record equity
        equity_curve.append({
            'timestamp': i,
            'equity': equity,
            'capital': equity,
            'in_position': len(open_trades) > 0,
            'btc_price': row['close'],
            'base_risk': base_risk,
            'adjusted_risk': adjusted_risk,
            'consecutive_losses': consecutive_losses,
            'streak_level': streak_level,
            'reduction_pct': reduction_pct,
            'tier': current_tier,
        })

        # Track price during losing streak
        if consecutive_losses >= 3:
            if streak_high is None:
                streak_high = row['high']
                streak_low = row['low']
            else:
                streak_high = max(streak_high, row['high'])
                streak_low = min(streak_low, row['low'])

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

                trade_record = {
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
                    "base_risk": trade["base_risk"],
                    "adjusted_risk": trade["adjusted_risk"],
                    "streak_level_at_entry": trade["streak_level_at_entry"],
                    "consecutive_losses_at_entry": trade["consecutive_losses_at_entry"],
                    "reduction_pct_at_entry": trade["reduction_pct_at_entry"],
                    "tier_at_entry": trade["tier_at_entry"],
                }

                trades.append(trade_record)
                pnl_history.append(pnl)
                equity += pnl

                if not partial_exit:
                    if pnl < 0:
                        consecutive_losses += 1

                        # Track streak
                        if consecutive_losses == 3:
                            streak_start_price = current_streak_trades[0]['entry_price'] if current_streak_trades else trade['entry_price']
                            streak_high = row['high']
                            streak_low = row['low']

                        current_streak_trades.append(trade_record)

                    else:
                        # Win - check if we need to record a completed losing streak
                        if consecutive_losses >= 3:
                            # Record the losing streak
                            losing_streaks.append({
                                'streak_length': consecutive_losses,
                                'max_streak_level': max_streak_level,
                                'start_time': current_streak_trades[0]['entry_time'] if current_streak_trades else None,
                                'end_time': current_streak_trades[-1]['exit_time'] if current_streak_trades else None,
                                'first_entry_price': current_streak_trades[0]['entry_price'] if current_streak_trades else None,
                                'last_exit_price': current_streak_trades[-1]['exit_price'] if current_streak_trades else None,
                                'streak_high': streak_high,
                                'streak_low': streak_low,
                                'total_loss': sum(t['pnl'] for t in current_streak_trades),
                                'trades': [{'entry_time': t['entry_time'], 'exit_time': t['exit_time'],
                                           'entry_price': t['entry_price'], 'exit_price': t['exit_price'],
                                           'pnl': t['pnl']} for t in current_streak_trades],
                            })

                            # Set recovery mode
                            recovery_mode = True
                            recovery_streak_level = max_streak_level

                        # Reset streak tracking
                        consecutive_losses = 0
                        current_streak_trades = []
                        streak_start_price = None
                        streak_high = None
                        streak_low = None
                        max_streak_level = 0
                        recovery_mode = False

                    trades_to_close.append(idx)

        for idx in sorted(trades_to_close, reverse=True):
            open_trades.pop(idx)

        # Entry logic
        hour_ok = entry_hours is None or hr in entry_hours
        if hour_ok and len(open_trades) == 0:
            # Skip if positioning too weak
            if abs(positioning_score) < min_pos_score:
                continue

            # Require stronger positioning after consecutive losses (original logic)
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

                # Use streak-adjusted tiered risk
                size = adjusted_risk / unit_risk if unit_risk > 0 else 0

                # Position score adjustments
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
                        "base_risk": base_risk,
                        "adjusted_risk": adjusted_risk,
                        "streak_level_at_entry": streak_level,
                        "consecutive_losses_at_entry": consecutive_losses,
                        "reduction_pct_at_entry": reduction_pct,
                        "tier_at_entry": current_tier,
                    })

    # Close remaining trades
    if open_trades:
        last_row = df.iloc[-1]
        for trade in open_trades:
            pnl = (last_row['close'] - trade["entry_price"]) * trade["size"]
            pnl_history.append(pnl)
            equity += pnl

    # Record any remaining losing streak
    if consecutive_losses >= 3 and current_streak_trades:
        losing_streaks.append({
            'streak_length': consecutive_losses,
            'max_streak_level': max_streak_level,
            'start_time': current_streak_trades[0]['entry_time'],
            'end_time': current_streak_trades[-1]['exit_time'],
            'first_entry_price': current_streak_trades[0]['entry_price'],
            'last_exit_price': current_streak_trades[-1]['exit_price'],
            'streak_high': streak_high,
            'streak_low': streak_low,
            'total_loss': sum(t['pnl'] for t in current_streak_trades),
            'trades': [{'entry_time': t['entry_time'], 'exit_time': t['exit_time'],
                       'entry_price': t['entry_price'], 'exit_price': t['exit_price'],
                       'pnl': t['pnl']} for t in current_streak_trades],
        })

    return trades, equity_curve, losing_streaks


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
        max_drawdown_abs = (peak_equity - equity_df['equity']).max()

        equity_df['returns'] = equity_df['equity'].pct_change()
        avg_return = equity_df['returns'].mean()
        std_return = equity_df['returns'].std()
        sharpe = (avg_return / std_return) * np.sqrt(35040) if std_return > 0 else 0
    else:
        final_equity = initial_capital
        max_drawdown = 0
        max_drawdown_abs = 0
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

    # Count trades with reduced risk
    trades_with_reduction = [t for t in complete_trades if t.get('reduction_pct_at_entry', 0) > 0]

    # Average risk metrics
    avg_base_risk = np.mean([t.get('base_risk', 5000) for t in complete_trades])
    avg_adjusted_risk = np.mean([t.get('adjusted_risk', 5000) for t in complete_trades])

    return {
        'initial_capital': initial_capital,
        'final_equity': final_equity,
        'total_return_pct': (final_equity - initial_capital) / initial_capital * 100,
        'total_pnl': total_pnl,
        'max_drawdown_pct': max_drawdown,
        'max_drawdown_abs': max_drawdown_abs,
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
        'avg_base_risk': avg_base_risk,
        'avg_adjusted_risk': avg_adjusted_risk,
        'trades_with_reduction': len(trades_with_reduction),
        'reduction_trade_pct': len(trades_with_reduction) / total_trades * 100 if total_trades > 0 else 0,
    }


def calculate_streak_statistics(losing_streaks):
    """Calculate statistics about losing streaks"""
    if not losing_streaks:
        return {
            'total_streaks_3plus': 0,
            'total_streaks_6plus': 0,
            'total_streaks_9plus': 0,
            'avg_streak_length': 0,
            'max_streak_length': 0,
            'avg_streak_loss': 0,
            'total_streak_loss': 0,
        }

    streaks_3plus = [s for s in losing_streaks if s['streak_length'] >= 3]
    streaks_6plus = [s for s in losing_streaks if s['streak_length'] >= 6]
    streaks_9plus = [s for s in losing_streaks if s['streak_length'] >= 9]

    return {
        'total_streaks_3plus': len(streaks_3plus),
        'total_streaks_6plus': len(streaks_6plus),
        'total_streaks_9plus': len(streaks_9plus),
        'avg_streak_length': np.mean([s['streak_length'] for s in losing_streaks]) if losing_streaks else 0,
        'max_streak_length': max([s['streak_length'] for s in losing_streaks]) if losing_streaks else 0,
        'avg_streak_loss': np.mean([s['total_loss'] for s in losing_streaks]) if losing_streaks else 0,
        'total_streak_loss': sum([s['total_loss'] for s in losing_streaks]) if losing_streaks else 0,
    }


def generate_html_report(all_stats, all_streak_stats, output_dir):
    """Generate HTML report with results and equity curves"""

    html = """<!DOCTYPE html>
<html>
<head>
    <title>Tiered Capital with Streak Mitigation - Backtest Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        h3 { color: #666; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: right; }
        th { background: #4CAF50; color: white; }
        tr:nth-child(even) { background: #f9f9f9; }
        tr:hover { background: #f1f1f1; }
        td:first-child { text-align: left; font-weight: bold; }
        .positive { color: #4CAF50; }
        .negative { color: #f44336; }
        .chart-container { background: white; padding: 20px; margin: 20px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .rules-box { background: #e8f5e9; padding: 20px; border-left: 4px solid #4CAF50; margin: 20px 0; }
        .tier-box { background: #e3f2fd; padding: 20px; border-left: 4px solid #2196F3; margin: 20px 0; }
        .streak-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
        .streak-card { background: white; padding: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }
        .streak-card h4 { margin: 0 0 10px 0; color: #666; }
        .streak-card .number { font-size: 2em; font-weight: bold; color: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Tiered Capital with Streak Mitigation - Backtest Report</h1>
        <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>

        <div class="tier-box">
            <h3>Tiered Risk Structure</h3>
            <table>
                <tr><th>Equity Range</th><th>Risk Per Trade</th></tr>
                <tr><td>$0 - $150k</td><td>$5,000 (5% of $100k)</td></tr>
                <tr><td>$150k - $225k</td><td>$10,500 (7% of $150k)</td></tr>
                <tr><td>$225k - $337.5k</td><td>$14,625 (6.5% of $225k)</td></tr>
                <tr><td>$337.5k - $507k</td><td>$20,250 (6% of $337.5k)</td></tr>
                <tr><td>$507k - $760k</td><td>$28,000 (5.5% of $507k)</td></tr>
                <tr><td>$760k - $1.2m</td><td>$38,000 (fixed)</td></tr>
                <tr><td>$1.2m+</td><td>$54,000 (fixed)</td></tr>
            </table>
        </div>

        <div class="rules-box">
            <h3>Streak Mitigation Rules</h3>
            <ul>
                <li><strong>3 consecutive losses:</strong> Reduce next trade risk by 40%. After first win, return to initial tiered risk.</li>
                <li><strong>6 consecutive losses:</strong> Reduce by additional 30% (total ~58% reduction). After first win, return to initial tiered risk.</li>
                <li><strong>9 consecutive losses:</strong> Reduce by additional 30% (total ~70.6% reduction). After first win, return to 5% of capital.</li>
            </ul>
        </div>

        <h2>Performance Summary</h2>
        <table>
            <tr>
                <th>Strategy</th>
                <th>Return %</th>
                <th>Final Equity</th>
                <th>Max DD %</th>
                <th>Max DD $</th>
                <th>Trades</th>
                <th>Win Rate</th>
                <th>Sharpe</th>
                <th>Max CL</th>
                <th>Trades w/ Reduction</th>
            </tr>
"""

    for name, stats in sorted(all_stats.items(), key=lambda x: x[1]['final_equity'], reverse=True):
        return_class = 'positive' if stats['total_return_pct'] > 0 else 'negative'
        html += f"""
            <tr>
                <td>{name}</td>
                <td class="{return_class}">{stats['total_return_pct']:.1f}%</td>
                <td>${stats['final_equity']:,.0f}</td>
                <td>{stats['max_drawdown_pct']:.1f}%</td>
                <td>${stats['max_drawdown_abs']:,.0f}</td>
                <td>{stats['total_trades']}</td>
                <td>{stats['win_rate_pct']:.1f}%</td>
                <td>{stats['sharpe_ratio']:.2f}</td>
                <td>{stats['max_losing_streak']}</td>
                <td>{stats['trades_with_reduction']} ({stats['reduction_trade_pct']:.1f}%)</td>
            </tr>
"""

    html += """
        </table>

        <h2>Losing Streak Statistics</h2>
        <div class="streak-stats">
"""

    for name, streak_stats in all_streak_stats.items():
        html += f"""
            <div class="streak-card">
                <h4>{name}</h4>
                <div class="number">{streak_stats['total_streaks_3plus']}</div>
                <p>Streaks >= 3</p>
                <p>6+ Streaks: {streak_stats['total_streaks_6plus']} | 9+ Streaks: {streak_stats['total_streaks_9plus']}</p>
                <p>Max Length: {streak_stats['max_streak_length']} | Avg: {streak_stats['avg_streak_length']:.1f}</p>
                <p>Total Streak Loss: ${streak_stats['total_streak_loss']:,.0f}</p>
            </div>
"""

    html += """
        </div>

        <h2>Equity Curves</h2>
"""

    # Add placeholder for equity chart
    html += """
        <div class="chart-container" id="equity-chart" style="height: 500px;"></div>

        <script>
"""

    # Load equity curves from CSVs and create chart
    chart_data = []
    for name in all_stats.keys():
        equity_file = os.path.join(output_dir, f'equity_{name}.csv')
        if os.path.exists(equity_file):
            eq_df = pd.read_csv(equity_file)
            # Sample every 100th point for performance
            eq_df = eq_df.iloc[::100]
            chart_data.append({
                'name': name,
                'x': eq_df['timestamp'].tolist(),
                'y': eq_df['equity'].tolist(),
            })

    html += f"""
            var data = {json.dumps(chart_data)};
            var traces = data.map(function(d) {{
                return {{
                    x: d.x,
                    y: d.y,
                    name: d.name,
                    type: 'scatter',
                    mode: 'lines'
                }};
            }});

            var layout = {{
                title: 'Equity Curves - Tiered Capital with Streak Mitigation',
                xaxis: {{ title: 'Date' }},
                yaxis: {{ title: 'Equity ($)', tickformat: '$,.0f' }},
                legend: {{ x: 0, y: 1.1, orientation: 'h' }},
                hovermode: 'x unified'
            }};

            Plotly.newPlot('equity-chart', traces, layout);
        </script>
    </div>
</body>
</html>
"""

    return html


def main():
    print("=" * 70)
    print("TIERED CAPITAL WITH STREAK MITIGATION BACKTEST")
    print("=" * 70)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print("\nRisk Tiers:")
    for min_eq, max_eq, risk in RISK_TIERS:
        if max_eq == float('inf'):
            print(f"  ${min_eq/1000:.0f}k+: ${risk/1000:.1f}k per trade")
        else:
            print(f"  ${min_eq/1000:.0f}k - ${max_eq/1000:.0f}k: ${risk/1000:.1f}k per trade")

    print("\nStreak Mitigation Rules:")
    print("  3 losses: -40% risk, recover to initial after win")
    print("  6 losses: -30% additional, recover to initial after win")
    print("  9 losses: -30% additional, recover to 5% capital after win")
    print("=" * 70)

    # Load data
    print("\nLoading 15-minute data...")
    df_base = merge_all_data_15min()
    print(f"Loaded {len(df_base):,} candles from {df_base.index.min()} to {df_base.index.max()}")

    # Compute indicators
    print(f"\nComputing indicators (ATR={ATR_PERIOD}, RSI={RSI_PERIOD}, SMA={SMA_PERIOD})...")
    df = compute_indicators(df_base)

    # Output directory
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, 'results_tiered_streak')
    os.makedirs(output_dir, exist_ok=True)

    all_stats = {}
    all_streak_stats = {}
    all_losing_streaks = {}

    print("\n" + "=" * 70)
    print("RUNNING TIERED STREAK MITIGATION BACKTESTS")
    print("=" * 70)

    for strategy_name, strategy_config in STRATEGY_CONFIGS.items():
        full_name = f"{strategy_name}_TieredStreak"
        print(f"\nRunning: {full_name}")

        trades, equity_curve, losing_streaks = run_tiered_streak_backtest(
            df,
            strategy_config,
            ENTRY_FILTER,
            entry_hours=ASIAN_HOURS
        )

        stats = calculate_statistics(trades, equity_curve)
        streak_stats = calculate_streak_statistics(losing_streaks)

        if stats:
            all_stats[full_name] = {
                'strategy': strategy_name,
                'version': 'TieredStreak_AHours',
                **stats
            }
            all_streak_stats[full_name] = streak_stats
            all_losing_streaks[full_name] = losing_streaks

            print(f"  Trades: {stats['total_trades']}, Return: {stats['total_return_pct']:.1f}%, "
                  f"Final: ${stats['final_equity']:,.0f}")
            print(f"  WR: {stats['win_rate_pct']:.1f}%, MaxDD: {stats['max_drawdown_pct']:.1f}%, "
                  f"MaxCL: {stats['max_losing_streak']}")
            print(f"  Streaks 3+: {streak_stats['total_streaks_3plus']}, "
                  f"6+: {streak_stats['total_streaks_6plus']}, "
                  f"9+: {streak_stats['total_streaks_9plus']}")
            print(f"  Trades with reduction: {stats['trades_with_reduction']} ({stats['reduction_trade_pct']:.1f}%)")

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

            # Export losing streaks with price history
            if losing_streaks:
                streaks_data = []
                for streak in losing_streaks:
                    streaks_data.append({
                        'streak_length': streak['streak_length'],
                        'max_streak_level': streak['max_streak_level'],
                        'start_time': streak['start_time'],
                        'end_time': streak['end_time'],
                        'first_entry_price': streak['first_entry_price'],
                        'last_exit_price': streak['last_exit_price'],
                        'streak_high': streak['streak_high'],
                        'streak_low': streak['streak_low'],
                        'price_range': streak['streak_high'] - streak['streak_low'] if streak['streak_high'] and streak['streak_low'] else None,
                        'total_loss': streak['total_loss'],
                        'num_trades': len(streak['trades']),
                    })

                streaks_df = pd.DataFrame(streaks_data)
                streaks_file = os.path.join(output_dir, f'losing_streaks_{full_name}.csv')
                streaks_df.to_csv(streaks_file, index=False)
                print(f"  Exported: losing_streaks_{full_name}.csv")
        else:
            print(f"  No valid trades")

    # Save statistics
    stats_file = os.path.join(output_dir, 'strategy_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\nSaved statistics to: {stats_file}")

    # Save streak statistics
    streak_stats_file = os.path.join(output_dir, 'streak_statistics.json')
    with open(streak_stats_file, 'w') as f:
        json.dump(all_streak_stats, f, indent=2, default=str)
    print(f"Saved streak statistics to: {streak_stats_file}")

    # Generate HTML report
    html_report = generate_html_report(all_stats, all_streak_stats, output_dir)
    report_file = os.path.join(output_dir, 'backtest_report.html')
    with open(report_file, 'w') as f:
        f.write(html_report)
    print(f"Generated report: {report_file}")

    # =========================================
    # COMPARISON: With vs Without Streak Mitigation
    # =========================================
    print("\n" + "=" * 70)
    print("COMPARISON: TIERED vs TIERED + STREAK MITIGATION")
    print("=" * 70)

    # Load tiered-only results for comparison
    tiered_stats_file = os.path.join(base_dir, 'results_tiered_capital', 'strategy_statistics.json')
    if os.path.exists(tiered_stats_file):
        with open(tiered_stats_file, 'r') as f:
            tiered_stats = json.load(f)

        print(f"\n{'Strategy':<25} {'Version':<15} {'Return%':>10} {'Final $':>15} {'MaxDD%':>8} {'MaxCL':>6}")
        print("-" * 85)

        for strategy_name in STRATEGY_CONFIGS.keys():
            # Tiered only results
            tiered_name = f"{strategy_name}_Tiered"
            if tiered_name in tiered_stats:
                ts = tiered_stats[tiered_name]
                print(f"{strategy_name:<25} {'Tiered':<15} {ts['total_return_pct']:>9.1f}% "
                      f"${ts['final_equity']:>13,.0f} {ts['max_drawdown_pct']:>7.1f}% {ts['max_losing_streak']:>6}")

            # Tiered + Streak results
            streak_name = f"{strategy_name}_TieredStreak"
            if streak_name in all_stats:
                ss = all_stats[streak_name]
                print(f"{'':<25} {'Tiered+Streak':<15} {ss['total_return_pct']:>9.1f}% "
                      f"${ss['final_equity']:>13,.0f} {ss['max_drawdown_pct']:>7.1f}% {ss['max_losing_streak']:>6}")

            print()
    else:
        print("\nTiered-only results not found for comparison.")

    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 70)
    print("TIERED + STREAK MITIGATION RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Strategy':<30} {'Return%':>10} {'Final $':>15} {'MaxDD%':>8} {'Streaks 3+':>10} {'Reduced Trades':>15}")
    print("-" * 95)
    for name, stats in sorted(all_stats.items(), key=lambda x: x[1]['final_equity'], reverse=True):
        streak_stats = all_streak_stats[name]
        print(f"{name:<30} {stats['total_return_pct']:>9.1f}% ${stats['final_equity']:>13,.0f} "
              f"{stats['max_drawdown_pct']:>7.1f}% {streak_stats['total_streaks_3plus']:>10} "
              f"{stats['trades_with_reduction']:>7} ({stats['reduction_trade_pct']:.1f}%)")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
