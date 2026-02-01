#!/usr/bin/env python3
"""
Adaptive Streak Reduction Backtest
===================================
Implements the action items from losing_streak_analysis.html:

1. ADX Filter (Solution 1a) - Market regime detection
   - Only enter when ADX > threshold (trending market)
   - ADX < threshold indicates range-bound conditions to avoid

2. Progressive Positioning Threshold (Solution 4b)
   - Increase min_pos_long requirement after each consecutive loss
   - 0 losses: 0.4, 2 losses: 0.6, 3 losses: 0.8, 4+ losses: 1.0
   - 5+ losses: 24-hour cooldown period

3. Sequential Strategy (Market Regime First)
   - Step 1: Check market regime (ADX)
   - Step 2: Apply regime-specific entry filters
   - Step 3: Use adaptive positioning thresholds

This approach addresses the three root causes:
- 45% of streaks: Choppy range-bound markets (ADX filter)
- 30% of streaks: Trend reversals (Higher Highs/Lower Lows check)
- 25% of streaks: Low volatility compression (Adaptive ATR)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
    (0, 150000, 5000),
    (150000, 225000, 10500),
    (225000, 337500, 14625),
    (337500, 507000, 20250),
    (507000, 760000, 28000),
    (760000, 1200000, 38000),
    (1200000, float('inf'), 54000),
]

# Streak mitigation rules (same as original)
STREAK_RULES = {
    3: {'reduction': 0.40, 'recovery': 'initial'},
    6: {'reduction': 0.30, 'recovery': 'initial'},
    9: {'reduction': 0.30, 'recovery': '5pct'},
}

# Positioning thresholds
TOP_TRADER_STRONG = 0.60
TOP_TRADER_MODERATE = 0.55

# Asian hours filter
ASIAN_HOURS = set(range(0, 12))

# Optimal parameters
ATR_PERIOD = 14
RSI_PERIOD = 56
SMA_PERIOD = 200
ADX_PERIOD = 14

# NEW: Adaptive entry filters
ADAPTIVE_ENTRY_FILTER = {
    'min_pos_long_base': 0.4,           # Base threshold
    'rsi_long_range': [20, 45],
    'pullback_range': [0.5, 3.0],
    'min_pos_score': 0.15,

    # Progressive positioning threshold (Solution 4b)
    'progressive_pos_threshold': {
        0: 0.4,    # No losses: base threshold
        1: 0.5,    # 1 loss: slightly higher
        2: 0.6,    # 2 losses: moderate
        3: 0.8,    # 3 losses: high
        4: 1.0,    # 4+ losses: very high (require strong signal)
    },
    'cooldown_after_losses': 5,         # Enter cooldown after this many losses
    'cooldown_bars': 96,                # 24 hours of 15-min bars

    # ADX filter (Solution 1a)
    'adx_filter_enabled': True,
    'adx_trending_threshold': 20,       # ADX > 20 = trending market
    'adx_strong_trend_threshold': 30,   # ADX > 30 = strong trend

    # Higher volume requirement in range-bound markets
    'vol_min_trending': 0.7,            # Volume min when ADX > 20
    'vol_min_ranging': 1.2,             # Higher volume min when ADX < 20
}

# Strategy configurations with adaptive features
ADAPTIVE_STRATEGY_CONFIGS = {
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
    "Sequential_Regime": {
        "pos_exit_enabled": True,
        "pos_exit_threshold": 0.0,
        "adx_filter": True,
        "progressive_pos": True,
        "cooldown_enabled": True,
        "hh_hl_filter": True,              # Higher highs/higher lows filter
        "adaptive_atr_sl": True,           # Use min(ATR_5, ATR_14) for SL
        "pos_momentum_filter": True,       # Check positioning rate of change
    },
}


def get_risk_for_equity(equity: float) -> float:
    """Get the risk amount for current equity level."""
    for min_eq, max_eq, risk in RISK_TIERS:
        if min_eq <= equity < max_eq:
            return risk
    return RISK_TIERS[-1][2]


def get_tier_name(equity: float) -> str:
    """Get human-readable tier name."""
    for i, (min_eq, max_eq, risk) in enumerate(RISK_TIERS):
        if min_eq <= equity < max_eq:
            if max_eq == float('inf'):
                return f"Tier {i+1}: ${min_eq/1000:.0f}k+ (${risk/1000:.1f}k risk)"
            else:
                return f"Tier {i+1}: ${min_eq/1000:.0f}k-${max_eq/1000:.0f}k (${risk/1000:.1f}k risk)"
    return "Unknown Tier"


def get_streak_adjusted_risk(base_risk: float, consecutive_losses: int, equity: float) -> tuple:
    """Apply streak mitigation rules to base risk."""
    if consecutive_losses < 3:
        return base_risk, 0.0, 0

    reduction = 1.0
    streak_level = 0

    if consecutive_losses >= 3:
        reduction *= (1 - STREAK_RULES[3]['reduction'])
        streak_level = 3
    if consecutive_losses >= 6:
        reduction *= (1 - STREAK_RULES[6]['reduction'])
        streak_level = 6
    if consecutive_losses >= 9:
        reduction *= (1 - STREAK_RULES[9]['reduction'])
        streak_level = 9

    adjusted_risk = base_risk * reduction
    reduction_pct = 1 - reduction

    return adjusted_risk, reduction_pct, streak_level


def get_progressive_pos_threshold(consecutive_losses: int, base_threshold: float) -> float:
    """Get adaptive positioning threshold based on consecutive losses."""
    thresholds = ADAPTIVE_ENTRY_FILTER['progressive_pos_threshold']

    if consecutive_losses in thresholds:
        return thresholds[consecutive_losses]
    elif consecutive_losses >= max(thresholds.keys()):
        return thresholds[max(thresholds.keys())]
    else:
        return base_threshold


def compute_indicators(df, atr_period=ATR_PERIOD, rsi_period=RSI_PERIOD,
                       sma_period=SMA_PERIOD, adx_period=ADX_PERIOD):
    """Compute technical indicators including ADX."""
    df = df.copy()

    bb_period = 80
    volume_ma_period = 96

    # Price indicators
    df['sma20'] = df['close'].rolling(bb_period).mean()
    df['sma50'] = df['close'].rolling(sma_period).mean()
    df['std20'] = df['close'].rolling(bb_period).std()
    df['upper_band'] = df['sma20'] + 2 * df['std20']
    df['lower_band'] = df['sma20'] - 2 * df['std20']

    # ATR - both standard and short-term
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(atr_period).mean()
    df['atr_short'] = tr.rolling(5).mean()  # Short-term ATR for adaptive SL

    # Adaptive ATR (min of short and long)
    df['atr_adaptive'] = df[['atr', 'atr_short']].min(axis=1)

    # RSI
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
    df['price_change'] = df['close'].pct_change()
    df['bullish_volume'] = (df['price_change'] > 0) & (df['vol_ratio'] > 1.0)

    # === ADX CALCULATION (Solution 1a) ===
    # True Range already computed above

    # Directional Movement
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()

    # Only keep positive values where the move is larger
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # Smoothed values (using Wilder's smoothing)
    atr_smooth = tr.ewm(alpha=1/adx_period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1/adx_period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1/adx_period, adjust=False).mean()

    # Directional Indicators
    df['plus_di'] = 100 * plus_dm_smooth / atr_smooth
    df['minus_di'] = 100 * minus_dm_smooth / atr_smooth

    # DX and ADX
    di_diff = (df['plus_di'] - df['minus_di']).abs()
    di_sum = df['plus_di'] + df['minus_di']
    dx = 100 * di_diff / di_sum.replace(0, np.nan)
    df['adx'] = dx.ewm(alpha=1/adx_period, adjust=False).mean()

    # Market regime based on ADX
    df['is_trending'] = df['adx'] > ADAPTIVE_ENTRY_FILTER['adx_trending_threshold']
    df['is_strong_trend'] = df['adx'] > ADAPTIVE_ENTRY_FILTER['adx_strong_trend_threshold']

    # === HIGHER HIGHS / HIGHER LOWS (Solution 3a) ===
    lookback = 20
    df['swing_high_1'] = df['high'].rolling(lookback).max()
    df['swing_high_2'] = df['high'].rolling(lookback).max().shift(lookback)
    df['swing_low_1'] = df['low'].rolling(lookback).min()
    df['swing_low_2'] = df['low'].rolling(lookback).min().shift(lookback)

    df['making_hh'] = df['swing_high_1'] > df['swing_high_2']
    df['making_hl'] = df['swing_low_1'] > df['swing_low_2']
    df['trend_confirmed'] = df['making_hh'] & df['making_hl']

    # === POSITIONING MOMENTUM (Solution 4a) ===
    # Will be calculated after positioning score is computed

    return df


def calculate_positioning_score(row) -> float:
    """Calculate positioning score."""
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
    """Calculate volume quality score (0-2 scale)."""
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


def run_adaptive_backtest(df, strategy_config, entry_filter, entry_hours=None,
                          initial_capital=INITIAL_CAPITAL, strategy_name="Adaptive"):
    """
    Run backtest with adaptive streak reduction features.
    """
    params = {**strategy_config}
    trades = []
    equity_curve = []
    losing_streaks = []
    entry_blocks = []  # Track when entries were blocked and why

    # Entry filter params
    min_pos_long_base = entry_filter.get('min_pos_long_base', 0.4)
    rsi_long_range = entry_filter.get('rsi_long_range', [20, 45])
    pullback_range = entry_filter.get('pullback_range', [0.5, 3.0])
    min_pos_score = entry_filter.get('min_pos_score', 0.15)

    # Adaptive features
    adx_filter = params.get('adx_filter', False)
    progressive_pos = params.get('progressive_pos', False)
    cooldown_enabled = params.get('cooldown_enabled', False)
    hh_hl_filter = params.get('hh_hl_filter', False)
    adaptive_atr_sl = params.get('adaptive_atr_sl', False)
    pos_momentum_filter = params.get('pos_momentum_filter', False)

    adx_threshold = entry_filter.get('adx_trending_threshold', 20)
    cooldown_bars = entry_filter.get('cooldown_bars', 96)
    cooldown_after_losses = entry_filter.get('cooldown_after_losses', 5)
    vol_min_trending = entry_filter.get('vol_min_trending', 0.7)
    vol_min_ranging = entry_filter.get('vol_min_ranging', 1.2)

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
    cooldown_until = None  # Timestamp when cooldown ends

    # Positioning momentum tracking
    pos_score_history = []

    # Streak tracking
    current_streak_trades = []
    streak_start_price = None
    streak_high = None
    streak_low = None
    max_streak_level = 0

    # Stats tracking
    blocked_by_adx = 0
    blocked_by_pos = 0
    blocked_by_cooldown = 0
    blocked_by_hh_hl = 0
    blocked_by_pos_momentum = 0

    for i, row in df.iterrows():
        hr = i.hour

        # Skip if missing critical data
        if pd.isna(row.get('sma20')) or pd.isna(row.get('atr')) or pd.isna(row.get('sma50')) or pd.isna(row.get('vol_ma')):
            continue

        positioning_score = calculate_positioning_score(row)
        volume_score = calculate_volume_score(row)
        vol_ratio = row.get('vol_ratio', 1.0)
        quality_score = positioning_score + volume_score

        # Track positioning history for momentum
        pos_score_history.append({'time': i, 'score': positioning_score})
        if len(pos_score_history) > 20:
            pos_score_history.pop(0)

        # Get base tiered risk
        base_risk = get_risk_for_equity(equity)
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
            'adx': row.get('adx', 0),
            'is_trending': row.get('is_trending', False),
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
                    "entry_adx": trade.get("entry_adx", 0),
                    "entry_regime": trade.get("entry_regime", "unknown"),
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

                        # Check if cooldown should be activated
                        if cooldown_enabled and consecutive_losses >= cooldown_after_losses:
                            cooldown_until = i + timedelta(hours=24)

                    else:
                        # Win - check if we need to record a completed losing streak
                        if consecutive_losses >= 3:
                            losing_streaks.append({
                                'streak_length': consecutive_losses,
                                'max_streak_level': max_streak_level,
                                'start_time': current_streak_trades[0]['entry_time'] if current_streak_trades else None,
                                'end_time': current_streak_trades[-1]['exit_time'] if current_streak_trades else None,
                                'first_entry_price': current_streak_trades[0]['entry_price'] if current_streak_trades else None,
                                'last_exit_price': current_streak_trades[-1]['exit_price'] if current_streak_trades else None,
                                'streak_high': streak_high,
                                'streak_low': streak_low,
                                'price_range': (streak_high - streak_low) if streak_high and streak_low else 0,
                                'total_loss': sum(t['pnl'] for t in current_streak_trades),
                                'num_trades': len(current_streak_trades),
                            })

                        # Reset streak tracking
                        consecutive_losses = 0
                        current_streak_trades = []
                        streak_start_price = None
                        streak_high = None
                        streak_low = None
                        max_streak_level = 0
                        cooldown_until = None

                    trades_to_close.append(idx)

        for idx in sorted(trades_to_close, reverse=True):
            open_trades.pop(idx)

        # ============ ENTRY LOGIC WITH ADAPTIVE FILTERS ============
        hour_ok = entry_hours is None or hr in entry_hours
        if hour_ok and len(open_trades) == 0:

            # Check cooldown
            if cooldown_enabled and cooldown_until is not None and i < cooldown_until:
                blocked_by_cooldown += 1
                continue

            # Skip if positioning too weak
            if abs(positioning_score) < min_pos_score:
                continue

            # === ADX FILTER (Solution 1a) ===
            adx_value = row.get('adx', 0)
            is_trending = row.get('is_trending', False)

            if adx_filter:
                if pd.isna(adx_value) or adx_value < adx_threshold:
                    blocked_by_adx += 1
                    entry_blocks.append({
                        'time': i,
                        'reason': 'adx_filter',
                        'adx': adx_value,
                        'pos_score': positioning_score
                    })
                    continue

            # === PROGRESSIVE POSITIONING THRESHOLD (Solution 4b) ===
            if progressive_pos:
                min_pos_long = get_progressive_pos_threshold(consecutive_losses, min_pos_long_base)
            else:
                min_pos_long = min_pos_long_base

            # === HIGHER HIGHS / HIGHER LOWS FILTER (Solution 3a) ===
            if hh_hl_filter:
                trend_confirmed = row.get('trend_confirmed', True)
                if not trend_confirmed:
                    blocked_by_hh_hl += 1
                    continue

            # === POSITIONING MOMENTUM FILTER (Solution 4a) ===
            if pos_momentum_filter and len(pos_score_history) >= 16:
                pos_4h_ago = pos_score_history[-16]['score']
                pos_momentum = positioning_score - pos_4h_ago
                if pos_momentum < -0.3:  # Positioning declining rapidly
                    blocked_by_pos_momentum += 1
                    continue

            # === VOLUME FILTER (regime-adaptive) ===
            if adx_filter:
                vol_min = vol_min_trending if is_trending else vol_min_ranging
            else:
                vol_min = 0.7

            if vol_ratio < vol_min:
                continue

            entry_signal = False

            pullback = row.get('pullback_pct', 0)
            uptrend = row.get('uptrend', False)
            rsi = row.get('rsi', 50)

            if pd.isna(rsi) or pd.isna(pullback):
                continue

            # Check if positioning meets the (potentially adaptive) threshold
            if positioning_score >= min_pos_long:
                if uptrend and pullback_range[0] < pullback < pullback_range[1]:
                    if rsi_long_range[0] < rsi < rsi_long_range[1]:
                        entry_signal = True
            else:
                blocked_by_pos += 1

            if entry_signal:
                entry_price = row['close']

                # === ADAPTIVE ATR STOP LOSS (Solution 2a) ===
                if adaptive_atr_sl:
                    atr_for_sl = row.get('atr_adaptive', row['atr'])
                else:
                    atr_for_sl = row['atr']

                stop_distance = atr_for_sl * stop_atr_mult

                tp_mult = tp_atr_mult_base
                if abs(positioning_score) >= 1.5:
                    tp_mult *= 1.3
                elif abs(positioning_score) >= 1.0:
                    tp_mult *= 1.15

                stop_price = entry_price - stop_distance
                target_price = entry_price + row['atr'] * tp_mult

                unit_risk = abs(entry_price - stop_price)
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
                        "entry_adx": adx_value,
                        "entry_regime": "trending" if is_trending else "ranging",
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
            'start_time': current_streak_trades[0]['entry_time'] if current_streak_trades else None,
            'end_time': current_streak_trades[-1]['exit_time'] if current_streak_trades else None,
            'first_entry_price': current_streak_trades[0]['entry_price'] if current_streak_trades else None,
            'last_exit_price': current_streak_trades[-1]['exit_price'] if current_streak_trades else None,
            'streak_high': streak_high,
            'streak_low': streak_low,
            'price_range': (streak_high - streak_low) if streak_high and streak_low else 0,
            'total_loss': sum(t['pnl'] for t in current_streak_trades),
            'num_trades': len(current_streak_trades),
        })

    # Calculate statistics
    trades_df = pd.DataFrame(trades)

    if len(trades_df) > 0:
        non_partial_trades = trades_df[~trades_df['is_partial']]
        winning_trades = non_partial_trades[non_partial_trades['pnl'] > 0]
        losing_trades_df = non_partial_trades[non_partial_trades['pnl'] <= 0]

        total_trades = len(non_partial_trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades_df)

        win_rate = win_count / total_trades * 100 if total_trades > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades_df['pnl'].mean()) if len(losing_trades_df) > 0 else 0

        gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades_df['pnl'].sum()) if len(losing_trades_df) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Calculate max drawdown
        equity_df = pd.DataFrame(equity_curve)
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['peak'] - equity_df['equity']) / equity_df['peak'] * 100
        max_dd = equity_df['drawdown'].max()
        max_dd_abs = (equity_df['peak'] - equity_df['equity']).max()

        # Sharpe ratio (simplified)
        if len(pnl_history) > 1:
            returns = pd.Series(pnl_history)
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Winning/losing streaks
        non_partial_trades['is_win'] = non_partial_trades['pnl'] > 0
        streaks = []
        current_streak = 0
        current_type = None
        for _, row in non_partial_trades.iterrows():
            is_win = row['is_win']
            if current_type is None:
                current_type = is_win
                current_streak = 1
            elif is_win == current_type:
                current_streak += 1
            else:
                streaks.append((current_type, current_streak))
                current_type = is_win
                current_streak = 1
        if current_streak > 0:
            streaks.append((current_type, current_streak))

        win_streaks = [s[1] for s in streaks if s[0]]
        loss_streaks = [s[1] for s in streaks if not s[0]]
        max_win_streak = max(win_streaks) if win_streaks else 0
        max_loss_streak = max(loss_streaks) if loss_streaks else 0
    else:
        total_trades = 0
        win_count = 0
        loss_count = 0
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        gross_profit = 0
        gross_loss = 0
        profit_factor = 0
        max_dd = 0
        max_dd_abs = 0
        sharpe = 0
        max_win_streak = 0
        max_loss_streak = 0

    # Streak statistics
    streaks_3plus = [s for s in losing_streaks if s['streak_length'] >= 3]
    streaks_6plus = [s for s in losing_streaks if s['streak_length'] >= 6]
    streaks_9plus = [s for s in losing_streaks if s['streak_length'] >= 9]

    statistics = {
        'strategy': strategy_name,
        'version': 'Adaptive_TieredStreak_AHours',
        'initial_capital': initial_capital,
        'final_equity': equity,
        'total_return_pct': (equity - initial_capital) / initial_capital * 100,
        'total_pnl': equity - initial_capital,
        'max_drawdown_pct': max_dd,
        'max_drawdown_abs': max_dd_abs,
        'sharpe_ratio': sharpe,
        'profit_factor': profit_factor,
        'total_trades': total_trades,
        'winning_trades': win_count,
        'losing_trades': loss_count,
        'win_rate_pct': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': avg_win / avg_loss if avg_loss > 0 else 0,
        'expectancy': (avg_win * win_rate / 100 - avg_loss * (100 - win_rate) / 100) if total_trades > 0 else 0,
        'max_winning_streak': max_win_streak,
        'max_losing_streak': max_loss_streak,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'blocked_by_adx': blocked_by_adx,
        'blocked_by_pos_threshold': blocked_by_pos,
        'blocked_by_cooldown': blocked_by_cooldown,
        'blocked_by_hh_hl': blocked_by_hh_hl,
        'blocked_by_pos_momentum': blocked_by_pos_momentum,
    }

    streak_statistics = {
        'total_streaks_3plus': len(streaks_3plus),
        'total_streaks_6plus': len(streaks_6plus),
        'total_streaks_9plus': len(streaks_9plus),
        'avg_streak_length': np.mean([s['streak_length'] for s in streaks_3plus]) if streaks_3plus else 0,
        'max_streak_length': max([s['streak_length'] for s in streaks_3plus]) if streaks_3plus else 0,
        'avg_streak_loss': np.mean([s['total_loss'] for s in streaks_3plus]) if streaks_3plus else 0,
        'total_streak_loss': sum([s['total_loss'] for s in streaks_3plus]) if streaks_3plus else 0,
    }

    return {
        'trades': trades,
        'equity_curve': equity_curve,
        'losing_streaks': losing_streaks,
        'statistics': statistics,
        'streak_statistics': streak_statistics,
        'entry_blocks': entry_blocks,
    }


def main():
    """Run all adaptive strategy backtests."""
    print("=" * 70)
    print("ADAPTIVE STREAK REDUCTION BACKTEST")
    print("=" * 70)
    print("\nLoading data...")

    # Load data
    df = merge_all_data_15min()
    print(f"Loaded {len(df)} rows of data")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Compute indicators
    print("\nComputing indicators (including ADX)...")
    df = compute_indicators(df)

    # Results directory
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}
    all_statistics = {}
    all_streak_stats = {}

    # Run each strategy
    for strategy_name, config in ADAPTIVE_STRATEGY_CONFIGS.items():
        print(f"\n{'=' * 50}")
        print(f"Running: {strategy_name}")
        print(f"{'=' * 50}")

        result = run_adaptive_backtest(
            df,
            config,
            ADAPTIVE_ENTRY_FILTER,
            entry_hours=ASIAN_HOURS,
            strategy_name=strategy_name
        )

        all_results[strategy_name] = result
        all_statistics[strategy_name] = result['statistics']
        all_streak_stats[strategy_name] = result['streak_statistics']

        stats = result['statistics']
        streak_stats = result['streak_statistics']

        print(f"\nResults for {strategy_name}:")
        print(f"  Total Return: {stats['total_return_pct']:.1f}%")
        print(f"  Final Equity: ${stats['final_equity']:,.0f}")
        print(f"  Max Drawdown: {stats['max_drawdown_pct']:.1f}%")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Win Rate: {stats['win_rate_pct']:.1f}%")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"  Max Losing Streak: {stats['max_losing_streak']}")
        print(f"\n  Streak Stats:")
        print(f"    Streaks 3+: {streak_stats['total_streaks_3plus']}")
        print(f"    Streaks 6+: {streak_stats['total_streaks_6plus']}")
        print(f"    Streaks 9+: {streak_stats['total_streaks_9plus']}")
        print(f"    Total Streak Loss: ${streak_stats['total_streak_loss']:,.0f}")
        print(f"\n  Entry Blocks:")
        print(f"    Blocked by ADX: {stats['blocked_by_adx']}")
        print(f"    Blocked by Pos Threshold: {stats['blocked_by_pos_threshold']}")
        print(f"    Blocked by Cooldown: {stats['blocked_by_cooldown']}")

        # Save trades
        trades_df = pd.DataFrame(result['trades'])
        trades_df.to_csv(os.path.join(results_dir, f'trades_{strategy_name}.csv'), index=False)

        # Save equity curve
        equity_df = pd.DataFrame(result['equity_curve'])
        equity_df.to_csv(os.path.join(results_dir, f'equity_{strategy_name}.csv'), index=False)

        # Save losing streaks
        if result['losing_streaks']:
            streaks_df = pd.DataFrame(result['losing_streaks'])
            streaks_df.to_csv(os.path.join(results_dir, f'losing_streaks_{strategy_name}.csv'), index=False)

    # Save all statistics
    with open(os.path.join(results_dir, 'strategy_statistics.json'), 'w') as f:
        json.dump(all_statistics, f, indent=2, default=str)

    with open(os.path.join(results_dir, 'streak_statistics.json'), 'w') as f:
        json.dump(all_streak_stats, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print("\n{:<25} {:>12} {:>10} {:>10} {:>8} {:>10}".format(
        "Strategy", "Return", "Max DD", "Trades", "WR%", "MaxStreak"))
    print("-" * 80)

    for name, stats in all_statistics.items():
        print("{:<25} {:>11.1f}% {:>9.1f}% {:>10} {:>7.1f}% {:>10}".format(
            name,
            stats['total_return_pct'],
            stats['max_drawdown_pct'],
            stats['total_trades'],
            stats['win_rate_pct'],
            stats['max_losing_streak']
        ))

    print("\n" + "=" * 70)
    print("STREAK COMPARISON")
    print("=" * 70)

    print("\n{:<25} {:>10} {:>10} {:>10} {:>15}".format(
        "Strategy", "Streaks3+", "Streaks6+", "MaxLen", "TotalLoss"))
    print("-" * 75)

    for name, stats in all_streak_stats.items():
        print("{:<25} {:>10} {:>10} {:>10} {:>15,.0f}".format(
            name,
            stats['total_streaks_3plus'],
            stats['total_streaks_6plus'],
            stats['max_streak_length'],
            stats['total_streak_loss']
        ))

    print(f"\nResults saved to: {results_dir}")

    return all_results


if __name__ == "__main__":
    main()
