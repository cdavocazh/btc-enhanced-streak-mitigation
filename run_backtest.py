#!/usr/bin/env python3
"""
BTC Enhanced Strategy with Streak Mitigation
=============================================

Optimized Configuration:
- Positioning: TopTraderFocused
- Entry Threshold: 1.5
- Strong Positioning Threshold: 2.0
- Mitigation: StrictEntry_3loss (require score ≥0.5 after 3 consecutive losses)
- Skip Neutral: Yes (threshold 0.25)

Expected Performance:
- Return: ~1700%+ (vs 1655% baseline)
- Max Consecutive Losses: ~20-22 (vs 25 baseline)
- Win Rate: ~28.3%

Data Sources:
- OHLC: BTC hourly candles from 2020
- Binance Positioning: From 2021-12-01 (backfilled from Binance Vision archive)
  - Top trader position L/S ratio
  - Top trader account L/S ratio
  - Global trader L/S ratio
  - Funding rate
  - Open Interest
"""

import pandas as pd
import numpy as np
import itertools
import json
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, List
import os

# Configuration
CSV_FILE = 'BTC_OHLC_1h_gmt8_updated.csv'
RESULTS_DIR = 'backtest_results'

# ============================================================================
# OPTIMIZED STRATEGY PARAMETERS
# ============================================================================

# Capital & Risk
INITIAL_EQ = 100000
RISK_PCT_INIT = 0.05
STOP_PCT = 0.005
ATR_MULT = 3.0

# Time Windows
ASIA_HRS = set(range(0, 12))
US_HRS = set(range(15, 21))

# Technical Indicator Periods
ATR_PERIOD = 14
BB_PERIOD = 20
RSI_PERIOD = 14
MA200_PERIOD = 200
VOLUME_MA_PERIOD = 24

# Volume Parameters
VOLUME_SURGE_THRESHOLD = 1.5
VOLUME_LOW_THRESHOLD = 0.5

# ============================================================================
# OPTIMIZED POSITIONING PARAMETERS (TopTraderFocused)
# ============================================================================

# TopTraderFocused Configuration
TOP_TRADER_STRONG_THRESHOLD = 0.60   # Strong signal threshold
TOP_TRADER_MODERATE_THRESHOLD = 0.55  # Moderate signal threshold
TOP_TRADER_STRONG_WEIGHT = 1.5       # Weight for strong signals
TOP_TRADER_MODERATE_WEIGHT = 1.0     # Weight for moderate signals

# Account signals (reduced weight in TopTraderFocused)
TOP_ACCOUNT_WEIGHT = 0.25

# Other positioning factors
GLOBAL_LS_RATIO_CONTRARIAN_HIGH = 1.5
GLOBAL_LS_RATIO_CONTRARIAN_LOW = 0.7
FUNDING_RATE_EXTREME_POSITIVE = 0.0005
FUNDING_RATE_EXTREME_NEGATIVE = -0.0005

# OI Parameters
OI_SURGE_THRESHOLD = 1.05
OI_DROP_THRESHOLD = 0.95

# ============================================================================
# OPTIMIZED ENTRY/EXIT THRESHOLDS
# ============================================================================

ENTRY_THRESHOLD = 1.5           # Minimum positioning score for entry
STRONG_POSITIONING_THRESHOLD = 2.0  # Strong positioning signal threshold

# ============================================================================
# STREAK MITIGATION PARAMETERS
# ============================================================================

SKIP_NEUTRAL_ENABLED = True
SKIP_NEUTRAL_THRESHOLD = 0.25   # Skip trades with |score| < 0.25

STRICT_ENTRY_AFTER_LOSSES = True
STRICT_ENTRY_TRIGGER = 3        # After 3 consecutive losses
STRICT_ENTRY_MIN_SCORE = 0.5    # Require score >= 0.5 after streak


def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"[{timestamp}] {message}")


def load_ohlc_data():
    """Load BTC OHLC data"""
    df = pd.read_csv(CSV_FILE, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df


def load_binance_data():
    """Load Binance positioning data from binance-futures-data repo"""
    try:
        from load_binance_data import load_all_binance_data
        df = load_all_binance_data()
        return df
    except ImportError as e:
        log(f"Warning: Could not import load_binance_data module: {e}")
        return None
    except Exception as e:
        log(f"Warning: Error loading Binance data: {e}")
        return None


def calculate_indicators(df):
    """Calculate all technical indicators including volume metrics"""
    df = df.copy()

    # Bollinger Bands
    df['sma20'] = df['close'].rolling(BB_PERIOD).mean()
    df['std20'] = df['close'].rolling(BB_PERIOD).std()
    df['upper_band'] = df['sma20'] + 2 * df['std20']
    df['lower_band'] = df['sma20'] - 2 * df['std20']

    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr20'] = tr.rolling(ATR_PERIOD).mean()
    df['atr20_median_all'] = df['atr20'].expanding().median()
    df['atr20_roll_med180'] = df['atr20'].rolling(window=180).median()

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    df['rsi14'] = 100 - 100 / (1 + gain / loss)

    # SMA 200
    df['sma200'] = df['close'].rolling(MA200_PERIOD).mean()

    # Breakout levels
    df['high_3h'] = df['high'].shift(1).rolling(3).max()
    df['low_3h'] = df['low'].shift(1).rolling(3).min()

    # Volume indicators
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    df['volume_ma24'] = df['volume'].rolling(VOLUME_MA_PERIOD).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma24']
    df['volume_surge'] = df['volume_ratio'] > VOLUME_SURGE_THRESHOLD
    df['volume_low'] = df['volume_ratio'] < VOLUME_LOW_THRESHOLD

    # Volume-weighted price movement
    df['vwap_24h'] = (df['close'] * df['volume']).rolling(24).sum() / df['volume'].rolling(24).sum()

    # Price momentum
    df['price_change_1h'] = df['close'].pct_change()
    df['price_change_4h'] = df['close'].pct_change(4)
    df['price_change_24h'] = df['close'].pct_change(24)

    return df


def merge_binance_data(ohlc_df, binance_df):
    """Merge OHLC data with Binance positioning data"""
    if binance_df is None:
        log("Warning: No Binance data available, running without positioning factors")
        return ohlc_df

    ohlc = ohlc_df.copy()
    ohlc.index = ohlc.index.tz_localize('Asia/Singapore').tz_convert('UTC').tz_localize(None)

    binance = binance_df.copy()
    if binance.index.tz is not None:
        binance.index = binance.index.tz_localize(None)

    merged = pd.merge(
        ohlc.reset_index(),
        binance.reset_index(),
        on='timestamp',
        how='left'
    )
    merged.set_index('timestamp', inplace=True)

    binance_cols = [col for col in binance.columns if col in merged.columns]
    merged[binance_cols] = merged[binance_cols].ffill()

    has_binance = merged['top_trader_position_ls_ratio'].notna().sum()
    log(f"Merged data: {len(merged)} rows total, {has_binance} with Binance positioning data")

    return merged


def calculate_positioning_score_top_trader_focused(row) -> float:
    """
    Calculate positioning score using TopTraderFocused method.

    This method gives higher weight to top trader position signals
    and reduced weight to account-level signals.
    """
    score = 0.0

    # Top Trader Position (strongest signal - increased weight)
    top_long_pct = getattr(row, 'top_trader_position_long_pct', None)
    top_short_pct = getattr(row, 'top_trader_position_short_pct', None)

    if top_long_pct is not None and not pd.isna(top_long_pct):
        if top_long_pct > TOP_TRADER_STRONG_THRESHOLD:
            score += TOP_TRADER_STRONG_WEIGHT
        elif top_long_pct > TOP_TRADER_MODERATE_THRESHOLD:
            score += TOP_TRADER_MODERATE_WEIGHT

    if top_short_pct is not None and not pd.isna(top_short_pct):
        if top_short_pct > TOP_TRADER_STRONG_THRESHOLD:
            score -= TOP_TRADER_STRONG_WEIGHT
        elif top_short_pct > TOP_TRADER_MODERATE_THRESHOLD:
            score -= TOP_TRADER_MODERATE_WEIGHT

    # Top Account (reduced weight in TopTraderFocused)
    acct_long_pct = getattr(row, 'top_trader_account_long_pct', None)
    acct_short_pct = getattr(row, 'top_trader_account_short_pct', None)

    if acct_long_pct is not None and not pd.isna(acct_long_pct):
        if acct_long_pct > TOP_TRADER_MODERATE_THRESHOLD:
            score += TOP_ACCOUNT_WEIGHT
    if acct_short_pct is not None and not pd.isna(acct_short_pct):
        if acct_short_pct > TOP_TRADER_MODERATE_THRESHOLD:
            score -= TOP_ACCOUNT_WEIGHT

    # Global L/S Ratio (contrarian)
    global_ls = getattr(row, 'global_ls_ratio', None)
    if global_ls is not None and not pd.isna(global_ls):
        if global_ls < GLOBAL_LS_RATIO_CONTRARIAN_LOW:
            score += 0.5
        elif global_ls > GLOBAL_LS_RATIO_CONTRARIAN_HIGH:
            score -= 0.5

    # Funding Rate (reversal signal)
    funding_rate = getattr(row, 'funding_rate', None)
    if funding_rate is not None and not pd.isna(funding_rate):
        if funding_rate > FUNDING_RATE_EXTREME_POSITIVE:
            score -= 0.5
        elif funding_rate < FUNDING_RATE_EXTREME_NEGATIVE:
            score += 0.5

    # OI confirmation
    oi_vs_ma = getattr(row, 'oi_vs_ma24', None)
    price_change = getattr(row, 'price_change_4h', None)
    if oi_vs_ma is not None and not pd.isna(oi_vs_ma) and oi_vs_ma > OI_SURGE_THRESHOLD:
        if price_change is not None and not pd.isna(price_change):
            if price_change > 0.01:
                score += 0.25
            elif price_change < -0.01:
                score -= 0.25

    return score


def run_optimized_backtest(df, atr_med, strategy_name="Optimized"):
    """
    Run the Optimized BTC Strategy with:
    - TopTraderFocused positioning score
    - Entry=1.5, Strong=2.0 thresholds
    - StrictEntry_3loss mitigation
    - SkipNeutral (0.25) mitigation
    """
    df_bt = df.copy()

    variant = f"{strategy_name}: Stop {STOP_PCT*100:.2f}%, ATR x{ATR_MULT}, Entry≥{ENTRY_THRESHOLD}"

    risk_amount = INITIAL_EQ * RISK_PCT_INIT
    equity = INITIAL_EQ
    open_trade = None
    pnl_history = []
    trade_log = []
    consecutive_losses = 0

    for r in df_bt.itertuples():
        hr = r.Index.hour

        if pd.isna(r.sma20) or pd.isna(r.atr20):
            continue

        # Calculate positioning score using TopTraderFocused method
        positioning_score = calculate_positioning_score_top_trader_focused(r)

        volume_surge = getattr(r, 'volume_surge', False)
        volume_confirms_long = getattr(r, 'volume_confirms_long', False)
        volume_confirms_short = getattr(r, 'volume_confirms_short', False)
        volume_warning = getattr(r, 'volume_warning', False)
        oi_surge = getattr(r, 'oi_surge', False)

        # ENTRY LOGIC
        if open_trade is None and hr in ASIA_HRS and r.atr20 > atr_med:

            # ========================================
            # MITIGATION: Skip Neutral Positioning
            # ========================================
            if SKIP_NEUTRAL_ENABLED and abs(positioning_score) < SKIP_NEUTRAL_THRESHOLD:
                continue

            # ========================================
            # MITIGATION: StrictEntry after losses
            # ========================================
            if STRICT_ENTRY_AFTER_LOSSES:
                if consecutive_losses >= STRICT_ENTRY_TRIGGER:
                    if abs(positioning_score) < STRICT_ENTRY_MIN_SCORE:
                        continue

            # ========================================
            # BASE SIGNALS
            # ========================================
            long_mr_base = (r.close < r.lower_band) and (r.rsi14 < 30)
            short_mr_base = (r.close > r.upper_band) and (r.rsi14 > 70)
            long_bo_base = (r.close > r.high_3h) and (r.rsi14 > 60)
            short_bo_base = (r.close < r.low_3h) and (r.rsi14 < 40)

            # ========================================
            # ENHANCED SIGNALS WITH POSITIONING FILTERS
            # ========================================
            # MR signals - block if strongly against
            long_mr_enhanced = long_mr_base and positioning_score > -1.5
            short_mr_enhanced = short_mr_base and positioning_score < 1.5

            # BO signals - block if against
            long_bo_enhanced = long_bo_base and positioning_score > -1.0
            short_bo_enhanced = short_bo_base and positioning_score < 1.0

            # ========================================
            # STRONG POSITIONING SIGNALS
            # ========================================
            strong_positioning_long = (
                positioning_score >= STRONG_POSITIONING_THRESHOLD and
                r.rsi14 < 45 and
                r.close < r.sma20 and
                (volume_surge or oi_surge)
            )

            strong_positioning_short = (
                positioning_score <= -STRONG_POSITIONING_THRESHOLD and
                r.rsi14 > 55 and
                r.close > r.sma20 and
                (volume_surge or oi_surge)
            )

            # Combine all signals
            any_long = long_mr_enhanced or long_bo_enhanced or strong_positioning_long
            any_short = short_mr_enhanced or short_bo_enhanced or strong_positioning_short

            if any_long or any_short:
                side = "long" if any_long else "short"
                entry_price = r.close
                stop_price = entry_price * (1 - STOP_PCT) if side == "long" else entry_price * (1 + STOP_PCT)
                target_price = entry_price + ATR_MULT * r.atr20 if side == "long" else entry_price - ATR_MULT * r.atr20

                unit_risk = abs(entry_price - stop_price)
                size = risk_amount / unit_risk if unit_risk > 0 else 0

                # ========================================
                # POSITION SIZING ADJUSTMENTS
                # ========================================
                if abs(positioning_score) >= 1.5:
                    size *= 1.3  # 30% larger on strong conviction
                elif abs(positioning_score) >= 1.0:
                    size *= 1.2  # 20% larger on good conviction
                elif abs(positioning_score) >= 0.5:
                    size *= 1.1  # 10% larger on moderate conviction

                if volume_surge and oi_surge:
                    size *= 1.1  # Additional 10% on high conviction

                if volume_warning:
                    size *= 0.8  # 20% smaller on low volume

                # Determine entry type
                if any_long:
                    if strong_positioning_long:
                        entry_type = "strong_long"
                    elif long_mr_enhanced and long_mr_base:
                        entry_type = "mr_long"
                    else:
                        entry_type = "bo_long"
                else:
                    if strong_positioning_short:
                        entry_type = "strong_short"
                    elif short_mr_enhanced and short_mr_base:
                        entry_type = "mr_short"
                    else:
                        entry_type = "bo_short"

                open_trade = {
                    "variant": variant,
                    "side": side,
                    "entry_time": r.Index,
                    "entry_price": entry_price,
                    "stop": stop_price,
                    "target": target_price,
                    "size": size,
                    "entry_type": entry_type,
                    "positioning_score": positioning_score,
                    "volume_ratio": getattr(r, 'volume_ratio', 0),
                    "rsi": r.rsi14,
                    "atr": r.atr20,
                    "consecutive_losses_at_entry": consecutive_losses
                }

        # EXIT LOGIC
        elif open_trade:
            exit_price = None
            exit_reason = None

            # US session exit
            if hr in US_HRS and hr not in ASIA_HRS:
                exit_price = r.close
                exit_reason = "us_session"
            else:
                # Stop loss / Take profit
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

                trade_log.append({
                    "variant": open_trade["variant"],
                    "side": open_trade["side"],
                    "entry_time": str(open_trade["entry_time"]),
                    "entry_price": open_trade["entry_price"],
                    "stop": open_trade["stop"],
                    "target": open_trade["target"],
                    "size": int(open_trade["size"]),
                    "exit_time": str(r.Index),
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "pnl": pnl,
                    "entry_type": open_trade["entry_type"],
                    "positioning_score": open_trade["positioning_score"],
                    "volume_ratio": open_trade["volume_ratio"],
                    "entry_rsi": open_trade["rsi"],
                    "entry_atr": open_trade["atr"],
                    "consecutive_losses_at_entry": open_trade["consecutive_losses_at_entry"]
                })

                pnl_history.append(pnl)
                equity += pnl

                # Update consecutive losses counter
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                open_trade = None

    # Calculate metrics
    pnl_arr = np.array(pnl_history)
    wins = pnl_arr[pnl_arr > 0]
    losses = pnl_arr[pnl_arr < 0]
    win_rate = len(wins) / len(pnl_arr) * 100 if len(pnl_arr) > 0 else 0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    cum_return = (equity - INITIAL_EQ) / INITIAL_EQ * 100

    eq_array = INITIAL_EQ + np.cumsum(pnl_history)
    drawdowns = np.maximum.accumulate(eq_array) - eq_array
    max_dd = float(drawdowns.max() / np.maximum.accumulate(eq_array).max() * 100) if len(eq_array) > 0 else 0

    consec_losses = [sum(1 for _ in grp) for k, grp in itertools.groupby(pnl_arr < 0) if k]
    consec_wins = [sum(1 for _ in grp) for k, grp in itertools.groupby(pnl_arr > 0) if k]
    max_consec_losses = max(consec_losses) if consec_losses else 0
    max_consec_wins = max(consec_wins) if consec_wins else 0

    # Count streaks of 4+ losses
    streaks_4plus = sum(1 for s in consec_losses if s >= 4)
    streaks_6plus = sum(1 for s in consec_losses if s >= 6)

    trade_df = pd.DataFrame(trade_log)

    # Entry type performance breakdown
    entry_type_stats = {}
    if len(trade_df) > 0 and 'entry_type' in trade_df.columns:
        for entry_type in trade_df['entry_type'].unique():
            type_trades = trade_df[trade_df['entry_type'] == entry_type]
            type_wins = len(type_trades[type_trades['pnl'] > 0])
            type_pnl = type_trades['pnl'].sum()
            entry_type_stats[entry_type] = {
                'count': len(type_trades),
                'win_rate': type_wins / len(type_trades) * 100 if len(type_trades) > 0 else 0,
                'total_pnl': type_pnl
            }

    # Recent performance
    if len(trade_df) > 0:
        trade_df['exit_time_dt'] = pd.to_datetime(trade_df['exit_time'])
        now = trade_df['exit_time_dt'].max()

        trades_7d = trade_df[trade_df['exit_time_dt'] >= now - pd.Timedelta(days=7)]
        pnl_7d = float(trades_7d['pnl'].sum()) if len(trades_7d) > 0 else 0
        wins_7d = len(trades_7d[trades_7d['pnl'] > 0])
        win_rate_7d = wins_7d / len(trades_7d) * 100 if len(trades_7d) > 0 else 0

        trades_30d = trade_df[trade_df['exit_time_dt'] >= now - pd.Timedelta(days=30)]
        pnl_30d = float(trades_30d['pnl'].sum()) if len(trades_30d) > 0 else 0
        wins_30d = len(trades_30d[trades_30d['pnl'] > 0])
        win_rate_30d = wins_30d / len(trades_30d) * 100 if len(trades_30d) > 0 else 0

        trades_3m = trade_df[trade_df['exit_time_dt'] >= now - pd.Timedelta(days=90)]
        pnl_3m = float(trades_3m['pnl'].sum()) if len(trades_3m) > 0 else 0
        wins_3m = len(trades_3m[trades_3m['pnl'] > 0])
        win_rate_3m = wins_3m / len(trades_3m) * 100 if len(trades_3m) > 0 else 0

        trades_1y = trade_df[trade_df['exit_time_dt'] >= now - pd.Timedelta(days=365)]
        pnl_1y = float(trades_1y['pnl'].sum()) if len(trades_1y) > 0 else 0
        wins_1y = len(trades_1y[trades_1y['pnl'] > 0])
        win_rate_1y = wins_1y / len(trades_1y) * 100 if len(trades_1y) > 0 else 0

        trade_df = trade_df.drop(columns=['exit_time_dt'])
    else:
        pnl_7d = pnl_30d = pnl_3m = pnl_1y = 0
        win_rate_7d = win_rate_30d = win_rate_3m = win_rate_1y = 0
        trades_7d = trades_30d = trades_3m = trades_1y = pd.DataFrame()

    metrics = {
        "Variant": variant,
        "Capital_Risked": f"{RISK_PCT_INIT*100:.1f}%",
        "Trades": int(len(pnl_arr)),
        "Win_rate_pct": round(win_rate, 1),
        "Win_Loss_ratio": round(win_loss_ratio, 2),
        "Cum_return_pct": round(cum_return, 0),
        "Max_DD_pct": round(max_dd, 1),
        "Max_consec_losses": int(max_consec_losses),
        "Max_consec_wins": int(max_consec_wins),
        "Streaks_4plus": streaks_4plus,
        "Streaks_6plus": streaks_6plus,
        "Win_rate_7d_pct": round(win_rate_7d, 0),
        "Trades_7d": int(len(trades_7d)) if len(trade_df) > 0 else 0,
        "PnL_7d": round(pnl_7d, 0),
        "Win_rate_30d_pct": round(win_rate_30d, 0),
        "Trades_30d": int(len(trades_30d)) if len(trade_df) > 0 else 0,
        "PnL_30d": round(pnl_30d, 0),
        "Win_rate_3m_pct": round(win_rate_3m, 0),
        "Trades_3m": int(len(trades_3m)) if len(trade_df) > 0 else 0,
        "PnL_3m": round(pnl_3m, 0),
        "Win_rate_1y_pct": round(win_rate_1y, 0),
        "Trades_1y": int(len(trades_1y)) if len(trade_df) > 0 else 0,
        "PnL_1y": round(pnl_1y, 0),
        "Entry_Type_Stats": entry_type_stats
    }

    live_position = None
    if open_trade:
        live_position = {
            "variant": variant,
            "entry_time": str(open_trade["entry_time"]),
            "position": open_trade["side"],
            "entry_price": round(open_trade["entry_price"], 0),
            "stop_price": round(open_trade["stop"], 0),
            "tp_price": round(open_trade["target"], 0),
            "entry_type": open_trade["entry_type"],
            "positioning_score": round(open_trade["positioning_score"], 2),
            "volume_ratio": round(open_trade["volume_ratio"], 2)
        }

    equity_curve = [float(x) for x in list(eq_array)[-500:]]

    equity_curve_ts = []
    running_equity = INITIAL_EQ
    for trade in trade_log:
        running_equity += trade['pnl']
        equity_curve_ts.append({
            'exit_time': trade['exit_time'],
            'equity': running_equity,
            'pnl': trade['pnl'],
            'entry_type': trade['entry_type']
        })

    return trade_df, metrics, live_position, equity_curve, equity_curve_ts


def calculate_enhanced_signals(df):
    """Calculate enhanced signals using volume and Binance positioning data"""
    df = df.copy()

    has_binance = 'top_trader_position_long_pct' in df.columns and df['top_trader_position_long_pct'].notna().any()

    # Volume signals
    df['volume_confirms_long'] = (df['volume_surge']) & (df['price_change_1h'] > 0)
    df['volume_confirms_short'] = (df['volume_surge']) & (df['price_change_1h'] < 0)
    df['volume_warning'] = df['volume_low']

    if has_binance:
        # Positioning signals
        df['top_trader_bullish'] = df['top_trader_position_long_pct'] > TOP_TRADER_MODERATE_THRESHOLD
        df['top_trader_bearish'] = df['top_trader_position_short_pct'] > TOP_TRADER_MODERATE_THRESHOLD

        if 'top_trader_account_long_pct' in df.columns:
            df['top_account_bullish'] = df['top_trader_account_long_pct'] > TOP_TRADER_MODERATE_THRESHOLD
            df['top_account_bearish'] = df['top_trader_account_short_pct'] > TOP_TRADER_MODERATE_THRESHOLD
        else:
            df['top_account_bullish'] = False
            df['top_account_bearish'] = False

        if 'global_ls_ratio' in df.columns:
            df['global_contrarian_long'] = df['global_ls_ratio'] < GLOBAL_LS_RATIO_CONTRARIAN_LOW
            df['global_contrarian_short'] = df['global_ls_ratio'] > GLOBAL_LS_RATIO_CONTRARIAN_HIGH
        else:
            df['global_contrarian_long'] = False
            df['global_contrarian_short'] = False

        if 'funding_rate' in df.columns:
            df['funding_extreme_positive'] = df['funding_rate'] > FUNDING_RATE_EXTREME_POSITIVE
            df['funding_extreme_negative'] = df['funding_rate'] < FUNDING_RATE_EXTREME_NEGATIVE
        else:
            df['funding_extreme_positive'] = False
            df['funding_extreme_negative'] = False

        if 'oi_vs_ma24' in df.columns:
            df['oi_surge'] = df['oi_vs_ma24'] > OI_SURGE_THRESHOLD
            df['oi_drop'] = df['oi_vs_ma24'] < OI_DROP_THRESHOLD
        else:
            df['oi_surge'] = False
            df['oi_drop'] = False

        log("Enhanced signals calculated with Binance positioning data")
    else:
        df['top_trader_bullish'] = False
        df['top_trader_bearish'] = False
        df['top_account_bullish'] = False
        df['top_account_bearish'] = False
        df['global_contrarian_long'] = False
        df['global_contrarian_short'] = False
        df['funding_extreme_positive'] = False
        df['funding_extreme_negative'] = False
        df['oi_surge'] = False
        df['oi_drop'] = False
        log("Warning: Running with volume signals only (no Binance positioning data)")

    return df


def main():
    log("=" * 70)
    log("BTC Enhanced Strategy with Streak Mitigation")
    log("=" * 70)
    log("")
    log("OPTIMIZED CONFIGURATION:")
    log(f"  Positioning Method: TopTraderFocused")
    log(f"  Entry Threshold: {ENTRY_THRESHOLD}")
    log(f"  Strong Positioning Threshold: {STRONG_POSITIONING_THRESHOLD}")
    log(f"  Skip Neutral: {SKIP_NEUTRAL_ENABLED} (threshold: {SKIP_NEUTRAL_THRESHOLD})")
    log(f"  Strict Entry After Losses: {STRICT_ENTRY_AFTER_LOSSES}")
    log(f"    - Trigger after: {STRICT_ENTRY_TRIGGER} consecutive losses")
    log(f"    - Min score required: {STRICT_ENTRY_MIN_SCORE}")
    log("")

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load OHLC data
    log("Loading OHLC data...")
    df_raw = load_ohlc_data()
    log(f"Loaded {len(df_raw):,} rows")
    log(f"  Date range: {df_raw.index.min()} to {df_raw.index.max()}")

    # Calculate technical indicators
    log("Calculating technical indicators...")
    df = calculate_indicators(df_raw)
    atr_med = df['atr20'].median()

    # Load and merge Binance positioning data
    log("Loading Binance positioning data...")
    binance_df = load_binance_data()
    if binance_df is not None:
        df = merge_binance_data(df, binance_df)

    # Calculate enhanced signals
    log("Calculating enhanced signals...")
    df = calculate_enhanced_signals(df)

    # Run optimized backtest
    log("")
    log("=" * 70)
    log("RUNNING OPTIMIZED BACKTEST")
    log("=" * 70)

    trade_log_df, metrics, live_position, equity_curve, equity_curve_ts = run_optimized_backtest(
        df, atr_med, strategy_name="Optimized"
    )
    log(f"  -> {metrics['Trades']} trades, {metrics['Win_rate_pct']:.1f}% win rate, {metrics['Cum_return_pct']:,.0f}% return")
    log(f"  -> Max Consecutive Losses: {metrics['Max_consec_losses']}")
    log(f"  -> Streaks 4+: {metrics['Streaks_4plus']}, Streaks 6+: {metrics['Streaks_6plus']}")

    # Save results
    log("")
    log("Saving results...")

    # Metrics JSON
    metrics_data = {
        "metrics": metrics,
        "live_position": live_position,
        "equity_curve": equity_curve,
        "last_updated": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        "data_latest_timestamp": str(df_raw.index.max()),
        "strategy_parameters": {
            "stop_pct": STOP_PCT,
            "atr_mult": ATR_MULT,
            "risk_pct": RISK_PCT_INIT,
            "entry_threshold": ENTRY_THRESHOLD,
            "strong_positioning_threshold": STRONG_POSITIONING_THRESHOLD,
            "skip_neutral_enabled": SKIP_NEUTRAL_ENABLED,
            "skip_neutral_threshold": SKIP_NEUTRAL_THRESHOLD,
            "strict_entry_after_losses": STRICT_ENTRY_AFTER_LOSSES,
            "strict_entry_trigger": STRICT_ENTRY_TRIGGER,
            "strict_entry_min_score": STRICT_ENTRY_MIN_SCORE,
            "top_trader_strong_threshold": TOP_TRADER_STRONG_THRESHOLD,
            "top_trader_moderate_threshold": TOP_TRADER_MODERATE_THRESHOLD,
        }
    }
    with open(f"{RESULTS_DIR}/metrics.json", 'w') as f:
        json.dump(metrics_data, f, indent=2)
    log(f"  Saved {RESULTS_DIR}/metrics.json")

    # Trade log CSV
    if len(trade_log_df) > 0:
        trade_log_df.to_csv(f"{RESULTS_DIR}/trade_log.csv", index=False)
        log(f"  Saved {RESULTS_DIR}/trade_log.csv ({len(trade_log_df)} trades)")

    # Equity curve CSV
    if equity_curve_ts:
        equity_df = pd.DataFrame(equity_curve_ts)
        equity_df.to_csv(f"{RESULTS_DIR}/equity_curve_ts.csv", index=False)
        log(f"  Saved {RESULTS_DIR}/equity_curve_ts.csv ({len(equity_curve_ts)} points)")

    # Print summary
    log("")
    log("=" * 70)
    log("BACKTEST RESULTS SUMMARY")
    log("=" * 70)

    log("")
    log("OPTIMIZED STRATEGY PERFORMANCE")
    log("-" * 50)
    log(f"  Total Trades:       {metrics['Trades']:,}")
    log(f"  Win Rate:           {metrics['Win_rate_pct']:.1f}%")
    log(f"  Win/Loss Ratio:     {metrics['Win_Loss_ratio']:.2f}")
    log(f"  Cumulative Return:  {metrics['Cum_return_pct']:,.0f}%")
    log(f"  Max Drawdown:       {metrics['Max_DD_pct']:.1f}%")
    log(f"  Max Consec Losses:  {metrics['Max_consec_losses']}")
    log(f"  Streaks 4+:         {metrics['Streaks_4plus']}")
    log(f"  Streaks 6+:         {metrics['Streaks_6plus']}")
    log(f"  7-Day PnL:          ${metrics['PnL_7d']:,.0f}")
    log(f"  30-Day PnL:         ${metrics['PnL_30d']:,.0f}")
    log(f"  1-Year PnL:         ${metrics['PnL_1y']:,.0f}")

    log("")
    log("ENTRY TYPE BREAKDOWN")
    log("-" * 50)
    for entry_type, stats in sorted(metrics.get('Entry_Type_Stats', {}).items(),
                                    key=lambda x: x[1]['total_pnl'], reverse=True):
        log(f"  {entry_type}:")
        log(f"    Trades: {stats['count']}, Win Rate: {stats['win_rate']:.0f}%, PnL: ${stats['total_pnl']:,.0f}")

    if live_position:
        log("")
        log("=" * 70)
        log(f"LIVE POSITION: {live_position['position'].upper()} @ ${live_position['entry_price']:,.0f}")
        log(f"  Type: {live_position['entry_type']}")
        log(f"  Positioning Score: {live_position['positioning_score']:.2f}")
        log(f"  Volume Ratio: {live_position['volume_ratio']:.2f}x")
        log(f"  Stop: ${live_position['stop_price']:,.0f} | Target: ${live_position['tp_price']:,.0f}")
    else:
        log("")
        log("No live position")

    log("")
    log("=" * 70)
    log("Backtest completed successfully")


if __name__ == "__main__":
    main()
