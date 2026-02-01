#!/usr/bin/env python3
"""
Tiered Capital Backtest
=======================
Runs backtest with tiered position sizing based on equity levels.

Tier Structure (capital risked per trade):
- Initial ($100k starting): 5% of $100k = $5k
- $150k - $225k: 7% of $150k = $10.5k
- $225k - $337.5k: 6.5% of $225k = $14.625k
- $337.5k - $507k: 6% of $337.5k = $20.25k
- $507k - $760k: 5.5% of $507k = $28k
- $760k - $1.2m: $38k
- $1.2m onwards: $54k
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


def run_tiered_backtest(df, strategy_config, entry_filter, entry_hours=None, initial_capital=INITIAL_CAPITAL):
    """
    Run backtest with tiered capital risk management.

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
    tier_transitions = []

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

    for i, row in df.iterrows():
        hr = i.hour

        # Skip if missing critical data
        if pd.isna(row.get('sma20')) or pd.isna(row.get('atr')) or pd.isna(row.get('sma50')) or pd.isna(row.get('vol_ma')):
            continue

        positioning_score = calculate_positioning_score(row)
        volume_score = calculate_volume_score(row)
        vol_ratio = row.get('vol_ratio', 1.0)
        quality_score = positioning_score + volume_score

        # Check for tier transition
        new_tier = get_tier_name(equity)
        if new_tier != current_tier:
            tier_transitions.append({
                'timestamp': str(i),
                'equity': equity,
                'from_tier': current_tier,
                'to_tier': new_tier,
            })
            current_tier = new_tier

        # Get current risk based on equity tier
        current_risk = get_risk_for_equity(equity)

        # Record equity
        equity_curve.append({
            'timestamp': i,
            'equity': equity,
            'capital': equity,
            'in_position': len(open_trades) > 0,
            'btc_price': row['close'],
            'current_risk': current_risk,
            'tier': current_tier,
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
                    "risk_used": trade["risk_used"],
                    "tier_at_entry": trade["tier_at_entry"],
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

                # Use tiered risk instead of fixed BASE_RISK
                tiered_risk = get_risk_for_equity(equity)
                size = tiered_risk / unit_risk if unit_risk > 0 else 0

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
                        "risk_used": tiered_risk,
                        "tier_at_entry": current_tier,
                    })

    # Close remaining trades
    if open_trades:
        last_row = df.iloc[-1]
        for trade in open_trades:
            pnl = (last_row['close'] - trade["entry_price"]) * trade["size"]
            pnl_history.append(pnl)
            equity += pnl

    return trades, equity_curve, tier_transitions


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

    # Risk-adjusted metrics
    avg_risk_used = np.mean([t.get('risk_used', 5000) for t in complete_trades])

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
        'avg_risk_per_trade': avg_risk_used,
    }


def main():
    print("=" * 70)
    print("TIERED CAPITAL BACKTEST - ASIAN HOURS")
    print("=" * 70)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print("\nRisk Tiers:")
    for min_eq, max_eq, risk in RISK_TIERS:
        if max_eq == float('inf'):
            print(f"  ${min_eq/1000:.0f}k+: ${risk/1000:.1f}k per trade")
        else:
            print(f"  ${min_eq/1000:.0f}k - ${max_eq/1000:.0f}k: ${risk/1000:.1f}k per trade")
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
    output_dir = os.path.join(base_dir, 'results_tiered_capital')
    os.makedirs(output_dir, exist_ok=True)

    all_stats = {}
    all_transitions = {}

    print("\n" + "=" * 70)
    print("RUNNING TIERED CAPITAL BACKTESTS")
    print("=" * 70)

    for strategy_name, strategy_config in STRATEGY_CONFIGS.items():
        full_name = f"{strategy_name}_Tiered"
        print(f"\nRunning: {full_name}")

        trades, equity_curve, tier_transitions = run_tiered_backtest(
            df,
            strategy_config,
            ENTRY_FILTER,
            entry_hours=ASIAN_HOURS
        )

        stats = calculate_statistics(trades, equity_curve)

        if stats:
            all_stats[full_name] = {
                'strategy': strategy_name,
                'version': 'Tiered_AHours',
                **stats
            }
            all_transitions[full_name] = tier_transitions

            print(f"  Trades: {stats['total_trades']}, Return: {stats['total_return_pct']:.1f}%, "
                  f"Final: ${stats['final_equity']:,.0f}")
            print(f"  WR: {stats['win_rate_pct']:.1f}%, MaxDD: {stats['max_drawdown_pct']:.1f}%, "
                  f"MaxCL: {stats['max_losing_streak']}")
            print(f"  Tier Transitions: {len(tier_transitions)}, Avg Risk: ${stats['avg_risk_per_trade']:,.0f}")

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

            # Export tier transitions
            if tier_transitions:
                transitions_df = pd.DataFrame(tier_transitions)
                transitions_file = os.path.join(output_dir, f'tier_transitions_{full_name}.csv')
                transitions_df.to_csv(transitions_file, index=False)
                print(f"  Exported: tier_transitions_{full_name}.csv")
        else:
            print(f"  No valid trades")

    # Save statistics
    stats_file = os.path.join(output_dir, 'strategy_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\nSaved statistics to: {stats_file}")

    # =========================================
    # COMPARISON: Tiered vs Fixed Risk
    # =========================================
    print("\n" + "=" * 70)
    print("COMPARISON: TIERED vs FIXED RISK (Asian Hours)")
    print("=" * 70)

    # Load fixed risk results for comparison
    fixed_stats_file = os.path.join(base_dir, 'results_asian_hours', 'strategy_statistics.json')
    if os.path.exists(fixed_stats_file):
        with open(fixed_stats_file, 'r') as f:
            fixed_stats = json.load(f)

        print(f"\n{'Strategy':<25} {'Version':<10} {'Return%':>10} {'Final $':>15} {'MaxDD%':>8} {'Trades':>7}")
        print("-" * 80)

        for strategy_name in STRATEGY_CONFIGS.keys():
            # Fixed risk results
            fixed_name = f"{strategy_name}_AHours"
            if fixed_name in fixed_stats:
                fs = fixed_stats[fixed_name]
                print(f"{strategy_name:<25} {'Fixed':<10} {fs['total_return_pct']:>9.1f}% "
                      f"${fs['final_equity']:>13,.0f} {fs['max_drawdown_pct']:>7.1f}% {fs['total_trades']:>7}")

            # Tiered risk results
            tiered_name = f"{strategy_name}_Tiered"
            if tiered_name in all_stats:
                ts = all_stats[tiered_name]
                print(f"{'':<25} {'Tiered':<10} {ts['total_return_pct']:>9.1f}% "
                      f"${ts['final_equity']:>13,.0f} {ts['max_drawdown_pct']:>7.1f}% {ts['total_trades']:>7}")

            print()
    else:
        print("\nFixed risk results not found for comparison.")

    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 70)
    print("TIERED CAPITAL RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Strategy':<25} {'Return%':>10} {'Final $':>15} {'MaxDD%':>8} {'MaxDD$':>12} {'AvgRisk':>10}")
    print("-" * 85)
    for name, stats in sorted(all_stats.items(), key=lambda x: x[1]['final_equity'], reverse=True):
        print(f"{name:<25} {stats['total_return_pct']:>9.1f}% ${stats['final_equity']:>13,.0f} "
              f"{stats['max_drawdown_pct']:>7.1f}% ${stats['max_drawdown_abs']:>10,.0f} "
              f"${stats['avg_risk_per_trade']:>8,.0f}")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
