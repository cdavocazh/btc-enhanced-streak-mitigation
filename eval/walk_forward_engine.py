#!/usr/bin/env python3
"""
Walk-Forward Optimization Engine for BTC Strategies
====================================================
Implements rolling-window backtesting with in-sample optimization
and out-of-sample validation (Robert Pardo methodology).

This module:
1. Loads price and positioning data
2. Creates rolling training/testing windows
3. Optimizes parameters on in-sample data
4. Validates on out-of-sample data
5. Tracks efficiency (OOS/IS ratio)
6. Measures parameter stability
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import itertools

# Add parent directories to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'backtest_15min'))
sys.path.insert(0, os.path.join(REPO_DIR, 'backtest_15min_new'))
sys.path.insert(0, os.path.join(REPO_DIR, 'backtest_15min_new_streak_a'))

from config import (
    WalkForwardConfig, DEFAULT_WF_CONFIG,
    TIERED_STRATEGY_CONFIGS, ADAPTIVE_STRATEGY_CONFIGS,
    STRATEGY_PARAMETER_RANGES, ENTRY_FILTER_CONFIGS, ADAPTIVE_ENTRY_FILTER,
    RISK_TIERS, STREAK_RULES, ASIAN_HOURS,
    ATR_PERIOD, RSI_PERIOD, SMA_PERIOD, ADX_PERIOD,
    TOP_TRADER_STRONG, TOP_TRADER_MODERATE,
    is_adaptive_strategy, get_strategy_config
)


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    strategy_name: str
    start_time: datetime
    end_time: datetime
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    max_losing_streak: int
    parameters: Dict[str, Any]
    is_in_sample: bool = True


@dataclass
class WalkForwardResult:
    """Results from a complete walk-forward optimization."""
    strategy_name: str
    entry_filter: str
    windows: List[Dict[str, Any]]
    overall_is_return: float
    overall_oos_return: float
    oos_efficiency: float
    parameter_stability: float
    status: str  # HEALTHY, WARNING, CRITICAL
    timestamp: str
    data_bars: int


def log(msg: str):
    """Print timestamped log message."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}")


class WalkForwardEngine:
    """
    Walk-Forward Optimization Engine.

    Implements rolling-window backtesting to validate strategy robustness.
    """

    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or DEFAULT_WF_CONFIG
        self.price_data: Optional[pd.DataFrame] = None
        self.results_dir = os.path.join(SCRIPT_DIR, 'results')
        os.makedirs(self.results_dir, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load and merge price + positioning data."""
        log("Loading data...")

        try:
            # Try to import from backtest_15min
            from load_15min_data import merge_all_data_15min
            df = merge_all_data_15min()
            log(f"Loaded {len(df)} rows from merge_all_data_15min()")
        except ImportError:
            log("Could not import load_15min_data, loading from CSV files...")
            df = self._load_data_from_csv()

        if df is None or len(df) == 0:
            raise ValueError("No data loaded")

        self.price_data = df
        return df

    def _load_data_from_csv(self) -> Optional[pd.DataFrame]:
        """Fallback: Load data directly from CSV files."""
        # Try loading from binance-futures-data
        data_dir = os.path.join(REPO_DIR, 'binance-futures-data', 'data')
        price_file = os.path.join(data_dir, 'price.csv')

        if not os.path.exists(price_file):
            log(f"Price file not found: {price_file}")
            return None

        df = pd.read_csv(price_file)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        return df

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators including ADX."""
        df = df.copy()

        bb_period = 80
        volume_ma_period = 96

        # Price indicators
        df['sma20'] = df['close'].rolling(bb_period).mean()
        df['sma50'] = df['close'].rolling(SMA_PERIOD).mean()
        df['std20'] = df['close'].rolling(bb_period).std()
        df['upper_band'] = df['sma20'] + 2 * df['std20']
        df['lower_band'] = df['sma20'] - 2 * df['std20']

        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(ATR_PERIOD).mean()
        df['atr_short'] = tr.rolling(5).mean()
        df['atr_adaptive'] = df[['atr', 'atr_short']].min(axis=1)

        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
        loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
        df['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        # Trend and pullback
        df['uptrend'] = df['close'] > df['sma50']
        df['pullback_pct'] = (df['high'].rolling(16).max() - df['close']) / df['close'] * 100

        # Volume indicators
        if 'volume' in df.columns:
            df['vol_ma'] = df['volume'].rolling(volume_ma_period).mean()
            df['vol_ratio'] = df['volume'] / df['vol_ma'].replace(0, np.nan)
            df['vol_ma_4h'] = df['volume'].rolling(16).mean()
            df['vol_trend'] = df['vol_ma_4h'] - df['vol_ma_4h'].shift(4)
            df['vol_increasing'] = df['vol_trend'] > 0
            df['price_change'] = df['close'].pct_change()
            df['bullish_volume'] = (df['price_change'] > 0) & (df['vol_ratio'] > 1.0)
        else:
            df['vol_ratio'] = 1.0
            df['vol_increasing'] = True
            df['bullish_volume'] = True

        # ADX calculation
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr_smooth = tr.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()

        df['plus_di'] = 100 * plus_dm_smooth / atr_smooth.replace(0, np.nan)
        df['minus_di'] = 100 * minus_dm_smooth / atr_smooth.replace(0, np.nan)

        di_diff = (df['plus_di'] - df['minus_di']).abs()
        di_sum = df['plus_di'] + df['minus_di']
        dx = 100 * di_diff / di_sum.replace(0, np.nan)
        df['adx'] = dx.ewm(alpha=1/ADX_PERIOD, adjust=False).mean()

        df['is_trending'] = df['adx'] > 20
        df['is_strong_trend'] = df['adx'] > 30

        # Higher Highs / Higher Lows
        lookback = 20
        df['swing_high_1'] = df['high'].rolling(lookback).max()
        df['swing_high_2'] = df['high'].rolling(lookback).max().shift(lookback)
        df['swing_low_1'] = df['low'].rolling(lookback).min()
        df['swing_low_2'] = df['low'].rolling(lookback).min().shift(lookback)

        df['making_hh'] = df['swing_high_1'] > df['swing_high_2']
        df['making_hl'] = df['swing_low_1'] > df['swing_low_2']
        df['trend_confirmed'] = df['making_hh'] & df['making_hl']

        return df

    def calculate_positioning_score(self, row) -> float:
        """Calculate positioning score from market data."""
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

    def calculate_volume_score(self, row) -> float:
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

    def run_backtest_window(
        self,
        df: pd.DataFrame,
        strategy_name: str,
        entry_filter: Dict[str, Any],
        strategy_config: Dict[str, Any],
        initial_capital: float = 100000,
        entry_hours: set = None
    ) -> BacktestResult:
        """
        Run a backtest on a data window.

        This is a simplified backtest for evaluation purposes.
        For full backtest, use the scripts in backtest_15min_new or backtest_15min_new_streak_a.
        """
        if entry_hours is None:
            entry_hours = ASIAN_HOURS

        # Compute indicators if not already done
        if 'atr' not in df.columns:
            df = self.compute_indicators(df)

        trades = []
        equity = initial_capital
        pnl_history = []
        open_trades = []
        consecutive_losses = 0
        max_losing_streak = 0

        # Entry filter params
        min_pos_long = entry_filter.get('min_pos_long', 0.4)
        rsi_long_range = entry_filter.get('rsi_long_range', (20, 45))
        pullback_range = entry_filter.get('pullback_range', (0.5, 3.0))
        min_pos_score = entry_filter.get('min_pos_score', 0.15)

        # Strategy params
        is_adaptive = is_adaptive_strategy(strategy_name)
        adx_filter = strategy_config.get('adx_filter', False)
        progressive_pos = strategy_config.get('progressive_pos', False)

        # Fixed params
        stop_atr_mult = 1.8
        tp_atr_mult = 4.5

        for i, row in df.iterrows():
            if pd.isna(row.get('atr')) or pd.isna(row.get('sma50')):
                continue

            hr = i.hour if hasattr(i, 'hour') else 0
            positioning_score = self.calculate_positioning_score(row)

            # Get tiered risk
            base_risk = self._get_risk_for_equity(equity)
            adjusted_risk = self._get_streak_adjusted_risk(base_risk, consecutive_losses)

            # Check exits for open trades
            trades_to_close = []
            for idx, trade in enumerate(open_trades):
                trade['bars_held'] += 1
                exit_price = None
                exit_reason = None

                if trade["side"] == "long":
                    if row['low'] <= trade["stop"]:
                        exit_price = trade["stop"]
                        exit_reason = "stop_loss"
                    elif row['high'] >= trade["target"]:
                        exit_price = trade["target"]
                        exit_reason = "take_profit"

                if exit_price is not None:
                    pnl = (exit_price - trade["entry_price"]) * trade["size"]
                    pnl_history.append(pnl)
                    equity += pnl

                    trades.append({
                        "entry_time": trade["entry_time"],
                        "exit_time": i,
                        "entry_price": trade["entry_price"],
                        "exit_price": exit_price,
                        "exit_reason": exit_reason,
                        "pnl": pnl,
                        "size": trade["size"],
                    })

                    if pnl < 0:
                        consecutive_losses += 1
                        max_losing_streak = max(max_losing_streak, consecutive_losses)
                    else:
                        consecutive_losses = 0

                    trades_to_close.append(idx)

            for idx in sorted(trades_to_close, reverse=True):
                open_trades.pop(idx)

            # Entry logic
            hour_ok = hr in entry_hours
            if hour_ok and len(open_trades) == 0:
                if abs(positioning_score) < min_pos_score:
                    continue

                # ADX filter for adaptive strategies
                if adx_filter:
                    adx_value = row.get('adx', 0)
                    if pd.isna(adx_value) or adx_value < 20:
                        continue

                # Progressive positioning for adaptive strategies
                if progressive_pos:
                    pos_threshold = self._get_progressive_pos_threshold(consecutive_losses)
                else:
                    pos_threshold = min_pos_long

                entry_signal = False
                pullback = row.get('pullback_pct', 0)
                uptrend = row.get('uptrend', False)
                rsi = row.get('rsi', 50)

                if pd.isna(rsi) or pd.isna(pullback):
                    continue

                if positioning_score >= pos_threshold:
                    if uptrend and pullback_range[0] < pullback < pullback_range[1]:
                        if rsi_long_range[0] < rsi < rsi_long_range[1]:
                            entry_signal = True

                if entry_signal:
                    entry_price = row['close']
                    atr = row['atr']
                    stop_distance = atr * stop_atr_mult
                    stop_price = entry_price - stop_distance
                    target_price = entry_price + atr * tp_atr_mult

                    unit_risk = abs(entry_price - stop_price)
                    size = adjusted_risk / unit_risk if unit_risk > 0 else 0

                    if size > 0:
                        open_trades.append({
                            "side": "long",
                            "entry_time": i,
                            "entry_price": entry_price,
                            "stop": stop_price,
                            "target": target_price,
                            "size": size,
                            "bars_held": 0,
                        })

        # Close remaining trades
        if open_trades and len(df) > 0:
            last_row = df.iloc[-1]
            for trade in open_trades:
                pnl = (last_row['close'] - trade["entry_price"]) * trade["size"]
                pnl_history.append(pnl)
                equity += pnl

        # Calculate statistics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['pnl'] > 0])
        losing_trades = len([t for t in trades if t['pnl'] <= 0])

        total_return_pct = (equity - initial_capital) / initial_capital * 100
        win_rate_pct = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Max drawdown
        running_equity = [initial_capital]
        for pnl in pnl_history:
            running_equity.append(running_equity[-1] + pnl)
        peak = pd.Series(running_equity).cummax()
        drawdown = (peak - pd.Series(running_equity)) / peak * 100
        max_drawdown_pct = drawdown.max() if len(drawdown) > 0 else 0

        # Profit factor
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio
        if len(pnl_history) > 1:
            returns = pd.Series(pnl_history)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        return BacktestResult(
            strategy_name=strategy_name,
            start_time=df.index[0] if len(df) > 0 else datetime.now(),
            end_time=df.index[-1] if len(df) > 0 else datetime.now(),
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            win_rate_pct=win_rate_pct,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            max_losing_streak=max_losing_streak,
            parameters=strategy_config,
        )

    def _get_risk_for_equity(self, equity: float) -> float:
        """Get risk amount for current equity level."""
        for min_eq, max_eq, risk in RISK_TIERS:
            if min_eq <= equity < max_eq:
                return risk
        return RISK_TIERS[-1][2]

    def _get_streak_adjusted_risk(self, base_risk: float, consecutive_losses: int) -> float:
        """Apply streak mitigation to risk."""
        if consecutive_losses < 3:
            return base_risk

        reduction = 1.0
        if consecutive_losses >= 3:
            reduction *= (1 - STREAK_RULES[3]['reduction'])
        if consecutive_losses >= 6:
            reduction *= (1 - STREAK_RULES[6]['reduction'])
        if consecutive_losses >= 9:
            reduction *= (1 - STREAK_RULES[9]['reduction'])

        return base_risk * reduction

    def _get_progressive_pos_threshold(self, consecutive_losses: int) -> float:
        """Get adaptive positioning threshold based on losses."""
        thresholds = {0: 0.4, 1: 0.5, 2: 0.6, 3: 0.8, 4: 1.0}
        if consecutive_losses in thresholds:
            return thresholds[consecutive_losses]
        elif consecutive_losses >= max(thresholds.keys()):
            return thresholds[max(thresholds.keys())]
        return 0.4

    def run_walk_forward(
        self,
        strategy_name: str,
        entry_filter_name: str = "baseline"
    ) -> WalkForwardResult:
        """
        Run walk-forward optimization for a strategy.

        Creates rolling windows and tracks IS vs OOS performance.
        """
        log(f"Running walk-forward for {strategy_name} with {entry_filter_name} filter")

        # Load data if not already loaded
        if self.price_data is None:
            self.load_data()

        df = self.compute_indicators(self.price_data.copy())

        # Get strategy config
        strategy_config = get_strategy_config(strategy_name)
        if strategy_config is None:
            log(f"Unknown strategy: {strategy_name}")
            return None

        entry_filter = ENTRY_FILTER_CONFIGS.get(entry_filter_name, ENTRY_FILTER_CONFIGS['baseline'])

        # Create windows
        total_bars = len(df)
        training_bars = self.config.training_window
        testing_bars = self.config.testing_window
        step_size = self.config.step_size

        windows = []
        window_idx = 0
        start_idx = 0

        while start_idx + training_bars + testing_bars <= total_bars:
            train_start = start_idx
            train_end = start_idx + training_bars
            test_start = train_end
            test_end = min(train_end + testing_bars, total_bars)

            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]

            # Run IS backtest
            is_result = self.run_backtest_window(
                train_df, strategy_name, entry_filter, strategy_config
            )
            is_result.is_in_sample = True

            # Run OOS backtest
            oos_result = self.run_backtest_window(
                test_df, strategy_name, entry_filter, strategy_config
            )
            oos_result.is_in_sample = False

            # Calculate efficiency
            efficiency = (oos_result.total_return_pct / is_result.total_return_pct
                         if is_result.total_return_pct != 0 else 0)

            window_result = {
                'window': window_idx,
                'train_start': str(train_df.index[0]) if len(train_df) > 0 else None,
                'train_end': str(train_df.index[-1]) if len(train_df) > 0 else None,
                'test_start': str(test_df.index[0]) if len(test_df) > 0 else None,
                'test_end': str(test_df.index[-1]) if len(test_df) > 0 else None,
                'is_return_pct': is_result.total_return_pct,
                'is_sharpe': is_result.sharpe_ratio,
                'is_max_dd': is_result.max_drawdown_pct,
                'is_trades': is_result.total_trades,
                'oos_return_pct': oos_result.total_return_pct,
                'oos_sharpe': oos_result.sharpe_ratio,
                'oos_max_dd': oos_result.max_drawdown_pct,
                'oos_trades': oos_result.total_trades,
                'efficiency': efficiency,
            }

            windows.append(window_result)
            window_idx += 1
            start_idx += step_size

        if len(windows) == 0:
            log("No windows created - insufficient data")
            return None

        # Calculate overall metrics
        overall_is_return = np.mean([w['is_return_pct'] for w in windows])
        overall_oos_return = np.mean([w['oos_return_pct'] for w in windows])
        oos_efficiency = overall_oos_return / overall_is_return if overall_is_return != 0 else 0

        # Parameter stability (std of efficiency across windows)
        efficiencies = [w['efficiency'] for w in windows if w['efficiency'] is not None]
        parameter_stability = 1 - np.std(efficiencies) if len(efficiencies) > 1 else 1.0

        # Determine status
        if oos_efficiency >= 0.7 and parameter_stability >= 0.7:
            status = "HEALTHY"
        elif oos_efficiency >= 0.5 or parameter_stability >= 0.5:
            status = "WARNING"
        else:
            status = "CRITICAL"

        result = WalkForwardResult(
            strategy_name=strategy_name,
            entry_filter=entry_filter_name,
            windows=windows,
            overall_is_return=overall_is_return,
            overall_oos_return=overall_oos_return,
            oos_efficiency=oos_efficiency,
            parameter_stability=parameter_stability,
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data_bars=total_bars,
        )

        # Save report
        self._save_report(result)

        return result

    def _save_report(self, result: WalkForwardResult):
        """Save walk-forward report to JSON."""
        filename = f"wf_report_{result.strategy_name}_{result.entry_filter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)

        log(f"Report saved: {filepath}")


def main():
    """Run walk-forward analysis for all strategies."""
    engine = WalkForwardEngine()

    # Load data first
    try:
        engine.load_data()
    except Exception as e:
        log(f"Error loading data: {e}")
        return

    # Strategies to evaluate
    strategies = [
        # From backtest_15min_new
        ("Baseline", "baseline"),
        ("MultiTP_30", "baseline"),
        ("Conservative", "baseline"),
        # From backtest_15min_new_streak_a
        ("Adaptive_Baseline", "baseline"),
        ("Adaptive_ProgPos_Only", "baseline"),
        ("Adaptive_Conservative", "baseline"),
    ]

    results = []
    for strategy_name, entry_filter in strategies:
        log(f"\n{'='*60}")
        log(f"Evaluating: {strategy_name}")
        log(f"{'='*60}")

        result = engine.run_walk_forward(strategy_name, entry_filter)
        if result:
            results.append(result)
            log(f"  IS Return: {result.overall_is_return:.2f}%")
            log(f"  OOS Return: {result.overall_oos_return:.2f}%")
            log(f"  OOS Efficiency: {result.oos_efficiency:.2f}")
            log(f"  Parameter Stability: {result.parameter_stability:.2f}")
            log(f"  Status: {result.status}")

    # Summary
    log(f"\n{'='*60}")
    log("SUMMARY")
    log(f"{'='*60}")
    for r in results:
        log(f"{r.strategy_name}: {r.status} (OOS Eff: {r.oos_efficiency:.2f})")


if __name__ == "__main__":
    main()
