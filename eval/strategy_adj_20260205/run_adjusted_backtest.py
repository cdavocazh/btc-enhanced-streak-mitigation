#!/usr/bin/env python3
"""
Adjusted Strategy Backtest Runner - 2026-02-04
===============================================
Runs walk-forward backtests for adjusted strategy parameters.

Usage:
    python run_adjusted_backtest.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(EVAL_DIR)
sys.path.insert(0, EVAL_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'backtest_15min'))
sys.path.insert(0, os.path.join(REPO_DIR, 'backtest_15min_new'))

from adjusted_config import (
    BASELINE_PARAMS, ADJUSTMENT_EXPERIMENTS, STRATEGY_VARIANTS,
    WF_CONFIG, ASIAN_HOURS, ATR_PERIOD, RSI_PERIOD, SMA_PERIOD, ADX_PERIOD
)


def log(msg: str):
    """Print timestamped log message."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}")


@dataclass
class ExperimentResult:
    """Result from a single experiment."""
    experiment_name: str
    strategy_name: str
    is_return_pct: float
    oos_return_pct: float
    oos_efficiency: float
    total_trades: int
    win_rate_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    windows_evaluated: int
    status: str
    parameters: Dict[str, Any]


class AdjustedBacktestRunner:
    """Runs backtests with adjusted parameters."""

    def __init__(self):
        self.results_dir = SCRIPT_DIR
        self.price_data = None
        self.results: List[ExperimentResult] = []

        # Risk tiers
        self.risk_tiers = [
            (0, 150000, 5000),
            (150000, 225000, 10500),
            (225000, 337500, 14625),
            (337500, 507000, 20250),
            (507000, 760000, 28000),
            (760000, 1200000, 38000),
            (1200000, float('inf'), 54000),
        ]

        # Streak rules
        self.streak_rules = {
            3: 0.40,
            6: 0.30,
            9: 0.30,
        }

    def load_data(self) -> pd.DataFrame:
        """Load price and positioning data."""
        log("Loading data...")
        try:
            from load_15min_data import merge_all_data_15min
            df = merge_all_data_15min()
            log(f"Loaded {len(df)} rows")
        except ImportError:
            log("Could not import load_15min_data, using fallback...")
            df = self._load_fallback_data()

        self.price_data = df
        return df

    def _load_fallback_data(self) -> pd.DataFrame:
        """Fallback data loading from CSV."""
        data_dir = os.path.join(REPO_DIR, 'binance-futures-data', 'data')
        price_file = os.path.join(data_dir, 'price.csv')

        if not os.path.exists(price_file):
            raise FileNotFoundError(f"Price file not found: {price_file}")

        df = pd.read_csv(price_file)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        return df

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators."""
        df = df.copy()

        bb_period = 80
        volume_ma_period = 96

        # SMA and Bollinger
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

        # RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
        loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
        df['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        # Trend and pullback
        df['uptrend'] = df['close'] > df['sma50']
        df['pullback_pct'] = (df['high'].rolling(16).max() - df['close']) / df['close'] * 100

        # Volume
        if 'volume' in df.columns:
            df['vol_ma'] = df['volume'].rolling(volume_ma_period).mean()
            df['vol_ratio'] = df['volume'] / df['vol_ma'].replace(0, np.nan)
        else:
            df['vol_ratio'] = 1.0

        # ADX
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

        return df

    def calculate_positioning_score(self, row) -> float:
        """Calculate positioning score."""
        score = 0.0

        top_long = row.get('top_trader_position_long_pct', None)
        top_short = row.get('top_trader_position_short_pct', None)

        if top_long is not None and not pd.isna(top_long):
            if top_long > 0.60:
                score += 1.5
            elif top_long > 0.55:
                score += 1.0

        if top_short is not None and not pd.isna(top_short):
            if top_short > 0.60:
                score -= 1.5
            elif top_short > 0.55:
                score -= 1.0

        acct_long = row.get('top_trader_account_long_pct', None)
        acct_short = row.get('top_trader_account_short_pct', None)

        if acct_long is not None and not pd.isna(acct_long):
            if acct_long > 0.55:
                score += 0.25
        if acct_short is not None and not pd.isna(acct_short):
            if acct_short > 0.55:
                score -= 0.25

        global_ls = row.get('global_ls_ratio', None)
        if global_ls is not None and not pd.isna(global_ls):
            if global_ls < 0.7:
                score += 0.5
            elif global_ls > 1.5:
                score -= 0.5

        return score

    def get_risk_for_equity(self, equity: float) -> float:
        """Get risk amount for current equity."""
        for min_eq, max_eq, risk in self.risk_tiers:
            if min_eq <= equity < max_eq:
                return risk
        return self.risk_tiers[-1][2]

    def get_streak_adjusted_risk(self, base_risk: float, consecutive_losses: int) -> float:
        """Apply streak mitigation."""
        if consecutive_losses < 3:
            return base_risk

        reduction = 1.0
        for threshold, red_pct in sorted(self.streak_rules.items()):
            if consecutive_losses >= threshold:
                reduction *= (1 - red_pct)

        return base_risk * reduction

    def run_backtest_window(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        is_adaptive: bool = False,
        initial_capital: float = 100000
    ) -> Dict[str, Any]:
        """Run backtest on a data window."""
        if 'atr' not in df.columns:
            df = self.compute_indicators(df)

        trades = []
        equity = initial_capital
        pnl_history = []
        open_trades = []
        consecutive_losses = 0
        max_losing_streak = 0

        # Extract parameters
        min_pos_score = params.get('min_pos_score', 0.15)
        rsi_range = params.get('rsi_long_range', (20, 45))
        pullback_range = params.get('pullback_range', (0.5, 3.0))
        stop_atr_mult = params.get('stop_atr_mult', 1.8)
        tp_atr_mult = params.get('tp_atr_mult', 4.5)
        adx_threshold = params.get('adx_threshold', 20)
        progressive_pos = params.get('progressive_pos', {})
        cooldown_bars = params.get('cooldown_bars', 96)
        min_pos_long = params.get('min_pos_long', 0.4)

        bars_since_loss = 999

        for i, row in df.iterrows():
            if pd.isna(row.get('atr')) or pd.isna(row.get('sma50')):
                continue

            hr = i.hour if hasattr(i, 'hour') else 0
            positioning_score = self.calculate_positioning_score(row)

            base_risk = self.get_risk_for_equity(equity)
            adjusted_risk = self.get_streak_adjusted_risk(base_risk, consecutive_losses)

            # Check exits
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
                        bars_since_loss = 0
                    else:
                        consecutive_losses = 0

                    trades_to_close.append(idx)

            for idx in sorted(trades_to_close, reverse=True):
                open_trades.pop(idx)

            bars_since_loss += 1

            # Entry logic
            hour_ok = hr in ASIAN_HOURS
            if not hour_ok or len(open_trades) > 0:
                continue

            if abs(positioning_score) < min_pos_score:
                continue

            # ADX filter for adaptive strategies
            if is_adaptive:
                adx_value = row.get('adx', 0)
                if pd.isna(adx_value) or adx_value < adx_threshold:
                    continue

                # Cooldown after losses
                if consecutive_losses >= 5 and bars_since_loss < cooldown_bars:
                    continue

                # Progressive positioning
                if progressive_pos:
                    pos_threshold = progressive_pos.get(min(consecutive_losses, 4), min_pos_long)
                else:
                    pos_threshold = min_pos_long
            else:
                pos_threshold = min_pos_long

            pullback = row.get('pullback_pct', 0)
            uptrend = row.get('uptrend', False)
            rsi = row.get('rsi', 50)

            if pd.isna(rsi) or pd.isna(pullback):
                continue

            entry_signal = False
            if positioning_score >= pos_threshold:
                if uptrend and pullback_range[0] < pullback < pullback_range[1]:
                    if rsi_range[0] < rsi < rsi_range[1]:
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

        # Sharpe ratio
        if len(pnl_history) > 1:
            returns = pd.Series(pnl_history)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        return {
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': sharpe_ratio,
            'win_rate_pct': win_rate_pct,
            'total_trades': total_trades,
            'max_losing_streak': max_losing_streak,
        }

    def run_walk_forward(
        self,
        strategy_name: str,
        experiment_name: str,
        params: Dict[str, Any]
    ) -> ExperimentResult:
        """Run walk-forward optimization for a strategy with adjusted params."""
        log(f"Running WFO: {strategy_name} / {experiment_name}")

        is_adaptive = "Adaptive" in strategy_name

        if self.price_data is None:
            self.load_data()

        df = self.compute_indicators(self.price_data.copy())

        # Window config
        bars_per_day = 96
        training_bars = WF_CONFIG['training_window_days'] * bars_per_day
        testing_bars = WF_CONFIG['testing_window_days'] * bars_per_day
        step_size = WF_CONFIG['step_size_days'] * bars_per_day

        total_bars = len(df)
        windows = []
        start_idx = 0

        while start_idx + training_bars + testing_bars <= total_bars:
            train_start = start_idx
            train_end = start_idx + training_bars
            test_start = train_end
            test_end = min(train_end + testing_bars, total_bars)

            train_df = df.iloc[train_start:train_end]
            test_df = df.iloc[test_start:test_end]

            # Run IS backtest
            is_result = self.run_backtest_window(train_df, params, is_adaptive)

            # Run OOS backtest
            oos_result = self.run_backtest_window(test_df, params, is_adaptive)

            # Calculate efficiency
            efficiency = (oos_result['total_return_pct'] / is_result['total_return_pct']
                         if is_result['total_return_pct'] != 0 else 0)

            windows.append({
                'is_return': is_result['total_return_pct'],
                'oos_return': oos_result['total_return_pct'],
                'efficiency': efficiency,
                'is_trades': is_result['total_trades'],
                'oos_trades': oos_result['total_trades'],
                'oos_sharpe': oos_result['sharpe_ratio'],
                'oos_max_dd': oos_result['max_drawdown_pct'],
                'oos_win_rate': oos_result['win_rate_pct'],
            })

            start_idx += step_size

        if not windows:
            return ExperimentResult(
                experiment_name=experiment_name,
                strategy_name=strategy_name,
                is_return_pct=0,
                oos_return_pct=0,
                oos_efficiency=0,
                total_trades=0,
                win_rate_pct=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                windows_evaluated=0,
                status="NO_DATA",
                parameters=params,
            )

        # Aggregate results
        overall_is = np.mean([w['is_return'] for w in windows])
        overall_oos = np.mean([w['oos_return'] for w in windows])
        oos_efficiency = overall_oos / overall_is if overall_is != 0 else 0
        total_trades = sum([w['oos_trades'] for w in windows])
        avg_sharpe = np.mean([w['oos_sharpe'] for w in windows])
        avg_max_dd = np.mean([w['oos_max_dd'] for w in windows])
        avg_win_rate = np.mean([w['oos_win_rate'] for w in windows])

        # Determine status
        if oos_efficiency >= 0.7:
            status = "HEALTHY"
        elif oos_efficiency >= 0.5:
            status = "WARNING"
        else:
            status = "CRITICAL"

        result = ExperimentResult(
            experiment_name=experiment_name,
            strategy_name=strategy_name,
            is_return_pct=overall_is,
            oos_return_pct=overall_oos,
            oos_efficiency=oos_efficiency,
            total_trades=total_trades,
            win_rate_pct=avg_win_rate,
            max_drawdown_pct=avg_max_dd,
            sharpe_ratio=avg_sharpe,
            windows_evaluated=len(windows),
            status=status,
            parameters=params,
        )

        self.results.append(result)
        return result

    def run_all_experiments(self):
        """Run all adjustment experiments for all strategies."""
        log("=" * 60)
        log("ADJUSTED STRATEGY BACKTEST - 2026-02-04")
        log("=" * 60)

        # Load data once
        self.load_data()

        # Run baseline for comparison
        log("\n--- BASELINE STRATEGIES ---")
        for strategy in STRATEGY_VARIANTS:
            params = BASELINE_PARAMS.get(strategy, {})
            self.run_walk_forward(strategy, "Baseline", params)

        # Run each experiment
        for experiment in ADJUSTMENT_EXPERIMENTS:
            log(f"\n--- EXPERIMENT: {experiment['name']} ---")
            log(f"Description: {experiment['description']}")

            for strategy in STRATEGY_VARIANTS:
                # Merge baseline params with experiment changes
                base_params = BASELINE_PARAMS.get(strategy, {}).copy()
                for key, value in experiment['changes'].items():
                    base_params[key] = value

                self.run_walk_forward(strategy, experiment['name'], base_params)

        log("\n" + "=" * 60)
        log("ALL EXPERIMENTS COMPLETE")
        log("=" * 60)

    def save_results(self):
        """Save results to JSON."""
        results_file = os.path.join(self.results_dir, 'experiment_results.json')

        data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'wf_config': WF_CONFIG,
            'experiments': [asdict(r) for r in self.results],
        }

        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        log(f"Results saved: {results_file}")

    def print_summary(self):
        """Print summary of results."""
        log("\n" + "=" * 80)
        log("EXPERIMENT SUMMARY")
        log("=" * 80)

        # Group by experiment
        experiments = {}
        for r in self.results:
            if r.experiment_name not in experiments:
                experiments[r.experiment_name] = []
            experiments[r.experiment_name].append(r)

        print(f"\n{'Experiment':<20} {'Strategy':<25} {'IS Ret%':>8} {'OOS Ret%':>9} {'OOS Eff':>8} {'Trades':>7} {'Status':<10}")
        print("-" * 95)

        for exp_name, results in experiments.items():
            for r in results:
                print(f"{exp_name:<20} {r.strategy_name:<25} {r.is_return_pct:>8.2f} {r.oos_return_pct:>9.2f} {r.oos_efficiency:>8.2f} {r.total_trades:>7} {r.status:<10}")
            print("-" * 95)


def main():
    runner = AdjustedBacktestRunner()
    runner.run_all_experiments()
    runner.save_results()
    runner.print_summary()


if __name__ == "__main__":
    main()
