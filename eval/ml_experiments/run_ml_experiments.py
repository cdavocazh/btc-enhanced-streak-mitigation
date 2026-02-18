#!/usr/bin/env python3
"""
ML Experiments for BTC Strategy Optimization
=============================================

Implements 4 machine learning approaches to improve the BTC trading strategy:

1. XGBoost Trade Quality Classifier
   - Predicts whether a trade setup will be profitable
   - Uses positioning, volume, ADX, RSI features
   - Walk-forward trained to avoid look-ahead bias

2. Random Forest Regime Detector
   - Classifies market into regimes (trending/ranging/volatile)
   - Adjusts strategy parameters per regime
   - Replaces static ADX threshold

3. Gradient Boosted Streak Predictor
   - Predicts probability of next trade being a loss
   - Dynamic position sizing based on predicted loss probability
   - Replaces static streak mitigation rules

4. LSTM-style Sequential Feature Extraction (via PyTorch)
   - Learns temporal patterns in positioning data
   - Generates learned features fed into XGBoost
   - Captures time-dependent signal quality

All experiments use walk-forward validation to measure OOS performance.
"""

import os

# Fix OpenMP conflict between PyTorch and XGBoost/LightGBM on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
import joblib

# PyTorch for sequential model
import torch
import torch.nn as nn

# Add parent directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(EVAL_DIR)

# The backtest_15min folder may be in the main repo (not in worktree)
# Try worktree first, then main repo
MAIN_REPO_DIR = os.path.join(os.path.expanduser('~'), 'Github', 'btc-enhanced-streak-mitigation')
for data_dir in [REPO_DIR, MAIN_REPO_DIR]:
    backtest_15min_path = os.path.join(data_dir, 'backtest_15min')
    if os.path.exists(backtest_15min_path):
        sys.path.insert(0, backtest_15min_path)
        break

sys.path.insert(0, os.path.join(REPO_DIR, 'backtest_15min_new'))
sys.path.insert(0, os.path.join(REPO_DIR, 'backtest_15min_new_streak_a'))
sys.path.insert(0, EVAL_DIR)

from config import (
    RISK_TIERS, STREAK_RULES, ASIAN_HOURS,
    ATR_PERIOD, RSI_PERIOD, SMA_PERIOD, ADX_PERIOD,
    TOP_TRADER_STRONG, TOP_TRADER_MODERATE,
)


def log(msg: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# DATA LOADING AND FEATURE ENGINEERING
# ============================================================

def load_and_prepare_data() -> pd.DataFrame:
    """Load price + positioning data and compute all features."""
    try:
        from load_15min_data import merge_all_data_15min
        df = merge_all_data_15min()
        log(f"Loaded {len(df)} rows from merge_all_data_15min()")
    except ImportError:
        log("Could not import load_15min_data, loading from CSV...")
        data_dir = os.path.join(REPO_DIR, 'binance-futures-data', 'data')
        df = pd.read_csv(os.path.join(data_dir, 'price.csv'))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    df = compute_all_features(df)
    return df


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and ML features."""
    df = df.copy()

    bb_period = 80
    volume_ma_period = 96

    # Price indicators
    df['sma20'] = df['close'].rolling(bb_period).mean()
    df['sma50'] = df['close'].rolling(SMA_PERIOD).mean()
    df['std20'] = df['close'].rolling(bb_period).std()
    df['upper_band'] = df['sma20'] + 2 * df['std20']
    df['lower_band'] = df['sma20'] - 2 * df['std20']

    # BB position (where price is relative to bands)
    bb_width = df['upper_band'] - df['lower_band']
    df['bb_position'] = (df['close'] - df['lower_band']) / bb_width.replace(0, np.nan)

    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(ATR_PERIOD).mean()
    df['atr_short'] = tr.rolling(5).mean()
    df['atr_adaptive'] = df[['atr', 'atr_short']].min(axis=1)
    df['atr_ratio'] = df['atr_short'] / df['atr'].replace(0, np.nan)

    # RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    df['rsi'] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # RSI momentum
    df['rsi_change_4h'] = df['rsi'] - df['rsi'].shift(16)
    df['rsi_change_12h'] = df['rsi'] - df['rsi'].shift(48)

    # Trend
    df['uptrend'] = (df['close'] > df['sma50']).astype(int)
    df['pullback_pct'] = (df['high'].rolling(16).max() - df['close']) / df['close'] * 100

    # Price momentum features
    df['price_change_1h'] = df['close'].pct_change(4) * 100
    df['price_change_4h'] = df['close'].pct_change(16) * 100
    df['price_change_12h'] = df['close'].pct_change(48) * 100
    df['price_change_24h'] = df['close'].pct_change(96) * 100

    # Volatility features
    df['realized_vol_1h'] = df['close'].pct_change().rolling(4).std() * np.sqrt(96 * 365) * 100
    df['realized_vol_4h'] = df['close'].pct_change().rolling(16).std() * np.sqrt(96 * 365) * 100
    df['realized_vol_24h'] = df['close'].pct_change().rolling(96).std() * np.sqrt(96 * 365) * 100
    df['vol_regime'] = df['realized_vol_4h'] / df['realized_vol_24h'].replace(0, np.nan)

    # Volume
    if 'volume' in df.columns:
        df['vol_ma'] = df['volume'].rolling(volume_ma_period).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma'].replace(0, np.nan)
        df['vol_ma_4h'] = df['volume'].rolling(16).mean()
        df['vol_trend'] = df['vol_ma_4h'] - df['vol_ma_4h'].shift(4)
        df['vol_increasing'] = (df['vol_trend'] > 0).astype(int)
        df['price_change'] = df['close'].pct_change()
        df['bullish_volume'] = ((df['price_change'] > 0) & (df['vol_ratio'] > 1.0)).astype(int)
        df['vol_change_4h'] = df['vol_ratio'] - df['vol_ratio'].shift(16)
    else:
        df['vol_ratio'] = 1.0
        df['vol_increasing'] = 1
        df['bullish_volume'] = 1
        df['vol_change_4h'] = 0.0

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
    df['adx_change_4h'] = df['adx'] - df['adx'].shift(16)

    # DI spread (trend direction strength)
    df['di_spread'] = df['plus_di'] - df['minus_di']

    # Higher Highs / Higher Lows
    lookback = 20
    df['swing_high_1'] = df['high'].rolling(lookback).max()
    df['swing_high_2'] = df['high'].rolling(lookback).max().shift(lookback)
    df['swing_low_1'] = df['low'].rolling(lookback).min()
    df['swing_low_2'] = df['low'].rolling(lookback).min().shift(lookback)
    df['making_hh'] = (df['swing_high_1'] > df['swing_high_2']).astype(int)
    df['making_hl'] = (df['swing_low_1'] > df['swing_low_2']).astype(int)
    df['trend_confirmed'] = (df['making_hh'] & df['making_hl']).astype(int)

    # Positioning features (vectorized)
    df['positioning_score'] = calculate_positioning_score_vectorized(df)
    df['pos_score_4h_ago'] = df['positioning_score'].shift(16)
    df['pos_score_12h_ago'] = df['positioning_score'].shift(48)
    df['pos_momentum_4h'] = df['positioning_score'] - df['pos_score_4h_ago']
    df['pos_momentum_12h'] = df['positioning_score'] - df['pos_score_12h_ago']

    # Volume score (vectorized)
    df['volume_score'] = calculate_volume_score_vectorized(df)

    # Hour features
    if hasattr(df.index, 'hour'):
        df['hour'] = df.index.hour
        df['is_asian'] = df['hour'].isin(ASIAN_HOURS).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    else:
        df['hour'] = 0
        df['is_asian'] = 1
        df['hour_sin'] = 0
        df['hour_cos'] = 1

    # Day of week features
    if hasattr(df.index, 'dayofweek'):
        df['day_of_week'] = df.index.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    else:
        df['day_of_week'] = 0
        df['dow_sin'] = 0
        df['dow_cos'] = 1

    return df


def calculate_positioning_score_vectorized(df: pd.DataFrame) -> pd.Series:
    """Vectorized positioning score calculation."""
    score = pd.Series(0.0, index=df.index)

    if 'top_trader_position_long_pct' in df.columns:
        tl = df['top_trader_position_long_pct']
        score += np.where(tl > TOP_TRADER_STRONG, 1.5,
                 np.where(tl > TOP_TRADER_MODERATE, 1.0, 0.0))
    if 'top_trader_position_short_pct' in df.columns:
        ts = df['top_trader_position_short_pct']
        score -= np.where(ts > TOP_TRADER_STRONG, 1.5,
                 np.where(ts > TOP_TRADER_MODERATE, 1.0, 0.0))
    if 'top_trader_account_long_pct' in df.columns:
        al = df['top_trader_account_long_pct']
        score += np.where(al > TOP_TRADER_MODERATE, 0.25, 0.0)
    if 'top_trader_account_short_pct' in df.columns:
        ash = df['top_trader_account_short_pct']
        score -= np.where(ash > TOP_TRADER_MODERATE, 0.25, 0.0)
    if 'global_ls_ratio' in df.columns:
        gl = df['global_ls_ratio']
        score += np.where(gl < 0.7, 0.5, np.where(gl > 1.5, -0.5, 0.0))
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        score += np.where(fr > 0.0005, -0.5, np.where(fr < -0.0005, 0.5, 0.0))

    return score


def calculate_volume_score_vectorized(df: pd.DataFrame) -> pd.Series:
    """Vectorized volume score calculation."""
    score = pd.Series(0.0, index=df.index)

    if 'vol_ratio' in df.columns:
        vr = df['vol_ratio'].fillna(1.0)
        score += np.where(vr > 1.5, 1.0,
                 np.where(vr > 1.2, 0.7,
                 np.where(vr > 1.0, 0.5,
                 np.where(vr > 0.8, 0.3, 0.0))))
    if 'vol_increasing' in df.columns:
        score += df['vol_increasing'].fillna(0) * 0.5
    if 'bullish_volume' in df.columns:
        score += df['bullish_volume'].fillna(0) * 0.5

    return score.clip(upper=2.0)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURE_COLUMNS = [
    'positioning_score', 'volume_score',
    'pos_momentum_4h', 'pos_momentum_12h',
    'adx', 'adx_change_4h', 'di_spread',
    'rsi', 'rsi_change_4h', 'rsi_change_12h',
    'atr_ratio', 'bb_position',
    'vol_ratio', 'vol_increasing', 'vol_change_4h',
    'price_change_1h', 'price_change_4h', 'price_change_12h',
    'realized_vol_4h', 'vol_regime',
    'pullback_pct', 'uptrend', 'trend_confirmed',
    'making_hh', 'making_hl',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'is_asian',
]


def get_feature_matrix(df: pd.DataFrame, feature_cols: List[str] = None) -> pd.DataFrame:
    """Extract feature matrix from DataFrame, dropping NaN rows."""
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS
    available = [c for c in feature_cols if c in df.columns]
    return df[available].dropna()


# ============================================================
# LABELING: Generate trade outcomes for supervised learning
# ============================================================

def generate_trade_labels(df: pd.DataFrame, stop_mult: float = 1.8,
                          tp_mult: float = 4.5) -> pd.DataFrame:
    """
    For each bar, simulate a hypothetical long entry and determine outcome.
    Label = 1 if TP hit before SL, 0 otherwise.
    Uses numpy for fast vectorized lookforward.
    """
    df = df.copy()

    close = df['close'].values.astype(np.float64)
    high_vals = df['high'].values.astype(np.float64)
    low_vals = df['low'].values.astype(np.float64)
    atr_vals = df['atr'].values.astype(np.float64)

    n = len(df)
    max_bars = 96

    labels = np.full(n, np.nan)
    pnl_ratios = np.full(n, np.nan)
    bars_to_exit = np.full(n, np.nan)

    # Process in chunks for efficiency
    valid_mask = ~np.isnan(atr_vals) & (atr_vals > 0)
    valid_indices = np.where(valid_mask)[0]

    for i in valid_indices:
        entry = close[i]
        sl = entry - atr_vals[i] * stop_mult
        tp = entry + atr_vals[i] * tp_mult

        end = min(i + max_bars + 1, n)
        if i + 1 >= end:
            continue

        future_low = low_vals[i+1:end]
        future_high = high_vals[i+1:end]

        sl_hit = np.where(future_low <= sl)[0]
        tp_hit = np.where(future_high >= tp)[0]

        sl_bar = sl_hit[0] if len(sl_hit) > 0 else max_bars + 1
        tp_bar = tp_hit[0] if len(tp_hit) > 0 else max_bars + 1

        if tp_bar <= sl_bar and tp_bar < max_bars + 1:
            labels[i] = 1
            pnl_ratios[i] = tp_mult / stop_mult
            bars_to_exit[i] = tp_bar + 1
        elif sl_bar < tp_bar and sl_bar < max_bars + 1:
            labels[i] = 0
            pnl_ratios[i] = -1.0
            bars_to_exit[i] = sl_bar + 1
        else:
            final_bar = min(i + max_bars, n - 1)
            final_pnl = (close[final_bar] - entry) / (atr_vals[i] * stop_mult)
            labels[i] = 1 if final_pnl > 0 else 0
            pnl_ratios[i] = final_pnl
            bars_to_exit[i] = max_bars

    df['trade_label'] = labels
    df['trade_pnl_ratio'] = pnl_ratios
    df['bars_to_exit'] = bars_to_exit

    return df


# ============================================================
# APPROACH 1: XGBoost Trade Quality Classifier
# ============================================================

@dataclass
class XGBExperimentResult:
    name: str = "XGBoost_TradeQuality"
    description: str = "XGBoost classifier predicting trade win/loss"
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0
    test_precision: float = 0.0
    test_recall: float = 0.0
    test_f1: float = 0.0
    oos_efficiency: float = 0.0
    filtered_win_rate: float = 0.0
    baseline_win_rate: float = 0.0
    trades_filtered_pct: float = 0.0
    improvement_vs_baseline: float = 0.0
    feature_importance: Dict[str, float] = None
    windows_evaluated: int = 0
    total_predictions: int = 0
    status: str = "PENDING"


def run_xgboost_experiment(df: pd.DataFrame) -> XGBExperimentResult:
    """
    Walk-forward XGBoost trade quality classifier.
    Train on past data, predict on future windows.
    """
    log("=" * 60)
    log("APPROACH 1: XGBoost Trade Quality Classifier")
    log("=" * 60)

    result = XGBExperimentResult()

    # Filter to Asian hours
    if 'hour' in df.columns:
        df_filtered = df[df['hour'].isin(ASIAN_HOURS)].copy()
    else:
        df_filtered = df.copy()

    # Get features and labels
    feature_df = get_feature_matrix(df_filtered)
    label_mask = df_filtered.loc[feature_df.index, 'trade_label'].notna()
    valid_idx = feature_df.index[label_mask.loc[feature_df.index]]

    X = feature_df.loc[valid_idx].values
    y = df_filtered.loc[valid_idx, 'trade_label'].values

    log(f"Total samples: {len(X)}, Win rate: {y.mean():.3f}")

    if len(X) < 200:
        log("Insufficient data for XGBoost experiment")
        result.status = "INSUFFICIENT_DATA"
        return result

    # Walk-forward: 70% train, 30% test in rolling windows
    train_size = int(len(X) * 0.7)
    step = max(96, len(X) // 20)

    all_predictions = []
    all_actuals = []
    all_probabilities = []
    feature_importances = np.zeros(X.shape[1])
    n_windows = 0

    scaler = StandardScaler()

    for start in range(train_size, len(X) - step, step):
        train_X = X[:start]
        train_y = y[:start]
        test_X = X[start:start + step]
        test_y = y[start:start + step]

        if len(test_X) == 0:
            continue

        # Scale
        train_X_s = scaler.fit_transform(train_X)
        test_X_s = scaler.transform(test_X)

        # Train XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
            eval_metric='logloss',
        )

        model.fit(train_X_s, train_y, verbose=False)

        # Predict
        probs = model.predict_proba(test_X_s)[:, 1]
        preds = (probs > 0.5).astype(int)

        all_predictions.extend(preds)
        all_actuals.extend(test_y)
        all_probabilities.extend(probs)
        feature_importances += model.feature_importances_
        n_windows += 1

    if n_windows == 0:
        result.status = "NO_WINDOWS"
        return result

    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    all_probabilities = np.array(all_probabilities)
    feature_importances /= n_windows

    # Metrics
    result.test_accuracy = accuracy_score(all_actuals, all_predictions)
    result.test_precision = precision_score(all_actuals, all_predictions, zero_division=0)
    result.test_recall = recall_score(all_actuals, all_predictions, zero_division=0)
    result.test_f1 = f1_score(all_actuals, all_predictions, zero_division=0)
    result.baseline_win_rate = all_actuals.mean() * 100
    result.total_predictions = len(all_predictions)
    result.windows_evaluated = n_windows

    # Calculate filtered win rate (only take trades where model says > 0.5)
    high_conf_mask = all_probabilities > 0.55
    if high_conf_mask.sum() > 0:
        result.filtered_win_rate = all_actuals[high_conf_mask].mean() * 100
        result.trades_filtered_pct = (1 - high_conf_mask.mean()) * 100
    else:
        result.filtered_win_rate = result.baseline_win_rate
        result.trades_filtered_pct = 100.0

    result.improvement_vs_baseline = result.filtered_win_rate - result.baseline_win_rate

    # OOS efficiency approximation
    if result.baseline_win_rate > 0:
        result.oos_efficiency = result.filtered_win_rate / max(result.baseline_win_rate, 1)
    else:
        result.oos_efficiency = 0

    # Feature importance
    feature_names = [c for c in FEATURE_COLUMNS if c in df_filtered.columns][:X.shape[1]]
    if len(feature_names) == len(feature_importances):
        result.feature_importance = dict(sorted(
            zip(feature_names, feature_importances.tolist()),
            key=lambda x: -x[1]
        )[:15])

    result.status = "HEALTHY" if result.improvement_vs_baseline > 3 else (
        "WARNING" if result.improvement_vs_baseline > 0 else "CRITICAL"
    )

    log(f"Baseline Win Rate: {result.baseline_win_rate:.1f}%")
    log(f"Filtered Win Rate: {result.filtered_win_rate:.1f}%")
    log(f"Improvement: {result.improvement_vs_baseline:+.1f}%")
    log(f"Trades Filtered: {result.trades_filtered_pct:.1f}%")
    log(f"Top Features: {list(result.feature_importance.keys())[:5]}")

    return result


# ============================================================
# APPROACH 2: Random Forest Regime Detector
# ============================================================

@dataclass
class RFRegimeResult:
    name: str = "RandomForest_RegimeDetector"
    description: str = "RF classifier for market regime detection"
    regime_accuracy: float = 0.0
    regime_distribution: Dict[str, float] = None
    win_rate_by_regime: Dict[str, float] = None
    best_regime: str = ""
    best_regime_win_rate: float = 0.0
    worst_regime: str = ""
    worst_regime_win_rate: float = 0.0
    regime_filtered_win_rate: float = 0.0
    baseline_win_rate: float = 0.0
    improvement_vs_baseline: float = 0.0
    oos_efficiency: float = 0.0
    feature_importance: Dict[str, float] = None
    windows_evaluated: int = 0
    status: str = "PENDING"


def classify_regime(row) -> int:
    """Classify market regime from features (for labeling)."""
    adx = row.get('adx', 20)
    vol_regime = row.get('vol_regime', 1.0)

    if pd.isna(adx):
        adx = 20
    if pd.isna(vol_regime):
        vol_regime = 1.0

    if adx > 30 and vol_regime < 1.2:
        return 0  # Strong trend, normal vol
    elif adx > 20 and vol_regime < 1.5:
        return 1  # Moderate trend
    elif adx <= 20 and vol_regime > 1.3:
        return 2  # Ranging, high vol
    elif adx <= 20:
        return 3  # Ranging, low vol
    else:
        return 4  # Volatile trending


def run_rf_regime_experiment(df: pd.DataFrame) -> RFRegimeResult:
    """
    Walk-forward Random Forest regime detector.
    Classifies market regimes and measures trade outcomes per regime.
    """
    log("=" * 60)
    log("APPROACH 2: Random Forest Regime Detector")
    log("=" * 60)

    result = RFRegimeResult()

    df_work = df.copy()
    df_work['regime_label'] = df_work.apply(classify_regime, axis=1)

    # Asian hours filter
    if 'hour' in df_work.columns:
        df_work = df_work[df_work['hour'].isin(ASIAN_HOURS)].copy()

    feature_df = get_feature_matrix(df_work)
    valid_mask = df_work.loc[feature_df.index, 'trade_label'].notna()
    valid_idx = feature_df.index[valid_mask.loc[feature_df.index]]

    X = feature_df.loc[valid_idx].values
    y_regime = df_work.loc[valid_idx, 'regime_label'].values
    y_trade = df_work.loc[valid_idx, 'trade_label'].values

    log(f"Total samples: {len(X)}")

    if len(X) < 200:
        result.status = "INSUFFICIENT_DATA"
        return result

    # Walk-forward
    train_size = int(len(X) * 0.7)
    step = max(96, len(X) // 20)

    all_regime_preds = []
    all_regime_actuals = []
    all_trade_outcomes = []
    feature_importances = np.zeros(X.shape[1])
    n_windows = 0

    scaler = StandardScaler()

    for start in range(train_size, len(X) - step, step):
        train_X = X[:start]
        train_y = y_regime[:start]
        test_X = X[start:start + step]
        test_y_regime = y_regime[start:start + step]
        test_y_trade = y_trade[start:start + step]

        if len(test_X) == 0:
            continue

        train_X_s = scaler.fit_transform(train_X)
        test_X_s = scaler.transform(test_X)

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
        )
        model.fit(train_X_s, train_y)

        preds = model.predict(test_X_s)
        all_regime_preds.extend(preds)
        all_regime_actuals.extend(test_y_regime)
        all_trade_outcomes.extend(test_y_trade)
        feature_importances += model.feature_importances_
        n_windows += 1

    if n_windows == 0:
        result.status = "NO_WINDOWS"
        return result

    all_regime_preds = np.array(all_regime_preds)
    all_regime_actuals = np.array(all_regime_actuals)
    all_trade_outcomes = np.array(all_trade_outcomes)
    feature_importances /= n_windows

    result.regime_accuracy = accuracy_score(all_regime_actuals, all_regime_preds)
    result.baseline_win_rate = all_trade_outcomes.mean() * 100
    result.windows_evaluated = n_windows

    # Win rate by predicted regime
    regime_names = {0: 'StrongTrend', 1: 'ModerateTrend', 2: 'RangingHighVol',
                    3: 'RangingLowVol', 4: 'VolatileTrend'}

    win_rates = {}
    regime_dist = {}
    for regime_id, regime_name in regime_names.items():
        mask = all_regime_preds == regime_id
        regime_dist[regime_name] = mask.mean() * 100
        if mask.sum() > 5:
            win_rates[regime_name] = all_trade_outcomes[mask].mean() * 100
        else:
            win_rates[regime_name] = 0.0

    result.win_rate_by_regime = win_rates
    result.regime_distribution = regime_dist

    if win_rates:
        valid_regimes = {k: v for k, v in win_rates.items() if v > 0}
        if valid_regimes:
            result.best_regime = max(valid_regimes, key=valid_regimes.get)
            result.best_regime_win_rate = valid_regimes[result.best_regime]
            result.worst_regime = min(valid_regimes, key=valid_regimes.get)
            result.worst_regime_win_rate = valid_regimes[result.worst_regime]

    # Filter: only trade in favorable regimes (win rate > baseline)
    favorable_regimes = [k for k, v in regime_names.items()
                         if win_rates.get(regime_names[k], 0) > result.baseline_win_rate]
    if favorable_regimes:
        favorable_mask = np.isin(all_regime_preds, favorable_regimes)
        if favorable_mask.sum() > 0:
            result.regime_filtered_win_rate = all_trade_outcomes[favorable_mask].mean() * 100
        else:
            result.regime_filtered_win_rate = result.baseline_win_rate
    else:
        result.regime_filtered_win_rate = result.baseline_win_rate

    result.improvement_vs_baseline = result.regime_filtered_win_rate - result.baseline_win_rate

    if result.baseline_win_rate > 0:
        result.oos_efficiency = result.regime_filtered_win_rate / max(result.baseline_win_rate, 1)

    # Feature importance
    feature_names = [c for c in FEATURE_COLUMNS if c in df.columns][:X.shape[1]]
    if len(feature_names) == len(feature_importances):
        result.feature_importance = dict(sorted(
            zip(feature_names, feature_importances.tolist()),
            key=lambda x: -x[1]
        )[:15])

    result.status = "HEALTHY" if result.improvement_vs_baseline > 3 else (
        "WARNING" if result.improvement_vs_baseline > 0 else "CRITICAL"
    )

    log(f"Regime Accuracy: {result.regime_accuracy:.1f}%")
    log(f"Baseline Win Rate: {result.baseline_win_rate:.1f}%")
    log(f"Regime-Filtered Win Rate: {result.regime_filtered_win_rate:.1f}%")
    log(f"Best Regime: {result.best_regime} ({result.best_regime_win_rate:.1f}%)")
    log(f"Worst Regime: {result.worst_regime} ({result.worst_regime_win_rate:.1f}%)")

    return result


# ============================================================
# APPROACH 3: LightGBM Streak Predictor
# ============================================================

@dataclass
class LGBStreakResult:
    name: str = "LightGBM_StreakPredictor"
    description: str = "LightGBM predicting loss streaks for dynamic sizing"
    prediction_accuracy: float = 0.0
    streak_precision: float = 0.0
    streak_recall: float = 0.0
    dynamic_sizing_return: float = 0.0
    static_sizing_return: float = 0.0
    improvement_pct: float = 0.0
    max_dd_dynamic: float = 0.0
    max_dd_static: float = 0.0
    dd_improvement_pct: float = 0.0
    oos_efficiency: float = 0.0
    avg_position_scale: float = 0.0
    feature_importance: Dict[str, float] = None
    windows_evaluated: int = 0
    status: str = "PENDING"


def run_lgb_streak_experiment(df: pd.DataFrame) -> LGBStreakResult:
    """
    LightGBM model to predict loss probability for dynamic position sizing.
    Replace static streak mitigation with ML-predicted risk scaling.
    """
    log("=" * 60)
    log("APPROACH 3: LightGBM Streak Predictor")
    log("=" * 60)

    result = LGBStreakResult()

    df_work = df.copy()

    if 'hour' in df_work.columns:
        df_work = df_work[df_work['hour'].isin(ASIAN_HOURS)].copy()

    feature_df = get_feature_matrix(df_work)
    valid_mask = df_work.loc[feature_df.index, 'trade_label'].notna()
    valid_idx = feature_df.index[valid_mask.loc[feature_df.index]]

    X = feature_df.loc[valid_idx].values
    y = df_work.loc[valid_idx, 'trade_label'].values
    pnl_ratios = df_work.loc[valid_idx, 'trade_pnl_ratio'].values

    # Target: predict loss (inverse of trade_label)
    y_loss = 1 - y

    log(f"Total samples: {len(X)}, Loss rate: {y_loss.mean():.3f}")

    if len(X) < 200:
        result.status = "INSUFFICIENT_DATA"
        return result

    # Walk-forward
    train_size = int(len(X) * 0.7)
    step = max(96, len(X) // 20)

    all_loss_probs = []
    all_actuals = []
    all_pnl_ratios = []
    feature_importances = np.zeros(X.shape[1])
    n_windows = 0

    for start in range(train_size, len(X) - step, step):
        train_X = X[:start]
        train_y = y_loss[:start]
        test_X = X[start:start + step]
        test_y = y_loss[start:start + step]
        test_pnl = pnl_ratios[start:start + step]

        if len(test_X) == 0:
            continue

        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )

        model.fit(train_X, train_y)
        loss_probs = model.predict_proba(test_X)[:, 1]

        all_loss_probs.extend(loss_probs)
        all_actuals.extend(test_y)
        all_pnl_ratios.extend(test_pnl)
        feature_importances += model.feature_importances_
        n_windows += 1

    if n_windows == 0:
        result.status = "NO_WINDOWS"
        return result

    all_loss_probs = np.array(all_loss_probs)
    all_actuals = np.array(all_actuals)
    all_pnl_ratios = np.array(all_pnl_ratios)
    feature_importances /= n_windows

    # Accuracy metrics
    preds = (all_loss_probs > 0.5).astype(int)
    result.prediction_accuracy = accuracy_score(all_actuals, preds) * 100
    result.streak_precision = precision_score(all_actuals, preds, zero_division=0) * 100
    result.streak_recall = recall_score(all_actuals, preds, zero_division=0) * 100
    result.windows_evaluated = n_windows

    # Simulate dynamic position sizing
    # Scale position by (1 - loss_probability): high prob loss -> small position
    base_risk = 5000  # $5k base
    dynamic_equity = [100000.0]
    static_equity = [100000.0]

    for i in range(len(all_pnl_ratios)):
        loss_prob = all_loss_probs[i]
        pnl_ratio = all_pnl_ratios[i]

        # Dynamic: scale position inversely to loss probability
        scale = max(0.2, 1.0 - loss_prob * 0.8)
        dynamic_pnl = base_risk * scale * pnl_ratio
        static_pnl = base_risk * pnl_ratio

        dynamic_equity.append(dynamic_equity[-1] + dynamic_pnl)
        static_equity.append(static_equity[-1] + static_pnl)

    dynamic_equity = np.array(dynamic_equity)
    static_equity = np.array(static_equity)

    result.static_sizing_return = (static_equity[-1] - 100000) / 100000 * 100
    result.dynamic_sizing_return = (dynamic_equity[-1] - 100000) / 100000 * 100
    result.improvement_pct = result.dynamic_sizing_return - result.static_sizing_return

    # Max drawdown
    dynamic_peak = np.maximum.accumulate(dynamic_equity)
    static_peak = np.maximum.accumulate(static_equity)
    result.max_dd_dynamic = ((dynamic_peak - dynamic_equity) / dynamic_peak * 100).max()
    result.max_dd_static = ((static_peak - static_equity) / static_peak * 100).max()
    result.dd_improvement_pct = result.max_dd_static - result.max_dd_dynamic

    # Average position scale
    result.avg_position_scale = np.mean([max(0.2, 1.0 - p * 0.8) for p in all_loss_probs])

    if result.static_sizing_return != 0:
        result.oos_efficiency = result.dynamic_sizing_return / abs(result.static_sizing_return)

    # Feature importance
    feature_names = [c for c in FEATURE_COLUMNS if c in df.columns][:X.shape[1]]
    if len(feature_names) == len(feature_importances):
        result.feature_importance = dict(sorted(
            zip(feature_names, feature_importances.tolist()),
            key=lambda x: -x[1]
        )[:15])

    result.status = "HEALTHY" if result.dd_improvement_pct > 2 else (
        "WARNING" if result.dd_improvement_pct > 0 else "CRITICAL"
    )

    log(f"Static Sizing Return: {result.static_sizing_return:.2f}%")
    log(f"Dynamic Sizing Return: {result.dynamic_sizing_return:.2f}%")
    log(f"Return Improvement: {result.improvement_pct:+.2f}%")
    log(f"DD Static: {result.max_dd_static:.2f}% -> Dynamic: {result.max_dd_dynamic:.2f}%")
    log(f"DD Improvement: {result.dd_improvement_pct:+.2f}%")

    return result


# ============================================================
# APPROACH 4: LSTM Sequential Feature Extractor
# ============================================================

class LSTMFeatureExtractor(nn.Module):
    """LSTM network for extracting temporal features from positioning data."""

    def __init__(self, input_dim, hidden_dim=32, num_layers=1, output_dim=8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.classifier = nn.Linear(output_dim, 1)

    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        features = self.fc(h_n[-1])
        logits = self.classifier(features)
        return logits.squeeze(-1), features


@dataclass
class LSTMResult:
    name: str = "LSTM_SequentialFeatures"
    description: str = "LSTM temporal patterns + XGBoost ensemble"
    lstm_accuracy: float = 0.0
    ensemble_accuracy: float = 0.0
    lstm_win_rate: float = 0.0
    ensemble_win_rate: float = 0.0
    baseline_win_rate: float = 0.0
    improvement_lstm: float = 0.0
    improvement_ensemble: float = 0.0
    oos_efficiency: float = 0.0
    training_loss_final: float = 0.0
    windows_evaluated: int = 0
    status: str = "PENDING"


def run_lstm_experiment(df: pd.DataFrame) -> LSTMResult:
    """
    LSTM to capture temporal patterns, then combine features with XGBoost.
    """
    log("=" * 60)
    log("APPROACH 4: LSTM Sequential Features + XGBoost Ensemble")
    log("=" * 60)

    result = LSTMResult()

    df_work = df.copy()
    if 'hour' in df_work.columns:
        df_work = df_work[df_work['hour'].isin(ASIAN_HOURS)].copy()

    feature_df = get_feature_matrix(df_work)
    valid_mask = df_work.loc[feature_df.index, 'trade_label'].notna()
    valid_idx = feature_df.index[valid_mask.loc[feature_df.index]]

    X = feature_df.loc[valid_idx].values
    y = df_work.loc[valid_idx, 'trade_label'].values

    # Use last 90 days for LSTM to keep training fast
    max_samples = 96 * 90  # ~90 days of 15-min bars (Asian hours = ~half)
    if len(X) > max_samples:
        X = X[-max_samples:]
        y = y[-max_samples:]

    log(f"Total samples (after trim): {len(X)}")

    if len(X) < 300:
        result.status = "INSUFFICIENT_DATA"
        return result

    # Check for NaN/inf in features
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0 or inf_count > 0:
        log(f"WARNING: X has {nan_count} NaN, {inf_count} Inf values. Replacing with 0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Create sequences for LSTM (lookback = 8 bars = 2 hours)
    seq_len = 8
    log(f"Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log(f"Scaled. Creating sequences...")

    # Vectorized sequence creation
    n_seqs = len(X) - seq_len
    seq_indices = np.arange(seq_len).reshape(1, -1) + np.arange(n_seqs).reshape(-1, 1)
    X_seq = X_scaled[seq_indices]
    X_flat = X_scaled[seq_len:]
    y_seq = y[seq_len:]
    log(f"Sequences created: {X_seq.shape}")

    # Walk-forward: 3 windows only for speed
    n_total = len(X_seq)
    n_test_per_window = n_total // 5  # 20% test per window
    train_sizes = [int(n_total * 0.6), int(n_total * 0.7), int(n_total * 0.8)]
    log(f"LSTM sequences: {X_seq.shape}, walk-forward: {len(train_sizes)} windows")

    all_lstm_preds = []
    all_ensemble_preds = []
    all_actuals = []
    n_windows = 0

    device = torch.device('cpu')

    for wi, train_end in enumerate(train_sizes):
        log(f"  Window {wi+1}/{len(train_sizes)}: train_end={train_end}")
        test_end = min(train_end + n_test_per_window, n_total)
        if test_end <= train_end:
            continue

        # Use last 14 days of training data only
        train_start = max(0, train_end - 96 * 14)
        log(f"  Creating tensors...")
        train_X_seq_t = torch.FloatTensor(X_seq[train_start:train_end]).to(device)
        train_X_flat_t = X_flat[train_start:train_end]
        train_y_t = torch.FloatTensor(y_seq[train_start:train_end]).to(device)
        test_X_seq_t = torch.FloatTensor(X_seq[train_end:test_end]).to(device)
        test_X_flat_t = X_flat[train_end:test_end]
        test_y_t = y_seq[train_end:test_end]

        if len(test_X_seq_t) == 0:
            continue

        log(f"  LSTM window {n_windows+1}: train={len(train_X_seq_t)}, test={len(test_X_seq_t)}")

        # Train LSTM
        input_dim = X_seq.shape[2]
        model = LSTMFeatureExtractor(input_dim, hidden_dim=16, num_layers=1, output_dim=4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        batch_size = min(512, len(train_X_seq_t))
        n_epochs = 5

        for epoch in range(n_epochs):
            indices_perm = torch.randperm(len(train_X_seq_t))
            total_loss = 0
            n_batches = 0

            for batch_start_idx in range(0, len(train_X_seq_t), batch_size):
                batch_idx = indices_perm[batch_start_idx:batch_start_idx + batch_size]
                batch_X = train_X_seq_t[batch_idx]
                batch_y = train_y_t[batch_idx]

                optimizer.zero_grad()
                logits, _ = model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        result.training_loss_final = total_loss / max(n_batches, 1)

        # Extract LSTM features
        model.eval()
        with torch.no_grad():
            test_logits, test_features = model(test_X_seq_t)
            test_lstm_probs = torch.sigmoid(test_logits).cpu().numpy()
            test_lstm_features = test_features.cpu().numpy()

            train_logits, train_features = model(train_X_seq_t)
            train_lstm_features = train_features.cpu().numpy()

        # LSTM-only predictions
        lstm_preds = (test_lstm_probs > 0.5).astype(int)

        # Ensemble: combine LSTM features with original features for XGBoost
        train_ensemble = np.hstack([train_X_flat_t, train_lstm_features])
        test_ensemble = np.hstack([test_X_flat_t, test_lstm_features])

        xgb_model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbosity=0,
            eval_metric='logloss',
        )
        xgb_model.fit(train_ensemble, y_seq[train_start:train_end], verbose=False)
        ensemble_probs = xgb_model.predict_proba(test_ensemble)[:, 1]
        ensemble_preds = (ensemble_probs > 0.5).astype(int)

        all_lstm_preds.extend(lstm_preds)
        all_ensemble_preds.extend(ensemble_preds)
        all_actuals.extend(test_y_t)
        n_windows += 1

    if n_windows == 0:
        result.status = "NO_WINDOWS"
        return result

    all_lstm_preds = np.array(all_lstm_preds)
    all_ensemble_preds = np.array(all_ensemble_preds)
    all_actuals = np.array(all_actuals)

    result.lstm_accuracy = accuracy_score(all_actuals, all_lstm_preds) * 100
    result.ensemble_accuracy = accuracy_score(all_actuals, all_ensemble_preds) * 100
    result.baseline_win_rate = all_actuals.mean() * 100
    result.windows_evaluated = n_windows

    # Win rate when model predicts win
    lstm_win_mask = all_lstm_preds == 1
    ensemble_win_mask = all_ensemble_preds == 1

    if lstm_win_mask.sum() > 0:
        result.lstm_win_rate = all_actuals[lstm_win_mask].mean() * 100
    if ensemble_win_mask.sum() > 0:
        result.ensemble_win_rate = all_actuals[ensemble_win_mask].mean() * 100

    result.improvement_lstm = result.lstm_win_rate - result.baseline_win_rate
    result.improvement_ensemble = result.ensemble_win_rate - result.baseline_win_rate

    if result.baseline_win_rate > 0:
        result.oos_efficiency = result.ensemble_win_rate / max(result.baseline_win_rate, 1)

    result.status = "HEALTHY" if result.improvement_ensemble > 3 else (
        "WARNING" if result.improvement_ensemble > 0 else "CRITICAL"
    )

    log(f"Baseline Win Rate: {result.baseline_win_rate:.1f}%")
    log(f"LSTM Win Rate: {result.lstm_win_rate:.1f}% ({result.improvement_lstm:+.1f}%)")
    log(f"Ensemble Win Rate: {result.ensemble_win_rate:.1f}% ({result.improvement_ensemble:+.1f}%)")
    log(f"LSTM Accuracy: {result.lstm_accuracy:.1f}%")
    log(f"Ensemble Accuracy: {result.ensemble_accuracy:.1f}%")

    return result


# ============================================================
# MAIN RUNNER
# ============================================================

def run_all_experiments() -> Dict[str, Any]:
    """Run all 4 ML experiments and collect results."""
    log("=" * 70)
    log("BTC STRATEGY ML OPTIMIZATION EXPERIMENTS")
    log("=" * 70)

    # Load and prepare data
    log("\nLoading and preparing data...")
    df = load_and_prepare_data()
    log(f"Data shape: {df.shape}")
    log(f"Date range: {df.index.min()} to {df.index.max()}")

    # Generate trade labels
    log("\nGenerating trade labels...")
    df = generate_trade_labels(df)
    valid_labels = df['trade_label'].notna().sum()
    win_rate = df['trade_label'].mean() * 100
    log(f"Valid labels: {valid_labels}, Overall win rate: {win_rate:.1f}%")

    # Run experiments
    results = {}

    # 1. XGBoost Trade Quality
    try:
        xgb_result = run_xgboost_experiment(df)
        results['xgboost'] = asdict(xgb_result)
        log(f"\nXGBoost Status: {xgb_result.status}")
    except Exception as e:
        log(f"XGBoost experiment failed: {e}")
        results['xgboost'] = {'name': 'XGBoost_TradeQuality', 'status': 'ERROR', 'error': str(e)}

    # 2. Random Forest Regime
    try:
        rf_result = run_rf_regime_experiment(df)
        results['random_forest'] = asdict(rf_result)
        log(f"\nRF Regime Status: {rf_result.status}")
    except Exception as e:
        log(f"RF Regime experiment failed: {e}")
        results['random_forest'] = {'name': 'RandomForest_RegimeDetector', 'status': 'ERROR', 'error': str(e)}

    # 3. LightGBM Streak Predictor
    try:
        lgb_result = run_lgb_streak_experiment(df)
        results['lightgbm'] = asdict(lgb_result)
        log(f"\nLightGBM Status: {lgb_result.status}")
    except Exception as e:
        log(f"LightGBM experiment failed: {e}")
        results['lightgbm'] = {'name': 'LightGBM_StreakPredictor', 'status': 'ERROR', 'error': str(e)}

    # 4. LSTM + XGBoost Ensemble
    try:
        lstm_result = run_lstm_experiment(df)
        results['lstm'] = asdict(lstm_result)
        log(f"\nLSTM Status: {lstm_result.status}")
    except Exception as e:
        log(f"LSTM experiment failed: {e}")
        results['lstm'] = {'name': 'LSTM_SequentialFeatures', 'status': 'ERROR', 'error': str(e)}

    # Save results
    output = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'data_shape': list(df.shape),
        'date_range': [str(df.index.min()), str(df.index.max())],
        'overall_win_rate': win_rate,
        'valid_labels': int(valid_labels),
        'experiments': results,
    }

    results_file = os.path.join(SCRIPT_DIR, 'ml_experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    log(f"\nResults saved: {results_file}")

    # Print summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    for key, r in results.items():
        name = r.get('name', key)
        status = r.get('status', 'UNKNOWN')
        improvement = r.get('improvement_vs_baseline',
                           r.get('improvement_ensemble',
                                r.get('improvement_pct', 0)))
        log(f"  {name}: {status} (improvement: {improvement:+.2f}%)")

    return output


if __name__ == "__main__":
    run_all_experiments()
