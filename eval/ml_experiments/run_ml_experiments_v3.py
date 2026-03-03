#!/usr/bin/env python3
"""
ML Experiments v3 — Comprehensive Multi-Strategy Comparison
==============================================================

Tests 8 ML configurations side-by-side on the same real strategy entry signals:

  A) Baseline:         No ML — raw strategy
  B) Hard Gate (v2.1): RF regime reject + XGB quality reject (current approach)
  C) Soft Regime Only: RF regime scales position (no reject), no XGB filter
  D) Soft Regime+XGB:  RF regime scales + XGB scales (no rejections at all)
  E) XGB-Only Soft:    No regime gate, XGB probability scales position
  F) LGB-Only Sizing:  No gates, only LightGBM streak-aware sizing
  G) Ensemble Soft:    All layers as soft signals — combined scaling factor
  H) Adaptive Gate:    Regime-dependent thresholds — soft in moderate, hard in extreme

All trained on ALL bars (113k–146k samples) with walk-forward monthly retraining.
Evaluation on real strategy entry signals only (54 trades in last year).
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(EVAL_DIR)
MAIN_REPO_DIR = os.path.join(os.path.expanduser('~'), 'Github', 'btc-enhanced-streak-mitigation')
for data_dir in [REPO_DIR, MAIN_REPO_DIR]:
    backtest_15min_path = os.path.join(data_dir, 'backtest_15min')
    if os.path.exists(backtest_15min_path):
        sys.path.insert(0, backtest_15min_path)
        break

sys.path.insert(0, os.path.join(REPO_DIR, 'backtest_15min_new'))
sys.path.insert(0, os.path.join(REPO_DIR, 'backtest_15min_new_streak_a'))
sys.path.insert(0, EVAL_DIR)

from run_ml_experiments import (
    load_and_prepare_data, compute_all_features,
    get_feature_matrix, FEATURE_COLUMNS, classify_regime,
    generate_trade_labels, LSTMFeatureExtractor,
)
from config import (
    ASIAN_HOURS, ATR_PERIOD, RSI_PERIOD, SMA_PERIOD,
    TOP_TRADER_STRONG, TOP_TRADER_MODERATE,
    RISK_TIERS, STREAK_RULES,
)


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# STRATEGY CONFIGS — each defines how ML layers are applied
# ============================================================

STRATEGY_CONFIGS = {
    'A_Baseline': {
        'description': 'No ML — raw strategy baseline',
        'regime_mode': 'none',        # none | hard | soft
        'xgb_mode': 'none',           # none | hard | soft
        'lgb_sizing': False,
        'lstm_boost': False,
    },
    'B_HardGate': {
        'description': 'Hard RF regime reject + hard XGB quality reject (v2.1)',
        'regime_mode': 'hard',
        'xgb_mode': 'hard',
        'lgb_sizing': True,
        'lstm_boost': True,
    },
    'C_SoftRegimeOnly': {
        'description': 'RF regime scales position 0.5x-1.0x (no reject), no XGB',
        'regime_mode': 'soft',
        'xgb_mode': 'none',
        'lgb_sizing': False,
        'lstm_boost': False,
        'regime_unfavorable_scale': 0.5,
        'regime_favorable_scale': 1.0,
    },
    'D_SoftRegimeXGB': {
        'description': 'RF regime soft + XGB soft scaling (no rejections)',
        'regime_mode': 'soft',
        'xgb_mode': 'soft',
        'lgb_sizing': False,
        'lstm_boost': False,
        'regime_unfavorable_scale': 0.5,
        'regime_favorable_scale': 1.0,
    },
    'E_XGBOnlySoft': {
        'description': 'XGB win probability directly scales position (0.3x-1.2x)',
        'regime_mode': 'none',
        'xgb_mode': 'soft',
        'lgb_sizing': False,
        'lstm_boost': False,
    },
    'F_LGBOnlySizing': {
        'description': 'No gates, LightGBM loss-predictor sizing only',
        'regime_mode': 'none',
        'xgb_mode': 'none',
        'lgb_sizing': True,
        'lstm_boost': False,
    },
    'G_EnsembleSoft': {
        'description': 'All layers as soft signals — combined weighted scaling',
        'regime_mode': 'soft',
        'xgb_mode': 'soft',
        'lgb_sizing': True,
        'lstm_boost': True,
        'regime_unfavorable_scale': 0.6,
        'regime_favorable_scale': 1.1,
    },
    'H_AdaptiveGate': {
        'description': 'Regime-adaptive: soft in moderate, hard reject only in extreme',
        'regime_mode': 'adaptive',
        'xgb_mode': 'soft',
        'lgb_sizing': True,
        'lstm_boost': True,
        'regime_unfavorable_scale': 0.4,
        'regime_favorable_scale': 1.15,
    },
}


# ============================================================
# REAL STRATEGY ENTRY SIGNAL GENERATION (same as v2.1)
# ============================================================

def generate_real_entry_signals(df):
    min_pos_long = 0.4
    rsi_long_range = (20, 45)
    pullback_range = (0.5, 3.0)
    stop_atr_mult = 1.8
    tp_atr_mult_base = 4.5
    consec_loss_threshold = 3
    consec_loss_min_pos = 0.75

    close = df['close'].values.astype(np.float64)
    high_vals = df['high'].values.astype(np.float64)
    low_vals = df['low'].values.astype(np.float64)
    atr_vals = df['atr'].values.astype(np.float64)
    rsi_vals = df['rsi'].values.astype(np.float64)
    pos_scores = df['positioning_score'].values.astype(np.float64)
    uptrend_vals = df['uptrend'].values.astype(np.float64)
    pullback_vals = df['pullback_pct'].values.astype(np.float64)

    hours = df.index.hour if hasattr(df.index, 'hour') else np.zeros(len(df))
    asian_mask = np.isin(hours, list(ASIAN_HOURS))

    n = len(df)
    max_hold_bars = 96

    entry_mask = np.zeros(n, dtype=bool)
    trade_labels = np.full(n, np.nan)
    trade_pnl_ratios = np.full(n, np.nan)
    trade_exit_bars = np.full(n, np.nan)

    in_trade = False
    trade_entry_idx = -1
    trade_stop = trade_target = 0.0
    consecutive_losses = 0

    for i in range(n):
        if np.isnan(atr_vals[i]) or atr_vals[i] <= 0: continue
        if np.isnan(rsi_vals[i]) or np.isnan(pos_scores[i]): continue
        if np.isnan(pullback_vals[i]) or np.isnan(uptrend_vals[i]): continue

        if in_trade:
            if low_vals[i] <= trade_stop:
                trade_labels[trade_entry_idx] = 0
                trade_pnl_ratios[trade_entry_idx] = -1.0
                trade_exit_bars[trade_entry_idx] = i - trade_entry_idx
                in_trade = False
                consecutive_losses += 1
            elif high_vals[i] >= trade_target:
                pnl_ratio = (trade_target - close[trade_entry_idx]) / (close[trade_entry_idx] - trade_stop)
                trade_labels[trade_entry_idx] = 1
                trade_pnl_ratios[trade_entry_idx] = pnl_ratio
                trade_exit_bars[trade_entry_idx] = i - trade_entry_idx
                in_trade = False
                consecutive_losses = 0
            elif (i - trade_entry_idx) >= max_hold_bars:
                final_pnl = (close[i] - close[trade_entry_idx]) / (close[trade_entry_idx] - trade_stop)
                trade_labels[trade_entry_idx] = 1 if final_pnl > 0 else 0
                trade_pnl_ratios[trade_entry_idx] = final_pnl
                trade_exit_bars[trade_entry_idx] = i - trade_entry_idx
                in_trade = False
                consecutive_losses = consecutive_losses + 1 if final_pnl <= 0 else 0
            continue

        if not asian_mask[i]: continue
        ps = pos_scores[i]
        if abs(ps) < 0.15: continue
        if consecutive_losses >= consec_loss_threshold and ps < consec_loss_min_pos: continue
        if ps < min_pos_long: continue
        if uptrend_vals[i] < 1: continue
        if not (pullback_range[0] < pullback_vals[i] < pullback_range[1]): continue
        if not (rsi_long_range[0] < rsi_vals[i] < rsi_long_range[1]): continue

        entry_mask[i] = True
        in_trade = True
        trade_entry_idx = i

        tp_mult = tp_atr_mult_base
        if abs(ps) >= 1.5: tp_mult *= 1.3
        elif abs(ps) >= 1.0: tp_mult *= 1.15

        trade_stop = close[i] - atr_vals[i] * stop_atr_mult
        trade_target = close[i] + atr_vals[i] * tp_mult

    return entry_mask, trade_labels, trade_pnl_ratios, trade_exit_bars


def get_risk_amount(equity):
    for min_eq, max_eq, risk in RISK_TIERS:
        if min_eq <= equity < max_eq:
            return risk
    return RISK_TIERS[-1][2]


def apply_streak_reduction(base_risk, consecutive_losses):
    risk = base_risk
    for streak_level in sorted(STREAK_RULES.keys()):
        if consecutive_losses >= streak_level:
            risk *= (1 - STREAK_RULES[streak_level]['reduction'])
    return risk


# ============================================================
# ML MODEL TRAINING
# ============================================================

def train_ml_models(train_X_ab, train_y_ab, train_reg_ab,
                    train_entry_y, train_entry_positions, X_all, idx_to_pos,
                    X_seq_all, X_scaled_all, allbar_labels, all_indices,
                    seq_len, cutoff_pos, entry_indices, first_entry_of_month, df,
                    regime_names):
    """Train all ML models. Returns dict of trained models + metadata."""
    scaler = StandardScaler()
    train_X_s = scaler.fit_transform(train_X_ab)

    # --- RF Regime ---
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                 max_features='sqrt', random_state=42, n_jobs=-1)
    rf.fit(train_X_s, train_reg_ab)

    # Determine favorable regimes from real entry WR
    base_wr = train_entry_y.mean() if len(train_entry_y) > 0 else 0.3
    favorable = []
    unfavorable = []

    valid_entry = ~np.isnan(train_entry_y)
    valid_positions = [train_entry_positions[j] for j in range(len(train_entry_positions))
                       if j < len(valid_entry) and valid_entry[j]]
    if len(valid_positions) > 0:
        entry_X_s = scaler.transform(X_all[valid_positions])
        entry_regime_preds = rf.predict(entry_X_s)
        clean_y = train_entry_y[valid_entry][:len(entry_regime_preds)]
        regime_wr = {}
        for rid in regime_names:
            m = entry_regime_preds == rid
            if m.sum() > 2:
                regime_wr[rid] = clean_y[m].mean()
                if clean_y[m].mean() >= base_wr * 0.85:
                    favorable.append(rid)
                else:
                    unfavorable.append(rid)

    if not favorable:
        # Fallback to all-bar analysis
        train_regime_preds = rf.predict(train_X_s)
        for rid in regime_names:
            m = train_regime_preds == rid
            if m.sum() > 50 and train_y_ab[m].mean() > train_y_ab.mean():
                favorable.append(rid)
            elif m.sum() > 50:
                unfavorable.append(rid)

    if not favorable:
        favorable = list(regime_names.keys())

    # Per-regime win rates for the report
    regime_win_rates = {}
    train_regime_preds_all = rf.predict(train_X_s)
    for rid in regime_names:
        m = train_regime_preds_all == rid
        if m.sum() > 0:
            regime_win_rates[regime_names[rid]] = float(train_y_ab[m].mean())

    # --- XGBoost Quality (improved: deeper, better calibrated) ---
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=15,
        reg_alpha=1.0, reg_lambda=3.0, random_state=42,
        verbosity=0, eval_metric='logloss', gamma=0.3,
    )
    xgb_model.fit(train_X_s, train_y_ab, verbose=False)
    train_xgb_probs = xgb_model.predict_proba(train_X_s)[:, 1]

    # Improved threshold: percentile-based but with entry-specific calibration
    xgb_thresh = np.percentile(train_xgb_probs, 30)
    xgb_thresh = max(0.25, min(0.45, xgb_thresh))

    # Compute entry-specific XGB calibration
    if len(valid_positions) > 0:
        entry_probs = xgb_model.predict_proba(scaler.transform(X_all[valid_positions]))[:, 1]
        clean_y_entry = train_entry_y[valid_entry][:len(entry_probs)]
        # Find threshold that maximizes profit factor on entry signals
        best_thresh = xgb_thresh
        best_score = 0
        for t in np.arange(0.20, 0.50, 0.02):
            kept = entry_probs >= t
            if kept.sum() > 3:
                wr = clean_y_entry[kept].mean()
                keep_pct = kept.mean()
                # Score: balance WR improvement vs trade retention
                score = wr * keep_pct * 100
                if score > best_score:
                    best_score = score
                    best_thresh = t
        xgb_thresh = best_thresh

    # --- LightGBM Loss Predictor ---
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7, min_child_samples=10,
        reg_alpha=0.5, reg_lambda=2.0, random_state=42, verbose=-1,
    )
    lgb_model.fit(train_X_s, 1 - train_y_ab)

    # --- LSTM Ensemble ---
    lstm_model = None
    ens_xgb = None
    lstm_available = False
    device = torch.device('cpu')

    if len(X_seq_all) > 0:
        lstm_lookback = min(cutoff_pos, 96 * 30)
        lstm_start = max(seq_len, cutoff_pos - lstm_lookback)
        lstm_positions = list(range(lstm_start, cutoff_pos))
        lstm_valid_pos = [p for p in lstm_positions
                          if p >= seq_len and (p - seq_len) < len(X_seq_all)
                          and p < len(allbar_labels) and not np.isnan(allbar_labels[p])]

        if len(lstm_valid_pos) > 200:
            lstm_seqs = X_seq_all[[p - seq_len for p in lstm_valid_pos]]
            lstm_flat = X_scaled_all[lstm_valid_pos]
            lstm_y = allbar_labels[lstm_valid_pos]

            t_seq = torch.FloatTensor(lstm_seqs).to(device)
            t_y = torch.FloatTensor(lstm_y).to(device)

            lstm_model = LSTMFeatureExtractor(lstm_seqs.shape[2], hidden_dim=16, num_layers=1, output_dim=4).to(device)
            opt = torch.optim.Adam(lstm_model.parameters(), lr=0.003, weight_decay=1e-4)
            crit = nn.BCEWithLogitsLoss()

            lstm_model.train()
            bs = min(512, len(t_seq))
            for ep in range(10):
                perm = torch.randperm(len(t_seq))
                for s in range(0, len(t_seq), bs):
                    bi = perm[s:s+bs]
                    opt.zero_grad()
                    lo, _ = lstm_model(t_seq[bi])
                    l = crit(lo, t_y[bi])
                    l.backward()
                    opt.step()

            lstm_model.eval()
            with torch.no_grad():
                _, train_feats = lstm_model(t_seq)
                train_lstm_feats = train_feats.cpu().numpy()

            ens_train = np.hstack([lstm_flat, train_lstm_feats])
            ens_xgb = xgb.XGBClassifier(
                n_estimators=80, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42, verbosity=0, eval_metric='logloss',
            )
            ens_xgb.fit(ens_train, lstm_y, verbose=False)
            lstm_available = True

    return {
        'scaler': scaler, 'rf': rf, 'xgb_model': xgb_model, 'lgb_model': lgb_model,
        'lstm_model': lstm_model, 'ens_xgb': ens_xgb,
        'favorable': favorable, 'unfavorable': unfavorable,
        'xgb_thresh': xgb_thresh, 'lstm_available': lstm_available,
        'device': device, 'regime_win_rates': regime_win_rates,
        'train_samples': len(train_X_ab),
    }


# ============================================================
# APPLY ML TO A SINGLE TRADE
# ============================================================

def apply_ml_strategy(strategy_name, config, models, test_x, test_x_s, pos, df, idx,
                      X_seq_all, X_scaled_all, seq_len, regime_names):
    """
    Apply ML strategy to a single trade entry.
    Returns: (take_trade: bool, scale: float, info: dict)
    """
    regime_mode = config.get('regime_mode', 'none')
    xgb_mode = config.get('xgb_mode', 'none')
    use_lgb = config.get('lgb_sizing', False)
    use_lstm = config.get('lstm_boost', False)

    scale = 1.0
    info = {'regime': '', 'xgb_prob': 0, 'lgb_scale': 1.0, 'lstm_boost': 1.0, 'action': 'TAKEN'}

    # --- L1: RF Regime ---
    if regime_mode != 'none':
        regime_pred = models['rf'].predict(test_x_s)[0]
        regime_name = regime_names.get(int(regime_pred), '?')
        info['regime'] = regime_name
        is_favorable = int(regime_pred) in models['favorable']

        if regime_mode == 'hard':
            if not is_favorable:
                info['action'] = f'REJECTED(regime:{regime_name})'
                return False, 0.0, info

        elif regime_mode == 'soft':
            unfav_scale = config.get('regime_unfavorable_scale', 0.5)
            fav_scale = config.get('regime_favorable_scale', 1.0)
            if is_favorable:
                scale *= fav_scale
            else:
                scale *= unfav_scale

        elif regime_mode == 'adaptive':
            unfav_scale = config.get('regime_unfavorable_scale', 0.4)
            fav_scale = config.get('regime_favorable_scale', 1.15)
            # Check regime win rate — hard reject only if WR < 20%
            regime_wr = models['regime_win_rates'].get(regime_name, 0.3)
            if regime_wr < 0.20:
                info['action'] = f'REJECTED(regime:{regime_name},wr={regime_wr:.0%})'
                return False, 0.0, info
            elif is_favorable:
                scale *= fav_scale
            else:
                # Scale proportionally to regime WR
                scale *= max(unfav_scale, regime_wr * 1.5)

    # --- L2: XGBoost Quality ---
    if xgb_mode != 'none':
        xgb_prob = models['xgb_model'].predict_proba(test_x_s)[0, 1]
        info['xgb_prob'] = float(xgb_prob)

        if xgb_mode == 'hard':
            if xgb_prob < models['xgb_thresh']:
                info['action'] = f'REJECTED(xgb:{xgb_prob:.3f}<{models["xgb_thresh"]:.3f})'
                return False, 0.0, info

        elif xgb_mode == 'soft':
            # Map XGB probability to scale: 0.3x at prob=0.15, 1.2x at prob=0.55+
            xgb_scale = 0.3 + (xgb_prob - 0.15) * (0.9 / 0.40)
            xgb_scale = max(0.3, min(1.2, xgb_scale))
            scale *= xgb_scale

    # --- L3: LightGBM Sizing ---
    if use_lgb:
        loss_prob = models['lgb_model'].predict_proba(test_x_s)[0, 1]
        lgb_scale = max(0.4, 1.0 - loss_prob * 0.5)
        info['lgb_scale'] = float(lgb_scale)
        scale *= lgb_scale

    # --- L4: LSTM Boost ---
    if use_lstm and models['lstm_available']:
        if pos >= seq_len and (pos - seq_len) < len(X_seq_all):
            device = models['device']
            with torch.no_grad():
                t_s = torch.FloatTensor(X_seq_all[pos - seq_len:pos - seq_len + 1]).to(device)
                _, feat = models['lstm_model'](t_s)
                feat_np = feat.cpu().numpy()
                t_flat = X_scaled_all[pos:pos+1]
                ens_input = np.hstack([t_flat, feat_np])
                ens_prob = models['ens_xgb'].predict_proba(ens_input)[0, 1]
                if ens_prob > 0.5:
                    boost = 1.0 + (ens_prob - 0.5) * 0.3
                else:
                    boost = 0.8 + ens_prob * 0.4
                info['lstm_boost'] = float(boost)
                scale *= boost

    # Final scale clamp
    scale = max(0.15, min(1.5, scale))
    info['final_scale'] = float(scale)
    return True, scale, info


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():
    log("=" * 80)
    log("ML EXPERIMENTS v3 — COMPREHENSIVE MULTI-STRATEGY COMPARISON")
    log("=" * 80)

    # --- Load and prepare data ---
    log("Loading data...")
    df = load_and_prepare_data()
    log(f"Data: {df.shape}, {df.index.min()} to {df.index.max()}")

    log("Generating hypothetical trade labels for ALL bars...")
    df = generate_trade_labels(df, stop_mult=1.8, tp_mult=4.5)
    allbar_count = df['trade_label'].notna().sum()
    log(f"All-bar labels: {allbar_count:,} valid, WR: {df['trade_label'].mean()*100:.1f}%")

    log("Generating real strategy entry signals...")
    entry_mask, trade_labels, trade_pnl_ratios, trade_exit_bars = generate_real_entry_signals(df)
    total_entries = entry_mask.sum()
    valid_entries = (~np.isnan(trade_labels[entry_mask])).sum()
    entry_wr = np.nanmean(trade_labels[entry_mask]) * 100
    log(f"Real strategy: {total_entries} entries, {valid_entries} completed, WR: {entry_wr:.1f}%")

    df['real_entry'] = entry_mask
    df['real_label'] = trade_labels
    df['real_pnl_ratio'] = trade_pnl_ratios

    df['regime_label'] = df.apply(classify_regime, axis=1)
    regime_names = {0: 'StrongTrend', 1: 'ModerateTrend', 2: 'RangingHighVol',
                    3: 'RangingLowVol', 4: 'VolatileTrend'}

    feature_df = get_feature_matrix(df)
    entry_indices = df.index[entry_mask & df['real_label'].notna()]
    entry_indices = entry_indices.intersection(feature_df.index)
    log(f"Entry signals with features: {len(entry_indices)}")

    X_all = feature_df.values
    all_indices = feature_df.index
    allbar_labels = df.loc[feature_df.index, 'trade_label'].values
    allbar_regimes = df.loc[feature_df.index, 'regime_label'].values

    # Eval period
    data_end = df.index[-1]
    one_year_ago = data_end - pd.Timedelta(days=365)
    eval_entries = [idx for idx in entry_indices if idx >= one_year_ago]
    train_entries_all = [idx for idx in entry_indices if idx < one_year_ago]
    log(f"Train entries: {len(train_entries_all)}, Eval entries: {len(eval_entries)}")

    # LSTM sequences
    seq_len = 8
    X_clean = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    scaler_lstm_global = StandardScaler()
    X_scaled_all = scaler_lstm_global.fit_transform(X_clean)
    n_seqs = len(X_clean) - seq_len
    X_seq_all = X_scaled_all[np.arange(seq_len).reshape(1, -1) + np.arange(n_seqs).reshape(-1, 1)] if n_seqs > 0 else np.array([])

    # Group by month
    eval_months = {}
    for idx in eval_entries:
        mk = idx.strftime('%Y-%m')
        eval_months.setdefault(mk, []).append(idx)
    log(f"Eval months: {list(eval_months.keys())}")

    idx_to_pos = {idx: i for i, idx in enumerate(all_indices)}

    # ============================================================
    # Initialize per-strategy state
    # ============================================================
    strategy_state = {}
    for sname in STRATEGY_CONFIGS:
        strategy_state[sname] = {
            'equity': 100000.0,
            'consec_losses': 0,
            'trades': [],
            'equity_curve': [100000.0],
        }

    # ============================================================
    # Walk-forward monthly evaluation
    # ============================================================
    for month_key in sorted(eval_months.keys()):
        month_entries = eval_months[month_key]
        log(f"\n  Month {month_key}: {len(month_entries)} real entry signals")

        first_entry = month_entries[0]
        cutoff_pos = idx_to_pos.get(first_entry, len(X_all))

        # Training data
        train_X_ab = X_all[:cutoff_pos]
        train_y_ab = allbar_labels[:cutoff_pos]
        train_reg_ab = allbar_regimes[:cutoff_pos]
        valid_ab = ~np.isnan(train_y_ab)
        train_X_ab = train_X_ab[valid_ab]
        train_y_ab = train_y_ab[valid_ab]
        train_reg_ab = train_reg_ab[valid_ab]

        # Entry-specific training data
        train_entry_idx_list = [idx for idx in entry_indices if idx < first_entry]
        train_entry_positions = [idx_to_pos[idx] for idx in train_entry_idx_list if idx in idx_to_pos]
        train_entry_y = np.array([df.loc[idx, 'real_label'] for idx in train_entry_idx_list])

        if len(train_X_ab) < 1000:
            log(f"    Insufficient training data ({len(train_X_ab)})")
            # All strategies act as baseline
            for idx in month_entries:
                pos = idx_to_pos.get(idx)
                if pos is None: continue
                pnl_r = df.loc[idx, 'real_pnl_ratio']
                win = df.loc[idx, 'real_label']
                if np.isnan(pnl_r) or np.isnan(win): continue
                ps = df.loc[idx, 'positioning_score']
                for sname in STRATEGY_CONFIGS:
                    st = strategy_state[sname]
                    risk = get_risk_amount(st['equity'])
                    risk = apply_streak_reduction(risk, st['consec_losses'])
                    if abs(ps) >= 1.5: risk *= 1.2
                    elif abs(ps) >= 1.0: risk *= 1.1
                    pnl = risk * pnl_r
                    st['equity'] += pnl
                    st['consec_losses'] = 0 if win == 1 else st['consec_losses'] + 1
                    st['equity_curve'].append(st['equity'])
                    st['trades'].append({
                        'timestamp': str(idx), 'price': round(float(df.loc[idx, 'close']), 1),
                        'pos_score': round(float(ps), 2), 'actual_win': int(win),
                        'pnl_ratio': round(float(pnl_r), 3), 'risk': round(float(risk), 2),
                        'trade_pnl': round(float(pnl), 2), 'equity': round(float(st['equity']), 2),
                        'ml_action': 'TAKEN', 'final_scale': 1.0, 'regime': '', 'xgb_prob': 0,
                    })
            continue

        # Train ML models once per month
        models = train_ml_models(
            train_X_ab, train_y_ab, train_reg_ab,
            train_entry_y, train_entry_positions, X_all, idx_to_pos,
            X_seq_all, X_scaled_all, allbar_labels, all_indices,
            seq_len, cutoff_pos, entry_indices, first_entry, df, regime_names
        )

        log(f"    Trained on {models['train_samples']:,} bars, "
            f"favorable={[regime_names[r] for r in models['favorable']]}, "
            f"xgb_thresh={models['xgb_thresh']:.3f}, lstm={'Y' if models['lstm_available'] else 'N'}")

        # Evaluate each strategy on this month's entries
        strategy_counts = {s: {'taken': 0, 'rejected': 0} for s in STRATEGY_CONFIGS}

        for idx in month_entries:
            pos = idx_to_pos.get(idx)
            if pos is None: continue
            pnl_r = df.loc[idx, 'real_pnl_ratio']
            win = df.loc[idx, 'real_label']
            if np.isnan(pnl_r) or np.isnan(win): continue
            ps = df.loc[idx, 'positioning_score']
            close_price = df.loc[idx, 'close']

            test_x = X_all[pos:pos+1]
            test_x_s = models['scaler'].transform(test_x)

            for sname, config in STRATEGY_CONFIGS.items():
                st = strategy_state[sname]

                if sname == 'A_Baseline':
                    # Pure baseline — no ML
                    risk = get_risk_amount(st['equity'])
                    risk = apply_streak_reduction(risk, st['consec_losses'])
                    if abs(ps) >= 1.5: risk *= 1.2
                    elif abs(ps) >= 1.0: risk *= 1.1
                    pnl = risk * pnl_r
                    st['equity'] += pnl
                    st['consec_losses'] = 0 if win == 1 else st['consec_losses'] + 1
                    st['equity_curve'].append(st['equity'])
                    st['trades'].append({
                        'timestamp': str(idx), 'price': round(float(close_price), 1),
                        'pos_score': round(float(ps), 2), 'actual_win': int(win),
                        'pnl_ratio': round(float(pnl_r), 3), 'risk': round(float(risk), 2),
                        'trade_pnl': round(float(pnl), 2), 'equity': round(float(st['equity']), 2),
                        'ml_action': 'TAKEN', 'final_scale': 1.0, 'regime': '', 'xgb_prob': 0,
                    })
                    strategy_counts[sname]['taken'] += 1
                    continue

                # Apply ML strategy
                take_trade, scale, info = apply_ml_strategy(
                    sname, config, models, test_x, test_x_s, pos, df, idx,
                    X_seq_all, X_scaled_all, seq_len, regime_names
                )

                if not take_trade:
                    strategy_counts[sname]['rejected'] += 1
                    st['trades'].append({
                        'timestamp': str(idx), 'price': round(float(close_price), 1),
                        'pos_score': round(float(ps), 2), 'actual_win': int(win),
                        'pnl_ratio': round(float(pnl_r), 3), 'risk': 0,
                        'trade_pnl': 0, 'equity': round(float(st['equity']), 2),
                        'ml_action': info['action'], 'final_scale': 0,
                        'regime': info.get('regime', ''), 'xgb_prob': info.get('xgb_prob', 0),
                    })
                    continue

                # Take trade with ML-adjusted sizing
                risk = get_risk_amount(st['equity'])
                risk = apply_streak_reduction(risk, st['consec_losses'])
                if abs(ps) >= 1.5: risk *= 1.2
                elif abs(ps) >= 1.0: risk *= 1.1
                risk *= scale

                pnl = risk * pnl_r
                st['equity'] += pnl
                st['consec_losses'] = 0 if win == 1 else st['consec_losses'] + 1
                st['equity_curve'].append(st['equity'])
                strategy_counts[sname]['taken'] += 1

                st['trades'].append({
                    'timestamp': str(idx), 'price': round(float(close_price), 1),
                    'pos_score': round(float(ps), 2), 'actual_win': int(win),
                    'pnl_ratio': round(float(pnl_r), 3), 'risk': round(float(risk), 2),
                    'trade_pnl': round(float(pnl), 2), 'equity': round(float(st['equity']), 2),
                    'ml_action': 'TAKEN', 'final_scale': round(float(scale), 3),
                    'regime': info.get('regime', ''), 'xgb_prob': round(float(info.get('xgb_prob', 0)), 4),
                })

        # Log per-strategy counts for this month
        count_strs = [f"{s.split('_',1)[1][:8]}:{strategy_counts[s]['taken']}T/{strategy_counts[s]['rejected']}R"
                      for s in STRATEGY_CONFIGS if s != 'A_Baseline']
        log(f"    {' | '.join(count_strs)}")

    # ============================================================
    # COMPUTE RESULTS
    # ============================================================
    log(f"\n{'='*80}")
    log(f"COMPREHENSIVE RESULTS — ALL STRATEGIES")
    log(f"{'='*80}")

    all_results = {}

    for sname, config in STRATEGY_CONFIGS.items():
        st = strategy_state[sname]
        trades = st['trades']
        taken = [t for t in trades if t.get('ml_action') == 'TAKEN']
        rejected = [t for t in trades if 'REJECTED' in t.get('ml_action', '')]

        total = len(taken)
        wins = sum(1 for t in taken if t['actual_win'] == 1)
        wr = wins / total * 100 if total > 0 else 0
        pnl = sum(t['trade_pnl'] for t in taken)

        eq = np.array(st['equity_curve'])
        peak = np.maximum.accumulate(eq)
        dd = ((peak - eq) / np.maximum(peak, 1) * 100).max()

        gross_win = sum(t['trade_pnl'] for t in taken if t['actual_win'] == 1)
        gross_loss = abs(sum(t['trade_pnl'] for t in taken if t['actual_win'] == 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

        ret_pct = (eq[-1] - 100000) / 100000 * 100
        calmar = ret_pct / dd if dd > 0 else float('inf')

        # Max loss streak
        ms = cs = 0
        for t in taken:
            if t['actual_win'] == 0: cs += 1; ms = max(ms, cs)
            else: cs = 0

        # Rejection analysis
        rej_wins = sum(1 for t in rejected if t['actual_win'] == 1)
        rej_losses = len(rejected) - rej_wins
        rej_accuracy = rej_losses / len(rejected) * 100 if rejected else 0

        # Avg scale
        scales = [t.get('final_scale', 1.0) for t in taken if t.get('final_scale', 1.0) != 1.0 or sname == 'A_Baseline']
        win_scales = [t.get('final_scale', 1.0) for t in taken if t['actual_win'] == 1]
        loss_scales = [t.get('final_scale', 1.0) for t in taken if t['actual_win'] == 0]

        result = {
            'name': sname,
            'description': config['description'],
            'trades': total, 'wins': wins, 'win_rate': round(wr, 1),
            'total_pnl': round(pnl, 2), 'final_equity': round(float(eq[-1]), 2),
            'return_pct': round(ret_pct, 1), 'max_dd': round(dd, 1),
            'profit_factor': round(pf, 2), 'calmar_ratio': round(calmar, 2),
            'max_loss_streak': ms,
            'trades_rejected': len(rejected),
            'rej_would_win': rej_wins, 'rej_would_lose': rej_losses,
            'rej_accuracy': round(rej_accuracy, 1),
            'avg_scale_all': round(np.mean(scales), 3) if scales else 1.0,
            'avg_scale_wins': round(np.mean(win_scales), 3) if win_scales else 1.0,
            'avg_scale_losses': round(np.mean(loss_scales), 3) if loss_scales else 1.0,
            'scale_diff': round((np.mean(win_scales) if win_scales else 0) - (np.mean(loss_scales) if loss_scales else 0), 3),
            'equity_curve': [round(float(x), 2) for x in eq],
        }

        # Monthly breakdown
        monthly = {}
        for t in taken:
            m = t['timestamp'][:7]
            if m not in monthly: monthly[m] = {'trades': 0, 'wins': 0, 'pnl': 0}
            monthly[m]['trades'] += 1
            monthly[m]['wins'] += t['actual_win']
            monthly[m]['pnl'] += t['trade_pnl']
        result['monthly'] = monthly

        all_results[sname] = result

    # ============================================================
    # PRINT COMPARISON TABLE
    # ============================================================
    log(f"\n  {'Strategy':22} {'Trds':>5} {'WR':>6} {'PnL':>12} {'Ret%':>7} {'MaxDD':>7} {'PF':>6} {'Calmar':>7} {'Rej':>4} {'RejAcc':>7} {'AvgScl':>7}")
    log(f"  {'─'*105}")

    for sname in STRATEGY_CONFIGS:
        r = all_results[sname]
        log(f"  {sname:22} {r['trades']:>5} {r['win_rate']:>5.1f}% ${r['total_pnl']:>10,.0f} {r['return_pct']:>6.1f}% "
            f"{r['max_dd']:>6.1f}% {r['profit_factor']:>5.2f} {r['calmar_ratio']:>6.2f} "
            f"{r['trades_rejected']:>4} {r['rej_accuracy']:>6.1f}% {r['avg_scale_all']:>6.3f}")

    # Best strategy analysis
    log(f"\n  RANKING BY CALMAR RATIO (risk-adjusted return):")
    ranked = sorted(all_results.items(), key=lambda x: x[1]['calmar_ratio'], reverse=True)
    for i, (sname, r) in enumerate(ranked):
        marker = " <<<" if i == 0 else ""
        log(f"  {i+1}. {sname:22} Calmar={r['calmar_ratio']:.2f} (Ret={r['return_pct']:.1f}%, DD={r['max_dd']:.1f}%){marker}")

    log(f"\n  RANKING BY TOTAL PnL (absolute return):")
    ranked_pnl = sorted(all_results.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
    for i, (sname, r) in enumerate(ranked_pnl):
        marker = " <<<" if i == 0 else ""
        log(f"  {i+1}. {sname:22} PnL=${r['total_pnl']:>10,.0f} (WR={r['win_rate']:.1f}%, Trades={r['trades']}){marker}")

    log(f"\n  RANKING BY PROFIT FACTOR:")
    ranked_pf = sorted(all_results.items(), key=lambda x: x[1]['profit_factor'], reverse=True)
    for i, (sname, r) in enumerate(ranked_pf):
        marker = " <<<" if i == 0 else ""
        log(f"  {i+1}. {sname:22} PF={r['profit_factor']:.2f} (GrossW/GrossL, WR={r['win_rate']:.1f}%){marker}")

    # Scale analysis
    log(f"\n  ML SIZING DIFFERENTIATION (scale_wins - scale_losses):")
    for sname in STRATEGY_CONFIGS:
        r = all_results[sname]
        if sname != 'A_Baseline':
            diff = r['scale_diff']
            marker = " ✓" if diff > 0.02 else " ✗" if diff < -0.02 else " ~"
            log(f"  {sname:22} W={r['avg_scale_wins']:.3f} L={r['avg_scale_losses']:.3f} Diff={diff:>+.3f}{marker}")

    # Monthly comparison (all strategies)
    log(f"\n  MONTHLY PnL COMPARISON:")
    all_months_sorted = sorted(set(m for r in all_results.values() for m in r['monthly']))
    header = f"  {'Month':>7} " + " ".join(f"{s[:12]:>12}" for s in STRATEGY_CONFIGS)
    log(header)
    log(f"  {'─'*len(header)}")
    for mo in all_months_sorted:
        vals = []
        for s in STRATEGY_CONFIGS:
            m_data = all_results[s]['monthly'].get(mo, {'pnl': 0})
            vals.append(f"${m_data['pnl']:>10,.0f}")
        log(f"  {mo:>7} " + " ".join(f"{v:>12}" for v in vals))

    # Save all results
    save_results = {}
    for sname, r in all_results.items():
        save_r = {k: v for k, v in r.items() if k != 'equity_curve'}
        save_r['trades_detail'] = strategy_state[sname]['trades']
        save_r['equity_curve'] = r['equity_curve']
        save_results[sname] = save_r

    out_file = os.path.join(SCRIPT_DIR, 'ml_experiments_v3_results.json')
    with open(out_file, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    log(f"\nResults saved: {out_file}")

    return save_results


if __name__ == "__main__":
    run_pipeline()
