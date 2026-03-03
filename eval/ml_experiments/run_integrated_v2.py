#!/usr/bin/env python3
"""
Integrated ML Pipeline v2.1 — EXPANDED TRAINING HORIZON
=========================================================

Changes from v2:
  - ML models trained on ALL bars (hypothetical trade labels via generate_trade_labels)
    instead of only the ~109 real entry signals. This gives 50,000-100,000+ labeled
    training bars vs the previous 109 entries.
  - Full data from 2021-12-01 used for training horizon.
  - Evaluation still only on real entry signals (honest comparison).

The real strategy entry logic:
  1. Asian hours (0-11 UTC)
  2. positioning_score >= 0.4
  3. uptrend (close > sma200)
  4. pullback 0.5-3.0% from 16-bar high
  5. RSI(56) in [20, 45]
  6. No open trade
  7. After 3 losses: require positioning_score >= 0.75

ML layers applied ONLY to bars that pass these real entry filters:
  L1 (Gate):   RF Regime Detector — reject unfavorable regimes
  L2 (Filter): XGBoost Trade Quality — reject low-confidence setups
  L3 (Sizing): LightGBM Streak Pred. — dynamic position scaling
  L4 (Boost):  LSTM+XGBoost Ensemble — temporal confidence multiplier

Comparison modes:
  A) Real Baseline: actual strategy, no ML
  B) ML-Enhanced: same entries, but ML layers can reject or resize trades
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
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
# REAL STRATEGY ENTRY SIGNAL GENERATION
# ============================================================

def generate_real_entry_signals(df):
    """
    Reproduce the actual strategy's entry signals.
    Returns a boolean mask of bars where the real strategy would enter.
    Also returns trade outcomes via bar-by-bar SL/TP simulation.
    """
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
    uptrend = df['uptrend'].values.astype(np.float64)
    pullback = df['pullback_pct'].values.astype(np.float64)

    hours = df.index.hour if hasattr(df.index, 'hour') else np.zeros(len(df))
    asian_mask = np.isin(hours, list(ASIAN_HOURS))

    n = len(df)
    max_hold_bars = 96  # 24 hours at 15-min

    entry_mask = np.zeros(n, dtype=bool)
    trade_labels = np.full(n, np.nan)
    trade_pnl_ratios = np.full(n, np.nan)
    trade_exit_bars = np.full(n, np.nan)

    in_trade = False
    trade_entry_idx = -1
    trade_stop = 0.0
    trade_target = 0.0
    consecutive_losses = 0

    for i in range(n):
        if np.isnan(atr_vals[i]) or atr_vals[i] <= 0:
            continue
        if np.isnan(rsi_vals[i]) or np.isnan(pos_scores[i]):
            continue
        if np.isnan(pullback[i]) or np.isnan(uptrend[i]):
            continue

        # --- Check exit if in trade ---
        if in_trade:
            if low_vals[i] <= trade_stop:
                pnl_ratio = -1.0
                trade_labels[trade_entry_idx] = 0
                trade_pnl_ratios[trade_entry_idx] = pnl_ratio
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
                if final_pnl <= 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
            continue

        # --- Check entry conditions ---
        if not asian_mask[i]:
            continue
        ps = pos_scores[i]
        if abs(ps) < 0.15:
            continue
        if consecutive_losses >= consec_loss_threshold:
            if ps < consec_loss_min_pos:
                continue
        if ps < min_pos_long:
            continue
        if uptrend[i] < 1:
            continue
        if not (pullback_range[0] < pullback[i] < pullback_range[1]):
            continue
        if not (rsi_long_range[0] < rsi_vals[i] < rsi_long_range[1]):
            continue

        entry_mask[i] = True
        in_trade = True
        trade_entry_idx = i

        tp_mult = tp_atr_mult_base
        if abs(ps) >= 1.5:
            tp_mult *= 1.3
        elif abs(ps) >= 1.0:
            tp_mult *= 1.15

        trade_stop = close[i] - atr_vals[i] * stop_atr_mult
        trade_target = close[i] + atr_vals[i] * tp_mult

    return entry_mask, trade_labels, trade_pnl_ratios, trade_exit_bars


def get_risk_amount(equity):
    """Get tiered risk amount based on current equity."""
    for min_eq, max_eq, risk in RISK_TIERS:
        if min_eq <= equity < max_eq:
            return risk
    return RISK_TIERS[-1][2]


def apply_streak_reduction(base_risk, consecutive_losses):
    """Apply streak mitigation rules."""
    risk = base_risk
    for streak_level in sorted(STREAK_RULES.keys()):
        if consecutive_losses >= streak_level:
            risk *= (1 - STREAK_RULES[streak_level]['reduction'])
    return risk


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():
    log("=" * 70)
    log("INTEGRATED ML PIPELINE v2.1 — EXPANDED TRAINING (ALL BARS)")
    log("=" * 70)

    # --- Load data ---
    log("Loading data...")
    df = load_and_prepare_data()
    log(f"Data: {df.shape}, {df.index.min()} to {df.index.max()}")

    # --- Generate hypothetical trade labels for ALL bars ---
    log("Generating hypothetical trade labels for ALL bars (for ML training)...")
    df = generate_trade_labels(df, stop_mult=1.8, tp_mult=4.5)
    all_bar_label_count = df['trade_label'].notna().sum()
    all_bar_wr = df['trade_label'].mean() * 100
    log(f"All-bar labels: {all_bar_label_count:,} valid bars, WR: {all_bar_wr:.1f}%")

    # --- Generate REAL strategy entry signals ---
    log("Generating real strategy entry signals (bar-by-bar simulation)...")
    entry_mask, trade_labels, trade_pnl_ratios, trade_exit_bars = generate_real_entry_signals(df)
    total_entries = entry_mask.sum()
    valid_entries = (~np.isnan(trade_labels[entry_mask])).sum()
    entry_wr = np.nanmean(trade_labels[entry_mask]) * 100
    log(f"Real strategy: {total_entries} entries, {valid_entries} completed, WR: {entry_wr:.1f}%")

    # Store real entry labels in df
    df['real_entry'] = entry_mask
    df['real_label'] = trade_labels
    df['real_pnl_ratio'] = trade_pnl_ratios

    # --- Regime labels ---
    df['regime_label'] = df.apply(classify_regime, axis=1)
    regime_names = {0: 'StrongTrend', 1: 'ModerateTrend', 2: 'RangingHighVol',
                    3: 'RangingLowVol', 4: 'VolatileTrend'}

    # --- Feature matrix (all bars) ---
    feature_df = get_feature_matrix(df)
    entry_indices = df.index[entry_mask & df['real_label'].notna()]
    entry_indices = entry_indices.intersection(feature_df.index)
    log(f"Entry signals with features: {len(entry_indices)}")

    X_all = feature_df.values
    all_indices = feature_df.index

    # All-bar labels aligned to feature matrix
    allbar_labels = df.loc[feature_df.index, 'trade_label'].values
    allbar_regimes = df.loc[feature_df.index, 'regime_label'].values

    # --- Eval period: last 1 year ---
    data_end = df.index[-1]
    one_year_ago = data_end - pd.Timedelta(days=365)

    eval_entries = [idx for idx in entry_indices if idx >= one_year_ago]
    train_entries = [idx for idx in entry_indices if idx < one_year_ago]
    log(f"Real entry signals — Train: {len(train_entries)}, Eval: {len(eval_entries)}")
    log(f"Eval period: {one_year_ago} to {data_end}")

    # Count all-bar training samples available before eval period
    allbar_train_mask = (feature_df.index < one_year_ago) & (~np.isnan(allbar_labels[:len(feature_df)]))
    allbar_train_count = allbar_train_mask.sum() if hasattr(allbar_train_mask, 'sum') else 0
    log(f"All-bar training samples available: {allbar_train_count:,}")

    # --- LSTM sequence preparation ---
    seq_len = 8
    X_clean = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    scaler_lstm_global = StandardScaler()
    X_scaled_all = scaler_lstm_global.fit_transform(X_clean)
    n_seqs = len(X_clean) - seq_len
    if n_seqs > 0:
        seq_idx_arr = np.arange(seq_len).reshape(1, -1) + np.arange(n_seqs).reshape(-1, 1)
        X_seq_all = X_scaled_all[seq_idx_arr]
    else:
        X_seq_all = np.array([])

    # --- Walk-forward: retrain monthly ---
    eval_months = {}
    for idx in eval_entries:
        month_key = idx.strftime('%Y-%m')
        if month_key not in eval_months:
            eval_months[month_key] = []
        eval_months[month_key].append(idx)

    log(f"Eval months: {list(eval_months.keys())}")

    # --- Map timestamp → position in feature matrix ---
    idx_to_pos = {idx: i for i, idx in enumerate(all_indices)}

    # --- Run baseline and ML-enhanced in parallel ---
    baseline_trades = []
    ml_trades = []
    baseline_equity = 100000.0
    ml_equity = 100000.0
    baseline_consec_losses = 0
    ml_consec_losses = 0

    for month_key in sorted(eval_months.keys()):
        month_entries = eval_months[month_key]
        log(f"\n  Month {month_key}: {len(month_entries)} real entry signals")

        # Training cutoff: everything before the first entry of this month
        first_entry_of_month = month_entries[0]
        cutoff_pos = idx_to_pos.get(first_entry_of_month, len(X_all))

        # =====================
        # ALL-BAR TRAINING DATA (for XGBoost/RF/LGB)
        # =====================
        train_X_allbar = X_all[:cutoff_pos]
        train_y_allbar = allbar_labels[:cutoff_pos]
        train_regimes_allbar = allbar_regimes[:cutoff_pos]

        # Remove NaN labels
        valid_allbar = ~np.isnan(train_y_allbar)
        train_X_ab = train_X_allbar[valid_allbar]
        train_y_ab = train_y_allbar[valid_allbar]
        train_reg_ab = train_regimes_allbar[valid_allbar]

        # Also get the real entry signals for regime WR analysis
        train_entry_idx_list = [idx for idx in entry_indices if idx < first_entry_of_month]
        train_entry_positions = [idx_to_pos[idx] for idx in train_entry_idx_list if idx in idx_to_pos]
        train_entry_y = np.array([df.loc[idx, 'real_label'] for idx in train_entry_idx_list])
        valid_entry = ~np.isnan(train_entry_y)
        train_entry_y = train_entry_y[valid_entry]

        if len(train_X_ab) < 1000:
            log(f"    Insufficient all-bar training data ({len(train_X_ab)}), using baseline only")
            for idx in month_entries:
                pos = idx_to_pos.get(idx)
                if pos is None:
                    continue
                pnl_r = df.loc[idx, 'real_pnl_ratio']
                win = df.loc[idx, 'real_label']
                if np.isnan(pnl_r) or np.isnan(win):
                    continue
                ps = df.loc[idx, 'positioning_score']

                # Baseline trade
                b_risk = get_risk_amount(baseline_equity)
                b_risk = apply_streak_reduction(b_risk, baseline_consec_losses)
                if abs(ps) >= 1.5: b_risk *= 1.2
                elif abs(ps) >= 1.0: b_risk *= 1.1
                b_pnl = b_risk * pnl_r
                baseline_equity += b_pnl
                baseline_consec_losses = 0 if win == 1 else baseline_consec_losses + 1
                baseline_trades.append({
                    'timestamp': str(idx), 'price': round(float(df.loc[idx, 'close']), 1),
                    'pos_score': round(float(ps), 2), 'actual_win': int(win),
                    'pnl_ratio': round(float(pnl_r), 3), 'risk': round(float(b_risk), 2),
                    'trade_pnl': round(float(b_pnl), 2), 'equity': round(float(baseline_equity), 2),
                })

                # ML trade (no ML available, same as baseline)
                m_risk = get_risk_amount(ml_equity)
                m_risk = apply_streak_reduction(m_risk, ml_consec_losses)
                if abs(ps) >= 1.5: m_risk *= 1.2
                elif abs(ps) >= 1.0: m_risk *= 1.1
                m_pnl = m_risk * pnl_r
                ml_equity += m_pnl
                ml_consec_losses = 0 if win == 1 else ml_consec_losses + 1
                ml_trades.append({
                    'timestamp': str(idx), 'price': round(float(df.loc[idx, 'close']), 1),
                    'pos_score': round(float(ps), 2), 'actual_win': int(win),
                    'pnl_ratio': round(float(pnl_r), 3), 'risk': round(float(m_risk), 2),
                    'trade_pnl': round(float(m_pnl), 2), 'equity': round(float(ml_equity), 2),
                    'ml_action': 'no_model', 'regime': '', 'xgb_prob': 0, 'lgb_scale': 1.0, 'lstm_boost': 1.0,
                })
            continue

        # =====================
        # TRAIN ML MODELS (on all bars)
        # =====================
        scaler = StandardScaler()
        train_X_s = scaler.fit_transform(train_X_ab)

        # L1: RF Regime — trained on all bars
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                     max_features='sqrt', random_state=42, n_jobs=-1)
        rf.fit(train_X_s, train_reg_ab)

        # Determine favorable regimes using REAL entry win rates (not all-bar WR)
        # Use train entry signals to assess which regimes have above-baseline WR
        base_wr = train_entry_y.mean() if len(train_entry_y) > 0 else 0.3
        favorable = []

        if len(train_entry_positions) > 0 and len(valid_entry) > 0:
            # Predict regimes for training entry signals
            valid_entry_positions = [train_entry_positions[j] for j in range(len(train_entry_positions)) if j < len(valid_entry) and valid_entry[j]]
            if len(valid_entry_positions) > 0:
                entry_X_for_regime = X_all[valid_entry_positions]
                entry_X_for_regime_s = scaler.transform(entry_X_for_regime)
                entry_regime_preds = rf.predict(entry_X_for_regime_s)
                for rid in regime_names:
                    m = entry_regime_preds == rid
                    if m.sum() > 2 and train_entry_y[:len(m)][m].mean() > base_wr * 0.9:
                        favorable.append(rid)

        # Fallback: if no favorable regimes identified, use top regimes from all-bar analysis
        if not favorable:
            train_regime_preds = rf.predict(train_X_s)
            for rid in regime_names:
                m = train_regime_preds == rid
                if m.sum() > 50 and train_y_ab[m].mean() > train_y_ab.mean():
                    favorable.append(rid)

        # If STILL no favorable, mark all as favorable (skip regime gate)
        if not favorable:
            favorable = list(regime_names.keys())

        # L2: XGBoost Quality — trained on all bars
        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=10,
            reg_alpha=0.5, reg_lambda=2.0, random_state=42,
            verbosity=0, eval_metric='logloss',
        )
        xgb_model.fit(train_X_s, train_y_ab, verbose=False)
        # Threshold: calibrate on training data — we want to keep top ~70%
        train_xgb_probs = xgb_model.predict_proba(train_X_s)[:, 1]
        xgb_thresh = np.percentile(train_xgb_probs, 30)  # keep top 70%
        xgb_thresh = max(0.25, min(0.50, xgb_thresh))

        # L3: LightGBM Loss Predictor — trained on all bars
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, min_child_samples=10,
            reg_alpha=0.5, reg_lambda=2.0, random_state=42, verbose=-1,
        )
        train_y_loss = 1 - train_y_ab  # predict loss
        lgb_model.fit(train_X_s, train_y_loss)

        # L4: LSTM Ensemble — trained on recent all-bar data
        lstm_available = False
        device = torch.device('cpu')
        if len(X_seq_all) > 0:
            # Use last 30 days of training bars for LSTM (efficient + recent patterns)
            lstm_lookback = min(cutoff_pos, 96 * 30)  # 30 days of 15-min bars
            lstm_start = max(seq_len, cutoff_pos - lstm_lookback)
            lstm_positions = list(range(lstm_start, cutoff_pos))

            # Filter to valid labels
            lstm_valid_pos = [p for p in lstm_positions
                              if p >= seq_len
                              and (p - seq_len) < len(X_seq_all)
                              and p < len(allbar_labels)
                              and not np.isnan(allbar_labels[p])]

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

        log(f"    ML trained on {len(train_X_ab):,} all-bar samples, "
            f"favorable={[regime_names[r] for r in favorable]}, "
            f"xgb_thresh={xgb_thresh:.3f}, lstm={'Y' if lstm_available else 'N'}")

        # =====================
        # EVALUATE this month's real entry signals
        # =====================
        ml_rejected = 0
        ml_taken = 0

        for idx in month_entries:
            pos = idx_to_pos.get(idx)
            if pos is None:
                continue
            pnl_r = df.loc[idx, 'real_pnl_ratio']
            win = df.loc[idx, 'real_label']
            if np.isnan(pnl_r) or np.isnan(win):
                continue
            ps = df.loc[idx, 'positioning_score']
            close_price = df.loc[idx, 'close']

            # --- BASELINE TRADE (always taken) ---
            b_risk = get_risk_amount(baseline_equity)
            b_risk = apply_streak_reduction(b_risk, baseline_consec_losses)
            if abs(ps) >= 1.5: b_risk *= 1.2
            elif abs(ps) >= 1.0: b_risk *= 1.1
            b_pnl = b_risk * pnl_r
            baseline_equity += b_pnl
            baseline_consec_losses = 0 if win == 1 else baseline_consec_losses + 1
            baseline_trades.append({
                'timestamp': str(idx), 'price': round(float(close_price), 1),
                'pos_score': round(float(ps), 2), 'actual_win': int(win),
                'pnl_ratio': round(float(pnl_r), 3), 'risk': round(float(b_risk), 2),
                'trade_pnl': round(float(b_pnl), 2), 'equity': round(float(baseline_equity), 2),
            })

            # --- ML-ENHANCED TRADE ---
            test_x = X_all[pos:pos+1]
            test_x_s = scaler.transform(test_x)

            # L1: Regime gate
            regime_pred = rf.predict(test_x_s)[0]
            regime_name = regime_names.get(int(regime_pred), '?')
            regime_pass = int(regime_pred) in favorable

            # L2: XGBoost quality
            xgb_prob = xgb_model.predict_proba(test_x_s)[0, 1]
            xgb_pass = xgb_prob >= xgb_thresh

            if not regime_pass or not xgb_pass:
                ml_rejected += 1
                ml_trades.append({
                    'timestamp': str(idx), 'price': round(float(close_price), 1),
                    'pos_score': round(float(ps), 2), 'actual_win': int(win),
                    'pnl_ratio': round(float(pnl_r), 3), 'risk': 0,
                    'trade_pnl': 0, 'equity': round(float(ml_equity), 2),
                    'ml_action': 'REJECTED',
                    'reject_reason': f"{'regime(' + regime_name + ')' if not regime_pass else 'xgb(' + str(round(xgb_prob, 3)) + ')'}",
                    'regime': regime_name, 'xgb_prob': round(float(xgb_prob), 4),
                    'lgb_scale': 0, 'lstm_boost': 0,
                })
                continue

            # L3: LightGBM sizing
            loss_prob = lgb_model.predict_proba(test_x_s)[0, 1]
            lgb_scale = max(0.3, 1.0 - loss_prob * 0.6)

            # L4: LSTM boost
            boost = 1.0
            if lstm_available and pos >= seq_len and (pos - seq_len) < len(X_seq_all):
                with torch.no_grad():
                    t_s = torch.FloatTensor(X_seq_all[pos - seq_len:pos - seq_len + 1]).to(device)
                    _, feat = lstm_model(t_s)
                    feat_np = feat.cpu().numpy()
                    t_flat = X_scaled_all[pos:pos+1]
                    ens_input = np.hstack([t_flat, feat_np])
                    ens_prob = ens_xgb.predict_proba(ens_input)[0, 1]
                    if ens_prob > 0.5:
                        boost = 1.0 + (ens_prob - 0.5) * 0.4
                    else:
                        boost = 0.7 + ens_prob * 0.6

            final_scale = max(0.2, min(1.5, lgb_scale * boost))

            # ML-adjusted risk
            m_risk = get_risk_amount(ml_equity)
            m_risk = apply_streak_reduction(m_risk, ml_consec_losses)
            if abs(ps) >= 1.5: m_risk *= 1.2
            elif abs(ps) >= 1.0: m_risk *= 1.1
            m_risk *= final_scale

            m_pnl = m_risk * pnl_r
            ml_equity += m_pnl
            ml_consec_losses = 0 if win == 1 else ml_consec_losses + 1
            ml_taken += 1

            ml_trades.append({
                'timestamp': str(idx), 'price': round(float(close_price), 1),
                'pos_score': round(float(ps), 2), 'actual_win': int(win),
                'pnl_ratio': round(float(pnl_r), 3), 'risk': round(float(m_risk), 2),
                'trade_pnl': round(float(m_pnl), 2), 'equity': round(float(ml_equity), 2),
                'ml_action': 'TAKEN',
                'regime': regime_name, 'xgb_prob': round(float(xgb_prob), 4),
                'lgb_scale': round(float(lgb_scale), 3), 'lstm_boost': round(float(boost), 3),
                'final_scale': round(float(final_scale), 3),
            })

        log(f"    Taken: {ml_taken}, Rejected: {ml_rejected}")

    # ============================================================
    # RESULTS
    # ============================================================
    log(f"\n{'='*70}")
    log(f"RESULTS: BASELINE vs ML-ENHANCED (Expanded Training)")
    log(f"{'='*70}")

    # Baseline stats
    b_total = len(baseline_trades)
    b_wins = sum(1 for t in baseline_trades if t['actual_win'] == 1)
    b_pnl = sum(t['trade_pnl'] for t in baseline_trades)
    b_eq = [100000.0] + [t['equity'] for t in baseline_trades]
    b_arr = np.array(b_eq)
    b_peak = np.maximum.accumulate(b_arr)
    b_dd = ((b_peak - b_arr) / np.maximum(b_peak, 1) * 100).max()

    # ML stats
    ml_taken_trades = [t for t in ml_trades if t.get('ml_action') == 'TAKEN']
    ml_rejected_trades = [t for t in ml_trades if t.get('ml_action') == 'REJECTED']
    m_total = len(ml_taken_trades)
    m_wins = sum(1 for t in ml_taken_trades if t['actual_win'] == 1)
    m_pnl = sum(t['trade_pnl'] for t in ml_taken_trades)

    # ML equity curve
    m_eq = [100000.0]
    for t in ml_trades:
        if t.get('ml_action') == 'TAKEN':
            m_eq.append(m_eq[-1] + t['trade_pnl'])
    m_arr = np.array(m_eq)
    m_peak = np.maximum.accumulate(m_arr)
    m_dd = ((m_peak - m_arr) / np.maximum(m_peak, 1) * 100).max()

    # Rejected trade analysis
    rej_wins = sum(1 for t in ml_rejected_trades if t['actual_win'] == 1)
    rej_losses = len(ml_rejected_trades) - rej_wins

    # Profit factors
    b_gross_win = sum(t['trade_pnl'] for t in baseline_trades if t['actual_win'] == 1)
    b_gross_loss = abs(sum(t['trade_pnl'] for t in baseline_trades if t['actual_win'] == 0))
    b_pf = b_gross_win / b_gross_loss if b_gross_loss > 0 else 0

    m_gross_win = sum(t['trade_pnl'] for t in ml_taken_trades if t['actual_win'] == 1)
    m_gross_loss = abs(sum(t['trade_pnl'] for t in ml_taken_trades if t['actual_win'] == 0))
    m_pf = m_gross_win / m_gross_loss if m_gross_loss > 0 else 0

    # Streaks
    def max_loss_streak(trades):
        ms = cs = 0
        for t in trades:
            if t['actual_win'] == 0: cs += 1; ms = max(ms, cs)
            else: cs = 0
        return ms

    # Risk-adjusted return
    b_return_pct = (b_arr[-1] - 100000) / 100000 * 100
    m_return_pct = (m_arr[-1] - 100000) / 100000 * 100
    b_calmar = b_return_pct / b_dd if b_dd > 0 else 0
    m_calmar = m_return_pct / m_dd if m_dd > 0 else 0

    log(f"")
    log(f"  {'':35} {'BASELINE':>12}   {'ML-ENHANCED':>12}   {'DELTA':>10}")
    log(f"  {'─'*80}")
    log(f"  {'Total Trades':35} {b_total:>12}   {m_total:>12}   {m_total - b_total:>+10}")
    log(f"  {'Wins':35} {b_wins:>12}   {m_wins:>12}   {m_wins - b_wins:>+10}")
    log(f"  {'Win Rate':35} {b_wins/b_total*100 if b_total else 0:>11.1f}%   {m_wins/m_total*100 if m_total else 0:>11.1f}%   {(m_wins/m_total*100 if m_total else 0) - (b_wins/b_total*100 if b_total else 0):>+9.1f}%")
    log(f"  {'Total PnL':35} ${b_pnl:>11,.0f}   ${m_pnl:>11,.0f}   ${m_pnl - b_pnl:>+9,.0f}")
    log(f"  {'Final Equity':35} ${b_arr[-1]:>11,.0f}   ${m_arr[-1]:>11,.0f}   ${m_arr[-1] - b_arr[-1]:>+9,.0f}")
    log(f"  {'Return %':35} {b_return_pct:>11.1f}%   {m_return_pct:>11.1f}%   {m_return_pct - b_return_pct:>+9.1f}%")
    log(f"  {'Max Drawdown':35} {b_dd:>11.1f}%   {m_dd:>11.1f}%   {m_dd - b_dd:>+9.1f}%")
    log(f"  {'Profit Factor':35} {b_pf:>12.2f}   {m_pf:>12.2f}   {m_pf - b_pf:>+10.2f}")
    log(f"  {'Calmar Ratio (Return/MaxDD)':35} {b_calmar:>12.2f}   {m_calmar:>12.2f}   {m_calmar - b_calmar:>+10.2f}")
    log(f"  {'Max Loss Streak':35} {max_loss_streak(baseline_trades):>12}   {max_loss_streak(ml_taken_trades):>12}")
    log(f"")
    log(f"  ML FILTERING ANALYSIS:")
    log(f"  {'Trades rejected by ML':35} {len(ml_rejected_trades):>12}")
    log(f"  {'  - Would have been wins':35} {rej_wins:>12}")
    log(f"  {'  - Would have been losses':35} {rej_losses:>12}")
    if ml_rejected_trades:
        log(f"  {'Rejection accuracy':35} {rej_losses/len(ml_rejected_trades)*100:>11.1f}%")
    log(f"  {'  Rejected by regime gate':35} {sum(1 for t in ml_rejected_trades if 'regime' in t.get('reject_reason', '')):>12}")
    log(f"  {'  Rejected by XGBoost quality':35} {sum(1 for t in ml_rejected_trades if 'xgb' in t.get('reject_reason', '')):>12}")

    # ML sizing analysis (for taken trades)
    if ml_taken_trades:
        scales = [t.get('final_scale', 1.0) for t in ml_taken_trades]
        win_scales = [t.get('final_scale', 1.0) for t in ml_taken_trades if t['actual_win'] == 1]
        loss_scales = [t.get('final_scale', 1.0) for t in ml_taken_trades if t['actual_win'] == 0]
        log(f"")
        log(f"  ML SIZING ANALYSIS (taken trades):")
        log(f"  {'Avg final_scale (all)':35} {np.mean(scales):>12.3f}")
        log(f"  {'Avg final_scale (wins)':35} {np.mean(win_scales) if win_scales else 0:>12.3f}")
        log(f"  {'Avg final_scale (losses)':35} {np.mean(loss_scales) if loss_scales else 0:>12.3f}")
        log(f"  {'Scale differentiation (W-L)':35} {(np.mean(win_scales) if win_scales else 0) - (np.mean(loss_scales) if loss_scales else 0):>+12.3f}")

    # Monthly comparison
    log(f"\n  MONTHLY COMPARISON:")
    log(f"  {'Month':>7}  {'B_Trds':>6}  {'B_WR':>5}  {'B_PnL':>12}  {'M_Trds':>6}  {'M_WR':>5}  {'M_PnL':>12}  {'Rej':>4}  {'RejAcc':>6}")
    log(f"  {'─'*80}")

    b_monthly = {}
    for t in baseline_trades:
        m = t['timestamp'][:7]
        if m not in b_monthly: b_monthly[m] = {'trades': 0, 'wins': 0, 'pnl': 0}
        b_monthly[m]['trades'] += 1
        b_monthly[m]['wins'] += t['actual_win']
        b_monthly[m]['pnl'] += t['trade_pnl']

    m_monthly = {}
    for t in ml_taken_trades:
        m = t['timestamp'][:7]
        if m not in m_monthly: m_monthly[m] = {'trades': 0, 'wins': 0, 'pnl': 0}
        m_monthly[m]['trades'] += 1
        m_monthly[m]['wins'] += t['actual_win']
        m_monthly[m]['pnl'] += t['trade_pnl']

    r_monthly = {}
    for t in ml_rejected_trades:
        m = t['timestamp'][:7]
        if m not in r_monthly: r_monthly[m] = {'total': 0, 'losses': 0}
        r_monthly[m]['total'] += 1
        if t['actual_win'] == 0:
            r_monthly[m]['losses'] += 1

    all_months = sorted(set(list(b_monthly.keys()) + list(m_monthly.keys())))
    for mo in all_months:
        b = b_monthly.get(mo, {'trades': 0, 'wins': 0, 'pnl': 0})
        m = m_monthly.get(mo, {'trades': 0, 'wins': 0, 'pnl': 0})
        r = r_monthly.get(mo, {'total': 0, 'losses': 0})
        b_wr = f"{b['wins']/b['trades']*100:.0f}%" if b['trades'] > 0 else "  -"
        m_wr = f"{m['wins']/m['trades']*100:.0f}%" if m['trades'] > 0 else "  -"
        r_acc = f"{r['losses']/r['total']*100:.0f}%" if r['total'] > 0 else "  -"
        log(f"  {mo:>7}  {b['trades']:>6}  {b_wr:>5}  ${b['pnl']:>11,.0f}  {m['trades']:>6}  {m_wr:>5}  ${m['pnl']:>11,.0f}  {r['total']:>4}  {r_acc:>6}")

    # Per-trade log
    log(f"\n  ML TRADE LOG (all entries):")
    log(f"  {'#':>3}  {'Timestamp':>19}  {'Price':>10}  {'PosScr':>7}  {'Win':>4}  {'B_PnL':>10}  {'M_Action':>10}  {'Regime':>15}  {'XGB_P':>6}  {'Scale':>6}  {'M_PnL':>10}")
    log(f"  {'─'*130}")
    for i, (bt, mt) in enumerate(zip(baseline_trades, ml_trades)):
        action = mt.get('ml_action', '?')
        regime = mt.get('regime', '')
        xgb_p = mt.get('xgb_prob', 0)
        scale = mt.get('final_scale', mt.get('lgb_scale', 0))
        m_pnl_val = mt.get('trade_pnl', 0)
        rej_reason = mt.get('reject_reason', '')
        action_str = action
        if action == 'REJECTED':
            action_str = f"REJ({rej_reason[:8]})"
        log(f"  {i+1:>3}  {bt['timestamp'][:19]:>19}  ${bt['price']:>9,.1f}  {bt['pos_score']:>7.2f}  "
            f"{'W' if bt['actual_win'] else 'L':>4}  ${bt['trade_pnl']:>9,.0f}  {action_str:>10}  {regime:>15}  "
            f"{xgb_p:>6.3f}  {scale:>6.3f}  ${m_pnl_val:>9,.0f}")

    # Save
    results = {
        'pipeline': 'ML-Enhanced on Real Strategy Signals (Expanded All-Bar Training)',
        'training_approach': 'All bars with hypothetical trade labels (generate_trade_labels)',
        'data_range': f"{df.index.min()} to {df.index.max()}",
        'allbar_training_samples': int(len(train_X_ab)),
        'baseline': {
            'trades': b_total, 'wins': b_wins,
            'win_rate': round(b_wins/b_total*100, 1) if b_total else 0,
            'total_pnl': round(b_pnl, 2), 'final_equity': round(float(b_arr[-1]), 2),
            'return_pct': round(b_return_pct, 2),
            'max_dd': round(b_dd, 2), 'profit_factor': round(b_pf, 2),
            'calmar_ratio': round(b_calmar, 2),
        },
        'ml_enhanced': {
            'trades_taken': m_total, 'wins': m_wins,
            'trades_rejected': len(ml_rejected_trades),
            'rejected_would_win': rej_wins, 'rejected_would_lose': rej_losses,
            'rejection_accuracy': round(rej_losses/len(ml_rejected_trades)*100, 1) if ml_rejected_trades else 0,
            'win_rate': round(m_wins/m_total*100, 1) if m_total else 0,
            'total_pnl': round(m_pnl, 2), 'final_equity': round(float(m_arr[-1]), 2),
            'return_pct': round(m_return_pct, 2),
            'max_dd': round(m_dd, 2), 'profit_factor': round(m_pf, 2),
            'calmar_ratio': round(m_calmar, 2),
        },
        'baseline_trades': baseline_trades,
        'ml_trades': ml_trades,
    }

    out_file = os.path.join(SCRIPT_DIR, 'integrated_v2_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\nResults saved: {out_file}")


if __name__ == "__main__":
    run_pipeline()
