#!/usr/bin/env python3
"""
BTC Strategy Dashboard - Asian Hours Version
=============================================
Streamlit dashboard for monitoring BTC trading strategies (Asian Hours version).

Features:
- Real-time indicator scores (price, volume, positioning, technical)
- BTC price history and volume charts
- Current trade status and entry conditions
- Stop loss levels for active positions
- Data refresh functionality
- Asian session time indicator (0-11 UTC)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timezone
import sys

# Add parent directory to path for data loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backtest_15min'))

# Page config
st.set_page_config(
    page_title="BTC Strategy Dashboard - Asian Hours",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results_asian_hours')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'binance-futures-data')
INITIAL_CAPITAL = 100000
BASE_RISK = 6000

# Entry hours - Asian session only
ENTRY_HRS_ASIAN = set(range(0, 12))

# Indicator periods (optimized for Asian hours)
ATR_PERIOD = 14
RSI_PERIOD = 56
SMA_PERIOD = 200
BB_PERIOD = 80
VOLUME_MA_PERIOD = 96

# Positioning thresholds
TOP_TRADER_STRONG = 0.60
TOP_TRADER_MODERATE = 0.55

# Strategy configurations
STRATEGIES = {
    "Baseline_AHours": {
        "description": "Max Returns - Positioning exit only",
        "config": {
            "pos_exit_enabled": True,
            "pos_exit_threshold": 0.0,
            "multi_tp_enabled": False,
        }
    },
    "PosVol_Combined_AHours": {
        "description": "Position + Volume collapse protection",
        "config": {
            "pos_exit_enabled": True,
            "pos_exit_threshold": 0.0,
            "pos_decline_be_enabled": True,
            "vol_collapse_enabled": True,
            "multi_tp_enabled": False,
        }
    },
    "MultiTP_30_AHours": {
        "description": "Multi-TP with 30% partial exit",
        "config": {
            "pos_exit_enabled": True,
            "multi_tp_enabled": True,
            "tp1_atr_mult": 2.0,
            "tp1_exit_pct": 0.3,
        }
    },
    "VolFilter_Adaptive_AHours": {
        "description": "Volume filter + adaptive sizing",
        "config": {
            "vol_min_for_entry": 0.8,
            "pos_exit_enabled": True,
            "vol_collapse_enabled": True,
            "vol_size_adjust": True,
        }
    },
    "Conservative_AHours": {
        "description": "Conservative multi-TP + BE protection",
        "config": {
            "vol_min_for_entry": 0.7,
            "pos_exit_enabled": True,
            "multi_tp_enabled": True,
            "tp1_atr_mult": 2.0,
            "tp1_exit_pct": 0.4,
            "pos_decline_be_enabled": True,
        }
    },
}

# Default parameters
DEFAULT_PARAMS = {
    "stop_atr_mult": 1.8,
    "tp_atr_mult_base": 4.5,
    "min_pos_long": 0.4,
    "rsi_long_range": [20, 45],
    "vol_min_for_entry": 0.0,
    "pullback_range": [0.5, 3.0],
}


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load and prepare data with incremental updates"""
    try:
        from load_15min_data import merge_all_data_15min
        df = merge_all_data_15min()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def compute_indicators(df):
    """Compute technical indicators"""
    df = df.copy()

    # Price indicators
    df['sma20'] = df['close'].rolling(BB_PERIOD).mean()
    df['sma200'] = df['close'].rolling(SMA_PERIOD).mean()
    df['std20'] = df['close'].rolling(BB_PERIOD).std()
    df['upper_band'] = df['sma20'] + 2 * df['std20']
    df['lower_band'] = df['sma20'] - 2 * df['std20']

    # ATR (using optimized period for Asian hours)
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
    df['rsi'] = 100 - 100 / (1 + gain / loss)

    # Trend and pullback
    df['uptrend'] = df['close'] > df['sma200']
    df['pullback_pct'] = (df['high'].rolling(16).max() - df['close']) / df['close'] * 100

    # Volume indicators
    df['vol_ma'] = df['volume'].rolling(VOLUME_MA_PERIOD).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma']
    df['vol_ma_4h'] = df['volume'].rolling(16).mean()
    df['vol_trend'] = df['vol_ma_4h'] - df['vol_ma_4h'].shift(4)
    df['vol_increasing'] = df['vol_trend'] > 0

    # Volume-price relationship
    df['price_change'] = df['close'].pct_change()
    df['bullish_volume'] = (df['price_change'] > 0) & (df['vol_ratio'] > 1.0)

    return df


def calculate_positioning_score(row) -> float:
    """Calculate positioning score"""
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
    """Calculate volume quality score"""
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


def check_entry_conditions(row, strategy_config):
    """Check if entry conditions are met for a strategy"""
    params = {**DEFAULT_PARAMS, **strategy_config}

    conditions = {}

    # Calculate scores
    pos_score = calculate_positioning_score(row)
    vol_score = calculate_volume_score(row)

    # Check Asian hours
    current_hour = row.name.hour if hasattr(row, 'name') and hasattr(row.name, 'hour') else datetime.now(timezone.utc).hour
    conditions['asian_hours'] = {
        'value': current_hour,
        'required': '0-11 UTC',
        'met': current_hour in ENTRY_HRS_ASIAN
    }

    # Check each condition
    conditions['positioning_score'] = {
        'value': pos_score,
        'required': f">= {params['min_pos_long']}",
        'met': pos_score >= params['min_pos_long']
    }

    conditions['positioning_min'] = {
        'value': abs(pos_score),
        'required': '>= 0.15',
        'met': abs(pos_score) >= 0.15
    }

    rsi = row.get('rsi', 50)
    conditions['rsi'] = {
        'value': rsi,
        'required': f"{params['rsi_long_range'][0]} - {params['rsi_long_range'][1]}",
        'met': params['rsi_long_range'][0] < rsi < params['rsi_long_range'][1] if not pd.isna(rsi) else False
    }

    uptrend = row.get('uptrend', False)
    conditions['uptrend'] = {
        'value': uptrend,
        'required': 'True',
        'met': uptrend
    }

    pullback = row.get('pullback_pct', 0)
    conditions['pullback'] = {
        'value': pullback,
        'required': f"{params['pullback_range'][0]}% - {params['pullback_range'][1]}%",
        'met': params['pullback_range'][0] < pullback < params['pullback_range'][1] if not pd.isna(pullback) else False
    }

    vol_ratio = row.get('vol_ratio', 1.0)
    vol_min = params.get('vol_min_for_entry', 0.0)
    conditions['volume_ratio'] = {
        'value': vol_ratio,
        'required': f">= {vol_min}",
        'met': vol_ratio >= vol_min if not pd.isna(vol_ratio) else False
    }

    # All conditions met?
    all_met = all(c['met'] for c in conditions.values())

    return conditions, all_met


def calculate_trade_levels(row, strategy_config):
    """Calculate entry, stop loss, and take profit levels"""
    params = {**DEFAULT_PARAMS, **strategy_config}

    entry_price = row['close']
    atr = row['atr']

    stop_distance = atr * params['stop_atr_mult']
    stop_price = entry_price - stop_distance

    # TP calculation with positioning adjustment
    pos_score = calculate_positioning_score(row)
    tp_mult = params['tp_atr_mult_base']
    if abs(pos_score) >= 1.5:
        tp_mult *= 1.3
    elif abs(pos_score) >= 1.0:
        tp_mult *= 1.15

    tp_price = entry_price + atr * tp_mult

    # Multi-TP level if enabled
    tp1_price = None
    if params.get('multi_tp_enabled', False):
        tp1_atr_mult = params.get('tp1_atr_mult', 2.5)
        tp1_price = entry_price + atr * tp1_atr_mult

    return {
        'entry': entry_price,
        'stop_loss': stop_price,
        'take_profit': tp_price,
        'tp1': tp1_price,
        'risk_reward': (tp_price - entry_price) / (entry_price - stop_price) if entry_price != stop_price else 0,
    }


def load_strategy_stats():
    """Load strategy statistics"""
    stats_file = os.path.join(RESULTS_DIR, 'strategy_statistics.json')
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)
    return {}


def get_session_status():
    """Get current trading session status"""
    now = datetime.now(timezone.utc)
    current_hour = now.hour

    if current_hour in ENTRY_HRS_ASIAN:
        return "🟢 ACTIVE", "Asian Session (Trading Window Open)", current_hour, 12 - current_hour
    else:
        hours_until = (24 - current_hour) if current_hour >= 12 else 0
        return "🔴 INACTIVE", "Non-Asian Session (No New Entries)", current_hour, hours_until


def main():
    # Header
    st.title("🌏 BTC Strategy Dashboard - Asian Hours")
    st.markdown("*Real-time monitoring for Asian Hours trading strategies (0-11 UTC only)*")

    # Refresh button and session status
    col1, col2, col3, col4 = st.columns([1, 1, 2, 2])
    with col1:
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
    with col2:
        st.write(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

    # Session status
    session_icon, session_text, current_hour, hours_remaining = get_session_status()
    with col3:
        st.metric("Session Status", session_icon)
    with col4:
        if hours_remaining > 0:
            st.metric("Current UTC Hour", f"{current_hour}:00",
                      f"{hours_remaining}h until session {'ends' if 'ACTIVE' in session_icon else 'starts'}")
        else:
            st.metric("Current UTC Hour", f"{current_hour}:00")

    # Session alert
    if "ACTIVE" in session_icon:
        st.success(f"🟢 **{session_text}** - New entries allowed. {hours_remaining} hours remaining in session.")
    else:
        st.warning(f"🔴 **{session_text}** - Waiting for Asian session to start.")

    # Load data
    with st.spinner("Loading data..."):
        df = load_data()

    if df is None or len(df) == 0:
        st.error("Failed to load data")
        return

    # Compute indicators
    df = compute_indicators(df)

    # Get latest row
    latest = df.iloc[-1]
    latest_time = df.index[-1]

    # Calculate current scores
    pos_score = calculate_positioning_score(latest)
    vol_score = calculate_volume_score(latest)

    # Top metrics row
    st.markdown("---")
    st.subheader("📊 Current Indicators")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("BTC Price", f"${latest['close']:,.0f}",
                  f"{latest['price_change']*100:.2f}%" if not pd.isna(latest.get('price_change')) else "N/A")

    with col2:
        st.metric("RSI (56)", f"{latest['rsi']:.1f}" if not pd.isna(latest.get('rsi')) else "N/A",
                  "Oversold" if latest.get('rsi', 50) < 30 else ("Overbought" if latest.get('rsi', 50) > 70 else "Neutral"))

    with col3:
        st.metric("Volume Ratio", f"{latest['vol_ratio']:.2f}" if not pd.isna(latest.get('vol_ratio')) else "N/A",
                  "High" if latest.get('vol_ratio', 1) > 1.2 else ("Low" if latest.get('vol_ratio', 1) < 0.8 else "Normal"))

    with col4:
        trend_status = "Uptrend ↑" if latest.get('uptrend', False) else "Downtrend ↓"
        st.metric("Trend (SMA200)", trend_status)

    with col5:
        st.metric("Positioning Score", f"{pos_score:.2f}",
                  "Bullish" if pos_score > 0.5 else ("Bearish" if pos_score < -0.5 else "Neutral"))

    with col6:
        st.metric("Volume Score", f"{vol_score:.2f}",
                  "Strong" if vol_score > 1.5 else ("Weak" if vol_score < 0.5 else "Moderate"))

    # Detailed positioning breakdown
    st.markdown("---")
    st.subheader("🎯 Positioning Details")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        top_long = latest.get('top_trader_position_long_pct', 0)
        st.metric("Top Traders Long %", f"{top_long*100:.1f}%" if not pd.isna(top_long) else "N/A")

    with col2:
        top_short = latest.get('top_trader_position_short_pct', 0)
        st.metric("Top Traders Short %", f"{top_short*100:.1f}%" if not pd.isna(top_short) else "N/A")

    with col3:
        global_ls = latest.get('global_ls_ratio', 1)
        st.metric("Global L/S Ratio", f"{global_ls:.3f}" if not pd.isna(global_ls) else "N/A")

    with col4:
        funding = latest.get('funding_rate', 0)
        st.metric("Funding Rate", f"{funding*100:.4f}%" if not pd.isna(funding) else "N/A",
                  "Negative (Bullish)" if funding < 0 else "Positive (Bearish)")

    # Price and Volume Chart
    st.markdown("---")
    st.subheader("📈 Price & Volume History")

    # Time range selector
    time_range = st.selectbox("Time Range", ["1 Day", "3 Days", "1 Week", "1 Month", "3 Months"], index=2)

    range_map = {
        "1 Day": 96,
        "3 Days": 288,
        "1 Week": 672,
        "1 Month": 2880,
        "3 Months": 8640,
    }

    df_chart = df.tail(range_map.get(time_range, 672))

    # Create price chart with indicators
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("BTC Price with Bollinger Bands", "Volume", "RSI"))

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_chart.index,
        open=df_chart['open'],
        high=df_chart['high'],
        low=df_chart['low'],
        close=df_chart['close'],
        name="BTC"
    ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['upper_band'],
                             line=dict(color='rgba(128,128,128,0.5)', width=1),
                             name='Upper BB'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['lower_band'],
                             line=dict(color='rgba(128,128,128,0.5)', width=1),
                             fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                             name='Lower BB'), row=1, col=1)

    # SMA200
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['sma200'],
                             line=dict(color='orange', width=1),
                             name='SMA 200'), row=1, col=1)

    # Highlight Asian hours on chart
    for idx in df_chart.index:
        if idx.hour in ENTRY_HRS_ASIAN:
            fig.add_vrect(x0=idx, x1=idx + pd.Timedelta(minutes=15),
                          fillcolor="rgba(0,255,0,0.05)", line_width=0,
                          row=1, col=1)

    # Volume
    colors = ['green' if c > o else 'red' for c, o in zip(df_chart['close'], df_chart['open'])]
    fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['volume'],
                         marker_color=colors, name='Volume'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['vol_ma'],
                             line=dict(color='yellow', width=1),
                             name='Vol MA'), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['rsi'],
                             line=dict(color='purple', width=1),
                             name='RSI'), row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=45, line_dash="dot", line_color="gray", row=3, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="gray", row=3, col=1)

    fig.update_layout(
        height=700,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Strategy Entry Conditions
    st.markdown("---")
    st.subheader("🎲 Strategy Entry Conditions")

    strategy_stats = load_strategy_stats()

    tabs = st.tabs(list(STRATEGIES.keys()))

    for tab, (strategy_name, strategy_info) in zip(tabs, STRATEGIES.items()):
        with tab:
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"**{strategy_info['description']}**")

                # Check conditions
                conditions, all_met = check_entry_conditions(latest, strategy_info['config'])

                st.markdown("##### Entry Conditions")
                for cond_name, cond_data in conditions.items():
                    icon = "✅" if cond_data['met'] else "❌"
                    value_str = f"{cond_data['value']:.2f}" if isinstance(cond_data['value'], (int, float)) else str(cond_data['value'])
                    st.write(f"{icon} **{cond_name}**: {value_str} (required: {cond_data['required']})")

                if all_met:
                    st.success("✅ All conditions met - ENTRY SIGNAL!")
                else:
                    st.warning("⏳ Waiting for conditions...")

            with col2:
                # Trade levels
                levels = calculate_trade_levels(latest, strategy_info['config'])

                st.markdown("##### Trade Levels (if entering now)")
                st.write(f"📍 **Entry**: ${levels['entry']:,.0f}")
                st.write(f"🛑 **Stop Loss**: ${levels['stop_loss']:,.0f} ({(levels['entry']-levels['stop_loss'])/levels['entry']*100:.2f}%)")
                st.write(f"🎯 **Take Profit**: ${levels['take_profit']:,.0f} ({(levels['take_profit']-levels['entry'])/levels['entry']*100:.2f}%)")
                if levels['tp1']:
                    st.write(f"🎯 **TP1 (Partial)**: ${levels['tp1']:,.0f}")
                st.write(f"📊 **Risk/Reward**: {levels['risk_reward']:.2f}")

                # Historical stats
                if strategy_name in strategy_stats:
                    stats = strategy_stats[strategy_name]
                    st.markdown("##### Historical Performance")
                    st.write(f"Total Return: {stats.get('total_return_pct', 0):.1f}%")
                    st.write(f"Win Rate: {stats.get('win_rate_pct', 0):.1f}%")
                    st.write(f"Max Drawdown: {stats.get('max_drawdown_pct', 0):.1f}%")
                    st.write(f"Max Losing Streak: {stats.get('max_losing_streak', 0)}")

    # Indicator History Charts
    st.markdown("---")
    st.subheader("📉 Indicator Score History")

    # Calculate historical scores
    df_scores = df.tail(range_map.get(time_range, 672)).copy()

    pos_scores = []
    vol_scores = []
    for idx, row in df_scores.iterrows():
        pos_scores.append(calculate_positioning_score(row))
        vol_scores.append(calculate_volume_score(row))

    df_scores['pos_score'] = pos_scores
    df_scores['vol_score'] = vol_scores

    # Score chart
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         vertical_spacing=0.1,
                         subplot_titles=("Positioning Score", "Volume Score"))

    fig2.add_trace(go.Scatter(x=df_scores.index, y=df_scores['pos_score'],
                              line=dict(color='cyan', width=1),
                              fill='tozeroy', fillcolor='rgba(0,255,255,0.2)',
                              name='Positioning'), row=1, col=1)
    fig2.add_hline(y=0.4, line_dash="dash", line_color="green", row=1, col=1)
    fig2.add_hline(y=0, line_dash="solid", line_color="gray", row=1, col=1)

    fig2.add_trace(go.Scatter(x=df_scores.index, y=df_scores['vol_score'],
                              line=dict(color='yellow', width=1),
                              fill='tozeroy', fillcolor='rgba(255,255,0,0.2)',
                              name='Volume'), row=2, col=1)
    fig2.add_hline(y=1.0, line_dash="dash", line_color="green", row=2, col=1)

    fig2.update_layout(
        height=400,
        showlegend=False,
        template="plotly_dark",
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption(f"Data as of: {latest_time} | Total bars: {len(df):,} | Dashboard: Asian Hours Version (0-11 UTC)")


if __name__ == "__main__":
    main()
