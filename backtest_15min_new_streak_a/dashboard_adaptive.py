#!/usr/bin/env python3
"""
BTC Adaptive Streak Reduction Dashboard
========================================
Streamlit dashboard for monitoring adaptive BTC trading strategies.

Features:
- Real-time indicator scores (positioning, ADX, volume)
- ADX market regime indicator (trending vs ranging)
- Progressive positioning threshold display
- Cooldown status tracking
- Entry condition monitoring for all adaptive strategies
- BTC price charts with ADX overlay
- Performance comparison across strategies
- AUTO-REFRESH every 5 minutes

Run with:
    streamlit run backtest_15min_new_streak_a/dashboard_adaptive.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timezone, timedelta
import sys
import time

# Add parent directory to path for data loading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backtest_15min'))

# Page config
st.set_page_config(
    page_title="BTC Adaptive Strategy Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh configuration
AUTO_REFRESH_INTERVAL = 300  # 5 minutes in seconds

# Constants
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'binance-futures-data', 'data')
INITIAL_CAPITAL = 100000

# Entry hours - Asian session only
ASIAN_HOURS = set(range(0, 12))

# Indicator periods
ATR_PERIOD = 14
RSI_PERIOD = 56
SMA_PERIOD = 200
ADX_PERIOD = 14
BB_PERIOD = 80
VOLUME_MA_PERIOD = 96

# Positioning thresholds
TOP_TRADER_STRONG = 0.60
TOP_TRADER_MODERATE = 0.55

# Tiered risk configuration
RISK_TIERS = [
    (0, 150000, 5000),
    (150000, 225000, 10500),
    (225000, 337500, 14625),
    (337500, 507000, 20250),
    (507000, 760000, 28000),
    (760000, 1200000, 38000),
    (1200000, float('inf'), 54000),
]

# Progressive positioning thresholds
PROGRESSIVE_POS_THRESHOLD = {
    0: 0.4,
    1: 0.5,
    2: 0.6,
    3: 0.8,
    4: 1.0,
}

# Adaptive strategy configurations
ADAPTIVE_STRATEGIES = {
    "Adaptive_Baseline": {
        "description": "Full adaptive: ADX + Progressive Pos + Cooldown",
        "config": {
            "adx_filter": True,
            "progressive_pos": True,
            "cooldown_enabled": True,
            "adx_threshold": 20,
        },
        "color": "#2ecc71"
    },
    "Adaptive_Conservative": {
        "description": "Full adaptive + Multi-TP + BE protection",
        "config": {
            "adx_filter": True,
            "progressive_pos": True,
            "cooldown_enabled": True,
            "multi_tp_enabled": True,
            "tp1_atr_mult": 2.0,
            "tp1_exit_pct": 0.4,
            "pos_decline_be_enabled": True,
        },
        "color": "#3498db"
    },
    "Adaptive_ADX_Only": {
        "description": "ADX market regime filter only",
        "config": {
            "adx_filter": True,
            "progressive_pos": False,
            "cooldown_enabled": False,
            "adx_threshold": 20,
        },
        "color": "#9b59b6"
    },
    "Adaptive_ProgPos_Only": {
        "description": "Progressive positioning + Cooldown only",
        "config": {
            "adx_filter": False,
            "progressive_pos": True,
            "cooldown_enabled": True,
        },
        "color": "#e74c3c"
    },
}

# Default parameters
DEFAULT_PARAMS = {
    "stop_atr_mult": 1.8,
    "tp_atr_mult_base": 4.5,
    "min_pos_long": 0.4,
    "rsi_long_range": [20, 45],
    "pullback_range": [0.5, 3.0],
    "adx_threshold": 20,
    "vol_min_trending": 0.7,
    "vol_min_ranging": 1.2,
    "cooldown_bars": 96,
    "cooldown_after_losses": 5,
}


def get_data_timestamp():
    """Get a timestamp key that changes every 5 minutes for cache invalidation."""
    now = datetime.now()
    # Round down to nearest 5 minutes
    minutes = (now.minute // 5) * 5
    return f"{now.year}{now.month}{now.day}{now.hour}{minutes}"


@st.cache_data(ttl=60)  # Short TTL, rely on timestamp key for refresh
def load_data(_timestamp_key: str):
    """Load and prepare data with incremental updates.

    Args:
        _timestamp_key: Cache key that changes every 5 minutes to force refresh
    """
    try:
        from load_15min_data import merge_all_data_15min
        df = merge_all_data_15min()
        return df
    except Exception as e:
        st.warning(f"Could not load from merge_all_data_15min: {e}")
        # Fallback to CSV
        try:
            price_file = os.path.join(DATA_DIR, 'price.csv')
            if os.path.exists(price_file):
                df = pd.read_csv(price_file)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                return df
        except Exception as e2:
            st.error(f"Error loading data: {e2}")
        return None


def compute_indicators(df):
    """Compute technical indicators including ADX"""
    df = df.copy()

    # Price indicators
    df['sma20'] = df['close'].rolling(BB_PERIOD).mean()
    df['sma200'] = df['close'].rolling(SMA_PERIOD).mean()
    df['std20'] = df['close'].rolling(BB_PERIOD).std()
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
    df['uptrend'] = df['close'] > df['sma200']
    df['pullback_pct'] = (df['high'].rolling(16).max() - df['close']) / df['close'] * 100

    # Volume indicators
    if 'volume' in df.columns:
        df['vol_ma'] = df['volume'].rolling(VOLUME_MA_PERIOD).mean()
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


def get_progressive_threshold(consecutive_losses: int) -> float:
    """Get adaptive positioning threshold based on losses"""
    if consecutive_losses in PROGRESSIVE_POS_THRESHOLD:
        return PROGRESSIVE_POS_THRESHOLD[consecutive_losses]
    elif consecutive_losses >= max(PROGRESSIVE_POS_THRESHOLD.keys()):
        return PROGRESSIVE_POS_THRESHOLD[max(PROGRESSIVE_POS_THRESHOLD.keys())]
    return 0.4


def check_entry_conditions(row, strategy_config, consecutive_losses=0):
    """Check if entry conditions are met for an adaptive strategy"""
    params = {**DEFAULT_PARAMS, **strategy_config}

    conditions = {}

    # Calculate scores
    pos_score = calculate_positioning_score(row)
    vol_score = calculate_volume_score(row)
    adx = row.get('adx', 0)
    is_trending = row.get('is_trending', False)

    # Check Asian hours
    current_hour = row.name.hour if hasattr(row, 'name') and hasattr(row.name, 'hour') else datetime.now(timezone.utc).hour
    conditions['asian_hours'] = {
        'value': current_hour,
        'required': '0-11 UTC',
        'met': current_hour in ASIAN_HOURS
    }

    # ADX filter
    if params.get('adx_filter', False):
        conditions['adx_filter'] = {
            'value': f"{adx:.1f}",
            'required': f">= {params.get('adx_threshold', 20)}",
            'met': adx >= params.get('adx_threshold', 20) if not pd.isna(adx) else False
        }

    # Progressive positioning
    if params.get('progressive_pos', False):
        min_pos = get_progressive_threshold(consecutive_losses)
        conditions['progressive_pos'] = {
            'value': f"{pos_score:.2f}",
            'required': f">= {min_pos:.2f} (losses: {consecutive_losses})",
            'met': pos_score >= min_pos
        }
    else:
        conditions['positioning_score'] = {
            'value': f"{pos_score:.2f}",
            'required': f">= {params['min_pos_long']}",
            'met': pos_score >= params['min_pos_long']
        }

    # Basic positioning minimum
    conditions['positioning_min'] = {
        'value': f"{abs(pos_score):.2f}",
        'required': '>= 0.15',
        'met': abs(pos_score) >= 0.15
    }

    # RSI
    rsi = row.get('rsi', 50)
    conditions['rsi'] = {
        'value': f"{rsi:.1f}" if not pd.isna(rsi) else "N/A",
        'required': f"{params['rsi_long_range'][0]} - {params['rsi_long_range'][1]}",
        'met': params['rsi_long_range'][0] < rsi < params['rsi_long_range'][1] if not pd.isna(rsi) else False
    }

    # Uptrend
    uptrend = row.get('uptrend', False)
    conditions['uptrend'] = {
        'value': "Yes" if uptrend else "No",
        'required': 'Yes',
        'met': uptrend
    }

    # Pullback
    pullback = row.get('pullback_pct', 0)
    conditions['pullback'] = {
        'value': f"{pullback:.2f}%" if not pd.isna(pullback) else "N/A",
        'required': f"{params['pullback_range'][0]}% - {params['pullback_range'][1]}%",
        'met': params['pullback_range'][0] < pullback < params['pullback_range'][1] if not pd.isna(pullback) else False
    }

    # Volume (regime-adaptive)
    vol_ratio = row.get('vol_ratio', 1.0)
    if params.get('adx_filter', False):
        vol_min = params['vol_min_trending'] if is_trending else params['vol_min_ranging']
    else:
        vol_min = 0.7
    conditions['volume_ratio'] = {
        'value': f"{vol_ratio:.2f}" if not pd.isna(vol_ratio) else "N/A",
        'required': f">= {vol_min}",
        'met': vol_ratio >= vol_min if not pd.isna(vol_ratio) else False
    }

    # Market regime
    conditions['market_regime'] = {
        'value': "Trending" if is_trending else "Ranging",
        'required': "Info only",
        'met': True  # Always passes, just info
    }

    all_met = all(c['met'] for c in conditions.values() if c['required'] != "Info only")

    return conditions, all_met


def load_strategy_stats():
    """Load strategy statistics from results"""
    stats_file = os.path.join(RESULTS_DIR, 'strategy_statistics.json')
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)
    return {}


def load_streak_stats():
    """Load streak statistics from results"""
    stats_file = os.path.join(RESULTS_DIR, 'streak_statistics.json')
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)
    return {}


def create_price_chart(df, show_adx=True):
    """Create price chart with ADX overlay"""
    if len(df) == 0:
        return None

    # Use last 48 hours (192 bars of 15min)
    df_chart = df.tail(192).copy()

    if show_adx:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('BTC Price', 'ADX (Market Regime)', 'Volume')
        )
    else:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=('BTC Price', 'Volume')
        )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df_chart.index,
            open=df_chart['open'],
            high=df_chart['high'],
            low=df_chart['low'],
            close=df_chart['close'],
            name='BTC'
        ),
        row=1, col=1
    )

    # Bollinger Bands
    if 'upper_band' in df_chart.columns:
        fig.add_trace(
            go.Scatter(x=df_chart.index, y=df_chart['upper_band'],
                      line=dict(color='rgba(150, 150, 150, 0.5)', width=1),
                      name='Upper BB'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_chart.index, y=df_chart['lower_band'],
                      line=dict(color='rgba(150, 150, 150, 0.5)', width=1),
                      fill='tonexty', fillcolor='rgba(150, 150, 150, 0.1)',
                      name='Lower BB'),
            row=1, col=1
        )

    # SMA 200
    if 'sma200' in df_chart.columns:
        fig.add_trace(
            go.Scatter(x=df_chart.index, y=df_chart['sma200'],
                      line=dict(color='orange', width=2),
                      name='SMA 200'),
            row=1, col=1
        )

    if show_adx:
        # ADX
        if 'adx' in df_chart.columns:
            fig.add_trace(
                go.Scatter(x=df_chart.index, y=df_chart['adx'],
                          line=dict(color='purple', width=2),
                          name='ADX'),
                row=2, col=1
            )
            # Threshold line
            fig.add_hline(y=20, line_dash="dash", line_color="gray",
                         annotation_text="Trending Threshold", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="red",
                         annotation_text="Strong Trend", row=2, col=1)

        # +DI / -DI
        if 'plus_di' in df_chart.columns:
            fig.add_trace(
                go.Scatter(x=df_chart.index, y=df_chart['plus_di'],
                          line=dict(color='green', width=1),
                          name='+DI'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_chart.index, y=df_chart['minus_di'],
                          line=dict(color='red', width=1),
                          name='-DI'),
                row=2, col=1
            )

        vol_row = 3
    else:
        vol_row = 2

    # Volume
    if 'volume' in df_chart.columns:
        colors = ['green' if c > o else 'red' for c, o in zip(df_chart['close'], df_chart['open'])]
        fig.add_trace(
            go.Bar(x=df_chart.index, y=df_chart['volume'],
                  marker_color=colors, name='Volume'),
            row=vol_row, col=1
        )

    fig.update_layout(
        height=700 if show_adx else 500,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )

    return fig


def create_performance_comparison():
    """Create performance comparison chart"""
    stats = load_strategy_stats()
    if not stats:
        return None

    strategies = list(stats.keys())
    returns = [stats[s].get('total_return_pct', 0) for s in strategies]
    drawdowns = [stats[s].get('max_drawdown_pct', 0) for s in strategies]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Return %',
        x=strategies,
        y=returns,
        marker_color='green'
    ))

    fig.add_trace(go.Bar(
        name='Max Drawdown %',
        x=strategies,
        y=drawdowns,
        marker_color='red'
    ))

    fig.update_layout(
        title='Strategy Performance Comparison',
        barmode='group',
        height=400,
        template='plotly_dark'
    )

    return fig


# ===============================
# Main Dashboard
# ===============================

def main():
    st.title("🎯 BTC Adaptive Streak Reduction Dashboard")

    # Initialize session state for auto-refresh
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    # Check if auto-refresh is needed (every 5 minutes)
    time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Auto-refresh toggle
        auto_refresh = st.toggle("🔄 Auto-refresh (5 min)", value=True)

        # Strategy selection
        selected_strategy = st.selectbox(
            "Select Strategy",
            options=list(ADAPTIVE_STRATEGIES.keys()),
            format_func=lambda x: f"{x}"
        )

        # Simulated consecutive losses for progressive threshold demo
        sim_losses = st.slider(
            "Simulated Consecutive Losses",
            min_value=0, max_value=6, value=0,
            help="Adjust to see how progressive threshold changes"
        )

        st.divider()

        # Show strategy description
        st.subheader("Strategy Info")
        st.write(ADAPTIVE_STRATEGIES[selected_strategy]['description'])

        # Refresh button
        if st.button("🔄 Refresh Now", type="primary"):
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            st.rerun()

        st.divider()

        # Current time
        now = datetime.now(timezone.utc)
        st.write(f"**Time (UTC):** {now.strftime('%Y-%m-%d %H:%M:%S')}")
        st.write(f"**Current Hour:** {now.hour}")

        in_asian = now.hour in ASIAN_HOURS
        if in_asian:
            st.success("🌏 Asian Session Active")
        else:
            st.warning("⏰ Outside Asian Hours")

        st.divider()

        # Auto-refresh status
        if auto_refresh:
            next_refresh = AUTO_REFRESH_INTERVAL - time_since_refresh
            if next_refresh > 0:
                st.info(f"⏱️ Next refresh in {int(next_refresh // 60)}:{int(next_refresh % 60):02d}")
            else:
                st.info("🔄 Refreshing...")
        else:
            st.write("Auto-refresh disabled")

    # Auto-refresh logic
    if auto_refresh and time_since_refresh >= AUTO_REFRESH_INTERVAL:
        st.cache_data.clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()

    # Load data with timestamp key for cache invalidation
    data_timestamp = get_data_timestamp()
    with st.spinner("Loading data..."):
        df = load_data(data_timestamp)

    if df is None or len(df) == 0:
        st.error("❌ Failed to load data")
        st.stop()

    # Compute indicators
    df = compute_indicators(df)

    # Get latest row
    latest = df.iloc[-1]

    # ===============================
    # Top Metrics Row
    # ===============================
    st.header("📊 Current Market State")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        price = latest['close']
        st.metric("BTC Price", f"${price:,.2f}")

    with col2:
        adx = latest.get('adx', 0)
        is_trending = latest.get('is_trending', False)
        regime = "Trending 📈" if is_trending else "Ranging 📊"
        st.metric("ADX", f"{adx:.1f}", regime)

    with col3:
        pos_score = calculate_positioning_score(latest)
        pos_color = "green" if pos_score > 0 else "red" if pos_score < 0 else "gray"
        st.metric("Positioning", f"{pos_score:+.2f}")

    with col4:
        vol_ratio = latest.get('vol_ratio', 1.0)
        vol_status = "High 🔥" if vol_ratio > 1.2 else "Normal" if vol_ratio > 0.8 else "Low ⚠️"
        st.metric("Volume Ratio", f"{vol_ratio:.2f}", vol_status)

    with col5:
        rsi = latest.get('rsi', 50)
        rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        st.metric("RSI", f"{rsi:.1f}", rsi_status)

    st.divider()

    # ===============================
    # Progressive Threshold Display
    # ===============================
    st.header("🎚️ Progressive Positioning Threshold")

    prog_col1, prog_col2 = st.columns([2, 3])

    with prog_col1:
        current_threshold = get_progressive_threshold(sim_losses)
        st.metric(
            "Current Threshold",
            f"{current_threshold:.2f}",
            f"Based on {sim_losses} consecutive losses"
        )

        # Show threshold ladder
        st.write("**Threshold Ladder:**")
        for losses, threshold in PROGRESSIVE_POS_THRESHOLD.items():
            marker = "👉" if losses == sim_losses else "  "
            st.write(f"{marker} {losses} losses: `{threshold:.2f}`")

        if sim_losses >= 5:
            st.error("⏸️ COOLDOWN ACTIVE (24 hours)")

    with prog_col2:
        # Create threshold visualization
        fig_thresh = go.Figure()

        losses_list = list(PROGRESSIVE_POS_THRESHOLD.keys())
        thresholds_list = list(PROGRESSIVE_POS_THRESHOLD.values())

        fig_thresh.add_trace(go.Bar(
            x=[f"{l} losses" for l in losses_list],
            y=thresholds_list,
            marker_color=['#2ecc71' if l == sim_losses else '#3498db' for l in losses_list],
            text=[f"{t:.2f}" for t in thresholds_list],
            textposition='auto'
        ))

        # Add current positioning as horizontal line
        fig_thresh.add_hline(
            y=pos_score if pos_score > 0 else 0,
            line_dash="dash",
            line_color="yellow",
            annotation_text=f"Current Pos: {pos_score:.2f}"
        )

        fig_thresh.update_layout(
            title="Positioning Threshold by Loss Count",
            height=300,
            template='plotly_dark'
        )

        st.plotly_chart(fig_thresh, width='stretch')

    st.divider()

    # ===============================
    # Entry Conditions
    # ===============================
    st.header(f"📋 Entry Conditions: {selected_strategy}")

    conditions, all_met = check_entry_conditions(
        latest,
        ADAPTIVE_STRATEGIES[selected_strategy]['config'],
        consecutive_losses=sim_losses
    )

    if all_met:
        st.success("✅ ALL CONDITIONS MET - Entry Signal Active")
    else:
        st.warning("⏳ Waiting for conditions...")

    # Display conditions in columns
    cond_cols = st.columns(4)
    for i, (name, cond) in enumerate(conditions.items()):
        with cond_cols[i % 4]:
            icon = "✅" if cond['met'] else "❌"
            st.write(f"**{name.replace('_', ' ').title()}**")
            st.write(f"{icon} Value: `{cond['value']}`")
            st.write(f"Required: {cond['required']}")

    st.divider()

    # ===============================
    # Price Chart with ADX
    # ===============================
    st.header("📈 Price Chart with ADX")

    show_adx = st.checkbox("Show ADX Panel", value=True)
    fig = create_price_chart(df, show_adx=show_adx)
    if fig:
        st.plotly_chart(fig, width='stretch')

    st.divider()

    # ===============================
    # Strategy Comparison
    # ===============================
    st.header("📊 Strategy Performance")

    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        st.subheader("Performance Metrics")
        stats = load_strategy_stats()
        if stats:
            perf_data = []
            for name, s in stats.items():
                perf_data.append({
                    'Strategy': name,
                    'Return %': f"{s.get('total_return_pct', 0):.1f}%",
                    'Max DD %': f"{s.get('max_drawdown_pct', 0):.1f}%",
                    'Win Rate': f"{s.get('win_rate_pct', 0):.1f}%",
                    'Trades': s.get('total_trades', 0),
                    'Profit Factor': f"{s.get('profit_factor', 0):.2f}",
                })
            st.dataframe(pd.DataFrame(perf_data), hide_index=True)
        else:
            st.info("Run backtest to generate performance data")

    with perf_col2:
        st.subheader("Streak Statistics")
        streak_stats = load_streak_stats()
        if streak_stats:
            streak_data = []
            for name, s in streak_stats.items():
                streak_data.append({
                    'Strategy': name,
                    'Streaks 3+': s.get('total_streaks_3plus', 0),
                    'Streaks 6+': s.get('total_streaks_6plus', 0),
                    'Max Streak': s.get('max_streak_length', 0),
                    'Avg Loss': f"${s.get('avg_streak_loss', 0):,.0f}",
                })
            st.dataframe(pd.DataFrame(streak_data), hide_index=True)
        else:
            st.info("Run backtest to generate streak data")

    # Performance comparison chart
    fig_perf = create_performance_comparison()
    if fig_perf:
        st.plotly_chart(fig_perf, width='stretch')

    st.divider()

    # ===============================
    # Data Info
    # ===============================
    st.header("📁 Data Status")

    data_col1, data_col2 = st.columns(2)

    with data_col1:
        st.write(f"**Total rows:** {len(df):,}")
        st.write(f"**Date range:** {df.index[0]} to {df.index[-1]}")
        st.write(f"**Last update:** {df.index[-1]}")

    with data_col2:
        # Check data freshness
        last_ts = df.index[-1]
        if hasattr(last_ts, 'tzinfo') and last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - last_ts.to_pydatetime().replace(tzinfo=timezone.utc)
        if age.total_seconds() < 900:  # 15 minutes
            st.success(f"✅ Data is fresh ({age.seconds // 60} min ago)")
        elif age.total_seconds() < 3600:  # 1 hour
            st.warning(f"⚠️ Data is {age.seconds // 60} minutes old")
        else:
            st.error(f"❌ Data is {age.seconds // 3600:.1f} hours old")

        # Show last refresh time
        st.write(f"**Dashboard refreshed:** {st.session_state.last_refresh.strftime('%H:%M:%S')}")

    # ===============================
    # Auto-refresh footer
    # ===============================
    st.divider()

    time_remaining = max(0, AUTO_REFRESH_INTERVAL - time_since_refresh)
    mins = int(time_remaining // 60)
    secs = int(time_remaining % 60)

    if auto_refresh:
        # Display refresh countdown with progress bar
        refresh_col1, refresh_col2 = st.columns([4, 1])
        with refresh_col1:
            progress = min(1.0, 1 - (time_remaining / AUTO_REFRESH_INTERVAL))
            st.progress(progress, text=f"🔄 Auto-refresh in {mins}:{secs:02d} | Data updates every 5 minutes")

        with refresh_col2:
            if st.button("⏭️ Refresh"):
                st.cache_data.clear()
                st.session_state.last_refresh = datetime.now()
                st.rerun()

        # Use st.empty with a hidden component that triggers rerun
        # The page will naturally rerun when user interacts
        # For true auto-refresh, we use a workaround with experimental_rerun
        if time_remaining <= 0:
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
            st.rerun()

        # Add auto-refresh meta tag for browsers that support it
        st.markdown(
            f'<meta http-equiv="refresh" content="{int(time_remaining)}">',
            unsafe_allow_html=True
        )
    else:
        st.caption("⏸️ Auto-refresh disabled. Enable in sidebar for live updates.")


if __name__ == "__main__":
    main()
