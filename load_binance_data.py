#!/usr/bin/env python3
"""
Binance Positioning Data Loader

Loads and processes data from the binance-futures-data repository,
resampling 5-minute data to hourly to match OHLC data.

Data Sources:
- top_trader_position_ratio.csv (5-min intervals)
- top_trader_account_ratio.csv (5-min intervals)
- global_ls_ratio.csv (5-min intervals)
- funding_rate.csv (8-hour intervals)
- open_interest.csv (5-min intervals)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os

# Path to binance-futures-data repository (relative to btc-enhanced-strategy)
BINANCE_DATA_DIR = "../binance-futures-data/data"

# File paths
DATA_FILES = {
    "top_trader_position": f"{BINANCE_DATA_DIR}/top_trader_position_ratio.csv",
    "top_trader_account": f"{BINANCE_DATA_DIR}/top_trader_account_ratio.csv",
    "global_ls_ratio": f"{BINANCE_DATA_DIR}/global_ls_ratio.csv",
    "funding_rate": f"{BINANCE_DATA_DIR}/funding_rate.csv",
    "open_interest": f"{BINANCE_DATA_DIR}/open_interest.csv",
}


def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"[{timestamp}] {message}")


def load_csv_with_timestamp(filepath, timestamp_col='timestamp'):
    """Load CSV file and parse timestamp column"""
    try:
        df = pd.read_csv(filepath)
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format='mixed', utc=True)
        df = df.set_index(timestamp_col)
        df = df.sort_index()
        return df
    except FileNotFoundError:
        log(f"⚠ File not found: {filepath}")
        return None
    except Exception as e:
        log(f"⚠ Error loading {filepath}: {e}")
        return None


def resample_to_hourly(df, agg_method='last'):
    """
    Resample 5-minute data to hourly.

    Args:
        df: DataFrame with datetime index
        agg_method: Aggregation method ('last', 'mean', 'first')

    Returns:
        Resampled DataFrame
    """
    if df is None or len(df) == 0:
        return None

    if agg_method == 'last':
        return df.resample('h').last()
    elif agg_method == 'mean':
        return df.resample('h').mean()
    elif agg_method == 'first':
        return df.resample('h').first()
    else:
        return df.resample('h').last()


def load_top_trader_position():
    """Load and process top trader position ratio data"""
    df = load_csv_with_timestamp(DATA_FILES["top_trader_position"])
    if df is None:
        return None

    # Resample to hourly (use last value in each hour)
    df_hourly = resample_to_hourly(df)

    # Rename columns to match expected format
    df_hourly = df_hourly.rename(columns={
        'ls_ratio': 'top_trader_position_ls_ratio',
        'long_pct': 'top_trader_position_long_pct',
        'short_pct': 'top_trader_position_short_pct'
    })

    # Drop symbol column if present
    if 'symbol' in df_hourly.columns:
        df_hourly = df_hourly.drop(columns=['symbol'])

    return df_hourly


def load_top_trader_account():
    """Load and process top trader account ratio data"""
    df = load_csv_with_timestamp(DATA_FILES["top_trader_account"])
    if df is None:
        return None

    # Resample to hourly
    df_hourly = resample_to_hourly(df)

    # Rename columns
    df_hourly = df_hourly.rename(columns={
        'ls_ratio': 'top_trader_account_ls_ratio',
        'long_pct': 'top_trader_account_long_pct',
        'short_pct': 'top_trader_account_short_pct'
    })

    if 'symbol' in df_hourly.columns:
        df_hourly = df_hourly.drop(columns=['symbol'])

    return df_hourly


def load_global_ls_ratio():
    """Load and process global long/short ratio data"""
    df = load_csv_with_timestamp(DATA_FILES["global_ls_ratio"])
    if df is None:
        return None

    # Resample to hourly
    df_hourly = resample_to_hourly(df)

    # Rename columns
    df_hourly = df_hourly.rename(columns={
        'ls_ratio': 'global_ls_ratio',
        'long_pct': 'global_long_pct',
        'short_pct': 'global_short_pct'
    })

    if 'symbol' in df_hourly.columns:
        df_hourly = df_hourly.drop(columns=['symbol'])

    return df_hourly


def load_funding_rate():
    """Load and process funding rate data"""
    df = load_csv_with_timestamp(DATA_FILES["funding_rate"])
    if df is None:
        return None

    # Funding rate is already at 8-hour intervals
    # Resample to hourly by forward-filling
    df_hourly = df.resample('h').ffill()

    # Calculate funding rate change
    df_hourly['funding_rate_change'] = df_hourly['funding_rate'].diff()

    # Calculate rolling averages for funding rate
    df_hourly['funding_rate_ma8'] = df_hourly['funding_rate'].rolling(8).mean()
    df_hourly['funding_rate_ma24'] = df_hourly['funding_rate'].rolling(24).mean()

    if 'symbol' in df_hourly.columns:
        df_hourly = df_hourly.drop(columns=['symbol'])

    return df_hourly


def load_open_interest():
    """Load and process open interest data"""
    df = load_csv_with_timestamp(DATA_FILES["open_interest"])
    if df is None:
        return None

    # Resample to hourly
    df_hourly = resample_to_hourly(df)

    # Calculate OI change and percentage change
    df_hourly['oi_change'] = df_hourly['sum_open_interest'].diff()
    df_hourly['oi_change_pct'] = df_hourly['sum_open_interest'].pct_change() * 100

    # Calculate OI moving averages
    df_hourly['oi_ma24'] = df_hourly['sum_open_interest'].rolling(24).mean()
    df_hourly['oi_vs_ma24'] = df_hourly['sum_open_interest'] / df_hourly['oi_ma24']

    if 'symbol' in df_hourly.columns:
        df_hourly = df_hourly.drop(columns=['symbol'])

    return df_hourly


def load_all_binance_data():
    """
    Load all Binance positioning data and merge into a single DataFrame.

    Returns:
        DataFrame with all positioning metrics at hourly intervals
    """
    log("Loading Binance positioning data...")

    # Load all data sources
    top_position = load_top_trader_position()
    top_account = load_top_trader_account()
    global_ratio = load_global_ls_ratio()
    funding = load_funding_rate()
    oi = load_open_interest()

    # Start with the most complete dataset
    dataframes = []

    if top_position is not None:
        dataframes.append(('top_trader_position', top_position))
        log(f"  ✓ Top trader position: {len(top_position)} hourly records")

    if top_account is not None:
        dataframes.append(('top_trader_account', top_account))
        log(f"  ✓ Top trader account: {len(top_account)} hourly records")

    if global_ratio is not None:
        dataframes.append(('global_ratio', global_ratio))
        log(f"  ✓ Global L/S ratio: {len(global_ratio)} hourly records")

    if funding is not None:
        dataframes.append(('funding_rate', funding))
        log(f"  ✓ Funding rate: {len(funding)} hourly records")

    if oi is not None:
        dataframes.append(('open_interest', oi))
        log(f"  ✓ Open interest: {len(oi)} hourly records")

    if not dataframes:
        log("❌ No Binance data available")
        return None

    # Merge all dataframes
    merged = dataframes[0][1]

    for name, df in dataframes[1:]:
        merged = pd.merge(
            merged,
            df,
            left_index=True,
            right_index=True,
            how='outer'
        )

    # Sort by index
    merged = merged.sort_index()

    # Forward fill missing values (data may have gaps)
    merged = merged.ffill()

    log(f"✓ Merged Binance data: {len(merged)} hourly records")
    log(f"  Date range: {merged.index.min()} to {merged.index.max()}")
    log(f"  Columns: {list(merged.columns)}")

    return merged


def get_data_summary():
    """Print summary of available Binance data"""
    log("=" * 60)
    log("BINANCE DATA SUMMARY")
    log("=" * 60)

    for name, filepath in DATA_FILES.items():
        df = load_csv_with_timestamp(filepath)
        if df is not None:
            log(f"{name}:")
            log(f"  Rows: {len(df):,}")
            log(f"  From: {df.index.min()}")
            log(f"  To:   {df.index.max()}")
            log(f"  Columns: {list(df.columns)}")
        else:
            log(f"{name}: Not available")

    log("=" * 60)


if __name__ == "__main__":
    # Test loading data
    get_data_summary()

    log("")
    log("Loading merged hourly data...")
    df = load_all_binance_data()

    if df is not None:
        log("")
        log("Sample data (last 5 rows):")
        print(df.tail())
