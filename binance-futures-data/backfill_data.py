#!/usr/bin/env python3
"""
Binance Futures Data Backfill Script

Efficiently backfills historical data for all metrics from two sources:
1. Binance Vision Archive (https://data.binance.vision) - Historical data from 2021-12-01 to ~2023-04-14
2. Binance Futures API - Recent data (last 30 days for most metrics)

Features:
1. Checks existing data coverage and identifies gaps
2. Downloads historical data from Binance Vision archive (no rate limits)
3. Backfills recent data using Binance API (with rate limiting)
4. Tracks API call usage (24h, 1h, 5min windows)
5. Reports data coverage vs available data
6. Runs for configurable duration (default 20 minutes)
7. Applies 10% safety margin on API rate limits

Data Sources:
- Binance Vision: 2021-12-01 to 2023-04-14 (daily zip files)
- Binance API: 2023-04-14 onwards (30 day rolling window for most metrics)

Rate Limits (API only):
- Most endpoints: 1000 requests/5min (IP limit)
- With 10% safety margin: 900 requests/5min
"""

import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import os
import time
import json
import sys
import argparse
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
SYMBOL = "BTCUSDT"
PERIOD = "5m"
DATA_DIR = "data"
API_LOG_FILE = f"{DATA_DIR}/api_call_log.json"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
BINANCE_VISION_BASE = "https://data.binance.vision/data/futures/um/daily/metrics"

# Rate limit configuration (for API calls only)
RATE_LIMIT_5MIN = 1000  # requests per 5 minutes
SAFETY_MARGIN = 0.10    # 10% safety margin
EFFECTIVE_RATE_LIMIT = int(RATE_LIMIT_5MIN * (1 - SAFETY_MARGIN))  # 900 requests/5min

# Binance Vision archive date range
ARCHIVE_START_DATE = datetime(2021, 12, 1, tzinfo=timezone.utc)
ARCHIVE_END_DATE = datetime(2023, 4, 14, tzinfo=timezone.utc)

# Data availability for API (in days from now)
API_DATA_AVAILABILITY = {
    "top_trader_position": 30,
    "top_trader_account": 30,
    "global_ls_ratio": 30,
    "funding_rate": 365 * 2,  # Effectively unlimited
    "open_interest": 30,
}

# File paths
FILES = {
    "top_trader_position": f"{DATA_DIR}/top_trader_position_ratio.csv",
    "top_trader_account": f"{DATA_DIR}/top_trader_account_ratio.csv",
    "global_ls_ratio": f"{DATA_DIR}/global_ls_ratio.csv",
    "funding_rate": f"{DATA_DIR}/funding_rate.csv",
    "open_interest": f"{DATA_DIR}/open_interest.csv",
}

# API endpoints
ENDPOINTS = {
    "top_trader_position": "/futures/data/topLongShortPositionRatio",
    "top_trader_account": "/futures/data/topLongShortAccountRatio",
    "global_ls_ratio": "/futures/data/globalLongShortAccountRatio",
    "funding_rate": "/fapi/v1/fundingRate",
    "open_interest": "/futures/data/openInterestHist",
}

# Column mappings from Binance Vision metrics file
VISION_COLUMN_MAPPING = {
    "top_trader_position": {
        "ls_ratio": "sum_toptrader_long_short_ratio",
        "long_pct": None,  # Calculate from ratio
        "short_pct": None,  # Calculate from ratio
    },
    "top_trader_account": {
        "ls_ratio": "count_toptrader_long_short_ratio",
        "long_pct": None,
        "short_pct": None,
    },
    "global_ls_ratio": {
        "ls_ratio": "count_long_short_ratio",
        "long_pct": None,
        "short_pct": None,
    },
    "open_interest": {
        "sum_open_interest": "sum_open_interest",
        "sum_open_interest_value": "sum_open_interest_value",
    },
}


def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"[{timestamp}] {message}")


def ensure_data_dir():
    """Create data directory if it doesn't exist"""
    os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# API CALL LOGGING
# ============================================================

def load_api_log():
    """Load API call log from file"""
    try:
        if os.path.exists(API_LOG_FILE):
            with open(API_LOG_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        log(f"Warning: Error loading API log: {e}")
    return {"calls": []}


def save_api_log(api_log):
    """Save API call log to file"""
    try:
        # Keep only last 24 hours of logs
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=24)).timestamp()
        api_log["calls"] = [c for c in api_log["calls"] if c["timestamp"] > cutoff]

        with open(API_LOG_FILE, 'w') as f:
            json.dump(api_log, f, indent=2)
    except Exception as e:
        log(f"Warning: Error saving API log: {e}")


def record_api_call(api_log, endpoint, success=True):
    """Record an API call in the log"""
    api_log["calls"].append({
        "timestamp": datetime.now(timezone.utc).timestamp(),
        "endpoint": endpoint,
        "success": success
    })


def get_api_call_counts(api_log):
    """Get API call counts for different time windows"""
    now = datetime.now(timezone.utc).timestamp()

    counts = {
        "5min": 0,
        "1hour": 0,
        "24hour": 0
    }

    for call in api_log.get("calls", []):
        ts = call["timestamp"]
        age_seconds = now - ts

        if age_seconds <= 300:  # 5 minutes
            counts["5min"] += 1
        if age_seconds <= 3600:  # 1 hour
            counts["1hour"] += 1
        if age_seconds <= 86400:  # 24 hours
            counts["24hour"] += 1

    return counts


def can_make_api_call(api_log):
    """Check if we can make another API call within rate limits"""
    counts = get_api_call_counts(api_log)
    return counts["5min"] < EFFECTIVE_RATE_LIMIT


def get_wait_time_for_rate_limit(api_log):
    """Calculate how long to wait before next API call is allowed"""
    now = datetime.now(timezone.utc).timestamp()

    # Get all calls in last 5 minutes
    recent_calls = [c["timestamp"] for c in api_log.get("calls", [])
                   if now - c["timestamp"] <= 300]

    if len(recent_calls) < EFFECTIVE_RATE_LIMIT:
        return 0

    # Sort and find the oldest call that will expire first
    recent_calls.sort()
    oldest_in_window = recent_calls[0]
    wait_time = 300 - (now - oldest_in_window) + 1  # +1 second buffer

    return max(0, wait_time)


# ============================================================
# DATA COVERAGE ANALYSIS
# ============================================================

def get_data_coverage(filepath):
    """Get the date range of existing data in a file"""
    try:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if len(df) > 0 and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
                return {
                    "rows": len(df),
                    "start": df['timestamp'].min(),
                    "end": df['timestamp'].max()
                }
    except Exception as e:
        log(f"Warning: Error reading {filepath}: {e}")
    return None


def analyze_data_gaps(metric_name, filepath):
    """Analyze gaps in existing data and what needs to be backfilled"""
    coverage = get_data_coverage(filepath)
    now = datetime.now(timezone.utc)

    result = {
        "metric": metric_name,
        "archive_start": ARCHIVE_START_DATE,
        "archive_end": ARCHIVE_END_DATE,
        "api_available_days": API_DATA_AVAILABILITY.get(metric_name, 30),
        "has_data": coverage is not None,
        "data_rows": 0,
        "data_start": None,
        "data_end": None,
        "needs_archive_backfill": False,
        "archive_backfill_start": None,
        "archive_backfill_end": None,
        "needs_api_backfill": False,
        "api_backfill_start": None,
        "api_backfill_end": None,
        "coverage_pct": 0
    }

    # Total available range: from archive start to now
    total_start = ARCHIVE_START_DATE
    total_end = now

    if coverage:
        result["data_rows"] = coverage["rows"]
        result["data_start"] = coverage["start"]
        result["data_end"] = coverage["end"]

        # Check if we need archive backfill (older than existing data)
        if coverage["start"] > ARCHIVE_START_DATE:
            result["needs_archive_backfill"] = True
            result["archive_backfill_start"] = ARCHIVE_START_DATE
            result["archive_backfill_end"] = min(coverage["start"], ARCHIVE_END_DATE)

        # Check if we need API backfill (gap between archive end and existing data start)
        # or recent data update
        api_start = now - timedelta(days=API_DATA_AVAILABILITY.get(metric_name, 30))

        # If existing data ends before now, we need to update
        if coverage["end"] < now - timedelta(hours=1):
            result["needs_api_backfill"] = True
            result["api_backfill_start"] = max(coverage["end"], api_start)
            result["api_backfill_end"] = now

        # Calculate coverage percentage
        total_duration = (total_end - total_start).total_seconds()
        covered_duration = (coverage["end"] - coverage["start"]).total_seconds()
        result["coverage_pct"] = min(100, (covered_duration / total_duration) * 100)
    else:
        # No data at all - need full backfill
        result["needs_archive_backfill"] = True
        result["archive_backfill_start"] = ARCHIVE_START_DATE
        result["archive_backfill_end"] = ARCHIVE_END_DATE

        result["needs_api_backfill"] = True
        api_start = now - timedelta(days=API_DATA_AVAILABILITY.get(metric_name, 30))
        result["api_backfill_start"] = max(ARCHIVE_END_DATE, api_start)
        result["api_backfill_end"] = now

    return result


# ============================================================
# BINANCE VISION ARCHIVE DOWNLOAD
# ============================================================

def download_vision_metrics_file(date):
    """
    Download a single day's metrics file from Binance Vision.

    Args:
        date: datetime object for the date to download

    Returns:
        DataFrame with all metrics for that day, or None if failed
    """
    date_str = date.strftime('%Y-%m-%d')
    url = f"{BINANCE_VISION_BASE}/{SYMBOL}/{SYMBOL}-metrics-{date_str}.zip"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                # Get the CSV file from the zip
                csv_name = f"{SYMBOL}-metrics-{date_str}.csv"
                if csv_name in zf.namelist():
                    with zf.open(csv_name) as csv_file:
                        df = pd.read_csv(csv_file)
                        df['timestamp'] = pd.to_datetime(df['create_time'], utc=True)
                        return df
        elif response.status_code == 404:
            return None  # File doesn't exist for this date
        else:
            log(f"Warning: HTTP {response.status_code} for {date_str}")
            return None
    except Exception as e:
        log(f"Warning: Error downloading {date_str}: {e}")
        return None


def process_vision_data_for_metric(df, metric_name):
    """
    Process raw Binance Vision data for a specific metric.

    Args:
        df: Raw DataFrame from Binance Vision
        metric_name: Name of the metric to extract

    Returns:
        Processed DataFrame for that metric
    """
    if df is None or len(df) == 0:
        return None

    if metric_name == "funding_rate":
        # Funding rate is not in the metrics file
        return None

    if metric_name == "open_interest":
        result = df[['timestamp', 'symbol', 'sum_open_interest', 'sum_open_interest_value']].copy()
        result['sum_open_interest'] = result['sum_open_interest'].astype(float)
        result['sum_open_interest_value'] = result['sum_open_interest_value'].astype(float)
        return result

    # Long/short ratio metrics
    mapping = VISION_COLUMN_MAPPING.get(metric_name)
    if not mapping:
        return None

    ratio_col = mapping.get("ls_ratio")
    if ratio_col not in df.columns:
        return None

    result = pd.DataFrame()
    result['timestamp'] = df['timestamp']
    result['symbol'] = df['symbol']
    result['ls_ratio'] = df[ratio_col].astype(float)

    # Calculate long/short percentages from ratio
    # ratio = long_pct / short_pct and long_pct + short_pct = 1
    # So: long_pct = ratio / (1 + ratio), short_pct = 1 / (1 + ratio)
    result['long_pct'] = result['ls_ratio'] / (1 + result['ls_ratio'])
    result['short_pct'] = 1 / (1 + result['ls_ratio'])

    return result


def backfill_from_archive(metric_name, start_date, end_date, max_duration_seconds=600):
    """
    Backfill data from Binance Vision archive.

    Args:
        metric_name: Name of the metric to backfill
        start_date: Start date for backfill
        end_date: End date for backfill
        max_duration_seconds: Maximum time to run

    Returns:
        Dictionary with backfill results
    """
    filepath = FILES.get(metric_name)
    start_time = time.time()

    result = {
        "metric": metric_name,
        "source": "archive",
        "files_downloaded": 0,
        "rows_added": 0,
        "errors": 0,
        "completed": False,
        "duration_seconds": 0
    }

    if metric_name == "funding_rate":
        # Funding rate is not in the archive
        result["completed"] = True
        return result

    # Generate list of dates to download
    dates_to_download = []
    current_date = start_date
    while current_date <= end_date:
        dates_to_download.append(current_date)
        current_date += timedelta(days=1)

    log(f"    Archive: Need to download {len(dates_to_download)} days of data")

    all_data = []

    for i, date in enumerate(dates_to_download):
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed >= max_duration_seconds:
            log(f"    Archive: Time limit reached after {i} files")
            break

        # Download and process
        df = download_vision_metrics_file(date)
        if df is not None:
            processed = process_vision_data_for_metric(df, metric_name)
            if processed is not None and len(processed) > 0:
                all_data.append(processed)
                result["files_downloaded"] += 1
        else:
            result["errors"] += 1

        # Progress update every 30 files
        if (i + 1) % 30 == 0:
            log(f"    Archive: Downloaded {i + 1}/{len(dates_to_download)} files...")

        # Small delay to be nice to the server
        time.sleep(0.05)

    # Combine and save all data
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        new_rows = append_to_csv(filepath, combined)
        result["rows_added"] = new_rows
        log(f"    Archive: Added {new_rows:,} rows from {result['files_downloaded']} files")

    if result["files_downloaded"] == len(dates_to_download):
        result["completed"] = True

    result["duration_seconds"] = time.time() - start_time
    return result


# ============================================================
# BINANCE API DATA FETCHING
# ============================================================

def fetch_api_data(metric_name, start_time=None, end_time=None, limit=500):
    """
    Fetch data from Binance API.
    Returns (dataframe, success)
    """
    endpoint = ENDPOINTS.get(metric_name)
    if not endpoint:
        return None, False

    url = f"{BINANCE_FUTURES_BASE}{endpoint}"

    params = {
        "symbol": SYMBOL,
        "limit": limit
    }

    # Add period for non-funding-rate endpoints
    if metric_name != "funding_rate":
        params["period"] = PERIOD

    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        if not data:
            return None, True  # Success but no data

        df = pd.DataFrame(data)

        # Process based on metric type
        if metric_name == "funding_rate":
            df['timestamp'] = pd.to_datetime(df['fundingTime'].astype(int), unit='ms', utc=True)
            df['funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
            df['mark_price'] = pd.to_numeric(df['markPrice'], errors='coerce')
            # Drop rows with invalid funding rate
            df = df.dropna(subset=['funding_rate'])
            if len(df) == 0:
                return None, True
            return df[['timestamp', 'symbol', 'funding_rate', 'mark_price']], True

        elif metric_name == "open_interest":
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
            df['sum_open_interest'] = df['sumOpenInterest'].astype(float)
            df['sum_open_interest_value'] = df['sumOpenInterestValue'].astype(float)
            return df[['timestamp', 'symbol', 'sum_open_interest', 'sum_open_interest_value']], True

        else:  # Long/short ratio metrics
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
            df['ls_ratio'] = df['longShortRatio'].astype(float)
            df['long_pct'] = df['longAccount'].astype(float)
            df['short_pct'] = df['shortAccount'].astype(float)
            return df[['timestamp', 'symbol', 'ls_ratio', 'long_pct', 'short_pct']], True

    except Exception as e:
        log(f"Error: API fetch failed for {metric_name}: {e}")
        return None, False


def append_to_csv(filepath, df):
    """Append new data to CSV, avoiding duplicates"""
    if df is None or len(df) == 0:
        return 0

    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        existing['timestamp'] = pd.to_datetime(existing['timestamp'], format='mixed', utc=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)

        # Combine and remove duplicates
        combined = pd.concat([existing, df])
        combined = combined.drop_duplicates(subset=['timestamp'], keep='last')
        combined = combined.sort_values('timestamp')

        new_rows = len(combined) - len(existing)
        combined.to_csv(filepath, index=False)
        return new_rows
    else:
        df = df.sort_values('timestamp')
        df.to_csv(filepath, index=False)
        return len(df)


def backfill_from_api(metric_name, api_log, max_duration_seconds=600):
    """
    Backfill data from Binance API.

    Args:
        metric_name: Name of the metric to backfill
        api_log: API call log for rate limiting
        max_duration_seconds: Maximum time to run

    Returns:
        Dictionary with backfill results
    """
    filepath = FILES.get(metric_name)
    start_time = time.time()

    result = {
        "metric": metric_name,
        "source": "api",
        "api_calls": 0,
        "rows_added": 0,
        "errors": 0,
        "completed": False,
        "duration_seconds": 0
    }

    # Get current data coverage
    coverage = get_data_coverage(filepath)
    now = datetime.now(timezone.utc)

    if coverage:
        # Start from where existing data ends
        backfill_start = coverage["end"]
    else:
        # Start from 30 days ago (or funding rate goes back further)
        days_back = API_DATA_AVAILABILITY.get(metric_name, 30)
        backfill_start = now - timedelta(days=days_back)

    current_start_ts = int(backfill_start.timestamp() * 1000)
    now_ts = int(now.timestamp() * 1000)

    consecutive_errors = 0
    consecutive_empty = 0
    MAX_CONSECUTIVE_ERRORS = 5
    MAX_CONSECUTIVE_EMPTY = 3

    while current_start_ts < now_ts:
        # Check time limit
        elapsed = time.time() - start_time
        if elapsed >= max_duration_seconds:
            log(f"    API: Time limit reached ({elapsed:.0f}s)")
            break

        # Check consecutive errors/empty responses
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            log(f"    API: Too many consecutive errors ({consecutive_errors}), stopping")
            break

        if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
            log(f"    API: Too many consecutive empty responses, likely up to date")
            result["completed"] = True
            break

        # Check rate limit
        if not can_make_api_call(api_log):
            wait_time = get_wait_time_for_rate_limit(api_log)
            if wait_time > 0:
                remaining_time = max_duration_seconds - elapsed
                if wait_time > remaining_time:
                    log(f"    API: Rate limit wait ({wait_time:.0f}s) exceeds remaining time")
                    break
                log(f"    API: Rate limit reached, waiting {wait_time:.0f}s...")
                time.sleep(wait_time)

        # Fetch data
        df, success = fetch_api_data(metric_name, start_time=current_start_ts, limit=500)
        record_api_call(api_log, metric_name, success)
        result["api_calls"] += 1

        if not success:
            result["errors"] += 1
            consecutive_errors += 1
            time.sleep(1)
            continue

        consecutive_errors = 0

        if df is None or len(df) == 0:
            consecutive_empty += 1
            # Move forward in time
            current_start_ts += 500 * 5 * 60 * 1000  # Move forward ~41 hours
            time.sleep(0.2)
            continue

        consecutive_empty = 0

        # Save data
        new_rows = append_to_csv(filepath, df)
        result["rows_added"] += new_rows

        # Get the newest timestamp from this batch
        newest_ts = int(df['timestamp'].max().timestamp() * 1000)

        # Move to next batch
        current_start_ts = newest_ts + 1

        # Small delay
        time.sleep(0.1)

    if current_start_ts >= now_ts:
        result["completed"] = True

    result["duration_seconds"] = time.time() - start_time
    save_api_log(api_log)

    return result


# ============================================================
# MAIN BACKFILL LOGIC
# ============================================================

def backfill_metric(metric_name, api_log, max_duration_seconds=1200, skip_archive=False):
    """
    Backfill data for a single metric from both archive and API.

    Args:
        metric_name: Name of the metric to backfill
        api_log: API call log for rate limiting
        max_duration_seconds: Maximum time to run
        skip_archive: If True, skip archive backfill

    Returns:
        Dictionary with combined backfill results
    """
    filepath = FILES.get(metric_name)
    start_time = time.time()

    result = {
        "metric": metric_name,
        "archive_files": 0,
        "archive_rows": 0,
        "api_calls": 0,
        "api_rows": 0,
        "total_rows_added": 0,
        "errors": 0,
        "completed": False,
        "duration_seconds": 0
    }

    # Analyze what needs to be backfilled
    gap_info = analyze_data_gaps(metric_name, filepath)

    # Phase 1: Backfill from archive (if needed)
    if not skip_archive and gap_info["needs_archive_backfill"] and gap_info["archive_backfill_start"]:
        elapsed = time.time() - start_time
        remaining = max_duration_seconds - elapsed

        if remaining > 60:  # Only if we have at least 1 minute
            log(f"  {metric_name}: Backfilling from archive...")
            archive_result = backfill_from_archive(
                metric_name,
                gap_info["archive_backfill_start"],
                gap_info["archive_backfill_end"],
                max_duration_seconds=remaining * 0.7  # Use 70% of time for archive
            )
            result["archive_files"] = archive_result["files_downloaded"]
            result["archive_rows"] = archive_result["rows_added"]
            result["errors"] += archive_result["errors"]

    # Phase 2: Backfill from API (if needed)
    if gap_info["needs_api_backfill"]:
        elapsed = time.time() - start_time
        remaining = max_duration_seconds - elapsed

        if remaining > 30:  # Only if we have at least 30 seconds
            log(f"  {metric_name}: Backfilling from API...")
            api_result = backfill_from_api(
                metric_name,
                api_log,
                max_duration_seconds=remaining
            )
            result["api_calls"] = api_result["api_calls"]
            result["api_rows"] = api_result["rows_added"]
            result["errors"] += api_result["errors"]

            if api_result["completed"]:
                result["completed"] = True

    # Check if fully complete (all gaps filled)
    result["total_rows_added"] = result["archive_rows"] + result["api_rows"]

    # Re-check coverage
    final_gap = analyze_data_gaps(metric_name, filepath)
    if not final_gap["needs_archive_backfill"] and not final_gap["needs_api_backfill"]:
        result["completed"] = True

    result["duration_seconds"] = time.time() - start_time
    return result


def run_backfill(max_duration_minutes=20, skip_archive=False, metrics=None):
    """
    Run backfill for all metrics.

    Args:
        max_duration_minutes: Maximum total runtime in minutes
        skip_archive: If True, skip archive backfill (use API only)
        metrics: List of specific metrics to backfill (None = all)
    """
    log("=" * 70)
    log("BINANCE FUTURES DATA BACKFILL")
    log(f"Symbol: {SYMBOL} | Period: {PERIOD}")
    log(f"Archive Range: {ARCHIVE_START_DATE.strftime('%Y-%m-%d')} to {ARCHIVE_END_DATE.strftime('%Y-%m-%d')}")
    log(f"API Rate Limit: {EFFECTIVE_RATE_LIMIT} requests/5min (with 10% safety margin)")
    log(f"Max Duration: {max_duration_minutes} minutes")
    if skip_archive:
        log("Mode: API only (skipping archive)")
    log("=" * 70)

    ensure_data_dir()

    # Load API log
    api_log = load_api_log()

    # Show current API usage
    counts = get_api_call_counts(api_log)
    log("")
    log("CURRENT API USAGE:")
    log(f"  Last 5 minutes:  {counts['5min']:,} / {EFFECTIVE_RATE_LIMIT} requests")
    log(f"  Last 1 hour:     {counts['1hour']:,} requests")
    log(f"  Last 24 hours:   {counts['24hour']:,} requests")

    # Determine which metrics to process
    if metrics:
        files_to_process = {k: v for k, v in FILES.items() if k in metrics}
    else:
        files_to_process = FILES

    # Analyze all metrics
    log("")
    log("DATA COVERAGE ANALYSIS:")
    log("-" * 70)

    metrics_to_backfill = []

    for metric_name, filepath in files_to_process.items():
        gap_info = analyze_data_gaps(metric_name, filepath)

        if gap_info["has_data"]:
            data_start = gap_info["data_start"].strftime('%Y-%m-%d %H:%M')
            data_end = gap_info["data_end"].strftime('%Y-%m-%d %H:%M')
            coverage_str = f"{data_start} to {data_end}"
            coverage_pct = f"{gap_info['coverage_pct']:.1f}%"
            rows = f"{gap_info['data_rows']:,} rows"
        else:
            coverage_str = "No data"
            coverage_pct = "0.0%"
            rows = "0 rows"

        log(f"  {metric_name}:")
        log(f"    Coverage:   {coverage_pct} ({rows})")
        log(f"    Extracted:  {coverage_str}")

        needs_work = False
        if gap_info["needs_archive_backfill"] and not skip_archive:
            archive_days = (gap_info["archive_backfill_end"] - gap_info["archive_backfill_start"]).days
            log(f"    Archive:    Need {archive_days} days of historical data")
            needs_work = True
        if gap_info["needs_api_backfill"]:
            log(f"    API:        Need recent data update")
            needs_work = True

        if needs_work:
            metrics_to_backfill.append(metric_name)
        else:
            log(f"    Status:     Complete")

    log("-" * 70)

    if not metrics_to_backfill:
        log("")
        log("All metrics are fully backfilled!")
        return

    # Run backfill
    log("")
    log("STARTING BACKFILL:")
    log("-" * 70)

    start_time = time.time()
    max_duration_seconds = max_duration_minutes * 60
    results = []

    for metric_name in metrics_to_backfill:
        elapsed = time.time() - start_time
        remaining = max_duration_seconds - elapsed

        if remaining <= 0:
            log(f"  Time limit reached, skipping {metric_name}")
            break

        log(f"")
        log(f"  Processing {metric_name}...")
        result = backfill_metric(
            metric_name,
            api_log,
            max_duration_seconds=remaining,
            skip_archive=skip_archive
        )
        results.append(result)

        log(f"    Results: Archive={result['archive_rows']:,} rows, "
            f"API={result['api_rows']:,} rows, "
            f"Total={result['total_rows_added']:,} rows, "
            f"Completed={result['completed']}")

    # Summary
    log("")
    log("=" * 70)
    log("BACKFILL SUMMARY")
    log("=" * 70)

    total_archive_files = sum(r["archive_files"] for r in results)
    total_archive_rows = sum(r["archive_rows"] for r in results)
    total_api_calls = sum(r["api_calls"] for r in results)
    total_api_rows = sum(r["api_rows"] for r in results)
    total_rows = sum(r["total_rows_added"] for r in results)
    total_errors = sum(r["errors"] for r in results)
    total_duration = time.time() - start_time

    log(f"  Archive files downloaded: {total_archive_files}")
    log(f"  Archive rows added:       {total_archive_rows:,}")
    log(f"  API calls made:           {total_api_calls}")
    log(f"  API rows added:           {total_api_rows:,}")
    log(f"  Total rows added:         {total_rows:,}")
    log(f"  Errors:                   {total_errors}")
    log(f"  Duration:                 {total_duration:.1f} seconds")

    # Final API usage
    counts = get_api_call_counts(api_log)
    log("")
    log("FINAL API USAGE:")
    log(f"  Last 5 minutes:  {counts['5min']:,} / {EFFECTIVE_RATE_LIMIT} requests")
    log(f"  Last 1 hour:     {counts['1hour']:,} requests")
    log(f"  Last 24 hours:   {counts['24hour']:,} requests")

    # Show updated coverage
    log("")
    log("UPDATED DATA COVERAGE:")
    log("-" * 70)

    for metric_name, filepath in files_to_process.items():
        gap_info = analyze_data_gaps(metric_name, filepath)

        if gap_info["has_data"]:
            data_start = gap_info["data_start"].strftime('%Y-%m-%d %H:%M')
            data_end = gap_info["data_end"].strftime('%Y-%m-%d %H:%M')
            log(f"  {metric_name}: {gap_info['coverage_pct']:.1f}% ({data_start} to {data_end})")
        else:
            log(f"  {metric_name}: No data")

    log("-" * 70)

    # Check if more backfill needed
    incomplete = [r for r in results if not r["completed"]]
    if incomplete:
        log("")
        log("NOTE: Some metrics are not fully backfilled.")
        log("Run this script again to continue backfilling.")
    else:
        log("")
        log("All requested metrics have been fully backfilled!")

    log("")
    log("Done")


def show_status():
    """Show current status without running backfill"""
    log("=" * 70)
    log("BINANCE FUTURES DATA STATUS")
    log("=" * 70)

    ensure_data_dir()

    # Load API log
    api_log = load_api_log()

    # Show API usage
    counts = get_api_call_counts(api_log)
    log("")
    log("API USAGE:")
    log(f"  Last 5 minutes:  {counts['5min']:,} / {EFFECTIVE_RATE_LIMIT} requests")
    log(f"  Last 1 hour:     {counts['1hour']:,} requests")
    log(f"  Last 24 hours:   {counts['24hour']:,} requests")

    # Show data coverage
    log("")
    log("DATA COVERAGE:")
    log("-" * 70)
    log(f"{'Metric':<25} {'Rows':<12} {'Start':<12} {'End':<12} {'Coverage':<10}")
    log("-" * 70)

    for metric_name, filepath in FILES.items():
        coverage = get_data_coverage(filepath)

        if coverage:
            data_start = coverage["start"].strftime('%Y-%m-%d')
            data_end = coverage["end"].strftime('%Y-%m-%d')
            rows = f"{coverage['rows']:,}"

            # Calculate coverage
            total_days = (datetime.now(timezone.utc) - ARCHIVE_START_DATE).days
            covered_days = (coverage["end"] - coverage["start"]).days
            coverage_pct = min(100, (covered_days / total_days) * 100)
            coverage_str = f"{coverage_pct:.1f}%"
        else:
            data_start = "N/A"
            data_end = "N/A"
            rows = "0"
            coverage_str = "0%"

        log(f"  {metric_name:<23} {rows:<12} {data_start:<12} {data_end:<12} {coverage_str:<10}")

    log("-" * 70)

    # Show archive info
    log("")
    log("ARCHIVE INFO:")
    log(f"  Binance Vision archive: {ARCHIVE_START_DATE.strftime('%Y-%m-%d')} to {ARCHIVE_END_DATE.strftime('%Y-%m-%d')}")
    log(f"  API data availability:  Last 30 days (most metrics)")
    log(f"  Total historical range: {ARCHIVE_START_DATE.strftime('%Y-%m-%d')} to present")

    log("")
    log("Done")


def main():
    parser = argparse.ArgumentParser(
        description="Binance Futures Data Backfill Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backfill_data.py                    # Run full backfill for 20 minutes
  python backfill_data.py --duration 60      # Run backfill for 60 minutes
  python backfill_data.py --status           # Show current status only
  python backfill_data.py --api-only         # Skip archive, use API only
  python backfill_data.py --metric funding_rate  # Backfill specific metric

Data Sources:
  - Archive: Historical data from 2021-12-01 to 2023-04-14 (no rate limits)
  - API: Recent data (last 30 days for most metrics, rate limited)
        """
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=20,
        help="Maximum duration in minutes (default: 20)"
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status without running backfill"
    )

    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Skip archive backfill, use API only"
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=list(FILES.keys()),
        help="Backfill only a specific metric"
    )

    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        metrics = [args.metric] if args.metric else None
        run_backfill(
            max_duration_minutes=args.duration,
            skip_archive=args.api_only,
            metrics=metrics
        )


if __name__ == "__main__":
    main()
