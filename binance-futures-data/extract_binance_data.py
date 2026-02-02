#!/usr/bin/env python3
"""
Binance Futures Data Extractor
Extracts positioning metrics, funding rates, and open interest for BTCUSDT perpetuals.

Data Extracted (5-minute intervals):
1. Top Trader Long/Short Position Ratio
2. Top Trader Long/Short Account Ratio
3. Global Trader Long/Short Ratio
4. Funding Rate History
5. Open Interest Statistics

API Rate Limits:
- Most endpoints: 1000 requests/5min (IP limit)
- With 10% safety margin: 900 requests/5min
- Max 500 rows per request (1000 for funding rate)

Data Availability:
- Long/Short Ratios: Last 30 days
- Open Interest: Last 30 days (1 month)
- Funding Rate: Unlimited historical data (max 1000 per request)

Features:
- Paginated requests to fetch maximum data within rate limits
- Gap detection to fill missing data in historical records
- Append-only mode to avoid duplicates
"""

import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

# Timezone for display (GMT+8)
DISPLAY_TZ = ZoneInfo("Asia/Singapore")  # GMT+8
import os
import time
import sys
import json

# Configuration
SYMBOL = "BTCUSDT"
PERIOD = "5m"  # 5-minute intervals (highest granularity)
DATA_DIR = "data"
API_LOG_FILE = f"{DATA_DIR}/api_call_log.json"
BINANCE_FUTURES_BASE = "https://fapi.binance.com"

# Rate limit configuration
RATE_LIMIT_5MIN = 1000  # requests per 5 minutes
SAFETY_MARGIN = 0.10    # 10% safety margin
EFFECTIVE_RATE_LIMIT = int(RATE_LIMIT_5MIN * (1 - SAFETY_MARGIN))  # 900 requests/5min

# Data granularity in minutes
DATA_GRANULARITY = {
    "top_trader_position": 5,
    "top_trader_account": 5,
    "global_ls_ratio": 5,
    "funding_rate": 480,  # 8 hours = 480 minutes
    "open_interest": 5,
    "price": 5,  # 5-minute candles
}

# Max rows per API request (Binance API limits)
MAX_ROWS_PER_REQUEST = {
    "top_trader_position": 500,   # Max 500
    "top_trader_account": 500,    # Max 500
    "global_ls_ratio": 500,       # Max 500
    "funding_rate": 1000,         # Max 1000
    "open_interest": 500,         # Max 500
    "price": 1000,                # Klines API max is 1000
}

# Data availability (in days from now)
API_DATA_AVAILABILITY = {
    "top_trader_position": 30,
    "top_trader_account": 30,
    "global_ls_ratio": 30,
    "funding_rate": 365 * 2,  # Effectively unlimited
    "open_interest": 30,
    "price": 365 * 2,  # Klines has extensive history
}

# File names for each metric
FILES = {
    "top_trader_position": f"{DATA_DIR}/top_trader_position_ratio.csv",
    "top_trader_account": f"{DATA_DIR}/top_trader_account_ratio.csv",
    "global_ls_ratio": f"{DATA_DIR}/global_ls_ratio.csv",
    "funding_rate": f"{DATA_DIR}/funding_rate.csv",
    "open_interest": f"{DATA_DIR}/open_interest.csv",
    "price": f"{DATA_DIR}/price.csv",  # OHLC price data
}

# API endpoints
ENDPOINTS = {
    "top_trader_position": "/futures/data/topLongShortPositionRatio",
    "top_trader_account": "/futures/data/topLongShortAccountRatio",
    "global_ls_ratio": "/futures/data/globalLongShortAccountRatio",
    "funding_rate": "/fapi/v1/fundingRate",
    "open_interest": "/futures/data/openInterestHist",
    "price": "/fapi/v1/klines",  # Klines/candlestick endpoint
}


def log(message):
    """Print timestamped log message"""
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f"[{timestamp}] {message}")


def ensure_data_dir():
    """Create data directory if it doesn't exist"""
    os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================
# API CALL LOGGING & RATE LIMITING
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
# DATA ANALYSIS & GAP DETECTION
# ============================================================

def get_data_coverage(filepath):
    """Get the date range and row count of existing data in a file"""
    try:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if len(df) > 0 and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
                return {
                    "rows": len(df),
                    "start": df['timestamp'].min(),
                    "end": df['timestamp'].max(),
                    "df": df
                }
    except Exception as e:
        log(f"Warning: Error reading {filepath}: {e}")
    return None


def detect_gaps(filepath, metric_name):
    """
    Detect gaps in the data where timestamps are missing.

    Returns a list of gap ranges: [(gap_start, gap_end), ...]
    """
    coverage = get_data_coverage(filepath)
    if coverage is None:
        return []

    df = coverage["df"]
    granularity_minutes = DATA_GRANULARITY.get(metric_name, 5)
    expected_interval = timedelta(minutes=granularity_minutes)

    # Sort by timestamp
    df = df.sort_values('timestamp')
    timestamps = df['timestamp'].tolist()

    gaps = []

    for i in range(1, len(timestamps)):
        prev_ts = timestamps[i - 1]
        curr_ts = timestamps[i]
        actual_interval = curr_ts - prev_ts

        # Allow some tolerance (1.5x expected interval)
        if actual_interval > expected_interval * 1.5:
            gap_start = prev_ts + expected_interval
            gap_end = curr_ts - expected_interval

            # Only report gaps that are significant (more than 1 interval)
            if gap_end > gap_start:
                gaps.append((gap_start, gap_end))

    return gaps


def get_last_timestamp(filepath):
    """Get the last timestamp from an existing CSV file"""
    try:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            if len(df) > 0 and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
                return int(df['timestamp'].max().timestamp() * 1000)
    except Exception as e:
        log(f"Warning: Error reading {filepath}: {e}")
    return None


# ============================================================
# CSV VALIDATION & AUTO-FIX
# ============================================================

def validate_and_fix_csv(filepath):
    """
    Validate CSV file for data integrity issues and auto-fix them.

    Checks for:
    1. Empty files (no header or data)
    2. Malformed timestamps (e.g., "1-03" instead of "2026-01-03")
    3. Duplicate rows
    4. Out-of-order timestamps

    Returns:
        dict: {"valid": bool, "fixed": bool, "issues": list, "rows_removed": int}
    """
    result = {"valid": True, "fixed": False, "issues": [], "rows_removed": 0}

    if not os.path.exists(filepath):
        return result  # File doesn't exist, nothing to validate

    try:
        # Check if file is empty or has no header
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            if not first_line:
                result["valid"] = False
                result["issues"].append("Empty file")
                return result
            if 'timestamp' not in first_line.lower():
                result["valid"] = False
                result["issues"].append("Missing header or 'timestamp' column")
                return result

        # Read the file
        df = pd.read_csv(filepath)
        original_len = len(df)

        if len(df) == 0:
            result["valid"] = True  # Empty but has header is OK
            return result

        # Check for malformed timestamps
        # Valid format: YYYY-MM-DD HH:MM:SS+00:00 or similar
        # Invalid: 1-03 23:45:00+00:00 (missing year)
        import re
        malformed_mask = df['timestamp'].astype(str).str.match(r'^[0-9]{1,2}-[0-9]{1,2}\s')
        malformed_count = malformed_mask.sum()

        if malformed_count > 0:
            result["issues"].append(f"Found {malformed_count} malformed timestamps")

            # Try to fix malformed timestamps
            # Pattern: "1-03 23:45:00+00:00" should become "2026-01-03 23:45:00+00:00"
            # We'll infer the year from surrounding valid timestamps

            # Get the most common year from valid timestamps
            valid_timestamps = df[~malformed_mask]['timestamp'].astype(str)
            if len(valid_timestamps) > 0:
                years = valid_timestamps.str.extract(r'^(\d{4})-')[0]
                most_common_year = years.mode().iloc[0] if len(years.mode()) > 0 else "2026"

                # Fix malformed timestamps
                def fix_timestamp(ts):
                    ts_str = str(ts)
                    if pd.isna(ts) or ts_str == 'nan':
                        return ts
                    # Check if it matches the malformed pattern
                    match = re.match(r'^([0-9]{1,2})-([0-9]{1,2})\s(.+)$', ts_str)
                    if match:
                        month, day, rest = match.groups()
                        fixed = f"{most_common_year}-{month.zfill(2)}-{day.zfill(2)} {rest}"
                        return fixed
                    return ts_str

                df['timestamp'] = df['timestamp'].apply(fix_timestamp)
                result["fixed"] = True
                log(f"  Auto-fixed {malformed_count} malformed timestamps in {os.path.basename(filepath)}")

        # Try to parse all timestamps to validate
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        except Exception as e:
            # If parsing fails, try to identify and remove bad rows
            result["issues"].append(f"Timestamp parsing error: {e}")

            # Parse row by row, keeping only valid ones
            valid_rows = []
            for idx, row in df.iterrows():
                try:
                    pd.to_datetime(row['timestamp'], format='mixed', utc=True)
                    valid_rows.append(idx)
                except:
                    pass

            if len(valid_rows) < len(df):
                removed = len(df) - len(valid_rows)
                df = df.loc[valid_rows]
                result["rows_removed"] += removed
                result["fixed"] = True
                result["issues"].append(f"Removed {removed} rows with unparseable timestamps")
                log(f"  Removed {removed} invalid rows from {os.path.basename(filepath)}")

            # Try parsing again
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)

        # Remove duplicates
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        duplicates_removed = before_dedup - len(df)
        if duplicates_removed > 0:
            result["issues"].append(f"Removed {duplicates_removed} duplicate rows")
            result["rows_removed"] += duplicates_removed
            result["fixed"] = True

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Save if any fixes were made
        if result["fixed"]:
            # Convert timestamp back to string format for CSV
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S+00:00')
            df.to_csv(filepath, index=False)
            log(f"  Saved cleaned data to {os.path.basename(filepath)} ({len(df)} rows)")

        result["valid"] = True
        result["rows_removed"] = original_len - len(df)

    except pd.errors.EmptyDataError:
        result["valid"] = False
        result["issues"].append("File is empty or has no parseable data")
    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"Validation error: {str(e)}")

    return result


def validate_all_csv_files():
    """
    Validate all CSV files in the data directory.

    Returns:
        dict: Summary of validation results for each file
    """
    log("Validating CSV files...")
    results = {}

    for metric_name, filepath in FILES.items():
        if os.path.exists(filepath):
            result = validate_and_fix_csv(filepath)
            results[filepath] = result

            if result["issues"]:
                status = "FIXED" if result["fixed"] else "ISSUES"
                log(f"  {os.path.basename(filepath)}: {status} - {', '.join(result['issues'])}")

    # Summary
    fixed_count = sum(1 for r in results.values() if r.get("fixed"))
    issue_count = sum(1 for r in results.values() if r.get("issues"))

    if fixed_count > 0:
        log(f"Auto-fixed {fixed_count} files")
    if issue_count == 0:
        log("All CSV files validated successfully")

    return results


# ============================================================
# CSV OPERATIONS
# ============================================================

def append_to_csv(filepath, df):
    """Append new data to CSV, avoiding duplicates"""
    if df is None or len(df) == 0:
        return 0

    if os.path.exists(filepath):
        # Validate and fix existing CSV before reading
        validation = validate_and_fix_csv(filepath)
        if not validation["valid"]:
            log(f"  Warning: Could not validate {os.path.basename(filepath)}: {validation['issues']}")

        existing = pd.read_csv(filepath)
        existing['timestamp'] = pd.to_datetime(existing['timestamp'], format='mixed', utc=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)

        # Remove duplicates
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


# ============================================================
# API DATA FETCHING
# ============================================================

def fetch_api_data(metric_name, start_time=None, end_time=None, limit=None, max_retries=3):
    """
    Fetch data from Binance API.

    Args:
        metric_name: Name of the metric to fetch
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds
        limit: Max rows to fetch (defaults to MAX_ROWS_PER_REQUEST)
        max_retries: Maximum number of retry attempts for connection errors

    Returns:
        (dataframe, success)
    """
    endpoint = ENDPOINTS.get(metric_name)
    if not endpoint:
        return None, False

    if limit is None:
        limit = MAX_ROWS_PER_REQUEST.get(metric_name, 500)

    url = f"{BINANCE_FUTURES_BASE}{endpoint}"

    params = {
        "symbol": SYMBOL,
        "limit": limit
    }

    # Add period for non-funding-rate and non-price endpoints
    if metric_name not in ["funding_rate", "price"]:
        params["period"] = PERIOD

    # Special handling for price/klines endpoint
    if metric_name == "price":
        params["interval"] = "5m"  # 5-minute candles

    # Validate startTime for metrics with 30-day limit
    if start_time and metric_name in ["top_trader_position", "top_trader_account", "global_ls_ratio", "open_interest"]:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        max_age_ms = API_DATA_AVAILABILITY.get(metric_name, 30) * 24 * 60 * 60 * 1000
        min_start_time = now_ms - max_age_ms + (60 * 60 * 1000)  # Add 1 hour buffer
        if start_time < min_start_time:
            start_time = min_start_time
            log(f"  Adjusted startTime for {metric_name} (30-day limit)")

    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    # Retry logic with exponential backoff for connection errors
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            if not data:
                return None, True  # Success but no data

            # Process based on metric type
            if metric_name == "price":
                # Klines response is array of arrays:
                # [open_time, open, high, low, close, volume, close_time, quote_volume, trades, ...]
                df = pd.DataFrame(data, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
                    'taker_buy_quote_volume', 'ignore'
                ])
                df['timestamp'] = pd.to_datetime(df['open_time'].astype(int), unit='ms', utc=True)
                df['symbol'] = SYMBOL
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                df['quote_volume'] = df['quote_volume'].astype(float)
                return df[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'quote_volume']], True

            df = pd.DataFrame(data)

            if metric_name == "funding_rate":
                df['timestamp'] = pd.to_datetime(df['fundingTime'].astype(int), unit='ms', utc=True)
                df['funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
                df['mark_price'] = pd.to_numeric(df['markPrice'], errors='coerce')
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

        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + 1  # Exponential backoff: 2, 3, 5 seconds
                log(f"  Connection error for {metric_name}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                log(f"Error: API connection failed for {metric_name} after {max_retries} attempts: {e}")
                return None, False

        except requests.exceptions.HTTPError as e:
            if response.status_code == 400:
                log(f"Error: API 400 Bad Request for {metric_name}. URL: {response.url}")
            else:
                log(f"Error: API HTTP error for {metric_name}: {e}")
            return None, False

        except requests.exceptions.Timeout as e:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + 1
                log(f"  Timeout for {metric_name}, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            else:
                log(f"Error: API timeout for {metric_name} after {max_retries} attempts: {e}")
                return None, False

        except Exception as e:
            log(f"Error: API fetch failed for {metric_name}: {e}")
            return None, False

    # Should not reach here, but just in case
    return None, False


# ============================================================
# PAGINATED DATA EXTRACTION
# ============================================================

def extract_metric_data(metric_name, api_log, start_time_ms=None, end_time_ms=None, max_requests=None):
    """
    Extract data for a single metric using paginated requests.

    Args:
        metric_name: Name of the metric to extract
        api_log: API call log for rate limiting
        start_time_ms: Start timestamp in milliseconds (None = from last data point)
        end_time_ms: End timestamp in milliseconds (None = now)
        max_requests: Maximum number of API requests to make (None = use rate limit)

    Returns:
        Dictionary with extraction results
    """
    filepath = FILES.get(metric_name)

    result = {
        "metric": metric_name,
        "api_calls": 0,
        "rows_added": 0,
        "errors": 0,
        "completed": False,
        "last_timestamp": None,  # Track the last timestamp for display
    }

    # Determine start time
    if start_time_ms is None:
        last_ts = get_last_timestamp(filepath)
        if last_ts:
            start_time_ms = last_ts + 1
        else:
            # No existing data - start from API availability limit
            days_back = API_DATA_AVAILABILITY.get(metric_name, 30)
            start_time_ms = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000)

    # Determine end time
    if end_time_ms is None:
        end_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    current_start_ts = start_time_ms
    consecutive_errors = 0
    consecutive_empty = 0
    consecutive_no_new = 0  # Track when no NEW rows are added
    MAX_CONSECUTIVE_ERRORS = 5
    MAX_CONSECUTIVE_EMPTY = 2  # Stop after 2 empty/no-new-data responses

    # Determine max requests
    if max_requests is None:
        # Use remaining rate limit
        counts = get_api_call_counts(api_log)
        max_requests = EFFECTIVE_RATE_LIMIT - counts["5min"]

    requests_made = 0

    while current_start_ts < end_time_ms and requests_made < max_requests:
        # Check consecutive errors/empty responses
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            log(f"    {metric_name}: Too many consecutive errors ({consecutive_errors}), stopping")
            break

        if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
            log(f"    {metric_name}: Data is up to date")
            result["completed"] = True
            break

        # Check rate limit
        if not can_make_api_call(api_log):
            wait_time = get_wait_time_for_rate_limit(api_log)
            if wait_time > 0:
                log(f"    {metric_name}: Rate limit reached, waiting {wait_time:.0f}s...")
                time.sleep(wait_time)

        # Fetch data
        limit = MAX_ROWS_PER_REQUEST.get(metric_name, 500)
        df, success = fetch_api_data(metric_name, start_time=current_start_ts, limit=limit)
        record_api_call(api_log, metric_name, success)
        result["api_calls"] += 1
        requests_made += 1

        if not success:
            result["errors"] += 1
            consecutive_errors += 1
            time.sleep(1)
            continue

        consecutive_errors = 0

        if df is None or len(df) == 0:
            consecutive_empty += 1
            # Move forward in time
            granularity_ms = DATA_GRANULARITY.get(metric_name, 5) * 60 * 1000
            current_start_ts += limit * granularity_ms
            time.sleep(0.05)
            continue

        # Save data
        new_rows = append_to_csv(filepath, df)
        result["rows_added"] += new_rows

        # Check if we're getting duplicate data (no new rows added)
        if new_rows == 0:
            consecutive_empty += 1
        else:
            consecutive_empty = 0

        # Get the newest timestamp from this batch
        newest_ts = int(df['timestamp'].max().timestamp() * 1000)
        result["last_timestamp"] = df['timestamp'].max()  # Store as datetime

        # Move to next batch
        current_start_ts = newest_ts + 1

        # Progress update every 10 requests
        if requests_made % 10 == 0:
            log(f"    {metric_name}: {requests_made} requests, {result['rows_added']} rows added...")

        # Minimal delay between requests (50ms to avoid burst)
        time.sleep(0.05)

    if current_start_ts >= end_time_ms:
        result["completed"] = True

    return result


def fill_gaps_for_metric(metric_name, api_log, max_requests_per_gap=50):
    """
    Fill gaps in historical data for a metric.

    Args:
        metric_name: Name of the metric
        api_log: API call log for rate limiting
        max_requests_per_gap: Maximum requests to use per gap

    Returns:
        Dictionary with gap fill results
    """
    filepath = FILES.get(metric_name)

    result = {
        "metric": metric_name,
        "gaps_found": 0,
        "gaps_filled": 0,
        "api_calls": 0,
        "rows_added": 0,
    }

    # Detect gaps
    gaps = detect_gaps(filepath, metric_name)
    result["gaps_found"] = len(gaps)

    if not gaps:
        return result

    log(f"    {metric_name}: Found {len(gaps)} gaps in data")

    # Check API data availability
    now = datetime.now(timezone.utc)
    days_back = API_DATA_AVAILABILITY.get(metric_name, 30)
    api_cutoff = now - timedelta(days=days_back)

    for gap_start, gap_end in gaps:
        # Skip gaps that are too old for API
        if gap_end < api_cutoff:
            log(f"    {metric_name}: Gap {gap_start} to {gap_end} is too old for API (>{days_back} days)")
            continue

        # Adjust gap start if it's too old
        if gap_start < api_cutoff:
            gap_start = api_cutoff

        # Check rate limit
        if not can_make_api_call(api_log):
            wait_time = get_wait_time_for_rate_limit(api_log)
            if wait_time > 60:  # Don't wait more than 1 minute for gap filling
                log(f"    {metric_name}: Rate limit reached, skipping remaining gaps")
                break
            if wait_time > 0:
                time.sleep(wait_time)

        start_ms = int(gap_start.timestamp() * 1000)
        end_ms = int(gap_end.timestamp() * 1000)

        log(f"    {metric_name}: Filling gap from {gap_start.strftime('%Y-%m-%d %H:%M')} to {gap_end.strftime('%Y-%m-%d %H:%M')}")

        gap_result = extract_metric_data(
            metric_name,
            api_log,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            max_requests=max_requests_per_gap
        )

        result["api_calls"] += gap_result["api_calls"]
        result["rows_added"] += gap_result["rows_added"]

        if gap_result["completed"]:
            result["gaps_filled"] += 1

    return result


# ============================================================
# MAIN EXTRACTION FUNCTIONS
# ============================================================

def fetch_incremental(fill_gaps=True):
    """
    Fetch new data since the last extraction using paginated requests.
    Optionally fill gaps in historical data.

    Args:
        fill_gaps: If True, also fill gaps in historical data
    """
    log("Running incremental data extraction...")
    log(f"Rate limit: {EFFECTIVE_RATE_LIMIT} requests/5min")

    ensure_data_dir()

    # Load API log
    api_log = load_api_log()

    # Show current API usage
    counts = get_api_call_counts(api_log)
    log(f"Current API usage (5min): {counts['5min']}/{EFFECTIVE_RATE_LIMIT}")

    results = {}
    gap_results = {}

    metrics = list(FILES.keys())

    for metric_name in metrics:
        filepath = FILES.get(metric_name)

        # Show current coverage
        coverage = get_data_coverage(filepath)
        if coverage:
            log(f"")
            log(f"{metric_name}:")
            log(f"  Current: {coverage['rows']:,} rows, {coverage['start'].strftime('%Y-%m-%d %H:%M')} to {coverage['end'].strftime('%Y-%m-%d %H:%M')}")
        else:
            log(f"")
            log(f"{metric_name}: No existing data")

        # Fill gaps first (if enabled)
        if fill_gaps and coverage:
            gap_result = fill_gaps_for_metric(metric_name, api_log)
            gap_results[metric_name] = gap_result
            if gap_result["rows_added"] > 0:
                log(f"  Gap fill: {gap_result['rows_added']} rows added ({gap_result['gaps_filled']}/{gap_result['gaps_found']} gaps filled)")

        # Extract new data
        log(f"  Extracting new data...")
        result = extract_metric_data(metric_name, api_log)
        results[metric_name] = result

        if result["rows_added"] > 0:
            last_ts = result.get("last_timestamp")
            if last_ts:
                last_ts_gmt8 = last_ts.astimezone(DISPLAY_TZ)
                ts_str = last_ts_gmt8.strftime('%Y-%m-%d %H:%M:%S GMT+8')
                log(f"  New data: {result['rows_added']} rows added ({result['api_calls']} API calls) | Updated to: {ts_str}")
            else:
                log(f"  New data: {result['rows_added']} rows added ({result['api_calls']} API calls)")
        else:
            log(f"  New data: Up to date")

        # Save API log after each metric
        save_api_log(api_log)

    # Summary
    log("")
    log("=" * 60)
    log("EXTRACTION SUMMARY")
    log("=" * 60)

    total_api_calls = sum(r["api_calls"] for r in results.values())
    total_rows = sum(r["rows_added"] for r in results.values())
    total_gap_rows = sum(r.get("rows_added", 0) for r in gap_results.values())
    total_errors = sum(r["errors"] for r in results.values())

    for metric_name in metrics:
        r = results[metric_name]
        g = gap_results.get(metric_name, {})
        gap_str = f", gaps: {g.get('rows_added', 0)}" if g.get('rows_added', 0) > 0 else ""
        log(f"  {metric_name}: {r['rows_added']} new rows{gap_str} ({r['api_calls']} calls)")

    log("")
    log(f"Total: {total_rows + total_gap_rows:,} rows added, {total_api_calls} API calls, {total_errors} errors")

    # Final API usage
    counts = get_api_call_counts(api_log)
    log(f"Final API usage (5min): {counts['5min']}/{EFFECTIVE_RATE_LIMIT}")

    return results


def fetch_all_historical(days=30):
    """
    Fetch all historical data for the specified number of days.
    Use this for initial data population.

    Args:
        days: Number of days of historical data (max 30 for most metrics)
    """
    log(f"Fetching {days} days of historical data...")
    log(f"Rate limit: {EFFECTIVE_RATE_LIMIT} requests/5min")

    ensure_data_dir()

    # Load API log
    api_log = load_api_log()

    # Show current API usage
    counts = get_api_call_counts(api_log)
    log(f"Current API usage (5min): {counts['5min']}/{EFFECTIVE_RATE_LIMIT}")

    # Calculate start time
    now = datetime.now(timezone.utc)
    start_time_ms = int((now - timedelta(days=days)).timestamp() * 1000)
    end_time_ms = int(now.timestamp() * 1000)

    results = {}

    for metric_name in FILES.keys():
        log(f"")
        log(f"Extracting {metric_name}...")

        result = extract_metric_data(
            metric_name,
            api_log,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms
        )
        results[metric_name] = result

        log(f"  {result['rows_added']} rows added ({result['api_calls']} API calls)")

        # Save API log after each metric
        save_api_log(api_log)

    # Summary
    log("")
    log("=" * 60)
    log("EXTRACTION SUMMARY")
    log("=" * 60)

    total_rows = sum(r["rows_added"] for r in results.values())
    total_api_calls = sum(r["api_calls"] for r in results.values())

    for metric_name, r in results.items():
        log(f"  {metric_name}: {r['rows_added']} rows ({r['api_calls']} calls)")

    log("")
    log(f"Total: {total_rows:,} rows added, {total_api_calls} API calls")

    return results


def print_summary():
    """Print summary of current data files including gap analysis"""
    log("=" * 60)
    log("DATA SUMMARY")
    log("=" * 60)

    for name, filepath in FILES.items():
        if os.path.exists(filepath):
            coverage = get_data_coverage(filepath)
            if coverage:
                rows = coverage["rows"]
                start = coverage["start"]
                end = coverage["end"]

                # Detect gaps
                gaps = detect_gaps(filepath, name)

                log(f"{name}:")
                log(f"  Rows: {rows:,}")
                log(f"  From: {start}")
                log(f"  To:   {end}")

                if gaps:
                    log(f"  Gaps: {len(gaps)} detected")
                    # Show first few gaps
                    for i, (gap_start, gap_end) in enumerate(gaps[:3]):
                        duration = gap_end - gap_start
                        log(f"    - {gap_start.strftime('%Y-%m-%d %H:%M')} to {gap_end.strftime('%Y-%m-%d %H:%M')} ({duration})")
                    if len(gaps) > 3:
                        log(f"    ... and {len(gaps) - 3} more gaps")
                else:
                    log(f"  Gaps: None detected")
            else:
                log(f"{name}: File exists but no valid data")
        else:
            log(f"{name}: No data file found")

    # Show API usage
    api_log = load_api_log()
    counts = get_api_call_counts(api_log)
    log("")
    log("API USAGE:")
    log(f"  Last 5 minutes:  {counts['5min']:,} / {EFFECTIVE_RATE_LIMIT}")
    log(f"  Last 1 hour:     {counts['1hour']:,}")
    log(f"  Last 24 hours:   {counts['24hour']:,}")

    log("=" * 60)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Binance Futures Data Extractor")
    parser.add_argument(
        "--mode",
        choices=["historical", "incremental", "summary"],
        default="incremental",
        help="Extraction mode: historical (full backfill), incremental (new data + gap fill), summary (show current data)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of historical data to fetch (only for historical mode, max 30)"
    )
    parser.add_argument(
        "--no-gap-fill",
        action="store_true",
        help="Skip gap filling in incremental mode"
    )

    args = parser.parse_args()

    log("=" * 60)
    log("BINANCE FUTURES DATA EXTRACTOR")
    log(f"Symbol: {SYMBOL} | Period: {PERIOD}")
    log("=" * 60)

    if args.mode == "historical":
        days = min(args.days, 30)  # API limit is 30 days for most metrics
        results = fetch_all_historical(days=days)
        log("")
        log("Historical extraction complete")

    elif args.mode == "incremental":
        results = fetch_incremental(fill_gaps=not args.no_gap_fill)
        log("")
        log("Incremental extraction complete")

    elif args.mode == "summary":
        print_summary()

    log("")
    log("Done")


if __name__ == "__main__":
    main()
