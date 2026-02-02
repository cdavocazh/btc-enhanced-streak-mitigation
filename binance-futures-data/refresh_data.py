#!/usr/bin/env python3
"""
Unified Data Refresh Script
============================
Refreshes all Binance Futures data (positioning metrics + OHLC price data).
Designed to be called every 5 minutes via crontab or dashboard.

Features:
- Append-only updates (no data loss)
- Rate limit aware (10% safety margin)
- Quick mode for 5-minute refresh cycles
- Full mode for complete backfill

Usage:
    python refresh_data.py                  # Quick refresh (5-minute update)
    python refresh_data.py --full           # Full backfill
    python refresh_data.py --status         # Show status only
"""

import os
import sys
import argparse
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Change to script directory for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Import from extract_binance_data
from extract_binance_data import (
    log, ensure_data_dir, load_api_log, save_api_log,
    get_api_call_counts, can_make_api_call,
    get_data_coverage, extract_metric_data, FILES,
    EFFECTIVE_RATE_LIMIT, DATA_DIR,
    validate_all_csv_files, validate_and_fix_csv
)

# Timezone for display (GMT+8)
DISPLAY_TZ = ZoneInfo("Asia/Singapore")  # GMT+8

# Last timestamps file
LAST_TIMESTAMPS_FILE = f"{DATA_DIR}/last_timestamps.json"


def save_last_timestamps():
    """
    Save last timestamps for all metrics to JSON file with timezone info.
    Includes both UTC and GMT+8 representations.
    """
    timestamps = {}
    now = datetime.now(timezone.utc)

    for metric_name, filepath in FILES.items():
        coverage = get_data_coverage(filepath)
        if coverage and coverage["end"]:
            last_ts = coverage["end"]
            last_ts_gmt8 = last_ts.astimezone(DISPLAY_TZ)

            timestamps[metric_name] = {
                "last_timestamp_ms": int(last_ts.timestamp() * 1000),
                "last_timestamp_utc": last_ts.strftime('%Y-%m-%d %H:%M:%S UTC'),
                "last_timestamp_gmt8": last_ts_gmt8.strftime('%Y-%m-%d %H:%M:%S GMT+8'),
                "last_timestamp_iso": last_ts.isoformat(),
                "updated_at_utc": now.strftime('%Y-%m-%d %H:%M:%S UTC'),
                "updated_at_gmt8": now.astimezone(DISPLAY_TZ).strftime('%Y-%m-%d %H:%M:%S GMT+8'),
                "updated_at_iso": now.isoformat()
            }

    try:
        with open(LAST_TIMESTAMPS_FILE, 'w') as f:
            json.dump(timestamps, f, indent=2)
    except Exception as e:
        log(f"Warning: Could not save last timestamps: {e}")


def quick_refresh():
    """
    Quick refresh mode for 5-minute cron cycles.
    Only fetches new data since last update (no gap filling).
    Prioritizes price data, then positioning metrics.
    """
    log("=" * 60)
    log("QUICK DATA REFRESH")
    log(f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    log("=" * 60)

    ensure_data_dir()

    # Validate and auto-fix any corrupted CSV files before processing
    validate_all_csv_files()

    api_log = load_api_log()

    # Check current API usage
    counts = get_api_call_counts(api_log)
    available_calls = EFFECTIVE_RATE_LIMIT - counts["5min"]
    log(f"API calls available: {available_calls}/{EFFECTIVE_RATE_LIMIT}")

    if available_calls < 10:
        log("Warning: Low API quota, skipping some updates")

    results = {}

    # Priority order: price first, then positioning metrics
    metrics_priority = [
        "price",              # Most important for live signals
        "top_trader_position",
        "global_ls_ratio",
        "open_interest",
        "top_trader_account",
        "funding_rate",       # Least frequent updates (8h)
    ]

    # Allocate API calls per metric
    calls_per_metric = max(1, available_calls // len(metrics_priority))

    for metric_name in metrics_priority:
        if metric_name not in FILES:
            continue

        filepath = FILES[metric_name]
        coverage = get_data_coverage(filepath)

        if coverage:
            age_minutes = (datetime.now(timezone.utc) - coverage["end"]).total_seconds() / 60
            log(f"{metric_name}: Last update {age_minutes:.1f} min ago")

            # Skip if updated very recently (< 2 minutes for non-funding)
            if metric_name != "funding_rate" and age_minutes < 2:
                log(f"  Skipping (recently updated)")
                continue
        else:
            log(f"{metric_name}: No existing data")

        # Extract new data
        result = extract_metric_data(
            metric_name,
            api_log,
            max_requests=calls_per_metric
        )
        results[metric_name] = result

        if result["rows_added"] > 0:
            # Format last timestamp in GMT+8
            last_ts = result.get("last_timestamp")
            if last_ts:
                last_ts_gmt8 = last_ts.astimezone(DISPLAY_TZ)
                ts_str = last_ts_gmt8.strftime('%Y-%m-%d %H:%M:%S GMT+8')
                log(f"  Added {result['rows_added']} rows ({result['api_calls']} API calls) | Updated to: {ts_str}")
            else:
                log(f"  Added {result['rows_added']} rows ({result['api_calls']} API calls)")
        else:
            log(f"  Up to date")

        # Save API log after each metric
        save_api_log(api_log)

    # Summary
    log("")
    log("-" * 60)
    total_rows = sum(r.get("rows_added", 0) for r in results.values())
    total_calls = sum(r.get("api_calls", 0) for r in results.values())
    log(f"Total: {total_rows} rows added, {total_calls} API calls")

    # Final API usage
    counts = get_api_call_counts(api_log)
    log(f"API usage: {counts['5min']}/{EFFECTIVE_RATE_LIMIT}")

    # Save last timestamps with timezone info
    save_last_timestamps()

    log("=" * 60)

    return results


def full_refresh(max_duration_minutes=20):
    """
    Full refresh mode with gap filling.
    Use for initial setup or recovering from gaps.
    """
    log("=" * 60)
    log("FULL DATA REFRESH (with gap filling)")
    log(f"Max duration: {max_duration_minutes} minutes")
    log("=" * 60)

    ensure_data_dir()

    # Validate and auto-fix any corrupted CSV files before processing
    validate_all_csv_files()

    # Use the incremental fetch from extract_binance_data
    from extract_binance_data import fetch_incremental
    result = fetch_incremental(fill_gaps=True)

    # Save last timestamps with timezone info
    save_last_timestamps()

    return result


def show_status():
    """Show current data status without making any API calls."""
    log("=" * 60)
    log("DATA STATUS")
    log("=" * 60)

    ensure_data_dir()
    api_log = load_api_log()

    # API usage
    counts = get_api_call_counts(api_log)
    log(f"API Usage (5min): {counts['5min']}/{EFFECTIVE_RATE_LIMIT}")
    log(f"API Usage (1hr):  {counts['1hour']}")
    log(f"API Usage (24hr): {counts['24hour']}")
    log("")

    # Data coverage
    log("Data Coverage:")
    log("-" * 60)

    for metric_name, filepath in FILES.items():
        coverage = get_data_coverage(filepath)

        if coverage:
            age = datetime.now(timezone.utc) - coverage["end"]
            age_str = f"{age.total_seconds() / 60:.1f} min ago"
            rows = f"{coverage['rows']:,}"
            # Show in GMT+8
            end_time_gmt8 = coverage["end"].astimezone(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M GMT+8")
            log(f"  {metric_name:<25} {rows:>10} rows | last: {end_time_gmt8} ({age_str})")
        else:
            log(f"  {metric_name:<25} No data")

    log("-" * 60)

    # Save/update last timestamps with timezone info
    save_last_timestamps()


def main():
    parser = argparse.ArgumentParser(
        description="Unified Data Refresh Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python refresh_data.py              # Quick 5-minute refresh
    python refresh_data.py --full       # Full refresh with gap filling
    python refresh_data.py --status     # Show status only

For crontab (every 5 minutes):
    */5 * * * * cd /path/to/binance-futures-data && python refresh_data.py >> refresh.log 2>&1
        """
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Full refresh with gap filling (slower)"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=20,
        help="Max duration for full refresh in minutes (default: 20)"
    )

    parser.add_argument(
        "--status",
        action="store_true",
        help="Show status only, no data fetching"
    )

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.full:
        full_refresh(max_duration_minutes=args.duration)
    else:
        quick_refresh()

    log("")
    log("Done")


if __name__ == "__main__":
    main()
