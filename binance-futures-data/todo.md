# Binance Futures Data - Requirements & Fixes

## Requirements Log

### Requirement 1: Fix API Errors - COMPLETED
**Problem:**
- `ConnectionResetError(54, 'Connection reset by peer')` for top_trader_position
- `400 Client Error` for open_interest when startTime is > 30 days old

**Solution:**
- Added retry logic with exponential backoff (3 retries) for connection errors
- Added automatic startTime adjustment for metrics with 30-day API limit
- Added better error handling for HTTP 400 errors with URL logging

**Files Modified:**
- `extract_binance_data.py`: `fetch_api_data()` function

---

### Requirement 2: Optimize Data Refresh Speed - COMPLETED
**Problem:**
- Data refresh was slow due to conservative delays and incorrect API limits

**Solution:**
- Fixed klines API limit from 1500 to 1000 (actual max)
- Reduced inter-request delay from 100ms to 50ms
- Verified all other metrics use maximum allowed limits (500 for most, 1000 for funding_rate)

**Files Modified:**
- `extract_binance_data.py`: `MAX_ROWS_PER_REQUEST`, delay in `extract_metric_data()`

---

### Requirement 3: Add Date-Time to Terminal Output - COMPLETED
**Problem:**
- After "xxx rows added..." no indication of what date/time the data was updated to

**Solution:**
- Added last timestamp display in GMT+8 timezone format
- Output now shows: `Added X rows (Y API calls) | Updated to: YYYY-MM-DD HH:MM:SS GMT+8`

**Files Modified:**
- `refresh_data.py`: `quick_refresh()` output messages
- `extract_binance_data.py`: Added DISPLAY_TZ, updated `fetch_incremental()` output

---

### Requirement 4: Add Timezone to last_timestamps.json - COMPLETED
**Problem:**
- `last_timestamps.json` only showed UTC timestamps with no timezone indication

**Solution:**
- Added `save_last_timestamps()` function to refresh_data.py
- JSON now includes both UTC and GMT+8 representations for each metric:
  - `last_timestamp_utc`: "2026-01-30 10:25:00 UTC"
  - `last_timestamp_gmt8`: "2026-01-30 18:25:00 GMT+8"
  - `last_timestamp_iso`: ISO 8601 format with timezone
  - `updated_at_utc` / `updated_at_gmt8`: When the file was updated

**Files Modified:**
- `refresh_data.py`: Added `save_last_timestamps()`, called in `quick_refresh()`, `full_refresh()`, `show_status()`

---

## Changes Summary

### extract_binance_data.py
1. Added `from zoneinfo import ZoneInfo` and `DISPLAY_TZ = ZoneInfo("Asia/Singapore")`
2. Fixed `MAX_ROWS_PER_REQUEST["price"]` from 1500 to 1000
3. Added retry logic with exponential backoff in `fetch_api_data()`
4. Added startTime validation for 30-day limited APIs
5. Added `last_timestamp` field to extraction results
6. Reduced inter-request delay from 100ms to 50ms
7. Updated log messages to show GMT+8 timestamps

### refresh_data.py
1. Added `import json` and `from zoneinfo import ZoneInfo`
2. Added `DISPLAY_TZ = ZoneInfo("Asia/Singapore")` constant
3. Added `save_last_timestamps()` function
4. Updated `quick_refresh()` to show GMT+8 timestamps and save last_timestamps.json
5. Updated `full_refresh()` to save last_timestamps.json
6. Updated `show_status()` to show GMT+8 and save last_timestamps.json

---

## Testing

To verify changes:
```bash
# Check syntax
python -m py_compile extract_binance_data.py
python -m py_compile refresh_data.py

# Run status check
python refresh_data.py --status

# Run quick refresh
python refresh_data.py

# Check last_timestamps.json format
cat data/last_timestamps.json
```
