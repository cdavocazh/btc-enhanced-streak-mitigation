# Binance Futures Data Infrastructure

## Overview

This directory contains scripts for extracting, storing, and refreshing BTCUSDT perpetual futures data from Binance. The data is used by the trading signal dashboard for real-time analysis and backtesting.

## Directory Structure

```
binance-futures-data/
├── data/                           # Data storage directory
│   ├── price.csv                   # OHLC price data (5-min candles)
│   ├── top_trader_position_ratio.csv
│   ├── top_trader_account_ratio.csv
│   ├── global_ls_ratio.csv
│   ├── funding_rate.csv
│   ├── open_interest.csv
│   ├── api_call_log.json           # Rate limit tracking
│   └── metrics_combined.csv        # Legacy combined metrics
├── extract_binance_data.py         # Core extraction logic
├── refresh_data.py                 # Unified refresh script
├── backfill_data.py                # Historical backfill
├── setup_cron.sh                   # Cron job setup
├── STATUS.md                       # This file
└── CRON_SETUP.md                   # Cron job instructions
```

---

## Scripts

### 1. `extract_binance_data.py`

**Purpose:** Core data extraction module that fetches data from Binance Futures API.

**Data Extracted (5-minute intervals):**
| Metric | Endpoint | Data Availability |
|--------|----------|-------------------|
| `price` | `/fapi/v1/klines` | 2+ years |
| `top_trader_position` | `/futures/data/topLongShortPositionRatio` | 30 days |
| `top_trader_account` | `/futures/data/topLongShortAccountRatio` | 30 days |
| `global_ls_ratio` | `/futures/data/globalLongShortAccountRatio` | 30 days |
| `funding_rate` | `/fapi/v1/fundingRate` | 2+ years |
| `open_interest` | `/futures/data/openInterestHist` | 30 days |

**Key Features:**
- Paginated requests to fetch maximum data within rate limits
- Gap detection and filling for historical records
- Append-only storage to avoid duplicates
- Rate limit tracking with 10% safety margin (900/1000 requests per 5 min)

**Usage:**
```bash
# Import as a module (used by refresh_data.py)
from extract_binance_data import extract_metric_data, get_data_coverage

# Direct execution (incremental fetch)
python extract_binance_data.py
```

---

### 2. `refresh_data.py`

**Purpose:** Unified refresh script designed for 5-minute cron cycles.

**Modes:**
| Mode | Flag | Description |
|------|------|-------------|
| Quick | (default) | Fast 5-minute refresh, prioritizes price data |
| Full | `--full` | Complete backfill with gap filling |
| Status | `--status` | Show data freshness without fetching |

**Usage:**
```bash
# Quick refresh (for cron jobs)
python refresh_data.py

# Full refresh with gap filling
python refresh_data.py --full

# Check current data status
python refresh_data.py --status
```

**Priority Order:**
1. `price` - Most important for live signals
2. `top_trader_position` - Key positioning metric
3. `global_ls_ratio` - Market sentiment
4. `open_interest` - Market activity
5. `top_trader_account` - Secondary positioning
6. `funding_rate` - Updates every 8 hours

---

### 3. `backfill_data.py`

**Purpose:** Historical data backfill from two sources.

**Data Sources:**
| Source | Date Range | Rate Limits |
|--------|------------|-------------|
| Binance Vision Archive | 2021-12-01 to 2023-04-14 | None (public downloads) |
| Binance Futures API | Rolling 30 days | 900 req/5min |

**Usage:**
```bash
# Run backfill (default 20 min duration)
python backfill_data.py

# Custom duration
python backfill_data.py --duration 60
```

---

### 4. `setup_cron.sh`

**Purpose:** Manages cron job for automated data refresh.

**See:** [CRON_SETUP.md](./CRON_SETUP.md) for detailed instructions.

---

## Data Files

### CSV Format

All CSV files use consistent format:
- UTF-8 encoding
- Comma-separated
- ISO 8601 timestamps with UTC timezone
- Sorted by timestamp ascending

### Schema

**price.csv:**
```
timestamp, symbol, open, high, low, close, volume, quote_volume
```

**top_trader_position_ratio.csv / top_trader_account_ratio.csv / global_ls_ratio.csv:**
```
timestamp, symbol, long_short_ratio, long_account, short_account
```

**funding_rate.csv:**
```
timestamp, symbol, funding_rate, mark_price
```

**open_interest.csv:**
```
timestamp, symbol, sum_open_interest, sum_open_interest_value
```

---

## API Rate Limiting

Binance Futures API has the following limits:
- **1000 requests per 5 minutes** (IP-based)
- We apply a **10% safety margin** = 900 effective requests

Rate limit tracking is stored in `data/api_call_log.json`:
```json
{
  "calls": [
    {"timestamp": 1706543210.123, "endpoint": "/fapi/v1/klines", "success": true},
    ...
  ]
}
```

**Check current usage:**
```bash
python refresh_data.py --status
```

---

## Integration with Dashboard

The `trading_signal_dashboard.py` uses this data infrastructure:

1. **Data Loading:** Reads from `data/*.csv` files
2. **Refresh Button:** Triggers `refresh_data.py` via subprocess
3. **Auto-Refresh:** Clears cache and reloads every 5 minutes
4. **Freshness Indicator:** Shows how recent the data is

---

## Troubleshooting

### Data is stale (> 10 minutes old)
```bash
# Check API rate limit status
python refresh_data.py --status

# Force refresh
python refresh_data.py
```

### Missing historical data
```bash
# Run full backfill
python backfill_data.py --duration 60
```

### API rate limit exceeded
Wait 5 minutes for the rate limit window to reset. The scripts automatically track and respect rate limits.

### Cron job not running
```bash
./setup_cron.sh --status
```

---

## Changelog

- **2026-01-30:** Added OHLC price data extraction from `/fapi/v1/klines`
- **2026-01-30:** Created unified `refresh_data.py` script
- **2026-01-30:** Added auto-refresh to dashboard (5-minute interval)
- **2026-01-30:** Created `setup_cron.sh` for automated updates
