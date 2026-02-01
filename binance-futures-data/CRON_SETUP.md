# Cron Job Setup for Automated Data Refresh

This document explains how to set up automated data refresh using cron.

## Quick Setup

```bash
cd /Users/kris.zhang/Github/btc-enhanced-streak-mitigation/binance-futures-data
chmod +x setup_cron.sh
./setup_cron.sh
```

## What the Cron Job Does

- Runs `refresh_data.py` every 5 minutes
- Fetches latest data from Binance Futures API
- Appends new data to CSV files (no duplicates)
- Logs output to `refresh.log`

## Manual Setup (Alternative)

If you prefer to set up the cron job manually:

### 1. Edit crontab
```bash
crontab -e
```

### 2. Add this line
```cron
*/5 * * * * cd /Users/kris.zhang/Github/btc-enhanced-streak-mitigation/binance-futures-data && python refresh_data.py >> refresh.log 2>&1 # btc_data_refresh
```

### 3. Save and exit

## Managing the Cron Job

### Check Status
```bash
./setup_cron.sh --status
```

This shows:
- Current cron configuration
- Last 10 lines of refresh log

### View Logs
```bash
# Live log monitoring
tail -f refresh.log

# Last 50 lines
tail -50 refresh.log
```

### Remove Cron Job
```bash
./setup_cron.sh --remove
```

### Reinstall Cron Job
```bash
./setup_cron.sh
```

## Cron Schedule Explained

```
*/5 * * * *
│   │ │ │ │
│   │ │ │ └── Day of week (0-7, Sunday=0 or 7)
│   │ │ └──── Month (1-12)
│   │ └────── Day of month (1-31)
│   └──────── Hour (0-23)
└────────────  Minute (0-59, */5 = every 5 minutes)
```

## Using a Virtual Environment

If you're using a Python virtual environment, modify the cron entry:

```cron
*/5 * * * * cd /path/to/binance-futures-data && /path/to/venv/bin/python refresh_data.py >> refresh.log 2>&1 # btc_data_refresh
```

Or edit `setup_cron.sh` and change the `PYTHON_PATH` variable:
```bash
PYTHON_PATH="/path/to/venv/bin/python"
```

## Troubleshooting

### Cron job not running

1. **Check if cron service is running:**
   ```bash
   # macOS
   sudo launchctl list | grep cron

   # Linux
   systemctl status cron
   ```

2. **Verify cron is installed:**
   ```bash
   crontab -l
   ```

3. **Check system logs:**
   ```bash
   # macOS
   log show --predicate 'process == "cron"' --last 1h

   # Linux
   grep CRON /var/log/syslog
   ```

### Permission issues

Make sure the script is executable:
```bash
chmod +x setup_cron.sh
chmod +x refresh_data.py
```

### Python not found

If cron can't find Python, use the full path:
```bash
which python
# Then use this path in setup_cron.sh
```

### Log file growing too large

Add log rotation:
```bash
# Keep only last 1000 lines
tail -1000 refresh.log > refresh.log.tmp && mv refresh.log.tmp refresh.log
```

Or add to crontab (daily at midnight):
```cron
0 0 * * * tail -1000 /path/to/refresh.log > /path/to/refresh.log.tmp && mv /path/to/refresh.log.tmp /path/to/refresh.log
```

## Expected Log Output

Each 5-minute run should produce output like:
```
[2026-01-30 10:05:00 UTC] ============================================================
[2026-01-30 10:05:00 UTC] QUICK DATA REFRESH
[2026-01-30 10:05:00 UTC] Time: 2026-01-30 10:05:00 UTC
[2026-01-30 10:05:00 UTC] ============================================================
[2026-01-30 10:05:00 UTC] API calls available: 880/900
[2026-01-30 10:05:00 UTC] price: Last update 5.2 min ago
[2026-01-30 10:05:01 UTC]   Added 1 rows (1 API calls)
[2026-01-30 10:05:01 UTC] top_trader_position: Last update 5.1 min ago
[2026-01-30 10:05:02 UTC]   Added 1 rows (1 API calls)
...
[2026-01-30 10:05:05 UTC] ------------------------------------------------------------
[2026-01-30 10:05:05 UTC] Total: 5 rows added, 5 API calls
[2026-01-30 10:05:05 UTC] API usage: 25/900
[2026-01-30 10:05:05 UTC] ============================================================
[2026-01-30 10:05:05 UTC] Done
```

## Verifying Data Updates

After the cron job runs, verify data freshness:
```bash
python refresh_data.py --status
```

Expected output:
```
[2026-01-30 10:10:00 UTC] ============================================================
[2026-01-30 10:10:00 UTC] DATA STATUS
[2026-01-30 10:10:00 UTC] ============================================================
[2026-01-30 10:10:00 UTC] API Usage (5min): 25/900
[2026-01-30 10:10:00 UTC]
[2026-01-30 10:10:00 UTC] Data Coverage:
[2026-01-30 10:10:00 UTC] ------------------------------------------------------------
[2026-01-30 10:10:00 UTC]   price                         2,500 rows | last: 2026-01-30 10:05 (5.0 min ago)
[2026-01-30 10:10:00 UTC]   top_trader_position         438,000 rows | last: 2026-01-30 10:05 (5.0 min ago)
...
```

All metrics should show "X min ago" where X is less than 10 minutes for a healthy system.
