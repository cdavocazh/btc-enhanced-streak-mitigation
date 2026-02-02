# CLAUDE.md

This file provides guidance for Claude Code (claude.ai/code) when working with this repository.

## Project Overview

BTC Enhanced Strategy with Streak Mitigation - a BTC trading backtesting system that uses Binance Futures positioning data to generate trading signals with streak mitigation techniques to reduce consecutive losses.

## Repository Structure

```
btc-enhanced-streak-mitigation/
├── run_backtest.py              # Main backtest with optimized config (hourly)
├── load_binance_data.py         # Binance data loader
├── BTC_OHLC_1h_gmt8_updated.csv # Hourly price data (GMT+8)
├── backtest_15min_new/          # 15-minute backtests with tiered capital
├── backtest_15min_new_streak_a/ # Adaptive streak reduction strategies (ADX, ProgPos)
├── binance-futures-data/        # Live Binance Futures data extraction (auto-refresh)
├── validation/                  # Monte Carlo and walk-forward validation
└── telegram_signals/            # Telegram bot solutions
```

## Key Commands

### Run Backtests
```bash
# Main hourly backtest
python run_backtest.py

# 15-minute parameter experiments
python backtest_15min_new/run_parameter_experiments.py

# Tiered capital + streak mitigation
python backtest_15min_new/run_tiered_streak_mitigation.py

# Adaptive streak reduction (ADX filter, progressive positioning)
python backtest_15min_new_streak_a/run_adaptive_streak_backtest.py
```

### Run Dashboards
```bash
# Asian hours dashboard (0-11 UTC)
streamlit run backtest_15min_new/dashboard_asian_hours.py

# All hours dashboard
streamlit run backtest_15min_new/dashboard_all_hours.py
```

### Data Refresh (Binance Futures)
```bash
# Quick refresh (5-min incremental)
python binance-futures-data/refresh_data.py

# Full backfill with gap filling
python binance-futures-data/refresh_data.py --full

# Check data status
python binance-futures-data/refresh_data.py --status
```

### Validation
```bash
python validation/monte_carlo_validation.py
python validation/walk_forward_optimization.py
```

## Architecture

### Data Flow
1. **Price Data**: `BTC_OHLC_1h_gmt8_updated.csv` (hourly) or 15-min data loaded via `load_15min_data.py`
2. **Positioning Data**: Binance Futures API via `binance-futures-data/extract_binance_data.py`
3. **Backtest**: Combine price + positioning to generate signals and simulate trades

### Key Strategy Parameters
- **Positioning Method**: TopTraderFocused (weight: 1.5x for strong signals)
- **Entry Threshold**: 1.5 (minimum positioning score)
- **Skip Neutral**: 0.25 (avoid trades when |score| < 0.25)
- **Strict Entry**: After 3 losses, require score >= 0.5

### Tiered Risk Configuration
| Equity Range | Risk Amount |
|--------------|-------------|
| $0-$150k | $5,000 |
| $150k-$225k | $10,500 |
| $225k-$337.5k | $14,625 |
| $337.5k-$507k | $20,250 |
| $507k-$760k | $28,000 |
| $760k-$1.2M | $38,000 |
| $1.2M+ | $54,000 |

### Streak Mitigation Rules
| Streak Level | Risk Reduction |
|--------------|----------------|
| 3 consecutive losses | -40% |
| 6 consecutive losses | Additional -30% (total ~58%) |
| 9 consecutive losses | Additional -30% (total ~71%) |

## Data Infrastructure

### Binance Futures Data (auto-refresh every 5 minutes via launchd)

**Metrics collected:**
- Top trader position L/S ratio
- Top trader account L/S ratio
- Global trader L/S ratio
- Funding rate (8-hour intervals)
- Open Interest
- OHLC price (5-minute candles)

**LaunchD Status:**
```bash
# Check if running
launchctl list | grep btc

# View recent logs
tail -50 binance-futures-data/logs/launchd.log

# Reload after changes
launchctl unload ~/Library/LaunchAgents/com.btc.datarefresh.plist
launchctl load ~/Library/LaunchAgents/com.btc.datarefresh.plist
```

### CSV Validation
The data extraction includes auto-fix for:
- Empty/corrupted CSV files
- Malformed timestamps (e.g., "1-03" instead of "2026-01-03")
- Duplicate rows

## Important Files

| File | Purpose |
|------|---------|
| `run_backtest.py` | Main backtest configuration and execution |
| `load_binance_data.py` | Data loading utilities for positioning metrics |
| `binance-futures-data/extract_binance_data.py` | Core API extraction with CSV validation |
| `binance-futures-data/refresh_data.py` | Data refresh orchestration (quick/full modes) |
| `backtest_15min_new/status.md` | Detailed results for 15-min backtests |
| `backtest_15min_new_streak_a/results/strategy_statistics.json` | Adaptive strategy performance |

## Performance Benchmarks

### Best Strategies (Asian Hours, 15-min)
| Strategy | Return | Max DD | Win Rate |
|----------|--------|--------|----------|
| Adaptive_ProgPos_Only | 1754% | 24.1% | 43.6% |
| Adaptive_Baseline | 642% | 27.0% | 45.2% |
| Adaptive_Conservative | 354% | 24.8% | 47.7% |

### API Rate Limits
- Binance: 900 requests per 5 minutes
- Script uses 10% safety margin (810 effective limit)
- Normal refresh: 10-16 API calls per cycle (~15-20 seconds)
