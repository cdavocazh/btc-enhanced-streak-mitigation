# BTC Enhanced Strategy - Backtest 15min New

## Overview
This folder contains the parameter optimization and dashboard system for the BTC Enhanced Streak Mitigation strategy, with separate versions for **Asian Hours** (0-11 UTC) and **All Hours** trading sessions.

## Files

### Core Scripts
| File | Description |
|------|-------------|
| `run_parameter_experiments.py` | Parameter optimization script testing ATR, RSI, SMA periods and entry filters |
| `run_tiered_capital_backtest.py` | Tiered capital risk management backtest |
| `run_tiered_streak_mitigation.py` | Tiered capital + losing streak risk reduction backtest |
| `generate_reports.py` | HTML report generator with equity curves using Plotly.js |
| `dashboard_asian_hours.py` | Streamlit dashboard for Asian Hours version (0-11 UTC) |
| `dashboard_all_hours.py` | Streamlit dashboard for All Hours version |

### Results Directories
| Directory | Description |
|-----------|-------------|
| `results_asian_hours/` | Backtest results for Asian Hours trading (0-11 UTC) |
| `results_all_hours/` | Backtest results for All Hours trading |
| `results_tiered_capital/` | Tiered capital backtest results |
| `results_tiered_streak/` | Tiered capital + streak mitigation results |

---

## Strategy Performance Summary

### Asian Hours Version (0-11 UTC)
| Strategy | Return | Max DD | Trades | Win Rate | Sharpe | Profit Factor |
|----------|--------|--------|--------|----------|--------|---------------|
| Baseline_AHours | 421.4% | 27.2% | 157 | 37.6% | 1.57 | 1.64 |
| PosVol_Combined_AHours | 348.9% | 22.2% | 180 | 44.4% | 1.40 | 1.75 |
| MultiTP_30_AHours | 349.6% | 21.5% | 157 | 37.6% | 1.75 | 1.27 |
| VolFilter_Adaptive_AHours | 331.9% | 31.5% | 95 | 43.2% | 1.25 | 2.14 |
| Conservative_AHours | 311.6% | 21.5% | 100 | 47.0% | 1.75 | 1.45 |

### All Hours Version
| Strategy | Return | Max DD | Trades | Win Rate | Sharpe | Profit Factor |
|----------|--------|--------|--------|----------|--------|---------------|
| Baseline_AllHours | 514.2% | 36.3% | 262 | 34.4% | 1.47 | 1.44 |
| PosVol_Combined_AllHours | 418.2% | 24.2% | 317 | 39.1% | 1.34 | 1.45 |
| MultiTP_30_AllHours | 400.8% | 32.3% | 261 | 34.1% | 1.54 | 1.10 |
| VolFilter_Adaptive_AllHours | 346.8% | 32.8% | 211 | 35.5% | 1.09 | 1.45 |
| Conservative_AllHours | 378.3% | 28.3% | 208 | 42.3% | 1.54 | 1.09 |

---

## Optimal Parameters Found
- **ATR Period**: 14
- **RSI Period**: 56
- **SMA Period**: 200

---

## Strategy Descriptions

### 1. Baseline
The core strategy with standard position sizing (20% risk per trade). Uses positioning score, volume score, RSI, and trend filters.

### 2. PosVol_Combined
Combines positioning and volume scores with partial position exits. More conservative entry filters with higher win rate target.

### 3. MultiTP_30
Multi-take-profit strategy with 30% exit at TP1 (1.5R), remainder at TP2 (3R). Better for capturing partial profits on volatile moves.

### 4. VolFilter_Adaptive
Adaptive volume filtering with higher thresholds. Fewer trades but higher win/loss ratio (2.8+).

### 5. Conservative
Reduced position size (10% risk), tighter stops. Highest win rate (~47%) with lower volatility.

---

## Dashboard Features

### Both Dashboards Include:
- **Real-time indicator scores**: Positioning, Volume, RSI, Trend
- **BTC price/volume charts**: 24h price with Bollinger Bands, SMA200, RSI
- **Entry condition checking**: For all 5 strategies
- **Trade level calculations**: Entry, Stop Loss, Take Profit levels
- **Refresh button**: Manual data refresh

### Asian Hours Dashboard (Additional):
- **Session status indicator**: Shows if currently in Asian trading session
- **Chart highlighting**: Asian hours (0-11 UTC) shaded on price charts

---

## Running the Dashboards

```bash
# Asian Hours Dashboard
streamlit run backtest_15min_new/dashboard_asian_hours.py

# All Hours Dashboard
streamlit run backtest_15min_new/dashboard_all_hours.py
```

---

## Data Files in Results Directories

Each results directory contains:
- `btc_price_15min.csv` - Historical BTC price data
- `parameter_grid_results.json` - Parameter optimization results
- `entry_filter_results.json` - Entry filter comparison results
- `strategy_statistics.json` - Final strategy statistics
- `strategy_report.html` - Interactive HTML report with equity curves
- `equity_*.csv` - Equity curve data for each strategy
- `trades_*.csv` - Trade log for each strategy

---

## Tiered Capital Backtest Results (Asian Hours)

Tiered risk management adjusts position sizing based on equity levels:

| Equity Range | Risk Per Trade |
|--------------|----------------|
| $0 - $150k | $5,000 (5% of $100k) |
| $150k - $225k | $10,500 (7% of $150k) |
| $225k - $337.5k | $14,625 (6.5% of $225k) |
| $337.5k - $507k | $20,250 (6% of $337.5k) |
| $507k - $760k | $28,000 (5.5% of $507k) |
| $760k - $1.2m | $38,000 (fixed) |
| $1.2m+ | $54,000 (fixed) |

### Tiered Capital Performance (Asian Hours)
| Strategy | Return | Final Equity | Max DD | Avg Risk |
|----------|--------|--------------|--------|----------|
| Baseline_Tiered | 1333.2% | $1,433,201 | 51.4% | $26,733 |
| MultiTP_30_Tiered | 976.9% | $1,076,893 | 42.5% | $23,604 |
| PosVol_Combined_Tiered | 865.6% | $965,574 | 35.1% | $15,991 |
| VolFilter_Adaptive_Tiered | 809.3% | $909,261 | 32.3% | $12,907 |
| Conservative_Tiered | 777.3% | $877,284 | 30.3% | $15,152 |

### Comparison: Tiered vs Fixed Risk
| Strategy | Fixed Return | Tiered Return | Fixed Max DD | Tiered Max DD |
|----------|--------------|---------------|--------------|---------------|
| Baseline | 421.4% | **1333.2%** | 27.2% | 51.4% |
| MultiTP_30 | 349.6% | **976.9%** | 21.5% | 42.5% |
| PosVol_Combined | 348.9% | **865.6%** | 22.2% | 35.1% |
| VolFilter_Adaptive | 331.9% | **809.3%** | 31.5% | 32.3% |
| Conservative | 311.6% | **777.3%** | 21.5% | 30.3% |

**Key Insight**: Tiered capital significantly amplifies returns (2-3x) but also increases drawdowns. Best risk-adjusted tiered strategy is **Conservative_Tiered** with 777% return and only 30.3% max drawdown.

---

## Tiered Capital + Streak Mitigation Results (Asian Hours)

Streak mitigation reduces position size during losing streaks:

| Streak Level | Risk Reduction | Recovery After Win |
|--------------|----------------|-------------------|
| 3 consecutive losses | -40% | Return to initial tiered risk |
| 6 consecutive losses | -30% additional (total ~58%) | Return to initial tiered risk |
| 9 consecutive losses | -30% additional (total ~70.6%) | Return to 5% of capital |

### Tiered + Streak Mitigation Performance
| Strategy | Return | Final Equity | Max DD | Streaks 3+ | Reduced Trades |
|----------|--------|--------------|--------|------------|----------------|
| Baseline_TieredStreak | 1456.3% | $1,556,291 | 36.9% | 13 | 44 (28.0%) |
| MultiTP_30_TieredStreak | 846.7% | $946,717 | 36.4% | 13 | 44 (28.0%) |
| Conservative_TieredStreak | 775.1% | $875,150 | 25.0% | 8 | 15 (15.0%) |
| VolFilter_Adaptive_TieredStreak | 728.3% | $828,295 | 28.8% | 8 | 15 (15.8%) |
| PosVol_Combined_TieredStreak | 544.6% | $644,638 | 35.1% | 17 | 29 (16.1%) |

### Comparison: Tiered vs Tiered + Streak Mitigation
| Strategy | Tiered Return | +Streak Return | Tiered MaxDD | +Streak MaxDD |
|----------|---------------|----------------|--------------|---------------|
| Baseline | 1333.2% | **1456.3%** | 51.4% | **36.9%** |
| MultiTP_30 | 976.9% | 846.7% | 42.5% | **36.4%** |
| Conservative | 777.3% | 775.1% | 30.3% | **25.0%** |
| VolFilter_Adaptive | 809.3% | 728.3% | 32.3% | **28.8%** |
| PosVol_Combined | 865.6% | 544.6% | 35.1% | 35.1% |

### Losing Streak Statistics
| Strategy | Streaks 3+ | Streaks 6+ | Streaks 9+ | Max Streak | Total Loss |
|----------|------------|------------|------------|------------|------------|
| Baseline | 13 | 4 | 1 | 13 | $2,235,712 |
| MultiTP_30 | 13 | 4 | 1 | 13 | $1,591,736 |
| PosVol_Combined | 17 | 1 | 0 | 6 | $709,369 |
| Conservative | 8 | 1 | 0 | 7 | $488,329 |
| VolFilter_Adaptive | 8 | 1 | 0 | 6 | $340,850 |

**Key Insight**: Streak mitigation significantly reduces drawdowns (up to 15% improvement) while maintaining or even improving returns for Baseline. Best strategy is **Conservative_TieredStreak** with 775% return and only **25% max drawdown**.

---

## Key Findings

1. **Asian Hours vs All Hours**: Asian Hours version has better risk-adjusted returns (lower drawdowns) but All Hours has higher absolute returns.

2. **Best Risk/Return (Fixed)**: `PosVol_Combined_AHours` offers best balance with 348.9% return, 22.2% max DD, and 1.75 profit factor.

3. **Best Risk/Return (Tiered)**: `Conservative_Tiered` achieves 777% return with only 30.3% max DD.

4. **Highest Absolute Return**: `Baseline_Tiered` achieves 1333% return ($1.43M final equity).

5. **Highest Win Rate**: `Conservative` strategies achieve 47.0% win rate across all versions.

---

## Pending Work
- [x] Tiered capital backtest for Asian hours strategies
- [x] Tiered capital + streak mitigation backtest
- [ ] Live trading integration
- [ ] Alert system for entry signals
