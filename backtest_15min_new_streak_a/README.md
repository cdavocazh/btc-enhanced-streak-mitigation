# Adaptive Streak Reduction Strategies

This folder contains advanced BTC trading strategies focused on reducing losing streaks through adaptive filtering mechanisms.

## Quick Start

### Run the Dashboard

```bash
streamlit run backtest_15min_new_streak_a/dashboard_adaptive.py
```

### Run the Backtest

```bash
python backtest_15min_new_streak_a/run_adaptive_streak_backtest.py
```

## Strategies

| Strategy | Description | Key Features |
|----------|-------------|--------------|
| **Adaptive_Baseline** | Full adaptive system | ADX filter + Progressive Pos + Cooldown |
| **Adaptive_Conservative** | Most protective | Above + Multi-TP + BE protection |
| **Adaptive_ADX_Only** | Market regime filter | Only ADX trending/ranging detection |
| **Adaptive_ProgPos_Only** | Progressive thresholds | Dynamic positioning based on losses |

## Key Concepts

### 1. ADX Market Regime Filter

The ADX (Average Directional Index) helps identify market conditions:

- **ADX > 20**: Trending market - good for trend-following entries
- **ADX < 20**: Ranging market - avoid entries or use stricter filters
- **ADX > 30**: Strong trend - high confidence entries

```python
# Only enter in trending markets
if adx > 20:
    allow_entry = True
```

### 2. Progressive Positioning Threshold

Positioning requirements increase after consecutive losses:

| Consecutive Losses | Min Positioning Score |
|--------------------|----------------------|
| 0 | 0.40 |
| 1 | 0.50 |
| 2 | 0.60 |
| 3 | 0.80 |
| 4+ | 1.00 |

This forces the strategy to wait for stronger signals after losses.

### 3. Cooldown Period

After 5+ consecutive losses, the strategy enters a 24-hour cooldown:

- No entries allowed during cooldown
- Prevents revenge trading
- Allows market conditions to stabilize

### 4. Volume Regime Adaptation

Volume requirements adjust based on market regime:

- **Trending market**: `vol_ratio >= 0.7`
- **Ranging market**: `vol_ratio >= 1.2` (stricter)

## Dashboard Features

### Real-time Monitoring

- **ADX Indicator**: Current value + trending/ranging status
- **Positioning Score**: Live calculation from Binance data
- **Volume Ratio**: Compared to 24-hour MA
- **RSI**: Overbought/oversold detection

### Progressive Threshold Visualizer

- Interactive slider to simulate different loss counts
- See how thresholds change dynamically
- Visual comparison of current positioning vs required

### Entry Condition Checker

- All conditions displayed with ✅/❌ status
- Strategy-specific filters applied
- Clear indication when entry is valid

### Performance Comparison

- Side-by-side strategy metrics
- Return vs drawdown visualization
- Streak statistics for each strategy

## Performance Summary

| Strategy | Return | Max DD | Win Rate | Max Streak |
|----------|--------|--------|----------|------------|
| Adaptive_ProgPos_Only | 1754% | 24.1% | 43.6% | 5 |
| Adaptive_Baseline | 642% | 27.0% | 45.2% | 7 |
| Adaptive_ADX_Only | 551% | 33.7% | 38.9% | 9 |
| Adaptive_Conservative | 354% | 24.8% | 47.7% | 5 |

## Files

| File | Description |
|------|-------------|
| `run_adaptive_streak_backtest.py` | Main backtest script |
| `dashboard_adaptive.py` | Streamlit monitoring dashboard |
| `comparison_report.html` | Strategy comparison charts |
| `strategy_guide.html` | Interactive strategy documentation |
| `STATUS.md` | Current status and findings |
| `README.md` | This file |

### Results Directory

```
results/
├── strategy_statistics.json    # Performance metrics
├── streak_statistics.json      # Losing streak analysis
├── equity_*.csv               # Equity curves per strategy
├── trades_*.csv               # Trade logs per strategy
└── losing_streaks_*.csv       # Streak details
```

## Configuration

### Entry Filter

```python
ADAPTIVE_ENTRY_FILTER = {
    'min_pos_long_base': 0.4,
    'rsi_long_range': [20, 45],
    'pullback_range': [0.5, 3.0],
    'min_pos_score': 0.15,

    # ADX filter
    'adx_trending_threshold': 20,
    'adx_strong_trend_threshold': 30,

    # Cooldown
    'cooldown_after_losses': 5,
    'cooldown_bars': 96,  # 24 hours

    # Volume
    'vol_min_trending': 0.7,
    'vol_min_ranging': 1.2,
}
```

### Tiered Risk

The strategies use tiered position sizing based on equity:

| Equity Range | Risk Amount |
|--------------|-------------|
| $0 - $150k | $5,000 |
| $150k - $225k | $10,500 |
| $225k - $337.5k | $14,625 |
| $337.5k - $507k | $20,250 |
| $507k - $760k | $28,000 |
| $760k - $1.2M | $38,000 |
| $1.2M+ | $54,000 |

### Streak Mitigation

| Streak Level | Risk Reduction |
|--------------|----------------|
| 3 losses | -40% |
| 6 losses | Additional -30% (total ~58%) |
| 9 losses | Additional -30% (total ~71%) |

## Root Cause Analysis

The adaptive filters address these causes of losing streaks:

| Cause | Percentage | Solution |
|-------|------------|----------|
| Choppy range-bound markets | 45% | ADX filter |
| Trend reversals | 30% | HH/HL confirmation |
| Low volatility compression | 25% | Adaptive ATR |

## Dependencies

```bash
pip install streamlit pandas numpy plotly
```

## Integration with Eval Framework

The strategies in this folder are tracked by the evaluation framework:

```bash
# Run walk-forward evaluation
cd eval && python run_evaluation.py --full

# Check performance trends
cd eval && python run_evaluation.py --adapt
```

## Recommendations

### For Maximum Returns
Use **Adaptive_ProgPos_Only** - 1754% return with acceptable 24% drawdown.

### For Controlled Risk
Use **Adaptive_Conservative** - 354% return with best risk-adjusted metrics and 48% win rate.

### For Production Trading
Use **Adaptive_Baseline** - balanced approach with all protective filters enabled.
