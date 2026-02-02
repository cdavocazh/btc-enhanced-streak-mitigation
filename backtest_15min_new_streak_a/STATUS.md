# Adaptive Streak Reduction Backtest - Status

## Overview
This folder implements advanced streak mitigation strategies based on the analysis from `losing_streak_analysis.html`. The strategies address three root causes of losing streaks:

- **45% of streaks**: Choppy range-bound markets (ADX filter)
- **30% of streaks**: Trend reversals (Higher Highs/Lower Lows check)
- **25% of streaks**: Low volatility compression (Adaptive ATR)

## Strategy Configurations

### 1. Adaptive_Baseline
Full adaptive system with all filters enabled:
- ADX filter (trending threshold: 20)
- Progressive positioning threshold (0.4 → 1.0 based on losses)
- 24-hour cooldown after 5+ consecutive losses

### 2. Adaptive_Conservative
Baseline with reduced risk:
- Higher ADX threshold (25)
- Tighter positioning requirements
- Lower risk tiers

### 3. Adaptive_ADX_Only
ADX market regime filter only:
- No progressive positioning
- No cooldown periods
- Tests pure regime filtering

### 4. Adaptive_ProgPos_Only
Progressive positioning threshold only:
- No ADX filter
- Threshold increases: 0→0.4, 1→0.5, 2→0.6, 3→0.8, 4+→1.0
- Cooldown after 5+ losses

### 5. Sequential_Regime
Most restrictive sequential filter:
- ADX > 25 required
- Higher Highs/Lower Lows confirmation
- Position momentum check

---

## Performance Summary (Asian Hours, Tiered Capital)

| Strategy | Return | Max DD | Trades | Win Rate | Profit Factor |
|----------|--------|--------|--------|----------|---------------|
| **Adaptive_ProgPos_Only** | 1754.5% | 24.1% | 94 | 43.6% | 2.06 |
| Adaptive_Baseline | 641.9% | 27.0% | 62 | 45.2% | 2.44 |
| Adaptive_ADX_Only | 551.5% | 33.7% | 72 | 38.9% | 2.35 |
| Adaptive_Conservative | 353.5% | 24.8% | 65 | 47.7% | 1.64 |
| Sequential_Regime | 18.3% | 8.9% | 4 | 50.0% | 2.83 |

---

## Streak Statistics

| Strategy | Streaks 3+ | Streaks 6+ | Streaks 9+ | Max Streak | Total Streak Loss |
|----------|------------|------------|------------|------------|-------------------|
| **Adaptive_Conservative** | 4 | 0 | 0 | 5 | -$121,928 |
| Adaptive_Baseline | 4 | 1 | 0 | 7 | -$221,332 |
| Adaptive_ADX_Only | 5 | 3 | 1 | 9 | -$224,662 |
| Adaptive_ProgPos_Only | 8 | 0 | 0 | 5 | -$1,064,328 |
| Sequential_Regime | 0 | 0 | 0 | 0 | $0 |

---

## Key Findings

### 1. Progressive Positioning is the Winner
**Adaptive_ProgPos_Only** achieves the highest return (1754%) with reasonable drawdown (24.1%). The progressive threshold effectively:
- Prevents overtrading after losses
- Requires stronger signals during losing periods
- Cooldown period prevents revenge trading

### 2. ADX Filter Trade-off
ADX filtering reduces trade count but doesn't eliminate long streaks on its own. Works best when combined with other filters.

### 3. Best Risk-Adjusted Strategy
**Adaptive_Conservative** has the best streak characteristics:
- Lowest max streak (5 losses)
- No 6+ or 9+ streaks
- Lowest streak loss (-$122k)
- 47.7% win rate

### 4. Sequential Regime is Too Restrictive
Only 4 trades in the backtest period - filters are too aggressive for practical use.

---

## Filter Effectiveness

| Filter Type | Trades Blocked | Purpose |
|-------------|----------------|---------|
| ADX Filter | ~27,500 | Avoid range-bound markets |
| Positioning Threshold | ~10,500 | Require strong positioning signals |
| Cooldown Period | 48-192 | Prevent revenge trading |
| HH/HL Check | 0-27,411 | Confirm trend direction |
| Position Momentum | 0-603 | Verify momentum alignment |

---

## Files

| File | Description |
|------|-------------|
| `run_adaptive_streak_backtest.py` | Main backtest script |
| `comparison_report.html` | Strategy comparison charts |
| `strategy_guide.html` | Interactive strategy guide |
| `results/strategy_statistics.json` | Performance metrics |
| `results/streak_statistics.json` | Losing streak analysis |
| `results/equity_*.csv` | Equity curves |
| `results/trades_*.csv` | Trade logs |
| `results/losing_streaks_*.csv` | Individual streak details |

---

## Recommendations

### For Maximum Returns
Use **Adaptive_ProgPos_Only** with tiered capital. Accept higher absolute drawdown for 17x returns.

### For Controlled Risk
Use **Adaptive_Conservative**. Best risk-adjusted returns with max 5-loss streak and ~25% drawdown.

### For Production Trading
Consider **Adaptive_Baseline** - balanced approach with ADX + progressive positioning + cooldown.

---

## Pending Work
- [ ] Live trading integration
- [ ] Real-time signal alerts
- [ ] Multi-timeframe confirmation
- [ ] Dynamic ADX threshold based on volatility regime
