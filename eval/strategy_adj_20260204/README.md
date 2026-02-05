# Strategy Adjustment Experiment - 2026-02-04

## Overview

This folder contains parameter adjustment experiments based on the first walk-forward evaluation that showed all strategies in CRITICAL status (OOS Efficiency < 0.50).

## Problem Statement

The initial evaluation on 2026-02-04 revealed:
- All 6 strategies had OOS Efficiency of 0.45-0.48 (below 0.60 threshold)
- Win rate was 0% in evaluation (entry conditions too restrictive)
- Adaptive strategies performed slightly worse than non-adaptive (0.45 vs 0.48)
- ADX filter may have been blocking valid trades

## Experiments Conducted

| Experiment | Description | Key Changes |
|------------|-------------|-------------|
| **Baseline** | Original parameters for comparison | No changes |
| **Relaxed_Entry** | Wider RSI range, lower min position | RSI: (15,50), min_pos_score: 0.10 |
| **Lower_ADX** | Lower ADX threshold | ADX threshold: 15 (from 20) |
| **Aggressive_ProgPos** | Lower progressive positioning | Progressive thresholds reduced |
| **Wider_SL_TP** | More room for trades | SL: 2.2x ATR, TP: 5.0x ATR |
| **Combined_Relaxed** | All relaxation combined | Multiple parameters relaxed |
| **Selective_Tight** | Tighter but higher quality | Higher pos requirements, tighter SL/TP |

## Key Results

### Best Performers

| Rank | Experiment | Strategy | OOS Efficiency | OOS Return |
|------|------------|----------|----------------|------------|
| 1 | Relaxed_Entry | Adaptive_Baseline | **0.69** | 0.81% |
| 2 | Relaxed_Entry | Adaptive_ProgPos_Only | **0.69** | 0.81% |
| 3 | Selective_Tight | Baseline | 0.53 | **2.83%** |
| 4 | Wider_SL_TP | Baseline | 0.52 | 2.14% |

### Improvement Summary

| Experiment | Avg OOS Efficiency | vs Baseline | Recommendation |
|------------|-------------------|-------------|----------------|
| Relaxed_Entry | 0.63 | +28.5% | **Strongly Recommended** |
| Selective_Tight | 0.51 | +4.1% | Recommended |
| Wider_SL_TP | 0.51 | +4.1% | Recommended |
| Lower_ADX | 0.50 | +2.0% | Neutral |
| Aggressive_ProgPos | 0.50 | +2.0% | Neutral |
| Combined_Relaxed | 0.02 | -95.9% | Not Recommended |

## Files in This Folder

| File | Description |
|------|-------------|
| `adjusted_config.py` | Parameter configurations for all experiments |
| `run_adjusted_backtest.py` | Main script to run all experiments |
| `generate_report.py` | Script to generate HTML report |
| `experiment_results.json` | Raw results data in JSON format |
| `experiment_report.html` | Visual HTML report with charts and tables |
| `README.md` | This documentation file |

## How to Reproduce

```bash
# Navigate to eval folder
cd eval/strategy_adj_20260204

# Run all experiments (takes ~10 minutes)
python run_adjusted_backtest.py

# Generate HTML report
python generate_report.py

# View report
open experiment_report.html
```

## Walk-Forward Configuration

| Parameter | Value |
|-----------|-------|
| Training Window | 30 days (increased from 14) |
| Testing Window | 14 days (increased from 7) |
| Step Size | 7 days |
| Total Windows | ~60 per strategy |
| Data Range | 2021-12-01 to 2026-02-04 |

## Parameter Details

### Relaxed_Entry (Best OOS Efficiency)
```python
{
    "min_pos_score": 0.10,      # Reduced from 0.15
    "rsi_long_range": (15, 50), # Widened from (20, 45)
    "pullback_range": (0.3, 3.5), # Widened from (0.5, 3.0)
}
```

**Rationale**: Increase trade frequency by relaxing entry criteria while maintaining position-based filtering.

### Selective_Tight (Best OOS Returns)
```python
{
    "min_pos_long": 0.5,        # Increased from 0.4
    "min_pos_score": 0.20,      # Increased from 0.15
    "stop_atr_mult": 1.5,       # Tighter from 1.8
    "tp_atr_mult": 3.5,         # Tighter from 4.5
}
```

**Rationale**: Higher quality entries with tighter risk management for better trade selection.

## Recommendations

### For Production Use:

1. **Adaptive Strategies**: Use **Relaxed_Entry** parameters
   - Improves OOS efficiency to 0.69 (near HEALTHY threshold)
   - Increases trade count significantly
   - Best balance of trade frequency and quality

2. **Non-Adaptive Strategies**: Use **Selective_Tight** parameters
   - Achieves highest absolute returns (2.83%)
   - Maintains reasonable trade count
   - Better risk-adjusted performance

### Next Steps:

1. Run additional evaluations to confirm stability
2. Consider combining Relaxed_Entry + Selective_Tight elements
3. Test with longer evaluation windows (60+ days)
4. Monitor live performance before full deployment

## Interpretation Guide

### OOS Efficiency
- **> 0.70**: HEALTHY - Strategy generalizes well to new data
- **0.50 - 0.70**: WARNING - Some overfitting, use with caution
- **< 0.50**: CRITICAL - Likely overfit, not recommended

### Status Meanings
- **HEALTHY** (green): Safe for production use
- **WARNING** (yellow): Proceed with caution, monitor closely
- **CRITICAL** (red): Requires parameter adjustment before use

## Technical Notes

1. **Trade Count**: Higher trade count doesn't always mean better. Combined_Relaxed had 1130 trades but negative returns.

2. **ADX Filter**: Lowering ADX threshold increases trades but may reduce quality. Original 20 threshold appears reasonable.

3. **Progressive Positioning**: Lowering thresholds alone didn't improve performance. Needs to be combined with other adjustments.

4. **Risk Management**: Wider stops (2.2x ATR) showed slight improvement, suggesting current 1.8x may be too tight in volatile conditions.

---

*Generated by BTC Strategy Evaluation Framework*
*Last Updated: 2026-02-04*
