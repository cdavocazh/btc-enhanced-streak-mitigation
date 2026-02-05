# BTC Strategy Evaluation Framework

A Walk-Forward Optimization (WFO) system for continuously evaluating and adapting BTC trading strategies. Implements the "gold standard" of trading strategy validation (Robert Pardo, 2008).

## Overview

This framework provides automated:
- **Walk-Forward Validation**: Rolling-window backtesting with in-sample optimization and out-of-sample validation
- **Performance Tracking**: SQLite-based historical tracking with degradation alerts
- **Adaptive Learning**: Parameter adjustment based on performance trends
- **Scheduled Evaluations**: Automated periodic checks (quick/full/adapt)

## Quick Start

### Automated Full Pipeline (Recommended)

```bash
# Run complete evaluation + adjustment pipeline
./eval/eval_regular_exec.sh

# Quick check only
./eval/eval_regular_exec.sh --quick

# Evaluation without adjustments
./eval/eval_regular_exec.sh --eval-only

# Adjustments only (uses existing evaluation)
./eval/eval_regular_exec.sh --adj-only
```

### Individual Commands

```bash
# Quick performance check (run every 6 hours)
python eval/run_evaluation.py --quick

# Full walk-forward evaluation (run weekly)
python eval/run_evaluation.py --full

# Learning/adaptation check (dry run)
python eval/run_evaluation.py --adapt

# Apply parameter adjustments
python eval/run_evaluation.py --adapt --apply

# Run based on schedule
python eval/run_evaluation.py --scheduled

# Show performance summary
python eval/run_evaluation.py --summary
```

## Architecture

```
eval/
├── config.py                # Central configuration & strategy parameters
├── walk_forward_engine.py   # Rolling-window WFO implementation
├── performance_tracker.py   # SQLite tracking & alerts
├── strategy_learner.py      # Trend analysis & parameter adaptation
├── run_evaluation.py        # Main scheduler & runner
├── eval_regular_exec.sh     # ★ Automated pipeline script
├── __init__.py              # Package exports
├── README.md                # This file
├── STATUS.md                # Current status & iteration log
├── results/                 # Evaluation results (JSON)
├── learnings/               # Learning cycle reports (JSON)
├── logs/                    # Execution logs
├── performance_history.db   # SQLite database
└── strategy_adj_YYYYMMDD/   # ★ Auto-generated adjustment folders
    ├── adjusted_config.py   # Parameter configurations
    ├── run_adjusted_backtest.py
    ├── generate_report.py
    ├── experiment_results.json
    ├── experiment_report.html
    └── README.md
```

## Strategies Covered

### From backtest_15min_new (Tiered Capital + Streak Mitigation)
| Strategy | Description |
|----------|-------------|
| Baseline | Core strategy with tiered risk |
| PosVol_Combined | Position + volume exit rules |
| MultiTP_30 | Multi-take-profit (30% at TP1) |
| VolFilter_Adaptive | Adaptive volume filtering |
| Conservative | Reduced risk, higher win rate |

### From backtest_15min_new_streak_a (Adaptive Streak Reduction)
| Strategy | Description |
|----------|-------------|
| Adaptive_Baseline | ADX filter + progressive positioning + cooldown |
| Adaptive_Conservative | Above + multi-TP + breakeven rules |
| Adaptive_ADX_Only | Only ADX market regime filter |
| Adaptive_ProgPos_Only | Only progressive positioning threshold |

## Evaluation Modes

### 1. Quick Check (`--quick`)
- **Interval**: Every 6 hours
- **Purpose**: Lightweight performance snapshot
- **Actions**:
  - Calculate rolling 7-day metrics
  - Check for regime changes
  - Generate alerts if thresholds exceeded

### 2. Full Evaluation (`--full`)
- **Interval**: Weekly
- **Purpose**: Complete walk-forward analysis
- **Actions**:
  - Create rolling training/testing windows
  - Run backtests on each window
  - Calculate OOS efficiency & parameter stability
  - Record performance snapshots
  - Generate detailed reports

### 3. Adaptation Check (`--adapt`)
- **Interval**: Every 14 days
- **Purpose**: Analyze trends and propose adjustments
- **Actions**:
  - Analyze 30-day performance trend
  - Detect market regime changes
  - Propose parameter adjustments if declining
  - Apply changes (with `--apply` flag)

## Configuration

### Walk-Forward Windows
```python
training_window: 14 days  # In-sample training
testing_window: 7 days    # Out-of-sample testing
step_size: 7 days         # Roll forward interval
```

### Alert Thresholds
```python
return_degradation_warning: 20%   # Below baseline
return_degradation_critical: 40%  # Below baseline
max_drawdown_warning: 25%
max_drawdown_critical: 40%
win_rate_min: 35%
sharpe_min: 0.5
max_losing_streak: 7
out_of_sample_efficiency: 0.6     # OOS/IS ratio
```

### Adaptation Rules
```python
parameter_adjustment_rate: 10%    # Max adjustment per cycle
max_parameter_deviation: 30%      # Max from baseline
min_evaluations_before_adapt: 4   # Required history
min_confidence: 50%               # Required for adjustment
```

## Database Schema

### performance_snapshots
```sql
- timestamp, strategy_name, entry_filter
- total_return_pct, max_drawdown_pct, sharpe_ratio
- win_rate_pct, profit_factor, total_trades
- oos_return_pct, oos_efficiency, parameter_stability
- data_bars, evaluation_type
```

### alerts
```sql
- timestamp, strategy_name, severity (INFO/WARNING/CRITICAL)
- alert_type, message, current_value, threshold_value
- acknowledged
```

### parameter_history
```sql
- timestamp, strategy_name, parameter_name
- parameter_value, change_reason
```

### regime_history
```sql
- timestamp, regime, confidence
- volatility_20d, trend_strength
```

## Output Files

| File | Description |
|------|-------------|
| `results/quick_eval_*.json` | Quick check results |
| `results/full_eval_*.json` | Full evaluation summaries |
| `results/wf_report_*.json` | Walk-forward window details |
| `learnings/learning_*.json` | Learning cycle reports |
| `EVALUATION_REPORT_*.md` | Human-readable reports |
| `performance_history.db` | SQLite database |

## Automated Pipeline: eval_regular_exec.sh

The `eval_regular_exec.sh` script automates the entire evaluation and adjustment workflow:

### What It Does

1. **Runs full walk-forward evaluation** for all strategies
2. **Runs strategy learner** to analyze performance trends
3. **Creates adjustment folder** (`strategy_adj_YYYYMMDD/`)
4. **Generates parameter experiments** (6 different adjustments)
5. **Runs backtests** with adjusted parameters
6. **Generates HTML report** with visual results
7. **Updates STATUS.md** with findings

### Usage Modes

| Mode | Command | Description |
|------|---------|-------------|
| Full | `./eval_regular_exec.sh` | Complete pipeline |
| Quick | `./eval_regular_exec.sh --quick` | Quick performance check only |
| Eval Only | `./eval_regular_exec.sh --eval-only` | Evaluation without adjustments |
| Adj Only | `./eval_regular_exec.sh --adj-only` | Adjustments only |

### Output

- **Adjustment folder**: `eval/strategy_adj_YYYYMMDD/`
- **HTML Report**: `eval/strategy_adj_YYYYMMDD/experiment_report.html`
- **Log file**: `eval/logs/eval_exec_YYYYMMDD.log`

## Integration with Launchd

### Data Refresh (Every 5 minutes)
The Binance data refresh is handled by a separate launchd job:
```bash
# Check status
launchctl list | grep btc

# View logs
tail -f ~/Github/btc-enhanced-streak-mitigation/binance-futures-data/logs/launchd.log
```

### Scheduled Evaluation (Crontab)
```bash
# Weekly full evaluation (Sundays at 2am)
0 2 * * 0 cd /path/to/repo && ./eval/eval_regular_exec.sh >> eval/logs/weekly.log 2>&1

# Quick check every 6 hours
0 */6 * * * cd /path/to/repo && python eval/run_evaluation.py --quick >> eval/logs/quick.log 2>&1
```

## Key Metrics

### OOS Efficiency
```
OOS Efficiency = OOS Return / IS Return
```
- **> 0.7**: HEALTHY - Strategy generalizes well
- **0.5 - 0.7**: WARNING - Possible overfitting
- **< 0.5**: CRITICAL - Likely overfit

### Parameter Stability
```
Stability = 1 - StdDev(efficiency across windows)
```
- **> 0.7**: Parameters are stable across time
- **< 0.5**: Parameters may need frequent adjustment

## Best Practices

1. **Run quick checks frequently** (every 6 hours) to catch issues early
2. **Review alerts promptly** - especially CRITICAL severity
3. **Don't apply adaptations blindly** - review proposed changes first
4. **Monitor OOS efficiency** - declining efficiency suggests overfitting
5. **Keep historical data** - longer history improves trend detection

## References

- Robert Pardo, "The Evaluation and Optimization of Trading Strategies" (2008)
- Walk-Forward Analysis: QuantInsti methodology
- Adaptive Strategy Management: Academic literature on regime detection
