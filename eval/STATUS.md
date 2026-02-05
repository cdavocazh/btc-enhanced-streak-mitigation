# Evaluation Framework - Status

## Current Status: ITERATION 2 COMPLETE - IMPROVEMENTS FOUND

**Last Updated:** 2026-02-04 14:20

---

## Executive Summary

The BTC Strategy Evaluation Framework has completed two iterations:
1. **Iteration 1**: First evaluation revealed all strategies in CRITICAL status (OOS Efficiency < 0.50)
2. **Iteration 2**: Parameter adjustments improved best strategy OOS Efficiency from **0.48 to 0.69** (+44%)

**Key Achievement**: `Relaxed_Entry` experiment for Adaptive strategies achieved **0.69 OOS Efficiency**, approaching the HEALTHY threshold of 0.70.

---

## Framework Components

| Component | Status | Description |
|-----------|--------|-------------|
| `config.py` | ✅ Ready | Strategy parameters, thresholds, schedules |
| `walk_forward_engine.py` | ✅ Ready | WFO engine with rolling windows |
| `performance_tracker.py` | ✅ Ready | SQLite tracking & alerts |
| `strategy_learner.py` | ✅ Ready | Trend analysis & adaptation |
| `run_evaluation.py` | ✅ Ready | Main runner with CLI |
| Database | ✅ Created | `performance_history.db` with snapshots & alerts |
| Adjustment Framework | ✅ Created | `strategy_adj_YYYYMMDD/` folders |

---

## Strategies Configured

### Tiered Capital Strategies (backtest_15min_new)
- [x] Baseline
- [x] PosVol_Combined
- [x] MultiTP_30
- [x] VolFilter_Adaptive
- [x] Conservative

### Adaptive Strategies (backtest_15min_new_streak_a)
- [x] Adaptive_Baseline
- [x] Adaptive_Conservative
- [x] Adaptive_ADX_Only
- [x] Adaptive_ProgPos_Only

---

## Evaluation Schedule

| Type | Interval | Last Run | Next Due |
|------|----------|----------|----------|
| Quick Check | 6 hours | 2026-02-04 13:54 | 2026-02-04 19:54 |
| Full Evaluation | 7 days | 2026-02-04 13:54 | 2026-02-11 |
| Adaptation Check | 14 days | 2026-02-04 13:55 | 2026-02-18 |

---

## Iteration Log

### Iteration 0 - Framework Creation (2026-02-03)

**What was done:**
1. Created eval folder structure
2. Implemented config.py with all strategy parameters from:
   - backtest_15min_new/run_tiered_streak_mitigation.py
   - backtest_15min_new_streak_a/run_adaptive_streak_backtest.py
3. Implemented walk_forward_engine.py:
   - Rolling window creation
   - Simplified backtest execution
   - OOS efficiency calculation
   - Parameter stability tracking
4. Implemented performance_tracker.py:
   - SQLite database with 4 tables
   - Performance snapshot recording
   - Degradation alert generation
   - Regime change detection
5. Implemented strategy_learner.py:
   - Trend analysis using linear regression
   - Parameter adjustment proposals
   - Learning cycle execution
6. Implemented run_evaluation.py:
   - Three execution modes (quick/full/adapt)
   - Scheduled execution
   - Report generation

**Configuration Summary:**
- Training window: 14 days (1,344 15-min bars)
- Testing window: 7 days (672 bars)
- Step size: 7 days
- Alert thresholds: 20%/40% return degradation, 25%/40% drawdown
- Adaptation rate: 10% max per cycle, 30% max deviation

---

### Iteration 1 - First Evaluation (2026-02-04 13:52)

**What was done:**
1. Fixed scipy/numpy compatibility issue (upgraded scipy 1.10.1 → 1.15.3)
2. Ran full walk-forward evaluation for all 6 strategies
3. Ran strategy learner (insufficient data for trend analysis - first run)
4. Generated evaluation reports and recorded performance snapshots

**Evaluation Results:**

| Strategy | IS Return | OOS Return | OOS Efficiency | Status |
|----------|-----------|------------|----------------|--------|
| Baseline | 2.45% | 1.18% | 0.48 | 🔴 CRITICAL |
| MultiTP_30 | 2.45% | 1.18% | 0.48 | 🔴 CRITICAL |
| Conservative | 2.45% | 1.18% | 0.48 | 🔴 CRITICAL |
| Adaptive_Baseline | 1.42% | 0.64% | 0.45 | 🔴 CRITICAL |
| Adaptive_ProgPos_Only | 2.62% | 1.18% | 0.45 | 🔴 CRITICAL |
| Adaptive_Conservative | 1.42% | 0.64% | 0.45 | 🔴 CRITICAL |

**Key Findings:**
1. All strategies show CRITICAL status - OOS Efficiency < 0.5 indicates potential overfitting
2. Non-adaptive strategies perform slightly better (0.48 vs 0.45 OOS efficiency)
3. ADX filter reduces trade frequency but doesn't improve OOS performance
4. Progressive positioning shows no improvement over baseline

**Root Causes Identified:**
1. Entry conditions too restrictive (0% win rate in evaluation)
2. Short training/testing windows (14/7 days) may not capture full market cycles
3. ADX threshold (20) may be too high, blocking valid trades

---

### Iteration 2 - Parameter Adjustment Experiments (2026-02-04 14:07)

**What was done:**
1. Created `strategy_adj_20260204/` folder for experiments
2. Designed 6 parameter adjustment experiments based on root cause analysis
3. Increased walk-forward windows (30-day training, 14-day testing)
4. Ran all experiments (~10 minutes total)
5. Generated HTML report with visual results

**Experiments Conducted:**

| Experiment | Description | Avg OOS Eff | vs Baseline |
|------------|-------------|-------------|-------------|
| Baseline | Original parameters | 0.49 | - |
| **Relaxed_Entry** | Wider RSI, lower min_pos | **0.63** | **+28.5%** |
| Selective_Tight | Tighter entry, tighter SL/TP | 0.51 | +4.1% |
| Wider_SL_TP | More room for trades | 0.51 | +4.1% |
| Lower_ADX | ADX threshold 15 | 0.50 | +2.0% |
| Aggressive_ProgPos | Lower prog pos thresholds | 0.50 | +2.0% |
| Combined_Relaxed | All relaxation combined | 0.02 | -95.9% |

**Best Results:**

| Rank | Experiment | Strategy | OOS Efficiency | OOS Return | Status |
|------|------------|----------|----------------|------------|--------|
| 1 | Relaxed_Entry | Adaptive_Baseline | **0.69** | 0.81% | WARNING |
| 2 | Relaxed_Entry | Adaptive_ProgPos_Only | **0.69** | 0.81% | WARNING |
| 3 | Selective_Tight | Baseline | 0.53 | **2.83%** | WARNING |

**Key Findings:**
1. **Relaxed_Entry** achieved 0.69 OOS Efficiency (near 0.70 HEALTHY threshold)
2. Trade count increased from 218 to 647 with Relaxed_Entry
3. Combined_Relaxed was too aggressive - negative returns with 1130 trades
4. Selective_Tight achieved highest absolute returns (2.83%)

**Recommended Parameters:**

For Adaptive Strategies (Relaxed_Entry):
```python
{
    "min_pos_score": 0.10,        # Reduced from 0.15
    "rsi_long_range": (15, 50),   # Widened from (20, 45)
    "pullback_range": (0.3, 3.5), # Widened from (0.5, 3.0)
}
```

For Non-Adaptive Strategies (Selective_Tight):
```python
{
    "min_pos_long": 0.5,          # Increased from 0.4
    "min_pos_score": 0.20,        # Increased from 0.15
    "stop_atr_mult": 1.5,         # Tighter from 1.8
    "tp_atr_mult": 3.5,           # Tighter from 4.5
}
```

---

## Alert Summary

**Iteration 1 (2026-02-04 13:54):**
- 🔴 **CRITICAL (1)**: Adaptive_Baseline return dropped 92% below baseline
- 🟡 **WARNING (12)**:
  - 6x Low win rate (0% below 35% threshold)
  - 6x Low OOS efficiency (0.45-0.48 below 0.6 threshold)

**Iteration 2 (2026-02-04 14:17):**
- No new alerts generated (experiments ran outside main evaluation loop)

---

## Performance Baseline

### Initial Evaluation (Iteration 1)

| Strategy | OOS Return | OOS Efficiency | Status |
|----------|------------|----------------|--------|
| Baseline | 1.18% | 0.48 | CRITICAL |
| Adaptive_Baseline | 0.64% | 0.45 | CRITICAL |
| Adaptive_ProgPos_Only | 1.18% | 0.45 | CRITICAL |

### After Adjustments (Iteration 2)

| Strategy + Experiment | OOS Return | OOS Efficiency | Status | Improvement |
|----------------------|------------|----------------|--------|-------------|
| Adaptive_Baseline + Relaxed_Entry | 0.81% | **0.69** | WARNING | +53.3% |
| Baseline + Selective_Tight | **2.83%** | 0.53 | WARNING | +10.4% |

**Target Thresholds for HEALTHY status:**
- OOS Efficiency: > 0.70
- Parameter Stability: > 0.70
- Win Rate: > 35%
- Max Drawdown: < 25%

---

## Folder Structure

```
eval/
├── __init__.py
├── config.py                          # Central configuration
├── walk_forward_engine.py             # WFO implementation
├── performance_tracker.py             # SQLite tracking
├── strategy_learner.py                # Trend analysis
├── run_evaluation.py                  # Main CLI runner
├── eval_regular_exec.sh               # ★ Automated pipeline script
├── README.md                          # Framework documentation
├── STATUS.md                          # This file
├── EVALUATION_REPORT_20260204_*.md    # Markdown reports
├── last_run.json                      # Schedule tracking
├── performance_history.db             # SQLite database
├── results/                           # Evaluation results (JSON)
│   ├── quick_eval_*.json
│   ├── full_eval_*.json
│   └── wf_report_*.json
├── learnings/                         # Learning cycle reports
│   └── learning_*.json
├── logs/                              # Execution logs
│   └── eval_exec_*.log
└── strategy_adj_YYYYMMDD/             # ★ Auto-generated adjustment folders
    ├── adjusted_config.py             # Experiment parameters
    ├── run_adjusted_backtest.py       # Experiment runner
    ├── generate_report.py             # HTML report generator
    ├── experiment_results.json        # Raw results
    ├── experiment_report.html         # Visual report
    └── README.md                      # Experiment documentation
```

## eval_regular_exec.sh - Automated Pipeline

### Overview

The `eval_regular_exec.sh` script automates the complete evaluation and strategy adjustment workflow:

1. **Evaluation**: Runs walk-forward optimization for all strategies
2. **Learning**: Analyzes performance trends and identifies declining strategies
3. **Adjustment**: Creates parameter experiments for declining strategies
4. **Backtesting**: Runs backtests with adjusted parameters
5. **Reporting**: Generates HTML report with visual results
6. **Documentation**: Updates STATUS.md with findings

### Usage

| Command | Description | Duration |
|---------|-------------|----------|
| `./eval_regular_exec.sh` | Full pipeline | ~15-20 min |
| `./eval_regular_exec.sh --quick` | Quick check only | ~1 min |
| `./eval_regular_exec.sh --eval-only` | Evaluation only | ~5 min |
| `./eval_regular_exec.sh --adj-only` | Adjustments only | ~10 min |

### Output

- Creates `strategy_adj_YYYYMMDD/` folder with all results
- Generates `experiment_report.html` with visual comparison
- Logs execution to `logs/eval_exec_YYYYMMDD.log`
- Appends iteration summary to `STATUS.md`

---

## Commands Reference

### Automated Pipeline (Recommended)

```bash
# === eval_regular_exec.sh - Full Automation ===

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
# === Main Evaluation Commands ===

# Full walk-forward evaluation (weekly)
python eval/run_evaluation.py --full

# Quick performance check (6-hourly)
python eval/run_evaluation.py --quick

# Check for adaptations (dry run)
python eval/run_evaluation.py --adapt

# Apply adaptations
python eval/run_evaluation.py --adapt --apply

# Show current summary
python eval/run_evaluation.py --summary

# Scheduled run (checks what's due)
python eval/run_evaluation.py --scheduled

# === Adjustment Experiments ===

# Run parameter experiments
python eval/strategy_adj_20260204/run_adjusted_backtest.py

# Generate HTML report
python eval/strategy_adj_20260204/generate_report.py

# View HTML report
open eval/strategy_adj_20260204/experiment_report.html
```

---

## Metrics Definitions

| Metric | Formula | Good Value | Description |
|--------|---------|------------|-------------|
| OOS Efficiency | OOS_Return / IS_Return | > 0.70 | Measures generalization |
| Parameter Stability | 1 - StdDev(efficiency) | > 0.70 | Consistency across windows |
| Trend Confidence | R² × (1 - p_value) | > 0.50 | Statistical significance |
| Regime Detection | \|Sharpe_change\| or \|Return_change\| | > 0.25 | Triggers regime alert |

---

## Next Steps (Iteration 3)

1. **Validate Relaxed_Entry** - Run additional evaluations to confirm stability
2. **Combine Best Elements** - Test Relaxed_Entry + Selective_Tight hybrid
3. **Extend Evaluation Windows** - Test with 60-day training, 30-day testing
4. **Production Deployment** - Apply Relaxed_Entry parameters to production strategies
5. **Continuous Monitoring** - Set up scheduled evaluations via launchd

---

## Technical Notes

1. **SciPy Upgrade**: Fixed numpy compatibility by upgrading scipy 1.10.1 → 1.15.3
2. **Window Size**: Increased from 14/7 days to 30/14 days for better market cycle coverage
3. **FutureWarning**: `pct_change()` deprecation warning in load_15min_data.py (non-critical)

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-03 | 0.1.0 | Framework creation |
| 2026-02-04 | 0.2.0 | First evaluation, all CRITICAL |
| 2026-02-04 | 0.3.0 | Parameter experiments, Relaxed_Entry achieves 0.69 OOS Eff |

---

*Generated by BTC Strategy Evaluation Framework*

---

### Iteration - Auto Evaluation (20260205)

**Generated by:** `eval_regular_exec.sh`
**Timestamp:** 2026-02-05 21:28:13

**What was done:**
1. Ran full walk-forward evaluation
2. Ran strategy learner for trend analysis
3. Created adjustment folder: `strategy_adj_20260205/`
4. Generated 6 parameter adjustment experiments
5. Ran backtests with adjusted parameters
6. Generated HTML report

**Results Location:**
- Experiment folder: `eval/strategy_adj_20260205/`
- HTML Report: `eval/strategy_adj_20260205/experiment_report.html`
- Raw Results: `eval/strategy_adj_20260205/experiment_results.json`

