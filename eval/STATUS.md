# Evaluation Framework - Status

## Current Status: INITIALIZED

**Last Updated:** 2026-02-03

## Framework Components

| Component | Status | Description |
|-----------|--------|-------------|
| `config.py` | ✅ Ready | Strategy parameters, thresholds, schedules |
| `walk_forward_engine.py` | ✅ Ready | WFO engine with rolling windows |
| `performance_tracker.py` | ✅ Ready | SQLite tracking & alerts |
| `strategy_learner.py` | ✅ Ready | Trend analysis & adaptation |
| `run_evaluation.py` | ✅ Ready | Main runner with CLI |
| Database | 🟡 Pending | Will be created on first run |

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

## Evaluation Schedule

| Type | Interval | Last Run | Next Due |
|------|----------|----------|----------|
| Quick Check | 6 hours | Never | Now |
| Full Evaluation | 7 days | Never | Now |
| Adaptation Check | 14 days | Never | Now |

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

**Next Steps:**
1. Run first evaluation: `python eval/run_evaluation.py --full`
2. Review results and tune thresholds if needed
3. Set up scheduled runs via launchd/cron
4. Monitor alerts and iterate

---

### Iteration 1 - [PENDING]

**Planned:**
- Run first full evaluation
- Record baseline performance for all strategies
- Identify any data loading issues
- Tune alert thresholds based on actual values

---

## Alert Summary

*No alerts yet - run first evaluation to populate*

---

## Performance Baseline

*Will be populated after first full evaluation*

| Strategy | Expected Return | Expected Sharpe | Max DD Threshold |
|----------|-----------------|-----------------|------------------|
| Baseline | TBD | TBD | TBD |
| Adaptive_Baseline | TBD | TBD | TBD |
| Adaptive_ProgPos_Only | TBD | TBD | TBD |
| Conservative | TBD | TBD | TBD |

---

## Known Issues

1. **Data Loading**: Framework expects data from `backtest_15min/load_15min_data.py`
   - Fallback to CSV loading from `binance-futures-data/data/` available
   - May need path adjustments depending on data location

2. **Positioning Data**: Not all positioning metrics may be available
   - Engine uses fallback values (score=0) when data missing
   - Full evaluation requires positioning data for accurate results

---

## Commands Reference

```bash
# First run - full evaluation
python eval/run_evaluation.py --full

# Quick performance check
python eval/run_evaluation.py --quick

# Check for adaptations (dry run)
python eval/run_evaluation.py --adapt

# Apply adaptations
python eval/run_evaluation.py --adapt --apply

# Show current summary
python eval/run_evaluation.py --summary

# Scheduled run (checks what's due)
python eval/run_evaluation.py --scheduled
```

---

## Metrics Definitions

| Metric | Formula | Good Value |
|--------|---------|------------|
| OOS Efficiency | OOS_Return / IS_Return | > 0.7 |
| Parameter Stability | 1 - StdDev(efficiency) | > 0.7 |
| Trend Confidence | R² × (1 - p_value) | > 0.5 |
| Regime Detection | \|Sharpe_change\| or \|Return_change\| | > 0.25 triggers |

---

## File Outputs

After running evaluations, check these locations:

```
eval/
├── results/
│   ├── quick_eval_YYYYMMDD_HHMMSS.json
│   ├── full_eval_YYYYMMDD_HHMMSS.json
│   └── wf_report_Strategy_filter_YYYYMMDD_HHMMSS.json
├── learnings/
│   └── learning_Strategy_filter_YYYYMMDD_HHMMSS.json
├── performance_history.db
└── EVALUATION_REPORT_YYYYMMDD_HHMMSS.md
```
