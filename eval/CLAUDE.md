# CLAUDE.md — eval/

Instructions for Claude Code when working in the `/eval` directory and the `/agent` evaluation agent framework.

## What This Is

Walk-forward evaluation framework for BTC trading strategies, plus an LLM-powered agent framework (`/agent`) that automates the evaluation loop.

## Eval Framework

### Core Modules

| File | Class/Function | Purpose |
|------|---------------|---------|
| `config.py` | `WalkForwardConfig`, `STRATEGY_PARAMETER_RANGES` | Central configuration — all strategy params, thresholds, schedules |
| `walk_forward_engine.py` | `WalkForwardEngine` | Rolling-window walk-forward optimization |
| `performance_tracker.py` | `PerformanceTracker` | SQLite-backed performance tracking and alerts |
| `strategy_learner.py` | `StrategyLearner` | Trend analysis (linear regression), parameter adaptation proposals |
| `run_evaluation.py` | `EvaluationRunner` | Main orchestrator — quick/full/adapt modes |
| `eval_regular_exec.sh` | — | 10-step automated pipeline (eval + learn + adjust + backtest + report) |

### Key Commands

```bash
# Automated pipeline (recommended)
./eval/eval_regular_exec.sh              # Full pipeline (~15-20 min)
./eval/eval_regular_exec.sh --quick      # Quick check (~1 min)
./eval/eval_regular_exec.sh --eval-only  # Eval without adjustments
./eval/eval_regular_exec.sh --adj-only   # Adjustments only

# Individual scripts
python eval/run_evaluation.py --full     # Full walk-forward evaluation
python eval/run_evaluation.py --quick    # Quick performance check
python eval/run_evaluation.py --adapt    # Dry-run adaptation check
python eval/run_evaluation.py --adapt --apply  # Apply adaptations
```

### Output Conventions

**Evaluation reports:** `eval/EVALUATION_REPORT_YYYYMMDD_HHMMSS.md`
```markdown
# BTC Strategy Evaluation Report
**Date:** YYYY-MM-DD HH:MM:SS
**Type:** Full Walk-Forward Evaluation
## Strategy Performance Summary
| Strategy | IS Return | OOS Return | OOS Efficiency | Status |
```

**Results JSON:** `eval/results/full_eval_YYYYMMDD_HHMMSS.json`
```json
{
  "timestamp": "ISO-8601",
  "type": "full",
  "config": {"training_days": 14, "testing_days": 7},
  "strategies": {
    "Baseline": {"is_return": 2.48, "oos_return": 1.22, "oos_efficiency": 0.49, "status": "CRITICAL"}
  }
}
```

**Learnings:** `eval/learnings/learning_{Strategy}_{filter}_{timestamp}.json`
```json
{
  "strategy_name": "Baseline",
  "trend": {"trend_direction": "declining", "confidence": 0.99},
  "proposed_adjustments": [{"parameter_name": "min_pos_long", "current_value": 0.4, "proposed_value": 0.46}]
}
```

**Experiment folders:** `eval/strategy_adj_YYYYMMDD/`
- `adjusted_config.py` — Experiment parameters (BASELINE_PARAMS, ADJUSTMENT_EXPERIMENTS, WF_CONFIG)
- `run_adjusted_backtest.py` — Backtest runner (AdjustedBacktestRunner class)
- `generate_report.py` — HTML report generator
- `experiment_results.json` — Raw results
- `experiment_report.html` — Visual report

### Performance Thresholds

| OOS Efficiency | Status | Action |
|----------------|--------|--------|
| >= 0.70 | HEALTHY | No action |
| 0.50 - 0.70 | WARNING | Monitor, consider adjustments |
| < 0.50 | CRITICAL | Immediate parameter adjustment needed |

### Strategy Parameters (from `config.py`)

9 strategies with tunable parameters in `STRATEGY_PARAMETER_RANGES`:
- **Tiered** (5): Baseline, PosVol_Combined, MultiTP_30, VolFilter_Adaptive, Conservative
- **Adaptive** (4): Adaptive_Baseline, Adaptive_Conservative, Adaptive_ADX_Only, Adaptive_ProgPos_Only

Key parameter ranges:
- `min_pos_long`: 0.2 - 0.8 (positioning threshold)
- `rsi_long_range`: (15-35, 40-55) (RSI entry window)
- `pullback_range`: (0.3-1.0, 2.0-5.0) (pullback percentage)
- `adx_trending_threshold`: 15 - 30 (ADX filter for adaptive strategies)
- `cooldown_bars`: 48 - 144 (bars after consecutive losses)
- `progressive_pos_{0-4}`: 0.3 - 1.2 (progressive positioning by loss count)

### Data Sources (read-only)

- `binance-futures-data/data/price.csv` — OHLC 5-min candles
- `binance-futures-data/data/top_trader_position_ratio.csv` — L/S positioning
- `binance-futures-data/data/top_trader_account_ratio.csv` — Account L/S ratio
- `binance-futures-data/data/global_ls_ratio.csv` — Global sentiment
- `binance-futures-data/data/funding_rate.csv` — 8-hour funding
- `binance-futures-data/data/open_interest.csv` — Open interest
- `binance-futures-data/data/last_timestamps.json` — Data freshness

---

## Agent Framework (`/agent`)

### Overview

LLM-powered agent that automates the evaluation loop. Two implementations sharing the same tools:

| Implementation | SDK | LLM Client | Tool Decorator |
|----------------|-----|------------|----------------|
| `/agent/OAI` | OpenAI Agents SDK | `AsyncOpenAI` | `@function_tool` |
| `/agent/LangChain` | LangChain | `ChatOpenAI` | `@tool` |

**LLM:** MiniMax Coding Plan API (`https://api.minimax.io/v1`, model `MiniMax-M2.1`)

### Shared Tools (`/agent/shared/tools.py`)

14 tools — all logic lives here, both OAI and LangChain are thin wrappers:

| Tool | Type | Description |
|------|------|-------------|
| `read_latest_evaluation` | Read | Latest `full_eval_*.json` + `EVALUATION_REPORT_*.md` |
| `read_latest_learnings` | Read | Latest `learning_*.json` per strategy |
| `read_strategy_config` | Read | Parameter ranges from `eval/config.py` |
| `read_market_data_status` | Read | Data freshness + CSV row counts |
| `run_walk_forward_evaluation` | Execute | Runs `eval/run_evaluation.py --full` |
| `run_parameter_experiment` | Execute | Creates `strategy_adj_YYYYMMDD/`, runs backtest |
| `generate_evaluation_report` | Write | Generates `EVALUATION_REPORT_*.md` |
| `web_search` | External | Web search (DuckDuckGo) |
| `read_evaluation_history` | Read | Last N evaluation iterations for trend comparison |
| `run_monte_carlo_validation` | Execute | Brute-force MC shuffle test (+ antithetic variates) |
| `run_stratified_monte_carlo` | Execute | Stratified MC by regime/volatility/session (+ antithetic) |
| `read_monte_carlo_results` | Read | Existing MC results without re-running |
| `run_particle_filter` | Execute | Bootstrap particle filter for regime-adaptive parameters |
| `read_particle_filter_results` | Read | Existing particle filter posteriors |

### Agent Commands

```bash
export MINIMAX_API_KEY="your-key"

# OAI
python agent/OAI/run.py                          # Full cycle
python agent/OAI/run.py --review-only            # Read-only

# LangChain
python agent/LangChain/run.py                    # Full cycle
python agent/LangChain/run.py --review-only      # Read-only
```

### Agent Constraints

- Writes ONLY to `/eval` (strategy_adj folders, results, learnings, reports)
- Market data in `/binance-futures-data/data/` is READ-ONLY
- Human-in-the-loop for report review (y/n/feedback prompt)
- Builds on latest results — does not re-evaluate from scratch

### Adding New Tools

1. Add the function to `agent/shared/tools.py`
2. Add a `@function_tool` wrapper in `agent/OAI/agent.py`
3. Add a `@tool` wrapper in `agent/LangChain/agent.py`
4. Add to `ALL_TOOLS` list in both agent files

### MiniMax Gotchas

- OAI: Must call `set_tracing_disabled(True)` — MiniMax rejects tracing headers
- OAI: Must call `set_default_openai_client(AsyncOpenAI(...))` before creating Agent
- LangChain: Use `ChatOpenAI(base_url=..., model="MiniMax-M2.1")` — not a custom provider
- Complex tool params (dicts): Pass as JSON strings, parse inside the tool function

---

## Do NOT

- Do not modify `config.py` parameter ranges without running evaluation first
- Do not delete `strategy_adj_*` folders — they are the experiment history
- Do not delete `results/` or `learnings/` JSON files — they feed the strategy learner
- Do not modify `performance_history.db` manually — use `PerformanceTracker` API
- Do not hardcode MiniMax API keys — always use `MINIMAX_API_KEY` env var
