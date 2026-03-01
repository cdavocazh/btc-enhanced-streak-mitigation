# BTC Evaluation Agent

LLM-powered agent that autonomously evaluates, optimizes, and reports on BTC trading strategy performance. Two implementations provided:

- **OAI** — OpenAI Agents SDK with MiniMax backend
- **LangChain** — LangChain with MiniMax backend

Both use the same shared tools and system prompt, producing identical evaluation outputs.

## Setup

### 1. Install Dependencies

```bash
# For OpenAI Agents SDK implementation
pip install -r agent/OAI/requirements.txt

# For LangChain implementation
pip install -r agent/LangChain/requirements.txt
```

### 2. Set API Key

```bash
export MINIMAX_API_KEY="your-minimax-api-key"
```

Get your API key from [MiniMax Platform](https://platform.minimaxi.com/).

## Usage

### OpenAI Agents SDK

```bash
# Full evaluation cycle (assess, diagnose, experiment, report)
python agent/OAI/run.py

# Review-only mode (no changes, just analysis)
python agent/OAI/run.py --review-only

# Run a specific experiment
python agent/OAI/run.py --experiment "Relaxed_Entry"
```

### LangChain

```bash
# Full evaluation cycle
python agent/LangChain/run.py

# Review-only mode
python agent/LangChain/run.py --review-only

# Run a specific experiment
python agent/LangChain/run.py --experiment "Relaxed_Entry"
```

## Architecture

```
agent/
├── shared/              # Shared between both implementations
│   ├── config.py        # Paths, API config, constants
│   ├── prompts.py       # System prompt and task prompts
│   └── tools.py         # 14 tool functions (core logic)
├── OAI/                 # OpenAI Agents SDK
│   ├── agent.py         # Agent + tool definitions
│   └── run.py           # CLI entrypoint
└── LangChain/           # LangChain
    ├── agent.py         # Agent + tool definitions
    └── run.py           # CLI entrypoint
```

### Tools Available to the Agent (14 tools)

| # | Tool | Description |
|---|------|-------------|
| 1 | `read_latest_evaluation` | Read most recent full_eval JSON and EVALUATION_REPORT markdown |
| 2 | `read_latest_learnings` | Read strategy learning reports (trends, proposed adjustments) |
| 3 | `read_evaluation_history` | Read last N evaluation iterations for trend comparison |
| 4 | `read_strategy_config` | Read parameter ranges, strategy configs, thresholds from eval/config.py |
| 5 | `read_market_data_status` | Check data freshness and CSV row counts |
| 6 | `run_walk_forward_evaluation` | Run walk-forward evaluation for a strategy |
| 7 | `run_parameter_experiment` | Create strategy_adj folder, run backtest with adjusted params |
| 8 | `generate_evaluation_report` | Generate markdown report in standard format |
| 9 | `web_search` | Search web for strategy research (DuckDuckGo) |
| 10 | `run_monte_carlo_validation` | Brute-force MC shuffle test (+ antithetic variates) |
| 11 | `run_stratified_monte_carlo` | Stratified MC by regime/volatility/session strata (+ antithetic) |
| 12 | `read_monte_carlo_results` | Read existing MC results without re-running |
| 13 | `run_particle_filter` | Bootstrap particle filter for regime-adaptive parameters |
| 14 | `read_particle_filter_results` | Read existing particle filter posteriors |

### Agent Workflow

1. **Assess** — Read latest evaluation results and learnings
2. **Diagnose** — Identify CRITICAL/WARNING strategies (OOS efficiency thresholds)
3. **Research** — Optional web search for market analysis
4. **Plan** — Propose parameter adjustments based on diagnostics
5. **Execute** — Run parameter experiments
6. **Evaluate** — Walk-forward evaluation on adjusted parameters
7. **Validate** — Run Monte Carlo shuffle tests (brute-force + stratified, with antithetic variates)
8. **Report** — Generate evaluation report in standard format
9. **Present** — Human reviews the report (HITL checkpoint)
10. **Estimate** — (Optional) Run particle filter for regime parameter estimation and position sizing

### Human-in-the-Loop

The agent presents reports for human review at the end of each cycle. Options:
- **y** — Accept the evaluation
- **n** — Reject (no changes saved)
- **feedback text** — Agent revises based on your feedback

## Output

The agent writes to `/eval` following existing conventions:
- `eval/EVALUATION_REPORT_YYYYMMDD_HHMMSS.md` — Evaluation reports
- `eval/strategy_adj_YYYYMMDD/` — Parameter experiment folders
- `eval/results/` — JSON result files
- `eval/learnings/` — Strategy learning JSON files

## Configuration

### MiniMax API

| Setting | Value |
|---------|-------|
| Base URL | `https://api.minimax.io/v1` |
| Model | `MiniMax-M2.1` |
| API Key | `MINIMAX_API_KEY` env var |

### Performance Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| OOS Efficiency >= 0.70 | HEALTHY | No action needed |
| OOS Efficiency 0.50-0.70 | WARNING | Monitor, consider adjustments |
| OOS Efficiency < 0.50 | CRITICAL | Immediate parameter adjustment |

### Web Search

Uses DuckDuckGo for web search. No API key required.

### Antithetic Variates

Both Monte Carlo tools support `antithetic=True` for ~50% variance reduction:
- Brute-force: pairs each shuffle Z with reverse(Z), averages metrics
- Stratified: reverses PnL within each stratum independently

### Particle Filter

Online Bayesian parameter estimation tracking:
- `half_life` (2-50 bars): Mean-reversion speed
- `vol_scale` (0.5-3.0): Current volatility relative to historical median
- `signal_strength` (0.0-2.0): Positioning-signal alpha

Position sizing: reduces size when posterior is wide (high uncertainty).
