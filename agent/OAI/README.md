# OAI — OpenAI Agents SDK Implementation

BTC Evaluation Agent built with the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python), using MiniMax Coding Plan API as the LLM backend.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API key
export MINIMAX_API_KEY="your-minimax-api-key"

# 3. Run
python run.py                          # Full evaluation cycle
python run.py --review-only            # Read-only analysis
python run.py --experiment "Relaxed"   # Specific experiment
```

## How It Works

The agent uses the OpenAI Agents SDK's `Agent` + `Runner` pattern:

1. **`create_client()`** — Creates an `AsyncOpenAI` client pointed at MiniMax's endpoint (`https://api.minimax.io/v1`)
2. **`set_default_openai_client(client)`** — Tells the SDK to use MiniMax instead of OpenAI
3. **`set_tracing_disabled(True)`** — Required because MiniMax doesn't support OpenAI tracing
4. **`Agent(...)`** — Defines the agent with system prompt and tools
5. **`Runner.run(agent, prompt)`** — Executes the agent loop (async)

## Files

| File | Description |
|------|-------------|
| `agent.py` | Agent definition with 14 `@function_tool` wrappers around `shared/tools.py` |
| `run.py` | CLI entrypoint — parses args, selects prompt, runs agent, handles HITL review |
| `requirements.txt` | Python dependencies |

## Tools

All 14 tools are thin wrappers around `shared/tools.py`:

| Tool | Description |
|------|-------------|
| `read_latest_evaluation` | Read most recent eval results and report |
| `read_latest_learnings` | Read strategy learning reports and trends |
| `read_strategy_config` | Read parameter ranges and thresholds |
| `read_market_data_status` | Check data freshness and CSV row counts |
| `run_walk_forward_evaluation` | Run WFO for a specific strategy |
| `run_parameter_experiment` | Create experiment folder and run backtest |
| `generate_evaluation_report` | Generate markdown report |
| `web_search` | Web search (requires search API config) |
| `read_evaluation_history` | Read last N evaluation iterations for trends |
| `run_monte_carlo_validation` | Brute-force MC shuffle test (+ antithetic variates) |
| `run_stratified_monte_carlo` | Stratified MC by regime/volatility/session (+ antithetic) |
| `read_monte_carlo_results` | Read existing MC results without re-running |
| `run_particle_filter` | Bootstrap particle filter for regime parameters |
| `read_particle_filter_results` | Read existing particle filter posteriors |

## Human-in-the-Loop

After the agent completes its analysis, `run.py` prompts:

```
Accept this evaluation? [y/n/feedback]:
```

- **y** — Accept and exit
- **n** — Reject, no changes saved
- **any text** — Feedback sent back to the agent for revision

## Dependencies

- `openai-agents>=0.0.7` — OpenAI Agents SDK
- `openai>=1.66.0` — OpenAI Python client (used for AsyncOpenAI)

## Configuration

All configuration lives in `shared/config.py`:
- `MINIMAX_BASE_URL` = `https://api.minimax.io/v1`
- `MINIMAX_MODEL` = `MiniMax-M2.1`
- `MINIMAX_API_KEY` from `MINIMAX_API_KEY` env var
