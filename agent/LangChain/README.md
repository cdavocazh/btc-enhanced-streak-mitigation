# LangChain — LangChain Implementation

BTC Evaluation Agent built with [LangChain](https://python.langchain.com/), using MiniMax Coding Plan API as the LLM backend via `langchain_openai.ChatOpenAI`.

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

The agent uses LangChain's tool-calling agent pattern:

1. **`create_llm()`** — Creates `ChatOpenAI` with MiniMax's base URL and model
2. **`ChatPromptTemplate`** — System prompt + chat history + human input + agent scratchpad
3. **`create_tool_calling_agent(llm, tools, prompt)`** — Builds the agent
4. **`AgentExecutor`** — Runs the agent loop with error handling and iteration limit

## Files

| File | Description |
|------|-------------|
| `agent.py` | Agent definition with 14 `@tool` wrappers around `shared/tools.py`, LLM + executor factory |
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
- **any text** — Feedback sent back to the agent with chat history for revision

## AgentExecutor Settings

| Setting | Value | Reason |
|---------|-------|--------|
| `verbose` | `True` | Shows tool calls and reasoning |
| `handle_parsing_errors` | `True` | Graceful recovery from malformed LLM output |
| `max_iterations` | `20` | Complex evaluations may need many tool calls |
| `return_intermediate_steps` | `True` | Enables inspection of tool call chain |

## Dependencies

- `langchain>=0.3.0` — LangChain core framework
- `langchain-openai>=0.3.0` — OpenAI/MiniMax LLM integration
- `langchain-core>=0.3.0` — Core abstractions (tools, prompts)

## Configuration

All configuration lives in `shared/config.py`:
- `MINIMAX_BASE_URL` = `https://api.minimax.io/v1`
- `MINIMAX_MODEL` = `MiniMax-M2.1`
- `MINIMAX_API_KEY` from `MINIMAX_API_KEY` env var

## Comparison with OAI Implementation

| Aspect | OAI | LangChain |
|--------|-----|-----------|
| SDK | OpenAI Agents SDK | LangChain |
| LLM client | `AsyncOpenAI` | `ChatOpenAI` |
| Tool decorator | `@function_tool` | `@tool` |
| Execution | `Runner.run()` (async) | `AgentExecutor.invoke()` (sync) |
| Multi-turn | Manual | Built-in `chat_history` |
| Verbosity | Minimal output | Detailed tool-call logging |

Both produce identical evaluation outputs — choose based on your team's SDK preference.
