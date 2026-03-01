# CLAUDE.md — agent/OAI

Instructions for Claude Code when working in the `/agent/OAI` directory.

## What This Is

OpenAI Agents SDK implementation of the BTC Evaluation Agent, using MiniMax as the LLM backend via OpenAI-compatible API.

## Key Files

| File | Purpose |
|------|---------|
| `agent.py` | Agent definition, tool wrappers (`@function_tool`), MiniMax client setup |
| `run.py` | CLI entrypoint with `--review-only`, `--experiment`, full-cycle modes |
| `requirements.txt` | `openai-agents>=0.0.7`, `openai>=1.66.0` |

## Architecture

```
run.py  →  agent.py (create_agent)  →  shared/tools.py (actual logic)
                ↓
        OpenAI Agents SDK
                ↓
        MiniMax API (https://api.minimax.io/v1)
```

- `agent.py` wraps each shared tool with `@function_tool` decorator
- `create_client()` returns an `AsyncOpenAI` pointed at MiniMax
- `set_tracing_disabled(True)` is required — MiniMax does not support OpenAI tracing
- `set_default_openai_client(client)` must be called before creating the Agent
- The agent uses `Runner.run()` (async) for execution

## MiniMax-Specific Gotchas

- Must call `set_default_openai_client()` with a custom `AsyncOpenAI(base_url=...)` — the SDK defaults to OpenAI's endpoint otherwise
- Must call `set_tracing_disabled(True)` — MiniMax returns errors for tracing headers
- Model name is `MiniMax-M2.1` (not an OpenAI model ID)
- The API is OpenAI Chat Completions compatible but does NOT support assistants, threads, or streaming tool calls in the same way

## Tool Parameter Convention

OpenAI Agents SDK `@function_tool` infers parameter schemas from type hints. For complex parameters (dicts), pass as JSON strings and parse inside the function:

```python
@function_tool
def run_parameter_experiment(strategy_name: str, parameter_changes: str, experiment_name: str) -> str:
    changes = json.loads(parameter_changes)  # Parse JSON string to dict
    return shared_tools.run_parameter_experiment(strategy_name, changes, experiment_name)
```

## Commands

```bash
# Set API key first
export MINIMAX_API_KEY="your-key"

# Install deps
pip install -r agent/OAI/requirements.txt

# Run
python agent/OAI/run.py                          # Full evaluation cycle
python agent/OAI/run.py --review-only            # Read-only analysis
python agent/OAI/run.py --experiment "Relaxed"   # Specific experiment
```

## Do NOT

- Do not import `from openai import OpenAI` (synchronous) — the SDK requires `AsyncOpenAI`
- Do not enable tracing (`set_tracing_disabled(True)` must stay)
- Do not hardcode API keys — always read from `MINIMAX_API_KEY` env var
- Do not modify `shared/tools.py` from this directory — shared tools are the single source of truth
