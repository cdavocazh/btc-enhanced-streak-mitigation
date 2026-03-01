# CLAUDE.md — agent/LangChain

Instructions for Claude Code when working in the `/agent/LangChain` directory.

## What This Is

LangChain implementation of the BTC Evaluation Agent, using MiniMax as the LLM backend via `langchain_openai.ChatOpenAI` with a custom `base_url`.

## Key Files

| File | Purpose |
|------|---------|
| `agent.py` | Agent definition, tool wrappers (`@tool`), LLM + executor setup |
| `run.py` | CLI entrypoint with `--review-only`, `--experiment`, full-cycle modes |
| `requirements.txt` | `langchain>=0.3.0`, `langchain-openai>=0.3.0` |

## Architecture

```
run.py  →  agent.py (create_agent_executor)  →  shared/tools.py (actual logic)
                      ↓
              AgentExecutor
                      ↓
              create_tool_calling_agent(llm, tools, prompt)
                      ↓
              ChatOpenAI(base_url="https://api.minimax.io/v1")
```

- `agent.py` wraps each shared tool with LangChain's `@tool` decorator
- `create_llm()` returns a `ChatOpenAI` with MiniMax's `base_url`
- `create_tool_calling_agent()` builds the agent with a `ChatPromptTemplate`
- `AgentExecutor` runs the loop with `max_iterations=20` and `handle_parsing_errors=True`

## MiniMax-Specific Notes

- Use `ChatOpenAI` (not `ChatAnthropic` or other providers) — MiniMax is OpenAI-compatible
- Set `base_url="https://api.minimax.io/v1"` and `model="MiniMax-M2.1"`
- Set `api_key` from `MINIMAX_API_KEY` env var
- `temperature=0.1` for deterministic evaluation behavior
- MiniMax supports tool calling via the OpenAI function-calling protocol

## Prompt Template

The agent uses `ChatPromptTemplate` with:
```python
[
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history", optional=True),  # Multi-turn support
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),              # Required by tool-calling agents
]
```

The `agent_scratchpad` placeholder is **required** by `create_tool_calling_agent` — do not remove it.

## Tool Parameter Convention

LangChain's `@tool` decorator infers schemas from type hints and docstrings. For complex parameters (dicts), pass as JSON strings:

```python
@tool
def run_parameter_experiment(strategy_name: str, parameter_changes: str, experiment_name: str) -> str:
    """...
    Args:
        parameter_changes: JSON string of parameter name to new value
    """
    changes = json.loads(parameter_changes)
    return shared_tools.run_parameter_experiment(strategy_name, changes, experiment_name)
```

## Commands

```bash
# Set API key first
export MINIMAX_API_KEY="your-key"

# Install deps
pip install -r agent/LangChain/requirements.txt

# Run
python agent/LangChain/run.py                          # Full evaluation cycle
python agent/LangChain/run.py --review-only            # Read-only analysis
python agent/LangChain/run.py --experiment "Relaxed"   # Specific experiment
```

## Do NOT

- Do not use `ChatOpenAI()` without `base_url` — it defaults to OpenAI's endpoint
- Do not remove `MessagesPlaceholder("agent_scratchpad")` from the prompt template
- Do not set `max_iterations` too low — complex evaluations may need 10-15 tool calls
- Do not hardcode API keys — always read from `MINIMAX_API_KEY` env var
- Do not modify `shared/tools.py` from this directory — shared tools are the single source of truth
