# OpenAI Agents SDK vs LangChain — Architecture Comparison

A detailed comparison of the two agent frameworks used in this project, both backed by MiniMax Coding Plan API.

---

## Architecture Overview

### OpenAI Agents SDK (OAI)

```
User Prompt
    ↓
Runner.run(agent, prompt)          ← async only
    ↓
Agent (name, model, instructions, tools)
    ↓
LLM call → tool calls? → execute tools → loop back
         → final output? → return
         → handoff? → switch agent → loop
    ↓
AsyncOpenAI(base_url=MiniMax)
```

**Core pattern:** `Agent` + `Runner`. You define an Agent with a plain-string system prompt and a list of `@function_tool`-decorated functions, then run it with `Runner.run()`. The Runner loop handles tool execution, handoffs, and termination automatically.

**Key characteristics:**
- Async-only execution (`asyncio.run()` wrapper needed for sync)
- No prompt template system — instructions are plain strings
- Global state configuration: `set_default_openai_api()`, `set_default_openai_client()`, `set_tracing_disabled()`
- `max_turns` on Runner controls loop iterations
- `output_type` accepts Pydantic models for structured output

### LangChain

```
User Prompt
    ↓
AgentExecutor.invoke({"input": prompt})    ← sync or async
    ↓
create_tool_calling_agent(llm, tools, prompt_template)
    ↓
ChatPromptTemplate:
    [system] → [chat_history] → [human] → [agent_scratchpad]
    ↓
LLM call → tool calls? → execute tools → loop back
         → final output? → return
    ↓
ChatOpenAI(base_url=MiniMax)
```

**Core pattern:** `LLM` + `Tools` + `PromptTemplate` + `Agent` + `AgentExecutor`. More composable but more assembly required. `create_tool_calling_agent()` builds the agent, `AgentExecutor` runs the loop.

**Key characteristics:**
- Both sync (`.invoke()`) and async (`.ainvoke()`) natively supported
- Structured prompt composition via `ChatPromptTemplate` + `MessagesPlaceholder`
- `AgentExecutor` controls: `max_iterations`, `handle_parsing_errors`, `verbose`, `return_intermediate_steps`
- No global state mutations — configuration is per-instance

---

## Side-by-Side: This Project's Implementations

| Aspect | OAI (`agent/OAI/agent.py`) | LangChain (`agent/LangChain/agent.py`) |
|--------|---------------------------|---------------------------------------|
| Agent creation | 6 lines + 4 lines global setup | ~11 lines (LLM + template + agent + executor) |
| Tool decorator | `@function_tool` | `@tool` |
| Schema inference | Type hints | Type hints + docstrings |
| LLM client | `AsyncOpenAI(base_url=...)` | `ChatOpenAI(base_url=...)` |
| Execution | `await Runner.run(agent, prompt)` | `executor.invoke({"input": prompt})` |
| Output access | `result.final_output` | `result["output"]` |
| Chat history | Manual (no built-in) | Built-in via `chat_history` key |
| Debug visibility | Minimal (needs tracing) | `verbose=True` logs every tool call |
| MiniMax setup | 3 global `set_*()` calls required | 1 constructor with `base_url` |

---

## Strengths

### OpenAI Agents SDK

| Strength | Detail |
|----------|--------|
| **Minimal API surface** | ~5 core concepts: Agent, Runner, function_tool, Handoff, Guardrail. Fast to learn |
| **Built-in guardrails** | `InputGuardrail`, `OutputGuardrail`, `ToolGuardrail` as first-class primitives. Input guardrails run in parallel with agent execution for low latency |
| **Native handoffs** | Agents can delegate to other agents seamlessly — peer-to-peer or manager patterns |
| **Structured output** | `output_type=PydanticModel` on Agent — clean, type-safe enforcement |
| **Built-in tracing** | Automatic instrumentation with traces + spans. Export to OpenAI Dashboard, Logfire, Langfuse, AgentOps, etc. |
| **MCP support** | Model Context Protocol server tools as first-class feature |
| **Realtime API** | Low-latency voice/multimodal agent support built-in |
| **Sessions & memory** | Short-term (per-conversation) and long-term (cross-conversation) memory with `SQLiteSession` |

### LangChain

| Strength | Detail |
|----------|--------|
| **Truly model-agnostic** | 100+ provider packages. Swap models by changing one import. No global state workarounds |
| **Cleaner third-party LLM setup** | MiniMax = 1 constructor call. Anthropic, Google, Ollama all have dedicated packages |
| **Massive ecosystem** | 1,000+ integrations: vector stores, document loaders, retrievers, output parsers, memory backends |
| **RAG pipeline** | Full retrieval stack: loaders, splitters, embeddings, vector stores, retrievers — all composable |
| **Rich memory options** | ConversationBufferMemory, SummaryMemory, VectorStoreRetrieverMemory, LangMem for long-term |
| **LangGraph** | Graph-based orchestration for complex stateful, branching multi-agent workflows |
| **Prompt templates** | `ChatPromptTemplate` with variable injection and message placeholders |
| **Sync + async** | Both first-class. No `asyncio.run()` wrapper needed |
| **Verbose debugging** | `verbose=True` + `return_intermediate_steps=True` gives immediate tool-call visibility |
| **LangSmith** | Production-grade observability, tracing, evaluation, and prompt management |

---

## Weaknesses

### OpenAI Agents SDK

| Weakness | Detail |
|----------|--------|
| **Non-OpenAI providers require workarounds** | 3 global calls needed for MiniMax: `set_default_openai_api("chat_completions")`, `set_default_openai_client()`, `set_tracing_disabled(True)`. Brittle. |
| **Async-only Runner** | Must wrap in `asyncio.run()` for synchronous contexts |
| **No prompt template system** | Instructions are plain strings — no variable injection or composition |
| **Tracing incompatible with many providers** | Must disable tracing for non-OpenAI LLMs |
| **Pre-1.0 API** | Version 0.0.7+ — API may change. Not yet stable |
| **Limited multi-model orchestration** | Designed primarily for OpenAI models; mixing providers is awkward |
| **No built-in RAG** | Must bring your own retrieval pipeline |
| **No output parsers** | Only Pydantic `output_type` — no composable parser ecosystem |

### LangChain

| Weakness | Detail |
|----------|--------|
| **Steeper learning curve** | Many concepts: Chains, Agents, Tools, Memory, Callbacks, Runnables, LangGraph nodes/edges |
| **More boilerplate** | Agent creation requires: LLM constructor + prompt template (4 message types) + `create_tool_calling_agent()` + `AgentExecutor()` |
| **Frequent API churn** | Chains → LCEL → Runnables → `create_react_agent` → `create_agent`. Migration overhead is real |
| **No built-in guardrails** | Must implement as middleware, graph nodes, or external libraries (Guardrails AI, NeMo) |
| **Over-abstraction risk** | Composable primitives can lead to sprawling, hard-to-debug graphs |
| **Commercial lock-in potential** | LangSmith (paid) is the primary observability path |
| **Streaming structured output** | Still an area with limited documentation |

---

## Extensibility

### Adding Tools

| | OAI | LangChain |
|--|-----|-----------|
| **Basic** | `@function_tool` decorator | `@tool` decorator |
| **Advanced** | Implement `FunctionTool` class | `StructuredTool.from_function()` or `BaseTool` subclass |
| **Schema** | Auto-inferred from type hints | Auto-inferred from type hints + docstrings |
| **Complex params** | Pass as JSON string, parse inside | Same pattern |
| **Effort** | ~5 lines per tool | ~5 lines per tool |

Both are nearly identical for tool definition. In this project, both implementations wrap the same `shared/tools.py` functions with their respective decorators — the wrappers are 1:1 identical in structure.

As of March 2026, both implementations expose **14 tools** covering evaluation reading, walk-forward execution, parameter experiments, web search, Monte Carlo validation (with antithetic variates), and particle filter regime estimation.

### Custom Model Providers

| | OAI | LangChain |
|--|-----|-----------|
| **OpenAI-compatible** | `AsyncOpenAI(base_url=...)` + 3 global calls | `ChatOpenAI(base_url=...)` — one call |
| **Anthropic** | LiteLLM adapter or custom `ModelProvider` | `from langchain_anthropic import ChatAnthropic` |
| **Google** | LiteLLM adapter | `from langchain_google_genai import ChatGoogleGenerativeAI` |
| **Local (Ollama)** | Not natively supported | `from langchain_ollama import ChatOllama` |
| **Custom** | Implement `ModelProvider` + `Model` interface | Subclass `BaseChatModel` |

**Verdict:** LangChain has dramatically better multi-provider support. OAI SDK is OpenAI-first with workarounds for others.

### Multi-Agent Patterns

| | OAI | LangChain |
|--|-----|-----------|
| **Manager pattern** | Agents as tools invoked by orchestrator | Supervisor node in LangGraph |
| **Peer handoff** | First-class `Handoff` primitive | Custom edges in LangGraph |
| **Hierarchical** | Nested agents with handoffs | Subgraphs in LangGraph |
| **Ease of setup** | Simple — define handoffs on Agent | More complex — design graph topology |
| **Flexibility** | Limited to handoff patterns | Arbitrary DAG control flow |

### Memory

| | OAI | LangChain |
|--|-----|-----------|
| **Short-term** | `SQLiteSession` per conversation | `ConversationBufferMemory`, `ConversationSummaryMemory` |
| **Long-term** | Global memory (cross-conversation) | `VectorStoreRetrieverMemory`, LangMem toolkit |
| **Persistence** | SQLite | SQLite, Redis, MongoDB, PostgreSQL, vector stores |
| **Maturity** | New (2025) | Mature (3+ years of iteration) |

### Guardrails

| | OAI | LangChain |
|--|-----|-----------|
| **Input** | `InputGuardrail` — runs parallel to LLM | Middleware (v1.0) or custom graph node |
| **Output** | `OutputGuardrail` — validates before return | Middleware or custom graph node |
| **Tool** | `ToolGuardrail` — pre/post tool execution | Custom wrapper or graph node |
| **First-class?** | Yes — native primitives | No — bring your own |

---

## Feature Modules

| Feature | OAI | LangChain |
|---------|-----|-----------|
| **Tracing** | Built-in auto-instrumentation. Export to 10+ platforms | LangSmith (paid) or Langfuse/OpenTelemetry |
| **Streaming** | `Runner.run_streamed()` | `.stream()`, `.astream()`, `astream_events()` |
| **Structured Output** | `output_type=PydanticModel` | `llm.with_structured_output(PydanticModel)` |
| **Human-in-the-Loop** | Built-in mechanism | LangGraph `interrupt()` at any node |
| **RAG / Vector Stores** | Not included (bring your own) | Core strength: 100+ loaders, 20+ vector stores |
| **Output Parsers** | Only Pydantic validation | `PydanticOutputParser`, `JsonOutputParser`, `StrOutputParser`, etc. |
| **MCP Protocol** | First-class support | Via LangGraph integration |
| **Callbacks** | `TracingProcessor` (trace/span events) | `BaseCallbackHandler` (LLM, tool, chain, agent events) |
| **Document Loaders** | Not included | 100+ (PDF, CSV, HTML, Notion, S3, etc.) |
| **Text Splitting** | Not included | `RecursiveCharacterTextSplitter`, `TokenTextSplitter`, etc. |
| **Evaluation** | Via tracing + external tools | LangSmith evaluations framework |
| **Deployment** | Roll your own | LangServe, LangFlow |
| **Voice/Realtime** | Built-in Realtime API support | Not natively included |

---

## Third-Party LLM Compatibility (MiniMax)

### OAI SDK Setup (from this project)

```python
# 3 global calls required before agent creation
set_default_openai_api("chat_completions")   # MiniMax doesn't support Responses API
client = AsyncOpenAI(base_url="https://api.minimax.io/v1", api_key=KEY)
set_default_openai_client(client)
set_tracing_disabled(True)                    # MiniMax rejects tracing headers
```

### LangChain Setup (from this project)

```python
# 1 constructor call — no global state mutation
llm = ChatOpenAI(
    base_url="https://api.minimax.io/v1",
    api_key=KEY,
    model="MiniMax-M2.5",
    temperature=0.1,
)
```

**Winner: LangChain.** Cleaner, no global state, no special workarounds.

---

## Community & Ecosystem Maturity

| Metric | OAI SDK | LangChain |
|--------|---------|-----------|
| **Age** | ~1 year (March 2025) | ~3 years (2023) |
| **GitHub Stars** | ~19k (Python SDK) | ~118k |
| **API Stability** | Pre-1.0 (v0.0.7+) | v1.0 (October 2025) |
| **Integrations** | Growing (~50+) | 1,000+ |
| **Backing** | OpenAI | LangChain Inc. ($125M Series B, Sequoia) |
| **Observability** | Built-in tracing | LangSmith (commercial) |
| **Visual Builder** | None | LangFlow |
| **Deployment** | None | LangServe |
| **API Churn** | Low (new project) | High (multiple redesigns) |

---

## When to Use Which

| Scenario | Recommended |
|----------|-------------|
| Simple agent with OpenAI models | **OAI SDK** — minimal code, fast setup |
| Non-OpenAI LLM (MiniMax, Anthropic, local) | **LangChain** — cleaner provider support |
| Need guardrails as first-class feature | **OAI SDK** — built-in primitives |
| Complex multi-step RAG pipeline | **LangChain** — unmatched retrieval ecosystem |
| Multi-agent handoff patterns | **OAI SDK** — native Handoff support |
| Complex stateful workflows (DAG/graph) | **LangChain** — LangGraph |
| Production observability | Both (OAI built-in tracing, LangChain via LangSmith) |
| Team already uses LangChain | **LangChain** — leverage existing knowledge |
| Want smallest dependency footprint | **OAI SDK** — 2 packages vs 3+ |
| Need voice/realtime agents | **OAI SDK** — built-in Realtime API |

---

## Summary

**OpenAI Agents SDK** is focused and opinionated — fewer concepts, built-in guardrails and handoffs, excellent if you're in the OpenAI ecosystem. Non-OpenAI providers work but require workarounds.

**LangChain** is broad and composable — massive ecosystem, truly model-agnostic, better for complex pipelines involving RAG, memory, and multi-provider setups. More boilerplate and steeper learning curve.

For this project (MiniMax + evaluation tools), LangChain's provider setup is cleaner, but OAI SDK's guardrails would be valuable if we add input/output validation. Both produce identical evaluation outputs using the shared tool layer.
