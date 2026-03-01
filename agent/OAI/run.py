#!/usr/bin/env python3
"""
OAI Agent CLI Entrypoint
=========================
Run the BTC Evaluation Agent using OpenAI Agents SDK with MiniMax backend.
Streams real-time progress (tool calls, LLM thinking) to the terminal.

Usage:
    python run.py                          # Full autonomous evaluation loop
    python run.py --review-only            # Review latest results, no changes
    python run.py --experiment Relaxed     # Run specific experiment
"""

import argparse
import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents import Runner
from agents.stream_events import AgentUpdatedStreamEvent, RunItemStreamEvent
from OAI.agent import create_agent
from shared.prompts import (
    FULL_EVALUATION_PROMPT,
    REVIEW_ONLY_PROMPT,
    EXPERIMENT_PROMPT_TEMPLATE,
    EXECUTE_RECOMMENDATIONS_PROMPT,
)


# ============================================================
# Console progress helpers
# ============================================================

_TOOL_ICONS = {
    "read_latest_evaluation": "📊",
    "read_latest_learnings": "📚",
    "read_evaluation_history": "📈",
    "read_strategy_config": "⚙️",
    "read_market_data_status": "📡",
    "run_walk_forward_evaluation": "🔄",
    "run_parameter_experiment": "🧪",
    "generate_evaluation_report": "📝",
    "web_search": "🔍",
    "run_monte_carlo_validation": "🎲",
    "run_stratified_monte_carlo": "🎯",
    "read_monte_carlo_results": "📉",
    "run_particle_filter": "🔬",
    "read_particle_filter_results": "📊",
}


def _timestamp():
    return datetime.now().strftime("%H:%M:%S")


def _print_progress(msg: str, icon: str = "▸"):
    """Print a timestamped progress line."""
    print(f"  {icon} [{_timestamp()}] {msg}", flush=True)


async def run_agent_streamed(agent, prompt: str) -> str:
    """
    Run the agent with streaming, printing real-time progress.
    Returns the final text output.
    """
    result = Runner.run_streamed(agent, prompt)
    final_output = ""
    tool_count = 0

    async for event in result.stream_events():
        if isinstance(event, RunItemStreamEvent):
            if event.name == "tool_called":
                tool_count += 1
                item = event.item
                # item is a ToolCallItem with .description and .raw_item
                tool_name = getattr(item.raw_item, "name", "unknown") if hasattr(item, "raw_item") else "unknown"
                icon = _TOOL_ICONS.get(tool_name, "🔧")
                desc = getattr(item, "description", None) or tool_name
                _print_progress(f"Calling tool: {desc}", icon)

            elif event.name == "tool_output":
                item = event.item
                output_str = getattr(item, "output", "")
                # Print a brief summary of the output (first 120 chars)
                if output_str:
                    preview = output_str[:120].replace("\n", " ")
                    if len(output_str) > 120:
                        preview += "..."
                    _print_progress(f"Tool returned ({len(output_str)} chars): {preview}", "  ✓")

            elif event.name == "message_output_created":
                item = event.item
                raw = getattr(item, "raw_item", None)
                if raw and hasattr(raw, "content"):
                    # Collect all text content parts
                    for part in raw.content:
                        if hasattr(part, "text"):
                            final_output = part.text

        elif isinstance(event, AgentUpdatedStreamEvent):
            new_agent = event.new_agent
            _print_progress(f"Agent: {new_agent.name}", "🤖")

    # If we didn't capture output from streaming, fall back to result
    if not final_output and hasattr(result, "final_output"):
        final_output = result.final_output or ""

    if tool_count > 0:
        _print_progress(f"Done — {tool_count} tool call(s) completed", "✅")

    return final_output


async def run_agent(prompt: str):
    """Run the agent with the given prompt, with streaming + HITL loop."""
    agent = create_agent()

    print("=" * 60)
    print("BTC EVALUATION AGENT (OpenAI Agents SDK + MiniMax)")
    print("=" * 60)
    print()
    print(f"Started at {_timestamp()}")
    print()

    agent_output = await run_agent_streamed(agent, prompt)

    # Print final output
    print()
    print("-" * 60)
    print("AGENT OUTPUT:")
    print("-" * 60)
    print(agent_output)
    print()

    # HITL loop — keeps going until user says "n" or agent has nothing left
    iteration = 1
    while True:
        print("=" * 60)
        print(f"HUMAN REVIEW (iteration {iteration})")
        print("=" * 60)
        print()
        print("  y        = Accept & execute the recommendations above")
        print("  n        = Stop here, no further action")
        print("  <text>   = Send feedback for the agent to revise")
        print()
        response = input("Your choice: ").strip()

        if response.lower() == "n":
            print("\nStopped. No further action taken.")
            break

        elif response.lower() == "y":
            # Agent executes its own recommendations
            print()
            print(f"Executing recommendations... ({_timestamp()})")
            print("-" * 60)

            follow_up = EXECUTE_RECOMMENDATIONS_PROMPT.format(
                previous_output=agent_output
            )
            agent_output = await run_agent_streamed(agent, follow_up)

            print()
            print("-" * 60)
            print("EXECUTION RESULTS:")
            print("-" * 60)
            print(agent_output)
            print()

            iteration += 1

        else:
            # Feed custom feedback back to the agent
            print()
            print(f"Sending feedback to agent... ({_timestamp()})")
            print("-" * 60)

            follow_up = (
                f"The human reviewer provided this feedback: {response}\n\n"
                "Please revise your analysis and recommendations accordingly."
            )
            agent_output = await run_agent_streamed(agent, follow_up)

            print()
            print("-" * 60)
            print("REVISED OUTPUT:")
            print("-" * 60)
            print(agent_output)
            print()

            iteration += 1


def main():
    parser = argparse.ArgumentParser(
        description="BTC Evaluation Agent (OAI + MiniMax)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py                          # Full evaluation cycle
    python run.py --review-only            # Review only, no changes
    python run.py --experiment Relaxed     # Run specific experiment
        """,
    )

    parser.add_argument(
        "--review-only",
        action="store_true",
        help="Review latest results without making changes",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Run a specific named experiment",
    )

    args = parser.parse_args()

    # Determine prompt
    if args.review_only:
        prompt = REVIEW_ONLY_PROMPT
    elif args.experiment:
        prompt = EXPERIMENT_PROMPT_TEMPLATE.format(experiment_name=args.experiment)
    else:
        prompt = FULL_EVALUATION_PROMPT

    # Run the agent
    asyncio.run(run_agent(prompt))


if __name__ == "__main__":
    main()
