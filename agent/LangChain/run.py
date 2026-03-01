#!/usr/bin/env python3
"""
LangChain Agent CLI Entrypoint
===============================
Run the BTC Evaluation Agent using LangChain with MiniMax backend.
Streams real-time progress (tool calls, LLM thinking) to the terminal.

Usage:
    python run.py                          # Full autonomous evaluation loop
    python run.py --review-only            # Review latest results, no changes
    python run.py --experiment Relaxed     # Run specific experiment
"""

import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from LangChain.agent import create_agent_graph
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


def _extract_final_text(messages: list) -> str:
    """Extract the final AI text response from a message list."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            # Skip messages that are pure tool-call requests with no text
            if msg.tool_calls and not msg.content.strip():
                continue
            return msg.content
    return "(No output from agent)"


def run_agent_streamed(graph, messages: list) -> tuple:
    """
    Run the agent graph with streaming, printing real-time progress.
    Returns (final_text_output, all_messages).
    """
    tool_count = 0
    final_output = ""
    all_messages = list(messages)  # copy input

    # stream_mode="updates" yields {"node_name": state_update} per step
    for chunk in graph.stream({"messages": messages}, stream_mode="updates"):
        for node_name, update in chunk.items():
            if node_name == "tools":
                # Tool execution step — update contains {"messages": [ToolMessage(...)]}
                tool_msgs = update.get("messages", [])
                for tm in tool_msgs:
                    if isinstance(tm, ToolMessage):
                        tool_count += 1
                        content = tm.content or ""
                        preview = content[:120].replace("\n", " ")
                        if len(content) > 120:
                            preview += "..."
                        _print_progress(f"Tool returned ({len(content)} chars): {preview}", "  ✓")

            elif node_name == "agent":
                # Agent (LLM) step — update contains {"messages": [AIMessage(...)]}
                ai_msgs = update.get("messages", [])
                for ai_msg in ai_msgs:
                    if isinstance(ai_msg, AIMessage):
                        # Check if it has tool calls (thinking step)
                        if ai_msg.tool_calls:
                            for tc in ai_msg.tool_calls:
                                tool_name = tc.get("name", "unknown")
                                icon = _TOOL_ICONS.get(tool_name, "🔧")
                                args_preview = str(tc.get("args", {}))
                                if len(args_preview) > 80:
                                    args_preview = args_preview[:80] + "..."
                                _print_progress(f"Calling tool: {tool_name}({args_preview})", icon)
                        # Check for text content (final or intermediate response)
                        if ai_msg.content and ai_msg.content.strip():
                            if not ai_msg.tool_calls:
                                # This is a final text response
                                final_output = ai_msg.content

            # Collect all messages from the update
            step_msgs = update.get("messages", [])
            all_messages.extend(step_msgs)

    if tool_count > 0:
        _print_progress(f"Done — {tool_count} tool call(s) completed", "✅")

    # If we didn't capture final output from streaming, extract from messages
    if not final_output:
        final_output = _extract_final_text(all_messages)

    return final_output, all_messages


def run_agent(prompt: str):
    """Run the agent with the given prompt, with streaming + HITL loop."""
    graph = create_agent_graph()

    print("=" * 60)
    print("BTC EVALUATION AGENT (LangChain + MiniMax)")
    print("=" * 60)
    print()
    print(f"Started at {_timestamp()}")
    print()

    messages = [HumanMessage(content=prompt)]
    agent_output, all_messages = run_agent_streamed(graph, messages)

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
            all_messages.append(HumanMessage(content=follow_up))
            agent_output, all_messages = run_agent_streamed(graph, all_messages)

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
            all_messages.append(HumanMessage(content=follow_up))
            agent_output, all_messages = run_agent_streamed(graph, all_messages)

            print()
            print("-" * 60)
            print("REVISED OUTPUT:")
            print("-" * 60)
            print(agent_output)
            print()

            iteration += 1


def main():
    parser = argparse.ArgumentParser(
        description="BTC Evaluation Agent (LangChain + MiniMax)",
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
    run_agent(prompt)


if __name__ == "__main__":
    main()
