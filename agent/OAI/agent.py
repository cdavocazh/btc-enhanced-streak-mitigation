"""
OpenAI Agents SDK Implementation
=================================
BTC Evaluation Agent using OpenAI Agents SDK with MiniMax as the LLM backend.

MiniMax provides an OpenAI-compatible API, so we configure the SDK to use
MiniMax's endpoint directly via AsyncOpenAI client.
"""

import json
from typing import Dict, Any

from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.config import MINIMAX_BASE_URL, MINIMAX_MODEL, MINIMAX_API_KEY
from shared.prompts import SYSTEM_PROMPT
from shared import tools as shared_tools


# ============================================================
# Configure MiniMax as the OpenAI-compatible backend
# ============================================================

def create_client() -> AsyncOpenAI:
    """Create the MiniMax-backed OpenAI client."""
    if not MINIMAX_API_KEY:
        raise ValueError(
            "MINIMAX_API_KEY environment variable is required. "
            "Set it with: export MINIMAX_API_KEY='your-key-here'"
        )
    return AsyncOpenAI(
        base_url=MINIMAX_BASE_URL,
        api_key=MINIMAX_API_KEY,
    )


# ============================================================
# Tool Definitions (wrappers around shared tools)
# ============================================================

@function_tool
def read_latest_evaluation() -> str:
    """Read the most recent full evaluation results and report. Returns JSON with strategy performances, OOS efficiency, and statuses."""
    return shared_tools.read_latest_evaluation()


@function_tool
def read_latest_learnings() -> str:
    """Read the most recent strategy learning reports. Returns trend analysis and proposed parameter adjustments per strategy."""
    return shared_tools.read_latest_learnings()


@function_tool
def read_evaluation_history(limit: int = 5) -> str:
    """Read the last N evaluation iterations to compare trends across runs. Returns recent full_eval results, reports, and experiment folders.

    Args:
        limit: Number of recent iterations to include (default: 5)
    """
    return shared_tools.read_evaluation_history(limit)


@function_tool
def read_strategy_config() -> str:
    """Read the current strategy configuration including parameter ranges, strategy configs, and performance thresholds."""
    return shared_tools.read_strategy_config()


@function_tool
def read_market_data_status() -> str:
    """Check data freshness from binance-futures-data. Returns last update timestamps and CSV row counts."""
    return shared_tools.read_market_data_status()


@function_tool
def run_walk_forward_evaluation(
    strategy_name: str,
    entry_filter: str = "baseline",
    training_days: int = 14,
    testing_days: int = 7,
) -> str:
    """Run walk-forward evaluation for a specific strategy. Returns IS/OOS performance, efficiency, and trade count.

    Args:
        strategy_name: Strategy to evaluate (e.g. "Baseline", "Adaptive_ProgPos_Only")
        entry_filter: Entry filter config name (default: "baseline")
        training_days: In-sample training window in days (default: 14)
        testing_days: Out-of-sample testing window in days (default: 7)
    """
    return shared_tools.run_walk_forward_evaluation(
        strategy_name, entry_filter, training_days, testing_days
    )


@function_tool
def run_parameter_experiment(
    strategy_name: str,
    parameter_changes: str,
    experiment_name: str,
) -> str:
    """Create a strategy_adj folder, generate adjusted config, run backtest, and return results.

    Args:
        strategy_name: Strategy to experiment with (e.g. "Baseline", "Adaptive_Baseline")
        parameter_changes: JSON string of parameter name -> new value (e.g. '{"min_pos_score": 0.10, "adx_threshold": 15}')
        experiment_name: Descriptive name for this experiment (e.g. "Relaxed_Entry")
    """
    changes = json.loads(parameter_changes) if isinstance(parameter_changes, str) else parameter_changes
    return shared_tools.run_parameter_experiment(strategy_name, changes, experiment_name)


@function_tool
def generate_evaluation_report(results_json: str) -> str:
    """Generate a markdown evaluation report and save it to the eval folder.

    Args:
        results_json: JSON string with keys: type, training_days, testing_days, strategies (dict of name->metrics), recommendations (list), adjustments (list)
    """
    results_dict = json.loads(results_json) if isinstance(results_json, str) else results_json
    return shared_tools.generate_evaluation_report(results_dict)


@function_tool
def web_search(query: str) -> str:
    """Search the web for BTC trading strategy research, market analysis, or optimization techniques.

    Args:
        query: Search query string
    """
    return shared_tools.web_search(query)


@function_tool
def run_monte_carlo_validation(n_simulations: int = 1000, antithetic: bool = False) -> str:
    """Run brute-force Monte Carlo shuffle validation. Shuffles the full PnL sequence N times and compares actual vs. shuffled distributions. Returns p-values, confidence intervals, and significance. When antithetic=True, pairs each shuffle with its reverse for ~50% variance reduction. Runtime: ~30-60s.

    Args:
        n_simulations: Number of shuffle simulations (default: 1000)
        antithetic: Use antithetic variates for variance reduction (default: False)
    """
    return shared_tools.run_monte_carlo_validation(n_simulations, antithetic=antithetic)


@function_tool
def run_stratified_monte_carlo(strata: str = "all", n_simulations: int = 1000, antithetic: bool = False) -> str:
    """Run stratified Monte Carlo validation — shuffles trades WITHIN market-regime strata. Tests whether strategy edge is regime-dependent. Strata: "regime", "volatility", "session", "combined", or "all". When antithetic=True, pairs each shuffle with its within-strata reverse for variance reduction. Runtime: ~2-4 min.

    Args:
        strata: Stratification method (default: "all")
        n_simulations: Simulations per strata (default: 1000)
        antithetic: Use antithetic variates for variance reduction (default: False)
    """
    return shared_tools.run_stratified_monte_carlo(strata, n_simulations, antithetic=antithetic)


@function_tool
def read_monte_carlo_results() -> str:
    """Read all existing Monte Carlo results (brute-force, stratified, and WFO) without re-running. Quick access to past validation results."""
    return shared_tools.read_monte_carlo_results()


@function_tool
def run_particle_filter(n_particles: int = 500) -> str:
    """Run particle filter for regime-adaptive strategy parameter estimation. Maintains a distribution of hypotheses about current half_life, vol_scale, and signal_strength. Updates posterior with each trade. When uncertainty is high, recommends reduced position sizing. Returns posterior estimates, regime changes, and position scale recommendations. Runtime: ~1-3 min.

    Args:
        n_particles: Number of particles in the filter (default: 500)
    """
    return shared_tools.run_particle_filter(n_particles)


@function_tool
def read_particle_filter_results() -> str:
    """Read existing particle filter results without re-running. Returns latest posterior estimates, position scale recommendations, regime analysis, and uncertainty measures."""
    return shared_tools.read_particle_filter_results()


# ============================================================
# Agent Definition
# ============================================================

ALL_TOOLS = [
    read_latest_evaluation,
    read_latest_learnings,
    read_evaluation_history,
    read_strategy_config,
    read_market_data_status,
    run_walk_forward_evaluation,
    run_parameter_experiment,
    generate_evaluation_report,
    web_search,
    run_monte_carlo_validation,
    run_stratified_monte_carlo,
    read_monte_carlo_results,
    run_particle_filter,
    read_particle_filter_results,
]


def create_agent() -> Agent:
    """Create the BTC Evaluation Agent."""
    # Set up MiniMax as the default OpenAI client
    # CRITICAL: MiniMax only supports Chat Completions API, not the Responses API.
    # Without this, the SDK calls /responses which returns 404 on MiniMax.
    set_default_openai_api("chat_completions")
    client = create_client()
    set_default_openai_client(client)
    set_tracing_disabled(True)  # MiniMax doesn't support OpenAI tracing

    return Agent(
        name="BTC Evaluation Agent",
        model=MINIMAX_MODEL,
        instructions=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
    )
