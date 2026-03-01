"""
Shared Tool Functions
=====================
Tool functions used by both OAI and LangChain agent implementations.
Each function is a standalone callable that reads/writes project files.
"""

import os
import sys
import json
import glob
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from .config import (
    REPO_DIR, EVAL_DIR, DATA_DIR, RESULTS_DIR, LEARNINGS_DIR,
    OOS_HEALTHY, OOS_WARNING,
)


def _find_latest_file(directory: str, pattern: str) -> Optional[str]:
    """Find the most recently modified file matching a glob pattern."""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _find_latest_files(directory: str, pattern: str, limit: int = 10) -> List[str]:
    """Find the most recent files matching a glob pattern."""
    files = glob.glob(os.path.join(directory, pattern))
    files.sort(key=os.path.getmtime, reverse=True)
    return files[:limit]


def _read_json(filepath: str) -> Optional[Dict]:
    """Read and parse a JSON file."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {"error": str(e)}


def _read_text(filepath: str, max_lines: int = 200) -> str:
    """Read a text file, truncated to max_lines."""
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
        if len(lines) > max_lines:
            return "".join(lines[:max_lines]) + f"\n... (truncated, {len(lines)} total lines)"
        return "".join(lines)
    except FileNotFoundError:
        return f"File not found: {filepath}"


# ============================================================
# Tool 1: Read Latest Evaluation
# ============================================================

def read_latest_evaluation() -> str:
    """
    Read the most recent full evaluation results and report.
    Returns a JSON summary of all strategy performances, OOS efficiency, and statuses.
    """
    results = {}

    # Find latest full_eval JSON
    latest_eval = _find_latest_file(RESULTS_DIR, "full_eval_*.json")
    if latest_eval:
        data = _read_json(latest_eval)
        results["full_eval"] = {
            "file": os.path.basename(latest_eval),
            "data": data,
        }
    else:
        results["full_eval"] = {"error": "No full evaluation results found"}

    # Find latest evaluation report
    latest_report = _find_latest_file(EVAL_DIR, "EVALUATION_REPORT_*.md")
    if latest_report:
        results["report"] = {
            "file": os.path.basename(latest_report),
            "content": _read_text(latest_report),
        }
    else:
        results["report"] = {"error": "No evaluation report found"}

    # Find latest strategy_adj folder
    adj_dirs = glob.glob(os.path.join(EVAL_DIR, "strategy_adj_*"))
    if adj_dirs:
        latest_adj = max(adj_dirs, key=os.path.getmtime)
        experiment_file = os.path.join(latest_adj, "experiment_results.json")
        if os.path.exists(experiment_file):
            results["latest_experiment"] = {
                "folder": os.path.basename(latest_adj),
                "data": _read_json(experiment_file),
            }
        else:
            results["latest_experiment"] = {
                "folder": os.path.basename(latest_adj),
                "data": "No experiment_results.json found",
            }
    else:
        results["latest_experiment"] = {"error": "No strategy adjustment folders found"}

    return json.dumps(results, indent=2, default=str)


# ============================================================
# Tool 2: Read Latest Learnings
# ============================================================

def read_latest_learnings() -> str:
    """
    Read the most recent strategy learning reports.
    Returns trend analysis and proposed parameter adjustments per strategy.
    """
    learnings = {}

    # Find all learning files, grouped by strategy
    learning_files = _find_latest_files(LEARNINGS_DIR, "learning_*.json", limit=20)

    if not learning_files:
        return json.dumps({"error": "No learning files found"}, indent=2)

    # Group by strategy (take most recent per strategy)
    seen_strategies = set()
    for filepath in learning_files:
        data = _read_json(filepath)
        if data and "strategy_name" in data:
            strategy = data["strategy_name"]
            if strategy not in seen_strategies:
                seen_strategies.add(strategy)
                learnings[strategy] = {
                    "file": os.path.basename(filepath),
                    "timestamp": data.get("timestamp", "unknown"),
                    "trend_direction": data.get("trend", {}).get("trend_direction", "unknown"),
                    "trend_confidence": data.get("trend", {}).get("confidence", 0),
                    "proposed_adjustments": data.get("proposed_adjustments", []),
                    "applied": data.get("applied", False),
                    "notes": data.get("notes", ""),
                }

    return json.dumps(learnings, indent=2, default=str)


# ============================================================
# Tool 2b: Read Evaluation History (multiple iterations)
# ============================================================

def read_evaluation_history(limit: int = 5) -> str:
    """
    Read the last N evaluation iterations to see trends over time.
    Returns recent full_eval results, evaluation reports, and experiment folders
    so the agent can compare across iterations and spot improving/declining trends.

    Args:
        limit: Number of recent iterations to include (default: 5)
    """
    history = {
        "evaluations": [],
        "reports": [],
        "experiments": [],
    }

    # --- Recent full_eval JSON files ---
    eval_files = _find_latest_files(RESULTS_DIR, "full_eval_*.json", limit=limit)
    for filepath in eval_files:
        data = _read_json(filepath)
        summary = {
            "file": os.path.basename(filepath),
            "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M"),
        }
        # Extract key metrics per strategy
        strategies = data.get("strategies", {}) if isinstance(data, dict) else {}
        strat_summary = {}
        for name, sdata in strategies.items():
            strat_summary[name] = {
                "oos_efficiency": sdata.get("oos_efficiency", None),
                "oos_return": sdata.get("oos_return", None),
                "is_return": sdata.get("is_return", None),
                "total_trades": sdata.get("total_trades", None),
                "win_rate": sdata.get("win_rate", None),
                "max_drawdown": sdata.get("max_drawdown", None),
            }
        summary["strategies"] = strat_summary
        history["evaluations"].append(summary)

    # --- Recent evaluation reports ---
    report_files = _find_latest_files(EVAL_DIR, "EVALUATION_REPORT_*.md", limit=limit)
    for filepath in report_files:
        history["reports"].append({
            "file": os.path.basename(filepath),
            "modified": datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M"),
            "preview": _read_text(filepath, max_lines=30),
        })

    # --- Recent strategy_adj experiment folders ---
    adj_dirs = sorted(
        glob.glob(os.path.join(EVAL_DIR, "strategy_adj_*")),
        key=os.path.getmtime,
        reverse=True,
    )[:limit]

    for adj_dir in adj_dirs:
        entry = {
            "folder": os.path.basename(adj_dir),
            "modified": datetime.fromtimestamp(os.path.getmtime(adj_dir)).strftime("%Y-%m-%d %H:%M"),
        }
        # Read experiment_results.json if present
        exp_file = os.path.join(adj_dir, "experiment_results.json")
        if os.path.exists(exp_file):
            entry["results"] = _read_json(exp_file)
        else:
            entry["results"] = "No experiment_results.json"
        # Read adjusted_config.py header for experiment name
        cfg_file = os.path.join(adj_dir, "adjusted_config.py")
        if os.path.exists(cfg_file):
            with open(cfg_file, "r") as f:
                header_lines = []
                for line in f:
                    header_lines.append(line.rstrip())
                    if len(header_lines) >= 8:
                        break
            entry["config_header"] = "\n".join(header_lines)
        history["experiments"].append(entry)

    # --- Summary counts ---
    history["summary"] = {
        "evaluations_found": len(history["evaluations"]),
        "reports_found": len(history["reports"]),
        "experiments_found": len(history["experiments"]),
        "showing_latest": limit,
    }

    return json.dumps(history, indent=2, default=str)


# ============================================================
# Tool 3: Read Strategy Config
# ============================================================

def read_strategy_config() -> str:
    """
    Read the current strategy configuration from eval/config.py.
    Returns parameter ranges, strategy configs, and thresholds.
    """
    config_file = os.path.join(EVAL_DIR, "config.py")
    if not os.path.exists(config_file):
        return json.dumps({"error": "eval/config.py not found"})

    # Import config dynamically
    sys.path.insert(0, EVAL_DIR)
    try:
        import importlib
        if "config" in sys.modules:
            importlib.reload(sys.modules["config"])
        else:
            import config as eval_config  # noqa: F811

        eval_config = sys.modules.get("config")
        if eval_config is None:
            import config as eval_config  # noqa: F811

        result = {
            "strategy_parameter_ranges": eval_config.STRATEGY_PARAMETER_RANGES,
            "tiered_strategy_configs": eval_config.TIERED_STRATEGY_CONFIGS,
            "adaptive_strategy_configs": eval_config.ADAPTIVE_STRATEGY_CONFIGS,
            "entry_filter_configs": eval_config.ENTRY_FILTER_CONFIGS,
            "risk_tiers": eval_config.RISK_TIERS,
            "streak_rules": {str(k): v for k, v in eval_config.STREAK_RULES.items()},
            "thresholds": {
                "return_degradation_warning": eval_config.DEFAULT_THRESHOLDS.return_degradation_warning,
                "max_drawdown_warning": eval_config.DEFAULT_THRESHOLDS.max_drawdown_warning,
                "oos_efficiency_min": eval_config.DEFAULT_THRESHOLDS.out_of_sample_efficiency,
                "win_rate_min": eval_config.DEFAULT_THRESHOLDS.win_rate_min,
                "sharpe_min": eval_config.DEFAULT_THRESHOLDS.sharpe_min,
            },
        }
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to load config: {e}"})
    finally:
        sys.path.pop(0)


# ============================================================
# Tool 4: Read Market Data Status
# ============================================================

def read_market_data_status() -> str:
    """
    Check data freshness from binance-futures-data and CSV row counts.
    Returns data coverage summary.
    """
    status = {}

    # Read last_timestamps.json
    timestamps_file = os.path.join(DATA_DIR, "last_timestamps.json")
    if os.path.exists(timestamps_file):
        status["timestamps"] = _read_json(timestamps_file)
    else:
        status["timestamps"] = {"error": "last_timestamps.json not found"}

    # Count rows in each CSV
    csv_files = [
        "price.csv",
        "top_trader_position_ratio.csv",
        "top_trader_account_ratio.csv",
        "global_ls_ratio.csv",
        "funding_rate.csv",
        "open_interest.csv",
    ]

    status["csv_row_counts"] = {}
    for csv_name in csv_files:
        csv_path = os.path.join(DATA_DIR, csv_name)
        if os.path.exists(csv_path):
            try:
                with open(csv_path, "r") as f:
                    line_count = sum(1 for _ in f) - 1  # Subtract header
                status["csv_row_counts"][csv_name] = line_count
            except Exception as e:
                status["csv_row_counts"][csv_name] = f"Error: {e}"
        else:
            status["csv_row_counts"][csv_name] = "File not found"

    return json.dumps(status, indent=2, default=str)


# ============================================================
# Tool 5: Run Walk-Forward Evaluation
# ============================================================

def run_walk_forward_evaluation(
    strategy_name: str,
    entry_filter: str = "baseline",
    training_days: int = 14,
    testing_days: int = 7,
) -> str:
    """
    Run walk-forward evaluation for a specific strategy.
    Uses eval/run_evaluation.py infrastructure.
    Returns IS/OOS performance, efficiency, and trade count.
    """
    cmd = [
        sys.executable,
        os.path.join(EVAL_DIR, "run_evaluation.py"),
        "--full",
        "--training-days", str(training_days),
        "--testing-days", str(testing_days),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=EVAL_DIR,
        )

        output = result.stdout
        if result.returncode != 0:
            output += f"\nSTDERR:\n{result.stderr}"

        # Read the latest results file that was just generated
        latest_eval = _find_latest_file(RESULTS_DIR, "full_eval_*.json")
        if latest_eval:
            eval_data = _read_json(latest_eval)
            strategy_result = eval_data.get("strategies", {}).get(strategy_name, {})
            return json.dumps({
                "strategy": strategy_name,
                "entry_filter": entry_filter,
                "training_days": training_days,
                "testing_days": testing_days,
                "results": strategy_result,
                "results_file": os.path.basename(latest_eval),
                "stdout_tail": output[-2000:] if len(output) > 2000 else output,
            }, indent=2, default=str)

        return json.dumps({
            "strategy": strategy_name,
            "output": output[-3000:] if len(output) > 3000 else output,
            "returncode": result.returncode,
        }, indent=2, default=str)

    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Walk-forward evaluation timed out (10 min limit)"})
    except Exception as e:
        return json.dumps({"error": f"Failed to run evaluation: {e}"})


# ============================================================
# Tool 6: Run Parameter Experiment
# ============================================================

def run_parameter_experiment(
    strategy_name: str,
    parameter_changes: Dict[str, Any],
    experiment_name: str,
) -> str:
    """
    Create a strategy_adj folder, generate adjusted config, and run backtest.
    Returns experiment results.

    Args:
        strategy_name: e.g. "Baseline", "Adaptive_ProgPos_Only"
        parameter_changes: dict of parameter name -> new value
        experiment_name: descriptive name for this experiment
    """
    today = datetime.now().strftime("%Y%m%d")
    adj_dir = os.path.join(EVAL_DIR, f"strategy_adj_{today}")
    os.makedirs(adj_dir, exist_ok=True)

    # Find the most recent template for run_adjusted_backtest.py
    template_dirs = sorted(
        glob.glob(os.path.join(EVAL_DIR, "strategy_adj_*")),
        key=os.path.getmtime,
        reverse=True,
    )

    template_backtest = None
    template_report = None
    for td in template_dirs:
        if td == adj_dir:
            continue
        bt_file = os.path.join(td, "run_adjusted_backtest.py")
        rp_file = os.path.join(td, "generate_report.py")
        if os.path.exists(bt_file) and template_backtest is None:
            template_backtest = bt_file
        if os.path.exists(rp_file) and template_report is None:
            template_report = rp_file
        if template_backtest and template_report:
            break

    # Copy template files
    if template_backtest:
        shutil.copy2(template_backtest, os.path.join(adj_dir, "run_adjusted_backtest.py"))
    if template_report:
        shutil.copy2(template_report, os.path.join(adj_dir, "generate_report.py"))

    # Generate adjusted_config.py
    config_content = _generate_adjusted_config(strategy_name, parameter_changes, experiment_name)
    config_path = os.path.join(adj_dir, "adjusted_config.py")
    with open(config_path, "w") as f:
        f.write(config_content)

    # Run the backtest
    if not template_backtest:
        return json.dumps({
            "error": "No template run_adjusted_backtest.py found",
            "adj_dir": adj_dir,
        })

    try:
        result = subprocess.run(
            [sys.executable, "run_adjusted_backtest.py"],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=adj_dir,
        )

        # Read experiment results
        results_file = os.path.join(adj_dir, "experiment_results.json")
        if os.path.exists(results_file):
            experiment_data = _read_json(results_file)
        else:
            experiment_data = None

        return json.dumps({
            "experiment_name": experiment_name,
            "strategy": strategy_name,
            "parameter_changes": parameter_changes,
            "adj_dir": os.path.basename(adj_dir),
            "results": experiment_data,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
            "stderr_tail": result.stderr[-1000:] if result.stderr else "",
        }, indent=2, default=str)

    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Backtest timed out (15 min limit)"})
    except Exception as e:
        return json.dumps({"error": f"Failed to run experiment: {e}"})


def _generate_adjusted_config(
    strategy_name: str,
    parameter_changes: Dict[str, Any],
    experiment_name: str,
) -> str:
    """Generate adjusted_config.py content for an experiment."""

    # Base parameters (from existing configs)
    baseline_params = {
        "Baseline": {
            "min_pos_long": 0.4,
            "rsi_long_range": (20, 45),
            "pullback_range": (0.5, 3.0),
            "min_pos_score": 0.15,
            "stop_atr_mult": 1.8,
            "tp_atr_mult": 4.5,
        },
        "Adaptive_Baseline": {
            "min_pos_long": 0.4,
            "rsi_long_range": (20, 45),
            "pullback_range": (0.5, 3.0),
            "min_pos_score": 0.15,
            "adx_threshold": 20,
            "progressive_pos": {0: 0.4, 1: 0.5, 2: 0.6, 3: 0.8, 4: 1.0},
            "cooldown_bars": 96,
        },
        "Adaptive_ProgPos_Only": {
            "min_pos_long": 0.4,
            "rsi_long_range": (20, 45),
            "pullback_range": (0.5, 3.0),
            "min_pos_score": 0.15,
            "progressive_pos": {0: 0.4, 1: 0.5, 2: 0.6, 3: 0.8, 4: 1.0},
            "cooldown_bars": 96,
        },
    }

    return f'''"""
Adjusted Strategy Configuration - Agent Generated
==================================================
Experiment: {experiment_name}
Strategy: {strategy_name}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from typing import Dict, Any, List

BASELINE_PARAMS = {repr(baseline_params)}

ADJUSTMENT_EXPERIMENTS: List[Dict[str, Any]] = [
    {{
        "name": "{experiment_name}",
        "description": "Agent-proposed parameter adjustment",
        "changes": {repr(parameter_changes)},
        "rationale": "Proposed by evaluation agent to improve OOS efficiency",
    }},
]

STRATEGY_VARIANTS = {repr([strategy_name])}

WF_CONFIG = {{
    "training_window_days": 30,
    "testing_window_days": 14,
    "step_size_days": 7,
}}

ASIAN_HOURS = set(range(0, 12))
ATR_PERIOD = 14
RSI_PERIOD = 56
SMA_PERIOD = 200
ADX_PERIOD = 14
'''


# ============================================================
# Tool 7: Generate Evaluation Report
# ============================================================

def generate_evaluation_report(results_dict: Dict[str, Any]) -> str:
    """
    Generate a markdown evaluation report matching the project's standard format.
    Saves to eval/EVALUATION_REPORT_YYYYMMDD_HHMMSS.md.
    Returns the report content.
    """
    now = datetime.now(timezone.utc)
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    date_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Build report
    lines = [
        "# BTC Strategy Evaluation Report (Agent-Generated)",
        "",
        f"**Date:** {date_str}",
        f"**Type:** {results_dict.get('type', 'Agent Evaluation')}",
        f"**Training:** {results_dict.get('training_days', 14)} days",
        f"**Testing:** {results_dict.get('testing_days', 7)} days",
        "",
        "## Strategy Performance Summary",
        "",
        "| Strategy | IS Return | OOS Return | OOS Efficiency | Trades | Win Rate | Max DD | Status |",
        "|----------|-----------|------------|----------------|--------|----------|--------|--------|",
    ]

    strategies = results_dict.get("strategies", {})
    critical_list = []
    warning_list = []

    for name, data in strategies.items():
        oos_eff = data.get("oos_efficiency", 0)
        if oos_eff >= OOS_HEALTHY:
            status = "HEALTHY"
            status_icon = "🟢"
        elif oos_eff >= OOS_WARNING:
            status = "WARNING"
            status_icon = "🟡"
            warning_list.append(name)
        else:
            status = "CRITICAL"
            status_icon = "🔴"
            critical_list.append(name)

        lines.append(
            f"| {name} "
            f"| {data.get('is_return', 0):.2f}% "
            f"| {data.get('oos_return', 0):.2f}% "
            f"| {oos_eff:.2f} "
            f"| {data.get('total_trades', 'N/A')} "
            f"| {data.get('win_rate', 'N/A')} "
            f"| {data.get('max_drawdown', 'N/A')} "
            f"| {status_icon} {status} |"
        )

    # Key findings
    lines.extend(["", "## Key Findings", ""])
    if critical_list:
        lines.append(f"- **Critical strategies:** {', '.join(critical_list)}")
    if warning_list:
        lines.append(f"- **Warning strategies:** {', '.join(warning_list)}")
    if not critical_list and not warning_list:
        lines.append("- All strategies are performing within healthy parameters")

    # Recommendations
    recommendations = results_dict.get("recommendations", [])
    if recommendations:
        lines.extend(["", "## Recommendations", ""])
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"{i}. {rec}")

    # Parameter adjustments
    adjustments = results_dict.get("adjustments", [])
    if adjustments:
        lines.extend(["", "## Parameter Adjustments Applied", ""])
        lines.append("| Strategy | Parameter | Old Value | New Value | Change |")
        lines.append("|----------|-----------|-----------|-----------|--------|")
        for adj in adjustments:
            lines.append(
                f"| {adj.get('strategy', '')} "
                f"| {adj.get('parameter', '')} "
                f"| {adj.get('old_value', '')} "
                f"| {adj.get('new_value', '')} "
                f"| {adj.get('change_pct', '')}% |"
            )

    lines.extend([
        "",
        "---",
        "*Generated by BTC Strategy Evaluation Agent*",
    ])

    report_content = "\n".join(lines)

    # Save report
    report_path = os.path.join(EVAL_DIR, f"EVALUATION_REPORT_{timestamp_str}.md")
    with open(report_path, "w") as f:
        f.write(report_content)

    return json.dumps({
        "report_file": os.path.basename(report_path),
        "report_path": report_path,
        "content": report_content,
    }, indent=2)


# ============================================================
# Tool 8: Web Search
# ============================================================

def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for BTC trading strategy research, market analysis,
    or optimization techniques. Uses DuckDuckGo (no API key required).

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
    """
    try:
        # Try the newer `ddgs` package first, fall back to legacy `duckduckgo_search`
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })

        return json.dumps({
            "query": query,
            "result_count": len(results),
            "results": results,
        }, indent=2)

    except ImportError:
        return json.dumps({
            "query": query,
            "error": (
                "Search package not installed. "
                "Install with: pip install ddgs"
            ),
            "results": [],
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "query": query,
            "error": f"Search failed: {e}",
            "results": [],
        }, indent=2)


# ============================================================
# Tool 9: Run Monte Carlo Validation (Brute-Force)
# ============================================================

VALIDATION_DIR = os.path.join(REPO_DIR, "validation")
MC_RESULTS_DIR = os.path.join(VALIDATION_DIR, "results")


def run_monte_carlo_validation(n_simulations: int = 1000, antithetic: bool = False) -> str:
    """
    Run brute-force Monte Carlo shuffle validation.
    Shuffles the full PnL sequence N times and compares actual vs. shuffled distributions.
    Returns p-values, confidence intervals, and significance interpretation.

    When antithetic=True, uses antithetic variates for variance reduction:
    each shuffled sequence Z is paired with its reverse Z[::-1] and metrics
    are averaged. This halves estimator variance, tightening confidence intervals
    for Sharpe, drawdown, and VaR.

    Runtime: ~30-60 seconds for 1000 simulations.

    Args:
        n_simulations: Number of shuffle simulations to run (default: 1000)
        antithetic: Use antithetic variates for variance reduction (default: False)
    """
    script = os.path.join(VALIDATION_DIR, "monte_carlo_validation.py")
    if not os.path.exists(script):
        return json.dumps({"error": f"Script not found: {script}"})

    cmd = [sys.executable, script, "--simulations", str(n_simulations)]
    if antithetic:
        cmd.append("--antithetic")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=VALIDATION_DIR,
        )

        # Read the results file
        results_file = os.path.join(MC_RESULTS_DIR, "monte_carlo_results.json")
        if os.path.exists(results_file):
            mc_data = _read_json(results_file)
            return json.dumps({
                "type": "brute_force",
                "n_simulations": mc_data.get("n_simulations", n_simulations),
                "n_trades": mc_data.get("n_trades", 0),
                "antithetic": mc_data.get("antithetic", False),
                "variance_reduction_pct": mc_data.get("variance_reduction_pct"),
                "actual_metrics": mc_data.get("actual_metrics", {}),
                "simulation_stats": mc_data.get("simulation_stats", {}),
                "interpretation": mc_data.get("interpretation", {}),
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-1500:] if len(result.stdout) > 1500 else result.stdout,
            }, indent=2, default=str)

        return json.dumps({
            "type": "brute_force",
            "output": result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout,
            "stderr": result.stderr[-1000:] if result.stderr else "",
            "returncode": result.returncode,
        }, indent=2, default=str)

    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Monte Carlo validation timed out (5 min limit)"})
    except Exception as e:
        return json.dumps({"error": f"Failed to run Monte Carlo: {e}"})


# ============================================================
# Tool 10: Run Stratified Monte Carlo Validation
# ============================================================

def run_stratified_monte_carlo(
    strata: str = "all",
    n_simulations: int = 1000,
    antithetic: bool = False,
) -> str:
    """
    Run stratified Monte Carlo validation — shuffles trades WITHIN market-regime strata.
    Tests whether the strategy's edge is regime-dependent.

    Strata options:
    - "regime": Trending (ADX > 25) vs. Ranging
    - "volatility": High-vol (ATR > median) vs. Low-vol
    - "session": Asian (UTC 0-11) vs. Non-Asian
    - "combined": Regime × Volatility (4 buckets)
    - "all": Run all four strata (default)

    When antithetic=True, uses antithetic variates for variance reduction:
    each stratified shuffle is paired with its within-strata reverse.

    Runtime: ~2-4 minutes for 1000 simulations × all strata.

    Args:
        strata: Stratification method — "regime", "volatility", "session", "combined", or "all" (default: "all")
        n_simulations: Number of simulations per strata (default: 1000)
        antithetic: Use antithetic variates for variance reduction (default: False)
    """
    script = os.path.join(VALIDATION_DIR, "stratified_monte_carlo.py")
    if not os.path.exists(script):
        return json.dumps({"error": f"Script not found: {script}"})

    cmd = [
        sys.executable, script,
        "--simulations", str(n_simulations),
        "--strata", strata,
    ]
    if antithetic:
        cmd.append("--antithetic")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            cwd=VALIDATION_DIR,
        )

        # Read the results file
        results_file = os.path.join(MC_RESULTS_DIR, "stratified_monte_carlo_results.json")
        if os.path.exists(results_file):
            mc_data = _read_json(results_file)
            # Build a summary with key findings per strata
            summary = {
                "type": "stratified",
                "strata_requested": strata,
                "n_trades": mc_data.get("n_trades", 0),
                "strata_results": {},
                "returncode": result.returncode,
            }
            for skey, sdata in mc_data.get("strata_results", {}).items():
                summary["strata_results"][skey] = {
                    "strata_breakdown": sdata.get("strata_breakdown", {}),
                    "actual_return": sdata.get("actual_metrics", {}).get("total_return", 0),
                    "sim_return_mean": sdata.get("simulation_stats", {}).get("total_return", {}).get("mean", 0),
                    "sim_return_std": sdata.get("simulation_stats", {}).get("total_return", {}).get("std", 0),
                    "return_p_value": sdata.get("simulation_stats", {}).get("total_return", {}).get("p_value", None),
                    "dd_p_value": sdata.get("simulation_stats", {}).get("max_drawdown", {}).get("p_value", None),
                    "interpretation": sdata.get("interpretation", {}),
                }
            return json.dumps(summary, indent=2, default=str)

        return json.dumps({
            "type": "stratified",
            "output": result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout,
            "stderr": result.stderr[-1000:] if result.stderr else "",
            "returncode": result.returncode,
        }, indent=2, default=str)

    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Stratified Monte Carlo timed out (10 min limit)"})
    except Exception as e:
        return json.dumps({"error": f"Failed to run stratified Monte Carlo: {e}"})


# ============================================================
# Tool 11: Read Monte Carlo Results
# ============================================================

def read_monte_carlo_results() -> str:
    """
    Read all existing Monte Carlo validation results (both brute-force and stratified).
    Returns the latest results from validation/results/ without re-running the simulations.
    Use this for quick access to past MC results.
    """
    results = {}

    # Brute-force results
    bf_file = os.path.join(MC_RESULTS_DIR, "monte_carlo_results.json")
    if os.path.exists(bf_file):
        bf_data = _read_json(bf_file)
        results["brute_force"] = {
            "file": "monte_carlo_results.json",
            "modified": datetime.fromtimestamp(os.path.getmtime(bf_file)).strftime("%Y-%m-%d %H:%M"),
            "n_trades": bf_data.get("n_trades", 0),
            "n_simulations": bf_data.get("n_simulations", 0),
            "actual_metrics": bf_data.get("actual_metrics", {}),
            "interpretation": bf_data.get("interpretation", {}),
            "key_p_values": {
                "return": bf_data.get("simulation_stats", {}).get("total_return", {}).get("p_value"),
                "max_drawdown": bf_data.get("simulation_stats", {}).get("max_drawdown", {}).get("p_value"),
                "sharpe": bf_data.get("simulation_stats", {}).get("sharpe_ratio", {}).get("p_value"),
            },
        }
    else:
        results["brute_force"] = {"error": "No brute-force MC results found. Run run_monte_carlo_validation() first."}

    # Stratified results
    strat_file = os.path.join(MC_RESULTS_DIR, "stratified_monte_carlo_results.json")
    if os.path.exists(strat_file):
        strat_data = _read_json(strat_file)
        strata_summary = {}
        for skey, sdata in strat_data.get("strata_results", {}).items():
            strata_summary[skey] = {
                "strata_breakdown": sdata.get("strata_breakdown", {}),
                "interpretation": sdata.get("interpretation", {}),
                "return_p_value": sdata.get("simulation_stats", {}).get("total_return", {}).get("p_value"),
                "dd_p_value": sdata.get("simulation_stats", {}).get("max_drawdown", {}).get("p_value"),
            }
        results["stratified"] = {
            "file": "stratified_monte_carlo_results.json",
            "modified": datetime.fromtimestamp(os.path.getmtime(strat_file)).strftime("%Y-%m-%d %H:%M"),
            "n_trades": strat_data.get("n_trades", 0),
            "strata": strata_summary,
        }
    else:
        results["stratified"] = {"error": "No stratified MC results found. Run run_stratified_monte_carlo() first."}

    # WFO results
    wfo_file = os.path.join(MC_RESULTS_DIR, "wfo_results.json")
    if os.path.exists(wfo_file):
        wfo_data = _read_json(wfo_file)
        analysis = wfo_data.get("analysis", {})
        results["walk_forward_optimization"] = {
            "file": "wfo_results.json",
            "modified": datetime.fromtimestamp(os.path.getmtime(wfo_file)).strftime("%Y-%m-%d %H:%M"),
            "summary": analysis.get("summary", {}),
            "efficiency": analysis.get("efficiency", {}),
            "parameter_stability": analysis.get("parameter_stability", {}),
            "recommended_config": analysis.get("recommended_config", {}),
        }
    else:
        results["walk_forward_optimization"] = {"error": "No WFO results found."}

    return json.dumps(results, indent=2, default=str)


# ============================================================
# Tool 12: Run Particle Filter Analysis
# ============================================================

def run_particle_filter(n_particles: int = 500) -> str:
    """
    Run particle filter for regime-adaptive strategy parameter estimation.

    Maintains a cloud of N particles, each a hypothesis about current market
    regime parameters (half_life, vol_scale, signal_strength). Updates the
    posterior as each trade is processed. When the posterior is wide (high
    uncertainty), position size should be reduced.

    Returns posterior estimates, position scale recommendations, and
    regime change history.

    Runtime: ~1-3 minutes depending on trade count.

    Args:
        n_particles: Number of particles in the filter (default: 500)
    """
    script = os.path.join(VALIDATION_DIR, "particle_filter.py")
    if not os.path.exists(script):
        return json.dumps({"error": f"Script not found: {script}"})

    cmd = [sys.executable, script, "--particles", str(n_particles)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=VALIDATION_DIR,
        )

        # Read the results file
        results_file = os.path.join(MC_RESULTS_DIR, "particle_filter_results.json")
        if os.path.exists(results_file):
            pf_data = _read_json(results_file)
            pf_results = pf_data.get("results", {})
            return json.dumps({
                "type": "particle_filter",
                "n_particles": pf_data.get("n_particles", n_particles),
                "n_trades": pf_data.get("n_trades", 0),
                "final_posterior": pf_results.get("final_posterior", {}),
                "final_ess": pf_results.get("final_ess", 0),
                "resampling_events": pf_results.get("resampling_events", 0),
                "regime_changes": pf_results.get("regime_changes_detected", []),
                "position_scale_summary": pf_results.get("position_scale_history_summary", {}),
                "interpretation": pf_results.get("interpretation", {}),
                "returncode": result.returncode,
                "stdout_tail": result.stdout[-1500:] if len(result.stdout) > 1500 else result.stdout,
            }, indent=2, default=str)

        return json.dumps({
            "type": "particle_filter",
            "output": result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout,
            "stderr": result.stderr[-1000:] if result.stderr else "",
            "returncode": result.returncode,
        }, indent=2, default=str)

    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Particle filter timed out (5 min limit)"})
    except Exception as e:
        return json.dumps({"error": f"Failed to run particle filter: {e}"})


# ============================================================
# Tool 13: Read Particle Filter Results
# ============================================================

def read_particle_filter_results() -> str:
    """
    Read existing particle filter results without re-running.
    Returns the latest posterior estimates, position scale recommendations,
    and regime analysis from validation/results/particle_filter_results.json.
    """
    results_file = os.path.join(MC_RESULTS_DIR, "particle_filter_results.json")

    if not os.path.exists(results_file):
        return json.dumps({
            "error": "No particle filter results found. Run run_particle_filter() first."
        })

    pf_data = _read_json(results_file)
    pf_results = pf_data.get("results", {})

    return json.dumps({
        "file": "particle_filter_results.json",
        "modified": datetime.fromtimestamp(os.path.getmtime(results_file)).strftime("%Y-%m-%d %H:%M"),
        "n_particles": pf_data.get("n_particles", 0),
        "n_trades": pf_data.get("n_trades", 0),
        "final_posterior": pf_results.get("final_posterior", {}),
        "final_ess": pf_results.get("final_ess", 0),
        "resampling_events": pf_results.get("resampling_events", 0),
        "regime_changes_count": len(pf_results.get("regime_changes_detected", [])),
        "regime_changes_recent": pf_results.get("regime_changes_detected", [])[-5:],
        "position_scale_summary": pf_results.get("position_scale_history_summary", {}),
        "interpretation": pf_results.get("interpretation", {}),
    }, indent=2, default=str)


# ============================================================
# Tool Registry (for easy import by both frameworks)
# ============================================================

TOOL_FUNCTIONS = {
    "read_latest_evaluation": read_latest_evaluation,
    "read_latest_learnings": read_latest_learnings,
    "read_evaluation_history": read_evaluation_history,
    "read_strategy_config": read_strategy_config,
    "read_market_data_status": read_market_data_status,
    "run_walk_forward_evaluation": run_walk_forward_evaluation,
    "run_parameter_experiment": run_parameter_experiment,
    "generate_evaluation_report": generate_evaluation_report,
    "web_search": web_search,
    "run_monte_carlo_validation": run_monte_carlo_validation,
    "run_stratified_monte_carlo": run_stratified_monte_carlo,
    "read_monte_carlo_results": read_monte_carlo_results,
    "run_particle_filter": run_particle_filter,
    "read_particle_filter_results": read_particle_filter_results,
}
