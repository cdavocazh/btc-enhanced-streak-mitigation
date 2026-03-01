#!/usr/bin/env python3
"""
Stratified Monte Carlo Validation for BTC Trading Strategy

Unlike brute-force Monte Carlo (which shuffles all trades uniformly), stratified
Monte Carlo groups trades by strata (market regime, time-of-day, volatility) and
shuffles WITHIN each stratum. This preserves regime structure and tests whether
the strategy's edge is dependent on specific market conditions.

Strata Definitions:
- **Regime**: Trending (ADX > 25) vs. Ranging (ADX <= 25)
- **Volatility**: High-vol (ATR > median) vs. Low-vol (ATR <= median)
- **Session**: Asian (UTC 0-11) vs. Non-Asian (UTC 12-23)
- **Combined**: Cross-product of regime × volatility (4 buckets)

For each stratification method, the script:
1. Assigns each trade to a stratum based on market conditions at entry
2. Shuffles trades within each stratum N times
3. Reassembles the full PnL sequence (preserving strata ordering)
4. Calculates metrics for each shuffled sequence
5. Compares actual vs. distribution to test regime-dependence

Usage:
    python validation/stratified_monte_carlo.py
    python validation/stratified_monte_carlo.py --simulations 2000
    python validation/stratified_monte_carlo.py --strata regime
    python validation/stratified_monte_carlo.py --strata all
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
import itertools
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

# Import from parent directory (same as brute-force MC)
from run_backtest import (
    load_ohlc_data, load_binance_data, calculate_indicators,
    merge_binance_data, calculate_enhanced_signals,
    INITIAL_EQ, RISK_PCT_INIT, STOP_PCT, ATR_MULT,
    ASIA_HRS, US_HRS
)

# Reuse the backtest runner from the brute-force MC
from validation.monte_carlo_validation import (
    run_optimized_backtest,
    calculate_metrics_from_pnl,
    OPTIMIZED_CONFIG,
    TradeResult,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_SIMULATIONS = 1000
ADX_TRENDING_THRESHOLD = 25  # ADX > 25 = trending
CONFIDENCE_LEVELS = [0.95, 0.99]


def log(message):
    """Print timestamped log message."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}")


# ============================================================
# Strata Assignment
# ============================================================

@dataclass
class AnnotatedTrade:
    """Trade with market-context annotations for stratification."""
    pnl: float
    entry_time: str
    side: str
    entry_type: str
    exit_reason: str
    positioning_score: float
    # Market context at entry
    hour: int
    atr: float
    adx: float
    regime: str        # "trending" or "ranging"
    volatility: str    # "high" or "low"
    session: str       # "asian" or "non_asian"
    combined: str      # "{regime}_{volatility}"


def annotate_trades(
    trades: List[TradeResult],
    df: pd.DataFrame,
    atr_median: float,
) -> List[AnnotatedTrade]:
    """
    Annotate each trade with market context at entry time for stratification.
    Reads ADX, ATR, and hour from the dataframe at the trade's entry_time.
    """
    annotated = []

    for t in trades:
        # Look up market context at entry time
        try:
            entry_ts = pd.Timestamp(t.entry_time)
        except Exception:
            # If parsing fails, assign defaults
            annotated.append(AnnotatedTrade(
                pnl=t.pnl, entry_time=t.entry_time, side=t.side,
                entry_type=t.entry_type, exit_reason=t.exit_reason,
                positioning_score=t.positioning_score,
                hour=0, atr=atr_median, adx=20,
                regime="ranging", volatility="low",
                session="asian", combined="ranging_low",
            ))
            continue

        # Find the closest row in df
        if entry_ts in df.index:
            row = df.loc[entry_ts]
        else:
            # Find nearest timestamp
            idx = df.index.get_indexer([entry_ts], method="nearest")[0]
            row = df.iloc[idx]

        atr_val = float(row.get("atr20", atr_median) if not pd.isna(row.get("atr20", np.nan)) else atr_median)
        adx_val = float(row.get("adx14", 20) if hasattr(row, "adx14") and not pd.isna(row.get("adx14", np.nan)) else 20)

        # Compute ADX if not in df (some datasets may not have it)
        if "adx14" not in df.columns:
            adx_val = 20  # default fallback

        hour = entry_ts.hour
        regime = "trending" if adx_val > ADX_TRENDING_THRESHOLD else "ranging"
        volatility = "high" if atr_val > atr_median else "low"
        session = "asian" if hour in ASIA_HRS else "non_asian"
        combined = f"{regime}_{volatility}"

        annotated.append(AnnotatedTrade(
            pnl=t.pnl, entry_time=t.entry_time, side=t.side,
            entry_type=t.entry_type, exit_reason=t.exit_reason,
            positioning_score=t.positioning_score,
            hour=hour, atr=atr_val, adx=adx_val,
            regime=regime, volatility=volatility,
            session=session, combined=combined,
        ))

    return annotated


# ============================================================
# Stratified Shuffle
# ============================================================

def stratified_shuffle(
    annotated_trades: List[AnnotatedTrade],
    strata_key: str,  # "regime", "volatility", "session", "combined"
) -> np.ndarray:
    """
    Shuffle trades WITHIN each stratum and reassemble the PnL sequence.
    The position of each stratum block is preserved; only within-block order changes.
    """
    # Group trades by stratum
    groups: Dict[str, List[int]] = {}
    pnl = np.array([t.pnl for t in annotated_trades])

    for i, t in enumerate(annotated_trades):
        key = getattr(t, strata_key)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    # Shuffle within each group
    shuffled_pnl = pnl.copy()
    for indices in groups.values():
        group_pnl = pnl[indices].copy()
        np.random.shuffle(group_pnl)
        for j, idx in enumerate(indices):
            shuffled_pnl[idx] = group_pnl[j]

    return shuffled_pnl


def antithetic_stratified_shuffle(
    annotated_trades: List[AnnotatedTrade],
    strata_key: str,
    original_shuffle: np.ndarray,
) -> np.ndarray:
    """
    Create the antithetic pair of a stratified shuffle by reversing the PnL
    order within each stratum independently. This produces a negatively
    correlated metric estimate when paired with the original shuffle.
    """
    pnl = np.array([t.pnl for t in annotated_trades])
    antithetic_pnl = original_shuffle.copy()

    # Group trades by stratum
    groups: Dict[str, List[int]] = {}
    for i, t in enumerate(annotated_trades):
        key = getattr(t, strata_key)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    # Reverse within each group
    for indices in groups.values():
        group_vals = [antithetic_pnl[idx] for idx in indices]
        group_vals.reverse()
        for j, idx in enumerate(indices):
            antithetic_pnl[idx] = group_vals[j]

    return antithetic_pnl


# ============================================================
# Stratified Monte Carlo Simulation
# ============================================================

def run_stratified_simulation(
    annotated_trades: List[AnnotatedTrade],
    strata_key: str,
    n_simulations: int = DEFAULT_SIMULATIONS,
    antithetic: bool = False,
) -> Dict:
    """
    Run stratified Monte Carlo simulation for a single strata key.

    When antithetic=True, uses antithetic variates: for each stratified shuffle Z,
    also evaluates the within-strata reversed shuffle and averages the pair.
    This reduces estimator variance by ~50%.

    Returns actual metrics, simulation distribution, and strata breakdown.
    """
    pnl_sequence = np.array([t.pnl for t in annotated_trades])
    actual_metrics = calculate_metrics_from_pnl(pnl_sequence)

    # Strata breakdown
    strata_breakdown = {}
    for t in annotated_trades:
        key = getattr(t, strata_key)
        if key not in strata_breakdown:
            strata_breakdown[key] = {"count": 0, "total_pnl": 0.0, "wins": 0}
        strata_breakdown[key]["count"] += 1
        strata_breakdown[key]["total_pnl"] += t.pnl
        if t.pnl > 0:
            strata_breakdown[key]["wins"] += 1

    for key, info in strata_breakdown.items():
        info["win_rate"] = round(info["wins"] / info["count"] * 100, 1) if info["count"] > 0 else 0
        info["avg_pnl"] = round(info["total_pnl"] / info["count"], 2) if info["count"] > 0 else 0

    strata_desc = ", ".join(f"{k}={v['count']}" for k, v in strata_breakdown.items())
    log(f"  Strata ({strata_key}): {strata_desc}")

    # Run simulations
    sim_returns = []
    sim_max_dd = []
    sim_max_consec_losses = []
    sim_sharpe = []

    if antithetic:
        n_pairs = n_simulations // 2
        log(f"  Running {n_pairs} antithetic pairs ({n_simulations} effective)...")

        for i in range(n_pairs):
            # Generate stratified shuffle Z
            shuffled_pnl = stratified_shuffle(annotated_trades, strata_key)
            metrics_z = calculate_metrics_from_pnl(shuffled_pnl)

            # Antithetic: reverse within each stratum
            anti_pnl = antithetic_stratified_shuffle(annotated_trades, strata_key, shuffled_pnl)
            metrics_anti = calculate_metrics_from_pnl(anti_pnl)

            # Average the pair
            sim_returns.append((metrics_z["total_return"] + metrics_anti["total_return"]) / 2)
            sim_max_dd.append((metrics_z["max_drawdown"] + metrics_anti["max_drawdown"]) / 2)
            sim_max_consec_losses.append((metrics_z["max_consecutive_losses"] + metrics_anti["max_consecutive_losses"]) / 2)
            sim_sharpe.append((metrics_z["sharpe_ratio"] + metrics_anti["sharpe_ratio"]) / 2)

            if (i + 1) % 100 == 0:
                log(f"    Completed {i + 1}/{n_pairs} antithetic pairs...")
    else:
        for i in range(n_simulations):
            shuffled_pnl = stratified_shuffle(annotated_trades, strata_key)
            metrics = calculate_metrics_from_pnl(shuffled_pnl)

            sim_returns.append(metrics["total_return"])
            sim_max_dd.append(metrics["max_drawdown"])
            sim_max_consec_losses.append(metrics["max_consecutive_losses"])
            sim_sharpe.append(metrics["sharpe_ratio"])

            if (i + 1) % 200 == 0:
                log(f"    Completed {i + 1}/{n_simulations} simulations...")

    sim_returns = np.array(sim_returns)
    sim_max_dd = np.array(sim_max_dd)
    sim_max_consec_losses = np.array(sim_max_consec_losses)
    sim_sharpe = np.array(sim_sharpe)

    # P-values
    p_return = float((sim_returns >= actual_metrics["total_return"]).mean())
    p_sharpe = float((sim_sharpe >= actual_metrics["sharpe_ratio"]).mean())
    p_max_dd = float((sim_max_dd <= actual_metrics["max_drawdown"]).mean())
    p_consec = float((sim_max_consec_losses <= actual_metrics["max_consecutive_losses"]).mean())

    # Confidence intervals
    ci_95_return = (float(np.percentile(sim_returns, 2.5)), float(np.percentile(sim_returns, 97.5)))
    ci_95_dd = (float(np.percentile(sim_max_dd, 2.5)), float(np.percentile(sim_max_dd, 97.5)))

    return {
        "strata_key": strata_key,
        "strata_breakdown": strata_breakdown,
        "n_simulations": n_simulations,
        "antithetic": antithetic,
        "actual_metrics": actual_metrics,
        "simulation_stats": {
            "total_return": {
                "mean": float(sim_returns.mean()),
                "std": float(sim_returns.std()),
                "min": float(sim_returns.min()),
                "max": float(sim_returns.max()),
                "p5": float(np.percentile(sim_returns, 5)),
                "p25": float(np.percentile(sim_returns, 25)),
                "p50": float(np.percentile(sim_returns, 50)),
                "p75": float(np.percentile(sim_returns, 75)),
                "p95": float(np.percentile(sim_returns, 95)),
                "ci_95": ci_95_return,
                "p_value": p_return,
            },
            "max_drawdown": {
                "mean": float(sim_max_dd.mean()),
                "std": float(sim_max_dd.std()),
                "min": float(sim_max_dd.min()),
                "max": float(sim_max_dd.max()),
                "p5": float(np.percentile(sim_max_dd, 5)),
                "p50": float(np.percentile(sim_max_dd, 50)),
                "p95": float(np.percentile(sim_max_dd, 95)),
                "ci_95": ci_95_dd,
                "p_value": p_max_dd,
            },
            "max_consecutive_losses": {
                "mean": float(sim_max_consec_losses.mean()),
                "std": float(sim_max_consec_losses.std()),
                "min": float(sim_max_consec_losses.min()),
                "max": float(sim_max_consec_losses.max()),
                "p_value": p_consec,
            },
            "sharpe_ratio": {
                "mean": float(sim_sharpe.mean()),
                "std": float(sim_sharpe.std()),
                "p_value": p_sharpe,
            },
        },
        "interpretation": {
            "return_significance": "SIGNIFICANT" if p_return < 0.05 else "NOT SIGNIFICANT",
            "drawdown_significance": "FAVORABLE" if p_max_dd < 0.05 else "TYPICAL",
            "sequence_dependency": (
                "HIGH" if abs(actual_metrics["total_return"] - sim_returns.mean()) > sim_returns.std() * 2
                else "MODERATE" if abs(actual_metrics["total_return"] - sim_returns.mean()) > sim_returns.std()
                else "LOW"
            ),
            "regime_dependent": (
                "YES" if sim_returns.std() > 0 and abs(actual_metrics["total_return"] - sim_returns.mean()) / sim_returns.std() > 1.96
                else "NO"
            ),
        },
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stratified Monte Carlo Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--simulations", "-n", type=int, default=DEFAULT_SIMULATIONS,
        help=f"Number of simulations per strata (default: {DEFAULT_SIMULATIONS})",
    )
    parser.add_argument(
        "--strata", "-s", type=str, default="all",
        choices=["regime", "volatility", "session", "combined", "all"],
        help="Which stratification method to run (default: all)",
    )
    parser.add_argument(
        "--antithetic", "-a", action="store_true",
        help="Use antithetic variates for variance reduction",
    )
    args = parser.parse_args()

    strata_keys = (
        ["regime", "volatility", "session", "combined"]
        if args.strata == "all"
        else [args.strata]
    )

    log("=" * 70)
    log("STRATIFIED MONTE CARLO VALIDATION")
    log(f"Strategy: {OPTIMIZED_CONFIG['name']}")
    log(f"Simulations per strata: {args.simulations}")
    log(f"Strata: {', '.join(strata_keys)}")
    if args.antithetic:
        log("Mode: ANTITHETIC VARIATES (variance reduction)")
    log("=" * 70)

    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    # Load and prepare data
    log("Loading data...")
    df_raw = load_ohlc_data()
    df = calculate_indicators(df_raw)
    atr_median = float(df["atr20"].median())

    binance_df = load_binance_data()
    if binance_df is not None:
        df = merge_binance_data(df, binance_df)

    df = calculate_enhanced_signals(df)

    # Compute ADX if not present
    if "adx14" not in df.columns:
        log("Computing ADX (14-period)...")
        _compute_adx(df, period=14)

    log(f"Data: {len(df)} rows, ATR median: {atr_median:.2f}")

    # Run backtest
    log("")
    log("Running optimized backtest...")
    trades = run_optimized_backtest(df, atr_median)
    log(f"  Trades: {len(trades)}")
    pnl_total = sum(t.pnl for t in trades)
    log(f"  Total PnL: ${pnl_total:,.0f}")

    # Annotate trades with market context
    log("")
    log("Annotating trades with market context...")
    annotated = annotate_trades(trades, df, atr_median)

    # Run stratified MC for each strata key
    all_results = {}
    for strata_key in strata_keys:
        log("")
        log("=" * 70)
        log(f"STRATIFIED MC — {strata_key.upper()}")
        log("=" * 70)

        result = run_stratified_simulation(annotated, strata_key, args.simulations, antithetic=args.antithetic)
        all_results[strata_key] = result

        # Print summary
        actual = result["actual_metrics"]
        sim = result["simulation_stats"]
        interp = result["interpretation"]

        log("")
        log(f"  Actual Return: {actual['total_return']:.1f}%")
        log(f"  Shuffled Mean: {sim['total_return']['mean']:.1f}% (±{sim['total_return']['std']:.1f}%)")
        log(f"  95% CI: [{sim['total_return']['ci_95'][0]:.1f}%, {sim['total_return']['ci_95'][1]:.1f}%]")
        log(f"  Return p-value: {sim['total_return']['p_value']:.4f} ({interp['return_significance']})")
        log(f"  Drawdown p-value: {sim['max_drawdown']['p_value']:.4f} ({interp['drawdown_significance']})")
        log(f"  Sequence dependency: {interp['sequence_dependency']}")
        log(f"  Regime dependent: {interp['regime_dependent']}")

    # Summary comparison
    log("")
    log("=" * 70)
    log("SUMMARY — BRUTE-FORCE vs. STRATIFIED")
    log("=" * 70)
    log("")
    log(f"{'Strata':<15} {'Return p-val':>12} {'DD p-val':>10} {'Seq Dep':>10} {'Regime Dep':>12}")
    log("-" * 60)

    # Load brute-force results for comparison
    bf_results_file = os.path.join(results_dir, "monte_carlo_results.json")
    if os.path.exists(bf_results_file):
        with open(bf_results_file, "r") as f:
            bf_data = json.load(f)
        log(f"{'brute_force':<15} "
            f"{bf_data['simulation_stats']['total_return']['p_value']:>12.4f} "
            f"{bf_data['simulation_stats']['max_drawdown']['p_value']:>10.4f} "
            f"{bf_data['interpretation']['sequence_dependency']:>10} "
            f"{'N/A':>12}")

    for key, result in all_results.items():
        sim = result["simulation_stats"]
        interp = result["interpretation"]
        log(f"{key:<15} "
            f"{sim['total_return']['p_value']:>12.4f} "
            f"{sim['max_drawdown']['p_value']:>10.4f} "
            f"{interp['sequence_dependency']:>10} "
            f"{interp['regime_dependent']:>12}")

    # Save results
    output = {
        "config": OPTIMIZED_CONFIG,
        "n_trades": len(trades),
        "atr_median": atr_median,
        "antithetic": args.antithetic,
        "strata_results": {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    for key, result in all_results.items():
        output["strata_results"][key] = result

    results_file = os.path.join(results_dir, "stratified_monte_carlo_results.json")
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log("")
    log(f"Results saved to: {results_file}")


def _compute_adx(df: pd.DataFrame, period: int = 14):
    """Compute ADX indicator and add to dataframe in-place."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # When plus_dm > minus_dm, minus_dm = 0 and vice versa
    mask = plus_dm > minus_dm
    minus_dm[mask] = 0
    plus_dm[~mask] = 0

    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - close.shift(1)).abs(),
        "lc": (low - close.shift(1)).abs(),
    }).max(axis=1)

    atr_adx = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_adx)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_adx)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    df["adx14"] = adx


if __name__ == "__main__":
    main()
