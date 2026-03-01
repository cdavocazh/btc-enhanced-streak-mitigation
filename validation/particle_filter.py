#!/usr/bin/env python3
"""
Particle Filter for Regime-Adaptive Strategy Parameters
========================================================

A Sequential Monte Carlo (particle filter) approach to online Bayesian
estimation of strategy parameters. Unlike rolling-window point estimates,
the particle filter maintains a full posterior distribution over parameters,
providing uncertainty quantification that directly informs position sizing.

Parameters Tracked:
- half_life (2-50 bars): Mean-reversion speed
- vol_scale (0.5-3.0): Current volatility relative to historical median
- signal_strength (0.0-2.0): Positioning-signal predictive alpha

Algorithm (Bootstrap Particle Filter):
1. INITIALIZE: Draw N particles from prior distributions
2. For each trade observation:
   a. PREDICT: Evolve particles with random-walk noise (regime change flexibility)
   b. UPDATE: Compute likelihood of observed PnL under each particle's model
   c. REWEIGHT: Multiply weights by likelihood, normalize
   d. RESAMPLE: If ESS < threshold, systematic resampling
3. OUTPUT: Weighted posterior (mean, std, percentiles) + position scale

Position Sizing Integration:
- Uncertainty = weighted std of signal_strength across particles
- High uncertainty -> reduce position (min 30% of full size)
- Tight posterior -> full position size

Usage:
    python validation/particle_filter.py
    python validation/particle_filter.py --particles 1000
    python validation/particle_filter.py --resample-threshold 0.3
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional

# Import shared infrastructure
from run_backtest import (
    load_ohlc_data, load_binance_data, calculate_indicators,
    merge_binance_data, calculate_enhanced_signals,
    INITIAL_EQ, ASIA_HRS, US_HRS
)
from validation.monte_carlo_validation import (
    run_optimized_backtest,
    TradeResult,
    OPTIMIZED_CONFIG,
)
from validation.stratified_monte_carlo import (
    AnnotatedTrade,
    annotate_trades,
    _compute_adx,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_N_PARTICLES = 500
DEFAULT_RESAMPLE_THRESHOLD = 0.5  # Resample when ESS < N * threshold

# Prior parameter ranges
PRIOR_RANGES = {
    "half_life": (2.0, 50.0),       # bars
    "vol_scale": (0.5, 3.0),        # multiplier on ATR
    "signal_strength": (0.0, 2.0),  # positioning alpha
}

# Random walk noise std (per-step evolution)
EVOLUTION_NOISE = {
    "half_life": 1.0,
    "vol_scale": 0.05,
    "signal_strength": 0.03,
}

# Position scaling parameters
UNCERTAINTY_THRESHOLD = 0.5  # signal_strength std above this -> reduce position
MIN_POSITION_SCALE = 0.3    # Minimum position scale (30%)


def log(message):
    """Print timestamped log message."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{timestamp}] {message}")


# ============================================================
# Particle Filter Engine
# ============================================================

@dataclass
class Particle:
    """A single hypothesis about current strategy parameters."""
    half_life: float
    vol_scale: float
    signal_strength: float
    weight: float = 1.0


class ParticleFilterEngine:
    """
    Bootstrap Particle Filter for online regime-adaptive parameter estimation.

    Maintains N particles, each representing a hypothesis about current
    market regime parameters. Updates the posterior with each observed trade.
    """

    def __init__(
        self,
        n_particles: int = DEFAULT_N_PARTICLES,
        resample_threshold: float = DEFAULT_RESAMPLE_THRESHOLD,
    ):
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.particles: List[Particle] = []
        self.history: List[Dict] = []
        self.resampling_events = 0

        self._initialize_particles()

    def _initialize_particles(self):
        """Draw particles from uniform prior distributions."""
        self.particles = []
        w = 1.0 / self.n_particles

        for _ in range(self.n_particles):
            p = Particle(
                half_life=np.random.uniform(*PRIOR_RANGES["half_life"]),
                vol_scale=np.random.uniform(*PRIOR_RANGES["vol_scale"]),
                signal_strength=np.random.uniform(*PRIOR_RANGES["signal_strength"]),
                weight=w,
            )
            self.particles.append(p)

    def predict(self):
        """
        Evolve particles with random-walk noise.
        This allows the filter to track regime changes over time.
        """
        for p in self.particles:
            p.half_life += np.random.normal(0, EVOLUTION_NOISE["half_life"])
            p.vol_scale += np.random.normal(0, EVOLUTION_NOISE["vol_scale"])
            p.signal_strength += np.random.normal(0, EVOLUTION_NOISE["signal_strength"])

            # Clip to valid ranges
            p.half_life = np.clip(p.half_life, *PRIOR_RANGES["half_life"])
            p.vol_scale = np.clip(p.vol_scale, *PRIOR_RANGES["vol_scale"])
            p.signal_strength = np.clip(p.signal_strength, *PRIOR_RANGES["signal_strength"])

    def update(self, trade: AnnotatedTrade, atr: float):
        """
        Update particle weights based on observed trade PnL.

        Likelihood model: PnL ~ Normal(mu, sigma)
        where:
          mu = signal_strength * positioning_score * atr * vol_scale * direction
          sigma = atr * vol_scale * base_std_multiplier
        """
        observed_pnl = trade.pnl
        positioning_score = trade.positioning_score
        direction = 1.0 if trade.side == "long" else -1.0

        total_weight = 0.0
        for p in self.particles:
            likelihood = self._compute_likelihood(
                observed_pnl, positioning_score, direction, atr, p
            )
            p.weight *= likelihood
            total_weight += p.weight

        # Normalize weights
        if total_weight > 0:
            for p in self.particles:
                p.weight /= total_weight
        else:
            # All weights collapsed — reinitialize
            w = 1.0 / self.n_particles
            for p in self.particles:
                p.weight = w

    def _compute_likelihood(
        self,
        observed_pnl: float,
        positioning_score: float,
        direction: float,
        atr: float,
        particle: Particle,
    ) -> float:
        """
        Compute likelihood of observed PnL under particle's model.

        Uses a Normal distribution centered on the model's predicted PnL.
        """
        # Predicted PnL: depends on signal_strength, positioning, and volatility
        # A good signal_strength means positioning_score is predictive of direction
        predicted_pnl = (
            particle.signal_strength
            * positioning_score
            * direction
            * atr
            * particle.vol_scale
            * INITIAL_EQ * 0.05 / atr  # approximate position size
        )

        # Predicted std: scales with volatility
        # Use half_life to modulate: shorter half_life -> tighter distribution
        # (more mean-reverting -> more predictable -> tighter)
        base_std = atr * particle.vol_scale * INITIAL_EQ * 0.05 / atr
        half_life_factor = np.sqrt(particle.half_life / 10.0)  # normalize around 10 bars
        predicted_std = max(base_std * half_life_factor, 1.0)  # floor to avoid division by zero

        # Normal PDF
        z = (observed_pnl - predicted_pnl) / predicted_std
        likelihood = np.exp(-0.5 * z * z) / (predicted_std * np.sqrt(2 * np.pi))

        # Add a small floor to prevent weight collapse
        return max(likelihood, 1e-300)

    def effective_sample_size(self) -> float:
        """
        Compute effective sample size: ESS = 1 / sum(w_i^2).
        ESS close to N = particles well-distributed.
        ESS close to 1 = particle degeneracy, need resampling.
        """
        weights = np.array([p.weight for p in self.particles])
        return 1.0 / np.sum(weights ** 2)

    def resample(self):
        """
        Systematic resampling: replace low-weight particles with
        copies of high-weight particles. Preserves total particle count.
        """
        n = self.n_particles
        weights = np.array([p.weight for p in self.particles])

        # Systematic resampling
        positions = (np.arange(n) + np.random.uniform()) / n
        cumulative_weights = np.cumsum(weights)

        new_particles = []
        i, j = 0, 0
        while i < n:
            if positions[i] < cumulative_weights[j]:
                source = self.particles[j]
                new_particles.append(Particle(
                    half_life=source.half_life,
                    vol_scale=source.vol_scale,
                    signal_strength=source.signal_strength,
                    weight=1.0 / n,
                ))
                i += 1
            else:
                j += 1
                if j >= n:
                    j = n - 1

        self.particles = new_particles
        self.resampling_events += 1

    def get_posterior(self) -> Dict:
        """
        Compute weighted posterior statistics for each parameter.
        Returns mean, std, and percentiles.
        """
        weights = np.array([p.weight for p in self.particles])

        posterior = {}
        for param_name in ["half_life", "vol_scale", "signal_strength"]:
            values = np.array([getattr(p, param_name) for p in self.particles])

            # Weighted statistics
            w_mean = np.average(values, weights=weights)
            w_var = np.average((values - w_mean) ** 2, weights=weights)
            w_std = np.sqrt(w_var)

            # Weighted percentiles (approximate via sorted values)
            sorted_indices = np.argsort(values)
            sorted_values = values[sorted_indices]
            sorted_weights = weights[sorted_indices]
            cum_weights = np.cumsum(sorted_weights)

            def _weighted_percentile(p_val):
                idx = np.searchsorted(cum_weights, p_val / 100.0)
                idx = min(idx, len(sorted_values) - 1)
                return float(sorted_values[idx])

            posterior[param_name] = {
                "mean": float(w_mean),
                "std": float(w_std),
                "p5": _weighted_percentile(5),
                "p25": _weighted_percentile(25),
                "p50": _weighted_percentile(50),
                "p75": _weighted_percentile(75),
                "p95": _weighted_percentile(95),
            }

        return posterior

    def get_position_scale(self) -> float:
        """
        Compute position scale factor based on signal_strength uncertainty.

        When the posterior over signal_strength is wide, we are uncertain about
        the edge, so position size should be reduced.

        Returns: float in [MIN_POSITION_SCALE, 1.0]
        """
        posterior = self.get_posterior()
        ss_std = posterior["signal_strength"]["std"]

        if ss_std >= UNCERTAINTY_THRESHOLD:
            scale = max(MIN_POSITION_SCALE, 1.0 - ss_std / UNCERTAINTY_THRESHOLD * 0.7)
        else:
            scale = 1.0 - (ss_std / UNCERTAINTY_THRESHOLD) * (1.0 - MIN_POSITION_SCALE) * 0.3
            scale = min(1.0, max(MIN_POSITION_SCALE, scale))

        return float(scale)

    def run_on_trades(
        self,
        trades: List[AnnotatedTrade],
        df: pd.DataFrame,
    ) -> Dict:
        """
        Process all trades sequentially through the particle filter.

        For each trade:
        1. Predict (evolve particles)
        2. Update (reweight based on observed PnL)
        3. Resample if needed
        4. Record posterior snapshot

        Returns full results with time-series of posteriors.
        """
        log(f"Processing {len(trades)} trades through particle filter...")
        log(f"  Particles: {self.n_particles}")
        log(f"  Resample threshold: {self.resample_threshold}")

        position_scale_history = []
        posterior_history = []
        regime_changes = []
        prev_regime = None

        for i, trade in enumerate(trades):
            # 1. Predict
            self.predict()

            # 2. Update
            self.update(trade, trade.atr)

            # 3. Resample if ESS is low
            ess = self.effective_sample_size()
            if ess < self.n_particles * self.resample_threshold:
                self.resample()

            # 4. Record snapshot
            posterior = self.get_posterior()
            pos_scale = self.get_position_scale()
            position_scale_history.append(pos_scale)

            # Detect regime changes (signal_strength mean crosses 0.5 threshold)
            ss_mean = posterior["signal_strength"]["mean"]
            hl_mean = posterior["half_life"]["mean"]
            vs_mean = posterior["vol_scale"]["mean"]

            # Classify current regime
            if vs_mean > 1.5 and ss_mean > 1.0:
                current_regime = "trending_high_vol"
            elif vs_mean > 1.5:
                current_regime = "ranging_high_vol"
            elif ss_mean > 1.0:
                current_regime = "trending_low_vol"
            else:
                current_regime = "ranging_low_vol"

            if prev_regime is not None and current_regime != prev_regime:
                regime_changes.append({
                    "trade_index": i,
                    "entry_time": trade.entry_time,
                    "from_regime": prev_regime,
                    "to_regime": current_regime,
                    "ess": float(ess),
                })
            prev_regime = current_regime

            # Record posterior every 10 trades and at the last trade
            if i % 10 == 0 or i == len(trades) - 1:
                posterior_history.append({
                    "trade_index": i,
                    "entry_time": trade.entry_time,
                    "posterior": posterior,
                    "position_scale": pos_scale,
                    "ess": float(ess),
                    "regime": current_regime,
                })

            if (i + 1) % 50 == 0:
                log(f"  Processed {i + 1}/{len(trades)} trades (ESS: {ess:.0f}, scale: {pos_scale:.2f})")

        # Final summary
        final_posterior = self.get_posterior()
        final_scale = self.get_position_scale()
        final_ess = self.effective_sample_size()

        # Determine signal confidence
        ss_std = final_posterior["signal_strength"]["std"]
        if ss_std < 0.15:
            signal_confidence = "HIGH"
        elif ss_std < 0.3:
            signal_confidence = "MODERATE"
        else:
            signal_confidence = "LOW"

        return {
            "n_particles": self.n_particles,
            "resample_threshold": self.resample_threshold,
            "n_trades_processed": len(trades),
            "final_posterior": final_posterior,
            "final_ess": float(final_ess),
            "position_scale_history": position_scale_history,
            "posterior_history": posterior_history,
            "resampling_events": self.resampling_events,
            "regime_changes_detected": regime_changes,
            "interpretation": {
                "current_regime": prev_regime or "unknown",
                "signal_confidence": signal_confidence,
                "position_scale_recommendation": final_scale,
                "half_life_estimate": f"{final_posterior['half_life']['mean']:.1f} bars (±{final_posterior['half_life']['std']:.1f})",
                "vol_scale_estimate": f"{final_posterior['vol_scale']['mean']:.2f}x (±{final_posterior['vol_scale']['std']:.2f})",
                "signal_strength_estimate": f"{final_posterior['signal_strength']['mean']:.2f} (±{final_posterior['signal_strength']['std']:.2f})",
            },
        }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Particle Filter for Regime-Adaptive Strategy Parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python validation/particle_filter.py
    python validation/particle_filter.py --particles 1000
    python validation/particle_filter.py --resample-threshold 0.3
        """,
    )
    parser.add_argument(
        "--particles", "-p", type=int, default=DEFAULT_N_PARTICLES,
        help=f"Number of particles (default: {DEFAULT_N_PARTICLES})",
    )
    parser.add_argument(
        "--resample-threshold", "-r", type=float, default=DEFAULT_RESAMPLE_THRESHOLD,
        help=f"ESS/N threshold for resampling (default: {DEFAULT_RESAMPLE_THRESHOLD})",
    )
    args = parser.parse_args()

    log("=" * 70)
    log("PARTICLE FILTER — REGIME-ADAPTIVE PARAMETER ESTIMATION")
    log(f"Strategy: {OPTIMIZED_CONFIG['name']}")
    log(f"Particles: {args.particles}")
    log(f"Resample threshold: {args.resample_threshold}")
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

    # Run backtest to get trades
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

    # Run particle filter
    log("")
    log("=" * 70)
    log("RUNNING PARTICLE FILTER")
    log("=" * 70)

    engine = ParticleFilterEngine(
        n_particles=args.particles,
        resample_threshold=args.resample_threshold,
    )
    results = engine.run_on_trades(annotated, df)

    # Print summary
    log("")
    log("=" * 70)
    log("RESULTS")
    log("=" * 70)

    posterior = results["final_posterior"]
    interp = results["interpretation"]

    log("")
    log("FINAL POSTERIOR ESTIMATES:")
    log(f"  Half-life:        {interp['half_life_estimate']}")
    log(f"  Vol Scale:        {interp['vol_scale_estimate']}")
    log(f"  Signal Strength:  {interp['signal_strength_estimate']}")

    log("")
    log("REGIME ANALYSIS:")
    log(f"  Current Regime:    {interp['current_regime']}")
    log(f"  Signal Confidence: {interp['signal_confidence']}")
    log(f"  Regime Changes:    {len(results['regime_changes_detected'])}")
    log(f"  Resampling Events: {results['resampling_events']}")

    log("")
    log("POSITION SIZING:")
    log(f"  Recommendation:    {interp['position_scale_recommendation']:.2f}x")
    log(f"  Final ESS:         {results['final_ess']:.0f}/{args.particles}")

    # Position scale statistics
    scales = results["position_scale_history"]
    if scales:
        log(f"  Scale Range:       [{min(scales):.2f}, {max(scales):.2f}]")
        log(f"  Scale Mean:        {np.mean(scales):.2f}")

    # Print regime changes
    if results["regime_changes_detected"]:
        log("")
        log("REGIME CHANGE HISTORY:")
        for rc in results["regime_changes_detected"][-10:]:  # last 10
            log(f"  Trade {rc['trade_index']}: {rc['from_regime']} -> {rc['to_regime']} ({rc['entry_time']})")

    # Save results
    output = {
        "config": OPTIMIZED_CONFIG,
        "n_trades": len(trades),
        "n_particles": args.particles,
        "resample_threshold": args.resample_threshold,
        "atr_median": atr_median,
        "results": results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Simplify position_scale_history for JSON (keep every 5th + last)
    output["results"]["position_scale_history_summary"] = {
        "full_length": len(scales),
        "mean": float(np.mean(scales)) if scales else 0,
        "std": float(np.std(scales)) if scales else 0,
        "min": float(min(scales)) if scales else 0,
        "max": float(max(scales)) if scales else 0,
        "p5": float(np.percentile(scales, 5)) if scales else 0,
        "p50": float(np.percentile(scales, 50)) if scales else 0,
        "p95": float(np.percentile(scales, 95)) if scales else 0,
    }

    results_file = os.path.join(results_dir, "particle_filter_results.json")
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    log("")
    log(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
