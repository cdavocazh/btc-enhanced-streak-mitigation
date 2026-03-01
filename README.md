# BTC Enhanced Strategy with Streak Mitigation

An optimized BTC trading strategy that incorporates positioning data from Binance and implements streak mitigation techniques to reduce consecutive losing trades.

## Strategy Overview

### Optimized Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Positioning Method** | TopTraderFocused | Higher weight on top trader position signals |
| **Entry Threshold** | 1.5 | Minimum positioning score for entry |
| **Strong Positioning** | 2.0 | Threshold for strong positioning signals |
| **Skip Neutral** | 0.25 | Skip trades with \|score\| < 0.25 |
| **Strict Entry Trigger** | 3 losses | After 3 consecutive losses, require score ≥ 0.5 |

### Expected Performance

| Metric | Value | vs Baseline |
|--------|-------|-------------|
| **Return** | ~1700%+ | +7-8% |
| **Max Consecutive Losses** | ~20-22 | -3 to -5 |
| **Win Rate** | ~28.3% | similar |
| **Max Drawdown** | ~10-11% | similar |

## Key Features

### 1. TopTraderFocused Positioning Score
- Emphasizes top trader position signals (weight: 1.5x for strong, 1.0x for moderate)
- Reduced weight on account-level signals (0.25x)
- Incorporates funding rate reversals and OI confirmation

### 2. Streak Mitigation
- **Skip Neutral**: Avoids trades when positioning is unclear (|score| < 0.25)
- **StrictEntry After Losses**: After 3 consecutive losses, requires positioning score ≥ 0.5

### 3. Adaptive Position Sizing
- 30% larger on strong conviction (|score| ≥ 1.5)
- 20% larger on good conviction (|score| ≥ 1.0)
- 10% larger on moderate conviction (|score| ≥ 0.5)
- 20% smaller on low volume

## Directory Structure

```
btc-enhanced-streak-mitigation/
├── run_backtest.py              # Main backtest with optimized config
├── load_binance_data.py         # Binance data loader
├── BTC_OHLC_1h_gmt8_updated.csv # Price data
├── backtest_results/            # Backtest output
│   ├── metrics.json
│   ├── trade_log.csv
│   └── equity_curve_ts.csv
├── validation/                  # Statistical validation
│   ├── monte_carlo_validation.py      # Brute-force MC shuffle (+ antithetic variates)
│   ├── stratified_monte_carlo.py      # Stratified MC by regime/volatility/session
│   ├── particle_filter.py             # Bootstrap particle filter for regime estimation
│   ├── walk_forward_optimization.py
│   └── results/
├── agent/                       # LLM-powered evaluation agent (14 tools)
│   ├── shared/                        # Shared tools, config, prompts
│   ├── OAI/                           # OpenAI Agents SDK implementation
│   └── LangChain/                     # LangChain implementation
└── telegram_signals/            # Telegram bot solutions
    ├── solution1_cron_polling.py
    ├── solution2_realtime_websocket.py
    ├── solution3_cloud_serverless.py
    └── README.md
```

## Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy requests
```

### 2. Link Binance Data
```bash
# Create symlink to binance-futures-data repo
ln -sf /path/to/binance-futures-data ../binance-futures-data
```

### 3. Run Backtest
```bash
python run_backtest.py
```

### 4. Run Validation
```bash
# Monte Carlo validation (sequence independence)
python validation/monte_carlo_validation.py
python validation/monte_carlo_validation.py --antithetic   # ~50% variance reduction

# Stratified Monte Carlo (regime-dependent edge)
python validation/stratified_monte_carlo.py --strata all
python validation/stratified_monte_carlo.py --antithetic   # Antithetic within strata

# Particle filter (regime-adaptive parameter estimation)
python validation/particle_filter.py --particles 500

# Walk-Forward Optimization (out-of-sample testing)
python validation/walk_forward_optimization.py
```

### 5. Run Agent (LLM-powered evaluation)
```bash
export MINIMAX_API_KEY="your-key"
python agent/OAI/run.py           # OpenAI Agents SDK
python agent/LangChain/run.py     # LangChain
```

## Validation Results

### Monte Carlo Shuffle (1000 simulations)
- **Drawdown P-Value**: 0.032 (FAVORABLE)
- **Sequence Dependency**: LOW
- Strategy's max drawdown is significantly better than random shuffles
- **Antithetic variates** available for ~50% variance reduction on confidence intervals

### Stratified Monte Carlo
- Tests whether strategy edge is regime-dependent
- Strata: regime (ADX-based), volatility (ATR percentile), session (Asia/US/Other), combined
- Antithetic variates reverse PnL within each stratum independently

### Particle Filter (Bootstrap SMC)
- Online Bayesian estimation of regime parameters: half_life, vol_scale, signal_strength
- Position sizing recommendations based on posterior uncertainty
- Regime change detection (trending/ranging x high/low volatility)

### Walk-Forward Optimization (20 periods)
- **Profitable Periods**: 70%
- **Efficiency Ratio**: 0.97 (ROBUST)
- Parameters generalize well to unseen data

## Telegram Signal Bot

Three solutions for sending signals to Telegram:

| Solution | Latency | Cost | Best For |
|----------|---------|------|----------|
| Cron Polling | ~1 hour | FREE | Development/Testing |
| WebSocket | Real-time | $5-20/mo VPS | Production Trading |
| Serverless (AWS/GCP) | ~1 hour | $0-1/mo | High Reliability |

See `telegram_signals/README.md` for setup instructions.

## Configuration Reference

### Core Parameters
```python
INITIAL_EQ = 100000          # Starting capital
RISK_PCT_INIT = 0.05         # 5% risk per trade
STOP_PCT = 0.005             # 0.5% stop loss
ATR_MULT = 3.0               # 3x ATR for take profit
```

### Positioning Thresholds
```python
ENTRY_THRESHOLD = 1.5        # Min score for entry
STRONG_POSITIONING_THRESHOLD = 2.0
SKIP_NEUTRAL_THRESHOLD = 0.25
```

### Streak Mitigation
```python
STRICT_ENTRY_TRIGGER = 3     # After N consecutive losses
STRICT_ENTRY_MIN_SCORE = 0.5 # Required score after streak
```

## Data Sources

- **OHLC**: BTC hourly candles (2020-present)
- **Binance Positioning**:
  - Top trader position L/S ratio
  - Top trader account L/S ratio
  - Global trader L/S ratio
  - Funding rate
  - Open Interest

## License

MIT License
