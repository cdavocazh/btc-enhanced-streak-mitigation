"""
Agent Configuration
===================
Paths, API configuration, and constants shared by both OAI and LangChain agents.

API keys are loaded from agent/.env (never hardcode secrets in this file).
"""

import os
from dotenv import load_dotenv

# ============================================================
# Directory Paths
# ============================================================

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(AGENT_DIR)
EVAL_DIR = os.path.join(REPO_DIR, "eval")
DATA_DIR = os.path.join(REPO_DIR, "binance-futures-data", "data")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
LEARNINGS_DIR = os.path.join(EVAL_DIR, "learnings")

# ============================================================
# Load .env file (agent/.env)
# ============================================================

load_dotenv(os.path.join(AGENT_DIR, ".env"))

# ============================================================
# MiniMax API Configuration
# ============================================================

MINIMAX_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_MODEL = "MiniMax-M2.5"
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")

# ============================================================
# Agent Constraints
# ============================================================

# Directories the agent is allowed to write to
WRITABLE_DIRS = [
    EVAL_DIR,
    AGENT_DIR,
]

# Maximum backtest duration (minutes) to prevent runaway processes
MAX_BACKTEST_DURATION_MINUTES = 30

# OOS efficiency thresholds (from eval/config.py)
OOS_HEALTHY = 0.70
OOS_WARNING = 0.50
OOS_CRITICAL = 0.50  # Below this = critical

# ============================================================
# Strategy lists (mirror eval/config.py)
# ============================================================

TIERED_STRATEGIES = [
    "Baseline",
    "PosVol_Combined",
    "MultiTP_30",
    "VolFilter_Adaptive",
    "Conservative",
]

ADAPTIVE_STRATEGIES = [
    "Adaptive_Baseline",
    "Adaptive_Conservative",
    "Adaptive_ADX_Only",
    "Adaptive_ProgPos_Only",
]

ALL_STRATEGIES = TIERED_STRATEGIES + ADAPTIVE_STRATEGIES
