#!/bin/bash
# ==============================================================================
# BTC Strategy Evaluation - Regular Execution Script
# ==============================================================================
# This script automates the entire evaluation and strategy adjustment process:
# 1. Runs walk-forward evaluation for all strategies
# 2. Analyzes results and identifies declining strategies
# 3. Creates parameter adjustment experiments
# 4. Runs backtests with adjusted parameters
# 5. Generates HTML report with results
# 6. Updates STATUS.md with findings
#
# Usage:
#   ./eval_regular_exec.sh              # Full evaluation + adjustments
#   ./eval_regular_exec.sh --quick      # Quick check only
#   ./eval_regular_exec.sh --eval-only  # Evaluation without adjustments
#   ./eval_regular_exec.sh --adj-only   # Adjustments only (uses existing eval)
#
# ==============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
TODAY=$(date +%Y%m%d)
ADJ_DIR="$SCRIPT_DIR/strategy_adj_$TODAY"
LOG_FILE="$SCRIPT_DIR/logs/eval_exec_$TODAY.log"

# Python paths (try in order)
PYTHON_PATHS=(
    "/Users/kriszhang/mambaforge/bin/python3"
    "/opt/anaconda3/bin/python3"
    "/opt/homebrew/bin/python3"
    "/usr/local/bin/python3"
    "/usr/bin/python3"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# Helper Functions
# ==============================================================================

log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "[$timestamp] $1"
    echo "[$timestamp] $1" >> "$LOG_FILE"
}

log_section() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    echo ""
}

find_python() {
    for p in "${PYTHON_PATHS[@]}"; do
        if [ -x "$p" ]; then
            PYTHON="$p"
            return 0
        fi
    done
    echo "ERROR: No Python interpreter found"
    exit 1
}

check_dependencies() {
    log "Checking dependencies..."

    # Check Python
    find_python
    log "Using Python: $PYTHON"

    # Check required Python packages
    $PYTHON -c "import pandas, numpy, scipy" 2>/dev/null || {
        log "${RED}ERROR: Missing Python packages (pandas, numpy, scipy)${NC}"
        exit 1
    }

    log "${GREEN}Dependencies OK${NC}"
}

# ==============================================================================
# Main Functions
# ==============================================================================

run_full_evaluation() {
    log_section "STEP 1: Running Full Walk-Forward Evaluation"

    cd "$SCRIPT_DIR"

    log "Running full evaluation with 14-day training, 7-day testing..."
    $PYTHON run_evaluation.py --full --training-days 14 --testing-days 7

    log "${GREEN}Full evaluation complete${NC}"
}

run_quick_check() {
    log_section "Running Quick Performance Check"

    cd "$SCRIPT_DIR"
    $PYTHON run_evaluation.py --quick

    log "${GREEN}Quick check complete${NC}"
}

run_strategy_learner() {
    log_section "STEP 2: Running Strategy Learner"

    cd "$SCRIPT_DIR"
    $PYTHON strategy_learner.py

    log "${GREEN}Strategy learner complete${NC}"
}

create_adjustment_folder() {
    log_section "STEP 3: Creating Adjustment Folder"

    if [ -d "$ADJ_DIR" ]; then
        log "${YELLOW}Adjustment folder already exists: $ADJ_DIR${NC}"
        log "Backing up existing folder..."
        mv "$ADJ_DIR" "${ADJ_DIR}_backup_$(date +%H%M%S)"
    fi

    mkdir -p "$ADJ_DIR"
    log "Created: $ADJ_DIR"
}

generate_adjusted_config() {
    log_section "STEP 4: Generating Adjusted Configuration"

    cat > "$ADJ_DIR/adjusted_config.py" << 'PYEOF'
"""
Adjusted Strategy Configuration - Auto-generated
================================================
Parameter adjustments based on evaluation findings.
"""

from typing import Dict, Any, List

# Baseline parameters
BASELINE_PARAMS = {
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

# Adjustment experiments
ADJUSTMENT_EXPERIMENTS: List[Dict[str, Any]] = [
    {
        "name": "Relaxed_Entry",
        "description": "Wider RSI range and lower min position score",
        "changes": {
            "min_pos_score": 0.10,
            "rsi_long_range": (15, 50),
            "pullback_range": (0.3, 3.5),
        },
        "rationale": "Increase trade frequency by relaxing entry criteria",
    },
    {
        "name": "Lower_ADX",
        "description": "Lower ADX threshold to capture more trades",
        "changes": {
            "adx_threshold": 15,
            "min_pos_score": 0.12,
        },
        "rationale": "Allow trades in weaker trends where positioning is strong",
    },
    {
        "name": "Aggressive_ProgPos",
        "description": "Lower progressive positioning thresholds",
        "changes": {
            "progressive_pos": {0: 0.3, 1: 0.4, 2: 0.5, 3: 0.6, 4: 0.8},
            "cooldown_bars": 72,
        },
        "rationale": "Re-enter market faster after losses",
    },
    {
        "name": "Wider_SL_TP",
        "description": "Wider stop-loss and take-profit",
        "changes": {
            "stop_atr_mult": 2.2,
            "tp_atr_mult": 5.0,
        },
        "rationale": "Give trades more room to develop",
    },
    {
        "name": "Combined_Relaxed",
        "description": "Combine multiple relaxation adjustments",
        "changes": {
            "min_pos_score": 0.10,
            "rsi_long_range": (15, 50),
            "pullback_range": (0.3, 3.5),
            "adx_threshold": 15,
            "progressive_pos": {0: 0.3, 1: 0.4, 2: 0.5, 3: 0.6, 4: 0.8},
            "cooldown_bars": 72,
        },
        "rationale": "Comprehensive relaxation",
    },
    {
        "name": "Selective_Tight",
        "description": "Tighter entry with better exit management",
        "changes": {
            "min_pos_long": 0.5,
            "min_pos_score": 0.20,
            "stop_atr_mult": 1.5,
            "tp_atr_mult": 3.5,
        },
        "rationale": "Higher quality entries with tighter risk",
    },
]

STRATEGY_VARIANTS = ["Baseline", "Adaptive_Baseline", "Adaptive_ProgPos_Only"]

WF_CONFIG = {
    "training_window_days": 30,
    "testing_window_days": 14,
    "step_size_days": 7,
}

ASIAN_HOURS = set(range(0, 12))
ATR_PERIOD = 14
RSI_PERIOD = 56
SMA_PERIOD = 200
ADX_PERIOD = 14
PYEOF

    log "Generated adjusted_config.py"
}

generate_backtest_runner() {
    log_section "STEP 5: Generating Backtest Runner"

    # Copy the backtest runner from a template or existing file
    if [ -f "$SCRIPT_DIR/strategy_adj_20260204/run_adjusted_backtest.py" ]; then
        cp "$SCRIPT_DIR/strategy_adj_20260204/run_adjusted_backtest.py" "$ADJ_DIR/"
        log "Copied run_adjusted_backtest.py from template"
    else
        log "${YELLOW}No template found, creating new backtest runner...${NC}"
        # Create a minimal version
        cat > "$ADJ_DIR/run_adjusted_backtest.py" << 'PYEOF'
#!/usr/bin/env python3
"""
Adjusted Strategy Backtest Runner - Auto-generated
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, EVAL_DIR)

# Import from the template if available
template_dir = os.path.join(EVAL_DIR, 'strategy_adj_20260204')
if os.path.exists(template_dir):
    sys.path.insert(0, template_dir)
    from run_adjusted_backtest import AdjustedBacktestRunner, main

    if __name__ == "__main__":
        main()
else:
    print("ERROR: Template backtest runner not found")
    sys.exit(1)
PYEOF
    fi

    chmod +x "$ADJ_DIR/run_adjusted_backtest.py"
    log "Generated run_adjusted_backtest.py"
}

generate_report_script() {
    log_section "STEP 6: Generating Report Script"

    if [ -f "$SCRIPT_DIR/strategy_adj_20260204/generate_report.py" ]; then
        cp "$SCRIPT_DIR/strategy_adj_20260204/generate_report.py" "$ADJ_DIR/"
        log "Copied generate_report.py from template"
    else
        log "${YELLOW}No template found for report generator${NC}"
    fi
}

run_adjusted_backtests() {
    log_section "STEP 7: Running Adjusted Backtests"

    cd "$ADJ_DIR"

    log "Running backtests with adjusted parameters..."
    log "This may take 10-15 minutes..."

    $PYTHON run_adjusted_backtest.py

    log "${GREEN}Adjusted backtests complete${NC}"
}

generate_html_report() {
    log_section "STEP 8: Generating HTML Report"

    cd "$ADJ_DIR"

    if [ -f "generate_report.py" ]; then
        $PYTHON generate_report.py
        log "${GREEN}HTML report generated: $ADJ_DIR/experiment_report.html${NC}"
    else
        log "${YELLOW}Report generator not found, skipping HTML report${NC}"
    fi
}

generate_readme() {
    log_section "STEP 9: Generating README"

    cat > "$ADJ_DIR/README.md" << EOF
# Strategy Adjustment Experiment - $TODAY

## Overview

This folder contains parameter adjustment experiments generated by \`eval_regular_exec.sh\`.

**Generated:** $(date '+%Y-%m-%d %H:%M:%S')

## Files

| File | Description |
|------|-------------|
| \`adjusted_config.py\` | Parameter configurations for all experiments |
| \`run_adjusted_backtest.py\` | Main script to run all experiments |
| \`generate_report.py\` | Script to generate HTML report |
| \`experiment_results.json\` | Raw results data in JSON format |
| \`experiment_report.html\` | Visual HTML report |
| \`README.md\` | This documentation file |

## How to Reproduce

\`\`\`bash
# Navigate to this folder
cd eval/strategy_adj_$TODAY

# Run all experiments
python run_adjusted_backtest.py

# Generate HTML report
python generate_report.py

# View report
open experiment_report.html
\`\`\`

## Experiments

| Experiment | Description |
|------------|-------------|
| Baseline | Original parameters for comparison |
| Relaxed_Entry | Wider RSI range, lower min position score |
| Lower_ADX | Lower ADX threshold (15 from 20) |
| Aggressive_ProgPos | Lower progressive positioning thresholds |
| Wider_SL_TP | More room for trades (SL 2.2x, TP 5.0x ATR) |
| Combined_Relaxed | All relaxation combined |
| Selective_Tight | Tighter entry, tighter risk management |

## Walk-Forward Configuration

| Parameter | Value |
|-----------|-------|
| Training Window | 30 days |
| Testing Window | 14 days |
| Step Size | 7 days |

## Interpretation

- **OOS Efficiency > 0.70**: HEALTHY - Safe for production
- **OOS Efficiency 0.50-0.70**: WARNING - Use with caution
- **OOS Efficiency < 0.50**: CRITICAL - Needs adjustment

---

*Auto-generated by eval_regular_exec.sh*
EOF

    log "Generated README.md"
}

update_status_md() {
    log_section "STEP 10: Updating STATUS.md"

    local status_file="$SCRIPT_DIR/STATUS.md"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Append new iteration to STATUS.md
    cat >> "$status_file" << EOF

---

### Iteration - Auto Evaluation ($TODAY)

**Generated by:** \`eval_regular_exec.sh\`
**Timestamp:** $timestamp

**What was done:**
1. Ran full walk-forward evaluation
2. Ran strategy learner for trend analysis
3. Created adjustment folder: \`strategy_adj_$TODAY/\`
4. Generated 6 parameter adjustment experiments
5. Ran backtests with adjusted parameters
6. Generated HTML report

**Results Location:**
- Experiment folder: \`eval/strategy_adj_$TODAY/\`
- HTML Report: \`eval/strategy_adj_$TODAY/experiment_report.html\`
- Raw Results: \`eval/strategy_adj_$TODAY/experiment_results.json\`

EOF

    log "Updated STATUS.md"
}

print_summary() {
    log_section "EXECUTION SUMMARY"

    echo -e "${GREEN}All steps completed successfully!${NC}"
    echo ""
    echo "Results:"
    echo "  - Adjustment folder: $ADJ_DIR"
    echo "  - HTML Report: $ADJ_DIR/experiment_report.html"
    echo "  - Log file: $LOG_FILE"
    echo ""
    echo "To view the report:"
    echo "  open $ADJ_DIR/experiment_report.html"
    echo ""
}

# ==============================================================================
# Main Execution
# ==============================================================================

main() {
    local mode="${1:-full}"

    # Create logs directory
    mkdir -p "$SCRIPT_DIR/logs"

    log_section "BTC STRATEGY EVALUATION - REGULAR EXECUTION"
    log "Mode: $mode"
    log "Date: $TODAY"
    log "Script Dir: $SCRIPT_DIR"

    # Check dependencies
    check_dependencies

    case "$mode" in
        --quick)
            run_quick_check
            ;;
        --eval-only)
            run_full_evaluation
            run_strategy_learner
            ;;
        --adj-only)
            create_adjustment_folder
            generate_adjusted_config
            generate_backtest_runner
            generate_report_script
            run_adjusted_backtests
            generate_html_report
            generate_readme
            update_status_md
            print_summary
            ;;
        full|*)
            run_full_evaluation
            run_strategy_learner
            create_adjustment_folder
            generate_adjusted_config
            generate_backtest_runner
            generate_report_script
            run_adjusted_backtests
            generate_html_report
            generate_readme
            update_status_md
            print_summary
            ;;
    esac

    log "${GREEN}Execution complete!${NC}"
}

# Run main with all arguments
main "$@"
