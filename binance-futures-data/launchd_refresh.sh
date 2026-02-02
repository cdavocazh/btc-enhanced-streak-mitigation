#!/bin/bash
# LaunchD job script for refreshing Binance Futures data every 5 minutes
#
# Installation:
#   1. Make executable: chmod +x launchd_refresh.sh
#   2. Create logs dir: mkdir -p logs
#   3. Copy plist to ~/Library/LaunchAgents/
#   4. Load with: launchctl load ~/Library/LaunchAgents/com.btc.datarefresh.plist
#
# Unload with: launchctl unload ~/Library/LaunchAgents/com.btc.datarefresh.plist

# Capture start time for duration calculation
START_TIME=$(date +%s)

# Configuration - UPDATE THESE PATHS
SCRIPT_DIR="/Users/kris.zhang/Github/btc-enhanced-streak-mitigation/binance-futures-data"
LOG_DIR="${SCRIPT_DIR}/logs"
PYTHON_PATH="/Users/kris.zhang/mambaforge/bin/python3"

# Fallback Python paths (try in order)
PYTHON_PATHS=(
    "/opt/anaconda3/bin/python3"
    "/opt/homebrew/Caskroom/mambaforge/base/bin/python3"
    "/Users/kris.zhang/mambaforge/bin/python3"
    "/opt/homebrew/bin/python3"
    "/usr/local/bin/python3"
    "/usr/bin/python3"
)

# Find a working Python
find_python() {
    for p in "${PYTHON_PATHS[@]}"; do
        if [ -x "$p" ]; then
            PYTHON_PATH="$p"
            return 0
        fi
    done
    return 1
}

# Source shell profile to get proper PATH (needed for launchd)
if [ -f ~/.zshrc ]; then
    source ~/.zshrc 2>/dev/null
elif [ -f ~/.bash_profile ]; then
    source ~/.bash_profile 2>/dev/null
fi

# Create logs directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# Log start time
echo ""
echo "=========================================="
echo "Data Refresh Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Change to script directory
cd "${SCRIPT_DIR}" || {
    echo "ERROR: Could not change to script directory: ${SCRIPT_DIR}"
    exit 1
}

# Find Python
if ! find_python; then
    echo "ERROR: No Python interpreter found in known paths"
    exit 1
fi

echo "Using Python: ${PYTHON_PATH}"

# Check if Python is available
if ! command -v ${PYTHON_PATH} &> /dev/null; then
    echo "ERROR: Python not found at ${PYTHON_PATH}"
    exit 1
fi

# Check if refresh script exists
if [ ! -f "refresh_data.py" ]; then
    echo "ERROR: refresh_data.py not found in ${SCRIPT_DIR}"
    exit 1
fi

# Run the refresh script
${PYTHON_PATH} refresh_data.py

# Check exit status
EXIT_CODE=$?

# Calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))
DURATION_SEC=$((DURATION % 60))

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Data refresh completed successfully"
else
    echo "ERROR: Data refresh failed with exit code ${EXIT_CODE}"
fi

echo "Data Refresh Ended: $(date '+%Y-%m-%d %H:%M:%S')"
printf "Duration: %d:%02d (min:sec)\n" ${DURATION_MIN} ${DURATION_SEC}
echo "=========================================="

exit ${EXIT_CODE}
