#!/usr/bin/env python3
"""
Solution 1: Cron-Based Polling Telegram Signal Bot

Simple approach - runs periodically via cron, checks for new signals,
sends to Telegram. No external dependencies beyond requests.

Setup:
1. Create bot via @BotFather, get token
2. Get your chat_id (message @userinfobot or @get_id_bot)
3. Set environment variables:
   export TELEGRAM_BOT_TOKEN="your_token"
   export TELEGRAM_CHAT_ID="your_chat_id"
4. Add to crontab: 0 * * * * /path/to/python /path/to/solution1_cron_polling.py

Cost: FREE (Telegram API is free, runs on your machine)
"""

import os
import sys
import json
import requests
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_backtest import (
    load_ohlc_data, load_binance_data, calculate_indicators,
    merge_binance_data, calculate_enhanced_signals, run_enhanced_backtest
)

# Configuration
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
STATE_FILE = Path(__file__).parent / 'signal_state.json'


def send_telegram_message(message: str, parse_mode: str = 'HTML') -> bool:
    """Send message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Error: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': parse_mode
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")
        return False


def load_state() -> dict:
    """Load previous state"""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {'last_trade_time': None, 'in_position': False, 'position_details': None}


def save_state(state: dict):
    """Save current state"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def format_entry_signal(trade: dict, positioning_score: float) -> str:
    """Format entry signal message"""
    side_emoji = "🟢" if trade['side'] == 'long' else "🔴"

    return f"""
{side_emoji} <b>NEW POSITION ENTRY</b> {side_emoji}

<b>Direction:</b> {trade['side'].upper()}
<b>Entry Price:</b> ${trade['entry_price']:,.2f}
<b>Stop Loss:</b> ${trade['stop']:,.2f}
<b>Take Profit:</b> ${trade['target']:,.2f}
<b>Entry Type:</b> {trade['entry_type']}

<b>Positioning Score:</b> {positioning_score:.2f}
<b>Time:</b> {trade['entry_time']}

⚠️ Risk: 5% of capital per trade
"""


def format_exit_signal(trade: dict) -> str:
    """Format exit signal message"""
    pnl = trade.get('pnl', 0)
    pnl_emoji = "✅" if pnl > 0 else "❌"

    return f"""
{pnl_emoji} <b>POSITION CLOSED</b> {pnl_emoji}

<b>Direction:</b> {trade['side'].upper()}
<b>Entry Price:</b> ${trade['entry_price']:,.2f}
<b>Exit Price:</b> ${trade['exit_price']:,.2f}
<b>Exit Reason:</b> {trade['exit_reason']}

<b>PnL:</b> ${pnl:,.2f}
<b>Time:</b> {trade['exit_time']}
"""


def check_and_send_signals():
    """Main function - check for new signals and send to Telegram"""
    print(f"[{datetime.now(timezone.utc)}] Checking for signals...")

    # Load previous state
    state = load_state()

    # Run backtest to get current position
    df_raw = load_ohlc_data()
    df = calculate_indicators(df_raw)
    atr_med = df['atr20'].median()

    binance_df = load_binance_data()
    if binance_df is not None:
        df = merge_binance_data(df, binance_df)

    df = calculate_enhanced_signals(df)

    # Run backtest
    trade_log_df, metrics, live_position, _, _ = run_enhanced_backtest(
        df, atr_med, use_enhanced_signals=True
    )

    # Check for new trades
    if len(trade_log_df) > 0:
        latest_trade = trade_log_df.iloc[-1].to_dict()
        latest_trade_time = latest_trade.get('exit_time') or latest_trade.get('entry_time')

        # Check if this is a new closed trade
        if state['last_trade_time'] != latest_trade_time and 'exit_time' in latest_trade:
            print(f"New closed trade detected: {latest_trade_time}")
            message = format_exit_signal(latest_trade)
            if send_telegram_message(message):
                state['last_trade_time'] = latest_trade_time
                state['in_position'] = False
                state['position_details'] = None

    # Check for new open position
    if live_position:
        position_key = f"{live_position['entry_time']}_{live_position['position']}"

        if state.get('position_key') != position_key:
            print(f"New position opened: {position_key}")
            message = format_entry_signal({
                'side': live_position['position'],
                'entry_price': live_position['entry_price'],
                'stop': live_position['stop_price'],
                'target': live_position['tp_price'],
                'entry_type': live_position['entry_type'],
                'entry_time': live_position['entry_time']
            }, live_position['positioning_score'])

            if send_telegram_message(message):
                state['in_position'] = True
                state['position_key'] = position_key
                state['position_details'] = live_position

    # Save state
    save_state(state)
    print(f"[{datetime.now(timezone.utc)}] Check complete")


if __name__ == "__main__":
    check_and_send_signals()
