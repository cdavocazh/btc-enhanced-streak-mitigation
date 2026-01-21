#!/usr/bin/env python3
"""
Solution 2: Real-Time WebSocket Signal Bot

Connects to Binance WebSocket for real-time price updates,
calculates signals in real-time, sends to Telegram immediately.

Setup:
1. Create bot via @BotFather, get token
2. Get your chat_id
3. Set environment variables:
   export TELEGRAM_BOT_TOKEN="your_token"
   export TELEGRAM_CHAT_ID="your_chat_id"
4. Run: python solution2_realtime_websocket.py

Requirements:
    pip install websocket-client pandas numpy requests python-telegram-bot

Cost: FREE (Binance WebSocket is free, Telegram API is free)
      Requires always-on server (~$5-20/month for VPS if not local)
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import threading
import queue

import websocket
import pandas as pd
import numpy as np
import requests

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configuration
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
BINANCE_WS_URL = "wss://fstream.binance.com/ws/btcusdt@kline_1h"

# Strategy parameters (from optimized config)
INITIAL_EQ = 100000
RISK_PCT = 0.05
STOP_PCT = 0.005
ATR_MULT = 3.0
ASIA_HRS = set(range(0, 12))
US_HRS = set(range(15, 21))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Position:
    side: str
    entry_price: float
    stop_price: float
    target_price: float
    entry_time: str
    entry_type: str
    size: float
    positioning_score: float


class TelegramNotifier:
    """Handles Telegram notifications"""

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"

    def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Send message to Telegram"""
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': parse_mode
                },
                timeout=10
            )
            response.raise_for_status()
            logger.info("Telegram message sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_entry_signal(self, position: Position):
        """Send entry signal"""
        side_emoji = "🟢" if position.side == 'long' else "🔴"
        message = f"""
{side_emoji} <b>NEW POSITION ENTRY</b> {side_emoji}

<b>Direction:</b> {position.side.upper()}
<b>Entry Price:</b> ${position.entry_price:,.2f}
<b>Stop Loss:</b> ${position.stop_price:,.2f}
<b>Take Profit:</b> ${position.target_price:,.2f}
<b>Entry Type:</b> {position.entry_type}

<b>Positioning Score:</b> {position.positioning_score:.2f}
<b>Time:</b> {position.entry_time}

⚠️ Risk: 5% of capital per trade
"""
        self.send_message(message)

    def send_exit_signal(self, position: Position, exit_price: float, exit_reason: str, pnl: float):
        """Send exit signal"""
        pnl_emoji = "✅" if pnl > 0 else "❌"
        message = f"""
{pnl_emoji} <b>POSITION CLOSED</b> {pnl_emoji}

<b>Direction:</b> {position.side.upper()}
<b>Entry Price:</b> ${position.entry_price:,.2f}
<b>Exit Price:</b> ${exit_price:,.2f}
<b>Exit Reason:</b> {exit_reason}

<b>PnL:</b> ${pnl:,.2f}
<b>Time:</b> {datetime.now(timezone.utc).isoformat()}
"""
        self.send_message(message)


class RealtimeSignalEngine:
    """Real-time signal generation engine"""

    def __init__(self, notifier: TelegramNotifier):
        self.notifier = notifier
        self.candles: pd.DataFrame = pd.DataFrame()
        self.position: Optional[Position] = None
        self.consecutive_losses = 0

        # Technical indicators cache
        self.indicators = {}

        # Load historical data for indicators
        self._load_historical_data()

    def _load_historical_data(self):
        """Load historical data for indicator calculation"""
        try:
            from run_backtest import load_ohlc_data, load_binance_data, calculate_indicators, merge_binance_data

            df = load_ohlc_data()
            df = calculate_indicators(df)

            binance_df = load_binance_data()
            if binance_df is not None:
                from run_backtest import merge_binance_data, calculate_enhanced_signals
                df = merge_binance_data(df, binance_df)
                df = calculate_enhanced_signals(df)

            self.candles = df.tail(500).copy()
            self.atr_med = df['atr20'].median()
            logger.info(f"Loaded {len(self.candles)} historical candles")
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            self.candles = pd.DataFrame()
            self.atr_med = 100  # Default

    def _calculate_positioning_score(self, row) -> float:
        """Calculate TopTraderFocused positioning score"""
        score = 0.0

        # Top Trader Position
        if 'top_trader_position_long_pct' in row and pd.notna(row.get('top_trader_position_long_pct')):
            if row['top_trader_position_long_pct'] > 0.60:
                score += 1.5
            elif row['top_trader_position_long_pct'] > 0.55:
                score += 1.0

        if 'top_trader_position_short_pct' in row and pd.notna(row.get('top_trader_position_short_pct')):
            if row['top_trader_position_short_pct'] > 0.60:
                score -= 1.5
            elif row['top_trader_position_short_pct'] > 0.55:
                score -= 1.0

        # Additional factors...
        if 'global_ls_ratio' in row and pd.notna(row.get('global_ls_ratio')):
            if row['global_ls_ratio'] < 0.7:
                score += 0.5
            elif row['global_ls_ratio'] > 1.5:
                score -= 0.5

        if 'funding_rate' in row and pd.notna(row.get('funding_rate')):
            if row['funding_rate'] > 0.0005:
                score -= 0.5
            elif row['funding_rate'] < -0.0005:
                score += 0.5

        return score

    def process_candle(self, candle: dict):
        """Process new candle data"""
        timestamp = pd.Timestamp(candle['t'], unit='ms', tz='UTC')
        hr = timestamp.hour

        # Update candles DataFrame
        new_row = {
            'timestamp': timestamp,
            'open': float(candle['o']),
            'high': float(candle['h']),
            'low': float(candle['l']),
            'close': float(candle['c']),
            'volume': float(candle['v'])
        }

        # Check if candle is closed
        is_closed = candle['x']

        if not is_closed:
            return  # Only process closed candles

        logger.info(f"Processing closed candle: {timestamp}, close={new_row['close']}")

        # Get latest indicators from cached data
        if len(self.candles) == 0:
            return

        latest = self.candles.iloc[-1]

        # Calculate positioning score
        positioning_score = self._calculate_positioning_score(latest)

        # Skip if neutral (mitigation)
        if abs(positioning_score) < 0.25:
            return

        # Strict entry after losses (mitigation)
        if self.consecutive_losses >= 3 and abs(positioning_score) < 0.5:
            return

        close = new_row['close']
        atr = latest.get('atr20', 100)

        # EXIT LOGIC
        if self.position:
            exit_price = None
            exit_reason = None

            # US session exit
            if hr in US_HRS and hr not in ASIA_HRS:
                exit_price = close
                exit_reason = "us_session"
            else:
                # Stop loss / Take profit
                if self.position.side == "long":
                    if new_row['low'] <= self.position.stop_price:
                        exit_price = self.position.stop_price
                        exit_reason = "stop_loss"
                    elif new_row['high'] >= self.position.target_price:
                        exit_price = self.position.target_price
                        exit_reason = "take_profit"
                else:
                    if new_row['high'] >= self.position.stop_price:
                        exit_price = self.position.stop_price
                        exit_reason = "stop_loss"
                    elif new_row['low'] <= self.position.target_price:
                        exit_price = self.position.target_price
                        exit_reason = "take_profit"

            if exit_price:
                pnl = ((exit_price - self.position.entry_price) if self.position.side == "long"
                       else (self.position.entry_price - exit_price)) * self.position.size

                self.notifier.send_exit_signal(self.position, exit_price, exit_reason, pnl)

                if pnl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0

                self.position = None
                return

        # ENTRY LOGIC (only during Asian hours, no existing position)
        if self.position is None and hr in ASIA_HRS and atr > self.atr_med:
            # Get indicator values
            sma20 = latest.get('sma20', close)
            lower_band = latest.get('lower_band', close * 0.98)
            upper_band = latest.get('upper_band', close * 1.02)
            rsi = latest.get('rsi14', 50)
            high_3h = latest.get('high_3h', close)
            low_3h = latest.get('low_3h', close)

            # Base signals
            long_mr = (close < lower_band) and (rsi < 30)
            short_mr = (close > upper_band) and (rsi > 70)
            long_bo = (close > high_3h) and (rsi > 60)
            short_bo = (close < low_3h) and (rsi < 40)

            # Enhanced with positioning
            long_mr_ok = long_mr and positioning_score > -1.5
            short_mr_ok = short_mr and positioning_score < 1.5
            long_bo_ok = long_bo and positioning_score > -1.0
            short_bo_ok = short_bo and positioning_score < 1.0

            any_long = long_mr_ok or long_bo_ok
            any_short = short_mr_ok or short_bo_ok

            if any_long or any_short:
                side = "long" if any_long else "short"
                entry_price = close
                stop_price = entry_price * (1 - STOP_PCT) if side == "long" else entry_price * (1 + STOP_PCT)
                target_price = entry_price + ATR_MULT * atr if side == "long" else entry_price - ATR_MULT * atr

                risk_amount = INITIAL_EQ * RISK_PCT
                unit_risk = abs(entry_price - stop_price)
                size = risk_amount / unit_risk if unit_risk > 0 else 0

                # Position sizing adjustment
                if abs(positioning_score) >= 1.5:
                    size *= 1.3
                elif abs(positioning_score) >= 1.0:
                    size *= 1.2

                entry_type = "mr_long" if (long_mr_ok and any_long) else "bo_long" if any_long else "mr_short" if short_mr_ok else "bo_short"

                self.position = Position(
                    side=side,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    entry_time=str(timestamp),
                    entry_type=entry_type,
                    size=size,
                    positioning_score=positioning_score
                )

                self.notifier.send_entry_signal(self.position)


class BinanceWebSocket:
    """Binance WebSocket client"""

    def __init__(self, engine: RealtimeSignalEngine):
        self.engine = engine
        self.ws = None

    def on_message(self, ws, message):
        """Handle incoming message"""
        try:
            data = json.loads(message)
            if 'k' in data:  # Kline data
                self.engine.process_candle(data['k'])
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")

    def on_open(self, ws):
        logger.info("WebSocket connected to Binance")

    def run(self):
        """Run WebSocket connection"""
        while True:
            try:
                self.ws = websocket.WebSocketApp(
                    BINANCE_WS_URL,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                    on_open=self.on_open
                )
                self.ws.run_forever()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

            logger.info("Reconnecting in 5 seconds...")
            import time
            time.sleep(5)


def main():
    """Main entry point"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Error: Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables")
        sys.exit(1)

    logger.info("Starting Real-Time Signal Bot...")

    notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    engine = RealtimeSignalEngine(notifier)
    ws_client = BinanceWebSocket(engine)

    # Send startup notification
    notifier.send_message("🚀 <b>Signal Bot Started</b>\n\nMonitoring BTC/USDT for trading signals...")

    # Run WebSocket
    ws_client.run()


if __name__ == "__main__":
    main()
