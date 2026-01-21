#!/usr/bin/env python3
"""
Solution 3: Cloud Serverless Architecture (AWS Lambda / Google Cloud Functions)

Serverless architecture using cloud functions triggered by:
- CloudWatch Events (AWS) / Cloud Scheduler (GCP) for periodic checks
- Or triggered by external price alert services

This solution provides:
- No server management
- Auto-scaling
- Pay-per-execution pricing
- High availability

Setup (AWS):
1. Create Lambda function with this code
2. Set environment variables in Lambda console
3. Create CloudWatch Events rule to trigger every hour
4. Estimated cost: ~$1-5/month for hourly execution

Setup (GCP):
1. Create Cloud Function
2. Set environment variables
3. Create Cloud Scheduler job
4. Estimated cost: ~$0-2/month (free tier covers most usage)
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import urllib.request
import urllib.parse

# For local testing, these would be imported
# In Lambda/Cloud Function, use layers or package dependencies
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration from environment
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
S3_BUCKET = os.environ.get('S3_BUCKET', 'btc-signals-state')  # For AWS
GCS_BUCKET = os.environ.get('GCS_BUCKET', 'btc-signals-state')  # For GCP


class TelegramClient:
    """Simple Telegram client using urllib (no external dependencies)"""

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id

    def send_message(self, text: str, parse_mode: str = 'HTML') -> bool:
        """Send message using urllib (works in Lambda without requests)"""
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        data = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': parse_mode
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False


class StateManager:
    """Manage state in cloud storage (S3/GCS)"""

    def __init__(self, use_aws: bool = True):
        self.use_aws = use_aws

    def load_state(self) -> dict:
        """Load state from cloud storage"""
        try:
            if self.use_aws:
                import boto3
                s3 = boto3.client('s3')
                response = s3.get_object(Bucket=S3_BUCKET, Key='signal_state.json')
                return json.loads(response['Body'].read().decode('utf-8'))
            else:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(GCS_BUCKET)
                blob = bucket.blob('signal_state.json')
                return json.loads(blob.download_as_string())
        except Exception as e:
            logger.warning(f"Could not load state: {e}")
            return {'last_trade_id': None, 'position': None}

    def save_state(self, state: dict):
        """Save state to cloud storage"""
        try:
            if self.use_aws:
                import boto3
                s3 = boto3.client('s3')
                s3.put_object(
                    Bucket=S3_BUCKET,
                    Key='signal_state.json',
                    Body=json.dumps(state).encode('utf-8')
                )
            else:
                from google.cloud import storage
                client = storage.Client()
                bucket = client.bucket(GCS_BUCKET)
                blob = bucket.blob('signal_state.json')
                blob.upload_from_string(json.dumps(state))
        except Exception as e:
            logger.error(f"Could not save state: {e}")


def fetch_binance_data() -> dict:
    """Fetch current market data from Binance API"""
    # Klines (candlestick) data
    klines_url = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=200"

    # Positioning data
    position_url = "https://fapi.binance.com/futures/data/topLongShortPositionRatio?symbol=BTCUSDT&period=1h&limit=1"
    account_url = "https://fapi.binance.com/futures/data/topLongShortAccountRatio?symbol=BTCUSDT&period=1h&limit=1"
    global_url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=1h&limit=1"
    funding_url = "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1"

    data = {}

    try:
        # Fetch klines
        with urllib.request.urlopen(klines_url, timeout=10) as response:
            klines = json.loads(response.read().decode('utf-8'))
            data['klines'] = klines

        # Fetch positioning
        for name, url in [('position', position_url), ('account', account_url),
                          ('global', global_url), ('funding', funding_url)]:
            try:
                with urllib.request.urlopen(url, timeout=10) as response:
                    data[name] = json.loads(response.read().decode('utf-8'))
            except:
                data[name] = None

    except Exception as e:
        logger.error(f"Failed to fetch Binance data: {e}")

    return data


def calculate_indicators(klines: list) -> dict:
    """Calculate technical indicators from klines"""
    if not klines or len(klines) < 50:
        return {}

    # Extract OHLCV
    closes = [float(k[4]) for k in klines]
    highs = [float(k[2]) for k in klines]
    lows = [float(k[3]) for k in klines]
    volumes = [float(k[5]) for k in klines]

    # Current values
    close = closes[-1]
    high = highs[-1]
    low = lows[-1]

    # SMA20
    sma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else close

    # Bollinger Bands
    if len(closes) >= 20:
        std20 = (sum((c - sma20) ** 2 for c in closes[-20:]) / 20) ** 0.5
        upper_band = sma20 + 2 * std20
        lower_band = sma20 - 2 * std20
    else:
        upper_band = close * 1.02
        lower_band = close * 0.98

    # RSI (14 period)
    if len(closes) >= 15:
        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [c if c > 0 else 0 for c in changes[-14:]]
        losses = [-c if c < 0 else 0 for c in changes[-14:]]
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 100
    else:
        rsi = 50

    # ATR (14 period)
    if len(closes) >= 15:
        trs = []
        for i in range(1, min(15, len(closes))):
            tr = max(
                highs[-i] - lows[-i],
                abs(highs[-i] - closes[-i-1]),
                abs(lows[-i] - closes[-i-1])
            )
            trs.append(tr)
        atr = sum(trs) / len(trs)
    else:
        atr = close * 0.01

    # 3-hour high/low
    high_3h = max(highs[-4:-1]) if len(highs) >= 4 else high
    low_3h = min(lows[-4:-1]) if len(lows) >= 4 else low

    return {
        'close': close,
        'high': high,
        'low': low,
        'sma20': sma20,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'rsi': rsi,
        'atr': atr,
        'high_3h': high_3h,
        'low_3h': low_3h,
        'timestamp': klines[-1][0]
    }


def calculate_positioning_score(binance_data: dict) -> float:
    """Calculate TopTraderFocused positioning score"""
    score = 0.0

    # Top trader position
    if binance_data.get('position') and len(binance_data['position']) > 0:
        pos = binance_data['position'][0]
        long_pct = float(pos.get('longAccount', 0.5))
        short_pct = float(pos.get('shortAccount', 0.5))

        if long_pct > 0.60:
            score += 1.5
        elif long_pct > 0.55:
            score += 1.0
        if short_pct > 0.60:
            score -= 1.5
        elif short_pct > 0.55:
            score -= 1.0

    # Global L/S ratio
    if binance_data.get('global') and len(binance_data['global']) > 0:
        ratio = float(binance_data['global'][0].get('longShortRatio', 1.0))
        if ratio < 0.7:
            score += 0.5
        elif ratio > 1.5:
            score -= 0.5

    # Funding rate
    if binance_data.get('funding') and len(binance_data['funding']) > 0:
        rate = float(binance_data['funding'][0].get('fundingRate', 0))
        if rate > 0.0005:
            score -= 0.5
        elif rate < -0.0005:
            score += 0.5

    return score


def generate_signals(indicators: dict, positioning_score: float, state: dict) -> Optional[dict]:
    """Generate trading signals"""
    if not indicators:
        return None

    close = indicators['close']
    rsi = indicators['rsi']
    atr = indicators['atr']

    # Get current hour (UTC)
    hr = datetime.now(timezone.utc).hour

    # Check if we're in Asian hours
    if hr not in range(0, 12):  # Asian hours (simplified)
        return None

    # Skip neutral positioning
    if abs(positioning_score) < 0.25:
        return None

    # Base signals
    long_mr = (close < indicators['lower_band']) and (rsi < 30)
    short_mr = (close > indicators['upper_band']) and (rsi > 70)
    long_bo = (close > indicators['high_3h']) and (rsi > 60)
    short_bo = (close < indicators['low_3h']) and (rsi < 40)

    # Enhanced with positioning
    long_mr_ok = long_mr and positioning_score > -1.5
    short_mr_ok = short_mr and positioning_score < 1.5
    long_bo_ok = long_bo and positioning_score > -1.0
    short_bo_ok = short_bo and positioning_score < 1.0

    any_long = long_mr_ok or long_bo_ok
    any_short = short_mr_ok or short_bo_ok

    if not (any_long or any_short):
        return None

    # Generate signal
    side = "long" if any_long else "short"
    stop_pct = 0.005
    atr_mult = 3.0

    stop_price = close * (1 - stop_pct) if side == "long" else close * (1 + stop_pct)
    target_price = close + atr_mult * atr if side == "long" else close - atr_mult * atr

    entry_type = "mr" if (long_mr_ok or short_mr_ok) else "bo"

    return {
        'type': 'entry',
        'side': side,
        'entry_price': close,
        'stop_price': stop_price,
        'target_price': target_price,
        'entry_type': f"{entry_type}_{side}",
        'positioning_score': positioning_score,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }


def format_signal_message(signal: dict) -> str:
    """Format signal for Telegram"""
    if signal['type'] == 'entry':
        side_emoji = "🟢" if signal['side'] == 'long' else "🔴"
        return f"""
{side_emoji} <b>NEW SIGNAL</b> {side_emoji}

<b>Direction:</b> {signal['side'].upper()}
<b>Entry Price:</b> ${signal['entry_price']:,.2f}
<b>Stop Loss:</b> ${signal['stop_price']:,.2f}
<b>Take Profit:</b> ${signal['target_price']:,.2f}
<b>Entry Type:</b> {signal['entry_type']}

<b>Positioning Score:</b> {signal['positioning_score']:.2f}
<b>Time:</b> {signal['timestamp']}

⚠️ Risk: 5% of capital per trade
"""
    return ""


# ============================================================================
# AWS Lambda Handler
# ============================================================================
def lambda_handler(event, context):
    """AWS Lambda entry point"""
    logger.info("Lambda triggered")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("Missing Telegram credentials")
        return {'statusCode': 500, 'body': 'Missing credentials'}

    telegram = TelegramClient(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    state_mgr = StateManager(use_aws=True)

    # Load state
    state = state_mgr.load_state()

    # Fetch market data
    binance_data = fetch_binance_data()
    if not binance_data.get('klines'):
        return {'statusCode': 500, 'body': 'Failed to fetch data'}

    # Calculate indicators
    indicators = calculate_indicators(binance_data['klines'])
    positioning_score = calculate_positioning_score(binance_data)

    # Generate signals
    signal = generate_signals(indicators, positioning_score, state)

    if signal:
        message = format_signal_message(signal)
        telegram.send_message(message)
        state['last_signal'] = signal
        state_mgr.save_state(state)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'signal_generated': signal is not None,
            'positioning_score': positioning_score
        })
    }


# ============================================================================
# Google Cloud Function Handler
# ============================================================================
def gcp_handler(request):
    """Google Cloud Function entry point"""
    logger.info("Cloud Function triggered")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return 'Missing credentials', 500

    telegram = TelegramClient(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    state_mgr = StateManager(use_aws=False)

    # Load state
    state = state_mgr.load_state()

    # Fetch market data
    binance_data = fetch_binance_data()
    if not binance_data.get('klines'):
        return 'Failed to fetch data', 500

    # Calculate indicators
    indicators = calculate_indicators(binance_data['klines'])
    positioning_score = calculate_positioning_score(binance_data)

    # Generate signals
    signal = generate_signals(indicators, positioning_score, state)

    if signal:
        message = format_signal_message(signal)
        telegram.send_message(message)
        state['last_signal'] = signal
        state_mgr.save_state(state)

    return json.dumps({
        'signal_generated': signal is not None,
        'positioning_score': positioning_score
    })


# ============================================================================
# Local Testing
# ============================================================================
if __name__ == "__main__":
    # For local testing
    print("Testing signal generation...")

    # Mock event for Lambda
    result = lambda_handler({}, None)
    print(f"Result: {result}")
