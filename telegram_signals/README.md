# Telegram Signal Bot Solutions

Three architectural solutions for sending BTC trading signals to Telegram.

## Quick Setup (All Solutions)

### 1. Create Telegram Bot
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot` and follow instructions
3. Save the **Bot Token** (looks like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

### 2. Get Your Chat ID
1. Message [@userinfobot](https://t.me/userinfobot) or [@get_id_bot](https://t.me/get_id_bot)
2. It will reply with your **Chat ID** (a number like `123456789`)

### 3. Set Environment Variables
```bash
export TELEGRAM_BOT_TOKEN="your_bot_token_here"
export TELEGRAM_CHAT_ID="your_chat_id_here"
```

---

## Solution Comparison

| Feature | Solution 1: Cron Polling | Solution 2: WebSocket | Solution 3: Serverless |
|---------|-------------------------|----------------------|------------------------|
| **Latency** | ~1 hour (depends on cron) | Real-time (~seconds) | ~1 hour (scheduled) |
| **Complexity** | Simple | Moderate | Moderate |
| **Infrastructure** | Local machine / VPS | Always-on server | Cloud (AWS/GCP) |
| **Monthly Cost** | FREE (local) or $5-10 VPS | $5-20 VPS | $0-5 |
| **Reliability** | Low-Medium | Medium-High | High |
| **Scalability** | Low | Medium | High |
| **Maintenance** | Manual | Some monitoring | Minimal |
| **Setup Time** | 5 minutes | 15 minutes | 30 minutes |

---

## Solution 1: Cron-Based Polling

**Best for:** Simple setup, occasional signals, budget-conscious

### Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Cron Job      │────▶│  Python Script  │────▶│  Telegram API   │
│   (every 1h)    │     │  (backtest)     │     │  (HTTP POST)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Setup
```bash
# Install dependencies
pip install requests pandas numpy

# Test the script
python solution1_cron_polling.py

# Add to crontab (run every hour at minute 5)
crontab -e
# Add line: 5 * * * * cd /path/to/telegram_signals && /usr/bin/python3 solution1_cron_polling.py >> /var/log/signals.log 2>&1
```

### Costs
| Component | Cost |
|-----------|------|
| Telegram API | FREE |
| Local Machine | FREE |
| VPS (optional) | $5-10/month |
| **Total** | **$0-10/month** |

### Pros
- ✅ Simplest to set up
- ✅ No server management if running locally
- ✅ Uses existing backtest code
- ✅ Low resource usage

### Cons
- ❌ Up to 1-hour delay for signals
- ❌ Requires machine to be running
- ❌ May miss signals if machine is off
- ❌ Not real-time

---

## Solution 2: Real-Time WebSocket

**Best for:** Real-time signals, serious traders, dedicated server

### Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Binance WS     │────▶│  Signal Engine  │────▶│  Telegram API   │
│  (real-time)    │     │  (always-on)    │     │  (HTTP POST)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Setup
```bash
# Install dependencies
pip install websocket-client pandas numpy requests

# Run (use screen/tmux for persistence)
screen -S signals
python solution2_realtime_websocket.py
# Ctrl+A, D to detach

# Or use systemd service
sudo nano /etc/systemd/system/btc-signals.service
```

**Systemd Service:**
```ini
[Unit]
Description=BTC Signal Bot
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/telegram_signals
Environment=TELEGRAM_BOT_TOKEN=your_token
Environment=TELEGRAM_CHAT_ID=your_chat_id
ExecStart=/usr/bin/python3 solution2_realtime_websocket.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Costs
| Component | Cost |
|-----------|------|
| Telegram API | FREE |
| Binance WebSocket | FREE |
| VPS (DigitalOcean/Vultr) | $5-20/month |
| **Total** | **$5-20/month** |

**Recommended VPS:**
- DigitalOcean: $6/month (1 vCPU, 1GB RAM)
- Vultr: $5/month (1 vCPU, 1GB RAM)
- Hetzner: €3.29/month (cheapest reliable option)

### Pros
- ✅ Real-time signal delivery
- ✅ Immediate entry/exit notifications
- ✅ WebSocket is efficient (low bandwidth)
- ✅ Automatic reconnection

### Cons
- ❌ Requires always-on server
- ❌ Monthly VPS cost
- ❌ Needs monitoring for crashes
- ❌ More complex setup

---

## Solution 3: Cloud Serverless (AWS Lambda / GCP)

**Best for:** High reliability, scalability, minimal maintenance

### Architecture
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  CloudWatch     │────▶│  Lambda/Cloud   │────▶│  Telegram API   │
│  Scheduler      │     │  Function       │     │  (HTTP POST)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  S3/GCS         │
                        │  (state store)  │
                        └─────────────────┘
```

### AWS Lambda Setup

1. **Create Lambda Function:**
```bash
# Package the code
cd telegram_signals
zip -r lambda_package.zip solution3_cloud_serverless.py

# Create function via AWS Console or CLI
aws lambda create-function \
  --function-name btc-signal-bot \
  --runtime python3.9 \
  --handler solution3_cloud_serverless.lambda_handler \
  --zip-file fileb://lambda_package.zip \
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-role
```

2. **Set Environment Variables:**
```
TELEGRAM_BOT_TOKEN = your_token
TELEGRAM_CHAT_ID = your_chat_id
S3_BUCKET = btc-signals-state
```

3. **Create CloudWatch Events Rule:**
```bash
aws events put-rule \
  --name btc-signal-hourly \
  --schedule-expression "rate(1 hour)"

aws events put-targets \
  --rule btc-signal-hourly \
  --targets "Id"="1","Arn"="arn:aws:lambda:region:account:function:btc-signal-bot"
```

### Google Cloud Functions Setup

1. **Deploy Function:**
```bash
gcloud functions deploy btc-signal-bot \
  --runtime python39 \
  --trigger-http \
  --entry-point gcp_handler \
  --set-env-vars TELEGRAM_BOT_TOKEN=your_token,TELEGRAM_CHAT_ID=your_chat_id
```

2. **Create Cloud Scheduler:**
```bash
gcloud scheduler jobs create http btc-signal-hourly \
  --schedule="0 * * * *" \
  --uri="https://REGION-PROJECT.cloudfunctions.net/btc-signal-bot" \
  --http-method=GET
```

### Costs

**AWS Lambda:**
| Component | Cost |
|-----------|------|
| Lambda (1M free requests/month) | ~$0 |
| CloudWatch Events | ~$0 |
| S3 (state storage) | ~$0.01 |
| **Total** | **~$0-1/month** |

**Google Cloud:**
| Component | Cost |
|-----------|------|
| Cloud Functions (2M free/month) | ~$0 |
| Cloud Scheduler (3 free jobs) | ~$0 |
| Cloud Storage | ~$0.01 |
| **Total** | **~$0-1/month** |

### Pros
- ✅ Highest reliability (99.9%+ uptime)
- ✅ No server management
- ✅ Auto-scaling
- ✅ Very low cost
- ✅ Built-in monitoring/logging

### Cons
- ❌ Not real-time (scheduled execution)
- ❌ Cold start latency (~1-2s)
- ❌ Cloud vendor lock-in
- ❌ More complex initial setup
- ❌ Limited execution time (15 min max)

---

## Recommendation

| Use Case | Recommended Solution |
|----------|---------------------|
| **Testing/Development** | Solution 1 (Cron) |
| **Serious Trading (Real-time)** | Solution 2 (WebSocket) |
| **Production (Reliability)** | Solution 3 (Serverless) |
| **Budget-Conscious** | Solution 1 or 3 |

### My Recommendation: **Solution 2 (WebSocket) + Solution 3 (Serverless Backup)**

For production use:
1. Run WebSocket bot on VPS for real-time signals
2. Deploy serverless function as backup/verification
3. Total cost: ~$6-10/month with high reliability

---

## Signal Message Format

All solutions send messages in this format:

**Entry Signal:**
```
🟢 NEW POSITION ENTRY 🟢

Direction: LONG
Entry Price: $95,000.00
Stop Loss: $94,525.00
Take Profit: $96,500.00
Entry Type: bo_long

Positioning Score: 1.75
Time: 2026-01-21T08:00:00Z

⚠️ Risk: 5% of capital per trade
```

**Exit Signal:**
```
✅ POSITION CLOSED ✅

Direction: LONG
Entry Price: $95,000.00
Exit Price: $96,500.00
Exit Reason: take_profit

PnL: $15,000.00
Time: 2026-01-21T12:00:00Z
```

---

## Troubleshooting

### Bot not sending messages
1. Verify bot token: `curl https://api.telegram.org/bot<TOKEN>/getMe`
2. Verify chat ID: Send message to bot first, then check
3. Check logs for errors

### Missing signals
1. Verify cron is running: `crontab -l`
2. Check script logs
3. Verify environment variables

### WebSocket disconnecting
1. Check internet connection
2. Binance may rate limit - add backoff
3. Check VPS resources (memory)
