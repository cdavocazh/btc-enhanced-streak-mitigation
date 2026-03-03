#!/usr/bin/env python3
"""
Generate HTML Report for ML Experiment Results
"""

import os
import json
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_html_report():
    """Generate comprehensive HTML report from ML experiment results."""

    results_file = os.path.join(SCRIPT_DIR, 'ml_experiment_results.json')
    with open(results_file, 'r') as f:
        data = json.load(f)

    experiments = data['experiments']
    timestamp = data.get('timestamp', datetime.now().isoformat())

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Strategy ML Optimization Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 50%, #0a1628 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1500px; margin: 0 auto; }}

        h1 {{
            text-align: center;
            color: #00d4ff;
            margin-bottom: 5px;
            font-size: 2.4em;
            text-shadow: 0 0 20px rgba(0,212,255,0.3);
        }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; font-size: 1.1em; }}

        .overview-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .overview-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            text-align: center;
            transition: transform 0.2s;
        }}
        .overview-card:hover {{ transform: translateY(-3px); }}
        .overview-card .number {{
            font-size: 2.2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .overview-card .label {{ color: #888; font-size: 0.9em; }}

        .healthy {{ color: #00ff88; }}
        .warning {{ color: #ffaa00; }}
        .critical {{ color: #ff4444; }}
        .neutral {{ color: #aaa; }}

        .section {{
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .section h2 {{
            color: #00d4ff;
            font-size: 1.5em;
            margin-bottom: 5px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .section .approach-num {{
            background: rgba(0,212,255,0.2);
            color: #00d4ff;
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.1em;
        }}
        .section-desc {{
            color: #999;
            margin-bottom: 20px;
            font-size: 0.95em;
            line-height: 1.6;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .metric-card .metric-label {{ color: #888; font-size: 0.85em; margin-bottom: 5px; }}
        .metric-card .metric-value {{ font-size: 1.5em; font-weight: bold; }}

        .feature-importance {{
            margin-top: 15px;
        }}
        .feature-bar-container {{
            display: flex;
            align-items: center;
            margin-bottom: 6px;
            gap: 10px;
        }}
        .feature-name {{
            width: 180px;
            font-size: 0.85em;
            color: #aaa;
            text-align: right;
        }}
        .feature-bar {{
            flex: 1;
            height: 20px;
            border-radius: 4px;
            position: relative;
        }}
        .feature-bar-fill {{
            height: 100%;
            border-radius: 4px;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            transition: width 0.5s ease;
        }}
        .feature-value {{
            width: 50px;
            font-size: 0.8em;
            color: #888;
        }}

        .status-badge {{
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            display: inline-block;
        }}
        .status-healthy {{ background: rgba(0,255,136,0.2); color: #00ff88; border: 1px solid rgba(0,255,136,0.3); }}
        .status-warning {{ background: rgba(255,170,0,0.2); color: #ffaa00; border: 1px solid rgba(255,170,0,0.3); }}
        .status-critical {{ background: rgba(255,68,68,0.2); color: #ff4444; border: 1px solid rgba(255,68,68,0.3); }}
        .status-error {{ background: rgba(255,68,68,0.15); color: #ff6666; border: 1px solid rgba(255,68,68,0.2); }}

        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .comparison-table th, .comparison-table td {{
            padding: 14px 18px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }}
        .comparison-table th {{
            background: rgba(0,212,255,0.08);
            color: #00d4ff;
            font-weight: 600;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .comparison-table tr:hover {{ background: rgba(255,255,255,0.03); }}
        .comparison-table .number {{ text-align: right; font-family: 'SF Mono', 'Menlo', monospace; }}
        .highlight-row {{ background: rgba(0,255,136,0.08) !important; }}

        .recommendation {{
            background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(0,255,136,0.1));
            border: 1px solid rgba(0,255,136,0.2);
            border-radius: 12px;
            padding: 25px;
            margin: 25px 0;
        }}
        .recommendation h3 {{ color: #00ff88; margin-bottom: 15px; font-size: 1.2em; }}
        .recommendation ul {{ margin-left: 20px; }}
        .recommendation li {{ margin-bottom: 10px; line-height: 1.6; }}

        .approach-design {{
            background: rgba(0,212,255,0.05);
            border: 1px solid rgba(0,212,255,0.15);
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
        }}
        .approach-design h4 {{ color: #00d4ff; margin-bottom: 10px; }}
        .approach-design p {{ color: #aaa; line-height: 1.6; }}
        .approach-design code {{
            background: rgba(255,255,255,0.1);
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'SF Mono', monospace;
            color: #00ff88;
        }}

        .integration-section {{
            background: rgba(0,212,255,0.03);
            border: 1px solid rgba(0,212,255,0.1);
            border-radius: 12px;
            padding: 25px;
            margin: 25px 0;
        }}
        .integration-section h3 {{ color: #00d4ff; margin-bottom: 15px; }}

        .regime-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }}
        .regime-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        .regime-card .regime-name {{ font-size: 0.85em; color: #aaa; margin-bottom: 5px; }}
        .regime-card .regime-value {{ font-size: 1.3em; font-weight: bold; }}

        footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
            color: #555;
        }}

        @media (max-width: 768px) {{
            .overview-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .metrics-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ML Strategy Optimization Report</h1>
        <p class="subtitle">Machine Learning Approaches for BTC Enhanced Strategy | {timestamp[:19]}</p>
'''

    # Overview cards
    n_experiments = len(experiments)
    statuses = [e.get('status', 'UNKNOWN') for e in experiments.values()]
    n_healthy = statuses.count('HEALTHY')
    n_warning = statuses.count('WARNING')
    best_improvement = 0
    best_name = "N/A"
    for key, exp in experiments.items():
        imp = exp.get('improvement_vs_baseline',
                      exp.get('improvement_ensemble',
                             exp.get('improvement_pct', 0)))
        if imp and imp > best_improvement:
            best_improvement = imp
            best_name = exp.get('name', key)

    html += f'''
        <div class="overview-grid">
            <div class="overview-card">
                <div class="number healthy">{n_experiments}</div>
                <div class="label">ML Approaches Tested</div>
            </div>
            <div class="overview-card">
                <div class="number {'healthy' if n_healthy > 0 else 'warning'}">{n_healthy}</div>
                <div class="label">Showing Improvement</div>
            </div>
            <div class="overview-card">
                <div class="number {'healthy' if best_improvement > 0 else 'critical'}">{best_improvement:+.1f}%</div>
                <div class="label">Best Improvement</div>
            </div>
            <div class="overview-card">
                <div class="number neutral">{data.get("overall_win_rate", 0):.1f}%</div>
                <div class="label">Baseline Win Rate</div>
            </div>
        </div>
'''

    # ---- APPROACH 1: XGBoost ----
    xgb = experiments.get('xgboost', {})
    xgb_status = xgb.get('status', 'ERROR')
    xgb_status_class = f"status-{xgb_status.lower()}"

    html += f'''
        <div class="section">
            <h2><span class="approach-num">1</span> XGBoost Trade Quality Classifier
                <span class="status-badge {xgb_status_class}">{xgb_status}</span></h2>
            <p class="section-desc">
                Trains an XGBoost gradient-boosted tree to predict whether each trade setup will be
                profitable before entry. Uses 29 features including positioning momentum, ADX dynamics,
                volume regime, and time-of-day encoding. Walk-forward validated to prevent look-ahead bias.
            </p>

            <div class="approach-design">
                <h4>Design</h4>
                <p>For each potential trade entry, the model outputs a win probability. Entries below the
                confidence threshold (55%) are filtered out. This replaces the static entry filters
                (RSI range, pullback range, min positioning score) with a learned, multi-dimensional filter
                that adapts to changing market conditions across walk-forward windows.</p>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Baseline Win Rate</div>
                    <div class="metric-value neutral">{xgb.get('baseline_win_rate', 0):.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Filtered Win Rate</div>
                    <div class="metric-value {'healthy' if xgb.get('filtered_win_rate', 0) > xgb.get('baseline_win_rate', 0) else 'critical'}">{xgb.get('filtered_win_rate', 0):.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Improvement</div>
                    <div class="metric-value {'healthy' if xgb.get('improvement_vs_baseline', 0) > 0 else 'critical'}">{xgb.get('improvement_vs_baseline', 0):+.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Test Accuracy</div>
                    <div class="metric-value">{xgb.get('test_accuracy', 0):.1%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">{xgb.get('test_precision', 0):.1%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Trades Filtered</div>
                    <div class="metric-value">{xgb.get('trades_filtered_pct', 0):.1f}%</div>
                </div>
            </div>
'''

    # Feature importance for XGBoost
    fi = xgb.get('feature_importance', {})
    if fi:
        max_imp = max(fi.values()) if fi.values() else 1
        html += '''<div class="feature-importance"><h4 style="color:#00d4ff;margin-bottom:10px;">Feature Importance (Top 10)</h4>'''
        for fname, fval in list(fi.items())[:10]:
            pct = fval / max_imp * 100
            html += f'''
                <div class="feature-bar-container">
                    <div class="feature-name">{fname}</div>
                    <div class="feature-bar"><div class="feature-bar-fill" style="width:{pct:.0f}%"></div></div>
                    <div class="feature-value">{fval:.3f}</div>
                </div>'''
        html += '</div>'

    html += '</div>'

    # ---- APPROACH 2: Random Forest Regime ----
    rf = experiments.get('random_forest', {})
    rf_status = rf.get('status', 'ERROR')
    rf_status_class = f"status-{rf_status.lower()}"

    html += f'''
        <div class="section">
            <h2><span class="approach-num">2</span> Random Forest Regime Detector
                <span class="status-badge {rf_status_class}">{rf_status}</span></h2>
            <p class="section-desc">
                Classifies the current market into 5 regimes (Strong Trend, Moderate Trend, Ranging High Vol,
                Ranging Low Vol, Volatile Trend) and measures trade win rates per regime. Only trades in
                favorable regimes, replacing the static ADX threshold with a learned multi-feature classifier.
            </p>

            <div class="approach-design">
                <h4>Design</h4>
                <p>The current system uses a simple <code>ADX > 20</code> threshold. This approach learns
                regime boundaries from multiple features simultaneously (ADX, volatility ratio, DI spread,
                price momentum). Each regime has a measured win rate; trades are only taken in regimes where
                the win rate exceeds the baseline. This provides more nuanced filtering than a single threshold.</p>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Regime Accuracy</div>
                    <div class="metric-value">{rf.get('regime_accuracy', 0):.1%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Baseline Win Rate</div>
                    <div class="metric-value neutral">{rf.get('baseline_win_rate', 0):.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Regime-Filtered WR</div>
                    <div class="metric-value {'healthy' if rf.get('regime_filtered_win_rate', 0) > rf.get('baseline_win_rate', 0) else 'critical'}">{rf.get('regime_filtered_win_rate', 0):.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Best Regime</div>
                    <div class="metric-value healthy">{rf.get('best_regime', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Best Regime WR</div>
                    <div class="metric-value healthy">{rf.get('best_regime_win_rate', 0):.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Worst Regime</div>
                    <div class="metric-value critical">{rf.get('worst_regime', 'N/A')}</div>
                </div>
            </div>
'''

    # Regime distribution
    regime_dist = rf.get('regime_distribution', {})
    regime_wr = rf.get('win_rate_by_regime', {})
    if regime_dist:
        html += '<h4 style="color:#00d4ff;margin:15px 0 10px;">Regime Distribution & Win Rates</h4><div class="regime-grid">'
        for regime_name in regime_dist:
            dist_pct = regime_dist.get(regime_name, 0)
            wr = regime_wr.get(regime_name, 0)
            color_class = 'healthy' if wr > rf.get('baseline_win_rate', 0) else 'critical'
            html += f'''
                <div class="regime-card">
                    <div class="regime-name">{regime_name}</div>
                    <div class="regime-value {color_class}">{wr:.1f}%</div>
                    <div style="color:#666;font-size:0.8em;">{dist_pct:.1f}% of bars</div>
                </div>'''
        html += '</div>'

    # Feature importance
    fi = rf.get('feature_importance', {})
    if fi:
        max_imp = max(fi.values()) if fi.values() else 1
        html += '''<div class="feature-importance"><h4 style="color:#00d4ff;margin-bottom:10px;">Feature Importance (Top 10)</h4>'''
        for fname, fval in list(fi.items())[:10]:
            pct = fval / max_imp * 100
            html += f'''
                <div class="feature-bar-container">
                    <div class="feature-name">{fname}</div>
                    <div class="feature-bar"><div class="feature-bar-fill" style="width:{pct:.0f}%"></div></div>
                    <div class="feature-value">{fval:.3f}</div>
                </div>'''
        html += '</div>'

    html += '</div>'

    # ---- APPROACH 3: LightGBM Streak ----
    lgb_data = experiments.get('lightgbm', {})
    lgb_status = lgb_data.get('status', 'ERROR')
    lgb_status_class = f"status-{lgb_status.lower()}"

    html += f'''
        <div class="section">
            <h2><span class="approach-num">3</span> LightGBM Streak Predictor
                <span class="status-badge {lgb_status_class}">{lgb_status}</span></h2>
            <p class="section-desc">
                Predicts the probability that the next trade will be a loss, enabling dynamic position sizing.
                Replaces the static streak mitigation rules (40% reduction after 3 losses, etc.) with a
                continuously-updated probability model that adjusts position size before entering a trade.
            </p>

            <div class="approach-design">
                <h4>Design</h4>
                <p>Current system waits for 3 consecutive losses before reducing risk. This reactive approach
                means the first 3 losses in each streak happen at full size. The ML model predicts loss
                probability <em>before</em> each trade, scaling position size by <code>max(0.2, 1 - P(loss) * 0.8)</code>.
                High-risk setups get smaller positions proactively, reducing both drawdown and streak severity.</p>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Prediction Accuracy</div>
                    <div class="metric-value">{lgb_data.get('prediction_accuracy', 0):.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Static Sizing Return</div>
                    <div class="metric-value neutral">{lgb_data.get('static_sizing_return', 0):.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Dynamic Sizing Return</div>
                    <div class="metric-value {'healthy' if lgb_data.get('dynamic_sizing_return', 0) > lgb_data.get('static_sizing_return', 0) else 'critical'}">{lgb_data.get('dynamic_sizing_return', 0):.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Return Improvement</div>
                    <div class="metric-value {'healthy' if lgb_data.get('improvement_pct', 0) > 0 else 'critical'}">{lgb_data.get('improvement_pct', 0):+.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Static Max DD</div>
                    <div class="metric-value critical">{lgb_data.get('max_dd_static', 0):.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Dynamic Max DD</div>
                    <div class="metric-value {'healthy' if lgb_data.get('max_dd_dynamic', 0) < lgb_data.get('max_dd_static', 0) else 'critical'}">{lgb_data.get('max_dd_dynamic', 0):.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">DD Improvement</div>
                    <div class="metric-value {'healthy' if lgb_data.get('dd_improvement_pct', 0) > 0 else 'critical'}">{lgb_data.get('dd_improvement_pct', 0):+.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Position Scale</div>
                    <div class="metric-value">{lgb_data.get('avg_position_scale', 0):.2f}x</div>
                </div>
            </div>
'''

    fi = lgb_data.get('feature_importance', {})
    if fi:
        max_imp = max(fi.values()) if fi.values() else 1
        html += '''<div class="feature-importance"><h4 style="color:#00d4ff;margin-bottom:10px;">Feature Importance (Top 10)</h4>'''
        for fname, fval in list(fi.items())[:10]:
            pct = fval / max_imp * 100
            html += f'''
                <div class="feature-bar-container">
                    <div class="feature-name">{fname}</div>
                    <div class="feature-bar"><div class="feature-bar-fill" style="width:{pct:.0f}%"></div></div>
                    <div class="feature-value">{fval:.0f}</div>
                </div>'''
        html += '</div>'

    html += '</div>'

    # ---- APPROACH 4: LSTM ----
    lstm = experiments.get('lstm', {})
    lstm_status = lstm.get('status', 'ERROR')
    lstm_status_class = f"status-{lstm_status.lower()}"

    html += f'''
        <div class="section">
            <h2><span class="approach-num">4</span> LSTM Sequential Features + XGBoost Ensemble
                <span class="status-badge {lstm_status_class}">{lstm_status}</span></h2>
            <p class="section-desc">
                Uses a PyTorch LSTM network to capture temporal patterns in the positioning and price data
                over the last 4 hours (16 bars). The LSTM's learned feature representations are concatenated
                with the original features and fed into XGBoost for final classification.
            </p>

            <div class="approach-design">
                <h4>Design</h4>
                <p>Static indicators (RSI, ADX) lose temporal context. An LSTM processes the last 16 bars
                of feature data, learning patterns like "positioning was building for 3 hours then
                suddenly reversed" that static features cannot capture. The 8-dimensional LSTM output is
                combined with 29 original features, giving XGBoost both instantaneous and sequential
                information. This hybrid architecture captures what neither approach achieves alone.</p>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Baseline Win Rate</div>
                    <div class="metric-value neutral">{lstm.get('baseline_win_rate', 0):.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">LSTM-only Win Rate</div>
                    <div class="metric-value">{lstm.get('lstm_win_rate', 0):.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Ensemble Win Rate</div>
                    <div class="metric-value {'healthy' if lstm.get('ensemble_win_rate', 0) > lstm.get('baseline_win_rate', 0) else 'critical'}">{lstm.get('ensemble_win_rate', 0):.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">LSTM Improvement</div>
                    <div class="metric-value {'healthy' if lstm.get('improvement_lstm', 0) > 0 else 'critical'}">{lstm.get('improvement_lstm', 0):+.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Ensemble Improvement</div>
                    <div class="metric-value {'healthy' if lstm.get('improvement_ensemble', 0) > 0 else 'critical'}">{lstm.get('improvement_ensemble', 0):+.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">LSTM Accuracy</div>
                    <div class="metric-value">{lstm.get('lstm_accuracy', 0):.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Ensemble Accuracy</div>
                    <div class="metric-value">{lstm.get('ensemble_accuracy', 0):.1f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Final Training Loss</div>
                    <div class="metric-value">{lstm.get('training_loss_final', 0):.4f}</div>
                </div>
            </div>
        </div>
'''

    # ---- COMPARISON TABLE ----
    html += '''
        <div class="section">
            <h2>Approach Comparison</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Approach</th>
                        <th class="number">Win Rate Improvement</th>
                        <th class="number">OOS Efficiency</th>
                        <th class="number">Key Metric</th>
                        <th>Status</th>
                        <th>Integration Effort</th>
                    </tr>
                </thead>
                <tbody>
'''

    approaches = [
        {
            'name': 'XGBoost Trade Quality',
            'improvement': xgb.get('improvement_vs_baseline', 0),
            'oos': xgb.get('oos_efficiency', 0),
            'key_metric': f"Precision: {xgb.get('test_precision', 0):.1%}",
            'status': xgb.get('status', 'ERROR'),
            'effort': 'Medium',
        },
        {
            'name': 'RF Regime Detector',
            'improvement': rf.get('improvement_vs_baseline', 0),
            'oos': rf.get('oos_efficiency', 0),
            'key_metric': f"Regime Acc: {rf.get('regime_accuracy', 0):.1%}",
            'status': rf.get('status', 'ERROR'),
            'effort': 'Low',
        },
        {
            'name': 'LightGBM Streak Pred.',
            'improvement': lgb_data.get('improvement_pct', 0),
            'oos': lgb_data.get('oos_efficiency', 0),
            'key_metric': f"DD Improv: {lgb_data.get('dd_improvement_pct', 0):+.1f}%",
            'status': lgb_data.get('status', 'ERROR'),
            'effort': 'Medium',
        },
        {
            'name': 'LSTM + XGBoost',
            'improvement': lstm.get('improvement_ensemble', 0),
            'oos': lstm.get('oos_efficiency', 0),
            'key_metric': f"Ensemble Acc: {lstm.get('ensemble_accuracy', 0):.1f}%",
            'status': lstm.get('status', 'ERROR'),
            'effort': 'High',
        },
    ]

    best_approach = max(approaches, key=lambda x: x['improvement'])
    for a in approaches:
        status_class = f"status-{a['status'].lower()}"
        imp_class = 'healthy' if a['improvement'] > 0 else 'critical'
        is_best = a == best_approach
        row_class = 'highlight-row' if is_best else ''

        html += f'''
                    <tr class="{row_class}">
                        <td>{'&#11088; ' if is_best else ''}{a['name']}</td>
                        <td class="number {imp_class}">{a['improvement']:+.2f}%</td>
                        <td class="number">{a['oos']:.2f}</td>
                        <td class="number">{a['key_metric']}</td>
                        <td><span class="status-badge {status_class}">{a['status']}</span></td>
                        <td>{a['effort']}</td>
                    </tr>
'''

    html += '''
                </tbody>
            </table>
        </div>
'''

    # ---- RECOMMENDATIONS ----
    html += '''
        <div class="recommendation">
            <h3>Recommendations & Next Steps</h3>
            <ul>
'''

    sorted_approaches = sorted(approaches, key=lambda x: -x['improvement'])
    for i, a in enumerate(sorted_approaches):
        if a['improvement'] > 0:
            html += f'<li><strong>Priority {i+1}: {a["name"]}</strong> - Shows {a["improvement"]:+.2f}% win rate improvement. '
            if a['effort'] == 'Low':
                html += 'Low integration effort makes this a quick win.</li>'
            elif a['effort'] == 'Medium':
                html += 'Worth implementing in the next eval cycle.</li>'
            else:
                html += 'Higher effort but potentially strongest results with more data.</li>'

    html += '''
                <li><strong>Ensemble Strategy</strong>: Combine the top 2-3 approaches. Use RF regime detection
                as a first filter, XGBoost quality scoring as a second filter, and LightGBM for position sizing.
                This layered approach can compound improvements.</li>
                <li><strong>Periodic Retraining</strong>: Integrate ML model retraining into the existing
                <code>eval_regular_exec.sh</code> pipeline. Models should be retrained weekly as part of the
                walk-forward evaluation cycle.</li>
                <li><strong>Feature Engineering</strong>: Based on feature importance results, invest in
                engineering better versions of the top features (e.g., multi-timeframe positioning momentum,
                cross-metric correlations).</li>
                <li><strong>Overfitting Guard</strong>: Continue using walk-forward validation for all ML models.
                Track OOS efficiency over time - if it drops below 0.50, reduce model complexity or increase
                regularization.</li>
            </ul>
        </div>
'''

    # ---- INTEGRATION WITH EVAL FRAMEWORK ----
    html += '''
        <div class="integration-section">
            <h3>Integration with Eval Framework</h3>
            <p style="color:#aaa;line-height:1.8;margin-bottom:15px;">
                These ML approaches integrate with the existing periodic evaluation workflow described in
                <code>eval/README.md</code>. The recommended integration points are:
            </p>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Eval Schedule</th>
                        <th>ML Integration</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Quick Check (6h)</td>
                        <td>Run ML model inference only</td>
                        <td>Score current market conditions with trained models, check for regime changes</td>
                    </tr>
                    <tr>
                        <td>Full Evaluation (7d)</td>
                        <td>Retrain models + evaluate</td>
                        <td>Retrain all ML models on latest data, run walk-forward validation, compare to baseline</td>
                    </tr>
                    <tr>
                        <td>Adaptation Check (14d)</td>
                        <td>Model selection + hyperparameter tuning</td>
                        <td>Compare ML approaches, select best performers, adjust hyperparameters if OOS declining</td>
                    </tr>
                </tbody>
            </table>
        </div>
'''

    html += '''
        <footer>
            <p>BTC Enhanced Streak Mitigation - ML Optimization Experiments</p>
            <p>Walk-Forward Validated | No Look-Ahead Bias | All models trained on past data only</p>
        </footer>
    </div>
</body>
</html>
'''

    output_file = os.path.join(SCRIPT_DIR, 'ml_experiment_report.html')
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"Report generated: {output_file}")
    return output_file


if __name__ == "__main__":
    generate_html_report()
