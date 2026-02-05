#!/usr/bin/env python3
"""
Generate HTML Report for Strategy Adjustment Experiments
"""

import os
import json
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_html_report():
    """Generate HTML report from experiment results."""

    # Load results
    results_file = os.path.join(SCRIPT_DIR, 'experiment_results.json')
    with open(results_file, 'r') as f:
        data = json.load(f)

    experiments = data['experiments']

    # Group by experiment
    exp_groups = {}
    for r in experiments:
        name = r['experiment_name']
        if name not in exp_groups:
            exp_groups[name] = []
        exp_groups[name].append(r)

    # Find best performers
    best_oos_eff = max(experiments, key=lambda x: x['oos_efficiency'])
    best_oos_return = max(experiments, key=lambda x: x['oos_return_pct'])

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Strategy Adjustment Report - 2026-02-04</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #00d4ff;
            margin-bottom: 10px;
            font-size: 2.2em;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .card h3 {{
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.1em;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }}
        .metric-label {{
            color: #888;
        }}
        .metric-value {{
            font-weight: bold;
        }}
        .healthy {{ color: #00ff88; }}
        .warning {{ color: #ffaa00; }}
        .critical {{ color: #ff4444; }}

        .experiment-section {{
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .experiment-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        .experiment-header h2 {{
            color: #00d4ff;
            font-size: 1.3em;
        }}
        .experiment-desc {{
            color: #888;
            font-size: 0.9em;
            margin-bottom: 15px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{
            background: rgba(0,212,255,0.1);
            color: #00d4ff;
            font-weight: 600;
        }}
        tr:hover {{
            background: rgba(255,255,255,0.03);
        }}
        .number {{
            text-align: right;
            font-family: 'SF Mono', monospace;
        }}
        .status-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .status-healthy {{
            background: rgba(0,255,136,0.2);
            color: #00ff88;
        }}
        .status-warning {{
            background: rgba(255,170,0,0.2);
            color: #ffaa00;
        }}
        .status-critical {{
            background: rgba(255,68,68,0.2);
            color: #ff4444;
        }}

        .highlight-row {{
            background: rgba(0,212,255,0.1) !important;
        }}

        .findings {{
            background: rgba(0,255,136,0.1);
            border: 1px solid rgba(0,255,136,0.3);
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .findings h3 {{
            color: #00ff88;
            margin-bottom: 15px;
        }}
        .findings ul {{
            margin-left: 20px;
        }}
        .findings li {{
            margin-bottom: 8px;
            line-height: 1.5;
        }}

        .params-table {{
            font-size: 0.9em;
            margin-top: 10px;
        }}
        .params-table td {{
            padding: 8px 12px;
        }}

        footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid rgba(255,255,255,0.1);
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>BTC Strategy Adjustment Report</h1>
        <p class="subtitle">Walk-Forward Optimization Results | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary-cards">
            <div class="card">
                <h3>Best OOS Efficiency</h3>
                <div class="metric">
                    <span class="metric-label">Experiment:</span>
                    <span class="metric-value">{best_oos_eff['experiment_name']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Strategy:</span>
                    <span class="metric-value">{best_oos_eff['strategy_name']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">OOS Efficiency:</span>
                    <span class="metric-value healthy">{best_oos_eff['oos_efficiency']:.2f}</span>
                </div>
            </div>

            <div class="card">
                <h3>Best OOS Return</h3>
                <div class="metric">
                    <span class="metric-label">Experiment:</span>
                    <span class="metric-value">{best_oos_return['experiment_name']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Strategy:</span>
                    <span class="metric-value">{best_oos_return['strategy_name']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">OOS Return:</span>
                    <span class="metric-value healthy">{best_oos_return['oos_return_pct']:.2f}%</span>
                </div>
            </div>

            <div class="card">
                <h3>Configuration</h3>
                <div class="metric">
                    <span class="metric-label">Training Window:</span>
                    <span class="metric-value">{data['wf_config']['training_window_days']} days</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Testing Window:</span>
                    <span class="metric-value">{data['wf_config']['testing_window_days']} days</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Experiments:</span>
                    <span class="metric-value">{len(experiments)}</span>
                </div>
            </div>
        </div>

        <div class="findings">
            <h3>Key Findings</h3>
            <ul>
                <li><strong>Relaxed Entry</strong> parameters significantly improved Adaptive strategies OOS efficiency from 0.48 to <strong>0.69</strong> (near HEALTHY threshold of 0.70)</li>
                <li><strong>Selective Tight</strong> parameters achieved best overall returns with OOS Return of <strong>2.83%</strong> for Baseline strategy</li>
                <li><strong>Combined Relaxed</strong> increased trade count but reduced profitability - too aggressive</li>
                <li><strong>Lower ADX</strong> improved IS returns but didn't significantly improve OOS efficiency</li>
                <li><strong>Wider SL/TP</strong> provided modest improvements across all strategies</li>
            </ul>
        </div>
'''

    # Add experiment sections
    for exp_name, results in exp_groups.items():
        # Get description from first result
        desc = ""
        for exp in [
            {"name": "Baseline", "description": "Original parameters for comparison"},
            {"name": "Relaxed_Entry", "description": "Wider RSI range and lower min position score"},
            {"name": "Lower_ADX", "description": "Lower ADX threshold to capture more trades"},
            {"name": "Aggressive_ProgPos", "description": "Lower progressive positioning thresholds"},
            {"name": "Wider_SL_TP", "description": "Wider stop-loss and take-profit for more room"},
            {"name": "Combined_Relaxed", "description": "Combine multiple relaxation adjustments"},
            {"name": "Selective_Tight", "description": "Tighter entry with better exit management"},
        ]:
            if exp["name"] == exp_name:
                desc = exp["description"]
                break

        html += f'''
        <div class="experiment-section">
            <div class="experiment-header">
                <h2>{exp_name}</h2>
            </div>
            <p class="experiment-desc">{desc}</p>

            <table>
                <thead>
                    <tr>
                        <th>Strategy</th>
                        <th class="number">IS Return</th>
                        <th class="number">OOS Return</th>
                        <th class="number">OOS Efficiency</th>
                        <th class="number">Trades</th>
                        <th class="number">Win Rate</th>
                        <th class="number">Max DD</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
'''
        for r in results:
            status_class = f"status-{r['status'].lower()}"
            is_best = r == best_oos_eff or r == best_oos_return
            row_class = "highlight-row" if is_best else ""

            html += f'''
                    <tr class="{row_class}">
                        <td>{r['strategy_name']}</td>
                        <td class="number">{r['is_return_pct']:.2f}%</td>
                        <td class="number">{r['oos_return_pct']:.2f}%</td>
                        <td class="number">{r['oos_efficiency']:.2f}</td>
                        <td class="number">{r['total_trades']}</td>
                        <td class="number">{r['win_rate_pct']:.1f}%</td>
                        <td class="number">{r['max_drawdown_pct']:.1f}%</td>
                        <td><span class="status-badge {status_class}">{r['status']}</span></td>
                    </tr>
'''

        html += '''
                </tbody>
            </table>
        </div>
'''

    # Add comparison table
    html += '''
        <div class="experiment-section">
            <h2>Improvement vs Baseline</h2>
            <p class="experiment-desc">Comparison of all experiments against baseline parameters</p>
            <table>
                <thead>
                    <tr>
                        <th>Experiment</th>
                        <th class="number">Avg OOS Eff</th>
                        <th class="number">Baseline OOS Eff</th>
                        <th class="number">Improvement</th>
                        <th>Recommendation</th>
                    </tr>
                </thead>
                <tbody>
'''

    # Calculate averages
    baseline_avg = sum(r['oos_efficiency'] for r in exp_groups['Baseline']) / len(exp_groups['Baseline'])

    for exp_name, results in exp_groups.items():
        if exp_name == 'Baseline':
            continue

        avg_eff = sum(r['oos_efficiency'] for r in results) / len(results)
        improvement = (avg_eff - baseline_avg) / baseline_avg * 100 if baseline_avg != 0 else 0

        if improvement > 20:
            rec = "Strongly Recommended"
            rec_class = "healthy"
        elif improvement > 5:
            rec = "Recommended"
            rec_class = "warning"
        elif improvement > -5:
            rec = "Neutral"
            rec_class = ""
        else:
            rec = "Not Recommended"
            rec_class = "critical"

        html += f'''
                    <tr>
                        <td>{exp_name}</td>
                        <td class="number">{avg_eff:.2f}</td>
                        <td class="number">{baseline_avg:.2f}</td>
                        <td class="number {rec_class}">{improvement:+.1f}%</td>
                        <td class="{rec_class}">{rec}</td>
                    </tr>
'''

    html += '''
                </tbody>
            </table>
        </div>

        <footer>
            <p>BTC Enhanced Streak Mitigation Strategy Evaluation Framework</p>
            <p>Walk-Forward Configuration: 30-day training, 14-day testing, 7-day step</p>
        </footer>
    </div>
</body>
</html>
'''

    # Save HTML
    output_file = os.path.join(SCRIPT_DIR, 'experiment_report.html')
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"Report generated: {output_file}")
    return output_file


if __name__ == "__main__":
    generate_html_report()
