#!/usr/bin/env python3
"""
Generate HTML Reports for Asian Hours and All Hours versions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import sys

INITIAL_CAPITAL = 100000

STRATEGY_COLORS = {
    'Baseline': '#FF6B6B',
    'PosVol_Combined': '#4ECDC4',
    'MultiTP_30': '#45B7D1',
    'VolFilter_Adaptive': '#96CEB4',
    'Conservative': '#FFEAA7',
}


def load_equity_data(results_dir, strategy_name):
    """Load equity curve data for a strategy"""
    equity_file = os.path.join(results_dir, f'equity_{strategy_name}.csv')
    if not os.path.exists(equity_file):
        return None
    equity = pd.read_csv(equity_file)
    equity['timestamp'] = pd.to_datetime(equity['timestamp'])
    return equity


def load_trade_data(results_dir, strategy_name):
    """Load trade log for a strategy"""
    trades_file = os.path.join(results_dir, f'trades_{strategy_name}.csv')
    if not os.path.exists(trades_file):
        return None
    trades = pd.read_csv(trades_file)
    return trades


def calculate_whole_trade_stats(trades_df):
    """Calculate statistics using whole-trade PnL (aggregating partial exits)"""
    if trades_df is None or len(trades_df) == 0:
        return None

    # Group by entry_time to aggregate partial exits
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

    whole_trades = trades_df.groupby('entry_time').agg({
        'pnl': 'sum',
        'is_partial': 'any',
        'exit_time': 'last',
        'entry_price': 'first',
        'exit_price': 'last',
        'exit_reason': 'last',
    }).reset_index()

    total_trades = len(whole_trades)
    wins = len(whole_trades[whole_trades['pnl'] > 0])
    losses = len(whole_trades[whole_trades['pnl'] < 0])
    breakeven = len(whole_trades[whole_trades['pnl'] == 0])

    # Streak analysis
    streaks = []
    current_streak = 0
    for _, t in whole_trades.iterrows():
        if t['pnl'] > 0:
            if current_streak < 0:
                streaks.append(current_streak)
                current_streak = 1
            else:
                current_streak += 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
                current_streak = -1
            else:
                current_streak -= 1
    if current_streak != 0:
        streaks.append(current_streak)

    max_winning_streak = max([s for s in streaks if s > 0], default=0)
    max_losing_streak = abs(min([s for s in streaks if s < 0], default=0))

    winning_trades = whole_trades[whole_trades['pnl'] > 0]
    losing_trades = whole_trades[whole_trades['pnl'] < 0]

    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = abs(losing_trades['pnl'].mean()) if len(losing_trades) > 0 else 0

    return {
        'total_trades': total_trades,
        'winning_trades': wins,
        'losing_trades': losses,
        'breakeven_trades': breakeven,
        'win_rate_pct': wins / total_trades * 100 if total_trades > 0 else 0,
        'loss_rate_pct': losses / total_trades * 100 if total_trades > 0 else 0,
        'breakeven_pct': breakeven / total_trades * 100 if total_trades > 0 else 0,
        'max_winning_streak': max_winning_streak,
        'max_losing_streak': max_losing_streak,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': avg_win / avg_loss if avg_loss > 0 else float('inf'),
        'has_partial_tp': trades_df['is_partial'].any(),
    }


def generate_html_report(version_name, results_dir, all_stats, all_equity, param_results=None, filter_results=None):
    """Generate HTML report for a version"""

    hour_info = "0-11 UTC (Asian Session)" if "Asian" in version_name else "0-23 UTC (All Hours)"

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BTC Strategy Report - {version_name}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 30px;
            line-height: 1.6;
        }}
        h1 {{
            color: #4ecdc4;
            margin-bottom: 10px;
            font-size: 28px;
        }}
        .subtitle {{
            color: #888;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .section {{
            background: #16213e;
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 25px;
        }}
        .section-title {{
            color: #4ecdc4;
            font-size: 18px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{
            background: #0f3460;
            color: #4ecdc4;
            font-weight: 500;
        }}
        tr:hover {{
            background: rgba(78, 205, 196, 0.1);
        }}
        .positive {{ color: #4ecdc4; }}
        .negative {{ color: #ff6b6b; }}
        .config-box {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            background: #0f3460;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 25px;
        }}
        .config-item {{
            text-align: center;
        }}
        .config-item .label {{
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .config-item .value {{
            color: #4ecdc4;
            font-size: 18px;
            font-weight: 600;
        }}
        .chart-container {{
            background: #16213e;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .chart {{
            width: 100%;
            height: 400px;
        }}
        .chart-title {{
            color: #4ecdc4;
            font-size: 16px;
            margin-bottom: 10px;
        }}
        .legend-row {{
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
            margin-bottom: 15px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 20px;
            height: 3px;
            border-radius: 2px;
        }}
        footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 12px;
        }}
        .highlight-row {{
            background: rgba(78, 205, 196, 0.15);
        }}
    </style>
</head>
<body>
    <h1>BTC Strategy Performance Report - {version_name}</h1>
    <p class="subtitle">Entry Hours: {hour_info} | Whole-Trade PnL Calculation | Dec 2021 - Jan 2026</p>

    <div class="config-box">
        <div class="config-item">
            <div class="label">Initial Capital</div>
            <div class="value">${INITIAL_CAPITAL:,}</div>
        </div>
        <div class="config-item">
            <div class="label">Risk Per Trade</div>
            <div class="value">$6,000</div>
        </div>
        <div class="config-item">
            <div class="label">Entry Hours</div>
            <div class="value">{hour_info.split(' ')[0]}</div>
        </div>
        <div class="config-item">
            <div class="label">Data Period</div>
            <div class="value">~4 Years</div>
        </div>
    </div>
'''

    # Performance Summary Table
    html += '''
    <div class="section">
        <h2 class="section-title">Performance Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Return %</th>
                    <th>Max DD %</th>
                    <th>Sharpe</th>
                    <th>Profit Factor</th>
                    <th>Final Equity</th>
                </tr>
            </thead>
            <tbody>
'''

    # Sort by return
    sorted_stats = sorted(all_stats.items(), key=lambda x: x[1].get('total_return_pct', 0), reverse=True)
    best_return = sorted_stats[0][1].get('total_return_pct', 0) if sorted_stats else 0

    for name, stats in sorted_stats:
        ret_pct = stats.get('total_return_pct', 0)
        is_best = abs(ret_pct - best_return) < 0.1
        row_class = 'highlight-row' if is_best else ''
        ret_class = 'positive' if ret_pct > 0 else 'negative'

        html += f'''
                <tr class="{row_class}">
                    <td><strong>{name}</strong></td>
                    <td class="{ret_class}">{ret_pct:.1f}%</td>
                    <td class="negative">{stats.get('max_drawdown_pct', 0):.1f}%</td>
                    <td>{stats.get('sharpe_ratio', 0):.2f}</td>
                    <td>{stats.get('profit_factor', 0):.2f}</td>
                    <td>${stats.get('final_equity', INITIAL_CAPITAL):,.0f}</td>
                </tr>
'''

    html += '''
            </tbody>
        </table>
    </div>
'''

    # Equity Charts Section
    html += '''
    <div class="section">
        <h2 class="section-title">Equity Curves</h2>
        <div class="legend-row">
'''

    # Add legend
    for base_name in STRATEGY_COLORS.keys():
        color = STRATEGY_COLORS.get(base_name, '#ffffff')
        html += f'''
            <div class="legend-item">
                <div class="legend-color" style="background: {color};"></div>
                <span>{base_name}</span>
            </div>
'''

    html += '''
        </div>
        <div class="chart-container">
            <div id="combined-equity-chart" class="chart" style="height: 500px;"></div>
        </div>
'''

    # Individual charts
    for name in all_stats.keys():
        html += f'''
        <div class="chart-container">
            <div class="chart-title">{name}</div>
            <div id="equity-{name.replace('_', '-').replace(' ', '-')}" class="chart"></div>
        </div>
'''

    html += '''
    </div>
'''

    # Win Rate Analysis
    html += '''
    <div class="section">
        <h2 class="section-title">Win Rate Analysis (Whole-Trade PnL)</h2>
        <table>
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Total Trades</th>
                    <th>Win Rate</th>
                    <th>BE Rate</th>
                    <th>Loss Rate</th>
                    <th>Avg Win</th>
                    <th>Avg Loss</th>
                    <th>W/L Ratio</th>
                </tr>
            </thead>
            <tbody>
'''

    for name, stats in sorted_stats:
        # Load trade data to calculate whole-trade stats
        trades_df = load_trade_data(results_dir, name)
        whole_stats = calculate_whole_trade_stats(trades_df)

        if whole_stats:
            html += f'''
                <tr>
                    <td><strong>{name}</strong></td>
                    <td>{whole_stats['total_trades']}</td>
                    <td class="positive">{whole_stats['win_rate_pct']:.1f}%</td>
                    <td>{whole_stats['breakeven_pct']:.1f}%</td>
                    <td class="negative">{whole_stats['loss_rate_pct']:.1f}%</td>
                    <td class="positive">${whole_stats['avg_win']:,.0f}</td>
                    <td class="negative">${whole_stats['avg_loss']:,.0f}</td>
                    <td>{whole_stats['win_loss_ratio']:.2f}</td>
                </tr>
'''

    html += '''
            </tbody>
        </table>
    </div>
'''

    # Streak Analysis
    html += '''
    <div class="section">
        <h2 class="section-title">Streak Analysis</h2>
        <table>
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Max Winning Streak</th>
                    <th>Max Losing Streak</th>
                    <th>Total Trades</th>
                </tr>
            </thead>
            <tbody>
'''

    for name, stats in sorted_stats:
        trades_df = load_trade_data(results_dir, name)
        whole_stats = calculate_whole_trade_stats(trades_df)

        if whole_stats:
            html += f'''
                <tr>
                    <td><strong>{name}</strong></td>
                    <td class="positive">{whole_stats['max_winning_streak']}</td>
                    <td class="negative">{whole_stats['max_losing_streak']}</td>
                    <td>{whole_stats['total_trades']}</td>
                </tr>
'''

    html += '''
            </tbody>
        </table>
    </div>
'''

    # Parameter Grid Results (if available)
    if param_results:
        html += '''
    <div class="section">
        <h2 class="section-title">Parameter Optimization Results</h2>
        <table>
            <thead>
                <tr>
                    <th>ATR Period</th>
                    <th>RSI Period</th>
                    <th>SMA Period</th>
                    <th>Return %</th>
                    <th>Max DD %</th>
                    <th>Win Rate %</th>
                    <th>Trades</th>
                </tr>
            </thead>
            <tbody>
'''
        # Show top 5
        for i, result in enumerate(param_results[:5]):
            row_class = 'highlight-row' if i == 0 else ''
            html += f'''
                <tr class="{row_class}">
                    <td>{result['atr_period']}</td>
                    <td>{result['rsi_period']}</td>
                    <td>{result['sma_period']}</td>
                    <td class="positive">{result['total_return_pct']:.1f}%</td>
                    <td class="negative">{result['max_drawdown_pct']:.1f}%</td>
                    <td>{result['win_rate_pct']:.1f}%</td>
                    <td>{result['total_trades']}</td>
                </tr>
'''
        html += '''
            </tbody>
        </table>
    </div>
'''

    # Entry Filter Results (if available)
    if filter_results:
        html += '''
    <div class="section">
        <h2 class="section-title">Entry Filter Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Filter</th>
                    <th>Return %</th>
                    <th>Max DD %</th>
                    <th>Win Rate %</th>
                    <th>Trades</th>
                    <th>Max CL</th>
                </tr>
            </thead>
            <tbody>
'''
        for i, result in enumerate(filter_results[:7]):
            row_class = 'highlight-row' if i == 0 else ''
            html += f'''
                <tr class="{row_class}">
                    <td><strong>{result['filter_name']}</strong></td>
                    <td class="positive">{result['total_return_pct']:.1f}%</td>
                    <td class="negative">{result['max_drawdown_pct']:.1f}%</td>
                    <td>{result['win_rate_pct']:.1f}%</td>
                    <td>{result['total_trades']}</td>
                    <td>{result['max_losing_streak']}</td>
                </tr>
'''
        html += '''
            </tbody>
        </table>
    </div>
'''

    # Methodology notes
    html += '''
    <div class="section">
        <h2 class="section-title">Methodology Notes</h2>
        <ul style="margin-left: 20px; color: #aaa;">
            <li><strong>Whole-Trade PnL:</strong> For Multi-TP strategies, partial exit profits are combined with final exit PnL to determine if the complete trade was profitable.</li>
            <li><strong>Positioning Score:</strong> Calculated using top trader position percentages (0.55/0.60 thresholds), account ratios, global L/S ratio, and funding rate.</li>
            <li><strong>Entry Filters:</strong> RSI range, uptrend (close > SMA), pullback percentage, minimum positioning score.</li>
            <li><strong>Risk Management:</strong> Fixed $6,000 risk per trade with position sizing based on stop distance.</li>
        </ul>
    </div>
'''

    # Footer
    html += f'''
    <footer>
        <p>BTC Strategy Performance Report - {version_name} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Data Period: Dec 2021 - Jan 2026 | 15-Minute Intervals</p>
    </footer>
'''

    # JavaScript for charts
    html += '''
    <script>
'''

    # Add equity data as JavaScript
    html += '    const equityData = {\n'
    for name, equity_df in all_equity.items():
        if equity_df is not None and len(equity_df) > 0:
            # Resample to hourly for performance
            equity_df = equity_df.set_index('timestamp')
            equity_hourly = equity_df.resample('1h').last().dropna().reset_index()

            timestamps = equity_hourly['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
            equities = equity_hourly['equity'].tolist()

            html += f'        "{name}": {{\n'
            html += f'            timestamps: {json.dumps(timestamps[::4])},\n'
            html += f'            equity: {json.dumps([round(e, 2) for e in equities[::4]])},\n'
            html += f'        }},\n'
    html += '    };\n\n'

    # Strategy colors mapping
    color_mapping = {}
    for name in all_stats.keys():
        for base_name, color in STRATEGY_COLORS.items():
            if base_name in name:
                color_mapping[name] = color
                break
        if name not in color_mapping:
            color_mapping[name] = '#888888'

    html += f'    const strategyColors = {json.dumps(color_mapping)};\n\n'

    html += '''
    // Combined equity chart
    const combinedTraces = [];
    for (const [name, data] of Object.entries(equityData)) {
        if (data.timestamps && data.timestamps.length > 0) {
            combinedTraces.push({
                x: data.timestamps,
                y: data.equity,
                type: 'scatter',
                mode: 'lines',
                name: name,
                line: { color: strategyColors[name], width: 2 },
                hovertemplate: '<b>' + name + '</b><br>Time: %{x}<br>Equity: $%{y:,.0f}<extra></extra>',
            });
        }
    }

    // Add initial capital line
    if (combinedTraces.length > 0) {
        const firstTimestamp = equityData[Object.keys(equityData)[0]].timestamps[0];
        const lastTimestamp = equityData[Object.keys(equityData)[0]].timestamps.slice(-1)[0];

        combinedTraces.push({
            x: [firstTimestamp, lastTimestamp],
            y: [100000, 100000],
            type: 'scatter',
            mode: 'lines',
            name: 'Initial Capital',
            line: { color: '#666', width: 1, dash: 'dash' },
            hoverinfo: 'skip',
        });
    }

    const combinedLayout = {
        paper_bgcolor: '#16213e',
        plot_bgcolor: '#16213e',
        font: { color: '#eee' },
        title: { text: 'All Strategies Equity Comparison', font: { color: '#4ecdc4' } },
        xaxis: {
            type: 'date',
            gridcolor: '#333',
            rangeselector: {
                buttons: [
                    { count: 1, label: '1M', step: 'month', stepmode: 'backward' },
                    { count: 3, label: '3M', step: 'month', stepmode: 'backward' },
                    { count: 6, label: '6M', step: 'month', stepmode: 'backward' },
                    { count: 1, label: '1Y', step: 'year', stepmode: 'backward' },
                    { step: 'all', label: 'All' }
                ],
                bgcolor: '#0f3460',
                activecolor: '#4ecdc4',
                font: { color: '#eee' },
            },
        },
        yaxis: {
            title: 'Equity ($)',
            gridcolor: '#333',
            tickformat: '$,.0f',
        },
        legend: {
            orientation: 'h',
            y: -0.15,
            x: 0.5,
            xanchor: 'center',
        },
        margin: { t: 50, l: 80, r: 50, b: 80 },
        hovermode: 'x unified',
    };

    Plotly.newPlot('combined-equity-chart', combinedTraces, combinedLayout, { scrollZoom: true });

    // Individual equity charts
    for (const [name, data] of Object.entries(equityData)) {
        if (!data.timestamps || data.timestamps.length === 0) continue;

        const chartId = 'equity-' + name.replace(/_/g, '-').replace(/ /g, '-');

        // Calculate drawdown
        let peak = data.equity[0];
        const drawdowns = data.equity.map(eq => {
            if (eq > peak) peak = eq;
            return ((peak - eq) / peak) * 100;
        });

        const traces = [
            {
                x: data.timestamps,
                y: data.equity,
                type: 'scatter',
                mode: 'lines',
                name: 'Equity',
                line: { color: strategyColors[name], width: 2 },
                hovertemplate: 'Equity: $%{y:,.0f}<extra></extra>',
            },
            {
                x: data.timestamps,
                y: drawdowns,
                type: 'scatter',
                mode: 'lines',
                name: 'Drawdown %',
                yaxis: 'y2',
                line: { color: '#F44336', width: 1 },
                fill: 'tozeroy',
                fillcolor: 'rgba(244, 67, 54, 0.2)',
                hovertemplate: 'Drawdown: %{y:.1f}%<extra></extra>',
            },
        ];

        traces.push({
            x: [data.timestamps[0], data.timestamps.slice(-1)[0]],
            y: [100000, 100000],
            type: 'scatter',
            mode: 'lines',
            name: 'Initial Capital',
            line: { color: '#666', width: 1, dash: 'dash' },
            hoverinfo: 'skip',
        });

        const layout = {
            paper_bgcolor: '#16213e',
            plot_bgcolor: '#16213e',
            font: { color: '#eee' },
            xaxis: {
                type: 'date',
                gridcolor: '#333',
                rangeselector: {
                    buttons: [
                        { count: 1, label: '1M', step: 'month', stepmode: 'backward' },
                        { count: 3, label: '3M', step: 'month', stepmode: 'backward' },
                        { count: 1, label: '1Y', step: 'year', stepmode: 'backward' },
                        { step: 'all', label: 'All' }
                    ],
                    bgcolor: '#0f3460',
                    activecolor: '#4ecdc4',
                    font: { color: '#eee' },
                },
            },
            yaxis: {
                title: 'Equity ($)',
                gridcolor: '#333',
                tickformat: '$,.0f',
                side: 'left',
            },
            yaxis2: {
                title: 'Drawdown (%)',
                gridcolor: '#333',
                overlaying: 'y',
                side: 'right',
                range: [Math.max(...drawdowns) * 1.1, 0],
                tickformat: '.1f',
            },
            legend: {
                orientation: 'h',
                y: -0.2,
                x: 0.5,
                xanchor: 'center',
            },
            margin: { t: 20, l: 80, r: 80, b: 80 },
            hovermode: 'x unified',
        };

        Plotly.newPlot(chartId, traces, layout, { scrollZoom: true });
    }
    </script>
</body>
</html>
'''

    return html


def main():
    """Generate reports for both versions"""
    base_dir = os.path.dirname(__file__)

    versions = [
        ('Asian Hours', os.path.join(base_dir, 'results_asian_hours')),
        ('All Hours', os.path.join(base_dir, 'results_all_hours')),
    ]

    for version_name, results_dir in versions:
        print(f"\n{'='*60}")
        print(f"Generating report for: {version_name}")
        print(f"{'='*60}")

        if not os.path.exists(results_dir):
            print(f"Results directory not found: {results_dir}")
            continue

        # Load statistics
        stats_file = os.path.join(results_dir, 'strategy_statistics.json')
        if not os.path.exists(stats_file):
            print(f"Statistics file not found: {stats_file}")
            continue

        with open(stats_file, 'r') as f:
            all_stats = json.load(f)

        # Load parameter grid results if available
        param_file = os.path.join(results_dir, 'parameter_grid_results.json')
        param_results = None
        if os.path.exists(param_file):
            with open(param_file, 'r') as f:
                param_results = json.load(f)

        # Load entry filter results if available
        filter_file = os.path.join(results_dir, 'entry_filter_results.json')
        filter_results = None
        if os.path.exists(filter_file):
            with open(filter_file, 'r') as f:
                filter_results = json.load(f)

        # Load equity data
        all_equity = {}
        for name in all_stats.keys():
            equity = load_equity_data(results_dir, name)
            all_equity[name] = equity

        # Generate HTML report
        html = generate_html_report(version_name, results_dir, all_stats, all_equity, param_results, filter_results)

        # Save report
        report_file = os.path.join(results_dir, 'strategy_report.html')
        with open(report_file, 'w') as f:
            f.write(html)

        print(f"Saved report to: {report_file}")

    print("\n" + "=" * 60)
    print("Report generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
