#!/usr/bin/env python3
"""
Performance Tracker for BTC Strategies
=======================================
Records and analyzes strategy performance over time using SQLite database.

Features:
- Record performance snapshots
- Track performance degradation
- Generate alerts (INFO/WARNING/CRITICAL)
- Detect market regime changes
- Export historical data
"""

import os
import sys
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import (
    PerformanceThresholds, DEFAULT_THRESHOLDS,
    get_all_strategies
)


@dataclass
class PerformanceSnapshot:
    """A point-in-time performance snapshot."""
    timestamp: str
    strategy_name: str
    entry_filter: str
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    max_losing_streak: int
    oos_return_pct: Optional[float] = None
    oos_efficiency: Optional[float] = None
    parameter_stability: Optional[float] = None
    data_bars: int = 0
    evaluation_type: str = "quick"  # quick, full, adapt


@dataclass
class Alert:
    """Performance alert."""
    timestamp: str
    strategy_name: str
    severity: str  # INFO, WARNING, CRITICAL
    alert_type: str
    message: str
    current_value: float
    threshold_value: float
    acknowledged: bool = False


def log(msg: str):
    """Print timestamped log message."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}")


class PerformanceTracker:
    """
    Tracks strategy performance over time using SQLite.
    """

    def __init__(self, db_path: str = None, thresholds: PerformanceThresholds = None):
        if db_path is None:
            db_path = os.path.join(SCRIPT_DIR, 'performance_history.db')
        self.db_path = db_path
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Performance snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                entry_filter TEXT NOT NULL,
                total_return_pct REAL,
                max_drawdown_pct REAL,
                sharpe_ratio REAL,
                win_rate_pct REAL,
                profit_factor REAL,
                total_trades INTEGER,
                max_losing_streak INTEGER,
                oos_return_pct REAL,
                oos_efficiency REAL,
                parameter_stability REAL,
                data_bars INTEGER,
                evaluation_type TEXT
            )
        ''')

        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                severity TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT,
                current_value REAL,
                threshold_value REAL,
                acknowledged INTEGER DEFAULT 0
            )
        ''')

        # Parameter history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS parameter_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                parameter_name TEXT NOT NULL,
                parameter_value REAL,
                change_reason TEXT
            )
        ''')

        # Regime history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS regime_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                regime TEXT NOT NULL,
                confidence REAL,
                volatility_20d REAL,
                trend_strength REAL
            )
        ''')

        conn.commit()
        conn.close()

    def record_snapshot(self, snapshot: PerformanceSnapshot) -> int:
        """
        Record a performance snapshot.

        Returns the snapshot ID.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO performance_snapshots (
                timestamp, strategy_name, entry_filter,
                total_return_pct, max_drawdown_pct, sharpe_ratio,
                win_rate_pct, profit_factor, total_trades,
                max_losing_streak, oos_return_pct, oos_efficiency,
                parameter_stability, data_bars, evaluation_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            snapshot.timestamp, snapshot.strategy_name, snapshot.entry_filter,
            snapshot.total_return_pct, snapshot.max_drawdown_pct, snapshot.sharpe_ratio,
            snapshot.win_rate_pct, snapshot.profit_factor, snapshot.total_trades,
            snapshot.max_losing_streak, snapshot.oos_return_pct, snapshot.oos_efficiency,
            snapshot.parameter_stability, snapshot.data_bars, snapshot.evaluation_type
        ))

        snapshot_id = cursor.lastrowid
        conn.commit()
        conn.close()

        log(f"Recorded snapshot for {snapshot.strategy_name}: Return={snapshot.total_return_pct:.2f}%")

        # Check for performance degradation
        self.check_performance_degradation(snapshot)

        return snapshot_id

    def check_performance_degradation(self, snapshot: PerformanceSnapshot) -> List[Alert]:
        """
        Check for performance degradation and generate alerts.
        """
        alerts = []

        # Get historical baseline (last 7 days of snapshots)
        historical = self.get_historical_performance(
            snapshot.strategy_name,
            snapshot.entry_filter,
            days=7
        )

        if not historical:
            return alerts

        # Calculate historical averages
        avg_return = np.mean([h['total_return_pct'] for h in historical])
        avg_sharpe = np.mean([h['sharpe_ratio'] for h in historical])
        avg_win_rate = np.mean([h['win_rate_pct'] for h in historical])

        # Check return degradation
        if avg_return > 0:
            return_change = (snapshot.total_return_pct - avg_return) / abs(avg_return)
            if return_change < -self.thresholds.return_degradation_critical:
                alerts.append(self._create_alert(
                    snapshot.strategy_name,
                    "CRITICAL",
                    "RETURN_DEGRADATION",
                    f"Return dropped {abs(return_change)*100:.1f}% below baseline",
                    snapshot.total_return_pct,
                    avg_return * (1 - self.thresholds.return_degradation_critical)
                ))
            elif return_change < -self.thresholds.return_degradation_warning:
                alerts.append(self._create_alert(
                    snapshot.strategy_name,
                    "WARNING",
                    "RETURN_DEGRADATION",
                    f"Return dropped {abs(return_change)*100:.1f}% below baseline",
                    snapshot.total_return_pct,
                    avg_return * (1 - self.thresholds.return_degradation_warning)
                ))

        # Check max drawdown
        if snapshot.max_drawdown_pct > self.thresholds.max_drawdown_critical * 100:
            alerts.append(self._create_alert(
                snapshot.strategy_name,
                "CRITICAL",
                "MAX_DRAWDOWN",
                f"Max drawdown {snapshot.max_drawdown_pct:.1f}% exceeds critical threshold",
                snapshot.max_drawdown_pct,
                self.thresholds.max_drawdown_critical * 100
            ))
        elif snapshot.max_drawdown_pct > self.thresholds.max_drawdown_warning * 100:
            alerts.append(self._create_alert(
                snapshot.strategy_name,
                "WARNING",
                "MAX_DRAWDOWN",
                f"Max drawdown {snapshot.max_drawdown_pct:.1f}% exceeds warning threshold",
                snapshot.max_drawdown_pct,
                self.thresholds.max_drawdown_warning * 100
            ))

        # Check win rate
        if snapshot.win_rate_pct < self.thresholds.win_rate_min * 100:
            alerts.append(self._create_alert(
                snapshot.strategy_name,
                "WARNING",
                "LOW_WIN_RATE",
                f"Win rate {snapshot.win_rate_pct:.1f}% below minimum threshold",
                snapshot.win_rate_pct,
                self.thresholds.win_rate_min * 100
            ))

        # Check Sharpe ratio
        if snapshot.sharpe_ratio < self.thresholds.sharpe_min:
            alerts.append(self._create_alert(
                snapshot.strategy_name,
                "WARNING",
                "LOW_SHARPE",
                f"Sharpe ratio {snapshot.sharpe_ratio:.2f} below minimum threshold",
                snapshot.sharpe_ratio,
                self.thresholds.sharpe_min
            ))

        # Check OOS efficiency
        if snapshot.oos_efficiency is not None:
            if snapshot.oos_efficiency < self.thresholds.out_of_sample_efficiency:
                alerts.append(self._create_alert(
                    snapshot.strategy_name,
                    "WARNING",
                    "LOW_OOS_EFFICIENCY",
                    f"OOS efficiency {snapshot.oos_efficiency:.2f} below threshold",
                    snapshot.oos_efficiency,
                    self.thresholds.out_of_sample_efficiency
                ))

        # Check losing streak
        if snapshot.max_losing_streak > self.thresholds.max_losing_streak:
            alerts.append(self._create_alert(
                snapshot.strategy_name,
                "WARNING",
                "HIGH_LOSING_STREAK",
                f"Max losing streak {snapshot.max_losing_streak} exceeds threshold",
                snapshot.max_losing_streak,
                self.thresholds.max_losing_streak
            ))

        # Record alerts
        for alert in alerts:
            self._record_alert(alert)

        return alerts

    def _create_alert(
        self,
        strategy_name: str,
        severity: str,
        alert_type: str,
        message: str,
        current_value: float,
        threshold_value: float
    ) -> Alert:
        """Create an alert object."""
        return Alert(
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy_name=strategy_name,
            severity=severity,
            alert_type=alert_type,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value
        )

    def _record_alert(self, alert: Alert):
        """Record an alert to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO alerts (
                timestamp, strategy_name, severity, alert_type,
                message, current_value, threshold_value, acknowledged
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.timestamp, alert.strategy_name, alert.severity,
            alert.alert_type, alert.message, alert.current_value,
            alert.threshold_value, 0
        ))

        conn.commit()
        conn.close()

        log(f"[{alert.severity}] {alert.strategy_name}: {alert.message}")

    def get_historical_performance(
        self,
        strategy_name: str,
        entry_filter: str,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get historical performance snapshots."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        cursor.execute('''
            SELECT * FROM performance_snapshots
            WHERE strategy_name = ? AND entry_filter = ?
            AND timestamp >= ?
            ORDER BY timestamp DESC
        ''', (strategy_name, entry_filter, cutoff))

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        return [dict(zip(columns, row)) for row in rows]

    def get_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get latest performance summary for all strategies."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get most recent snapshot for each strategy
        cursor.execute('''
            SELECT ps.* FROM performance_snapshots ps
            INNER JOIN (
                SELECT strategy_name, entry_filter, MAX(timestamp) as max_ts
                FROM performance_snapshots
                GROUP BY strategy_name, entry_filter
            ) latest ON ps.strategy_name = latest.strategy_name
            AND ps.entry_filter = latest.entry_filter
            AND ps.timestamp = latest.max_ts
        ''')

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        summary = {}
        for row in rows:
            data = dict(zip(columns, row))
            key = f"{data['strategy_name']}_{data['entry_filter']}"
            summary[key] = data

        return summary

    def get_recent_alerts(self, days: int = 7, strategy_name: str = None) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        if strategy_name:
            cursor.execute('''
                SELECT * FROM alerts
                WHERE timestamp >= ? AND strategy_name = ?
                ORDER BY timestamp DESC
            ''', (cutoff, strategy_name))
        else:
            cursor.execute('''
                SELECT * FROM alerts
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', (cutoff,))

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        return [dict(zip(columns, row)) for row in rows]

    def detect_regime_change(self, strategy_name: str, entry_filter: str) -> Optional[Dict[str, Any]]:
        """
        Detect market regime changes by comparing recent vs historical performance.
        """
        recent = self.get_historical_performance(strategy_name, entry_filter, days=7)
        historical = self.get_historical_performance(strategy_name, entry_filter, days=30)

        if len(recent) < 2 or len(historical) < 4:
            return None

        # Calculate metrics for comparison
        recent_sharpe = np.mean([r['sharpe_ratio'] for r in recent])
        historical_sharpe = np.mean([r['sharpe_ratio'] for r in historical])

        recent_return = np.mean([r['total_return_pct'] for r in recent])
        historical_return = np.mean([r['total_return_pct'] for r in historical])

        # Calculate change ratios
        sharpe_change = (recent_sharpe - historical_sharpe) / abs(historical_sharpe) if historical_sharpe != 0 else 0
        return_change = (recent_return - historical_return) / abs(historical_return) if historical_return != 0 else 0

        # Detect regime change if significant shift
        if abs(sharpe_change) > 0.25 or abs(return_change) > 0.25:
            regime = "improving" if sharpe_change > 0 else "declining"
            confidence = min(abs(sharpe_change) + abs(return_change), 1.0)

            # Record to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO regime_history (timestamp, regime, confidence, volatility_20d, trend_strength)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now(timezone.utc).isoformat(), regime, confidence, 0, sharpe_change))
            conn.commit()
            conn.close()

            return {
                'regime': regime,
                'confidence': confidence,
                'sharpe_change': sharpe_change,
                'return_change': return_change,
            }

        return None

    def record_parameter_change(
        self,
        strategy_name: str,
        parameter_name: str,
        new_value: float,
        reason: str
    ):
        """Record a parameter change to history."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO parameter_history (timestamp, strategy_name, parameter_name, parameter_value, change_reason)
            VALUES (?, ?, ?, ?, ?)
        ''', (datetime.now(timezone.utc).isoformat(), strategy_name, parameter_name, new_value, reason))

        conn.commit()
        conn.close()

        log(f"Recorded parameter change: {strategy_name}.{parameter_name} = {new_value} ({reason})")

    def export_history(self, output_file: str = None) -> Dict[str, Any]:
        """Export all historical data to JSON."""
        if output_file is None:
            output_file = os.path.join(SCRIPT_DIR, 'results', f'history_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

        conn = sqlite3.connect(self.db_path)

        # Export all tables
        data = {
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'snapshots': [],
            'alerts': [],
            'parameter_history': [],
            'regime_history': [],
        }

        for table in ['performance_snapshots', 'alerts', 'parameter_history', 'regime_history']:
            cursor = conn.cursor()
            cursor.execute(f'SELECT * FROM {table}')
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            key = table.replace('performance_', '')
            data[key] = [dict(zip(columns, row)) for row in rows]

        conn.close()

        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        log(f"Exported history to: {output_file}")
        return data

    def get_rolling_metrics(
        self,
        strategy_name: str,
        entry_filter: str,
        window_days: int = 7
    ) -> Dict[str, float]:
        """Calculate rolling metrics over a window."""
        snapshots = self.get_historical_performance(strategy_name, entry_filter, days=window_days)

        if not snapshots:
            return {}

        return {
            'avg_return': np.mean([s['total_return_pct'] for s in snapshots]),
            'avg_sharpe': np.mean([s['sharpe_ratio'] for s in snapshots]),
            'avg_win_rate': np.mean([s['win_rate_pct'] for s in snapshots]),
            'max_drawdown': max([s['max_drawdown_pct'] for s in snapshots]),
            'total_trades': sum([s['total_trades'] for s in snapshots]),
            'snapshot_count': len(snapshots),
        }


def main():
    """Test the performance tracker."""
    tracker = PerformanceTracker()

    # Create test snapshot
    snapshot = PerformanceSnapshot(
        timestamp=datetime.now(timezone.utc).isoformat(),
        strategy_name="Adaptive_Baseline",
        entry_filter="baseline",
        total_return_pct=15.5,
        max_drawdown_pct=8.2,
        sharpe_ratio=1.45,
        win_rate_pct=45.0,
        profit_factor=1.85,
        total_trades=25,
        max_losing_streak=4,
        oos_return_pct=12.3,
        oos_efficiency=0.79,
        parameter_stability=0.85,
        data_bars=1000,
        evaluation_type="full"
    )

    # Record it
    tracker.record_snapshot(snapshot)

    # Get summary
    summary = tracker.get_performance_summary()
    log(f"Performance summary: {len(summary)} strategies tracked")

    # Get alerts
    alerts = tracker.get_recent_alerts(days=1)
    log(f"Recent alerts: {len(alerts)}")


if __name__ == "__main__":
    main()
