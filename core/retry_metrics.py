#!/usr/bin/env python3
"""
Retry Metrics Tracking System

Tracks and reports statistics on adaptive retry system performance including:
- Success rates per attempt
- Average retry counts
- Context reduction effectiveness
- Provider-specific performance

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.0
"""

import json
import sqlite3
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RetryAttemptMetrics:
    """Metrics for a single retry attempt"""
    attempt_number: int
    success: bool
    error_type: Optional[str] = None
    clinical_text_length: int = 0
    conversation_history_length: int = 0
    tool_context_length: int = 0
    switched_to_minimal: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExtractionRetryMetrics:
    """Metrics for retry attempts during a single extraction"""
    extraction_id: str
    provider: str
    model_name: str
    total_attempts: int = 0
    successful: bool = False
    final_attempt_number: int = 1
    attempts: List[RetryAttemptMetrics] = field(default_factory=list)
    original_text_length: int = 0
    final_text_length: int = 0
    total_retry_time_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_attempt(self, metrics: RetryAttemptMetrics):
        """Add a retry attempt"""
        self.attempts.append(metrics)
        self.total_attempts = len(self.attempts)
        if metrics.success:
            self.successful = True
            self.final_attempt_number = metrics.attempt_number


@dataclass
class RetryMetricsSummary:
    """Summary statistics for retry system performance"""
    total_extractions: int = 0
    total_retry_attempts: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0

    # Success rate by attempt number
    success_rate_by_attempt: Dict[int, float] = field(default_factory=dict)

    # Average metrics
    avg_attempts_per_extraction: float = 0.0
    avg_context_reduction: float = 0.0

    # Provider-specific
    metrics_by_provider: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Most common error types
    error_type_counts: Dict[str, int] = field(default_factory=dict)


class RetryMetricsTracker:
    """
    Tracks and persists retry metrics for analysis and visualization
    """

    def __init__(self, db_path: str = "cache/retry_metrics.db"):
        """
        Initialize retry metrics tracker

        Args:
            db_path: Path to SQLite database for persistence
        """
        self.db_path = db_path
        self._ensure_db_exists()
        logger.info(f"RetryMetricsTracker initialized: {db_path}")

    def _ensure_db_exists(self):
        """Ensure database and tables exist"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extraction_metrics (
                extraction_id TEXT PRIMARY KEY,
                provider TEXT,
                model_name TEXT,
                total_attempts INTEGER,
                successful BOOLEAN,
                final_attempt_number INTEGER,
                original_text_length INTEGER,
                final_text_length INTEGER,
                total_retry_time_seconds REAL,
                timestamp TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attempt_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                extraction_id TEXT,
                attempt_number INTEGER,
                success BOOLEAN,
                error_type TEXT,
                clinical_text_length INTEGER,
                conversation_history_length INTEGER,
                tool_context_length INTEGER,
                switched_to_minimal BOOLEAN,
                timestamp TEXT,
                FOREIGN KEY (extraction_id) REFERENCES extraction_metrics(extraction_id)
            )
        """)

        conn.commit()
        conn.close()

    def record_extraction(self, metrics: ExtractionRetryMetrics):
        """
        Record extraction retry metrics

        Args:
            metrics: ExtractionRetryMetrics to record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Insert extraction metrics
            cursor.execute("""
                INSERT OR REPLACE INTO extraction_metrics
                (extraction_id, provider, model_name, total_attempts, successful,
                 final_attempt_number, original_text_length, final_text_length,
                 total_retry_time_seconds, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.extraction_id,
                metrics.provider,
                metrics.model_name,
                metrics.total_attempts,
                metrics.successful,
                metrics.final_attempt_number,
                metrics.original_text_length,
                metrics.final_text_length,
                metrics.total_retry_time_seconds,
                metrics.timestamp
            ))

            # Insert attempt metrics
            for attempt in metrics.attempts:
                cursor.execute("""
                    INSERT INTO attempt_metrics
                    (extraction_id, attempt_number, success, error_type,
                     clinical_text_length, conversation_history_length,
                     tool_context_length, switched_to_minimal, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.extraction_id,
                    attempt.attempt_number,
                    attempt.success,
                    attempt.error_type,
                    attempt.clinical_text_length,
                    attempt.conversation_history_length,
                    attempt.tool_context_length,
                    attempt.switched_to_minimal,
                    attempt.timestamp
                ))

            conn.commit()
            logger.debug(f"Recorded metrics for extraction {metrics.extraction_id}")

        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_summary(self, provider: Optional[str] = None, limit_days: int = 30) -> RetryMetricsSummary:
        """
        Get summary statistics for retry metrics

        Args:
            provider: Filter by provider (None = all providers)
            limit_days: Only include last N days of data

        Returns:
            RetryMetricsSummary with aggregated statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        summary = RetryMetricsSummary()

        try:
            # Filter by provider if specified
            provider_filter = f"AND provider = '{provider}'" if provider else ""

            # Total extractions
            cursor.execute(f"""
                SELECT COUNT(*) FROM extraction_metrics
                WHERE datetime(timestamp) >= datetime('now', '-{limit_days} days')
                {provider_filter}
            """)
            summary.total_extractions = cursor.fetchone()[0]

            # Successful/failed extractions
            cursor.execute(f"""
                SELECT
                    SUM(CASE WHEN successful THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN NOT successful THEN 1 ELSE 0 END) as failed
                FROM extraction_metrics
                WHERE datetime(timestamp) >= datetime('now', '-{limit_days} days')
                {provider_filter}
            """)
            row = cursor.fetchone()
            summary.successful_extractions = row[0] or 0
            summary.failed_extractions = row[1] or 0

            # Total retry attempts
            cursor.execute(f"""
                SELECT SUM(total_attempts) FROM extraction_metrics
                WHERE datetime(timestamp) >= datetime('now', '-{limit_days} days')
                {provider_filter}
            """)
            summary.total_retry_attempts = cursor.fetchone()[0] or 0

            # Average attempts per extraction
            if summary.total_extractions > 0:
                summary.avg_attempts_per_extraction = summary.total_retry_attempts / summary.total_extractions

            # Success rate by attempt number
            cursor.execute(f"""
                SELECT
                    attempt_number,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
                FROM attempt_metrics
                WHERE extraction_id IN (
                    SELECT extraction_id FROM extraction_metrics
                    WHERE datetime(timestamp) >= datetime('now', '-{limit_days} days')
                    {provider_filter}
                )
                GROUP BY attempt_number
                ORDER BY attempt_number
            """)
            for row in cursor.fetchall():
                summary.success_rate_by_attempt[row[0]] = round(row[1], 2)

            # Average context reduction
            cursor.execute(f"""
                SELECT
                    AVG((original_text_length - final_text_length) * 100.0 / original_text_length)
                FROM extraction_metrics
                WHERE datetime(timestamp) >= datetime('now', '-{limit_days} days')
                AND original_text_length > 0
                {provider_filter}
            """)
            result = cursor.fetchone()[0]
            summary.avg_context_reduction = round(result, 2) if result else 0.0

            # Error type counts
            cursor.execute(f"""
                SELECT error_type, COUNT(*) as count
                FROM attempt_metrics
                WHERE error_type IS NOT NULL
                AND extraction_id IN (
                    SELECT extraction_id FROM extraction_metrics
                    WHERE datetime(timestamp) >= datetime('now', '-{limit_days} days')
                    {provider_filter}
                )
                GROUP BY error_type
                ORDER BY count DESC
                LIMIT 10
            """)
            for row in cursor.fetchall():
                summary.error_type_counts[row[0]] = row[1]

            # Provider-specific metrics
            cursor.execute(f"""
                SELECT
                    provider,
                    COUNT(*) as total,
                    SUM(CASE WHEN successful THEN 1 ELSE 0 END) as successful,
                    AVG(total_attempts) as avg_attempts
                FROM extraction_metrics
                WHERE datetime(timestamp) >= datetime('now', '-{limit_days} days')
                GROUP BY provider
            """)
            for row in cursor.fetchall():
                prov, total, successful, avg_attempts = row
                success_rate = (successful / total * 100) if total > 0 else 0
                summary.metrics_by_provider[prov] = {
                    'total': total,
                    'successful': successful,
                    'success_rate': round(success_rate, 2),
                    'avg_attempts': round(avg_attempts, 2)
                }

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
        finally:
            conn.close()

        return summary

    def get_recent_failures(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent failed extractions for debugging

        Args:
            limit: Maximum number of failures to return

        Returns:
            List of failure details
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        failures = []

        try:
            cursor.execute("""
                SELECT
                    e.extraction_id,
                    e.provider,
                    e.model_name,
                    e.total_attempts,
                    e.timestamp,
                    a.error_type
                FROM extraction_metrics e
                LEFT JOIN (
                    SELECT extraction_id, error_type
                    FROM attempt_metrics
                    WHERE error_type IS NOT NULL
                    ORDER BY timestamp DESC
                ) a ON e.extraction_id = a.extraction_id
                WHERE e.successful = 0
                ORDER BY e.timestamp DESC
                LIMIT ?
            """, (limit,))

            for row in cursor.fetchall():
                failures.append({
                    'extraction_id': row[0],
                    'provider': row[1],
                    'model_name': row[2],
                    'total_attempts': row[3],
                    'timestamp': row[4],
                    'last_error_type': row[5]
                })

        except Exception as e:
            logger.error(f"Failed to get recent failures: {e}")
        finally:
            conn.close()

        return failures

    def clear_old_metrics(self, days: int = 90):
        """
        Clear metrics older than specified days

        Args:
            days: Delete metrics older than this many days
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(f"""
                DELETE FROM attempt_metrics
                WHERE extraction_id IN (
                    SELECT extraction_id FROM extraction_metrics
                    WHERE datetime(timestamp) < datetime('now', '-{days} days')
                )
            """)

            cursor.execute(f"""
                DELETE FROM extraction_metrics
                WHERE datetime(timestamp) < datetime('now', '-{days} days')
            """)

            conn.commit()
            deleted = cursor.rowcount
            logger.info(f"Cleared {deleted} old metric records (older than {days} days)")

        except Exception as e:
            logger.error(f"Failed to clear old metrics: {e}")
            conn.rollback()
        finally:
            conn.close()


# Global metrics tracker instance
_global_tracker: Optional[RetryMetricsTracker] = None


def get_retry_metrics_tracker(db_path: str = "cache/retry_metrics.db") -> RetryMetricsTracker:
    """
    Get global retry metrics tracker instance

    Args:
        db_path: Path to metrics database

    Returns:
        RetryMetricsTracker instance
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = RetryMetricsTracker(db_path=db_path)
    return _global_tracker
