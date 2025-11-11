#!/usr/bin/env python3
"""
Performance Monitoring Module - Comprehensive metrics tracking
Version: 1.0.0
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center

Tracks:
- Component-level timing
- Cache hit rates
- Resource utilization
- Bottleneck detection
"""

import time
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import json
from pathlib import Path
from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation"""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self):
        """Mark operation as complete"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time


@dataclass
class CacheMetrics:
    """Cache performance metrics"""
    cache_name: str
    hits: int = 0
    misses: int = 0
    total_requests: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests * 100.0

    def record_hit(self):
        """Record a cache hit"""
        self.hits += 1
        self.total_requests += 1

    def record_miss(self):
        """Record a cache miss"""
        self.misses += 1
        self.total_requests += 1


@dataclass
class ComponentStats:
    """Statistics for a component"""
    component_name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: Optional[float] = None
    max_time: Optional[float] = None
    times: List[float] = field(default_factory=list)

    def add_measurement(self, duration: float):
        """Add a timing measurement"""
        self.total_calls += 1
        self.total_time += duration
        self.times.append(duration)

        if self.min_time is None or duration < self.min_time:
            self.min_time = duration
        if self.max_time is None or duration > self.max_time:
            self.max_time = duration

    @property
    def avg_time(self) -> float:
        """Calculate average time"""
        if self.total_calls == 0:
            return 0.0
        return self.total_time / self.total_calls

    @property
    def p50(self) -> float:
        """Calculate 50th percentile"""
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        idx = len(sorted_times) // 2
        return sorted_times[idx]

    @property
    def p95(self) -> float:
        """Calculate 95th percentile"""
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def p99(self) -> float:
        """Calculate 99th percentile"""
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]


class PerformanceMonitor:
    """
    Centralized performance monitoring system

    Features:
    - Component-level timing
    - Cache hit rate tracking
    - Percentile calculations (p50, p95, p99)
    - Bottleneck detection
    - Export to JSON for analysis
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.component_stats: Dict[str, ComponentStats] = {}
        self.cache_metrics: Dict[str, CacheMetrics] = {}
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self.lock = threading.Lock()
        self.session_start = time.time()

        if self.enabled:
            logger.info("🔍 Performance monitoring enabled")

    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Start timing an operation

        Returns: operation_id for later completion
        """
        if not self.enabled:
            return ""

        operation_id = f"{operation_name}_{time.time()}_{threading.get_ident()}"

        with self.lock:
            self.active_operations[operation_id] = PerformanceMetrics(
                operation_name=operation_name,
                start_time=time.time(),
                metadata=metadata or {}
            )

        return operation_id

    def end_operation(self, operation_id: str, additional_metadata: Optional[Dict[str, Any]] = None):
        """Complete an operation and record metrics"""
        if not self.enabled or not operation_id:
            return

        with self.lock:
            if operation_id not in self.active_operations:
                return

            operation = self.active_operations[operation_id]
            operation.complete()

            if additional_metadata:
                operation.metadata.update(additional_metadata)

            # Update component stats
            component_name = operation.operation_name
            if component_name not in self.component_stats:
                self.component_stats[component_name] = ComponentStats(component_name)

            self.component_stats[component_name].add_measurement(operation.duration)

            # Remove from active operations
            del self.active_operations[operation_id]

    def record_cache_hit(self, cache_name: str):
        """Record a cache hit"""
        if not self.enabled:
            return

        with self.lock:
            if cache_name not in self.cache_metrics:
                self.cache_metrics[cache_name] = CacheMetrics(cache_name)
            self.cache_metrics[cache_name].record_hit()

    def record_cache_miss(self, cache_name: str):
        """Record a cache miss"""
        if not self.enabled:
            return

        with self.lock:
            if cache_name not in self.cache_metrics:
                self.cache_metrics[cache_name] = CacheMetrics(cache_name)
            self.cache_metrics[cache_name].record_miss()

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self.lock:
            session_duration = time.time() - self.session_start

            # Component summaries
            component_summary = {}
            for name, stats in self.component_stats.items():
                component_summary[name] = {
                    'total_calls': stats.total_calls,
                    'total_time': round(stats.total_time, 2),
                    'avg_time': round(stats.avg_time, 2),
                    'min_time': round(stats.min_time, 2) if stats.min_time else 0,
                    'max_time': round(stats.max_time, 2) if stats.max_time else 0,
                    'p50': round(stats.p50, 2),
                    'p95': round(stats.p95, 2),
                    'p99': round(stats.p99, 2)
                }

            # Cache summaries
            cache_summary = {}
            for name, metrics in self.cache_metrics.items():
                cache_summary[name] = {
                    'hits': metrics.hits,
                    'misses': metrics.misses,
                    'total_requests': metrics.total_requests,
                    'hit_rate': round(metrics.hit_rate, 2)
                }

            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks()

            return {
                'session_duration': round(session_duration, 2),
                'components': component_summary,
                'caches': cache_summary,
                'bottlenecks': bottlenecks,
                'timestamp': datetime.now().isoformat()
            }

    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []

        # Find components with highest total time
        sorted_components = sorted(
            self.component_stats.items(),
            key=lambda x: x[1].total_time,
            reverse=True
        )

        if sorted_components:
            total_time = sum(stats.total_time for _, stats in sorted_components)

            for name, stats in sorted_components[:5]:  # Top 5 bottlenecks
                percentage = (stats.total_time / total_time * 100) if total_time > 0 else 0
                if percentage > 10:  # Only report if >10% of total time
                    bottlenecks.append({
                        'component': name,
                        'total_time': round(stats.total_time, 2),
                        'percentage': round(percentage, 1),
                        'avg_time': round(stats.avg_time, 2),
                        'calls': stats.total_calls
                    })

        return bottlenecks

    def export_to_file(self, filepath: str):
        """Export metrics to JSON file"""
        summary = self.get_summary()

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"📊 Performance metrics exported to {filepath}")

    def log_summary(self):
        """Log performance summary"""
        if not self.enabled:
            return

        summary = self.get_summary()

        logger.info("=" * 80)
        logger.info("📊 PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Session Duration: {summary['session_duration']}s")
        logger.info("")

        if summary['components']:
            logger.info("⏱️  Component Timings:")
            for name, stats in summary['components'].items():
                logger.info(f"  • {name}:")
                logger.info(f"      Calls: {stats['total_calls']}, "
                          f"Total: {stats['total_time']}s, "
                          f"Avg: {stats['avg_time']}s, "
                          f"P95: {stats['p95']}s")
            logger.info("")

        if summary['caches']:
            logger.info("💾 Cache Performance:")
            for name, metrics in summary['caches'].items():
                logger.info(f"  • {name}: "
                          f"Hit Rate: {metrics['hit_rate']}%, "
                          f"Hits: {metrics['hits']}, "
                          f"Misses: {metrics['misses']}")
            logger.info("")

        if summary['bottlenecks']:
            logger.info("🔴 Performance Bottlenecks:")
            for bottleneck in summary['bottlenecks']:
                logger.info(f"  • {bottleneck['component']}: "
                          f"{bottleneck['percentage']}% of time "
                          f"({bottleneck['total_time']}s across {bottleneck['calls']} calls)")
            logger.info("")

        logger.info("=" * 80)

    def reset(self):
        """Reset all metrics"""
        with self.lock:
            self.component_stats.clear()
            self.cache_metrics.clear()
            self.active_operations.clear()
            self.session_start = time.time()

        logger.info("🔄 Performance metrics reset")


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(enabled: bool = True) -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor(enabled=enabled)
    return _global_monitor


class TimingContext:
    """Context manager for timing operations"""

    def __init__(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None, enabled: bool = True):
        self.operation_name = operation_name
        self.metadata = metadata
        self.enabled = enabled
        self.operation_id = None
        self.monitor = get_performance_monitor(enabled=enabled)

    def __enter__(self):
        if self.enabled:
            self.operation_id = self.monitor.start_operation(self.operation_name, self.metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self.operation_id:
            error_metadata = {}
            if exc_type is not None:
                error_metadata = {
                    'error': True,
                    'error_type': exc_type.__name__,
                    'error_message': str(exc_val)
                }
            self.monitor.end_operation(self.operation_id, error_metadata)
        return False
