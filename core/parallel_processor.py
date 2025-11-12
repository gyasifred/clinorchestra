#!/usr/bin/env python3
"""
Parallel Processing System - Multi-row concurrent extraction
Version: 1.0.0
Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center

Features:
- Concurrent row processing (5-10x speedup for cloud APIs)
- Intelligent worker pool sizing based on provider
- Rate limiting for API compliance
- Robust error handling and recovery
- Progress tracking across workers
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from queue import Queue
import traceback
from core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessingTask:
    """Single processing task"""
    row_index: int
    clinical_text: str
    label_value: Any
    label_context: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result from processing a single row"""
    row_index: int
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration: float = 0.0


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call_time = 0
        self.lock = threading.Lock()

    def acquire(self):
        """Wait if necessary to respect rate limit"""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call_time

            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                time.sleep(sleep_time)

            self.last_call_time = time.time()


class ParallelProcessor:
    """
    Parallel processor for multi-row extraction

    Automatically determines optimal worker count based on provider:
    - Local models: 2 workers (GPU memory limited)
    - Cloud APIs: 5-10 workers (rate limited)

    Provides 5-10x speedup for I/O-bound tasks (API calls)
    """

    def __init__(self,
                 max_workers: int = 5,
                 provider: str = 'openai',
                 rate_limit: int = 60):
        """
        Initialize parallel processor

        Args:
            max_workers: Maximum number of concurrent workers
            provider: LLM provider name ('openai', 'anthropic', 'local', etc.)
            rate_limit: API calls per minute (for rate limiting)
        """
        self.max_workers = self._determine_optimal_workers(max_workers, provider)
        self.provider = provider
        self.rate_limiter = RateLimiter(calls_per_minute=rate_limit)
        self.stop_flag = threading.Event()

        logger.info(f"[INIT]  Parallel Processor initialized: {self.max_workers} workers, provider={provider}")

    def _determine_optimal_workers(self, requested: int, provider: str) -> int:
        """Determine optimal worker count based on provider"""
        if provider == 'local':
            # GPU memory limited - conservative
            optimal = min(requested, 2)
            logger.info(f"[LOCAL] Local model detected: limiting to {optimal} workers (GPU memory)")
            return optimal
        elif provider in ['openai', 'anthropic', 'google', 'azure']:
            # API rate limited - can handle more concurrent
            optimal = min(requested, 10)
            logger.info(f"[CLOUD]  Cloud API detected: using {optimal} workers")
            return optimal
        else:
            return requested

    def process_batch(self,
                     tasks: List[ProcessingTask],
                     process_function: Callable[[str, Any], Dict[str, Any]],
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> List[ProcessingResult]:
        """
        Process batch of tasks in parallel

        Args:
            tasks: List of processing tasks
            process_function: Function to process single row (text, label) -> result
            progress_callback: Optional callback for progress updates (completed, total)

        Returns:
            List of processing results (same order as input tasks)
        """
        if not tasks:
            return []

        total_tasks = len(tasks)
        logger.info(f"[START] Starting parallel processing: {total_tasks} tasks across {self.max_workers} workers")

        # Results dictionary (thread-safe)
        results_dict = {}
        results_lock = threading.Lock()
        completed_count = 0
        completed_lock = threading.Lock()

        def process_single_task(task: ProcessingTask) -> ProcessingResult:
            """Process single task with error handling"""
            if self.stop_flag.is_set():
                return ProcessingResult(
                    row_index=task.row_index,
                    success=False,
                    error="Processing stopped by user"
                )

            start_time = time.time()

            try:
                # Rate limiting for API calls
                if self.provider != 'local':
                    self.rate_limiter.acquire()

                # Process the row
                result = process_function(task.clinical_text, task.label_value)

                duration = time.time() - start_time

                return ProcessingResult(
                    row_index=task.row_index,
                    success=True,
                    result=result,
                    duration=duration
                )

            except Exception as e:
                duration = time.time() - start_time
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(f"Error processing row {task.row_index}: {error_msg}")
                logger.debug(traceback.format_exc())

                return ProcessingResult(
                    row_index=task.row_index,
                    success=False,
                    error=error_msg,
                    duration=duration
                )

        def task_done_callback(future):
            """Callback when task completes"""
            nonlocal completed_count

            try:
                result = future.result()

                with results_lock:
                    results_dict[result.row_index] = result

                with completed_lock:
                    completed_count += 1
                    current_completed = completed_count

                # Progress callback
                if progress_callback:
                    progress_callback(current_completed, total_tasks)

                # Log progress
                if current_completed % 10 == 0 or current_completed == total_tasks:
                    logger.info(f"[METRICS] Progress: {current_completed}/{total_tasks} ({current_completed/total_tasks*100:.1f}%)")

            except Exception as e:
                logger.error(f"Error in task callback: {e}")

        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []

            for task in tasks:
                future = executor.submit(process_single_task, task)
                future.add_done_callback(task_done_callback)
                futures.append(future)

            # Wait for all tasks to complete
            for future in as_completed(futures):
                if self.stop_flag.is_set():
                    logger.warning("[STOP]  Stop requested - cancelling remaining tasks")
                    executor.shutdown(wait=False)
                    break

        # Convert results dict to ordered list
        results = [results_dict.get(i) for i in range(total_tasks)]

        # Handle any missing results
        for i, result in enumerate(results):
            if result is None:
                results[i] = ProcessingResult(
                    row_index=i,
                    success=False,
                    error="Task was not completed"
                )

        # Log summary
        successful = sum(1 for r in results if r.success)
        failed = total_tasks - successful
        avg_duration = sum(r.duration for r in results) / total_tasks if results else 0

        logger.info("=" * 80)
        logger.info("[SUCCESS] Parallel Processing Complete")
        logger.info(f"   Total Tasks: {total_tasks}")
        logger.info(f"   Successful: {successful}")
        logger.info(f"   Failed: {failed}")
        logger.info(f"   Avg Duration: {avg_duration:.2f}s per task")
        logger.info(f"   Workers Used: {self.max_workers}")
        logger.info("=" * 80)

        return results

    def stop(self):
        """Signal processor to stop"""
        logger.warning("[STOP]  Stop signal received")
        self.stop_flag.set()

    def reset(self):
        """Reset stop flag"""
        self.stop_flag.clear()


def estimate_speedup(num_tasks: int, workers: int, avg_task_duration: float = 1.5) -> Dict[str, Any]:
    """
    Estimate speedup from parallel processing

    Args:
        num_tasks: Number of tasks to process
        workers: Number of parallel workers
        avg_task_duration: Average duration per task in seconds

    Returns:
        Dictionary with timing estimates
    """
    sequential_time = num_tasks * avg_task_duration
    parallel_time = (num_tasks / workers) * avg_task_duration * 1.1  # 10% overhead
    speedup = sequential_time / parallel_time
    time_saved = sequential_time - parallel_time

    return {
        'sequential_time': sequential_time,
        'parallel_time': parallel_time,
        'speedup': speedup,
        'time_saved': time_saved,
        'workers': workers,
        'efficiency': (speedup / workers) * 100  # Percentage of ideal speedup
    }
