#!/usr/bin/env python3
"""
Multi-GPU Parallel Processor for ClinOrchestra
Enables true multi-GPU parallelism using ProcessPoolExecutor

Key differences from standard parallel_processor.py:
1. Uses ProcessPoolExecutor instead of ThreadPoolExecutor
2. Each worker process loads model on a different GPU
3. GPU assignment via CUDA_VISIBLE_DEVICES or torch.cuda.set_device()

Author: Frederick Gyasi (gyasi@musc.edu)
Version: 1.0.0
"""

import os
import time
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from multiprocessing import Manager
import traceback


@dataclass
class MultiGPUTask:
    """Task for multi-GPU processing"""
    task_id: int
    row_index: int
    clinical_text: str
    label_value: Any
    prompt_variables: Optional[Dict[str, Any]] = None
    gpu_id: Optional[int] = None  # Which GPU to use


@dataclass
class MultiGPUResult:
    """Result from multi-GPU processing"""
    task_id: int
    row_index: int
    gpu_id: int
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration: float = 0.0


class MultiGPUProcessor:
    """
    Multi-GPU Parallel Processor for Local Models

    Distributes extraction tasks across multiple GPUs by:
    1. Spawning separate processes (not threads)
    2. Each process loads model on assigned GPU
    3. True parallel GPU utilization

    WARNING: Only use for local models. Cloud APIs should use
    standard parallel_processor.py (ThreadPoolExecutor is fine for API calls)
    """

    def __init__(self, num_gpus: int = None, app_state = None):
        """
        Initialize multi-GPU processor

        Args:
            num_gpus: Number of GPUs to use (default: all available)
            app_state: AppState instance for configuration
        """
        self.app_state = app_state

        # Detect available GPUs
        if torch.cuda.is_available():
            self.total_gpus = torch.cuda.device_count()
        else:
            raise RuntimeError("CUDA not available - multi-GPU processing requires GPUs")

        # Determine how many GPUs to use
        if num_gpus is None:
            self.num_gpus = self.total_gpus
        else:
            self.num_gpus = min(num_gpus, self.total_gpus)

        if self.num_gpus == 0:
            raise RuntimeError("No GPUs available for multi-GPU processing")

        print(f"[MULTI-GPU] Initialized with {self.num_gpus}/{self.total_gpus} GPUs")

        # GPU device IDs to use
        self.gpu_ids = list(range(self.num_gpus))

    def process_batch(self,
                     tasks: List[MultiGPUTask],
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> List[MultiGPUResult]:
        """
        Process batch across multiple GPUs

        Args:
            tasks: List of extraction tasks
            progress_callback: Optional callback for progress (completed, total)

        Returns:
            List of results (same order as tasks)
        """
        if not tasks:
            print("[ERROR] No tasks provided to process_batch")
            return []

        # Assign GPUs to tasks (round-robin)
        for i, task in enumerate(tasks):
            task.gpu_id = self.gpu_ids[i % self.num_gpus]
            task.task_id = i

        total_tasks = len(tasks)
        print(f"[START] Processing {total_tasks} tasks across {self.num_gpus} GPUs")
        print(f"[DEBUG] app_state type: {type(self.app_state)}")
        print(f"[DEBUG] Attempting to serialize app_state for worker processes...")

        # Shared progress counter
        manager = Manager()
        completed_count = manager.Value('i', 0)
        results_dict = manager.dict()

        # Process in parallel using separate processes
        try:
            print("[DEBUG] Creating ProcessPoolExecutor...")
            executor = ProcessPoolExecutor(max_workers=self.num_gpus)
            print(f"[DEBUG] ProcessPoolExecutor created with {self.num_gpus} workers")
        except Exception as e:
            print(f"[ERROR] Failed to create ProcessPoolExecutor: {e}")
            import traceback
            traceback.print_exc()
            return []

        with executor:
            # Submit tasks
            futures = {}
            submit_errors = 0
            for i, task in enumerate(tasks):
                try:
                    print(f"[DEBUG] Submitting task {i}/{total_tasks}...")
                    future = executor.submit(
                        _process_single_task_on_gpu,
                        task,
                        self.app_state,
                        completed_count,
                        total_tasks
                    )
                    futures[future] = task.task_id
                    if i == 0:
                        print(f"[DEBUG] First task submitted successfully")
                except Exception as e:
                    submit_errors += 1
                    print(f"[ERROR] Failed to submit task {task.task_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Create failed result immediately
                    results_dict[task.task_id] = MultiGPUResult(
                        task_id=task.task_id,
                        row_index=task.row_index,
                        gpu_id=-1,
                        success=False,
                        error=f"Submit failed: {str(e)}"
                    )

            print(f"[DEBUG] Submitted {len(futures)} tasks, {submit_errors} submit errors")

            # Collect results
            completed_tasks = 0
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    results_dict[task_id] = result
                    completed_tasks += 1

                    # Progress callback
                    if progress_callback:
                        with completed_count.get_lock():
                            current = completed_count.value
                        progress_callback(current, total_tasks)

                except Exception as e:
                    print(f"[ERROR] Task {task_id} failed during execution: {e}")
                    import traceback
                    traceback.print_exc()
                    results_dict[task_id] = MultiGPUResult(
                        task_id=task_id,
                        row_index=task_id,
                        gpu_id=-1,
                        success=False,
                        error=f"Execution failed: {str(e)}"
                    )

            print(f"[DEBUG] Completed collecting {completed_tasks} task results")

        # Convert dict to ordered list
        results = [results_dict.get(i) for i in range(total_tasks)]

        # Handle missing results
        for i, result in enumerate(results):
            if result is None:
                results[i] = MultiGPUResult(
                    task_id=i,
                    row_index=i,
                    gpu_id=-1,
                    success=False,
                    error="Task was not completed"
                )

        # Log summary by GPU
        self._log_summary(results)

        return results

    def _log_summary(self, results: List[MultiGPUResult]):
        """Log summary statistics per GPU"""
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        print("=" * 80)
        print("[COMPLETE] Multi-GPU Processing Summary")
        print(f"   Total: {len(results)}, Successful: {successful}, Failed: {failed}")

        # Per-GPU statistics
        gpu_stats = {}
        for gpu_id in self.gpu_ids:
            gpu_results = [r for r in results if r.gpu_id == gpu_id]
            if gpu_results:
                gpu_success = sum(1 for r in gpu_results if r.success)
                avg_time = sum(r.duration for r in gpu_results) / len(gpu_results)
                gpu_stats[gpu_id] = {
                    'total': len(gpu_results),
                    'success': gpu_success,
                    'avg_time': avg_time
                }

        for gpu_id, stats in gpu_stats.items():
            print(f"   GPU {gpu_id}: {stats['total']} tasks, "
                  f"{stats['success']} successful, "
                  f"{stats['avg_time']:.2f}s avg")

        print("=" * 80)


def _process_single_task_on_gpu(task: MultiGPUTask,
                                  app_state,
                                  completed_count,
                                  total_tasks) -> MultiGPUResult:
    """
    Process single task on assigned GPU
    This function runs in a separate process

    Args:
        task: Task to process
        app_state: AppState for configuration
        completed_count: Shared counter for progress
        total_tasks: Total number of tasks

    Returns:
        Result of processing
    """
    start_time = time.time()
    gpu_id = task.gpu_id

    try:
        print(f"[WORKER-{gpu_id}] Starting task {task.task_id}")

        # Set GPU for this process
        torch.cuda.set_device(gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"[WORKER-{gpu_id}] GPU set to {gpu_id}")

        # Import required managers and agent factory
        from core.agent_factory import create_agent
        print(f"[WORKER-{gpu_id}] Imported create_agent")

        # CRITICAL FIX: Recreate all managers in this worker process
        # Cannot share managers across processes - must recreate from app_state

        try:
            # Initialize LLM manager for this process
            print(f"[WORKER-{gpu_id}] Getting LLM manager...")
            llm_manager = app_state.get_llm_manager()
            print(f"[WORKER-{gpu_id}] LLM manager created")

            # Initialize regex preprocessor
            print(f"[WORKER-{gpu_id}] Getting regex preprocessor...")
            regex_preprocessor = app_state.get_regex_preprocessor()
            print(f"[WORKER-{gpu_id}] Regex preprocessor created")

            # Initialize extras manager
            print(f"[WORKER-{gpu_id}] Getting extras manager...")
            extras_manager = app_state.get_extras_manager()
            print(f"[WORKER-{gpu_id}] Extras manager created")

            # Initialize function registry
            print(f"[WORKER-{gpu_id}] Getting function registry...")
            function_registry = app_state.get_function_registry()
            print(f"[WORKER-{gpu_id}] Function registry created")

            # Initialize RAG engine if enabled
            print(f"[WORKER-{gpu_id}] Getting RAG engine...")
            rag_engine = app_state.get_or_initialize_rag_engine()
            print(f"[WORKER-{gpu_id}] RAG engine created")

            # Create agent for this process with ALL required parameters
            print(f"[WORKER-{gpu_id}] Creating agent...")
            agent = create_agent(
                llm_manager=llm_manager,
                rag_engine=rag_engine,
                extras_manager=extras_manager,
                function_registry=function_registry,
                regex_preprocessor=regex_preprocessor,
                app_state=app_state
            )
            print(f"[WORKER-{gpu_id}] Agent created successfully")
        except Exception as init_error:
            print(f"[WORKER-{gpu_id}] INITIALIZATION FAILED: {init_error}")
            import traceback
            traceback.print_exc()
            raise

        # Ensure model is on correct device
        if hasattr(agent, 'llm_manager') and hasattr(agent.llm_manager, 'model'):
            if agent.llm_manager.model is not None:
                agent.llm_manager.model.to(f'cuda:{gpu_id}')

        # Process the extraction
        result = agent.extract(
            clinical_text=task.clinical_text,
            label_value=task.label_value,
            prompt_variables=task.prompt_variables
        )

        duration = time.time() - start_time

        # Update progress
        with completed_count.get_lock():
            completed_count.value += 1
            current = completed_count.value

        if current % 10 == 0 or current == total_tasks:
            print(f"[PROGRESS] {current}/{total_tasks} ({current/total_tasks*100:.1f}%) - "
                  f"GPU {gpu_id} completed task {task.task_id}")

        return MultiGPUResult(
            task_id=task.task_id,
            row_index=task.row_index,
            gpu_id=gpu_id,
            success=True,
            result=result,
            duration=duration
        )

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] GPU {gpu_id}, Task {task.task_id}: {error_msg}")

        with completed_count.get_lock():
            completed_count.value += 1

        return MultiGPUResult(
            task_id=task.task_id,
            row_index=task.row_index,
            gpu_id=gpu_id,
            success=False,
            error=error_msg,
            duration=duration
        )


def check_multi_gpu_readiness() -> Dict[str, Any]:
    """
    Check if system is ready for multi-GPU processing

    Returns:
        Dictionary with readiness information
    """
    info = {
        'ready': False,
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': 0,
        'gpus': [],
        'issues': []
    }

    if not torch.cuda.is_available():
        info['issues'].append("CUDA not available")
        return info

    gpu_count = torch.cuda.device_count()
    info['gpu_count'] = gpu_count

    if gpu_count == 0:
        info['issues'].append("No GPUs detected")
        return info

    # Check each GPU
    for i in range(gpu_count):
        try:
            props = torch.cuda.get_device_properties(i)
            gpu_info = {
                'id': i,
                'name': props.name,
                'memory_gb': props.total_memory / 1024**3,
                'accessible': False
            }

            # Test access
            torch.cuda.set_device(i)
            test_tensor = torch.randn(10, 10).cuda()
            gpu_info['accessible'] = True
            del test_tensor
            torch.cuda.empty_cache()

            info['gpus'].append(gpu_info)
        except Exception as e:
            gpu_info['accessible'] = False
            gpu_info['error'] = str(e)
            info['gpus'].append(gpu_info)
            info['issues'].append(f"GPU {i} not accessible: {e}")

    # Check if at least 2 GPUs are accessible
    accessible_gpus = sum(1 for g in info['gpus'] if g['accessible'])

    if accessible_gpus < 2:
        info['issues'].append(f"Only {accessible_gpus} GPU(s) accessible, need 2+ for multi-GPU")
    else:
        info['ready'] = True

    return info


if __name__ == "__main__":
    """Test multi-GPU processor"""
    print("Checking multi-GPU readiness...")
    readiness = check_multi_gpu_readiness()

    print(f"\nCUDA Available: {readiness['cuda_available']}")
    print(f"GPU Count: {readiness['gpu_count']}")
    print(f"Ready for Multi-GPU: {readiness['ready']}")

    if readiness['gpus']:
        print("\nGPUs:")
        for gpu in readiness['gpus']:
            status = "✅" if gpu['accessible'] else "❌"
            print(f"  {status} GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")

    if readiness['issues']:
        print("\nIssues:")
        for issue in readiness['issues']:
            print(f"  ⚠️  {issue}")

    if readiness['ready']:
        print("\n✅ System is READY for multi-GPU processing!")
    else:
        print("\n❌ System is NOT ready for multi-GPU processing")
