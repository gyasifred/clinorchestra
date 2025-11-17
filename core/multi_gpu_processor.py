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
class MultiGPUConfig:
    """Serializable configuration for multi-GPU workers"""
    # Model configuration (from ModelConfig)
    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    max_seq_length: int
    model_type: str
    api_key: Optional[str]
    azure_endpoint: Optional[str]
    azure_deployment: Optional[str]
    google_project_id: Optional[str]
    local_model_path: Optional[str]
    quantization: Optional[str]
    gpu_layers: int

    # Paths
    extras_path: str
    functions_path: str
    patterns_path: str

    # RAG configuration
    rag_enabled: bool
    rag_top_k: int
    rag_documents_path: Optional[str]

    # Prompt configuration
    main_prompt: str
    minimal_prompt: str
    rag_refinement_prompt: str
    json_schema: Dict[str, Any]

    # Data configuration
    enable_phi_redaction: bool
    enable_pattern_normalization: bool
    phi_entity_types: List[str]

    # Agentic configuration
    agentic_enabled: bool
    max_iterations: int
    max_tool_calls: int


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

        # Create serializable config from app_state
        self.config = self._create_serializable_config(app_state)

    def _create_serializable_config(self, app_state) -> MultiGPUConfig:
        """
        Extract serializable configuration from app_state

        Args:
            app_state: AppState instance

        Returns:
            MultiGPUConfig with serializable data
        """
        import json

        # Get paths
        from pathlib import Path
        base_path = Path.cwd()
        extras_path = str(base_path / "extras")
        functions_path = str(base_path / "functions")
        patterns_path = str(base_path / "patterns")

        # Get RAG documents path if available
        rag_documents_path = None
        if app_state.rag_config.enabled and hasattr(app_state.rag_config, 'documents'):
            rag_documents_path = app_state.rag_config.documents

        # Get rag_refinement_prompt - handle both rag_prompt and rag_refinement_prompt
        rag_refinement_prompt = getattr(app_state.prompt_config, 'rag_refinement_prompt',
                                       getattr(app_state.prompt_config, 'rag_prompt', ''))

        return MultiGPUConfig(
            # Model config - use actual ModelConfig attributes
            provider=app_state.model_config.provider,
            model_name=app_state.model_config.model_name,
            temperature=app_state.model_config.temperature,
            max_tokens=app_state.model_config.max_tokens,
            max_seq_length=getattr(app_state.model_config, 'max_seq_length', 16384),
            model_type=getattr(app_state.model_config, 'model_type', 'chat'),
            api_key=getattr(app_state.model_config, 'api_key', None),
            azure_endpoint=getattr(app_state.model_config, 'azure_endpoint', None),
            azure_deployment=getattr(app_state.model_config, 'azure_deployment', None),
            google_project_id=getattr(app_state.model_config, 'google_project_id', None),
            local_model_path=getattr(app_state.model_config, 'local_model_path', None),
            quantization=getattr(app_state.model_config, 'quantization', '4bit'),
            gpu_layers=getattr(app_state.model_config, 'gpu_layers', -1),

            # Paths
            extras_path=extras_path,
            functions_path=functions_path,
            patterns_path=patterns_path,

            # RAG config
            rag_enabled=app_state.rag_config.enabled,
            rag_top_k=app_state.rag_config.rag_top_k,
            rag_documents_path=rag_documents_path,

            # Prompt config
            main_prompt=app_state.prompt_config.main_prompt or "",
            minimal_prompt=app_state.prompt_config.minimal_prompt or "",
            rag_refinement_prompt=rag_refinement_prompt,
            json_schema=json.loads(app_state.prompt_config.json_schema) if isinstance(app_state.prompt_config.json_schema, str) else (app_state.prompt_config.json_schema or {}),

            # Data config
            enable_phi_redaction=app_state.data_config.enable_phi_redaction,
            enable_pattern_normalization=app_state.data_config.enable_pattern_normalization,
            phi_entity_types=app_state.data_config.phi_entity_types or [],

            # Agentic config
            agentic_enabled=app_state.agentic_config.enabled,
            max_iterations=app_state.agentic_config.max_iterations,
            max_tool_calls=app_state.agentic_config.max_tool_calls
        )

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

        # CRITICAL FIX: Use 'spawn' method for CUDA compatibility
        import multiprocessing as mp
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, that's fine
            pass

        # Assign GPUs to tasks (round-robin)
        for i, task in enumerate(tasks):
            task.gpu_id = self.gpu_ids[i % self.num_gpus]
            task.task_id = i

        total_tasks = len(tasks)
        print(f"[START] Processing {total_tasks} tasks across {self.num_gpus} GPUs")
        print(f"[INFO] Using serializable config for worker processes")

        # FIXED: Use Manager.dict for results - no locks needed
        manager = Manager()
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
            # Submit tasks - FIXED: Only pass task and config
            futures = {}
            submit_errors = 0
            for i, task in enumerate(tasks):
                try:
                    if i == 0:
                        print(f"[DEBUG] Submitting first task...")
                    future = executor.submit(
                        _process_single_task_on_gpu,
                        task,
                        self.config  # Only pass task and config
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
                        progress_callback(completed_tasks, total_tasks)

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
                                  config: MultiGPUConfig) -> MultiGPUResult:
    """
    Process single task on assigned GPU
    This function runs in a separate process
    
    FIXED: Removed shared state parameters (completed_count, total_tasks)

    Args:
        task: Task to process
        config: Serializable configuration

    Returns:
        Result of processing
    """
    import sys
    import io
    from contextlib import redirect_stdout, redirect_stderr

    # Capture all output from this worker
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    start_time = time.time()
    gpu_id = task.gpu_id

    # Redirect stdout/stderr to capture worker output
    success = False
    result_obj = None
    error_msg = None
    duration = 0.0

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            print(f"[WORKER-{gpu_id}] Starting task {task.task_id}")

            # Set GPU for this process
            torch.cuda.set_device(gpu_id)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            print(f"[WORKER-{gpu_id}] GPU set to {gpu_id}")

            # Import required modules
            from core.llm_manager import LLMManager
            from core.regex_preprocessor import RegexPreprocessor
            from core.extras_manager import ExtrasManager
            from core.function_registry import FunctionRegistry
            from core.rag_engine import RAGEngine
            from core.agent_factory import create_agent
            print(f"[WORKER-{gpu_id}] Imported all required modules")

            # CRITICAL: Recreate all managers from serializable config
            print(f"[WORKER-{gpu_id}] Creating minimal app_state from config...")

            # Import config classes
            from core.app_state import PromptConfig, DataConfig, RAGConfig, AgenticConfig
            from core.model_config import ModelConfig

            # Create config objects from serialized data
            model_cfg = ModelConfig(
                provider=config.provider,
                model_name=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                max_seq_length=config.max_seq_length,
                model_type=config.model_type,
                api_key=config.api_key,
                azure_endpoint=config.azure_endpoint,
                azure_deployment=config.azure_deployment,
                google_project_id=config.google_project_id,
                local_model_path=config.local_model_path,
                quantization=config.quantization,
                gpu_layers=config.gpu_layers
            )

            prompt_cfg = PromptConfig(
                main_prompt=config.main_prompt,
                minimal_prompt=config.minimal_prompt,
                rag_prompt=config.rag_refinement_prompt,
                json_schema=config.json_schema
            )

            data_cfg = DataConfig(
                enable_phi_redaction=config.enable_phi_redaction,
                enable_pattern_normalization=config.enable_pattern_normalization,
                phi_entity_types=config.phi_entity_types
            )

            rag_cfg = RAGConfig(
                enabled=config.rag_enabled,
                k_value=config.rag_top_k
            )

            agentic_cfg = AgenticConfig(
                enabled=config.agentic_enabled,
                max_iterations=config.max_iterations,
                max_tool_calls=config.max_tool_calls
            )

            # Create mock app_state with config objects
            mock_app_state = type('MockAppState', (), {
                'model_config': model_cfg,
                'prompt_config': prompt_cfg,
                'data_config': data_cfg,
                'rag_config': rag_cfg,
                'agentic_config': agentic_cfg
            })()

            # Initialize LLM manager for this process
            print(f"[WORKER-{gpu_id}] Creating LLM manager...")
            from dataclasses import asdict
            llm_manager = LLMManager(config=asdict(model_cfg))
            print(f"[WORKER-{gpu_id}] LLM manager created")

            # Initialize regex preprocessor
            print(f"[WORKER-{gpu_id}] Creating regex preprocessor...")
            regex_preprocessor = RegexPreprocessor(storage_path=config.patterns_path)
            print(f"[WORKER-{gpu_id}] Regex preprocessor created")

            # Initialize extras manager
            print(f"[WORKER-{gpu_id}] Creating extras manager...")
            extras_manager = ExtrasManager(storage_path=config.extras_path)
            print(f"[WORKER-{gpu_id}] Extras manager created")

            # Initialize function registry
            print(f"[WORKER-{gpu_id}] Creating function registry...")
            function_registry = FunctionRegistry(storage_path=config.functions_path)
            print(f"[WORKER-{gpu_id}] Function registry created")

            # Initialize RAG engine if enabled
            rag_engine = None
            if config.rag_enabled and config.rag_documents_path:
                print(f"[WORKER-{gpu_id}] Creating RAG engine...")
                rag_engine = RAGEngine(config.rag_top_k)
                print(f"[WORKER-{gpu_id}] RAG engine created")

            # Create agent for this process
            print(f"[WORKER-{gpu_id}] Creating agent...")
            agent = create_agent(
                llm_manager=llm_manager,
                rag_engine=rag_engine,
                extras_manager=extras_manager,
                function_registry=function_registry,
                regex_preprocessor=regex_preprocessor,
                app_state=mock_app_state
            )
            print(f"[WORKER-{gpu_id}] Agent created successfully")

            # Ensure model is on correct device
            if hasattr(agent, 'llm_manager') and hasattr(agent.llm_manager, 'model'):
                if agent.llm_manager.model is not None:
                    print(f"[WORKER-{gpu_id}] Moving model to GPU {gpu_id}...")
                    agent.llm_manager.model.to(f'cuda:{gpu_id}')
                    print(f"[WORKER-{gpu_id}] Model moved to GPU {gpu_id}")

            # Process the extraction
            print(f"[WORKER-{gpu_id}] Running extraction...")
            result_obj = agent.extract(
                clinical_text=task.clinical_text,
                label_value=task.label_value,
                prompt_variables=task.prompt_variables
            )
            print(f"[WORKER-{gpu_id}] Extraction complete")

            duration = time.time() - start_time
            success = True

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        success = False

    # NOW we're outside the redirect context - output will actually show
    worker_output = stdout_capture.getvalue()
    worker_errors = stderr_capture.getvalue()

    if success:
        # Print success output to main process
        if worker_output:
            sys.__stdout__.write(f"\n{'='*80}\n[WORKER-{gpu_id} OUTPUT - Task {task.task_id}]\n{'='*80}\n{worker_output}\n")
            sys.__stdout__.flush()
        if worker_errors:
            sys.__stderr__.write(f"\n[WORKER-{gpu_id} ERRORS]\n{worker_errors}\n")
            sys.__stderr__.flush()

        return MultiGPUResult(
            task_id=task.task_id,
            row_index=task.row_index,
            gpu_id=gpu_id,
            success=True,
            result=result_obj,
            duration=duration
        )
    else:
        # Print failure output to main process
        sys.__stdout__.write(f"\n{'='*80}\n[WORKER-{gpu_id}] FAILED Task {task.task_id}: {error_msg}\n{'='*80}\n")
        sys.__stdout__.flush()
        if worker_output:
            sys.__stdout__.write(f"[WORKER-{gpu_id} OUTPUT]\n{worker_output}\n")
            sys.__stdout__.flush()
        if worker_errors:
            sys.__stderr__.write(f"[WORKER-{gpu_id} ERRORS]\n{worker_errors}\n")
            sys.__stderr__.flush()

        return MultiGPUResult(
            task_id=task.task_id,
            row_index=task.row_index,
            gpu_id=gpu_id,
            success=False,
            error=f"{error_msg}\n\nCaptured Output:\n{worker_output}\n\nCaptured Errors:\n{worker_errors}",
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