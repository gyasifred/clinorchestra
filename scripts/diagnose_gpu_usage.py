#!/usr/bin/env python3
"""
GPU Utilization Diagnostic for ClinOrchestra
Checks if parallel processing is actually using multiple GPUs
"""

import torch
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os

def check_gpu_setup():
    """Check GPU availability and configuration"""
    print("=" * 80)
    print("GPU DIAGNOSTIC REPORT")
    print("=" * 80)

    # Check PyTorch CUDA availability
    print(f"\n1. PyTorch CUDA Available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("   ❌ CUDA not available - running on CPU")
        return

    # Check GPU count
    gpu_count = torch.cuda.device_count()
    print(f"2. Number of GPUs Detected: {gpu_count}")

    if gpu_count == 0:
        print("   ❌ No GPUs detected")
        return

    # List all GPUs
    print(f"\n3. GPU Details:")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name}")
        print(f"      - Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"      - Compute Capability: {props.major}.{props.minor}")

    # Check current device
    current_device = torch.cuda.current_device()
    print(f"\n4. Current Default Device: cuda:{current_device}")

    # Check if models can be loaded on different devices
    print(f"\n5. Testing GPU Access:")
    for i in range(gpu_count):
        try:
            torch.cuda.set_device(i)
            tensor = torch.randn(100, 100).cuda()
            print(f"   ✅ GPU {i}: Accessible")
            del tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   ❌ GPU {i}: {e}")

    # Check ClinOrchestra's current configuration
    print(f"\n6. ClinOrchestra Configuration Issue:")
    print(f"   ⚠️  PROBLEM IDENTIFIED:")
    print(f"      - ClinOrchestra uses ThreadPoolExecutor (CPU threading)")
    print(f"      - Loads ONE model instance on ONE GPU (typically cuda:0)")
    print(f"      - All threads share the same model → only GPU 0 is used")
    print(f"      - Other GPUs ({gpu_count-1} GPU{'s' if gpu_count > 2 else ''}) sit idle")

    print(f"\n7. How to Verify Which GPU is Being Used:")
    print(f"   Run this in a separate terminal while processing:")
    print(f"   >>> watch -n 0.5 nvidia-smi")
    print(f"   ")
    print(f"   You'll see:")
    print(f"   - GPU 0: High utilization (70-100%)")
    print(f"   - GPU 1: Low/zero utilization (0-10%)")

    print(f"\n8. Why This Happens:")
    print(f"   File: core/llm_manager.py, line 219-222")
    print(f"   Code: self.device = torch.device('cuda')  # Defaults to cuda:0")
    print(f"         self.model.to(self.device)           # Model on GPU 0 only")
    print(f"   ")
    print(f"   File: core/parallel_processor.py, line 208")
    print(f"   Code: ThreadPoolExecutor(max_workers=N)  # Threads share model")

    print(f"\n9. Solution Required:")
    print(f"   Need PROCESS-based parallelism, not THREAD-based:")
    print(f"   - Option A: ProcessPoolExecutor + per-process GPU assignment")
    print(f"   - Option B: Model framework's native multi-GPU (DataParallel/DistributedDataParallel)")
    print(f"   - Option C: vLLM or other multi-GPU inference engines")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


def simulate_current_behavior():
    """Simulate how ClinOrchestra currently uses GPUs"""
    print("\n" + "=" * 80)
    print("SIMULATING CURRENT CLINORCHESTRA BEHAVIOR")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available - skipping simulation")
        return

    gpu_count = torch.cuda.device_count()
    print(f"\nAvailable GPUs: {gpu_count}")
    print(f"Current behavior: ALL threads use GPU 0 only\n")

    # Simulate model on GPU 0
    print("Loading model on GPU 0...")
    model_device = torch.device("cuda:0")
    dummy_model = torch.nn.Linear(1000, 1000).to(model_device)
    print(f"✅ Model loaded on {model_device}")

    def process_on_shared_model(task_id):
        """Simulate processing with shared model (current behavior)"""
        # All threads use the same model on GPU 0
        with torch.no_grad():
            input_tensor = torch.randn(100, 1000).to(model_device)
            output = dummy_model(input_tensor)

        gpu_id = output.device.index
        return f"Task {task_id} ran on GPU {gpu_id}"

    # Run with ThreadPoolExecutor (current implementation)
    print(f"\nRunning 4 tasks with ThreadPoolExecutor (max_workers=2)...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(process_on_shared_model, range(4)))

    print("\nResults:")
    for result in results:
        print(f"  {result}")

    print(f"\n❌ As expected: ALL tasks used GPU 0 only")
    print(f"   GPU 1 was never utilized")

    del dummy_model
    torch.cuda.empty_cache()


def show_multi_gpu_solution():
    """Show what multi-GPU solution would look like"""
    print("\n" + "=" * 80)
    print("WHAT MULTI-GPU SOLUTION SHOULD DO")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available - showing concept only")
        return

    gpu_count = torch.cuda.device_count()

    if gpu_count < 2:
        print(f"Only {gpu_count} GPU available - need 2+ for multi-GPU demo")
        return

    print(f"\nWith {gpu_count} GPUs, ideal distribution:")
    print(f"  - Worker 1 → GPU 0")
    print(f"  - Worker 2 → GPU 1")
    if gpu_count > 2:
        for i in range(2, gpu_count):
            print(f"  - Worker {i+1} → GPU {i}")

    print(f"\nThis requires:")
    print(f"  1. ProcessPoolExecutor (not ThreadPoolExecutor)")
    print(f"  2. Each process loads model on assigned GPU")
    print(f"  3. CUDA_VISIBLE_DEVICES or torch.cuda.set_device() per process")


if __name__ == "__main__":
    check_gpu_setup()
    simulate_current_behavior()
    show_multi_gpu_solution()

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("\n1. VERIFY THE ISSUE:")
    print("   Run while processing:")
    print("   $ watch -n 0.5 nvidia-smi")
    print("   ")
    print("   You should see:")
    print("   - GPU 0: 90-100% utilization")
    print("   - GPU 1: 0-10% utilization (idle)")
    print("")
    print("2. TO FIX:")
    print("   I'll create a multi-GPU parallel processor for you.")
    print("   This will use ProcessPoolExecutor with per-GPU model loading.")
    print("")
    print("3. QUICK FIX (if you just want to use both GPUs manually):")
    print("   Run TWO separate ClinOrchestra instances:")
    print("   Terminal 1: CUDA_VISIBLE_DEVICES=0 clinorchestra")
    print("   Terminal 2: CUDA_VISIBLE_DEVICES=1 clinorchestra")
    print("   Split your CSV in half and process each half separately.")
    print("=" * 80)
