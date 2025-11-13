# Multi-GPU Processing Guide for ClinOrchestra v1.0.0

**Enable True Multi-GPU Parallelism for Local Model Inference**

This guide explains why ClinOrchestra's default parallel processing doesn't use multiple GPUs and how to enable true multi-GPU support for your H100 cluster.

---

## The Problem

### Current Behavior (Default `parallel_processor.py`)

**What happens:**
1. Uses `ThreadPoolExecutor` for parallelism
2. Loads ONE model instance on ONE GPU (cuda:0)
3. All threads share the same model
4. **Result**: Only GPU 0 is utilized, other GPUs sit idle

**Why this happens:**
```python
# File: core/llm_manager.py (line 219-222)
self.device = torch.device("cuda")  # Defaults to cuda:0
self.model.to(self.device)           # Model loaded on GPU 0 only

# File: core/parallel_processor.py (line 208)
with ThreadPoolExecutor(max_workers=N):  # Threads share model
```

**ThreadPoolExecutor = CPU threading**, NOT GPU parallelism
- Perfect for I/O-bound tasks (API calls to OpenAI, Anthropic, etc.)
- **Does NOT distribute work across multiple GPUs**

### How to Verify

Run this while processing:
```bash
watch -n 0.5 nvidia-smi
```

You'll see:
- **GPU 0**: 90-100% utilization ✅
- **GPU 1**: 0-10% utilization ❌ (idle!)

---

## The Solution

Use **`MultiGPUProcessor`** with **ProcessPoolExecutor**:
- Spawns separate processes (not threads)
- Each process loads model on assigned GPU
- True parallel GPU utilization

---

## Quick Test: Diagnose GPU Usage

First, confirm the issue:

```bash
cd /home/user/clinorchestra
python scripts/diagnose_gpu_usage.py
```

This will:
1. Check GPU availability
2. Simulate current behavior (single GPU)
3. Explain what multi-GPU should do
4. Provide recommendations

---

## Option 1: Check Multi-GPU Readiness

```bash
cd /home/user/clinorchestra
python -c "from core.multi_gpu_processor import check_multi_gpu_readiness; import json; print(json.dumps(check_multi_gpu_readiness(), indent=2))"
```

Expected output for H100 cluster with 2 GPUs:
```json
{
  "ready": true,
  "cuda_available": true,
  "gpu_count": 2,
  "gpus": [
    {"id": 0, "name": "NVIDIA H100", "memory_gb": 80.0, "accessible": true},
    {"id": 1, "name": "NVIDIA H100", "memory_gb": 80.0, "accessible": true}
  ],
  "issues": []
}
```

---

## Option 2: Enable Multi-GPU in SDK

### Example: Process Dataset with 2 GPUs

```python
import pandas as pd
from core.app_state import AppState
from core.multi_gpu_processor import MultiGPUProcessor, MultiGPUTask

# Load dataset
df = pd.read_csv('patients.csv')

# Setup app_state (configure model, prompts, etc.)
app_state = AppState()

app_state.set_model_config({
    'provider': 'local',
    'model_name': 'unsloth/Phi-4',
    'temperature': 0.01,
    'max_tokens': 4096,
    'quantization': '4bit',
    'gpu_layers': -1,  # Use all GPU layers
    'max_seq_length': 16384
})

app_state.set_prompt_config({
    'main_prompt': "Extract clinical information...",
    'json_schema': {...}
})

# Create multi-GPU processor
processor = MultiGPUProcessor(
    num_gpus=2,  # Use both H100 GPUs
    app_state=app_state
)

# Create tasks
tasks = []
for idx, row in df.iterrows():
    task = MultiGPUTask(
        task_id=idx,
        row_index=idx,
        clinical_text=row['clinical_text'],
        label_value=row.get('diagnosis_id'),
        prompt_variables={
            'age': row.get('age'),
            'gender': row.get('gender')
        }
    )
    tasks.append(task)

# Process with progress callback
def progress_callback(completed, total):
    print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")

results = processor.process_batch(tasks, progress_callback=progress_callback)

# Extract successful results
successful_results = [r.result for r in results if r.success]
print(f"\n✅ Completed: {len(successful_results)}/{len(results)}")

# Check GPU distribution
gpu_usage = {}
for result in results:
    gpu_id = result.gpu_id
    gpu_usage[gpu_id] = gpu_usage.get(gpu_id, 0) + 1

print(f"\nGPU Distribution:")
for gpu_id, count in sorted(gpu_usage.items()):
    print(f"  GPU {gpu_id}: {count} tasks ({count/len(results)*100:.1f}%)")
```

---

## Option 3: Quick Workaround (No Code Changes)

If you just want to utilize both GPUs NOW without code changes:

### Run Two Separate Instances

**Terminal 1 (GPU 0):**
```bash
export CUDA_VISIBLE_DEVICES=0
clinorchestra
# Process first half of dataset
```

**Terminal 2 (GPU 1):**
```bash
export CUDA_VISIBLE_DEVICES=1
clinorchestra
# Process second half of dataset
```

**Steps:**
1. Split your CSV into two files: `patients_part1.csv`, `patients_part2.csv`
2. Terminal 1: Process part1 → outputs to `results_part1/`
3. Terminal 2: Process part2 → outputs to `results_part2/`
4. Combine results: `cat results_part1/*.json results_part2/*.json > all_results.json`

**Advantages:**
- No code changes needed
- Works immediately
- Each instance uses one GPU

**Disadvantages:**
- Manual splitting required
- Need to merge results
- Less elegant than true multi-GPU processor

---

## Option 4: Integration with UI (Future)

To add multi-GPU toggle to the web UI, you would need to:

1. Add toggle in `ui/config_tab.py`:
```python
use_multi_gpu = gr.Checkbox(
    label="Enable Multi-GPU Processing (Local Models Only)",
    value=False,
    info="Use multiple GPUs for parallel processing"
)
num_gpus = gr.Slider(
    minimum=1,
    maximum=torch.cuda.device_count() if torch.cuda.is_available() else 1,
    value=2,
    step=1,
    label="Number of GPUs to Use"
)
```

2. Modify `ui/processing_tab.py` to use `MultiGPUProcessor` when enabled:
```python
if app_state.optimization_config.use_multi_gpu and provider == 'local':
    from core.multi_gpu_processor import MultiGPUProcessor
    processor = MultiGPUProcessor(num_gpus=num_gpus, app_state=app_state)
else:
    # Use standard parallel processor
    processor = ParallelProcessor(max_workers=workers, provider=provider)
```

---

## Performance Comparison

### Single GPU (Current Default)
```
Dataset: 1000 patients
Model: Phi-4 (4-bit quantized)
Hardware: 1x H100

Sequential: ~1.5s per patient = 25 minutes total
Parallel (ThreadPoolExecutor, 2 workers): ~1.5s per patient = 12.5 minutes total
  → Both workers queue on GPU 0, ~2x speedup from better GPU utilization

GPU Utilization:
  GPU 0: 90-100% ✅
  GPU 1: 0-5% ❌ (wasted)
```

### Multi-GPU (with MultiGPUProcessor)
```
Dataset: 1000 patients
Model: Phi-4 (4-bit quantized)
Hardware: 2x H100

Multi-GPU (ProcessPoolExecutor, 2 workers): ~0.75s per patient = 6.25 minutes total
  → Each worker on separate GPU, true parallel execution

GPU Utilization:
  GPU 0: 90-100% ✅
  GPU 1: 90-100% ✅ (fully utilized!)

Speedup: 4x faster than sequential, 2x faster than single-GPU parallel
```

---

## Limitations and Considerations

### 1. Memory Requirements
- Each process loads a FULL model copy on its GPU
- For Phi-4 4-bit: ~4GB per GPU
- For Llama-3.1 70B 4-bit: ~40GB per GPU
- **Ensure each GPU has enough VRAM**

### 2. Process Overhead
- ProcessPoolExecutor has higher overhead than ThreadPoolExecutor
- Model loading happens ONCE per process (amortized over many tasks)
- **Worth it for batches >10 rows**

### 3. Only for Local Models
- **DO NOT use MultiGPUProcessor for cloud APIs** (OpenAI, Anthropic, etc.)
- Cloud APIs are I/O-bound → ThreadPoolExecutor is perfect
- Use MultiGPUProcessor ONLY when `provider='local'`

### 4. Serialization
- AppState and tasks must be picklable
- Complex objects may need special handling

---

## Troubleshooting

### Issue: "RuntimeError: CUDA out of memory"

**Solution:**
```python
# Reduce num_gpus or use smaller batches
processor = MultiGPUProcessor(num_gpus=1, app_state=app_state)

# Or use smaller quantization
app_state.set_model_config({
    'quantization': '4bit',  # Instead of '8bit' or None
    'gpu_layers': 20  # Instead of -1 (all layers)
})
```

### Issue: "GPU 1 not accessible"

**Check:**
```bash
# Ensure both GPUs are visible
nvidia-smi

# Check CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES  # Should be empty or "0,1"

# Test GPU access
python -c "import torch; print(torch.cuda.device_count())"  # Should be 2
```

### Issue: Processes hang or deadlock

**Solution:**
```python
# Use spawn method for multiprocessing (safer for CUDA)
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
```

---

## Monitoring GPU Usage

### Real-time monitoring:
```bash
watch -n 0.5 nvidia-smi
```

### Detailed GPU metrics:
```bash
nvidia-smi dmon -s u -d 1
```

### Per-process GPU usage:
```bash
nvidia-smi pmon -d 1
```

### Expected output during multi-GPU processing:
```
Every 0.5s: nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name                     | Bus-Id        Mem-Usage | Volatile Uncorr. ECC |
|===============================+======================+======================|
|   0  NVIDIA H100              | 00000000:17:00.0  45% |   95°C    P0    250W |
|                               |                  8GB / 80GB |         N/A          |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA H100              | 00000000:65:00.0  42% |   94°C    P0    245W |
|                               |                  8GB / 80GB |         N/A          |
+-------------------------------+----------------------+----------------------+

Both GPUs actively processing! ✅
```

---

## Summary

**Current Default Behavior:**
- ❌ ThreadPoolExecutor with shared model on GPU 0
- ❌ Only 1 GPU utilized (others idle)
- ✅ Perfect for cloud APIs

**Multi-GPU Solution:**
- ✅ ProcessPoolExecutor with per-GPU model instances
- ✅ All GPUs fully utilized
- ✅ 2-4x faster for local models
- ❌ Only for local models (not cloud APIs)

**To Enable:**
1. Use `MultiGPUProcessor` in SDK (Option 2)
2. OR run separate instances per GPU (Option 3)
3. OR integrate into UI (Option 4 - requires code changes)

---

## Next Steps

1. **Verify the issue:**
   ```bash
   python scripts/diagnose_gpu_usage.py
   ```

2. **Check readiness:**
   ```bash
   python core/multi_gpu_processor.py
   ```

3. **Test multi-GPU processing:**
   ```python
   # Use SDK example from Option 2 above
   ```

4. **Monitor GPU usage:**
   ```bash
   watch -n 0.5 nvidia-smi
   ```

---

**ClinOrchestra v1.0.0** - Multi-GPU Support for H100 Clusters

For standard parallel processing (cloud APIs), see [SDK_GUIDE.md](SDK_GUIDE.md)
