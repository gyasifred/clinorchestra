# GPU Usage and Isolation Guide

## Default Behavior

By default, ClinOrchestra restricts GPU usage to **GPU 0 only** to prevent resource leakage to other GPUs.

This happens automatically when you start the application:
```bash
python annotate.py
# Output: [GPU ISOLATION] Defaulting to GPU 0 only (set CUDA_VISIBLE_DEVICES to override)
```

## Customizing GPU Usage

### Use a Specific GPU

To use a different single GPU (e.g., GPU 1):
```bash
export CUDA_VISIBLE_DEVICES=1
python annotate.py
```

Or:
```bash
export CLINORCHESTRA_GPU_DEVICE=1
python annotate.py
```

### Use Multiple GPUs

To use multiple GPUs (e.g., GPU 0 and GPU 2):
```bash
export CUDA_VISIBLE_DEVICES=0,2
python annotate.py
```

### Use All Available GPUs

```bash
export CUDA_VISIBLE_DEVICES=""  # Empty string means use all GPUs
python annotate.py
```

Or simply don't set the variable if you want the default (GPU 0 only).

## Multi-GPU Processing

For batch processing with local models, enable multi-GPU in the Settings tab:
1. Go to **Settings → Optimizations**
2. Enable "Multi-GPU Processing"
3. Set number of GPUs to use (-1 for all available)

The system will automatically:
- Create separate worker processes for each GPU
- Assign each worker to a specific GPU using `CUDA_VISIBLE_DEVICES`
- Ensure complete GPU isolation between workers
- Prevent resource leakage

## GPU Isolation Technical Details

### When Isolation Occurs

GPU isolation is configured at **application startup** in `annotate.py` (lines 19-32), BEFORE any CUDA-capable libraries are imported. This ensures:

1. PyTorch only sees the specified GPU(s)
2. Sentence-transformers only uses the specified GPU(s)
3. FAISS GPU operations only use the specified GPU(s)
4. No CUDA contexts are created on unintended GPUs

### Architecture

```
annotate.py (line 19-32)
    ↓ Set CUDA_VISIBLE_DEVICES
    ↓ BEFORE importing torch, transformers, etc.
    ↓
Import libraries (line 34+)
    ↓ torch, sentence-transformers, faiss
    ↓ These only see restricted GPUs
    ↓
Application runs
    ↓ All GPU operations isolated
```

### Multi-GPU Worker Isolation

When using multi-GPU mode for batch processing:

```python
# Worker process for GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Physical GPU 0
torch.cuda.set_device(0)  # Appears as cuda:0 in worker

# Worker process for GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Physical GPU 1
torch.cuda.set_device(0)  # Appears as cuda:0 in worker
```

Each worker process:
1. Sets `CUDA_VISIBLE_DEVICES` to its assigned GPU
2. That GPU appears as `cuda:0` from the worker's perspective
3. Complete isolation - worker cannot see other GPUs
4. No resource leakage

## Troubleshooting GPU Leakage

If you still see GPU memory on unintended devices:

### 1. Check Environment Variables

```bash
echo $CUDA_VISIBLE_DEVICES  # Should show only intended GPU(s)
```

### 2. Restart Application

Changes to `CUDA_VISIBLE_DEVICES` only take effect at application startup. After changing the variable, fully restart the application.

### 3. Check for Pre-initialized CUDA

If you're running in a container or environment where CUDA is already initialized:
```bash
# Ensure clean start
unset CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0
python annotate.py
```

### 4. Monitor GPU Usage

```bash
watch -n 0.5 nvidia-smi
```

Expected behavior:
- **Single GPU mode**: Only GPU 0 should have memory allocated
- **Multi-GPU mode**: Only assigned GPUs should have memory allocated
- **No leakage**: Unassigned GPUs should show 0 MB usage

## RAG Engine GPU Usage

The RAG engine (FAISS + sentence-transformers) respects GPU isolation:

1. **Embeddings**: Use the GPU specified in `CUDA_VISIBLE_DEVICES`
2. **FAISS**: Uses `gpu_device` parameter (default: 0)
3. **Complete isolation**: No cross-GPU resource usage

Enable GPU FAISS in Settings → Optimizations:
- Check "Enable GPU FAISS"
- Or set `CLINORCHESTRA_ENABLE_GPU=1`

## Best Practices

### For Single-GPU Systems
```bash
# Default is optimal - no action needed
python annotate.py
```

### For Multi-GPU Systems (Development)
```bash
# Use GPU 0 only (prevents interference with other GPUs)
export CUDA_VISIBLE_DEVICES=0
python annotate.py
```

### For Multi-GPU Systems (Production)
```bash
# Use all GPUs for maximum performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
python annotate.py
# Enable multi-GPU processing in UI
```

### For Batch Processing
1. Enable "Multi-GPU Processing" in Settings
2. Set number of GPUs to use
3. System automatically isolates each GPU per worker
4. No manual CUDA_VISIBLE_DEVICES configuration needed

## Environment Variables Reference

| Variable | Purpose | Example |
|----------|---------|---------|
| `CUDA_VISIBLE_DEVICES` | Control which GPUs are visible | `0` or `0,1` or `` (all) |
| `CLINORCHESTRA_GPU_DEVICE` | Override default GPU selection | `1` or `0,2` |
| `CLINORCHESTRA_ENABLE_GPU` | Enable GPU FAISS acceleration | `1` or `true` |

## Implementation Details

### Files Modified for GPU Isolation

1. **annotate.py** (lines 19-32)
   - Sets `CUDA_VISIBLE_DEVICES` before imports
   - Prevents GPU leakage at application startup

2. **core/rag_engine.py**
   - `EmbeddingGenerator`: Accepts `device` parameter
   - `VectorStore`: Accepts `gpu_device` parameter
   - `RAGEngine`: Passes device to components

3. **core/multi_gpu_processor.py**
   - Sets `CUDA_VISIBLE_DEVICES` per worker process
   - Maps physical GPU to worker's local device 0
   - Complete process-level isolation

### How It Works

**Application Startup:**
```python
# annotate.py (line 19-32)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Before any imports!

import torch  # Only sees GPU 0
import sentence_transformers  # Only sees GPU 0
```

**Multi-GPU Worker:**
```python
# Worker process
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Physical GPU 1
# From worker's perspective, this is cuda:0
model.to('cuda:0')  # Actually uses physical GPU 1
```

This ensures complete isolation with zero resource leakage!
