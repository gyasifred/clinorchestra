# GPU Resource Isolation Fixes

## Problem Summary

When starting the application with multiple GPUs available, resources were being allocated on GPU 1 even when not needed, resulting in wasted VRAM (~600MB per process on secondary GPUs).

Example symptom:
```
GPU 0: ~46GB (main models + FAISS)
GPU 1: ~1.2GB (unwanted allocation)
```

## Root Causes Identified

### 1. FAISS GPU Device Hardcoded to 0
**Location:** `core/rag_engine.py:392, 405`

The GPU device ID was hardcoded to `0` in `faiss.index_cpu_to_gpu()` calls:
```python
# Before (WRONG)
gpu_test_index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, test_index)
self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
```

This meant FAISS would ALWAYS use GPU 0, regardless of which GPU was assigned to a worker process.

### 2. SentenceTransformer Auto-GPU Detection
**Location:** `core/rag_engine.py:243`

The embedding model was loaded without specifying a device:
```python
# Before (WRONG)
self.model = SentenceTransformer(self.model_name)
```

SentenceTransformer would auto-detect CUDA and potentially allocate memory across multiple GPUs.

### 3. CUDA_VISIBLE_DEVICES Ineffective Timing
**Location:** `core/multi_gpu_processor.py:405`

`CUDA_VISIBLE_DEVICES` was being set, but with incorrect device mapping logic for multi-GPU workers.

## Fixes Implemented

### Fix 1: Parameterize FAISS GPU Device

**File:** `core/rag_engine.py`

Added `gpu_device` parameter to `VectorStore.__init__()`:
```python
def __init__(self, embedding_generator: EmbeddingGenerator, cache_db_path: str,
             use_gpu: bool = False, gpu_device: int = 0):
    self.gpu_device = gpu_device  # Which GPU to use (0, 1, 2, etc.)
```

Updated FAISS calls to use the specified device:
```python
gpu_test_index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, test_index)
self.index = faiss.index_cpu_to_gpu(self.gpu_resources, self.gpu_device, self.index)
```

### Fix 2: Explicit SentenceTransformer Device Placement

**File:** `core/rag_engine.py`

Added `device` parameter to `EmbeddingGenerator.__init__()`:
```python
def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2",
             device: Optional[str] = None):
    self.device = device  # e.g., "cuda:0", "cuda:1", "cpu", or None
```

Set device explicitly when loading model:
```python
if self.device:
    self.model = SentenceTransformer(self.model_name, device=self.device)
else:
    self.model = SentenceTransformer(self.model_name)
```

### Fix 3: Proper GPU Device Configuration in RAGEngine

**File:** `core/rag_engine.py`

Added `gpu_device` to `RAGEngine.__init__()`:
```python
self.gpu_device = config.get('gpu_device', 0)  # Which GPU to use
```

Compute proper device string for embedding model:
```python
embedding_device = None
if self.use_gpu_faiss or (app_state.optimization_config.use_gpu_faiss):
    if torch.cuda.is_available():
        embedding_device = f"cuda:{self.gpu_device}"
```

Pass both parameters to components:
```python
self.embedding_generator = EmbeddingGenerator(self.embedding_model, device=embedding_device)
self.vector_store = VectorStore(self.embedding_generator, cache_path,
                                use_gpu=use_gpu, gpu_device=self.gpu_device)
```

### Fix 4: Correct Multi-GPU Worker Device Mapping

**File:** `core/multi_gpu_processor.py`

**Key Insight:** When using `CUDA_VISIBLE_DEVICES=N` in a worker process, that GPU appears as `cuda:0` from the worker's perspective.

Updated RAGEngine config in worker:
```python
rag_engine = RAGEngine({
    'gpu_device': 0  # Always 0 because CUDA_VISIBLE_DEVICES makes assigned GPU appear as device 0
})
```

Fixed model device placement:
```python
# Before
agent.llm_manager.model.to(f'cuda:{gpu_id}')  # WRONG - gpu_id is physical device

# After
agent.llm_manager.model.to('cuda:0')  # CORRECT - CUDA_VISIBLE_DEVICES makes it device 0
```

Updated CUDA device setting with clear comments:
```python
# Set CUDA_VISIBLE_DEVICES to restrict this process to assigned GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

# From this process's perspective, the assigned GPU is now device 0
torch.cuda.set_device(0)
```

## Impact

### Before
- FAISS: Always GPU 0 (hardcoded)
- SentenceTransformer: Auto-detect (could use any GPU)
- Multi-GPU workers: Incorrect device mapping
- Result: Resources leaked to unintended GPUs

### After
- FAISS: Uses specified `gpu_device`
- SentenceTransformer: Uses specified device (`cuda:N`)
- Multi-GPU workers: Correctly map to local device 0
- Result: Perfect GPU isolation per worker

## Testing

Run the validation script:
```bash
python test_gpu_isolation.py
```

This tests:
1. ✅ EmbeddingGenerator accepts and uses `device` parameter
2. ✅ VectorStore accepts and uses `gpu_device` parameter
3. ✅ RAGEngine passes `gpu_device` to components
4. ✅ No resource leaks to other GPUs

## Backward Compatibility

All changes are **fully backward compatible**:
- New parameters have default values (`gpu_device=0`, `device=None`)
- Existing code without these parameters continues to work
- Default behavior unchanged (GPU 0 for single-GPU systems)

## Files Modified

1. `core/rag_engine.py` - Added device parameters to EmbeddingGenerator, VectorStore, RAGEngine
2. `core/multi_gpu_processor.py` - Fixed device mapping in multi-GPU workers
3. `test_gpu_isolation.py` - Validation test script (can be removed after testing)

## Verification

To verify the fix is working:
```bash
# Monitor GPU usage while running
watch -n 0.5 nvidia-smi

# You should see:
# - Only GPU 0 with significant memory (if single-GPU mode)
# - Each worker using its assigned GPU (if multi-GPU mode)
# - No unexpected allocations on other GPUs
```

## Related Issues

This fix resolves the issue where starting the application would allocate ~600MB on GPU 1 even though all processing was on GPU 0.
