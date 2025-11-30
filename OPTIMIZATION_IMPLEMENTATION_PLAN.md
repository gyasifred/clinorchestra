# OPTIMIZATION IMPLEMENTATION PLAN

**Date:** November 30, 2025
**System:** ClinOrchestra v1.0.0
**Scope:** Performance optimization implementation based on latency bottleneck analysis

---

## IMPLEMENTATION CHECKLIST

### TIER 1: Quick Wins (Target: -50% latency, -60% cost)

#### 1.1 Prompt Caching (2-3 hours)
**Status:** PENDING
**Target Impact:** -40% tokens, -20% latency, -50% cost
**Applies To:** Both STRUCTURED and ADAPTIVE workflows

**Implementation Steps:**
- [ ] Add prompt caching support to `core/llm_service.py`
- [ ] Enable caching for system prompts in STRUCTURED Stage 1
- [ ] Enable caching for tool definitions in both workflows
- [ ] Add cache control headers for Anthropic API
- [ ] Test with both workflow types
- [ ] Verify cache hit rates in logs

**Files to Modify:**
- `core/llm_service.py` - Add cache control parameters
- `core/agent_system.py` - Mark cacheable prompt sections
- `core/adaptive_agent.py` - Mark cacheable sections for ADAPTIVE

**Testing:**
- Test STRUCTURED workflow with 5 records
- Test ADAPTIVE workflow with 3 records
- Verify cache headers in API requests
- Check latency reduction

---

#### 1.2 Tiered Model Usage (1-2 hours)
**Status:** PENDING
**Target Impact:** -30% LLM time, -40% cost
**Applies To:** STRUCTURED workflow (ADAPTIVE uses configurable model)

**Implementation Steps:**
- [ ] Add stage-specific model configuration
- [ ] Stage 1: Use faster model (haiku/gpt-4o-mini) for task analysis
- [ ] Stage 3: Use accurate model (sonnet/gpt-4o) for extraction
- [ ] Stage 4: Use faster model for refinement
- [ ] Add configuration option to enable/disable tiered models
- [ ] Test accuracy vs speed tradeoff

**Files to Modify:**
- `core/agent_system.py` - Add stage-specific model selection
- `core/app_state.py` - Add tiered model configuration options

**Testing:**
- Verify Stage 1 uses fast model
- Verify Stage 3 uses accurate model
- Compare accuracy with single-model baseline
- Measure latency improvement

---

#### 1.3 Async LLM Wrapper (3-4 hours)
**Status:** PENDING
**Target Impact:** -25% latency
**Applies To:** Both STRUCTURED and ADAPTIVE workflows

**Implementation Steps:**
- [ ] Convert `llm_service.py` to async/await pattern
- [ ] Wrap OpenAI client with async
- [ ] Wrap Anthropic client with async
- [ ] Wrap Google/Azure clients with async
- [ ] Update agent_system.py to use async calls
- [ ] Update adaptive_agent.py to use async calls
- [ ] Ensure backward compatibility

**Files to Modify:**
- `core/llm_service.py` - Convert to async methods
- `core/agent_system.py` - Use async LLM calls
- `core/adaptive_agent.py` - Use async LLM calls

**Testing:**
- Test STRUCTURED workflow async execution
- Test ADAPTIVE workflow async execution
- Verify no blocking calls remain
- Measure concurrent execution improvement

---

#### 1.4 GPU FAISS Auto-Detection (1-2 hours)
**Status:** PENDING
**Target Impact:** -90% RAG time (if GPU available)
**Applies To:** Both workflows (RAG engine)

**Implementation Steps:**
- [ ] Add CUDA availability check in `rag_engine.py`
- [ ] Auto-detect GPU and create GPU index
- [ ] Fallback to CPU if GPU unavailable
- [ ] Add logging for GPU/CPU mode
- [ ] Test on CPU-only system
- [ ] Test on GPU system (if available)

**Files to Modify:**
- `core/rag_engine.py` - Add GPU detection and index creation

**Testing:**
- Verify CPU fallback works
- Test RAG query performance
- Check CUDA detection logs
- Measure speedup if GPU available

---

### TIER 2: Medium Effort (Target: Additional -20% latency)

#### 2.1 Stage 1 Plan Caching (2-3 days)
**Status:** PENDING
**Target Impact:** -20-30% Stage 1 time
**Applies To:** STRUCTURED workflow

**Implementation Steps:**
- [ ] Create schema hash function
- [ ] Build plan cache storage (JSON/pickle)
- [ ] Cache tool requests by schema+task hash
- [ ] Add cache lookup before Stage 1
- [ ] Add cache invalidation mechanism
- [ ] Add cache hit/miss metrics

**Files to Modify:**
- `core/agent_system.py` - Add plan caching logic
- `core/cache_manager.py` - Extend caching system

**Testing:**
- Test cache hit on repeated schemas
- Test cache miss on new schemas
- Verify cache invalidation works
- Measure Stage 1 speedup

---

#### 2.2 Function Optimization (1 day)
**Status:** PENDING
**Target Impact:** -40% function overhead
**Applies To:** Both workflows

**Implementation Steps:**
- [ ] Identify simple math functions (BMI, BSA, basic arithmetic)
- [ ] Add "skip_execution" flag to function definitions
- [ ] Let LLM calculate simple math directly
- [ ] Add memoization for expensive calculations
- [ ] Cache function results by parameters

**Files to Modify:**
- `core/function_registry.py` - Add skip logic and memoization
- Function JSON files - Mark simple functions

**Testing:**
- Verify simple functions skipped
- Verify complex functions executed
- Test memoization cache hits
- Measure execution time reduction

---

#### 2.3 JSON Schema Pre-Compilation (1-2 hours)
**Status:** PENDING
**Target Impact:** -55% parsing time
**Applies To:** Both workflows

**Implementation Steps:**
- [ ] Pre-compile JSON schemas at startup
- [ ] Cache validator objects
- [ ] Reuse validators across extractions
- [ ] Add validator cache invalidation

**Files to Modify:**
- `core/agent_system.py` - Pre-compile schemas
- `core/adaptive_agent.py` - Pre-compile schemas

**Testing:**
- Verify validators created once
- Test validation performance
- Measure parsing speedup

---

## TESTING PROTOCOL

### For Each Fix:

1. **Unit Testing**
   - Test isolated functionality
   - Verify no regressions
   - Check error handling

2. **Integration Testing - STRUCTURED Workflow**
   - Run with malnutrition example (5 records)
   - Measure latency per record
   - Verify output accuracy unchanged
   - Check all 4 stages work correctly

3. **Integration Testing - ADAPTIVE Workflow**
   - Run with ADRD example (3 records)
   - Measure latency per record
   - Verify autonomous tool selection works
   - Check iteration loop functions correctly

4. **Performance Metrics**
   - Record baseline latency (before fix)
   - Record optimized latency (after fix)
   - Calculate % improvement
   - Verify no accuracy degradation

### Test Datasets:

**STRUCTURED Workflow Test:**
```
Dataset: examples/malnutrition_classification/test_data_small.csv
Records: 5
Expected: All 4 stages complete, valid JSON output
Baseline: ~30s/record
Target: Progressive improvement with each fix
```

**ADAPTIVE Workflow Test:**
```
Dataset: Custom ADRD test cases
Records: 3
Expected: Autonomous tool selection, valid output
Baseline: ~35s/record
Target: Progressive improvement with each fix
```

---

## ROLLOUT STRATEGY

### Phase 1: TIER 1 Fixes (This Week)
1. Implement fix #1 → Test → Commit
2. Implement fix #2 → Test → Commit
3. Implement fix #3 → Test → Commit
4. Implement fix #4 → Test → Commit
5. Full integration test with all TIER 1 fixes
6. Measure cumulative improvement
7. Push to remote

**Expected Result:** -50% latency, -60% cost

### Phase 2: TIER 2 Fixes (Next Week)
1. Implement fix #5 → Test → Commit
2. Implement fix #6 → Test → Commit
3. Implement fix #7 → Test → Commit
4. Full integration test with TIER 1 + TIER 2
5. Measure cumulative improvement
6. Push to remote

**Expected Result:** -70% latency total

---

## SUCCESS CRITERIA

### TIER 1 Completion:
- [ ] All 4 fixes implemented
- [ ] No test failures
- [ ] Latency reduced by 40-60%
- [ ] Cost reduced by 50-70%
- [ ] Both workflows functional
- [ ] All tests pass

### TIER 2 Completion:
- [ ] All 3 fixes implemented
- [ ] Cumulative latency reduction 65-75%
- [ ] No accuracy degradation
- [ ] Cache systems working
- [ ] Both workflows optimized

---

## RISK MITIGATION

1. **Backward Compatibility**
   - Keep original code paths as fallback
   - Add feature flags for new optimizations
   - Allow disabling optimizations via config

2. **Accuracy Preservation**
   - Test output accuracy after each fix
   - Compare with baseline outputs
   - Revert if accuracy degrades >5%

3. **Error Handling**
   - Graceful fallback on optimization failures
   - Detailed error logging
   - No silent failures

---

## PERFORMANCE TRACKING

Create performance log for each test run:

```
Date: YYYY-MM-DD
Fix: [Name]
Workflow: STRUCTURED / ADAPTIVE
Records: N
Baseline Latency: Xs/record
Optimized Latency: Ys/record
Improvement: Z%
Accuracy: Maintained / Degraded
Status: PASS / FAIL
Notes: [Any observations]
```

---

## FILES TO MODIFY (Summary)

### Core Files:
- `core/llm_service.py` - Prompt caching, async wrapper
- `core/agent_system.py` - Tiered models, plan caching, schema pre-compilation
- `core/adaptive_agent.py` - Async calls, schema pre-compilation
- `core/rag_engine.py` - GPU FAISS detection
- `core/function_registry.py` - Function optimization, memoization
- `core/cache_manager.py` - Plan caching storage
- `core/app_state.py` - Configuration options

### Test Files:
- Create `tests/test_optimizations.py` - Unit tests for each fix
- Create `tests/performance_benchmark.py` - Performance testing

---

## NEXT STEPS

1. Start with TIER 1.1 (Prompt Caching)
2. Implement → Test → Commit cycle
3. Track performance improvements
4. Document any issues or deviations
5. Move to next fix when current one passes

---

**Implementation Started:** November 30, 2025
**Target Completion (TIER 1):** December 7, 2025
**Target Completion (TIER 2):** December 14, 2025
