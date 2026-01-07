# Search Strategy Enhancement - v1.0.0

## Overview
Implemented LLM-driven search strategy with term variations for improved RAG and extras retrieval. This enhancement adds "leniency" by using synonyms, abbreviations, and related terms to cast a wider net for better recall while maintaining precision.

## What Changed

### 1. **Enhanced Prompts - Search Strategy Integration**

#### Stage 1 Analysis Prompt (`agent_system.py`)
- **Added**: Search strategy guidance for RAG query building
  - Primary keywords + term variations (synonyms, abbreviations, related terms)
  - Multi-faceted query approach (guideline, diagnostic, assessment angles)
  - Leniency tactics (broader/narrower terms, alternative phrasings)

- **Added**: Search strategy guidance for extras keyword expansion
  - Core concepts + expanded variations
  - Medical abbreviations and related terminology

- **Updated**: Response format to include `term_variations` field in RAG queries

**Impact**: LLM now generates smarter queries with terminology variations

#### Gap Analysis Prompt (`agent_system.py`)
- **Added**: Search strategy guidance for gap-filling tools
  - Instructions to use DIFFERENT term variations (avoid repetition)
  - Suggestions for broader/narrower terms and alternative phrasings

**Impact**: Stage 4 refinement uses varied terminology to find new information

#### Agentic Prompt (`prompt_templates.py`)
- **Updated**: Tool descriptions to include search strategy
  - `query_rag()`: Instructions for multi-term queries with variations
  - `query_extras()`: Instructions for keyword expansion

- **Updated**: Workflow phases to emphasize term variation usage
  - Phase 2: Execute tools with search strategy
  - Phase 3: Iterate with DIFFERENT term variations

**Impact**: Adaptive pipeline uses search strategy throughout execution

---

### 2. **New Methods - RAG Engine** (`rag_engine.py`)

#### `query_with_variations(primary_query, variations, k)`
- Combines primary query with term variations
- Executes expanded query for better recall
- Deduplicates results based on content similarity
- Returns top k unique results

**Example**:
```python
primary_query = "ASPEN pediatric malnutrition diagnostic criteria"
variations = ["undernutrition", "PEM", "SAM", "wasting", "stunting"]
results = rag_engine.query_with_variations(primary_query, variations, k=10)
```

**Benefit**: Retrieves more relevant chunks by searching with varied terminology

---

### 3. **New Methods - Extras Manager** (`extras_manager.py`)

#### `match_extras_with_variations(core_keywords, variations_map, threshold, top_k)`
- Expands core keywords with variations map
- Uses existing `match_extras_by_keywords` with expanded set
- Supports optional variations mapping per keyword

**Example**:
```python
core_keywords = ["malnutrition", "pediatric"]
variations_map = {
    "malnutrition": ["undernutrition", "PEM", "SAM", "wasting"],
    "pediatric": ["child", "infant", "neonatal"]
}
results = extras_manager.match_extras_with_variations(core_keywords, variations_map)
```

**Benefit**: Finds more relevant extras by matching against expanded terminology

---

### 4. **Integration - Agent System** (`agent_system.py`)

#### Updated `_execute_rag_tool()`
- Checks for `term_variations` in tool_request
- Calls `query_with_variations()` when variations present
- Falls back to standard `query()` when no variations

**Logic**:
```python
if term_variations and len(term_variations) > 0:
    results = rag_engine.query_with_variations(query, term_variations, k)
else:
    results = rag_engine.query(query, k)
```

**Benefit**: Automatically uses enhanced search when LLM provides variations

---

## How It Works

### End-to-End Flow

1. **LLM receives enhanced prompt** with search strategy instructions
2. **LLM generates query with variations**:
   ```json
   {
     "query": "ASPEN pediatric malnutrition diagnostic criteria",
     "term_variations": ["undernutrition", "PEM", "SAM", "wasting", "stunting"],
     "query_type": "guideline",
     "purpose": "Classification criteria with terminology variations"
   }
   ```
3. **Agent system parses response** and extracts `term_variations`
4. **RAG engine executes enhanced query**:
   - Combines: "ASPEN pediatric malnutrition diagnostic criteria undernutrition PEM SAM wasting stunting"
   - Fetches 2x results for deduplication
   - Returns top k unique chunks
5. **Better recall**: More relevant chunks retrieved due to term variations

---

## Benefits

### ✅ **Better Recall**
- Term variations catch content that uses different terminology
- Example: "PEM" variation finds chunks that don't use "malnutrition"

### ✅ **Leniency**
- Handles terminology variations automatically
- Works with abbreviations, synonyms, related terms
- Robust to different medical terminology styles

### ✅ **Multi-Faceted Search**
- Different query angles (guideline, diagnostic, assessment)
- Comprehensive coverage from multiple perspectives

### ✅ **Backward Compatible**
- Works with or without term variations
- Falls back gracefully if LLM doesn't provide variations
- No breaking changes to existing functionality

### ✅ **Both Pipelines**
- Structured pipeline: Enhanced Stage 1 and Gap Analysis
- Adaptive pipeline: Enhanced tool calling throughout iterations

---

## Version
**1.0.0** - All components maintain v1.0.0 versioning

---

## Files Modified

1. `/core/agent_system.py` - Prompts + RAG integration
2. `/core/prompt_templates.py` - Agentic prompt updates
3. `/core/rag_engine.py` - Added `query_with_variations()`
4. `/core/extras_manager.py` - Added `match_extras_with_variations()`

---

## Testing

### Recommended Test Cases

1. **With Variations**: Verify enhanced query executes correctly
2. **Without Variations**: Verify fallback to standard query
3. **Deduplication**: Verify unique results returned
4. **Extras Expansion**: Verify keyword expansion works
5. **Both Pipelines**: Test structured and adaptive modes

---

## Future Enhancements

- Medical terminology database for automatic variation generation
- Configurable leniency levels (low/medium/high)
- Analytics on variation effectiveness
- User-visible search strategy in UI

---

**Implemented by**: Claude (AI Assistant)
**Date**: 2026-01-07
**Version**: 1.0.0
