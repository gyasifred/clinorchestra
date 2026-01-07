# Prompt Analysis & Search Strategy Integration

## Executive Summary

Current prompt token usage analysis and concise improvements to integrate LLM-driven search strategy WITHOUT creating new prompts or excessive bloat.

---

## 📊 Current Prompt Analysis

### What the LLM Sees (Complete Prompt Chain)

#### **Structured Pipeline:**
```
Stage 1 (Analysis):
  ├─ Task Description: ~500-2000 tokens (varies by task template)
  ├─ Stage 1 Analysis Instructions: ~2500-3500 tokens
  ├─ Clinical Text: Variable (user input)
  ├─ Schema JSON: Variable (~200-1000 tokens)
  ├─ Available Tools Descriptions: ~300-800 tokens
  └─ TOTAL STAGE 1: ~3500-6500 tokens + clinical text

Stage 2: Tool Execution (no LLM call)

Stage 3 (Extraction):
  ├─ Main/Minimal Prompt: ~500-3000 tokens (task-dependent)
  ├─ Clinical Text: Variable
  ├─ RAG Outputs: ~500-3000 tokens (if RAG used)
  ├─ Function Outputs: ~200-1000 tokens (if functions used)
  ├─ Extras Outputs: ~300-1500 tokens (if extras used)
  ├─ Schema Instructions: ~300-800 tokens
  └─ TOTAL STAGE 3: ~2000-10000 tokens + clinical text

Stage 4 (Gap Analysis + Refinement):
  ├─ Gap Analysis Prompt: ~600-800 tokens
  ├─ Current Extraction: Variable (~500-2000 tokens)
  ├─ Tool Results: Variable
  ├─ Refinement Prompt: ~800-1200 tokens
  └─ TOTAL STAGE 4: ~2000-5000 tokens
```

#### **Adaptive Pipeline:**
```
Initial Iteration:
  ├─ Agentic Framework: ~1500-2000 tokens
  ├─ Task Prompt: ~500-3000 tokens (task-dependent)
  ├─ Clinical Text: Variable
  ├─ Schema: ~200-1000 tokens
  ├─ Tool Descriptions: ~300-800 tokens
  └─ TOTAL INITIAL: ~2500-7000 tokens + clinical text

Subsequent Iterations:
  ├─ Conversation History: Grows with each iteration
  ├─ Tool Results: Added incrementally
  └─ Budget: Max 10 iterations, 100 tool calls
```

---

## ⚠️ Token Efficiency Concerns

### Current Issues:
1. **Stage 1 Analysis Prompt is LONG** (~310 lines, ~2500-3500 tokens)
   - Contains extensive examples and explanations
   - Repeats instructions multiple times
   - Has verbose formatting

2. **Agentic Prompt is LENGTHY** (~215 lines, ~1500-2000 tokens)
   - Detailed workflow phases
   - Extensive examples
   - Verbose critical principles

3. **Adding search strategy could add 300-800+ tokens if not careful**

### Risk:
- **Total prompt + clinical text could approach context limits**
- **Higher costs per API call**
- **Slower processing**

---

## ✅ SOLUTION: Concise Search Strategy Integration

### Design Principle:
**"Compress, don't expand. Replace examples with strategy."**

### Approach:
1. **Replace verbose examples with search strategy guidance**
2. **Condense existing sections**
3. **Integrate search strategy into existing workflow steps**
4. **Use bullets, not paragraphs**
5. **Remove redundancy**

---

## 🎯 Specific Improvements

### **1. Stage 1 Analysis Prompt** (PRIORITY 1)

**Current Length:** ~310 lines, ~2500-3500 tokens
**Target Length:** ~250 lines, ~1800-2500 tokens
**Savings:** ~500-1000 tokens

#### Changes:

**A. Replace "3. BUILD MULTIPLE INTELLIGENT RAG QUERIES" section with:**

```python
3. BUILD SEARCH STRATEGY FOR RAG (Multi-Faceted Approach):

   Strategy: Create 3-5 queries targeting DIFFERENT aspects with VARIED terminology

   a) PRIMARY TERMS: Extract core keywords from task domain
      - Diagnosis, condition, assessment type
      - Use medical terminology from schema fields

   b) TERM VARIATIONS (LENIENCY FOR BETTER RECALL):
      For each concept, include:
      • Synonyms: "malnutrition" → ["undernutrition", "nutritional deficiency"]
      • Abbreviations: "malnutrition" → ["PEM", "SAM", "MAM"]
      • Related terms: "malnutrition" → ["wasting", "stunting", "failure to thrive"]
      • Broader terms: "ASPEN criteria" → ["pediatric assessment", "nutritional screening"]

   c) MULTI-FACETED QUERIES: Build 3-5 queries, each with different angle:
      • Guideline Query: "[organization] [condition] [guideline]"
        Example: "ASPEN pediatric malnutrition diagnostic criteria"
      • Diagnostic Query: "[condition] [classification] [severity]"
        Example: "moderate malnutrition z-score anthropometric"
      • Assessment Query: "[method] [scoring] [interpretation]"
        Example: "growth chart percentile WHO classification"
      • Treatment Query (if relevant): "[intervention] [guidelines]"
      • Domain-Specific: Use variations of key terms

   d) LENIENCY TACTICS:
      • Include common misspellings/variations
      • Use both formal and colloquial terms
      • Add age/population qualifiers: "pediatric", "adult", "geriatric"
      • Include metric variations: "z-score", "standard deviation", "percentile"

   Example Output:
   {
     "query": "ASPEN pediatric malnutrition undernutrition diagnostic criteria severity",
     "variations": ["PEM assessment", "nutritional deficiency screening", "wasting stunting"],
     "query_type": "guideline",
     "purpose": "Retrieve ASPEN classification with terminology variations"
   }
```

**B. Replace "4. IDENTIFY EXTRAS KEYWORDS" section with:**

```python
4. BUILD SEARCH STRATEGY FOR EXTRAS (Term Expansion):

   Extract 5-8 CORE concepts, then expand each:

   For each concept:
   • Core term: From schema field name
   • Variations: Synonyms, abbreviations, related terms
   • Context: Add qualifiers (age group, specialty, system)

   Example:
   Core: "malnutrition"
   Expanded: ["malnutrition", "undernutrition", "PEM", "SAM", "nutritional deficiency",
              "pediatric malnutrition", "wasting", "stunting"]

   Output: Return BOTH core terms AND variations
```

**C. Condense examples from ~70 lines to ~30 lines:**

Remove EXAMPLE 0, EXAMPLE 1, EXAMPLE 2, EXAMPLE 3 verbose explanations.

Replace with:
```
QUICK REFERENCE:
• No tools needed: Simple extraction of existing values → empty arrays
• Functions needed: Task requires calculations/conversions → include functions
• Guidelines needed: Task mentions standards/criteria → include RAG with varied terms
• Hints needed: Domain-specific task → include extras with term variations

For RAG/Extras: ALWAYS include term variations for better recall and leniency
```

**D. Add to response format:**

```python
"rag_queries": [
  {
    "query": "primary keywords string",
    "term_variations": ["synonym1", "abbrev1", "related1"],  // NEW
    "query_type": "guideline|diagnostic|assessment|treatment|domain_specific",
    "purpose": "what this retrieves"
  }
],
"extras_keywords": ["core1", "variation1", "core2", "variation2", ...]  // Include variations
```

---

### **2. Gap Analysis Prompt** (PRIORITY 2)

**Current Length:** ~75 lines, ~600-800 tokens
**Target Length:** ~65 lines, ~500-700 tokens
**Savings:** ~100 tokens

#### Changes:

**Add after line "3. REQUEST ADDITIONAL TOOLS IF NEEDED:"**

```python
   SEARCH STRATEGY for Additional Tools:

   • RAG: Use DIFFERENT keywords than before (see list above)
     - Try term variations: synonyms, abbreviations, broader/narrower terms
     - Target specific gaps identified in analysis
     - Include leniency: related concepts, alternative terminology

   • Extras: Expand keywords with variations
     - Core concept + synonyms + related terms
     - Use domain-specific qualifiers

   Example RAG: Instead of repeating "malnutrition criteria",
   try: "nutritional screening tools pediatric assessment methods"
```

**Simplify keyword guidance section (lines 1759-1771):**

Current: ~13 lines, verbose
New: ~6 lines, concise

```python
if previous_rag_keywords:
    keyword_guidance += f"\n   ⚠️  RAG Keywords Used: {', '.join(previous_rag_keywords[:15])}"
    keyword_guidance += "\n   → Use VARIATIONS: synonyms, abbreviations, related terms\n"
if previous_extras_keywords:
    keyword_guidance += f"\n   ⚠️  Extras Keywords Used: {', '.join(previous_extras_keywords[:15])}"
    keyword_guidance += "\n   → Expand with: domain-specific terms, qualifiers\n"
```

---

### **3. Agentic Prompt** (PRIORITY 3)

**Current Length:** ~215 lines, ~1500-2000 tokens
**Target Length:** ~180 lines, ~1200-1600 tokens
**Savings:** ~300-400 tokens

#### Changes in `get_agentic_extraction_prompt()`:

**Replace "PHASE 2 - AUTONOMOUSLY DETERMINE & EXECUTE REQUIRED TOOLS" verbose workflow with:**

```python
**PHASE 2 - TOOL EXECUTION WITH SEARCH STRATEGY:**

When calling query_rag():
  • Use MULTI-TERM queries: "concept1 concept2 concept3"
  • Include VARIATIONS: synonyms, abbreviations, related terms
  • Try DIFFERENT angles: guidelines, diagnostics, assessments
  • Build leniency: broader terms, alternative terminology

When calling query_extras():
  • Expand keywords: core + synonyms + related
  • Add qualifiers: age, specialty, system
  • Use domain-specific variations

Strategy: Cast a WIDE NET with term variations for better recall,
then let the system rank by relevance.

Example:
query_rag("ASPEN malnutrition undernutrition PEM criteria assessment pediatric",
          "Need classification using terminology variations")
```

**Condense PHASE 3 and PHASE 4 from ~35 lines to ~20 lines**

**Remove verbose example workflow (~40 lines) and replace with:**

```python
**WORKFLOW SUMMARY:**
1. Read task → Identify gaps
2. Call tools with search strategy (varied terms, multiple angles)
3. Review results → Call more tools if needed (use different term variations)
4. Complete extraction

**SEARCH STRATEGY TIPS:**
- RAG: Multiple queries with different terminology
- Extras: Core concepts + expanded variations
- Iterate with NEW term combinations if first attempt incomplete
```

---

## 🔧 Implementation Changes

### **File 1: `/home/user/clinorchestra/core/prompt_templates.py`**

**Changes:**
1. Update `STAGE1_ANALYSIS_PROMPT` (lines 535-605) with condensed version
2. Update `get_agentic_extraction_prompt()` (lines 1267-1480) with condensed version

**Estimated reduction:** ~500-800 tokens total

---

### **File 2: `/home/user/clinorchestra/core/agent_system.py`**

**Changes:**
1. Update `_build_stage1_analysis_prompt()` (lines 1929-2239) with search strategy
2. Update `_build_stage4_gap_analysis_prompt()` (lines 1707-1847) with search strategy

**Estimated reduction:** ~400-600 tokens total

---

### **File 3: `/home/user/clinorchestra/core/rag_engine.py`**

**Changes:**
Add `query_with_variations()` method (NEW - 20-30 lines)

```python
def query_with_variations(self,
                          primary_query: str,
                          variations: List[str],
                          k: int = 10) -> List[Dict]:
    """
    Execute RAG query with term variations for better recall.

    Args:
        primary_query: Main query string
        variations: List of term variations (synonyms, abbreviations, related)
        k: Number of results to return

    Returns:
        Deduplicated results combining all variation queries
    """
    all_results = []
    seen_content = set()

    # Build combined query with variations
    full_query = f"{primary_query} {' '.join(variations)}"

    # Query with expanded terms
    results = self.query(full_query, k=k*2)  # Fetch more for dedup

    # Deduplicate and return top k
    for result in results:
        content = result.get('text', '') or result.get('content', '')
        if content and content not in seen_content:
            all_results.append(result)
            seen_content.add(content)
            if len(all_results) >= k:
                break

    return all_results
```

---

### **File 4: `/home/user/clinorchestra/core/extras_manager.py`**

**Changes:**
Add `match_with_variations()` method (NEW - 20-30 lines)

```python
def match_with_variations(self,
                          core_keywords: List[str],
                          variations: Dict[str, List[str]],
                          threshold: float = 0.2,
                          top_k: int = 10) -> List[Dict]:
    """
    Match extras using core keywords + variations for better recall.

    Args:
        core_keywords: List of core concept keywords
        variations: Dict mapping core keywords to variations
        threshold: Minimum relevance score
        top_k: Max results to return

    Returns:
        Matched extras with relevance scores
    """
    # Expand keywords with variations
    all_keywords = set(core_keywords)
    for core in core_keywords:
        if core in variations:
            all_keywords.update(variations[core])

    # Use existing match_by_keywords with expanded set
    return self.match_by_keywords(list(all_keywords), threshold, top_k)
```

---

## 📈 Expected Outcomes

### Token Savings:
- **Stage 1:** -500 to -1000 tokens (condensed examples, integrated strategy)
- **Gap Analysis:** -100 tokens (simplified guidance)
- **Agentic:** -300 to -400 tokens (condensed workflow)
- **TOTAL SAVINGS:** ~900-1500 tokens per complete extraction

### Functionality Gains:
1. **Better Recall:** Term variations catch more relevant content
2. **Leniency:** Broader/narrower terms handle terminology variations
3. **Multi-Faceted:** Different query angles get comprehensive coverage
4. **Adaptive:** LLM chooses variations intelligently based on task

### Performance:
- **Same number of LLM calls** (no new stages)
- **Potentially MORE tool executions** (variations create multiple queries)
  - Mitigated by deduplication and async execution
- **Better quality** from comprehensive retrieval

---

## 🎯 Implementation Priority

1. **HIGH PRIORITY:** Stage 1 Analysis Prompt (biggest token saver + most impact)
2. **MEDIUM PRIORITY:** Gap Analysis Prompt (smaller but still important)
3. **MEDIUM PRIORITY:** Agentic Prompt (alternative pipeline)
4. **LOW PRIORITY:** RAG/Extras engine methods (nice-to-have, optional)

---

## ⚡ Quick Win: Minimal Implementation

**If you want FASTEST path:**

Just modify Stage 1 Analysis Prompt:
1. Add term variation guidance to RAG query section (20 lines)
2. Add term expansion to extras section (15 lines)
3. Update response format to include `term_variations` field (5 lines)
4. Condense examples (remove 40 lines, add 15 lines)

**NET CHANGE:** ~0 lines added, ~500-800 tokens saved, search strategy integrated

---

## 📝 Summary

**What I'll Implement:**
✅ Integrate search strategy into EXISTING prompts (not new prompts)
✅ Keep prompts CONCISE (actually REDUCE token count)
✅ Add leniency through term variations
✅ Multi-faceted search approach
✅ Work in BOTH pipelines (structured + adaptive)

**What I WON'T Do:**
❌ Create new prompt templates
❌ Increase prompt length excessively
❌ Add new LLM call stages
❌ Complicate existing architecture

---

## 🚀 Ready to Implement?

I can now:
1. Update the 4 files with condensed prompts + search strategy
2. Test with sample clinical text
3. Validate token reduction
4. Commit and push

**Shall I proceed with implementation?**
