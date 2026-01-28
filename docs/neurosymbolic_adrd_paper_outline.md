# Neurosymbolic AI for ADRD Classification: Comparing Pure LLM vs. Hybrid Approaches

## A Comprehensive Paper Outline and Pipeline Code Review

---

# PART I: PAPER OUTLINE

## Abstract (Proposed)

**Background:** Alzheimer's Disease and Related Dementias (ADRD) classification from clinical notes remains challenging due to complex diagnostic criteria requiring integration of cognitive assessments, functional status, and differential diagnosis. While Large Language Models (LLMs) show promise in clinical NLP tasks, their "black-box" reasoning raises concerns about reliability and interpretability in high-stakes medical decisions.

**Objective:** To compare the performance of pure LLM-based classification versus a neurosymbolic AI approach that integrates LLM reasoning with deterministic symbolic computation and knowledge retrieval for ADRD classification using NIA-AA criteria.

**Methods:** We developed ClinOrchestra, a neurosymbolic orchestration platform that combines: (1) LLM reasoning for natural language understanding, (2) symbolic functions for deterministic clinical score interpretation (MoCA, MMSE, CDR), (3) RAG-based retrieval from clinical guidelines, and (4) domain-specific knowledge hints. We evaluated both approaches on a clinical note dataset with ground-truth ADRD labels.

**Results:** [To be completed after experiments]

**Conclusions:** [To be completed after experiments]

---

## 1. Introduction

### 1.1 Background and Motivation
- **ADRD Prevalence and Impact:** Growing burden of Alzheimer's disease and related dementias globally
- **Clinical Classification Challenges:**
  - Complex multi-domain criteria (NIA-AA guidelines)
  - Requires integration of cognitive testing, functional assessment, and differential diagnosis
  - Distinction between MCI and dementia based on functional independence
- **Need for AI Assistance:** Shortage of specialists, time constraints in primary care

### 1.2 Problem Statement
- Pure LLM approaches may produce inconsistent interpretations of standardized clinical scores
- LLMs lack explicit encoding of clinical diagnostic criteria
- Need for verifiable, interpretable AI reasoning in medical classification

### 1.3 Research Questions
1. **RQ1:** How does pure LLM classification compare to neurosymbolic hybrid approaches for ADRD detection?
2. **RQ2:** Does integrating symbolic functions for clinical score interpretation improve classification accuracy?
3. **RQ3:** What role does RAG-based guideline retrieval play in classification performance?
4. **RQ4:** Can neurosymbolic approaches provide more interpretable reasoning chains?

### 1.4 Contributions
1. ClinOrchestra: An open-source neurosymbolic AI platform for clinical NLP
2. Comparative evaluation of LLM-only vs. neurosymbolic ADRD classification
3. Domain-specific symbolic functions for NIA-AA criteria evaluation
4. Analysis of interpretability and reasoning quality

---

## 2. Related Work

### 2.1 AI in ADRD Detection
- Traditional ML approaches (NLP + classifiers)
- Deep learning for dementia classification
- LLM-based clinical NLP

### 2.2 LLMs in Clinical Decision Support
- GPT models for medical text analysis
- Challenges: hallucination, inconsistency, lack of domain grounding

### 2.3 Neurosymbolic AI
- Definition and theoretical foundations
- Integration of neural and symbolic reasoning
- Applications in healthcare

### 2.4 NIA-AA Clinical Guidelines
- Dementia diagnostic criteria (Table 1 of guidelines)
- MCI criteria and distinction from dementia
- Cognitive assessment tools (MoCA, MMSE, CDR, Mini-Cog)

---

## 3. Methods

### 3.1 Study Design
- **Comparison Arms:**
  - **Arm 1 (Control):** Pure LLM classification with NIA-AA prompt only
  - **Arm 2 (Treatment):** Neurosymbolic approach (LLM + Functions + RAG + Extras)

### 3.2 Dataset
- **Source:** Clinical notes with confirmed ADRD diagnoses
- **Labels:** Binary classification (ADRD: YES/NO)
- **Inclusion Criteria:** Notes containing cognitive assessments and clinical evaluations
- **Sample Size:** [To be determined]
- **Train/Test Split:** [To be determined]

### 3.3 Pure LLM Approach (Arm 1)

#### 3.3.1 Prompt Design
```
Classify for ADRD per NIA-AA criteria.

DEMENTIA (YES):
Cognitive or behavioral symptoms that:
1. Interfere with ability to function at work or usual activities
2. Represent decline from previous levels
3. Not explained by delirium or major psychiatric disorder

AND minimum TWO domains:
a) Memory: repetitive questions, misplacing items, forgetting events, getting lost
b) Executive: poor safety understanding, cannot manage finances, poor decisions
c) Visuospatial: cannot recognize faces/objects, cannot find objects
d) Language: word-finding difficulty, hesitations, speech/writing errors
e) Behavior: mood fluctuations, agitation, apathy, loss of drive, withdrawal

MCI (NO):
- Cognitive concern reported
- Objective impairment in one or more domains
- Preservation of independence in functional abilities
- Not demented

Output: classification (YES/NO), rationale
```

#### 3.3.2 Model Configuration
- LLM: GPT-4 / Claude / Gemini (specify version)
- Temperature: 0.0 (deterministic)
- Max tokens: 4096

### 3.4 Neurosymbolic Approach (Arm 2)

#### 3.4.1 Architecture Overview
```
                    ┌─────────────────────────────────────────────┐
                    │           ClinOrchestra Pipeline            │
                    └─────────────────────────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │   Stage 1: Task Analysis & Tool Planning    │
                    │   (LLM analyzes text, identifies needed     │
                    │    tools, generates queries)                │
                    └─────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
         ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
         │   Functions      │  │   RAG Engine     │  │   Extras Manager │
         │ (Symbolic)       │  │ (Guidelines)     │  │ (Domain Hints)   │
         ├──────────────────┤  ├──────────────────┤  ├──────────────────┤
         │ • MoCA interpret │  │ • NIA-AA 2024    │  │ • Differential   │
         │ • MMSE interpret │  │   Guidelines     │  │   diagnoses      │
         │ • CDR interpret  │  │ • DSM-5 criteria │  │ • Cognitive test │
         │ • Domain count   │  │ • AD dementia    │  │   terminology    │
         │ • Criteria check │  │   criteria       │  │ • Functional     │
         └──────────────────┘  └──────────────────┘  │   assessment     │
                    │                    │           └──────────────────┘
                    └────────────────────┼────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │    Stage 3: Synthesis (LLM combines all     │
                    │    tool outputs into structured output)     │
                    └─────────────────────────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │   Stage 4: RAG Refinement (Optional)        │
                    │   (Evidence-based verification)             │
                    └─────────────────────────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────────┐
                    │        Final Classification Output          │
                    │  {classification: YES/NO, rationale: ...}   │
                    └─────────────────────────────────────────────┘
```

#### 3.4.2 Symbolic Functions
The following deterministic functions ground LLM reasoning:

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `interpret_moca` | MoCA score interpretation per Table 5 | score (0-30), years_education | impairment_detected, interpretation |
| `interpret_mmse` | MMSE score interpretation per Table 5 | score (0-30) | impairment_detected, severity |
| `interpret_mini_cog` | Mini-Cog interpretation per Table 5 | score (0-5) | impairment_detected, interpretation |
| `interpret_slums` | SLUMS interpretation (education-adjusted) | score (0-30), years_education | impairment_detected, interpretation |
| `interpret_m_ace` | M-ACE interpretation with PPV | score (0-30) | impairment_detected, PPV, interpretation |
| `interpret_gpcog` | GPCOG patient+informant interpretation | patient_score, informant_score | impairment_detected, reason |
| `count_domains` | Count affected NIA-AA cognitive domains | memory, executive, visuospatial, language, behavior | count, meets_criteria |
| `check_dementia_criteria` | Verify all NIA-AA dementia criteria | interferes_function, decline, delirium_excluded, psychiatric_excluded, domain_count | meets_dementia_criteria, classification |
| `calculate_cdr_severity` | CDR global score interpretation | cdr_global_score (0, 0.5, 1, 2, 3) | category, diagnostic_implications |

#### 3.4.3 RAG Knowledge Sources
- Alzheimer's Association Clinical Practice Guideline 2024 (3 PDF documents)
- NIA-AA diagnostic criteria tables
- DSM-5 neurocognitive disorder criteria

#### 3.4.4 Domain Extras (Knowledge Hints)
- ADRD differential diagnoses list
- Neuroimaging findings terminology
- Functional assessment terms (ADL, IADL)
- Cognitive symptom terminology
- Behavioral/psychiatric symptom terminology
- Biomarker terminology
- Reversible cause workup terms

### 3.5 Evaluation Metrics
- **Primary:** Accuracy, Sensitivity, Specificity, F1-Score
- **Secondary:** Positive Predictive Value (PPV), Negative Predictive Value (NPV)
- **Additional:** AUC-ROC, confusion matrix analysis
- **Interpretability:** Reasoning chain quality assessment

### 3.6 Statistical Analysis
- McNemar's test for paired comparison
- 95% confidence intervals
- Subgroup analysis by dementia subtype if applicable

---

## 4. Expected Results

### 4.1 Hypotheses
- **H1:** Neurosymbolic approach will achieve higher accuracy than pure LLM
- **H2:** Symbolic function integration will reduce score interpretation errors
- **H3:** RAG retrieval will improve guideline adherence
- **H4:** Neurosymbolic reasoning will be more interpretable

### 4.2 Results Tables (Template)

**Table 1: Classification Performance Comparison**
| Metric | Pure LLM | Neurosymbolic | p-value |
|--------|----------|---------------|---------|
| Accuracy | - | - | - |
| Sensitivity | - | - | - |
| Specificity | - | - | - |
| F1-Score | - | - | - |
| PPV | - | - | - |
| NPV | - | - | - |

**Table 2: Error Analysis**
| Error Type | Pure LLM (n) | Neurosymbolic (n) |
|------------|--------------|-------------------|
| Score misinterpretation | - | - |
| Criteria not applied | - | - |
| MCI vs Dementia confusion | - | - |
| Reversible cause overlooked | - | - |

### 4.3 Ablation Study Design
To isolate component contributions:
- LLM only (baseline)
- LLM + Functions only
- LLM + RAG only
- LLM + Extras only
- LLM + Functions + RAG
- LLM + Functions + RAG + Extras (full neurosymbolic)

---

## 5. Discussion

### 5.1 Key Findings Summary
- [To be written based on results]

### 5.2 Clinical Implications
- Value of symbolic grounding for medical AI
- Role of guideline-based RAG in clinical decision support
- Interpretability for clinician trust

### 5.3 Limitations
- Single-center data
- Retrospective analysis
- Specific to English-language notes
- LLM version dependency

### 5.4 Future Directions
- Multi-center validation
- Prospective clinical trial
- Extension to other neurodegenerative conditions
- Integration with EHR systems

---

## 6. Conclusion
- [To be written based on results]

---

## References
1. Atri A, et al. Alzheimer's Association clinical practice guideline for the diagnostic evaluation. *Alzheimer's & Dementia*. 2024.
2. McKhann GM, et al. The diagnosis of dementia due to Alzheimer's disease. *Alzheimer's & Dementia*. 2011.
3. [Additional references to be added]

---

# PART II: PIPELINE CODE REVIEW

## 1. Overview of ClinOrchestra Architecture

ClinOrchestra is a **task-agnostic neurosymbolic AI orchestration platform** designed for clinical data extraction. The platform combines:
- **Neural reasoning** (LLMs) for natural language understanding
- **Symbolic computation** (deterministic functions) for grounded calculations
- **Knowledge retrieval** (RAG) for guideline-based evidence
- **Domain hints** (Extras) for contextual knowledge injection

## 2. Core Components Analysis

### 2.1 ExtractionAgent (`core/agent_system.py`)

**Location:** `/home/user/clinorchestra/core/agent_system.py`

**Purpose:** Implements the STRUCTURED 4-stage extraction pipeline

**Key Components:**

```python
class ExtractionAgent:
    """
    UNIVERSAL AGENTIC SYSTEM - Works for ANY clinical extraction task

    This agent is task-agnostic and dynamically determines:
    - Required information based on YOUR task (defined in schema/prompts)
    - Which functions to call from registry (based on YOUR task needs)
    - Optimal RAG queries for YOUR clinical domain
    - Extras keywords matching YOUR task context
    """
```

**Stage Flow:**
1. **Stage 1: Task Analysis** - LLM analyzes the clinical text and determines required tools
2. **Stage 2: Tool Execution** - Functions, RAG, and Extras execute in parallel (async)
3. **Stage 3: Synthesis** - LLM combines all tool outputs into structured JSON
4. **Stage 4: RAG Refinement** - Optional evidence-based verification

**Code Quality Assessment:**
- Well-structured with clear separation of concerns
- Comprehensive error handling and retry logic
- Performance monitoring with `TimingContext`
- Tool deduplication to prevent redundant calls
- Adaptive retry mechanism with metrics tracking

### 2.2 Function Registry (`core/function_registry.py`)

**Purpose:** Manages and executes deterministic Python functions

**ADRD-Specific Functions (from `yaml_configs/adrd_functions.yaml`):**

```yaml
# MoCA Interpretation (Lines 1-60)
- name: calculate_moca_severity
  description: Interpret MoCA score with education correction
  # Key logic:
  # - ≥26: Normal Cognition (add 1 point if education ≤12 years)
  # - 18-25: Mild Cognitive Impairment Range
  # - <18: Significant Cognitive Impairment Range
  # IMPORTANT: Returns cognitive_impairment_level, NOT dementia diagnosis

# MMSE Interpretation (Lines 61-99)
- name: calculate_mmse_severity
  description: Interpret MMSE score and categorize severity
  # Key logic:
  # - ≥24: Normal or Mild Impairment
  # - 18-23: Mild Dementia
  # - 10-17: Moderate Dementia
  # - <10: Severe Dementia

# CDR Interpretation (Lines 100-169)
- name: calculate_cdr_severity
  description: Interpret CDR global score
  # CRITICAL: CDR 0.5 is AMBIGUOUS - requires functional assessment
  # - 0: No Cognitive Impairment
  # - 0.5: MCI or Questionable Dementia (REQUIRES FUNCTIONAL ASSESSMENT)
  # - 1: Mild Dementia
  # - 2: Moderate Dementia
  # - 3: Severe Dementia
```

**Code Quality Assessment:**
- Functions correctly implement clinical guidelines
- Clear documentation with citations (Nasreddine 2005, Folstein 1975, Morris 1993)
- Appropriate handling of edge cases (invalid scores, missing data)
- CDR 0.5 correctly flagged as ambiguous (critical for MCI vs dementia distinction)

### 2.3 Regex Preprocessor (`core/regex_preprocessor.py`)

**Purpose:** Normalizes clinical text before LLM processing

**ADRD-Specific Patterns (from `yaml_configs/adrd_patterns.yaml`):**

| Pattern | Description | Example |
|---------|-------------|---------|
| `extract_moca_score` | Extracts MoCA scores | "MoCA score 22/30" → "MoCA: 22/30" |
| `extract_mmse_score` | Extracts MMSE scores | "MMSE=24" → "MMSE: 24/30" |
| `extract_cdr_score` | Extracts CDR global scores | "CDR 0.5" → "CDR: 0.5" |
| `detect_alzheimer_diagnosis` | Detects AD mentions | "Alzheimer's disease" → "[ALZHEIMER_DISEASE]" |
| `detect_vascular_dementia` | Detects VaD mentions | "vascular dementia" → "[VASCULAR_DEMENTIA]" |
| `detect_lewy_body_dementia` | Detects LBD mentions | "Lewy body dementia" → "[LEWY_BODY_DEMENTIA]" |
| `detect_frontotemporal_dementia` | Detects FTD mentions | "FTD" → "[FRONTOTEMPORAL_DEMENTIA]" |

**Code Quality Assessment:**
- Comprehensive coverage of common score formats
- Handles alternative naming conventions (e.g., "Montreal Cognitive Assessment" vs "MoCA")
- Standardizes terminology for consistent LLM processing

### 2.4 RAG Engine (`core/rag_engine.py`)

**Purpose:** Retrieves relevant sections from clinical guidelines

**Key Features:**
- FAISS vector search for efficient similarity matching
- Batch embedding generation (25-40% faster)
- Query variations for improved recall
- Document chunking with configurable size/overlap

**ADRD RAG Sources:**
- Alzheimer's Association Clinical Practice Guideline 2024 (3 PDFs)
- Contains Tables 1, 3, 5 from guidelines (dementia criteria, AD criteria, cognitive tests)

### 2.5 Extras Manager (`core/extras_manager.py`)

**Purpose:** Provides domain-specific hints and contextual knowledge

**ADRD-Specific Extras (from `yaml_configs/adrd_extras.yaml`):**

| Extra ID | Type | Content Summary |
|----------|------|-----------------|
| `extra_adrd_differential_013` | keyword_list | Differential diagnoses (delirium, depression, NPH, etc.) |
| `extra_adrd_imaging_007` | keyword_list | Neuroimaging findings (atrophy, WMH, PET findings) |
| `extra_adrd_functional_002` | keyword_list | Functional assessment terms (ADL, IADL, Katz, Lawton) |
| `extra_adrd_criteria_012` | keyword_list | Diagnostic criteria references (NIA-AA, DSM-5, NINDS-AIREN) |
| `extra_adrd_cognitive_001` | keyword_list | Cognitive assessment terminology |
| `extra_adrd_symptoms_004` | keyword_list | Cognitive symptom terminology |
| `extra_adrd_behavioral_005` | keyword_list | BPSD terminology |
| `extra_adrd_biomarkers_008` | keyword_list | Biomarker terminology (CSF, PET) |
| `extra_adrd_reversible_009` | keyword_list | Reversible cause workup terms |

## 3. Pipeline Execution Flow

### 3.1 Stage 1: Task Analysis

**File:** `core/agent_system.py`, method `_execute_stage1_analysis()`

**Process:**
1. Builds analysis prompt with:
   - Clinical text
   - Available functions from registry
   - JSON output schema
   - Task-specific prompts
2. LLM generates:
   - Task understanding (what information is needed)
   - Function call requests (which functions to execute)
   - RAG queries (what to search in guidelines)
   - Extras keywords (what domain hints to retrieve)

**Key Code:**
```python
def _execute_stage1_analysis(self) -> bool:
    """
    Stage 1: LLM analyzes task and determines:
    - Required information
    - Functions to call
    - RAG queries
    - Extras keywords
    """
    analysis_prompt = self._build_stage1_analysis_prompt()

    for attempt in range(self.max_retries):
        response = self.llm_manager.generate(analysis_prompt, ...)
        task_understanding = self._parse_task_analysis_response(response)

        if task_understanding:
            self._convert_task_understanding_to_tool_requests()
            self._validate_and_improve_rag_queries()  # Enhanced query validation
            return True
```

### 3.2 Stage 2: Tool Execution

**File:** `core/agent_system.py`, method `_execute_stage2_tools()`

**Process:**
1. Executes all tools in parallel (async/threaded)
2. Three tool types:
   - **Functions:** Deterministic calculations (e.g., `calculate_moca_severity(22)`)
   - **RAG:** Vector search in guidelines (e.g., "NIA-AA dementia criteria")
   - **Extras:** Keyword-based hint retrieval (e.g., ["dementia", "functional"])

**Key Code:**
```python
def _execute_stage2_tools(self):
    """Execute all tool requests in parallel"""
    with ThreadPoolExecutor() as executor:
        futures = []
        for request in self.context.tool_requests:
            if request['type'] == 'function':
                futures.append(executor.submit(self._execute_function, request))
            elif request['type'] == 'rag':
                futures.append(executor.submit(self._execute_rag_query, request))
            elif request['type'] == 'extras':
                futures.append(executor.submit(self._execute_extras_lookup, request))
```

### 3.3 Stage 3: Synthesis

**File:** `core/agent_system.py`, method `_execute_stage3_extraction()`

**Process:**
1. Formats all tool outputs into a comprehensive prompt
2. LLM synthesizes information into structured JSON output
3. JSON validation and content validation applied

**Key Code:**
```python
def _execute_stage3_extraction(self) -> bool:
    """
    Stage 3: LLM synthesizes all tool outputs into final structured output
    """
    synthesis_prompt = self._build_synthesis_prompt()
    response = self.llm_manager.generate(synthesis_prompt, ...)

    # Parse and validate JSON
    extracted_json = self.json_parser.parse(response)
    validation_result = self.json_validator.validate(extracted_json)
```

### 3.4 Stage 4: RAG Refinement

**File:** `core/agent_system.py`, method `_execute_stage4_rag_refinement_with_retry()`

**Process:**
1. Optional refinement stage
2. Uses RAG evidence to verify/refine Stage 3 output
3. Evidence-based field enhancement

## 4. Strengths of the Current Implementation

### 4.1 Clinical Accuracy
- Functions correctly implement published guidelines (NIA-AA, DSM-5)
- Education-adjusted scoring where applicable (MoCA, SLUMS)
- CDR 0.5 ambiguity correctly handled (critical distinction)
- Comprehensive differential diagnosis coverage

### 4.2 Architecture Quality
- Clean separation of concerns (neural vs. symbolic)
- Task-agnostic design (works for any clinical task)
- Parallel tool execution for performance
- Adaptive retry with metrics tracking
- Tool deduplication to prevent waste

### 4.3 Interpretability
- Explicit tool call logs
- Reasoning chain from LLM visible in Stage 1 output
- Function outputs provide deterministic justification
- RAG citations provide evidence trail

### 4.4 Extensibility
- YAML-based configuration for easy task customization
- Modular components (add new functions without code changes)
- Multiple LLM provider support (OpenAI, Anthropic, Google, local)

## 5. Areas for Improvement

### 5.1 ADRD-Specific Enhancements
- **Add CDR Sum of Boxes function:** More sensitive than global CDR
- **Add FAQ (Functional Activities Questionnaire) interpretation:** Commonly used
- **Add Trail Making Test interpretation:** Executive function assessment
- **Add Clock Drawing Test scoring function:** Visuospatial/executive assessment

### 5.2 Pipeline Improvements
- **Confidence scoring:** Add uncertainty quantification to outputs
- **Explanation generation:** Structured explanation of reasoning
- **Contradition detection:** Flag conflicting evidence
- **Temporal reasoning:** Track cognitive decline over time

### 5.3 Evaluation Needs
- **Ground truth dataset:** Need labeled ADRD classification dataset
- **Inter-rater reliability:** Compare to clinician gold standard
- **Error categorization:** Systematic error analysis framework

## 6. Recommended Experiments

### 6.1 Experiment 1: Pure LLM vs. Neurosymbolic
**Design:** A/B comparison
- Arm A: Pure LLM with NIA-AA prompt only
- Arm B: Full neurosymbolic pipeline
**Metrics:** Accuracy, F1, PPV, NPV, sensitivity, specificity
**Expected Outcome:** Neurosymbolic outperforms on cases with explicit scores

### 6.2 Experiment 2: Ablation Study
**Design:** Remove one component at a time
- Full pipeline (baseline)
- Without functions
- Without RAG
- Without extras
**Metrics:** Performance delta from full pipeline
**Expected Outcome:** Functions provide largest contribution for score-based cases

### 6.3 Experiment 3: Score Interpretation Accuracy
**Design:** Focused evaluation on score interpretation
- Extract all cases with explicit MoCA/MMSE/CDR scores
- Compare score interpretation between pure LLM and neurosymbolic
**Metrics:** Score interpretation accuracy
**Expected Outcome:** Symbolic functions achieve near-100% accuracy

### 6.4 Experiment 4: MCI vs. Dementia Distinction
**Design:** Subset analysis on borderline cases
- Extract cases with CDR 0.5 or MoCA 18-25
- Compare classification accuracy
**Metrics:** Sensitivity for detecting dementia vs. MCI
**Expected Outcome:** Neurosymbolic better handles ambiguous cases with functional assessment hints

---

## Appendix A: Proposed Output Schema for ADRD Classification

```json
{
  "classification": {
    "type": "string",
    "enum": ["YES", "NO"],
    "description": "ADRD classification (YES=dementia, NO=MCI or normal)"
  },
  "confidence": {
    "type": "string",
    "enum": ["High", "Medium", "Low"],
    "description": "Confidence level of classification"
  },
  "cognitive_domains_affected": {
    "type": "array",
    "items": ["Memory", "Executive", "Visuospatial", "Language", "Behavior"],
    "description": "NIA-AA cognitive domains with impairment"
  },
  "domain_count": {
    "type": "integer",
    "description": "Number of affected domains (≥2 required for dementia)"
  },
  "functional_impairment": {
    "type": "boolean",
    "description": "Whether functional decline is documented"
  },
  "cognitive_scores": {
    "type": "object",
    "properties": {
      "moca": {"type": "number", "description": "MoCA score if present"},
      "mmse": {"type": "number", "description": "MMSE score if present"},
      "cdr": {"type": "number", "description": "CDR global score if present"}
    }
  },
  "exclusions_verified": {
    "type": "object",
    "properties": {
      "delirium_excluded": {"type": "boolean"},
      "psychiatric_excluded": {"type": "boolean"}
    }
  },
  "rationale": {
    "type": "string",
    "description": "Clinical reasoning for classification"
  },
  "evidence": {
    "type": "array",
    "items": {"type": "string"},
    "description": "Supporting evidence from clinical text"
  }
}
```

---

## Appendix B: NIA-AA Criteria Summary (for LLM Prompt)

### Dementia Criteria (Table 1)
Cognitive or behavioral symptoms that:
1. Interfere with ability to function at work or usual activities
2. Represent decline from previous levels of functioning
3. Not explained by delirium or major psychiatric disorder

AND cognitive impairment in minimum TWO domains:
- Memory (repetitive questions, misplacing items, forgetting events, getting lost)
- Executive (poor safety understanding, cannot manage finances, poor decisions)
- Visuospatial (cannot recognize faces/objects, cannot find objects)
- Language (word-finding difficulty, hesitations, speech/writing errors)
- Behavior (mood fluctuations, agitation, apathy, loss of drive, withdrawal)

### MCI Criteria (Table 1)
- Cognitive concern reflecting change, reported by patient/informant/clinician
- Objective evidence of impairment in ≥1 cognitive domain
- **Preservation of independence in functional abilities**
- Not demented

### Key Distinction
**MCI vs. Dementia = Functional Independence**
- MCI: Cognitive impairment WITH preserved independence
- Dementia: Cognitive impairment WITH functional decline

---

*Document prepared for ClinOrchestra Neurosymbolic ADRD Classification Project*
*Medical University of South Carolina, Biomedical Informatics Center*
*Date: 2026-01-28*
