# ADRD Phenotyping Using LLMs: Comparing Pure LLM vs. Knowledge-Integrated Approaches

## Using ClinOrchestra's ADAPTIVE Pipeline

---

# PART I: PAPER OUTLINE

## Abstract

**Background:** Alzheimer's Disease and Related Dementias (ADRD) classification from clinical notes remains challenging due to complex diagnostic criteria requiring integration of cognitive assessments, functional status, and differential diagnosis. While Large Language Models (LLMs) show promise in clinical NLP tasks, their "black-box" reasoning raises concerns about reliability and interpretability in high-stakes medical decisions.

**Objective:** To compare pure LLM-based classification versus a knowledge-integrated approach that augments LLM reasoning with computational functions, retrieval-augmented generation (RAG), and domain knowledge for ADRD classification using NIA-AA 2024 criteria.

**Methods:** We developed ClinOrchestra, a configurable clinical NLP platform combining: (1) LLM reasoning for natural language understanding, (2) computational functions for score interpretation, (3) RAG-based guideline retrieval, and (4) domain-specific knowledge hints. Using ClinOrchestra's ADAPTIVE pipeline with 8 ADRD-specific functions and 12 clinical extras, we evaluated both approaches on clinical notes with ground-truth ADRD labels.

**Results:** [To be completed after experiments]

**Conclusions:** [To be completed after experiments]

---

## 1. Introduction

### 1.1 Research Questions
1. **RQ1:** How does pure LLM classification compare to knowledge-integrated approaches for ADRD detection?
2. **RQ2:** Does integrating computational functions for score interpretation improve classification accuracy?
3. **RQ3:** What role does RAG-based guideline retrieval play in classification performance?
4. **RQ4:** Can knowledge-integrated approaches provide more interpretable reasoning chains?

### 1.2 Contributions
1. **ClinOrchestra:** Configurable clinical NLP platform (applicable to ANY clinical task)
2. Comparative evaluation of LLM-only vs. knowledge-integrated ADRD classification
3. 8 domain-specific computational functions for NIA-AA criteria evaluation
4. 12 task-specific clinical extras (knowledge hints)

---

## 2. Methods

### 2.1 ClinOrchestra Platform

**ClinOrchestra is a UNIVERSAL clinical NLP platform** - task-agnostic by design. While applied here to ADRD classification, it can be configured for ANY clinical extraction task by changing the schema, prompts, functions, and RAG documents.

### 2.2 ADAPTIVE Pipeline Architecture

```
         ┌──────────────────────────────────────┐
         │      LLM Agent (Autonomous)          │
         │  • Analyzes clinical text            │
         │  • Calls tools iteratively           │◄────┐
         │  • Self-corrects on errors           │     │
         └──────────────────────────────────────┘     │
                    │                                 │
         ┌──────────┼──────────┬──────────┐          │
         │          │          │          │          │
         ▼          ▼          ▼          ▼          │
    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
    │Functions│ │  RAG    │ │ Extras  │ │ More    │  │
    │(8 ADRD) │ │ Engine  │ │(12 ADRD)│ │ Tools   │  │
    └─────────┘ └─────────┘ └─────────┘ └─────────┘  │
         │          │          │          │          │
         └──────────┴──────────┴──────────┘          │
                    │                                 │
                    ▼                                 │
         ┌──────────────────────────────────────┐    │
         │  Need More Tools? YES ───────────────────┘
         │                   NO → Output JSON   │
         └──────────────────────────────────────┘
```

### 2.3 Study Design
- **Arm 1 (Control):** Pure LLM with NIA-AA prompt only
- **Arm 2 (Treatment):** Knowledge-Integrated (LLM + Functions + RAG + Extras)

### 2.4 Output Schema (4 Keys)

```json
{
  "severity": "no_impairment | MCI | mild_dementia | moderate_dementia | severe_dementia",
  "syndrome": "amnestic | executive | language | visuospatial | behavioral | mixed",
  "diagnosis": {
    "classification": "ADRD | Non-ADRD",
    "specific_type": "...",
    "confidence": "high | moderate | low"
  },
  "clinical_summary": "Integrated reasoning"
}
```

---

## 3. Computational Functions (8 Total)

| # | Function | Purpose | Key Logic |
|---|----------|---------|-----------|
| 1 | `interpret_moca(score, years_education)` | MoCA per Table 5 | <26 threshold; +1 if education ≤12 |
| 2 | `interpret_mmse(score)` | MMSE per Table 5 | <26 threshold; low MCI sensitivity |
| 3 | `interpret_mini_cog(score)` | Mini-Cog per Table 5 | ≤3 threshold |
| 4 | `interpret_katz_adl(score)` | Basic ADL | 6=independent |
| 5 | `interpret_lawton_iadl(score)` | **CRITICAL: MCI vs Dementia** | 8=independent; MCI=preserved |
| 6 | `interpret_cdr(global_score)` | CDR staging | 0.5=MCI, ≥1=Dementia |
| 7 | `count_cognitive_domains(...)` | NIA-AA domain count | ≥2 for dementia |
| 8 | `check_nia_aa_criteria(...)` | Full criteria check | All 5 TRUE for dementia |

---

## 4. Task-Specific Extras (12 Knowledge Hints)

| # | ID | Name | Content Summary |
|---|-----|------|-----------------|
| 1 | `aadrd_nia_aa_dementia_criteria` | NIA-AA Dementia Criteria | 5 criteria + ≥2 domains required |
| 2 | `aadrd_nia_aa_mci_criteria` | NIA-AA MCI Criteria | Key: functional independence PRESERVED |
| 3 | `aadrd_functional_assessment` | Functional Assessment | Lawton IADL (8=indep) vs Katz ADL (6=indep) |
| 4 | `aadrd_cdr_staging` | CDR Staging | 0.5=MCI, ≥1=Dementia boundary |
| 5 | `aadrd_screening_thresholds` | Screening Thresholds | MoCA<26, MMSE<26, Mini-Cog≤3 |
| 6 | `aadrd_probable_ad_criteria` | Probable AD | Insidious onset, amnestic pattern common |
| 7 | `aadrd_dlb_criteria` | DLB Criteria | Visual hallucinations, RBD, parkinsonism |
| 8 | `aadrd_vascular_dementia_criteria` | VaD Criteria | CVD + temporal relationship |
| 9 | `aadrd_ftd_criteria` | FTD Criteria | Behavioral changes, earlier onset |
| 10 | `aadrd_non_adrd_exclusions` | Non-ADRD Causes | Delirium, psychiatric, medical, meds |
| 11 | `aadrd_mixed_dementia` | Mixed Dementia | Multiple pathologies common >85yo |
| 12 | `aadrd_clinical_judgment` | Clinical Judgment | Screening ≠ diagnosis, integrate all info |

---

## 5. Hypotheses

- **H1:** Knowledge-integrated achieves higher accuracy than pure LLM
- **H2:** Computational functions reduce score interpretation errors
- **H3:** RAG improves guideline adherence
- **H4:** Knowledge-integrated reasoning is more interpretable

---

## 6. Ablation Study Design

| Configuration | Functions | RAG | Extras |
|--------------|-----------|-----|--------|
| LLM only (baseline) | ✗ | ✗ | ✗ |
| LLM + Functions | ✓ | ✗ | ✗ |
| LLM + RAG | ✗ | ✓ | ✗ |
| LLM + Extras | ✗ | ✗ | ✓ |
| LLM + Functions + RAG | ✓ | ✓ | ✗ |
| Full Knowledge-Integrated | ✓ | ✓ | ✓ |

---

# PART II: EXPERIMENTAL PROMPTS

## Prompt 1: Pure LLM (Arm 1 - Control)

```
ADRD specialist per AA 2024 Guidelines.

TEXT:
{clinical_text}

TASK: Classify ADRD vs Non-ADRD.

DEMENTIA requires ALL:
1. Interferes with function
2. Represents decline
3. Delirium excluded
4. Psychiatric excluded
5. >=2 domains (memory, executive, visuospatial, language, behavior)

MCI: Cognitive impairment + PRESERVED function.
CDR: 0.5=MCI, >=1=Dementia.

OUTPUT:
- severity: no_impairment/MCI/mild_dementia/moderate_dementia/severe_dementia
- syndrome: amnestic/executive/language/visuospatial/behavioral/mixed
- diagnosis: {classification, specific_type, confidence}
- clinical_summary: Integrated reasoning

Return JSON only.
```

## Prompt 2: Knowledge-Integrated (Arm 2 - Treatment)

```
ADRD specialist per AA 2024 Guidelines.

TEXT:
{clinical_text}

TASK: Classify ADRD vs Non-ADRD.

TOOLS:
1. interpret_moca(score, years_education) - <26 threshold
2. interpret_mmse(score) - <26 threshold
3. interpret_mini_cog(score) - <=3 threshold
4. interpret_katz_adl(score) - Basic ADL (6=indep)
5. interpret_lawton_iadl(score) - IADL (8=indep) - MCI vs Dementia key
6. interpret_cdr(global_score) - 0.5=MCI, >=1=Dementia
7. count_cognitive_domains(memory, exec, visuo, lang, behav) - >=2 for dementia
8. check_nia_aa_criteria(...) - All 5 for dementia
9. query_rag(query) - Guidelines
10. query_extras(keywords) - Domain hints

WORKFLOW:
1. Identify scores in text
2. Call interpretation functions
3. Count domains, check criteria
4. Query RAG/extras if needed
5. Output classification

OUTPUT:
- severity: no_impairment/MCI/mild_dementia/moderate_dementia/severe_dementia
- syndrome: amnestic/executive/language/visuospatial/behavioral/mixed
- diagnosis: {classification, specific_type, confidence}
- clinical_summary: Reasoning with function results

Call tools. Output JSON.
```

## Prompt 3: Minimal (Retry/Fallback)

```
ADRD per NIA-AA 2024.

TEXT: {clinical_text}

Key: MCI=preserved function, Dementia=impaired, >=2 domains.

OUTPUT: {severity, syndrome, diagnosis: {classification, specific_type, confidence}, clinical_summary}
```

## Prompt 4: RAG Refinement

```
Refine ADRD classification using evidence.

TEXT: {clinical_text}
INITIAL: {initial_output}
EVIDENCE: {rag_chunks}

Validate: severity matches function, syndrome matches domains, diagnosis supported.

Return refined JSON.
```

---

# PART III: FUNCTION SPECIFICATIONS

## Function 1: interpret_moca
```python
def interpret_moca(score, years_education=None):
    """
    MoCA per AA 2024 Table 5. SCREENING only.
    Threshold: <26 suggests impairment.
    +1 if education <=12 years.
    """
    # Returns: {raw_score, adjusted_score, below_threshold, interpretation_note}
```

## Function 2: interpret_mmse
```python
def interpret_mmse(score):
    """
    MMSE per AA 2024 Table 5. SCREENING only.
    Threshold: <26 suggests impairment.
    NOTE: Low sensitivity for MCI.
    """
    # Returns: {score, below_threshold, interpretation_note}
```

## Function 3: interpret_mini_cog
```python
def interpret_mini_cog(score):
    """
    Mini-Cog per AA 2024 Table 5.
    Threshold: <=3 suggests impairment.
    """
    # Returns: {score, at_or_below_threshold, interpretation_note}
```

## Function 4: interpret_katz_adl
```python
def interpret_katz_adl(score):
    """
    Basic ADL (0-6, 6=independent).
    Bathing, dressing, toileting, transferring, continence, feeding.
    """
    # Returns: {score, functional_status, basic_adl_impaired}
```

## Function 5: interpret_lawton_iadl
```python
def interpret_lawton_iadl(score):
    """
    IADL (0-8, 8=independent).
    CRITICAL FOR MCI VS DEMENTIA:
    - MCI = IADLs PRESERVED
    - Dementia = IADLs IMPAIRED
    """
    # Returns: {score, functional_status, instrumental_adl_impaired, clinical_significance}
```

## Function 6: interpret_cdr
```python
def interpret_cdr(global_score):
    """
    CDR staging per Morris 1993.
    0=normal, 0.5=MCI, 1=mild, 2=moderate, 3=severe dementia.
    0.5 vs 1.0 = MCI/dementia boundary (functional).
    """
    # Returns: {global_score, staging, functional_impairment, clinical_meaning}
```

## Function 7: count_cognitive_domains
```python
def count_cognitive_domains(memory=None, executive=None, visuospatial=None, language=None, behavior=None):
    """
    Count affected domains per NIA-AA.
    Dementia requires >=2 of 5 domains.
    """
    # Returns: {count, affected_domains, meets_two_domain_criterion}
```

## Function 8: check_nia_aa_criteria
```python
def check_nia_aa_criteria(interferes_function=None, represents_decline=None,
                          delirium_excluded=None, psychiatric_excluded=None, domain_count=None):
    """
    Full NIA-AA dementia criteria check.
    All 5 must be TRUE for dementia.
    """
    # Returns: {criteria_status, all_criteria_met, unassessed_criteria}
```

---

# PART IV: EXTRAS CONTENT

## Extra 1: NIA-AA Dementia Criteria (CRITICAL)
```
NIA-AA DEMENTIA CRITERIA (AA 2024 Table 1):
Dementia requires cognitive/behavioral symptoms that:
(1) Interfere with function at work or usual activities
(2) Represent decline from previous level
(3) Not explained by delirium
(4) Not explained by major psychiatric disorder as PRIMARY cause
PLUS >=2 of 5 domains affected:
- Memory: repetitive questions, misplacing items, forgetting events
- Executive: poor judgment, cannot manage finances
- Visuospatial: cannot recognize faces/objects
- Language: word-finding difficulty
- Behavior: mood changes, apathy, disinhibition
```

## Extra 2: NIA-AA MCI Criteria (CRITICAL)
```
NIA-AA MCI CRITERIA:
(1) Cognitive concern from patient/informant/clinician
(2) Objective evidence of impairment in >=1 cognitive domain
(3) PRESERVATION OF INDEPENDENCE in functional abilities
(4) Not demented

CRITICAL: MCI = cognitive impairment + functional independence PRESERVED
         Dementia = cognitive impairment + functional IMPAIRMENT
```

## Extra 3: Functional Assessment (CRITICAL)
```
FUNCTIONAL ASSESSMENT - KEY FOR MCI VS DEMENTIA:

Lawton IADL (0-8, 8=independent):
- Phone, shopping, food prep, housekeeping, laundry, transport, meds, finances
- IADL impairment = dementia (not MCI)
- IADLs decline BEFORE basic ADLs

Katz ADL (0-6, 6=independent):
- Bathing, dressing, toileting, transferring, continence, feeding
- Basic ADL impairment = more advanced disease
```

## Extra 4: CDR Staging (CRITICAL)
```
CDR STAGING (Morris 1993):
0 = no impairment
0.5 = MCI (functional independence PRESERVED - NOT dementia)
1 = mild dementia (functional impairment PRESENT)
2 = moderate dementia
3 = severe dementia

CRITICAL: CDR 0.5 vs 1.0 is the MCI/dementia boundary based on FUNCTIONAL IMPAIRMENT
```

## Extra 5: Screening Thresholds
```
COGNITIVE SCREENING THRESHOLDS (AA 2024 Table 5) - SCREENING only, NOT diagnostic:
- MoCA <26: possible impairment (add 1 if education <=12)
- MMSE <26: possible impairment (low sensitivity for MCI)
- Mini-Cog <=3: possible impairment
```

## Extra 6: Probable AD Criteria
```
PROBABLE AD (McKhann 2011):
Meets dementia criteria + insidious onset + clear worsening + pattern:
- Amnestic (most common): memory + other domain
- Language: word-finding
- Visuospatial: spatial cognition
- Executive: reasoning/judgment

NOT if: substantial CVD, core DLB features, prominent bvFTD/PPA features
```

## Extra 7: DLB Criteria
```
DLB CRITERIA (McKeith 2017):
>=2 core features = probable DLB, 1 = possible DLB

CORE FEATURES:
1. Fluctuating cognition with variations in attention/alertness
2. Recurrent visual hallucinations (well-formed, detailed)
3. REM sleep behavior disorder
4. Spontaneous parkinsonism
```

## Extra 8: Vascular Dementia Criteria
```
VASCULAR DEMENTIA (NINDS-AIREN):
(1) Dementia established
(2) Cerebrovascular disease present
(3) Relationship: onset within 3 months of stroke, OR abrupt deterioration, OR stepwise progression

Features: early gait disturbance, falls, urinary symptoms, executive dysfunction
```

## Extra 9: FTD Criteria
```
bvFTD CRITERIA (Rascovsky 2011):
Progressive deterioration + THREE of six:
1. Early behavioral disinhibition
2. Early apathy/inertia
3. Early loss of sympathy/empathy
4. Early perseverative/compulsive behaviors
5. Hyperorality/dietary changes
6. Executive deficits with relative sparing of memory

KEY: Earlier onset (45-65), behavioral changes precede memory loss
```

## Extra 10: Non-ADRD Exclusions (CRITICAL)
```
NON-ADRD CAUSES TO EXCLUDE:
- DELIRIUM: acute onset, fluctuating, inattention (rule out first)
- PSYCHIATRIC: depression/pseudodementia (but can BE early ADRD)
- MEDICAL: hypothyroidism, B12 deficiency, infections, medications

Workup: CBC, CMP, TSH, B12, MRI/CT
```

## Extra 11: Mixed Dementia
```
MIXED DEMENTIA:
Multiple pathologies VERY COMMON in elderly (>50% over age 85).
Common: AD + vascular, AD + Lewy body
Pure single-etiology dementia is often the exception.
```

## Extra 12: Clinical Judgment (CRITICAL)
```
CLINICAL JUDGMENT PRINCIPLES:
1. Screening thresholds are NOT diagnostic cutoffs
2. Missing documentation ≠ absence of impairment
3. Functional impairment distinguishes MCI from dementia
4. Psychiatric symptoms can BE early ADRD manifestation
5. Multiple pathologies common in elderly
6. Rule out delirium and non-ADRD causes first
7. No single test determines diagnosis
8. Integrate ALL clinical information with expert judgment
```

---

# PART V: KEY CLINICAL DISTINCTIONS

## MCI vs Dementia

| Feature | MCI | Dementia |
|---------|-----|----------|
| Cognitive impairment | Present | Present |
| Functional independence | **PRESERVED** | **IMPAIRED** |
| CDR Global Score | 0.5 | ≥1.0 |
| Lawton IADL | 8/8 | <8/8 |
| Domain count | ≥1 | ≥2 |

## Why Computational Functions Matter

1. **Score Interpretation:** LLMs inconsistent with MoCA 24 vs 26 cutoffs
2. **CDR 0.5 Ambiguity:** Pure LLM may misclassify as dementia
3. **Functional Assessment:** Lawton IADL critical for boundary
4. **Domain Counting:** NIA-AA requires systematic evaluation
5. **Criteria Integration:** All 5 criteria must be verified

---

*ClinOrchestra: Configurable Clinical NLP Platform*
*Medical University of South Carolina, Biomedical Informatics Center*
