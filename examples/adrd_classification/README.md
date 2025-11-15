# ADRD Classification Example

**Task**: Binary classification of Alzheimer's Disease and Related Dementias (ADRD) vs Non-ADRD from clinical notes

**Type**: Classification (diagnostic prediction)

**Domain**: Geriatric Neurology, Neuropsychology, Dementia Care

---

## 📋 Task Overview

### Classification Goals

This example demonstrates how to use ClinOrchestra for **binary classification** of patients into:
- **ADRD**: Alzheimer's Disease, Vascular Dementia, Lewy Body Dementia, Frontotemporal Dementia, Mixed Dementia, or other neurodegenerative dementias
- **Non-ADRD**: Mild Cognitive Impairment, Depression, Delirium, Medication-induced impairment, Metabolic causes, or Normal aging

### Clinical Importance

- **Alzheimer's Disease** affects ~6.7 million Americans (2023)
- **ADRD** is the 7th leading cause of death in the US
- Early, accurate diagnosis enables:
  - Appropriate treatment (cholinesterase inhibitors, memantine)
  - Family planning and support
  - Clinical trial enrollment
  - Ruling out reversible causes

### Why This Task is Complex

1. **Multiple dementia subtypes** with overlapping symptoms (AD, VaD, LBD, FTD)
2. **Challenging differentials**: Depression, delirium, MCI, normal aging
3. **Multidomain assessment**: Cognitive, functional, behavioral, neurological, imaging
4. **Biomarker integration**: CSF, PET imaging when available
5. **Longitudinal progression** patterns distinguish types
6. **Reversible causes** must be ruled out (B12, thyroid, medications)

---

## 📁 Directory Structure

```
adrd_classification/
├── prompts/
│   ├── main_prompt.txt              # Comprehensive ADRD classification prompt
│   ├── minimal_prompt.txt           # Simplified fallback prompt
│   └── rag_refinement_prompt.txt    # Guideline-based refinement prompt
│
├── schemas/
│   └── adrd_classification_schema.json  # JSON schema for classification output
│
├── extras/
│   └── adrd_keywords.txt            # Domain-specific clinical keywords (400+)
│
├── functions/
│   └── adrd_functions.py            # Custom calculation functions
│
├── patterns/
│   └── adrd_patterns.txt            # Regex patterns for extracting scores/findings
│
├── rag_resources/
│   └── guidelines_and_references.md  # Clinical guidelines for RAG retrieval
│
└── README.md                        # This file
```

---

## 🔧 Setup Instructions

### 1. Install Functions

Register the ADRD-specific functions in ClinOrchestra:

```python
from core.function_registry import register_function
from examples.adrd_classification.functions.adrd_functions import (
    calculate_mmse_severity,
    calculate_moca_severity,
    calculate_cdr_severity,
    calculate_vascular_risk_score,
    assess_functional_independence
)

# Register functions
register_function("calculate_mmse_severity", calculate_mmse_severity)
register_function("calculate_moca_severity", calculate_moca_severity)
register_function("calculate_cdr_severity", calculate_cdr_severity)
register_function("calculate_vascular_risk_score", calculate_vascular_risk_score)
register_function("assess_functional_independence", assess_functional_independence)
```

### 2. Configure Extras

Load ADRD-specific keywords:

```python
extras_manager.load_keywords_from_file(
    "/home/user/clinorchestra/examples/adrd_classification/extras/adrd_keywords.txt"
)
```

### 3. Configure RAG

Set up RAG with clinical guidelines:

```python
# Option 1: Load from file
rag_engine.ingest_documents([
    "/home/user/clinorchestra/examples/adrd_classification/rag_resources/guidelines_and_references.md"
])

# Option 2: Use external medical knowledge bases
# - PubMed articles on ADRD diagnostic criteria
# - Clinical practice guidelines (NIA-AA, DSM-5)
# - Textbook chapters on dementia evaluation
```

### 4. Load Patterns

```python
from core.regex_preprocessor import RegexPreprocessor

preprocessor = RegexPreprocessor()
preprocessor.load_patterns_from_file(
    "/home/user/clinorchestra/examples/adrd_classification/patterns/adrd_patterns.txt"
)
```

### 5. Configure Prompts and Schema

```python
# Load main prompt
with open("examples/adrd_classification/prompts/main_prompt.txt") as f:
    main_prompt = f.read()

# Load minimal prompt (fallback)
with open("examples/adrd_classification/prompts/minimal_prompt.txt") as f:
    minimal_prompt = f.read()

# Load RAG refinement prompt (optional)
with open("examples/adrd_classification/prompts/rag_refinement_prompt.txt") as f:
    rag_prompt = f.read()

# Load schema
import json
with open("examples/adrd_classification/schemas/adrd_classification_schema.json") as f:
    schema = json.load(f)

# Configure ClinOrchestra
app_state.set_prompt_config(
    main_prompt=main_prompt,
    minimal_prompt=minimal_prompt,
    use_minimal=True,
    json_schema=schema,
    rag_prompt=rag_prompt  # For Stage 4 refinement
)
```

---

## 🎯 Classification Workflow

### STRUCTURED Pipeline (4 Stages)

```
Stage 1: TASK ANALYSIS
├─ LLM analyzes clinical note
├─ Plans which tools to use:
│  ├─ Functions: calculate_mmse_severity, calculate_vascular_risk_score
│  ├─ RAG: Query "NIA-AA Alzheimer criteria", "Lewy body dementia features"
│  └─ Extras: Extract keywords like "memory loss", "MMSE score", "atrophy"

Stage 2: TOOL EXECUTION (Parallel)
├─ Execute cognitive test interpretation functions
├─ Retrieve clinical guidelines from RAG
└─ Extract ADRD-specific terms with Extras

Stage 3: EXTRACTION
├─ LLM generates comprehensive classification
└─ Structured output per JSON schema

Stage 4: RAG REFINEMENT (Optional)
├─ Validate against NIA-AA, DSM-5 criteria
├─ Enhance diagnostic precision with guidelines
└─ Refine confidence assessment
```

### ADAPTIVE Pipeline (Iterative)

```
Iteration 1:
├─ LLM reads clinical note
├─ Requests tools (MMSE interpretation, RAG queries)
└─ Receives results

Iteration 2:
├─ LLM analyzes with tool results
├─ May request additional tools (vascular risk, functional assessment)
└─ Continues until confident

Final Iteration:
└─ Outputs classification with supporting evidence
```

---

## 📊 Schema Output Structure

### Top-Level Sections

1. **cognitive_assessment**:
   - Formal testing (MMSE, MoCA, CDR scores)
   - Cognitive domains (memory, language, executive, visuospatial)
   - Progression pattern (onset type, rate, duration)

2. **functional_assessment**:
   - ADL status (bathing, dressing, toileting, etc.)
   - IADL status (finances, medications, shopping, etc.)
   - Functional decline presence and severity

3. **clinical_features**:
   - Behavioral/psychiatric symptoms (hallucinations, agitation, apathy)
   - Neurological examination (parkinsonism, focal deficits)
   - Neuroimaging (atrophy patterns, vascular changes, PET findings)

4. **diagnostic_evaluation**:
   - Reversible causes ruled out (B12, thyroid, metabolic)
   - CSF biomarkers (if available)
   - Vascular risk factors
   - Family history

5. **classification**:
   - **diagnosis**: "ADRD" or "Non-ADRD"
   - **specific_diagnosis**: AD, VaD, LBD, FTD, MCI, Depression, etc.
   - **severity**: mild, moderate, severe (if ADRD)
   - **confidence**: high, moderate, low
   - **key_supporting_evidence**: Top findings supporting classification
   - **evidence_against_alternative**: Findings ruling out opposite diagnosis

6. **clinical_reasoning**:
   - Diagnostic criteria met (NIA-AA, DSM-5)
   - Differential diagnoses considered and ruled out
   - Atypical features, missing data
   - Confidence factors

---

## 🔍 Function Descriptions

### `calculate_mmse_severity(mmse_score)`
Interprets Mini-Mental State Examination score:
- 24-30: Normal or mild impairment
- 18-23: Mild dementia
- 10-17: Moderate dementia
- 0-9: Severe dementia

**Use when**: MMSE score mentioned in clinical note

### `calculate_moca_severity(moca_score, education_years=None)`
Interprets Montreal Cognitive Assessment with education correction:
- ≥26: Normal (add 1 point if education ≤12 years)
- 18-25: MCI range
- <18: Dementia range

**Use when**: MoCA score documented (more sensitive than MMSE for MCI)

### `calculate_cdr_severity(cdr_global_score)`
Interprets Clinical Dementia Rating:
- 0: No dementia
- 0.5: Very mild/questionable
- 1: Mild
- 2: Moderate
- 3: Severe

**Use when**: CDR score available (gold standard for staging)

### `calculate_vascular_risk_score(hypertension, diabetes, ...)`
Calculates vascular dementia risk based on cardiovascular risk factors:
- Low risk: 0 factors
- Moderate risk: 1-2 factors
- High risk: 3+ factors

**Use when**: Assessing vascular contribution to cognitive impairment

### `assess_functional_independence(adl_independent, iadl_independent)`
Assesses functional status and implications for dementia diagnosis:
- Both intact → Does not meet functional criteria for dementia
- IADL impaired, ADL intact → Consistent with mild dementia/MCI
- ADL impaired → Moderate-severe dementia

**Use when**: Determining if functional decline meets dementia threshold

---

## 📚 RAG Resources

### Guidelines Included

1. **NIA-AA Criteria** (2011, 2018): Alzheimer's disease diagnostic criteria
2. **DSM-5**: Major and mild neurocognitive disorder
3. **NINDS-AIREN**: Vascular dementia criteria
4. **DLB Consensus (2017)**: Lewy body dementia features
5. **Rascovsky FTD Criteria**: Frontotemporal dementia
6. **Cognitive Assessment Guidelines**: MMSE, MoCA, CDR interpretation
7. **Biomarker Guidelines**: CSF, PET imaging interpretation
8. **Functional Scales**: Katz ADL, Lawton IADL, FAQ
9. **Differential Diagnosis**: Delirium vs dementia, depression vs dementia, MCI criteria

### RAG Query Suggestions

**For general ADRD**:
- "NIA-AA criteria for probable Alzheimer's disease"
- "DSM-5 major neurocognitive disorder criteria"
- "Biomarker interpretation for Alzheimer's diagnosis"

**For specific subtypes**:
- "Vascular dementia NINDS-AIREN criteria stepwise decline"
- "Lewy body dementia visual hallucinations REM sleep behavior"
- "Frontotemporal dementia behavioral disinhibition apathy"

**For differential diagnosis**:
- "Distinguish delirium from dementia acute vs insidious onset"
- "Depression versus dementia pseudodementia features"
- "Mild cognitive impairment MCI vs early dementia criteria"

**For assessment tools**:
- "MMSE interpretation severity staging cutoffs"
- "MoCA education correction scoring guidelines"
- "CDR Clinical Dementia Rating global score interpretation"

---

## ⚙️ Extras Keywords

**400+ domain-specific terms** organized by category:

- **Cognitive Assessment**: MMSE, MoCA, CDR, neuropsych testing
- **Functional Assessment**: ADL, IADL, independence, assistance
- **Dementia Types**: Alzheimer, vascular, Lewy body, frontotemporal, MCI
- **Cognitive Symptoms**: Memory loss, disorientation, aphasia, executive dysfunction
- **BPSD**: Agitation, hallucinations, sundowning, wandering
- **Neurological**: Parkinsonism, gait abnormality, primitive reflexes
- **Neuroimaging**: Atrophy, white matter changes, infarcts, PET findings
- **Biomarkers**: CSF amyloid, tau, PET amyloid/tau
- **Lab Workup**: B12, thyroid, metabolic panel
- **Vascular Risks**: Hypertension, diabetes, stroke, atrial fibrillation
- **Medications**: Donepezil, memantine, anticholinergics
- **Criteria**: NIA-AA, DSM-5, NINDS-AIREN, DLB consensus
- **Scales**: Katz, Lawton, FAQ, NPI
- **Differential Dx**: Delirium, depression, NPH, CJD

**How extras help**: Automatically identifies relevant sections of clinical notes containing diagnostic information

---

## 🔄 Pattern Matching

**200+ regex patterns** for extracting:

### Cognitive Scores
```regex
MMSE_SCORE: MMSE\s*(?:score|=|:)?\s*(\d{1,2})\s*(?:/\s*30)?
MOCA_SCORE: MoCA\s*(?:score|=|:)?\s*(\d{1,2})\s*(?:/\s*30)?
CDR_SCORE: CDR\s*(?:global|score)?\s*(0|0\.5|1|2|3)
```

### Progression Patterns
```regex
INSIDIOUS_ONSET: insidious(?:ly)? onset|gradual(?:ly)? onset
STEPWISE_DECLINE: step(?:-)?wise (?:decline|progression)
FLUCTUATING_COGNITION: fluctuating (?:cognition|cognitive)
```

### Neuroimaging Findings
```regex
HIPPOCAMPAL_ATROPHY: hippocampal? atrophy
WHITE_MATTER_HYPERINTENSITIES: white matter (?:hyperintensities|disease|WMH)
LACUNAR_INFARCT: lacunar (?:infarcts?|stroke)
```

### Behavioral Symptoms
```regex
VISUAL_HALLUCINATIONS: (?:visual hallucinations?|seeing things)
REM_SLEEP_BEHAVIOR: REM sleep behavior (?:disorder|disturbance)
SUNDOWNING: sundowning|worse (?:in the|at) (?:evening|night)
```

**How patterns help**: Automatically extract structured data from unstructured clinical notes

---

## 💡 Usage Examples

### Example 1: Probable Alzheimer's Disease

**Input Clinical Note**:
```
78-year-old female with 3-year history of progressive memory loss.
MMSE 18/30. MoCA 15/30. Struggles with managing finances and medications
(IADL impaired). Still independent in bathing and dressing (ADL intact).
Gradual onset, slowly progressive. MRI shows bilateral hippocampal atrophy.
Normal B12, TSH. No focal neurological deficits.
```

**Expected Classification**:
```json
{
  "classification": {
    "diagnosis": "ADRD",
    "specific_diagnosis": "Probable Alzheimer's Disease",
    "severity": "mild",
    "confidence": "high",
    "key_supporting_evidence": [
      "Insidious onset with gradual progression over 3 years",
      "MMSE 18/30 and MoCA 15/30 consistent with mild dementia",
      "Prominent memory impairment with IADL loss, ADLs preserved",
      "Bilateral hippocampal atrophy on MRI (classic AD pattern)",
      "Reversible causes ruled out (normal B12, TSH)"
    ]
  }
}
```

### Example 2: Lewy Body Dementia

**Input Clinical Note**:
```
72-year-old male with 2-year history of cognitive fluctuations, worse in
evenings. Frequent visual hallucinations (sees children in room).
Mild parkinsonism (rigidity, bradykinesia). REM sleep behavior disorder
(acts out dreams). MoCA 20/30. Functional decline in complex tasks.
```

**Expected Classification**:
```json
{
  "classification": {
    "diagnosis": "ADRD",
    "specific_diagnosis": "Probable Lewy Body Dementia",
    "severity": "mild",
    "confidence": "high",
    "key_supporting_evidence": [
      "Meets 4/4 core DLB criteria: fluctuating cognition, visual hallucinations, REM sleep behavior disorder, parkinsonism",
      "Visual hallucinations early and prominent (highly specific for LBD)",
      "REM sleep behavior disorder often precedes cognitive symptoms in LBD",
      "MoCA 20/30 with functional decline meets dementia criteria",
      "Presentation consistent with 2017 DLB Consensus Criteria"
    ]
  }
}
```

### Example 3: Non-ADRD (Depression)

**Input Clinical Note**:
```
65-year-old female with subjective memory complaints for 3 months.
Depressed mood, anhedonia, poor sleep. MMSE 28/30 (normal). MoCA 27/30
(normal). Fully independent in all ADLs and IADLs. PHQ-9 score 18 (moderately
severe depression). Emphasizes memory problems but objective testing normal.
```

**Expected Classification**:
```json
{
  "classification": {
    "diagnosis": "Non-ADRD",
    "specific_diagnosis": "Major Depressive Disorder with cognitive complaints (pseudodementia)",
    "confidence": "moderate",
    "evidence_against_alternative": [
      "Normal cognitive testing (MMSE 28/30, MoCA 27/30) despite subjective complaints",
      "Fully independent in ADLs and IADLs (no functional decline)",
      "Prominent depressive symptoms (PHQ-9 18)",
      "Acute onset (3 months) atypical for neurodegenerative dementia",
      "Pattern more consistent with depression-related cognitive complaints than true dementia"
    ]
  }
}
```

---

## 🎓 Best Practices

### 1. Data Requirements

**Minimum required**:
- Clinical history with timeline
- At least one formal cognitive assessment (MMSE, MoCA, or CDR)
- Functional status (ADL/IADL)
- Basic medical history

**Optimal for high confidence**:
- Longitudinal data (progression over time)
- Multiple cognitive assessments
- Neuroimaging (MRI or CT)
- Lab workup ruling out reversible causes
- Biomarkers (CSF or PET) for atypical cases

### 2. Interpreting Confidence Levels

**High Confidence**:
- Meets formal diagnostic criteria (NIA-AA, DSM-5)
- Objective cognitive testing confirms impairment
- Clear functional decline from baseline
- Imaging supports diagnosis
- Reversible causes excluded

**Moderate Confidence**:
- Clinical syndrome present but limited biomarker data
- Some atypical features
- Missing key diagnostic information
- Conflicting findings

**Low Confidence**:
- Insufficient data for confident diagnosis
- Highly atypical presentation
- Significant missing information (no cognitive testing, no imaging)
- Multiple conflicting findings

### 3. Common Pitfalls to Avoid

❌ **Over-relying on a single finding**
✓ Use comprehensive assessment across multiple domains

❌ **Ignoring functional status**
✓ Dementia requires functional impairment; intact function → MCI or no dementia

❌ **Missing reversible causes**
✓ Always consider B12 deficiency, hypothyroidism, medications, depression

❌ **Confusing delirium with dementia**
✓ Delirium = acute onset, fluctuating; Dementia = insidious, progressive

❌ **Not considering mixed pathology**
✓ Many elderly have both AD and vascular pathology (mixed dementia)

### 4. Quality Checks

- **Criteria alignment**: Does classification match NIA-AA, DSM-5, or consensus criteria?
- **Internal consistency**: Do cognitive, functional, and imaging findings align?
- **Alternative explanations**: Were differential diagnoses adequately considered?
- **Evidence quality**: Is supporting evidence specific and quoted from clinical note?
- **Confidence calibration**: Does confidence level match strength of evidence?

---

## 📈 Performance Optimization

### For STRUCTURED Pipeline

1. **Enable all relevant tools**:
   - Functions for score interpretation
   - RAG for guideline retrieval
   - Extras for keyword extraction

2. **Stage 4 RAG refinement**:
   - Queries specific guidelines based on initial classification
   - Refines diagnostic precision
   - Validates against formal criteria

### For ADAPTIVE Pipeline

1. **Set appropriate budgets**:
   ```python
   app_state.set_agentic_config(
       enabled=True,
       max_iterations=10,
       max_tool_calls=100
   )
   ```

2. **LLM will iteratively**:
   - Request cognitive test interpretation
   - Query guidelines for criteria validation
   - Extract relevant keywords
   - Build evidence-based classification

---

## 🔬 Advanced Features

### Subtype Discrimination

The prompts guide LLM to distinguish:
- **AD** (insidious, memory-predominant, medial temporal atrophy)
- **VaD** (stepwise, executive dysfunction, vascular imaging)
- **LBD** (fluctuating, visual hallucinations, parkinsonism, REM sleep disorder)
- **FTD** (behavioral changes, young onset, frontal atrophy)
- **Mixed** (features of multiple types, common in elderly)

### Biomarker Integration

When CSF or PET data available, schema captures:
- CSF Aβ42, total tau, phospho-tau
- Amyloid PET status (positive/negative)
- Tau PET distribution
- FDG-PET hypometabolism pattern

Prompts instruct A/T/N framework application per NIA-AA 2018

### Severity Staging

Classification includes:
- **Mild** (CDR 0.5-1, MMSE 21-26): Noticeable symptoms, some IADL impairment
- **Moderate** (CDR 2, MMSE 10-20): Clear deficits, needs supervision, ADL impairment
- **Severe** (CDR 3, MMSE <10): Profound impairment, total care needed

---

## 📖 References

See `rag_resources/guidelines_and_references.md` for comprehensive citations

**Key References**:
1. McKhann et al. (2011) - NIA-AA Alzheimer's criteria
2. Jack et al. (2018) - NIA-AA Research Framework (A/T/N)
3. American Psychiatric Association (2013) - DSM-5
4. McKeith et al. (2017) - DLB Consensus Criteria
5. Rascovsky et al. (2011) - FTD Diagnostic Criteria
6. Roman et al. (1993) - NINDS-AIREN Vascular Dementia Criteria

---

## 🤝 Contributing

To improve this ADRD classification example:
1. Add more clinical note examples
2. Expand RAG resources with recent guidelines
3. Create additional function definitions
4. Enhance pattern matching rules
5. Test with real-world clinical data

---

## 📞 Support

For questions about this example:
- Review `rag_resources/guidelines_and_references.md` for clinical guidance
- Check ClinOrchestra main documentation for technical setup
- Consult published diagnostic criteria for clinical validation

---

**Created by**: ClinOrchestra Development Team
**Version**: 1.0.0
**Last Updated**: 2025-01-15
**License**: MIT
