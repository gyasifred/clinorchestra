"""
ADRD Classification Prompts for Conversational AI Training Data Annotation

These prompts generate natural clinical narratives - like a cognitive neurologist
dictating to a colleague about a patient's dementia evaluation.

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
"""

# =============================================================================
# ADRD MAIN PROMPT - Full Expert Clinical Narrative
# =============================================================================

ADRD_MAIN_PROMPT = """[ADRD CLASSIFICATION - Expert Clinical Data Curation for Conversational AI]

You are a board-certified cognitive neurologist curating training data for a conversational AI system. Your output should read like you're dictating a comprehensive evaluation to a colleague - natural, expert clinical language with complete diagnostic reasoning.

**PRIMARY DIRECTIVE:** Extract and synthesize all available clinical information to support the ground truth classification. Use natural clinical conversation, not bullet points or forms.

**GROUND TRUTH CLASSIFICATION (YOU MUST SUPPORT):**
{label_context}

**CLINICAL TEXT TO ANALYZE:**
{clinical_text}

**CRITICAL WORKFLOW:**

**STEP 1 - ESTABLISH THE CLINICAL PICTURE:**
Begin by describing who this patient is and why they're being evaluated. Set the stage:
- Age, presenting complaint, referral reason
- Timeline of symptoms (onset, duration, progression)
- Who provided the history (patient, family, both)
- Key context that shapes your evaluation

**STEP 2 - COGNITIVE ASSESSMENT:**
Discuss cognitive findings like you're presenting at a case conference:

A. FORMAL TESTING:
- Report scores naturally: "Her MMSE was 22 out of 30, losing points primarily on recall and orientation..."
- Interpret meaningfully: "The MoCA at 18 suggests more significant impairment than the MMSE captured..."
- Note CDR staging if available: "CDR of 1 places her in the mild dementia range..."

B. COGNITIVE DOMAINS:
For each affected domain, describe deficits conversationally:
- Memory: "Most striking is her episodic memory impairment - she couldn't recall any of three words at five minutes, even with cueing..."
- Language: "Word-finding difficulties are evident - she called a watch 'the time thing' and struggled with category fluency..."
- Executive: "She has trouble with multi-step tasks - her daughter says she burned several meals before they took over cooking..."
- Visuospatial: "Clock drawing showed poor planning with numbers clustered..."
- Attention: "Attention seems relatively preserved on digit span..."

C. PROGRESSION PATTERN:
This is CRITICAL for differential diagnosis:
- "The family describes an insidious onset over roughly two years, with gradually progressive decline - classic for Alzheimer's..."
- "The stepwise pattern with periods of stability between clear worsenings points toward vascular etiology..."
- "Fluctuating cognition with marked variations day-to-day is highly suggestive of Lewy body disease..."

**STEP 3 - FUNCTIONAL ASSESSMENT:**
Describe impact on daily life - this determines whether we're dealing with MCI vs dementia:

IADLs (lost earlier):
- "She stopped managing her own finances about a year ago after paying bills twice and missing others..."
- "Medication management became problematic - she was found to have weeks of pills still in the organizer..."
- "She got lost driving to her church of 30 years, which prompted this evaluation..."

ADLs (lost later):
- "She still bathes and dresses independently, though her daughter notes she sometimes wears the same clothes..."
- "Eating is independent once meals are prepared..."

CRITICAL DETERMINATION: Does cognitive impairment interfere with independence?
- "The cognitive deficits clearly interfere with her ability to live independently - she requires daily supervision and assistance with complex activities..."

**STEP 4 - CLINICAL FEATURES:**

A. BEHAVIORAL/PSYCHIATRIC SYMPTOMS:
Describe any BPSD naturally:
- "She's become apathetic, spending most days sitting in her chair with little initiative..."
- "The family reports increased anxiety and some paranoid ideation - she accused them of stealing her jewelry..."
- "No visual hallucinations or REM sleep behavior disorder to suggest Lewy body disease..."

B. NEUROLOGICAL EXAMINATION:
- "Motor exam shows subtle parkinsonian features - mild cogwheeling at the right wrist, slightly reduced arm swing..."
- "Gait is slow and slightly wide-based but without frank ataxia..."
- "No focal deficits to suggest stroke..."

C. NEUROIMAGING:
Interpret findings diagnostically:
- "MRI shows moderate hippocampal atrophy bilaterally, worse on the left, with Fazekas grade 2 white matter changes..."
- "The medial temporal atrophy pattern supports AD pathology, while the white matter disease suggests a mixed picture..."
- "Amyloid PET was positive, confirming amyloid deposition consistent with AD..."

**STEP 5 - DIAGNOSTIC EVALUATION:**

A. REVERSIBLE CAUSES:
- "We've ruled out the treatable causes - B12 and folate are normal, TSH is 1.8, metabolic panel unremarkable..."
- "RPR negative, HIV testing declined..."

B. VASCULAR RISK FACTORS:
- "Her vascular risk burden is significant - hypertension for 20 years, diabetes for 10, former smoker..."
- "No history of clinical stroke or TIA, though imaging shows silent infarcts..."

C. FAMILY HISTORY:
- "Her mother had 'senility' in her 80s, her sister was diagnosed with Alzheimer's at 72..."
- "This positive family history, while not early-onset, does increase her risk..."

**STEP 6 - CLASSIFICATION AND REASONING:**
This is where you synthesize everything to support the ground truth:

FOR ADRD CLASSIFICATION:
State the diagnosis with specificity and evidence:
- "This is ADRD - specifically, probable Alzheimer's disease dementia with possible vascular contribution..."
- "The clinical picture meets NIA-AA criteria for probable AD dementia: insidious onset, progressive amnestic presentation, exclusion of other causes..."
- "DSM-5 criteria for major neurocognitive disorder due to Alzheimer's disease are satisfied..."

Key supporting evidence (cite 3-5 strongest points):
1. "Progressive amnestic syndrome over 2+ years with impaired delayed recall"
2. "Functional decline interfering with independence (finances, medications, driving)"
3. "MRI with medial temporal atrophy pattern"
4. "Positive amyloid PET"
5. "Exclusion of reversible causes"

Evidence against Non-ADRD:
- "This is not MCI - functional impairment is clearly present"
- "Not primarily vascular - no stroke history, gradual not stepwise progression"
- "Not depression - no significant mood symptoms, poor response to prior SSRI trial"

FOR NON-ADRD CLASSIFICATION:
State why this is NOT dementia and what it IS:
- "This is Non-ADRD - the presentation is most consistent with Mild Cognitive Impairment, amnestic type..."
- "While cognitive testing shows impairment, she remains functionally independent - she manages her own finances, medications, and drives safely..."
- "Alternatively, this represents pseudodementia due to undertreated depression with cognitive symptoms..."

Key supporting evidence:
1. "Preserved functional independence despite subjective complaints"
2. "Scores in MCI range (MoCA 24) not dementia range"
3. "No progression over 6-month follow-up"
4. "Significant depressive symptoms with vegetative features"

**STEP 7 - DIFFERENTIAL DIAGNOSES:**
Consider alternatives like a good clinician:
- "I considered vascular dementia given her risk factors and white matter disease, but the gradual progressive course and amnestic predominance favor AD..."
- "Lewy body dementia was considered but there are no fluctuations, hallucinations, or parkinsonism..."
- "Depression can certainly impair cognition, but her presentation predates any mood symptoms and she didn't improve with treatment..."

**STEP 8 - CONFIDENCE AND LIMITATIONS:**
Be honest about uncertainty:
- "I have high confidence in this diagnosis given the classic presentation and supportive biomarkers..."
- "Confidence is moderate - while the clinical picture is suggestive, we lack biomarker confirmation..."
- "Longitudinal follow-up would strengthen this diagnosis - I'd want to see continued progression over the next 6-12 months..."

**SYNTHESIS STRUCTURE:**
Organize your response into these flowing narrative sections:

1. **CASE PRESENTATION**: Who, why, timeline, context
2. **COGNITIVE PROFILE**: Testing, domains, progression - the cognitive story
3. **FUNCTIONAL STATUS**: IADLs, ADLs, independence determination
4. **BEHAVIORAL & NEUROLOGICAL**: BPSD, exam findings, imaging
5. **DIAGNOSTIC WORKUP**: Labs, biomarkers, risk factors, family history
6. **CLASSIFICATION & REASONING**: Diagnosis, criteria, evidence for and against
7. **DIFFERENTIAL**: Alternatives considered and excluded
8. **CLINICAL INSIGHTS**: Confidence, prognosis, follow-up needs

**CRITICAL RULES:**
- Write as natural clinical narrative, not structured lists
- Sound like you're discussing the case with a colleague
- Support the ground truth classification with evidence
- Apply formal criteria (NIA-AA, DSM-5) and cite them
- Be specific with evidence: "MMSE 22" not "impaired cognition"
- Consider the differential diagnosis systematically
- Acknowledge uncertainty and missing information
- ANONYMIZE: Use "the patient", "this [age]-year-old", "the family"

[END TASK DESCRIPTION]

{rag_outputs}

{function_outputs}

{extras_outputs}

{json_schema_instructions}"""


# =============================================================================
# ADRD MINIMAL PROMPT - Concise Version
# =============================================================================

ADRD_MINIMAL_PROMPT = """[ADRD CLASSIFICATION - Expert Clinical Annotation]

Cognitive neurologist curating AI training data. Natural clinical narrative supporting ground truth classification.

**GROUND TRUTH:**
{label_context}

**DIRECTIVE:** Extract ADRD-relevant information in conversational clinical style. Use NIA-AA and DSM-5 criteria.

**SYNTHESIS (8 sections, natural narrative):**

1. **CASE**: Patient context, presenting complaint, symptom timeline, history source
2. **COGNITIVE**: Test scores (MMSE, MoCA, CDR) with interpretation. Domain-specific deficits (memory, language, executive, visuospatial, attention). Progression pattern (insidious/acute, gradual/stepwise/fluctuating)
3. **FUNCTIONAL**: IADLs (finances, medications, driving), ADLs. Critical: Does impairment interfere with independence?
4. **BEHAVIORAL/NEURO**: BPSD symptoms, motor findings, neuroimaging with diagnostic interpretation
5. **WORKUP**: Reversible causes ruled out, vascular risk factors, biomarkers if available, family history
6. **CLASSIFICATION**:
   - FOR ADRD: Specific type (AD, VaD, LBD, FTD, Mixed), severity, criteria met
   - FOR Non-ADRD: Alternative (MCI, depression, delirium, other), reasoning
7. **DIFFERENTIAL**: Alternatives considered, why excluded
8. **INSIGHTS**: Confidence level, supporting/limiting factors, prognosis

**RULES:**
- Natural clinical conversation, not bullet lists
- Support ground truth with specific evidence
- Cite criteria: "NIA-AA probable AD criteria met..."
- Acknowledge limitations: "Biomarker confirmation would strengthen..."
- ANONYMIZE: "the patient", "this [age]-year-old"

[END]

**CLINICAL TEXT:**
{clinical_text}

{rag_outputs}

{function_outputs}

{extras_outputs}

{json_schema_instructions}"""


# =============================================================================
# ADRD RAG REFINEMENT PROMPT
# =============================================================================

ADRD_RAG_REFINEMENT_PROMPT = """[ADRD RAG REFINEMENT - Evidence-Based Enhancement]

Refining ADRD classification using retrieved diagnostic criteria and guidelines. Enhance clinical narrative with formal criteria validation.

**GROUND TRUTH (MUST SUPPORT):**
{label_context}

**REFINEMENT OBJECTIVES:**

1. **VALIDATE CRITERIA APPLICATION:**
   - Verify NIA-AA criteria for AD are correctly applied
   - Check DSM-5 Major/Mild NCD criteria
   - For specific subtypes (VaD, LBD, FTD): confirm consensus criteria
   - Quote criteria: "Per NIA-AA, probable AD requires insidious onset, documented decline, and amnestic or non-amnestic presentation..."

2. **STRENGTHEN DIAGNOSTIC REASONING:**
   - Add guideline references to support classification
   - Enhance specificity: "CDR 1 = mild dementia per Morris criteria..."
   - Incorporate retrieved evidence about cutoffs, patterns, biomarkers

3. **ENHANCE DIFFERENTIAL DIAGNOSIS:**
   - Use guidelines to systematically exclude alternatives
   - For VaD: Apply NINDS-AIREN or VASCOG criteria
   - For LBD: Apply McKeith criteria (fluctuations, visual hallucinations, parkinsonism, RBD)
   - For FTD: Apply Rascovsky criteria (behavioral) or Gorno-Tempini (language)

4. **VALIDATE FUNCTIONAL IMPACT:**
   - MCI vs Dementia hinges on functional independence
   - Dementia = "significant interference with independence in everyday activities"
   - MCI = "minimal impairment in complex instrumental functions"

5. **REFINE SEVERITY STAGING:**
   - Per CDR: 0.5 = very mild/MCI, 1 = mild, 2 = moderate, 3 = severe
   - Per MMSE: 20-24 = mild, 13-20 = moderate, <12 = severe
   - Ensure consistency between score interpretation and stated severity

6. **CORRECT MISCLASSIFICATIONS:**
   - If initial extraction misapplied criteria, correct with guideline citation
   - If severity mismatch, adjust with scoring reference
   - If wrong subtype suggested, apply appropriate consensus criteria

**CRITICAL PRINCIPLES:**
- Preserve accurate information from initial extraction
- Quote guidelines when correcting: "Per DSM-5..."
- Flag inconsistencies: "MMSE of 22 suggests mild dementia, but CDR 0.5 suggests MCI..."
- Maintain natural clinical narrative tone
- ANONYMIZE throughout

**SYNTHESIS GUIDELINES:**

FOR ADRD CLASSIFICATION:
- Confirm specific subtype with criteria (AD, VaD, LBD, FTD, Mixed)
- Validate severity staging against formal cutoffs
- Ensure biomarkers align with diagnosis if available
- Strengthen evidence for excluding alternatives

FOR NON-ADRD CLASSIFICATION:
- Confirm why dementia criteria not met (preserved function, stable course)
- Validate alternative diagnosis (MCI stage, psychiatric differential)
- Ensure "normal" doesn't mean "no concerns" - describe what IS present
- Apply MCI criteria specifically if applicable

[END REFINEMENT TASK]

**ORIGINAL TEXT:**
{clinical_text}

**INITIAL EXTRACTION:**
{stage3_json_output}

**EVIDENCE BASE:**
{retrieved_evidence_chunks}

{json_schema_instructions}

Return ONLY JSON. No markdown. Refine using guideline evidence to strengthen diagnostic accuracy."""


# =============================================================================
# SIMPLIFIED SCHEMA FOR CONVERSATIONAL OUTPUT
# =============================================================================

ADRD_CONVERSATIONAL_SCHEMA = {
    "case_presentation": {
        "type": "string",
        "description": "Natural narrative: Patient context, presenting complaint, symptom onset/duration, history source, referral reason. Set the clinical stage.",
        "required": True
    },
    "cognitive_assessment": {
        "type": "string",
        "description": "Natural narrative: Formal test scores (MMSE, MoCA, CDR) with interpretation. Domain-specific deficits (memory, language, executive, visuospatial, attention) with evidence. Progression pattern (insidious/acute, gradual/stepwise/fluctuating, duration).",
        "required": True
    },
    "functional_status": {
        "type": "string",
        "description": "Natural narrative: IADL status (finances, medications, driving, cooking). ADL status. CRITICAL: Does cognitive impairment interfere with independence? This determines MCI vs dementia.",
        "required": True
    },
    "behavioral_neurological": {
        "type": "string",
        "description": "Natural narrative: BPSD symptoms (apathy, depression, anxiety, hallucinations, delusions, sleep disturbances). Neurological exam (parkinsonism, gait, focal deficits). Neuroimaging findings with diagnostic interpretation (atrophy pattern, vascular changes, PET if available).",
        "required": True
    },
    "diagnostic_workup": {
        "type": "string",
        "description": "Natural narrative: Reversible causes workup (B12, TSH, metabolic). Vascular risk factors. Biomarkers if available (CSF, amyloid PET). Family history of dementia.",
        "required": True
    },
    "classification": {
        "type": "string",
        "description": "Natural narrative: State ADRD or Non-ADRD. If ADRD: specific type (AD, VaD, LBD, FTD, Mixed), severity (mild/moderate/severe), formal criteria met (NIA-AA, DSM-5). If Non-ADRD: alternative diagnosis (MCI, depression, delirium), supporting reasoning. Include 3-5 key supporting evidence points and evidence against alternative classification.",
        "required": True
    },
    "differential_diagnoses": {
        "type": "string",
        "description": "Natural narrative: Other diagnoses considered. For each: why considered, why excluded. Cover relevant differentials (other dementia subtypes, MCI, psychiatric, metabolic).",
        "required": True
    },
    "clinical_insights": {
        "type": "string",
        "description": "Natural narrative: Diagnostic confidence (high/moderate/low). Factors supporting and limiting confidence. Prognosis. Recommended follow-up or additional testing. Missing information that would strengthen diagnosis.",
        "required": True
    }
}


# =============================================================================
# TEMPLATE REGISTRY ENTRY
# =============================================================================

ADRD_TEMPLATE = {
    "main": ADRD_MAIN_PROMPT,
    "minimal": ADRD_MINIMAL_PROMPT,
    "rag_prompt": ADRD_RAG_REFINEMENT_PROMPT,
    "description": "ADRD Classification - Expert clinical narratives for dementia vs non-dementia annotation",
    "version": "1.0.0",
    "schema": ADRD_CONVERSATIONAL_SCHEMA
}


def get_adrd_prompts():
    """Return ADRD prompt templates for use in extraction"""
    return ADRD_TEMPLATE


def get_adrd_schema():
    """Return ADRD conversational schema"""
    return ADRD_CONVERSATIONAL_SCHEMA
