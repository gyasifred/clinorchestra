#!/usr/bin/env python3
"""
Prompt Templates Module - Central repository for all prompt templates
Professional Quality v1.0.0 - Natural clinical language with guideline-based interpretation

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Lab: ClinicalNLP Lab, Biomedical Informatics Center
Version: 1.0.4 - Natural conversation with guideline-based evidence synthesis
"""

from typing import Dict, Any, List

# ============================================================================
# NO HARDCODED CLINICAL CONTEXT - Universal Platform
# ============================================================================

# ============================================================================
# DEFAULT PROMPT TEMPLATES (main, minimal, RAG refinement)
# ============================================================================

DEFAULT_MAIN_PROMPT = """[TASK DESCRIPTION - Edit this section with your extraction task]

You are a clinical expert analyzing medical records for structured information extraction to curate training data for a conversational AI.

YOUR TASK:
Extract and synthesize clinical information from the provided text to create a comprehensive narrative, following the JSON schema exactly. The goal is to annotate data with expert-level clinical analysis, focusing on factual evidence aligned with the ICD classification.

EXTRACTION GUIDELINES:
- Be precise and accurate
- Use null for truly unknown values
- Maintain clinical terminology
- Extract only factual information present in the text
- ANONYMIZE: NEVER use patient or family names; ALWAYS use "the patient", "the [age]-year-old", or "the family"

FUNCTION CALLING RULE:
Ensure all function parameters are provided in correct units. Convert units (e.g., cm to m) using a conversion function before calling the primary function.

INTELLIGENT QUERIES:
Build RAG queries from clinical text and ICD classification, targeting relevant guidelines (e.g., "ASPEN malnutrition criteria"). Summarize extras outputs in two sentences, focusing on diagnostic insights.

[END TASK DESCRIPTION]

CLINICAL TEXT TO ANALYZE:
{clinical_text}

ICD CLASSIFICATION:
{label_context}

{rag_outputs}

{function_outputs}

{extras_outputs}

{json_schema_instructions}"""

DEFAULT_MINIMAL_PROMPT = """[TASK DESCRIPTION - Concise Version]

Extract and synthesize clinical information in JSON format to curate training data for a conversational AI, aligning with the ICD classification.

FUNCTION CALLING RULE:
Ensure all function parameters are provided in correct units. Convert units (e.g., cm to m) using a conversion function before calling the primary function.

INTELLIGENT QUERIES:
Build RAG queries from clinical text and ICD classification, targeting relevant guidelines (e.g., "ASPEN malnutrition criteria"). Summarize extras outputs in two sentences, focusing on diagnostic insights.

[END TASK DESCRIPTION]

CLINICAL TEXT:
{clinical_text}

ICD CLASSIFICATION:
{label_context}

{rag_outputs}

{function_outputs}

{extras_outputs}

{json_schema_instructions}"""

DEFAULT_RAG_REFINEMENT_PROMPT = """[RAG REFINEMENT TASK]

You are refining a preliminary clinical extraction using evidence from authoritative sources, acting as a clinical expert to curate high-quality training data for a conversational AI. Synthesize findings to support the ICD classification of nutritional status.

CRITICAL ANONYMIZATION:
- NEVER use patient or family names
- ALWAYS use: "the patient", "the [age]-year-old", "the family"

REFINEMENT OBJECTIVES:

1. VALIDATE: Confirm clinical interpretations against guideline criteria (e.g., ASPEN), classification accuracy, and diagnostic reasoning.

2. CORRECT: Adjust misclassifications with guideline citations, update inappropriate differentials, and align recommendations with evidence-based practice. Replace any patient names with "the patient", "the [age]-year-old", etc.

3. ENHANCE: Add guideline interpretations (e.g., "Per ASPEN criteria..."), prognostic information, and diagnostic criteria met.

4. FILL GAPS: Specify severity if parameters are present but unstated, include guideline-indicated recommendations, and complete evidence-supported care plan elements.

5. ENSURE CONSISTENCY: Verify assessments align with clinical findings, symptoms support the ICD classification, and recommendations match severity/presentation.

6. HANDLE MISSING DATA: For null, none, or not documented fields, provide a normal response. Example: If labs are null, state, "No laboratory investigations were ordered, consistent with clinical presentation."

CRITICAL PRINCIPLES:
- Preserve fidelity: Never remove correct data or fabricate information
- Quote guidelines when correcting: "Per WHO standards..."
- Flag discrepancies: "Parameters suggest severe, but initial extraction stated moderate"
- Add value only when guidelines clearly apply
- Maintain conversational expert tone
- ANONYMIZE: Use "the patient", "the [age]-year-old", "the family"

Z-SCORE AND PERCENTILE VALIDATION:
- Validate z-score sign with percentile:
  • Percentiles <50th: Negative z-scores (below average, e.g., "1 z 2.36" is -2.36)
  • Percentiles >50th: Positive z-scores (above average, e.g., "85 ile z 1.04" is +1.04)
- Confirm alignment with clinical descriptions (e.g., "short stature" indicates below average; "well-nourished" indicates normal/above average)
- Flag discrepancies between z-scores, percentiles, and clinical narrative for review

SYNTHESIS GUIDELINES FOR MALNUTRITION ASSESSMENT:
- For ICD classification "MALNUTRITION PRESENT":
  • Synthesize evidence supporting presence (e.g., low z-scores, wasting, inadequate intake)
  • Classify severity (mild, moderate, severe) per guidelines (e.g., ASPEN)
  • Identify etiology (e.g., illness-related, non-illness-related)
- For ICD classification "NO MALNUTRITION":
  • Synthesize evidence supporting adequate nutritional status (e.g., normal z-scores, well-nourished appearance, adequate intake)
  • Highlight stable growth and absence of malnutrition signs
  • If risk factors exist, note them without implying current deficits
- Avoid negative assertions (e.g., do not state "malnutrition is absent"; focus on evidence of adequate nutrition)
- Present findings neutrally: "The ICD classification of [label_context] is supported by..."

[END RAG REFINEMENT TASK]

ORIGINAL TEXT:
{clinical_text}

ICD CLASSIFICATION:
{label_context}

INITIAL EXTRACTION:
{stage3_json_output}

EVIDENCE BASE:
{retrieved_evidence_chunks}"""

# ============================================================================
# MALNUTRITION TEMPLATE - FIXED v1.0.4: Natural clinical conversation
# ============================================================================

MALNUTRITION_MAIN_PROMPT = """[TASK DESCRIPTION - Pediatric Malnutrition Clinical Assessment]

You are a board-certified pediatric dietitian performing a comprehensive malnutrition assessment to curate training data for a conversational AI. Use natural, expert-level clinical language.

**CRITICAL: FORWARD-THINKING + TEMPORAL CAPTURE + NEVER LEAVE EMPTY**

1. **NEVER LEAVE FIELDS EMPTY/BLANK/NULL** - All fields must have content
2. **If data NOT documented**: Use retrieved guideline evidence to write what SHOULD BE DONE (recommended guidelines, monitoring schedule, what to assess)
3. **If data IS documented**: Extract it AND enhance by comparing with retrieved guidelines, adding guideline-recommended areas not yet addressed
4. **Capture ALL vitals/measurements with DATES**
5. **Calculate explicit TRENDS** (absolute change, %, rate, velocity, percentile trajectory)
6. **Identify assessment TYPE**: Single-point vs Serial same-encounter vs Longitudinal multi-encounter
7. Use retrieved clinical guidelines for interpretation (don't restate criteria)

**TEMPORAL DATA FORMAT EXAMPLES:**

✓ GOOD: "Weight: 12.5 kg on 1/15/25 (25th %ile, z-score -0.7), 11.8 kg on 2/14/25 (10th %ile, z-score -1.3), 11.2 kg on 3/20/25 (5th %ile, z-score -1.8). Loss of 1.3 kg (10.4%) over 64 days (20g/day decline), crossing 25th→5th percentile."

✗ BAD: "Weight decreased from 12.5 kg to 11.2 kg"

**GROUND TRUTH DIAGNOSIS (YOU MUST SUPPORT THIS):**
{label_context}

This is definitive. Extract and synthesize ALL evidence supporting this diagnosis. Use retrieved clinical guidelines for interpretation.

IF "MALNUTRITION PRESENT": Synthesize anthropometric deficits WITH TRENDS, exam findings WITH SERIAL CHANGES, inadequate intake WITH DURATION, growth faltering WITH VELOCITY, labs WITH TRENDS, severity/etiology per retrieved guidelines.

IF "MALNUTRITION ABSENT": Synthesize normal anthropometrics WITH STABLE TRACKING, well-nourished appearance WITH CONSISTENCY, adequate intake WITH SUSTAINABILITY, stable growth OVER TIME, normal labs WITH STABLE TRENDS.

**ANONYMIZE:** Use "the patient", "the [age]-year-old", "the family"

**SYNTHESIS STRUCTURE:**

1. **CASE PRESENTATION**: Setting, chief concern, timeline, family perspective. **Identify assessment type** (single/serial/longitudinal) and temporal context.

2. CLINICAL SYMPTOMS - TEMPORAL (NEVER LEAVE EMPTY):
   - **If symptoms ARE documented**: Document ALL with DATES: "Vomiting 3-4 episodes daily since 1/15/25, increased to 6-8 by 2/15/25"
   - Categories: GI (vomiting, diarrhea, reflux, pain), Systemic (fever, fatigue, irritability), Feeding (poor appetite, refusal), Functional (weakness, decreased activity)
   - Describe trajectory: New onset, progressive, stable, improving, resolved
   - Quote exact descriptions with dates, relate to nutrition
   - For serial: Document changes across visits
   - **If symptoms NOT documented**: Write guideline-based recommendations using retrieved evidence: "No specific symptoms documented in the record. Per [GUIDELINE NAME from retrieved evidence], comprehensive symptom assessment should include: (1) GI symptoms: frequency/severity of vomiting, diarrhea, reflux, abdominal pain; (2) Systemic: fever patterns, fatigue, irritability; (3) Feeding: appetite changes, food refusal; (4) Functional: activity level, developmental regression. Recommend systematic symptom review at each encounter with [INTERVAL] monitoring."
   - **If symptoms ARE documented**: Also compare with retrieved guidelines and add any guideline-recommended symptom categories not yet assessed
   
3. **GROWTH & ANTHROPOMETRICS - TEMPORAL CAPTURE (NEVER LEAVE EMPTY)**:
   - **If measurements ARE documented**: Extract ALL with DATES: "Weight 12.5 kg on 1/15 (5th %ile, z-score -1.8)", calculate TRENDS (absolute change, %, rate, velocity, trajectory), describe pattern, interpret using retrieved guidelines. Then compare with retrieved evidence to identify missing parameters (e.g., "Head circumference not documented - per [GUIDELINE NAME from retrieved evidence], should be measured at each visit for children <36 months")
   - **If measurements NOT documented**: Use retrieved guideline evidence to write comprehensive recommendations: "Growth parameters not documented in this encounter. Per [GUIDELINE NAME from retrieved evidence] for [diagnosis], baseline anthropometric assessment should include: Weight-for-age with percentile/z-score, Height/length-for-age with percentile/z-score, BMI-for-age or weight-for-length with percentile/z-score, Head circumference (if <36 months), Mid-upper arm circumference (if malnutrition suspected). Recommend obtaining complete growth assessment with serial measurements at [INTERVAL] to establish growth trajectory and percentile tracking patterns."

4. **PHYSICAL EXAM - TEMPORAL (NEVER LEAVE EMPTY)**:
   - **If exam documented**: Describe findings with dates, if serial exams show progression, quote exact findings, interpret using retrieved guidelines. Compare with retrieved evidence to identify missing exam components
   - **If exam NOT documented or incomplete**: Use retrieved guideline evidence: "Physical examination not fully documented. Per [GUIDELINE NAME from retrieved evidence] for malnutrition assessment, comprehensive exam should evaluate: (1) General appearance: alertness, hydration status; (2) Muscle wasting: temporalis, deltoids, quadriceps, interosseous; (3) Subcutaneous fat loss: orbital, triceps, subscapular; (4) Edema: periorbital, pedal, sacral; (5) Skin/hair changes; (6) Growth chart plotting. For serial assessment, recommend documenting these findings at each encounter to track progression or improvement."

5. **NUTRITION & INTAKE - TEMPORAL (NEVER LEAVE EMPTY)**:
   - **If intake documented**: Extract patterns over time: "Intake declined: 80-90% in Dec, 60-70% in Jan, 40-50% by Feb". Quote with timeframes, relate to needs
   - **If intake NOT documented**: Use retrieved guideline evidence: "Nutritional intake not quantified in the record. Per [GUIDELINE NAME from retrieved evidence], comprehensive intake assessment should document: (1) Percentage of estimated needs consumed; (2) Duration of inadequate intake; (3) Types of foods accepted/refused; (4) Feeding route (oral/enteral); (5) Appetite patterns with timing. For [diagnosis], recommend 3-day food diary or 24-hour recall with serial monitoring to quantify intake and identify patterns. Target intake: [XX] kcal/kg/day, [XX] g protein/kg/day per [GUIDELINE NAME from retrieved evidence]."

6. **DIAGNOSIS & REASONING**:
   - State diagnosis consistent with ground truth
   - Synthesize evidence WITH TEMPORAL PATTERNS supporting ground truth
   - Integrate retrieved guideline criteria
   - Specify severity/etiology with guideline reference
   - Reason with incomplete data using convergent temporal evidence

7. **LABS & SCREENING - TEMPORAL (NEVER LEAVE EMPTY)**:
   - **If labs ARE documented**: Extract ALL with DATES: "Albumin: 3.8 on 1/15, 3.5 on 2/15, 3.2 on 3/15", describe TRENDS: "Albumin declining 16% over 2 months", interpret using retrieved guidelines. Compare with retrieved evidence to identify missing lab work
   - **If labs NOT documented AND diagnosis is MALNUTRITION PRESENT**: Use retrieved guideline evidence to recommend: "Laboratory investigations not documented. Per [GUIDELINE NAME from retrieved evidence] for [severity] malnutrition, comprehensive metabolic assessment should include: BASELINE PANEL: (1) Complete blood count (assess anemia, immune function); (2) Comprehensive metabolic panel (electrolytes, renal, hepatic function); (3) Albumin and prealbumin (protein status - note limitations for acute illness); (4) Vitamins/minerals: Iron studies, Vitamin D 25-OH, Zinc, B12, folate; (5) Other: Thyroid function if growth faltering, Celiac screening if indicated. SERIAL MONITORING: Recommend weekly labs for first 2 weeks (electrolytes for refeeding syndrome risk), then every 2-4 weeks until nutritional rehabilitation. Specific monitoring intervals per [GUIDELINE NAME from retrieved evidence]: [schedule]."
   - **If labs NOT documented AND diagnosis is MALNUTRITION ABSENT**: Use clinical reasoning with retrieved evidence: "No laboratory investigations documented in this encounter. Given the clinical presentation of adequate nutritional status with [supporting evidence: normal growth trajectory, adequate intake, well-nourished appearance], extensive lab work may not be clinically indicated at this time per [GUIDELINE NAME from retrieved evidence]. Routine screening labs (CBC, CMP) could be considered at next well-child visit or if clinical status changes. If growth faltering develops, would recommend malnutrition-specific lab panel as outlined in [GUIDELINE NAME from retrieved evidence]."

8. **CARE PLAN - WITH TEMPORAL MONITORING**:
   - Goals, interventions with doses
   - **Monitoring schedule**: Specific intervals (Week 1: Day 7; Week 2: assessment; Weeks 3-4: weekly; Months 2-3: bi-weekly)
   - **Labs schedule**: Baseline panel; serial monitoring frequency
   - **Follow-up timeline**: Specific dates/intervals
   - **Escalation criteria**: With timepoints (Day 7: if <50g gain→NG tube; Week 2: if no response→specialist)
   - **Expected trajectory**: Timeline for recovery
   - Justify with temporal reasoning

9. **SOCIAL CONTEXT - TEMPORAL**:
   - **If documented**: Extract temporal changes in circumstances with dates, describe intervention progression
   - **If NOT documented**: State "Social determinants and family context not documented in the record."

10. **CLINICAL INSIGHTS - TEMPORAL SYNTHESIS**:
   - Summarize with TEMPORAL INTEGRATION and retrieved guideline references
   - **Prognosis with timeline**
   - **Decision points with dates**
   - **Risk factors with timeframes**
   - **Teaching about temporal patterns**
   - **Pearls about temporal monitoring**

**CRITICAL RULES:**
- **NEVER LEAVE ANY FIELD EMPTY/BLANK/NULL**: Every field must have substantive content (except social context may say "not documented")
- **DATA NOT DOCUMENTED**: Use retrieved guideline evidence to write what SHOULD BE DONE (recommended assessments, monitoring, guidelines)
- **DATA IS DOCUMENTED**: Extract it AND enhance with retrieved guidelines (compare to guidelines, add missing recommended elements)
- ANONYMIZE: Use "the patient", "the [age]-year-old", "the family"
- CAPTURE ALL TEMPORAL DATA: Every measurement/lab/exam/intake with date
- CALCULATE EXPLICIT TRENDS: Change, %, rate, velocity, trajectory
- IDENTIFY ASSESSMENT TYPE: Single/serial/longitudinal
- QUOTE EXACTLY: All values with dates
- REASON FORWARD: Recommend what should be done with timeline
- USE RETRIEVED GUIDELINES: For thresholds, criteria, and recommendations (reference guideline names like ASPEN, WHO, CDC)
- PRESERVE UNITS
- ALIGN WITH GROUND TRUTH: Support {label_context} with ALL temporal evidence

**REMEMBER**: The goal is comprehensive, guideline-based clinical documentation. Missing data is an opportunity to demonstrate expert knowledge by recommending evidence-based assessment and monitoring per retrieved clinical guidelines.

Z-SCORE INTERPRETATION:
"[NUMBER] z [NUMBER]" = PERCENTILE first, z-score second. <50th: negative. >50th: positive. Track changes over time.

[END TASK DESCRIPTION]

CLINICAL TEXT:
{clinical_text}

**GROUND TRUTH DIAGNOSIS:**
{label_context}

{rag_outputs}

{function_outputs}

{extras_outputs}

{json_schema_instructions}"""

MALNUTRITION_MINIMAL_PROMPT = """[TASK DESCRIPTION - Pediatric Malnutrition Assessment]

Expert pediatric dietitian performing malnutrition assessment for conversational AI training. Natural, expert-level clinical language. Use retrieved clinical guidelines for interpretation.

**CRITICAL: NEVER LEAVE EMPTY + FORWARD-THINKING + TEMPORAL CAPTURE**
**NEVER LEAVE FIELDS EMPTY**: If data NOT documented → use retrieved guideline evidence to write recommendations. If data IS documented → extract AND enhance with retrieved guidelines. Capture ALL measurements with DATES. Calculate TRENDS. Identify assessment type (single/serial/longitudinal).

**GROUND TRUTH DIAGNOSIS (MUST SUPPORT):**
{label_context}

IF "MALNUTRITION PRESENT": Synthesize deficits WITH TRENDS, exam WITH SERIAL CHANGES, intake WITH DURATION, growth WITH VELOCITY, labs WITH TRENDS per retrieved guidelines.

IF "MALNUTRITION ABSENT": Synthesize normal WITH STABLE TRACKING, well-nourished WITH CONSISTENCY, adequate intake WITH SUSTAINABILITY, stable growth OVER TIME per retrieved guidelines.

**ANONYMIZE:** "the patient", "the [age]-year-old", "the family"

**SYNTHESIS STRUCTURE:**

1. **CASE**: Setting, chief concern, timeline. **Identify type** (single/serial/longitudinal), temporal context.

2. SYMPTOMS - TEMPORAL (NEVER EMPTY): **If documented**: Extract ALL with DATES, categories (GI/Systemic/Feeding/Functional), trajectory, quote. **If NOT**: Use retrieved evidence for recommendations: "Not documented. Per [GUIDELINE NAME from retrieved evidence], assess: GI, systemic, feeding, functional symptoms with [interval] monitoring." **If documented**: Also compare with retrieved guidelines for missing categories.

3. **GROWTH - TEMPORAL (NEVER EMPTY)**: **If documented**: Extract ALL with DATES, calculate TRENDS, describe pattern, interpret with retrieved guidelines, identify missing parameters. **If NOT**: Use retrieved evidence: "Not documented. Per [GUIDELINE NAME from retrieved evidence], obtain weight/height/BMI-for-age with percentiles/z-scores, serial measurements at [interval]."

4. **EXAM - TEMPORAL (NEVER EMPTY)**: **If documented**: Describe with dates, progression if serial, quote, interpret with retrieved guidelines, note missing components. **If NOT**: Use retrieved evidence: "Not documented. Per [GUIDELINE NAME from retrieved evidence], examine: appearance, muscle/fat, edema, skin/hair, growth plotting at each visit."

5. **INTAKE - TEMPORAL (NEVER EMPTY)**: **If documented**: Extract patterns with dates, timeframes. **If NOT**: Use retrieved evidence: "Not quantified. Per [GUIDELINE NAME from retrieved evidence], assess: % needs, duration, foods, route, appetite. Recommend food diary, target [XX] kcal/kg/day per [GUIDELINE NAME from retrieved evidence]."

6. **DIAGNOSIS (NEVER EMPTY)**: State consistent with ground truth. Synthesize evidence WITH TEMPORAL PATTERNS. Use retrieved guideline criteria. Specify severity/etiology. Reason with incomplete data using convergent evidence.

7. **LABS - TEMPORAL (NEVER EMPTY)**: **If documented**: Extract ALL with DATES, trends, interpret with retrieved guidelines, note missing labs. **If NOT + malnutrition**: Use retrieved evidence: "Not documented. Per [GUIDELINE NAME from retrieved evidence], obtain: CBC, CMP, albumin, vitamins. Serial monitoring: weekly x2, then q2-4wks." **If NOT + adequate**: "Not documented. Given adequate status, may not be indicated. Consider routine screening at well-child per [GUIDELINE NAME from retrieved evidence]."

8. **CARE PLAN - TEMPORAL MONITORING**: Goals, interventions. **Schedule**: Week 1 (Day 7), Week 2, Weeks 3-4, Months 2-3. **Labs schedule**: Baseline, serial frequency. **Follow-up**: Dates. **Escalation**: With timepoints. **Trajectory**: Recovery timeline.

9. **SOCIAL - TEMPORAL**: **If documented**: Extract changes with dates, intervention progression. **If NOT**: State "Social determinants and family context not documented in the record."

10. **INSIGHTS - TEMPORAL (NEVER EMPTY)**: Summarize with TEMPORAL INTEGRATION and retrieved guideline references. Prognosis with timeline. Decision points with dates. Risks with timeframes. Teaching. Pearls.

**RULES:**
- **NEVER LEAVE FIELDS EMPTY**: If not documented → use retrieved guideline evidence. If documented → extract + enhance with retrieved guidelines (except social context may say "not documented")
- ANONYMIZE
- CAPTURE ALL TEMPORAL DATA with dates
- CALCULATE TRENDS explicitly
- IDENTIFY TYPE
- QUOTE EXACTLY with dates
- REASON FORWARD with timeline
- USE RETRIEVED GUIDELINES for criteria and recommendations (ASPEN, WHO, CDC)
- ALIGN WITH GROUND TRUTH

**REMEMBER**: Missing data = opportunity to show guideline-based recommendations using retrieved clinical guidelines.

[END]

CLINICAL TEXT:
{clinical_text}

**GROUND TRUTH:**
{label_context}

{rag_outputs}

{function_outputs}

{extras_outputs}

{json_schema_instructions}"""

MALNUTRITION_RAG_REFINEMENT_PROMPT = """[RAG REFINEMENT TASK - Malnutrition Assessment]

Refining preliminary malnutrition assessment using evidence from authoritative guidelines. Clinical expert curating training data for conversational AI.

**CRITICAL: NEVER LEAVE EMPTY + ENHANCE FORWARD-THINKING + TEMPORAL CAPTURE**
**NEVER LEAVE ANY FIELD EMPTY/BLANK/NULL**: Every field in refined output must have substantive content. If initial extraction has empty/null fields, FILL THEM using retrieved guideline evidence. If initial extraction has content, ENHANCE it using retrieved guidelines. Ensure comprehensive temporal capture: ALL measurements with dates, explicit trend calculations, assessment type, temporal significance.

**GROUND TRUTH DIAGNOSIS (MUST SUPPORT):**
{label_context}

If initial extraction contradicts ground truth, CORRECT IT. Use retrieved clinical guidelines for justification.

**ANONYMIZE:** "the patient", "the [age]-year-old", "the family"

**REFINEMENT OBJECTIVES:**

1. **VALIDATE**: Confirm interpretations against retrieved guideline thresholds and criteria. Validate temporal trend calculations.

2. **CORRECT**: Fix misclassifications with citations to retrieved guidelines. Correct temporal calculations. Align with ground truth. If initial said "Absent" but ground truth is "Present", reframe with temporal decline. If initial said "Present" but ground truth is "Absent", reframe with temporal stability.

3. **ENHANCE**: Add retrieved guideline references. **Add temporal detail**: Transform vague into specific with dates. Calculate missing trends. Identify assessment type. Add temporal interpretation. **Add forward-thinking**: Labs with serial schedule, care plan with monitoring intervals, insights with timeline.

4. **FILL GAPS - PRIMARY OBJECTIVE**:
   - **EMPTY/NULL FIELDS**: If initial extraction has empty, null, blank, or "not documented" fields, THIS IS YOUR PRIMARY TASK - FILL THEM using retrieved guideline evidence
   - **Symptoms field empty?** → Write: "Symptoms not documented in record. Per [GUIDELINE NAME from retrieved evidence], comprehensive symptom assessment should include: [detailed guideline-based symptom list with monitoring intervals]"
   - **Labs field empty?** → Write: "Labs not documented. Per [GUIDELINE NAME from retrieved evidence] for [severity] malnutrition, recommend: [detailed lab panel with specific tests and serial monitoring schedule from guidelines]"
   - **Growth field empty?** → Write: "Growth parameters not documented. Per [GUIDELINE NAME from retrieved evidence], obtain: [specific anthropometric measurements with percentile/z-score interpretation and monitoring frequency]"
   - **Exam field empty?** → Write: "Physical exam not documented. Per [GUIDELINE NAME from retrieved evidence], malnutrition exam should assess: [specific exam components from guidelines with serial documentation recommendations]"
   - **Intake field empty?** → Write: "Intake not quantified. Per [GUIDELINE NAME from retrieved evidence], assess: [specific intake assessment methods and target requirements from guidelines]"
   - **Social field empty?** → State: "Social determinants and family context not documented in the record."
   - **FIELDS WITH CONTENT**: Enhance by adding guideline-recommended elements not yet addressed, specify severity using retrieved guideline criteria, calculate missing trends, add monitoring schedules from guidelines

5. **ENSURE CONSISTENCY**: Verify diagnosis matches ground truth with temporal evidence. Check temporal consistency (dates, intervals, calculations). Care plan must have actionable timeline.

6. **HANDLE MISSING**: For labs with adequate nutrition: Explain appropriateness. For labs with malnutrition: Recommend specific labs WITH schedule. For anthropometrics: Recommend obtaining. For intake: Recommend quantification. For incomplete temporal: Recommend establishing trend.

**CRITICAL PRINCIPLES:**
- **ABSOLUTE RULE: NO EMPTY FIELDS**: Every field must have substantive content in your refined output (except social context may say "not documented")
- **EMPTY FIELDS ARE YOUR PRIMARY TASK**: Use retrieved guideline evidence to fill with evidence-based recommendations
- **FILLED FIELDS ALSO NEED WORK**: Enhance with retrieved guidelines, add missing guideline-recommended elements
- Preserve fidelity (don't remove correct documented data)
- Quote retrieved guidelines explicitly (include guideline names like ASPEN, WHO, CDC)
- Flag discrepancies between initial extraction and ground truth
- EMBED FORWARD-THINKING with timelines and monitoring schedules from retrieved evidence
- ENHANCE ALL TEMPORAL DATA: dates, trends, type, significance
- GROUND TRUTH IS ABSOLUTE: Correct to align using temporal evidence
- **MINDSET**: You are demonstrating expert clinical knowledge by providing guideline-based recommendations using retrieved clinical guidelines

**Z-SCORE AND PERCENTILE VALIDATION:**
- Validate z-score sign with percentile:
  • Percentiles <50th: Negative z-scores (below average, e.g., "1 z 2.36" is -2.36)
  • Percentiles >50th: Positive z-scores (above average, e.g., "85 ile z 1.04" is +1.04)
- Confirm alignment with clinical descriptions (e.g., "short stature" indicates below average; "well-nourished" indicates normal/above average)
- Flag discrepancies between z-scores, percentiles, and clinical narrative for review

**SYNTHESIS GUIDELINES FOR MALNUTRITION ASSESSMENT:**
- For ICD classification "MALNUTRITION PRESENT":
  • Synthesize evidence supporting presence (e.g., low z-scores, wasting, inadequate intake)
  • Classify severity (mild, moderate, severe) per retrieved guidelines (e.g., ASPEN)
  • Identify etiology (e.g., illness-related, non-illness-related)
- For ICD classification "NO MALNUTRITION":
  • Synthesize evidence supporting adequate nutritional status (e.g., normal z-scores, well-nourished appearance, adequate intake)
  • Highlight stable growth and absence of malnutrition signs
  • If risk factors exist, note them without implying current deficits
- Avoid negative assertions (e.g., do not state "malnutrition is absent"; focus on evidence of adequate nutrition)
- Present findings neutrally: "The ICD classification of [label_context] is supported by..."

**CRITICAL REMINDER ON ANONYMIZATION:**
- NEVER use patient names or family/caregiver names in the output
- ALWAYS use: "the patient", "the [age]-year-old", "the family", "the caregiver"
- Replace any names from initial extraction: "John" → "the patient", "Mary (mother)" → "the family"

**REMINDER ON TEMPORAL DATA:**
- Capture EVERY anthropometric measurement with its DATE: "Weight 12.5 kg on 1/15/25"
- Document ALL measurements over time, not just summary
- Calculate explicit trends between all time points

[END]

ORIGINAL TEXT:
{clinical_text}

**GROUND TRUTH:**
{label_context}

INITIAL EXTRACTION:
{stage3_json_output}

EVIDENCE BASE:
{retrieved_evidence_chunks}

{json_schema_instructions}"""

# ============================================================================
# DIABETES TEMPLATE (unchanged)
# ============================================================================

DIABETES_MAIN_PROMPT = """[TASK DESCRIPTION - Diabetes Assessment]

You are a clinical expert analyzing diabetes-related medical information.

YOUR TASK:
Extract comprehensive diabetes assessment data:

1. DIAGNOSIS:
   - Diabetes type (Type 1, Type 2, Gestational, Other)
   - Diagnosis date and method
   - Diagnostic criteria used

2. LABORATORY VALUES:
   - HbA1c levels and dates
   - Fasting blood glucose
   - Random/postprandial glucose
   - Other relevant labs

3. MEDICATIONS:
   - Current diabetes medications
   - Dosages and frequencies
   - Insulin regimens if applicable

4. COMPLICATIONS:
   - Documented complications
   - Risk factors
   - Screening results

[END TASK DESCRIPTION]

CLINICAL TEXT:
{clinical_text}

ICD CLASSIFICATION:
{label_context}

{rag_outputs}

{function_outputs}

{extras_outputs}

{json_schema_instructions}"""

DIABETES_MINIMAL_PROMPT = """[TASK DESCRIPTION - Diabetes Data]

Extract diabetes information: diagnosis, type, HbA1c, glucose values, medications, complications.

[END TASK DESCRIPTION]

CLINICAL TEXT:
{clinical_text}

ICD CLASSIFICATION:
{label_context}

{rag_outputs}

{function_outputs}

{extras_outputs}

{json_schema_instructions}"""

# ============================================================================
# STAGE 1 ANALYSIS PROMPT TEMPLATE
# ============================================================================

STAGE1_ANALYSIS_PROMPT = """[SYSTEM INSTRUCTION]
You are an agentic clinical analyst planning information extraction using available tools. Your job is to understand the extraction task, analyze the clinical text, and intelligently select tools that will help complete the extraction.

[EXTRACTION TASK]

The extraction task requires the following information to be extracted:

{task_description}

EXPECTED OUTPUT SCHEMA:
{json_schema}

[AVAILABLE TOOLS]

You have access to these tools:

{available_tools_description}

[YOUR AGENTIC TASK]

Analyze the clinical text and determine which tools are needed to extract the required information:

1. **Understand the Task**: Review the extraction task description and output schema to understand what information needs to be extracted.

2. **Analyze the Clinical Text**: Identify key information present in the clinical text that relates to the extraction task.

3. **Identify Functions**: Determine which functions need to be called and extract their parameters from the clinical text.
   - Extract numeric values (weight, height, age, etc.) from the clinical text
   - If a parameter cannot be extracted, leave it out (the system will attempt extraction)
   - For age calculations, extract date of birth and current date
   - For growth calculations, extract age, sex, and relevant measurements

4. **Build RAG Queries**: Create intelligent queries to retrieve relevant guidelines and evidence.
   - Build queries from the task description, clinical context, and label classification
   - Target specific guidelines, criteria, and standards relevant to the task
   - Use 3-7 meaningful keywords that capture the clinical context

5. **Build Extras Keywords**: Create keywords to match relevant hints and contextual information.
   - Extract key medical terms and concepts from the clinical text
   - Include terms from the label classification and task description
   - Use 3-5 specific, relevant keywords

CRITICAL PARAMETER EXTRACTION RULES:
- Extract ALL numeric parameters from the clinical text (weight, height, age, measurements, lab values)
- Look for patterns like "weight: 45.5 kg", "height 150cm", "5 year old", "DOB: 01/15/2020"
- Convert units appropriately (e.g., pounds to kg: multiply by 0.453592; feet to cm: multiply by 30.48)
- For dates, extract in format MM/DD/YYYY or similar
- For sex, identify from context (male/female indicators)

INTELLIGENT QUERY BUILDING:
- RAG queries should target guidelines, criteria, and standards relevant to the extraction task
- Extract key medical concepts from clinical text: diagnoses, conditions, assessments
- Include classification/label terms in queries
- Include schema-related terms (e.g., if schema has "malnutrition_status", include "malnutrition" in queries)

TOOL SELECTION STRATEGY:
- Call functions when: calculations are needed, specific values must be computed
- Use RAG when: guidelines, criteria, or evidence-based standards are needed for the task
- Use extras when: contextual information, coding, or reference data would help

[CLINICAL TEXT TO ANALYZE]
{clinical_text}

[LABEL CLASSIFICATION]
{label_context}

[REQUIRED OUTPUT FORMAT]

Return your response in this EXACT JSON format:
{{
  "analysis": "Brief analysis of what information needs to be extracted and which tools will help",
  "tool_requests": [
    {{
      "tool": "function",
      "name": "extract_age_from_dates",
      "arguments": {{"date_of_birth": "01/15/2020", "current_date": "10/30/2025"}},
      "reasoning": "Need to calculate age in months for growth assessment"
    }},
    {{
      "tool": "function",
      "name": "calculate_bmi",
      "arguments": {{"weight_kg": 45.5, "height_cm": 150}},
      "reasoning": "Need BMI for nutritional status assessment"
    }},
    {{
      "tool": "rag",
      "keywords": ["malnutrition criteria", "ASPEN guidelines", "pediatric nutrition"],
      "reasoning": "Need evidence-based criteria for nutritional assessment and diagnosis"
    }},
    {{
      "tool": "extras",
      "keywords": ["ICD-10", "malnutrition", "coding"],
      "reasoning": "Need contextual coding information for documentation"
    }}
  ]
}}

REMEMBER:
- Extract ALL relevant parameters from the clinical text
- Build intelligent queries that target guidelines and standards for the specific task
- Only request tools that will genuinely help with the extraction task
- The goal is to gather all information needed to complete the extraction schema"""

# ============================================================================
# TEMPLATE REGISTRY
# ============================================================================
PROMPT_TEMPLATE_REGISTRY_V1 = {
    "blank": {
        "main": DEFAULT_MAIN_PROMPT,
        "minimal": DEFAULT_MINIMAL_PROMPT,
        "rag_prompt": DEFAULT_RAG_REFINEMENT_PROMPT,
        "description": "Start from scratch - Generic extraction",
        "version": "2.0.0",
        "schema": {
            "extracted_data": {
                "type": "object",
                "description": "Container for extracted information",
                "required": True
            },
            "confidence": {
                "type": "string",
                "description": "Extraction confidence: high, medium, low",
                "required": False
            }
        }
    },
    "malnutrition": {
        "main": MALNUTRITION_MAIN_PROMPT,
        "minimal": MALNUTRITION_MINIMAL_PROMPT,
        "rag_prompt": MALNUTRITION_RAG_REFINEMENT_PROMPT,
        "description": "Pediatric Malnutrition Assessment with Forward-Thinking Clinical Reasoning and Temporal Data Capture",
        "version": "1.0.5",
        "schema": {
            "malnutrition_status": {
                "type": "string",
                "description": "Malnutrition classification: 'Malnutrition Present' or 'Malnutrition Absent'",
                "required": True
            },
            "case_presentation": {
                "type": "string",
                "description": "Natural case presentation with setting, chief concern, problem evolution, family perspective. Identify assessment type (single-point/serial/longitudinal) with temporal context.",
                "required": True
            },
            "clinical_symptoms_and_signs": {
                        "type": "string",
                        "description": "TEMPORAL SYMPTOM CAPTURE (NEVER LEAVE EMPTY): If documented: Extract ALL with ONSET DATES and PROGRESSION: (1) GI: vomiting, diarrhea, reflux, pain with frequency/severity/timing; (2) Systemic: fever, fatigue, irritability; (3) Feeding: appetite, refusal, satiety; (4) Functional: weakness, activity, development; (5) Trajectory: new/progressive/stable/improving/resolved. Quote exact descriptions. If NOT documented: Use retrieved guideline evidence to write symptom assessment recommendations with monitoring intervals. NEVER leave this field empty/null.",
                        "required": False
            },
            "growth_and_anthropometrics": {
                "type": "string",
                "description": "TEMPORAL CAPTURE (NEVER LEAVE EMPTY): If documented: Extract ALL with DATES, calculate EXPLICIT TRENDS (absolute, %, rate, velocity, trajectory), describe pattern, interpret using retrieved guidelines, note missing parameters. If NOT documented: Use retrieved guideline evidence to write comprehensive anthropometric assessment recommendations with specific measurements, percentile/z-score interpretation, and serial monitoring frequency. NEVER leave empty.",
                "required": True
            },
            "physical_exam": {
                "type": "string",
                "description": "TEMPORAL (NEVER LEAVE EMPTY): If documented: Extract findings with dates, progression if serial, quote exactly, interpret using retrieved guidelines, note missing exam components. If NOT documented: Use retrieved guideline evidence to write comprehensive exam recommendations covering appearance, muscle/fat assessment, edema, skin/hair, with serial documentation guidance. NEVER leave empty.",
                "required": True
            },
            "nutrition_and_intake": {
                "type": "string",
                "description": "TEMPORAL (NEVER LEAVE EMPTY): If documented: Extract patterns with dates and timeframes, relate to requirements. If NOT documented: Use retrieved guideline evidence to write intake assessment recommendations with methods (food diary, recall), target requirements (kcal/kg, protein/kg), and quantification strategy. NEVER leave empty.",
                "required": True
            },
            "diagnosis_and_reasoning": {
                "type": "string",
                "description": "TEMPORAL INTEGRATION: Diagnosis (aligned with ground truth), severity, etiology. Synthesize evidence WITH TEMPORAL PATTERNS. Use retrieved guideline criteria. Reason with incomplete data using convergent temporal evidence.",
                "required": True
            },
            "labs_and_screening": {
                "type": "string",
                "description": "TEMPORAL (NEVER LEAVE EMPTY): If documented: Extract ALL with DATES, describe TRENDS, interpret using retrieved guidelines, note missing labs. If NOT documented + malnutrition: Use retrieved guideline evidence to write comprehensive lab recommendations (CBC, CMP, albumin, vitamins, etc.) WITH serial monitoring schedule (weekly x2, then q2-4wks). If NOT documented + adequate nutrition: Use retrieved evidence to explain appropriateness and recommend routine screening schedule. NEVER leave empty - always provide guideline-based content.",
                "required": False
            },
            "care_plan": {
                "type": "string",
                "description": "TEMPORAL MONITORING: Goals, interventions with doses. MONITORING SCHEDULE: Specific intervals (Week 1: Day 7; Week 2; Weeks 3-4: weekly; Months 2-3: bi-weekly). LABS SCHEDULE: Baseline panel, serial frequency. FOLLOW-UP: Dates/intervals. ESCALATION: With timepoints (Day 7: if <50g→NG; Week 2: if no response→specialist). TRAJECTORY: Recovery timeline. Justify with temporal reasoning.",
                "required": True
            },
            "social_context": {
                "type": "string",
                "description": "TEMPORAL: If documented: Extract situation, barriers, changes with dates, intervention progression. If NOT documented: State 'Social determinants and family context not documented in the record.'",
                "required": False
            },
            "clinical_insights": {
                "type": "string",
                "description": "TEMPORAL SYNTHESIS: Summary with TEMPORAL INTEGRATION and retrieved guideline references. Prognosis with timeline. Decision points with dates. Risk factors with timeframes. Teaching about temporal patterns. Pearls about temporal monitoring.",
                "required": True
            }
        }
    },
    "diabetes": {
        "main": DIABETES_MAIN_PROMPT,
        "minimal": DIABETES_MINIMAL_PROMPT,
        "rag_prompt": DEFAULT_RAG_REFINEMENT_PROMPT,
        "description": "Diabetes Assessment and Management",
        "version": "2.0.0",
        "schema": {
            "diabetes_diagnosis": {
                "type": "string",
                "description": "Primary diabetes diagnosis",
                "required": True
            },
            "diabetes_type": {
                "type": "string",
                "description": "Type 1, Type 2, Gestational, or Other",
                "required": False
            },
            "hba1c_value": {
                "type": "number",
                "description": "Most recent HbA1c percentage",
                "required": False
            },
            "hba1c_date": {
                "type": "string",
                "description": "Date of HbA1c measurement",
                "required": False
            },
            "glucose_values": {
                "type": "array",
                "description": "Blood glucose readings with dates",
                "required": False
            },
            "current_medications": {
                "type": "array",
                "description": "List of diabetes medications",
                "required": False
            },
            "complications": {
                "type": "array",
                "description": "Documented diabetes complications",
                "required": False
            }
        }
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_template(template_name: str = "blank") -> Dict[str, Any]:
    """
    Retrieve a prompt template by name.
    
    Args:
        template_name: Name of template to retrieve
        
    Returns:
        Template dictionary with main, minimal, rag_prompt, schema, etc.
    """
    return PROMPT_TEMPLATE_REGISTRY_V1.get(
        template_name,
        PROMPT_TEMPLATE_REGISTRY_V1["blank"]
    )


def get_prompt_template(template_name: str = "blank") -> Dict[str, Any]:
    """
    Retrieve a prompt template by name (alias for get_template).
    
    Args:
        template_name: Name of template to retrieve
        
    Returns:
        Template dictionary
    """
    return get_template(template_name)


def list_templates() -> Dict[str, str]:
    """
    List all available templates with descriptions.
    
    Returns:
        Dictionary mapping template names to descriptions
    """
    return {
        name: template_data["description"]
        for name, template_data in PROMPT_TEMPLATE_REGISTRY_V1.items()
    }


def list_available_templates() -> Dict[str, str]:
    """
    List all available templates with descriptions (alias for list_templates).
    
    Returns:
        Dictionary mapping template names to descriptions
    """
    return list_templates()


def get_default_rag_refinement_prompt() -> str:
    """
    Get default RAG refinement prompt template.
    
    Returns:
        Default RAG refinement prompt template string
    """
    return DEFAULT_RAG_REFINEMENT_PROMPT


def get_stage1_analysis_prompt_template() -> str:
    """
    Get Stage 1 analysis prompt template.
    
    Returns:
        Stage 1 analysis prompt template string
    """
    return STAGE1_ANALYSIS_PROMPT


def get_json_enforcement_instructions() -> str:
    """
    Get JSON enforcement instructions.
    REMOVED: Returns empty string - no JSON enforcement.
    
    Returns:
        Empty string (JSON enforcement removed per user request)
    """
    return ""


def get_rag_json_enforcement_instructions() -> str:
    """
    Get RAG-specific JSON enforcement instructions.
    REMOVED: Returns empty string - no JSON enforcement.
    
    Returns:
        Empty string (JSON enforcement removed per user request)
    """
    return ""


def format_schema_as_instructions(schema: Dict[str, Any]) -> str:
    """
    Format JSON schema as human-readable instructions for the LLM.
    Supports both flat and nested schemas with proper formatting.
    
    Args:
        schema: JSON schema dictionary with field definitions
        
    Returns:
        Formatted schema instruction string
    """
    if not schema:
        return ""
    
    def format_field(name: str, spec: Dict[str, Any], indent: int = 0) -> List[str]:
        """
        Recursively format a field with proper indentation.
        
        Args:
            name: Field name
            spec: Field specification dictionary
            indent: Current indentation level
            
        Returns:
            List of formatted lines
        """
        lines = []
        prefix = "  " * indent
        field_type = spec.get('type', 'string')
        field_desc = spec.get('description', '')
        is_required = spec.get('required', False)
        
        req_marker = " [REQUIRED]" if is_required else " [OPTIONAL]"
        
        if field_type == 'object':
            lines.append(f'{prefix}"{name}": {{{req_marker}')
            if field_desc:
                lines.append(f'{prefix}  // {field_desc}')
            if 'properties' in spec:
                for prop_name, prop_spec in spec['properties'].items():
                    lines.extend(format_field(prop_name, prop_spec, indent + 1))
            lines.append(f'{prefix}}}')
        elif field_type == 'array':
            items_type = spec.get('items', {}).get('type', 'string')
            lines.append(f'{prefix}"{name}": [{items_type}]{req_marker}')
            if field_desc:
                lines.append(f'{prefix}  // {field_desc}')
        else:
            type_hint = {
                'string': '"text"',
                'number': '0',
                'boolean': 'true/false',
                'integer': '0'
            }.get(field_type, 'null')
            lines.append(f'{prefix}"{name}": {type_hint}{req_marker}')
            if field_desc:
                lines.append(f'{prefix}  // {field_desc}')
        
        return lines
    
    instructions = ["\n" + "=" * 80]
    instructions.append("EXPECTED JSON OUTPUT STRUCTURE")
    instructions.append("=" * 80)
    instructions.append("{")
    
    for i, (field_name, field_spec) in enumerate(schema.items()):
        field_lines = format_field(field_name, field_spec, indent=1)
        
        for j, line in enumerate(field_lines):
            if j == len(field_lines) - 1 and i == len(schema) - 1:
                instructions.append(line)
            elif j == 0 and not line.strip().startswith('//'):
                if '":' in line and i < len(schema) - 1:
                    instructions.append(line + ',')
                else:
                    instructions.append(line)
            else:
                instructions.append(line)
    
    instructions.append("}")
    instructions.append("=" * 80)
    instructions.append("")
    
    return "\n".join(instructions)


def format_tool_outputs_for_prompt(
    tool_results: List[Dict[str, Any]],
    include_rag: bool = True,
    include_functions: bool = True,
    include_extras: bool = True
) -> Dict[str, str]:
    """
    ENHANCED: Format tool execution results for inclusion in extraction prompts
    
    Args:
        tool_results: List of tool execution results from Stage 2
        include_rag: Whether to include RAG results
        include_functions: Whether to include function results
        include_extras: Whether to include extras results
        
    Returns:
        Dictionary with formatted strings for rag_outputs, function_outputs, extras_outputs
    """
    rag_output = ""
    function_output = ""
    extras_output = ""
    
    for result in tool_results:
        tool_type = result.get('type', '').lower()
        success = result.get('success', False)
        
        if not success:
            continue
        
        if tool_type == 'rag' and include_rag:
            # Format RAG chunks with clear source attribution
            results_list = result.get('results', [])
            if results_list:
                if not rag_output:
                    rag_output = "\n[RETRIEVED EVIDENCE FROM AUTHORITATIVE SOURCES]\n"
                    rag_output += "Use the following evidence to support your clinical interpretation.\n"
                    rag_output += "IMPORTANT: When referencing this evidence, cite the specific source document.\n\n"

                for i, chunk in enumerate(results_list, 1):
                    # Handle both dict and object formats
                    if isinstance(chunk, dict):
                        text = chunk.get('text', '') or chunk.get('content', '')
                        source = chunk.get('source', 'Unknown Source')
                        score = chunk.get('score', 0)
                        metadata = chunk.get('metadata', {})
                    else:
                        text = getattr(chunk, 'text', '') or getattr(chunk, 'content', '')
                        source = getattr(chunk, 'source', 'Unknown Source')
                        score = getattr(chunk, 'score', 0)
                        metadata = getattr(chunk, 'metadata', {})

                    if text:
                        # Extract more descriptive source name if available
                        source_filename = metadata.get('source_filename', '') if isinstance(metadata, dict) else ''
                        source_type = metadata.get('type', '') if isinstance(metadata, dict) else ''

                        rag_output += f"\n{'='*60}\n"
                        rag_output += f"EVIDENCE #{i} - Relevance Score: {score:.2f}\n"
                        rag_output += f"SOURCE: {source}\n"
                        if source_filename and source_filename != source:
                            rag_output += f"FILE: {source_filename}\n"
                        rag_output += f"{'='*60}\n\n"
                        rag_output += f"{text[:1500]}\n\n"  # Increased limit for more context
        
        elif tool_type == 'function' and include_functions:
            func_name = result.get('name', 'unknown')
            func_result = result.get('result', {})
            date_context = result.get('date_context', '')

            if not function_output:
                function_output = "\n[CALCULATED VALUES FROM FUNCTIONS]\n"

            # Include date context for serial measurements
            if date_context:
                function_output += f"\n{func_name} [{date_context}]:\n"
            else:
                function_output += f"\n{func_name}:\n"

            if isinstance(func_result, dict):
                for key, value in func_result.items():
                    function_output += f"  - {key}: {value}\n"
            else:
                function_output += f"  Result: {func_result}\n"
        
        elif tool_type == 'extras' and include_extras:
            # ENHANCED: Format extras as supplementary hints/tips
            items = result.get('results', [])
            keywords = result.get('keywords', [])
            
            if items:
                if not extras_output:
                    extras_output = "\n[SUPPLEMENTARY HINTS & TIPS]\n"
                    extras_output += f"(Retrieved based on keywords: {', '.join(keywords)})\n\n"
                
                for i, item in enumerate(items, 1):
                    content = item.get('content', '')
                    item_type = item.get('type', 'hint')
                    relevance = item.get('relevance_score', 0)
                    matched_kw = item.get('matched_keywords', [])
                    
                    if content:
                        extras_output += f"--- Hint #{i} ({item_type}) ---\n"
                        if matched_kw:
                            extras_output += f"Matched keywords: {', '.join(matched_kw)}\n"
                        extras_output += f"{content}\n\n"
    
    return {
        'rag_outputs': rag_output,
        'function_outputs': function_output,
        'extras_outputs': extras_output
    }