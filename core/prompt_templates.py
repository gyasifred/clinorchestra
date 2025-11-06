#!/usr/bin/env python3
"""
Prompt Templates Module - Central repository for all prompt templates
Professional Quality v1.0.0 - Natural clinical language with guideline-based interpretation

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Lab: ClinicalNLP Lab, Biomedical Informatics Center
Version: 1.0.4 - Natural conversation with guideline-based evidence synthesis

═══════════════════════════════════════════════════════════════════════════
🎯 TRULY UNIVERSAL SYSTEM - Works for ANY Clinical Task
═══════════════════════════════════════════════════════════════════════════

This module contains:
1. DEFAULT templates (generic, task-agnostic)
2. EXAMPLE templates (malnutrition, diabetes - illustrative only!)

The example templates (MALNUTRITION_*, DIABETES_*) are pre-configured for those
specific tasks but are NOT hardcoded requirements. They serve as:
- Reference implementations for complex tasks
- Templates you can adapt for your own tasks
- Examples of how to structure prompts

YOUR TASK can be completely different (sepsis, AKI, oncology, cardiac, etc.).
The system adapts to YOUR schema and prompts - it's not limited to the examples!
═══════════════════════════════════════════════════════════════════════════
"""

from typing import Dict, Any, List

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
# EXAMPLE TEMPLATE 1: MALNUTRITION ASSESSMENT
# ============================================================================
# 🎯 NOTE: This is an EXAMPLE template for malnutrition tasks.
#    It demonstrates complex temporal reasoning and growth assessment.
#    Your task can be completely different - this is NOT a system requirement!
# ============================================================================

MALNUTRITION_MAIN_PROMPT = """[TASK DESCRIPTION - Pediatric Malnutrition Clinical Assessment]

You are a board-certified pediatric dietitian performing a comprehensive malnutrition assessment to curate training data for a conversational AI. Use natural, expert-level clinical language.

**CRITICAL: FORWARD-THINKING + TEMPORAL CAPTURE**

1. Extract documented data AND recommend what should be done when missing
2. **Capture ALL vitals/measurements with DATES**
3. **Calculate explicit TRENDS** (absolute change, %, rate, velocity, percentile trajectory)
4. **Identify assessment TYPE**: Single-point vs Serial same-encounter vs Longitudinal multi-encounter
5. Use retrieved evidence from authoritative sources (ASPEN, WHO, CDC) for clinical interpretation (don't restate criteria)

**TEMPORAL DATA FORMAT EXAMPLES:**

✓ GOOD: "Weight: 12.5 kg on 1/15/25 (25th %ile, z-score -0.7), 11.8 kg on 2/14/25 (10th %ile, z-score -1.3), 11.2 kg on 3/20/25 (5th %ile, z-score -1.8). Loss of 1.3 kg (10.4%) over 64 days (20g/day decline), crossing 25th→5th percentile."

✗ BAD: "Weight decreased from 12.5 kg to 11.2 kg"

**CRITICAL Z-SCORE AND PERCENTILE CONVENTION - MUST FOLLOW EXACTLY:**

**Z-SCORE SIGN CONVENTION (NON-NEGOTIABLE):**
- **Percentile BELOW 50th** = NEGATIVE z-score (child is below average)
  * 3rd percentile = z-score **-1.88** (NOT +1.88)
  * 5th percentile = z-score **-1.64** (NOT +1.64)
  * 10th percentile = z-score **-1.28** (NOT +1.28)
  * 25th percentile = z-score **-0.67** (NOT +0.67)
- **50th percentile** = z-score 0 (exactly average)
- **Percentile ABOVE 50th** = POSITIVE z-score (child is above average)
  * 75th percentile = z-score +0.67
  * 90th percentile = z-score +1.28
  * 95th percentile = z-score +1.64

**WHO MALNUTRITION CLASSIFICATION BY Z-SCORE:**
Weight-for-Height or BMI-for-Age:
- **z < -3**: SEVERE ACUTE MALNUTRITION (SAM) - <1st percentile, immediate intervention needed
- **-3 ≤ z < -2**: MODERATE ACUTE MALNUTRITION (MAM) - 2nd-3rd percentile, nutritional rehabilitation
- **-2 ≤ z < -1**: MILD MALNUTRITION RISK - 3rd-15th percentile, close monitoring
- **-1 ≤ z ≤ +1**: NORMAL RANGE - 15th-85th percentile
- **z > +2**: OVERWEIGHT/OBESITY - >97th percentile

Height-for-Age (Stunting):
- **z < -3**: SEVERELY STUNTED (chronic malnutrition)
- **z < -2**: STUNTED (chronic undernutrition)

**ASPEN PEDIATRIC MALNUTRITION SEVERITY (Requires 2+ indicators):**
BMI-for-Age or Weight-for-Height:
- **Mild**: z-score -1 to -1.9 (3rd-15th percentile)
- **Moderate**: z-score -2 to -2.9 (0.5th-3rd percentile)
- **Severe**: z-score ≤ -3 (<0.5th percentile)

Deceleration in Weight Gain Velocity:
- **Mild**: Decline of 1 z-score
- **Moderate**: Decline of 2 z-scores
- **Severe**: Decline of 3 z-scores

**FUNCTIONS TO USE:**
- When you have z-score values, ALWAYS call interpret_zscore_malnutrition(zscore, measurement_type) to get proper WHO/ASPEN interpretation
- Use percentile_to_zscore() to convert percentiles to z-scores if only percentiles are given
- Use calculate_growth_percentile() to calculate various z-scores from anthropometric measurements

**GROUND TRUTH DIAGNOSIS (YOU MUST SUPPORT THIS):**
{label_context}

This is definitive. Extract and synthesize ALL evidence supporting this diagnosis. Use retrieved evidence from authoritative sources (ASPEN, WHO, CDC) for interpretation.

IF "MALNUTRITION PRESENT": Synthesize anthropometric deficits WITH TRENDS, exam findings WITH SERIAL CHANGES, inadequate intake WITH DURATION, growth faltering WITH VELOCITY, labs WITH TRENDS, severity/etiology per retrieved guidelines (cite specific sources: ASPEN, WHO, CDC).

IF "MALNUTRITION ABSENT": Synthesize normal anthropometrics WITH STABLE TRACKING, well-nourished appearance WITH CONSISTENCY, adequate intake WITH SUSTAINABILITY, stable growth OVER TIME, normal labs WITH STABLE TRENDS.

**ANONYMIZE:** Use "the patient", "the [age]-year-old", "the family"

**SYNTHESIS STRUCTURE:**

1. **CASE PRESENTATION**: Setting, chief concern, timeline, family perspective. **Identify assessment type** (single/serial/longitudinal) and temporal context.

2. CLINICAL SYMPTOMS - TEMPORAL:
   - Document ALL symptoms with DATES: "Vomiting 3-4 episodes daily since 1/15/25, increased to 6-8 by 2/15/25"
   - Categories: GI (vomiting, diarrhea, reflux, pain), Systemic (fever, fatigue, irritability), Feeding (poor appetite, refusal), Functional (weakness, decreased activity)
   - Describe trajectory: New onset, progressive, stable, improving, resolved
   - Quote exact descriptions with dates
   - Relate to nutrition: "Vomiting limits intake to <50%"
   - For serial: Document changes across visits
   - IF NOT DOCUMENTED: "No symptoms documented. Recommend systematic review of symptoms per retrieved evidence."

3. **GROWTH & ANTHROPOMETRICS - TEMPORAL CAPTURE**:
   - ALL measurements with DATES: "Weight 12.5 kg on 1/15 (5th %ile, z-score -1.8)"
   - Calculate TRENDS: Absolute change, %, rate, velocity, percentile trajectory
   - Describe trajectory: "Progressive decline" or "Stable tracking 25th %ile across 11/1, 12/15, 2/1"
   - Interpret using retrieved guidelines from authoritative sources (cite ASPEN, WHO, CDC as appropriate; don't restate thresholds)
   - IF MISSING: State what's missing, recommend obtaining with rationale

4. **PHYSICAL EXAM - TEMPORAL**:
   - If serial exams: Describe progression with dates
   - Quote exact findings
   - Interpret using retrieved guidelines (cite specific sources: ASPEN, WHO)
   - IF INCOMPLETE: Recommend exam with rationale

5. **NUTRITION & INTAKE - TEMPORAL**:
   - Document patterns over time: "Intake declined: 80-90% in Dec, 60-70% in Jan, 40-50% by Feb"
   - Quote with timeframes
   - IF MISSING: Use clinical reasoning about trajectory, recommend quantification

6. **DIAGNOSIS & REASONING**:
   - State diagnosis consistent with ground truth
   - Synthesize evidence WITH TEMPORAL PATTERNS supporting ground truth
   - Integrate retrieved guideline criteria (cite specific sources: ASPEN, WHO, CDC)
   - Specify severity/etiology with guideline reference from retrieved evidence
   - Reason with incomplete data using convergent temporal evidence

7. **LABS & SCREENING - TEMPORAL**:
   - ALL labs with DATES: "Albumin: 3.8 on 1/15, 3.5 on 2/15, 3.2 on 3/15"
   - Describe TRENDS: "Albumin declining 16% over 2 months"
   - Interpret using retrieved guidelines from authoritative sources
   - **IF MISSING**: For malnutrition, recommend specific labs WITH serial monitoring schedule per retrieved guidelines. For adequate nutrition, explain appropriateness.

8. **CARE PLAN - WITH TEMPORAL MONITORING**:
   - Goals, interventions with doses
   - **Monitoring schedule**: Specific intervals (Week 1: Day 7; Week 2: assessment; Weeks 3-4: weekly; Months 2-3: bi-weekly)
   - **Labs schedule**: Baseline panel; serial monitoring frequency
   - **Follow-up timeline**: Specific dates/intervals
   - **Escalation criteria**: With timepoints (Day 7: if <50g gain→NG tube; Week 2: if no response→specialist)
   - **Expected trajectory**: Timeline for recovery
   - Justify with temporal reasoning

9. **SOCIAL CONTEXT - TEMPORAL**:
   - Note temporal changes in circumstances with dates
   - Describe intervention progression
   - IF MISSING: Recommend assessment with rationale

10. **CLINICAL INSIGHTS - TEMPORAL SYNTHESIS**:
   - Summarize with TEMPORAL INTEGRATION and references to retrieved guidelines (cite specific sources: ASPEN, WHO, CDC)
   - **Prognosis with timeline**
   - **Decision points with dates**
   - **Risk factors with timeframes**
   - **Teaching about temporal patterns**
   - **Pearls about temporal monitoring**

**CRITICAL RULES:**
- ANONYMIZE: Use "the patient", "the [age]-year-old", "the family"
- CAPTURE ALL TEMPORAL DATA: Every measurement/lab/exam/intake with date
- CALCULATE EXPLICIT TRENDS: Change, %, rate, velocity, trajectory
- IDENTIFY ASSESSMENT TYPE: Single/serial/longitudinal
- QUOTE EXACTLY: All values with dates
- REASON FORWARD: Recommend what should be done with timeline
- USE RETRIEVED GUIDELINES: Cite specific authoritative sources (ASPEN, WHO, CDC) for thresholds and criteria (don't restate in narrative)
- PRESERVE UNITS
- ALIGN WITH GROUND TRUTH: Support {label_context} with ALL temporal evidence

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

Expert pediatric dietitian performing malnutrition assessment for conversational AI training. Natural, expert-level clinical language. Use retrieved evidence from authoritative sources (ASPEN, WHO, CDC) for interpretation.

**CRITICAL: FORWARD-THINKING + TEMPORAL CAPTURE**
Extract documented data AND recommend missing. Capture ALL measurements with DATES. Calculate TRENDS. Identify assessment type (single/serial/longitudinal).

**CRITICAL Z-SCORE CONVENTION:**
- Percentile <50th = NEGATIVE z-score: 3rd %ile = -1.88, 5th %ile = -1.64, 10th %ile = -1.28, 25th %ile = -0.67
- Percentile >50th = POSITIVE z-score: 75th %ile = +0.67, 90th %ile = +1.28, 95th %ile = +1.64
- WHO: z < -3 = Severe, -3 to -2 = Moderate, -2 to -1 = Mild risk, -1 to +1 = Normal, z > +2 = Overweight
- ASPEN: z ≤ -3 = Severe, -2 to -2.9 = Moderate, -1 to -1.9 = Mild (requires 2+ indicators)
- Velocity decline: 1 z-score = Mild, 2 z-scores = Moderate, 3 z-scores = Severe
- **USE FUNCTIONS**: interpret_zscore_malnutrition(zscore, measurement_type), percentile_to_zscore(), calculate_growth_percentile()

**GROUND TRUTH DIAGNOSIS (MUST SUPPORT):**
{label_context}

IF "MALNUTRITION PRESENT": Synthesize deficits WITH TRENDS, exam WITH SERIAL CHANGES, intake WITH DURATION, growth WITH VELOCITY, labs WITH TRENDS per retrieved guidelines (cite ASPEN, WHO, CDC).

IF "MALNUTRITION ABSENT": Synthesize normal WITH STABLE TRACKING, well-nourished WITH CONSISTENCY, adequate intake WITH SUSTAINABILITY, stable growth OVER TIME per retrieved guidelines.

**ANONYMIZE:** "the patient", "the [age]-year-old", "the family"

**SYNTHESIS STRUCTURE:**

1. **CASE**: Setting, chief concern, timeline. **Identify type** (single/serial/longitudinal), temporal context.

2. SYMPTOMS - TEMPORAL: Document ALL with DATES. Categories: GI, Systemic, Feeding, Functional. Trajectory: new/progressive/stable/improving/resolved. Quote with dates. For serial: changes across visits. Relate to nutrition. IF NOT DOCUMENTED: State it.

3. **GROWTH - TEMPORAL**: ALL measurements with DATES. Calculate TRENDS (absolute, %, rate, velocity, trajectory). Describe pattern. Interpret using retrieved guidelines (cite ASPEN, WHO, CDC). IF MISSING: Recommend with rationale.

4. **EXAM - TEMPORAL**: If serial, describe progression with dates. Quote findings. Interpret using retrieved guidelines. IF INCOMPLETE: Recommend.

5. **INTAKE - TEMPORAL**: Patterns over time with dates. Quote with timeframes. IF MISSING: Reason about trajectory, recommend quantification.

6. **DIAGNOSIS**: State consistent with ground truth. Synthesize evidence WITH TEMPORAL PATTERNS. Use retrieved guideline criteria (cite ASPEN, WHO, CDC). Specify severity/etiology. Reason with incomplete data.

7. **LABS - TEMPORAL**: ALL with DATES. Describe TRENDS. Interpret using retrieved guidelines. **IF MISSING**: For malnutrition, recommend specific labs WITH schedule. For adequate, explain appropriateness.

8. **CARE PLAN - TEMPORAL MONITORING**: Goals, interventions. **Schedule**: Week 1 (Day 7), Week 2, Weeks 3-4, Months 2-3. **Labs schedule**: Baseline, serial frequency. **Follow-up**: Dates. **Escalation**: With timepoints. **Trajectory**: Recovery timeline.

9. **SOCIAL - TEMPORAL**: Changes with dates. Intervention progression. IF MISSING: Recommend.

10. **INSIGHTS - TEMPORAL**: Summarize with TEMPORAL INTEGRATION and references to retrieved guidelines (cite ASPEN, WHO, CDC). Prognosis with timeline. Decision points with dates. Risks with timeframes. Teaching. Pearls.

**RULES:**
- ANONYMIZE
- CAPTURE ALL TEMPORAL DATA with dates
- CALCULATE TRENDS explicitly
- IDENTIFY TYPE
- QUOTE EXACTLY with dates
- REASON FORWARD with timeline
- USE RETRIEVED GUIDELINES: Cite specific authoritative sources (ASPEN, WHO, CDC) for criteria
- ALIGN WITH GROUND TRUTH

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

**CRITICAL: ENHANCE FORWARD-THINKING + TEMPORAL CAPTURE**
Refined output must recommend what SHOULD BE DONE when missing. Ensure comprehensive temporal capture: ALL measurements with dates, explicit trend calculations, assessment type, temporal significance.

**GROUND TRUTH DIAGNOSIS (MUST SUPPORT):**
{label_context}

If initial extraction contradicts ground truth, CORRECT IT. Use retrieved evidence from authoritative sources (ASPEN, WHO, CDC) for justification.

**ANONYMIZE:** "the patient", "the [age]-year-old", "the family"

**REFINEMENT OBJECTIVES:**

1. **VALIDATE**: Confirm interpretations against retrieved guideline thresholds and criteria from authoritative sources. Validate temporal trend calculations.

2. **CORRECT**: Fix misclassifications with citations from retrieved guidelines (cite specific sources: ASPEN, WHO, CDC). Correct temporal calculations. Align with ground truth. If initial said "Absent" but ground truth is "Present", reframe with temporal decline. If initial said "Present" but ground truth is "Absent", reframe with temporal stability.

3. **ENHANCE**: Add references to retrieved guidelines (cite ASPEN, WHO, CDC as appropriate). **Add temporal detail**: Transform vague into specific with dates. Calculate missing trends. Identify assessment type. Add temporal interpretation. **Add forward-thinking**: Labs with serial schedule, care plan with monitoring intervals, insights with timeline.

4. **FILL GAPS**: Specify severity. Calculate trends if data present. Add recommendations with timeframes. Transform "not documented" into recommendations citing retrieved guidelines and schedules.

5. **ENSURE CONSISTENCY**: Verify diagnosis matches ground truth with temporal evidence. Check temporal consistency (dates, intervals, calculations). Care plan must have actionable timeline.

6. **HANDLE MISSING**: For labs with adequate nutrition: Explain appropriateness. For labs with malnutrition: Recommend specific labs WITH schedule. For anthropometrics: Recommend obtaining. For intake: Recommend quantification. For incomplete temporal: Recommend establishing trend.

**CRITICAL PRINCIPLES:**
- Preserve fidelity
- Quote retrieved guidelines (cite specific authoritative sources: ASPEN, WHO, CDC)
- Flag discrepancies
- EMBED FORWARD-THINKING with timelines
- ENHANCE ALL TEMPORAL DATA: dates, trends, type, significance
- GROUND TRUTH IS ABSOLUTE: Correct to align using temporal evidence

**Z-SCORE AND PERCENTILE VALIDATION:**
- Validate z-score sign with percentile:
  • Percentiles <50th: Negative z-scores (below average, e.g., "3rd %ile" is z-score -1.88)
  • Percentiles >50th: Positive z-scores (above average, e.g., "85th %ile" is z-score +1.04)
- Confirm alignment with clinical descriptions (e.g., "short stature" indicates below average; "well-nourished" indicates normal/above average)
- Flag discrepancies between z-scores, percentiles, and clinical narrative for review
- Verify WHO/ASPEN classification: z < -3 = Severe, -3 to -2 = Moderate, -2 to -1 = Mild risk

**SYNTHESIS GUIDELINES:**
Use retrieved evidence from authoritative sources (ASPEN, WHO, CDC) for all criteria. Present with temporal context. Care plans with schedules. Insights with timelines. Final diagnosis must match ground truth.

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
# EXAMPLE TEMPLATE 2: DIABETES ASSESSMENT
# ============================================================================
# 🎯 NOTE: This is an EXAMPLE template for diabetes tasks.
#    It demonstrates lab value extraction and medication tracking.
#    Your task can be completely different - this is NOT a system requirement!
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
# TEMPLATE REGISTRY - Your Starting Points
# ============================================================================
# 🎯 UNIVERSAL SYSTEM: Pick a template below or create your own!
#
# - "blank": Generic template - customize for any task
# - "malnutrition": Example for nutritional assessment (adapt as needed)
# - "diabetes": Example for diabetes extraction (adapt as needed)
#
# The system works with ANY task. These are just convenient starting points!
# ============================================================================

PROMPT_TEMPLATE_REGISTRY_V1 = {
    "blank": {
        "main": DEFAULT_MAIN_PROMPT,
        "minimal": DEFAULT_MINIMAL_PROMPT,
        "rag_prompt": DEFAULT_RAG_REFINEMENT_PROMPT,
        "description": "⭐ Universal template - Customize for ANY clinical task",
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
        "description": "📚 EXAMPLE: Pediatric malnutrition with temporal reasoning (adapt for your task!)",
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
                        "description": "TEMPORAL SYMPTOM CAPTURE: Document ALL symptoms with ONSET DATES and PROGRESSION: (1) GI: vomiting, diarrhea, reflux, pain; (2) Systemic: fever, fatigue, irritability; (3) Feeding: appetite, refusal; (4) Functional: weakness, activity. Trajectory: new/progressive/stable/improving/resolved. Quote exact descriptions. If NOT documented: State it and recommend systematic review per retrieved evidence.",
                        "required": False
            },
            "growth_and_anthropometrics": {
                "type": "string",
                "description": "TEMPORAL CAPTURE: Extract ALL measurements with DATES. Calculate EXPLICIT TRENDS (absolute change, %, rate, velocity, percentile trajectory). Describe pattern. Interpret using retrieved guidelines. If MISSING: State what's missing and recommend obtaining with rationale.",
                "required": True
            },
            "physical_exam": {
                "type": "string",
                "description": "TEMPORAL: If serial exams, describe progression with dates. Quote exact findings. Interpret using retrieved guidelines. If INCOMPLETE: Recommend exam with rationale.",
                "required": True
            },
            "nutrition_and_intake": {
                "type": "string",
                "description": "TEMPORAL: Document patterns over time with dates and timeframes. Quote with timeframes. If MISSING: Use clinical reasoning about trajectory, recommend quantification.",
                "required": True
            },
            "diagnosis_and_reasoning": {
                "type": "string",
                "description": "TEMPORAL INTEGRATION: Diagnosis (aligned with ground truth), severity, etiology. Synthesize evidence WITH TEMPORAL PATTERNS. Use retrieved guideline criteria. Reason with incomplete data using convergent temporal evidence.",
                "required": True
            },
            "labs_and_screening": {
                "type": "string",
                "description": "TEMPORAL: Extract ALL labs with DATES. Describe TRENDS. Interpret using retrieved guidelines. IF MISSING: For malnutrition, recommend specific labs WITH serial monitoring schedule per guidelines. For adequate nutrition, explain appropriateness.",
                "required": False
            },
            "care_plan": {
                "type": "string",
                "description": "TEMPORAL MONITORING: Goals, interventions with doses. MONITORING SCHEDULE: Specific intervals (Week 1: Day 7; Week 2; Weeks 3-4: weekly; Months 2-3: bi-weekly). LABS SCHEDULE: Baseline panel, serial frequency. FOLLOW-UP: Dates/intervals. ESCALATION: With timepoints (Day 7: if <50g→NG; Week 2: if no response→specialist). TRAJECTORY: Recovery timeline. Justify with temporal reasoning.",
                "required": True
            },
            "social_context": {
                "type": "string",
                "description": "TEMPORAL: Note temporal changes in circumstances with dates. Describe intervention progression. If MISSING: Recommend assessment with rationale.",
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
        "description": "📚 EXAMPLE: Diabetes assessment with labs and meds (adapt for your task!)",
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
                    rag_output += "🔴 CRITICAL: Use the following evidence to support your interpretation.\n"
                    rag_output += "REQUIREMENTS:\n"
                    rag_output += "  • CITE specific sources in your output\n"
                    rag_output += "  • APPLY criteria and guidelines from these documents\n"
                    rag_output += "  • REFERENCE evidence when making decisions\n"
                    rag_output += "  • DO NOT ignore this evidence - it was retrieved for your use\n\n"

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
                function_output += "🔴 CRITICAL: Include ALL calculated values in your JSON output.\n"
                function_output += "REQUIREMENTS:\n"
                function_output += "  • USE exact calculated values (do not recalculate)\n"
                function_output += "  • INCLUDE all function results in your extraction\n"
                function_output += "  • REFERENCE these values in your reasoning\n\n"

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
                    extras_output += "🔴 CRITICAL: Apply these hints and patterns to your extraction.\n"
                    extras_output += f"Retrieved based on keywords: {', '.join(keywords)}\n"
                    extras_output += "REQUIREMENTS:\n"
                    extras_output += "  • APPLY guidance from these hints to your task\n"
                    extras_output += "  • USE patterns and examples shown\n"
                    extras_output += "  • FOLLOW recommendations provided\n\n"
                
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


# ============================================================================
# AGENTIC EXTRACTION PROMPT (v1.0.0 - Agentic with Async)
# ============================================================================

def get_agentic_extraction_prompt(clinical_text: str, label_context: str,
                                   json_schema: str, schema_instructions: str,
                                   user_task_prompt: str = "") -> str:
    """
    Build agentic extraction prompt using USER'S task-specific prompt as PRIMARY

    This function:
    1. Uses the user's main/minimal prompt as the PRIMARY task definition
    2. Fills in {clinical_text} and {label_context} placeholders
    3. Handles {rag_outputs}, {function_outputs}, {extras_outputs} placeholders
    4. Appends agentic tool-calling framework as the execution mechanism

    The user's task-specific prompts (like MALNUTRITION_MAIN_PROMPT) contain
    all the domain expertise, guidelines, synthesis structure, and requirements.
    The agentic framework explains HOW to use tools to complete that task.
    """

    # If user provided a task-specific prompt, use it as PRIMARY
    if user_task_prompt:
        # Fill in the placeholders that we have values for
        try:
            # Try to format the user's prompt with available values
            # Note: User prompts may have {rag_outputs}, {function_outputs}, {extras_outputs}
            # which we don't have yet in agentic mode (they come from tool calls)
            # So we replace them with instructions to use tools

            user_prompt_filled = user_task_prompt.format(
                clinical_text=clinical_text,
                label_context=label_context,
                rag_outputs="[You will retrieve guidelines/evidence by calling query_rag() tool multiple times]",
                function_outputs="[You will perform calculations by calling call_[function_name]() tools as needed]",
                extras_outputs="[You will get supplementary hints by calling query_extras() tool if needed]"
            )
        except KeyError as e:
            # If there are other placeholders we don't know about, leave them
            import re
            user_prompt_filled = user_task_prompt
            # Fill known placeholders using regex to avoid KeyError
            user_prompt_filled = re.sub(r'\{clinical_text\}', clinical_text, user_prompt_filled)
            user_prompt_filled = re.sub(r'\{label_context\}', label_context, user_prompt_filled)
            user_prompt_filled = re.sub(r'\{rag_outputs\}', '[You will retrieve guidelines/evidence by calling query_rag() tool]', user_prompt_filled)
            user_prompt_filled = re.sub(r'\{function_outputs\}', '[You will perform calculations by calling call_[function_name]() tools]', user_prompt_filled)
            user_prompt_filled = re.sub(r'\{extras_outputs\}', '[You will get supplementary hints by calling query_extras() tool]', user_prompt_filled)

        # Build the complete prompt: USER'S TASK PROMPT + AGENTIC FRAMEWORK
        prompt = f"""{user_prompt_filled}

{"=" * 80}
AGENTIC TOOL-CALLING EXECUTION FRAMEWORK
{"=" * 80}

The task description above defines WHAT to extract and HOW to synthesize the information.
This section defines HOW to use tools ITERATIVELY to gather the information you need.

**AVAILABLE TOOLS (call as many times as needed):**

1. **query_rag(query, purpose)**
   - Retrieve clinical guidelines, standards, and evidence from authoritative sources
   - Sources include: ASPEN, WHO, CDC, ADA, AHA, and other clinical guidelines
   - Call MULTIPLE times with different queries to gather comprehensive information
   - Refine queries based on what you learn
   - Example: query_rag("ASPEN pediatric malnutrition severity criteria", "need classification thresholds")

2. **call_[function_name](parameters)**
   - Perform medical calculations: z-scores, BMI, percentiles, growth calculations, lab interpretations
   - Call same function multiple times for serial measurements at different time points
   - Available functions are dynamically listed based on your registry
   - Example: call_percentile_to_zscore({{"percentile": 3}})

3. **query_extras(keywords)**
   - Get supplementary hints, tips, patterns, and task-specific guidance
   - Helps understand domain concepts and best practices
   - Example: query_extras({{"keywords": ["malnutrition", "pediatric", "assessment"]}})

**UNIVERSAL ITERATIVE EXECUTION WORKFLOW:**

**PHASE 1 - INITIAL ANALYSIS:**
1. Read the task prompt → Understand what needs to be extracted
2. Read the clinical text → Identify key metrics, measurements, words, entities
3. Execute INITIAL tool calls:
   - Call functions for calculations (BMI, z-scores, etc.)
   - Call query_extras for task-specific hints

**PHASE 2 - BUILD CONTEXT:**
4. Review function results and extras hints
5. Build RAG keywords based on what you learned (domain, criteria needed, guidelines)
6. Call query_rag to fetch clinical guidelines and standards

**PHASE 3 - ASSESS INFORMATION GAPS:**
7. Determine: Do I have ALL information needed to complete the task?
   - ✅ YES: Proceed to Phase 4
   - ❌ NO: Go to Phase 3b

**PHASE 3b - FILL GAPS:**
8. Identify what other information is needed
9. Determine which tools to call again (can call same tool with different queries)
10. Fetch additional information → Return to Phase 3

**PHASE 4 - COMPLETION:**
11. When you have all necessary information → Output final JSON extraction

**CRITICAL PRINCIPLES:**

- ✅ **Follow Task Definition Above**: The task description above specifies your requirements - follow them exactly
- ✅ **Iterative Rounds**: Call tools across multiple rounds, using results to inform next steps
- ✅ **Self-Assessment**: After each round, assess if you have enough information or need more
- ✅ **Multiple Tool Calls**: Call the same tool multiple times with different queries/parameters
- ✅ **Parallel Execution**: Tools execute in parallel (async) for performance - request multiple at once when possible
- ✅ **Support Ground Truth**: Ensure your extraction supports the ground truth diagnosis
- ✅ **Complete Before Output**: Only output JSON when you have ALL necessary information

**EXAMPLE PHASED EXECUTION:**

```
[PHASE 1 - Initial Analysis]
"Analyzing task: Need to extract malnutrition assessment with z-scores and ASPEN criteria"
"Clinical text shows: 3-year-old, weight 12.5kg (10th percentile), height 92cm (25th percentile)"
"Initial tool calls needed: Convert percentiles to z-scores, get task hints"
→ Call: call_percentile_to_zscore({{"percentile": 10}})
→ Call: call_percentile_to_zscore({{"percentile": 25}})
→ Call: query_extras({{"keywords": ["malnutrition", "pediatric", "z-score", "ASPEN"]}})

[Tools return: 10th percentile = -1.28 z-score, 25th percentile = -0.67 z-score, plus task hints]

[PHASE 2 - Build Context]
"Good! Now I have z-scores: weight -1.28, height -0.67. Extras hint ASPEN criteria needed."
"Building RAG query based on what I learned: need ASPEN pediatric malnutrition severity criteria"
→ Call: query_rag("ASPEN pediatric malnutrition severity classification criteria z-scores", "need severity thresholds")

[RAG returns: ASPEN criteria with z-score cutoffs and multi-indicator requirements]

[PHASE 3 - Assess Gaps]
"Assessing: Do I have all info needed?"
"✓ Have: z-scores, ASPEN criteria, clinical symptoms from text, physical exam findings"
"✓ Can determine: severity classification, diagnosis reasoning"
"Assessment: YES - I have all necessary information to complete the task"

[PHASE 4 - Completion]
"Ready to output final JSON with comprehensive malnutrition assessment"
→ Output JSON
```

**EXPECTED OUTPUT SCHEMA:**
{json_schema}

{schema_instructions}

**CRITICAL RULES:**
- Follow the task-specific requirements and synthesis structure defined at the top
- Use tools iteratively to gather needed information
- Support the ground truth diagnosis with evidence
- Output JSON only when you have completed the task requirements

**Begin your analysis. Call tools as needed to gather information. When ready, provide the final JSON in the exact schema format specified above.**
"""

    else:
        # Fallback: If no user prompt provided, use generic agentic prompt
        prompt = f"""You are a board-certified clinical expert performing structured information extraction from medical text.

**GROUND TRUTH DIAGNOSIS (YOU MUST SUPPORT THIS):**
{label_context}

**CLINICAL TEXT TO ANALYZE:**
{clinical_text}

**YOUR TASK:**
Extract structured clinical information to create comprehensive, expert-level annotations.
Your extraction must support the ground truth diagnosis using evidence from the clinical text.

**AVAILABLE TOOLS (call as many times as needed):**

1. **query_rag(query, purpose)**: Retrieve clinical guidelines and evidence
2. **call_[function_name](parameters)**: Perform medical calculations
3. **query_extras(keywords)**: Get supplementary hints

**AGENTIC WORKFLOW:**
1. Analyze the clinical text and ground truth
2. Call tools to gather needed information (iteratively!)
3. Learn from results, call more tools if needed
4. Complete extraction when you have sufficient information

**EXPECTED OUTPUT SCHEMA:**
{json_schema}

{schema_instructions}

**Begin your analysis. Call tools as needed. Output final JSON when ready.**
"""

    return prompt