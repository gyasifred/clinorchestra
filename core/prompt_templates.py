#!/usr/bin/env python3
"""
Prompt Templates Module - Central repository for all prompt templates
Professional Quality v1.0.0 - Natural clinical language with guideline-based interpretation

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Lab: ClinicalNLP Lab, Biomedical Informatics Center
Version: 1.0.0 - Natural conversation with guideline-based evidence synthesis

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

MALNUTRITION_MAIN_PROMPT = """[PEDIATRIC MALNUTRITION ASSESSMENT - Expert Clinical Data Curation]

You are a board-certified pediatric dietitian curating training data for conversational AI. Use natural, expert clinical language.

**PRIMARY DIRECTIVE:** Use documented values from clinical text. Call tools ONLY to: (1) calculate missing z-scores/percentiles, (2) interpret ambiguous findings, (3) validate calculations.

**CRITICAL WORKFLOW:**

**STEP 1 - IDENTIFY ASSESSMENT TYPE (REQUIRED):**
- Single-point: One encounter → Need ≥1 ASPEN indicator for diagnosis
- Serial same-encounter: Multiple measurements same visit → Need ≥2 indicators
- Longitudinal: Multiple encounters with dates → Need ≥2 indicators
- State type with justification: "Single-point - one encounter 3/15/25" OR "Longitudinal - three encounters (1/15, 2/14, 3/15)"

**STEP 2 - TEMPORAL DATA:**
- Extract ALL measurements with DATES: "Weight 12.5kg on 1/15 (25th %ile, z-score -0.7)"
- Serial/Longitudinal: Calculate trends (absolute, %, rate, velocity, percentile trajectory)
- Single-point: State "Velocity cannot be assessed - single measurement only" + recommend serial

**STEP 3 - Z-SCORE VALIDATION:**

Sign Convention:
- Percentile <50th = NEGATIVE z: 3rd %ile = -1.88, 5th = -1.64, 10th = -1.28, 25th = -0.67
- Percentile >50th = POSITIVE z: 75th = +0.67, 90th = +1.28, 95th = +1.64

Documentation Formats:
- Standard: "BMI-for-age z-score: -2.3" or "Weight at 5th percentile"
- "PERCENTILE z VALUE": "[NUMBER] z [NUMBER]" = percentile FIRST, z-score second
  * "3 z 1.88" = 3rd %ile, z should be -1.88 (NOT +1.88) - CORRECT IT
  * "85 ile z 1.04" = 85th %ile, z correctly +1.04

Validation: If %ile <50th shows positive z → CORRECT to negative. Call tools for ambiguous formats.

**STEP 4 - CLASSIFICATION:**

WHO (Weight-for-Height or BMI-for-Age):
- z<-3: SEVERE, <1st %ile
- -3≤z<-2: MODERATE, 2nd-3rd %ile
- -2≤z<-1: MILD RISK, 3rd-15th %ile
- -1≤z≤+1: NORMAL, 15th-85th %ile
- Height-for-Age z<-2: STUNTED

ASPEN Pediatric:
- Anthropometric: Severe z≤-3, Moderate -2 to -2.9, Mild -1 to -1.9
- Velocity (Serial/Long only): Severe 3 z decline, Moderate 2 z, Mild 1 z
- Other: Inadequate intake <50% ≥1wk; Physical findings (muscle wasting/fat loss)

**STEP 5 - COUNT INDICATORS:**
Count all 4: (1) Anthropometric, (2) Velocity (note if single-point cannot assess), (3) Intake, (4) Physical
State: "ASPEN indicators: X/4 met. [Meets/Exceeds] threshold (single ≥1; serial/long ≥2)"

**STEP 6 - SPECIFIC CRITERIA (NO VAGUE STATEMENTS):**
✓ "Moderate per ASPEN anthropometric z-score -2 to -2.9 (BMI z-score -2.3 on 3/15/25)"
✓ "Velocity: decline 2 z-scores (from -0.5 on 1/15 to -2.5 on 3/15, 59 days)"
❌ "Based on ASPEN criteria", "Meets WHO guidelines"

**STEP 7 - SINGLE-POINT ENHANCEMENT:**
Correlate: anthropometrics + labs + exam + symptoms
Example: "BMI z-score -2.1 corroborated by albumin 2.8 g/dL, edema on exam, lethargy"

**GROUND TRUTH (MUST SUPPORT):**
{label_context}

Extract ALL evidence supporting this using ASPEN, WHO, CDC guidelines.

**SYNTHESIS STRUCTURE (10 sections):**

1. **CASE**: Assessment type with justification FIRST, setting, concern, timeline, family perspective

2. **SYMPTOMS** (temporal): ALL with DATES, trajectory, impact. GI/Systemic/Feeding/Functional categories. If not documented: state and recommend review.

3. **GROWTH** (temporal): ALL with DATES. Serial/Long: trends, velocity. Single: velocity limitation, correlate with labs/clinical, recommend serial. Validate z-score signs. Document WHO/ASPEN with specific values. If missing: recommend.

4. **EXAM** (temporal): Serial: progression with dates. Single: correlate with anthropometric/labs. Quote exact findings. If incomplete: recommend.

5. **INTAKE** (temporal): Patterns with dates, timeframes. If missing: recommend quantification.

6. **DIAGNOSIS**: Consistent with ground truth. State type + threshold. Specific criteria with exact values. Count indicators: "X/4 met. [Meets/Exceeds] threshold". Temporal synthesis. Single: lab/clinical correlation. Severity with criterion reference.

7. **LABS** (temporal): ALL with DATES, TRENDS. Single: corroborate anthropometrics. Guidelines. If missing: for malnutrition recommend panel + schedule; for adequate explain appropriateness.

8. **CARE PLAN** (temporal): Goals, interventions. Schedule: Week 1 (Day 7), Week 2, Weeks 3-4, Months 2-3. Labs: baseline, serial. Follow-up: dates. Escalation: timepoints. Trajectory: timeline. Single: recommend serial.

9. **SOCIAL** (temporal): Changes with dates. If missing: recommend.

10. **INSIGHTS** (temporal): Summary with integration + guidelines. Prognosis with timeline. Decisions with dates. Risks with timeframes. Single: serial follow-up importance.

**CRITICAL RULES:**
- ANONYMIZE always
- Identify type FIRST with justification
- Capture temporal data with dates
- Validate z-score signs (%ile <50th = negative)
- For "PERCENTILE z VALUE": verify sign
- Trends only for serial/long; single note limitation
- Specific criteria with exact values
- Count indicators: X/4, threshold (≥1 single, ≥2 serial/long)
- Single: correlate labs/clinical
- Quote with dates
- Align with ground truth

[END]

CLINICAL TEXT:
{clinical_text}

GROUND TRUTH:
{label_context}

{rag_outputs}

{function_outputs}

{extras_outputs}

{json_schema_instructions}"""


MALNUTRITION_MINIMAL_PROMPT = """[PEDIATRIC MALNUTRITION - Expert Data Curation]

Pediatric dietitian curating AI training data. Natural expert language. Use ASPEN, WHO, CDC guidelines.

**DIRECTIVE:** Use documented values. Call tools to fill gaps, interpret ambiguous, validate.

**WORKFLOW:**

1. **ASSESSMENT TYPE:** Single (≥1 indicator) | Serial/Long (≥2 indicators). State with justification.

2. **TEMPORAL:** ALL with DATES. Serial/Long: calculate trends. Single: note velocity limitation, correlate labs/clinical.

3. **Z-SCORE VALIDATION:** %ile <50th = negative z (3rd=-1.88, 5th=-1.64, 10th=-1.28, 25th=-0.67). "PERCENTILE z VALUE" format: "3 z 1.88" = 3rd %ile, z should be -1.88 → CORRECT.

4. **CLASSIFICATION:**
- WHO: z<-3 Severe, -3 to -2 Moderate, -2 to -1 Mild, -1 to +1 Normal
- ASPEN Anthropometric: z≤-3 Severe, -2 to -2.9 Moderate, -1 to -1.9 Mild
- ASPEN Velocity (serial/long only): 1z Mild, 2z Moderate, 3z Severe. Single: cannot assess.
- 4 indicators: Anthropometric, Velocity, Intake <50% ≥1wk, Physical

5. **COUNT:** "ASPEN indicators: X/4 met. [Meets/Exceeds] threshold (single ≥1; serial/long ≥2)"

6. **SPECIFIC CRITERIA:**
✓ "Moderate per ASPEN z-score -2 to -2.9 (z-score -2.3 on 3/15)"
❌ "Based on ASPEN" - TOO VAGUE

**GROUND TRUTH (SUPPORT):**
{label_context}

**MALNUTRITION PRESENT:** Synthesize deficits with trends, exam with serial changes, intake with duration, velocity (if assessable), labs with trends per guidelines.

**MALNUTRITION ABSENT:** Synthesize normal with stable tracking, well-nourished with consistency, adequate intake, stable growth over time.

**ANONYMIZE:** "the patient", "the [age]-year-old", "the family"

**SYNTHESIS (10 sections):**

1. CASE: Setting, concern, timeline. State type with justification.
2. SYMPTOMS (temporal): ALL with DATES. GI/Systemic/Feeding/Functional. Trajectory. Quote. If not documented: state.
3. GROWTH (temporal): ALL with DATES. Serial/Long: trends, velocity. Single: limitation, correlate labs/clinical, recommend. Validate z-signs. WHO/ASPEN with specific values. If missing: recommend.
4. EXAM (temporal): Serial: progression. Single: correlate anthropometric/labs. Quote. If incomplete: recommend.
5. INTAKE (temporal): Patterns with dates. If missing: recommend.
6. DIAGNOSIS: Consistent with ground truth. State type + threshold. Specific criteria with values. Count: "X/4 met. [Meets/Exceeds] threshold". Temporal synthesis. Single: lab/clinical correlation. Severity with criterion.
7. LABS (temporal): ALL with DATES, TRENDS. Single: corroborate anthropometrics. Guidelines. If missing: for malnutrition panel + schedule; for adequate explain.
8. CARE PLAN (temporal): Goals, interventions. Schedule: Week 1 (Day 7), Week 2, Weeks 3-4, Months 2-3. Labs: baseline, serial. Follow-up: dates. Escalation: timepoints. Trajectory. Single: recommend serial.
9. SOCIAL (temporal): Changes with dates. If missing: recommend.
10. INSIGHTS (temporal): Summary with integration + guidelines. Prognosis with timeline. Decisions with dates. Risks with timeframes. Single: serial importance.

**RULES:**
- ANONYMIZE
- Identify type with justification
- Capture temporal with dates
- Validate z-signs (%ile <50th = negative)
- Trends only serial/long; single note limitation
- Specific criteria with values
- Count: X/4, threshold (≥1 single, ≥2 serial/long)
- Single: correlate labs/clinical
- Quote with dates
- Align with ground truth

[END]

CLINICAL TEXT:
{clinical_text}

GROUND TRUTH:
{label_context}

{rag_outputs}

{function_outputs}

{extras_outputs}

{json_schema_instructions}"""


MALNUTRITION_RAG_REFINEMENT_PROMPT = """[RAG REFINEMENT - Malnutrition]

Refining preliminary assessment using guideline evidence. Expert curating AI training data.

**DIRECTIVE:** Use text/initial values. Enhance or correct when necessary.

**GROUND TRUTH (SUPPORT):**
{label_context}

If initial contradicts, CORRECT using ASPEN, WHO, CDC guidelines.

**ANONYMIZE:** "the patient", "the [age]-year-old", "the family"

**REFINEMENT:**

1. **VALIDATE TYPE:** Confirm single/serial/long correct with dates. Verify interpretation matches type (single can't assess velocity, serial/long calculates trends).

2. **VALIDATE CRITERIA:** Check exact values present:
✓ "Moderate per ASPEN z-score -2 to -2.9 (BMI z-score -2.3 on 3/15/25)"
❌ "Based on ASPEN"
Verify indicators counted: "X/4 met". Confirm threshold: ≥1 single, ≥2 serial/long.

3. **VALIDATE Z-SIGNS:** Standard: "BMI z-score: -2.3" or "Weight 5th %ile". Alternative "PERCENTILE z VALUE": "[NUMBER] z [NUMBER]" = %ile first, z second. "3 z 1.88" = 3rd %ile, z should be -1.88 → CORRECT. Rule: %ile <50th MUST have negative z.

4. **VALIDATE COUNT:** Count: Anthropometric + Velocity + Intake + Physical. Threshold: Single ANY 1 = diagnostic. Serial/Long ≥2 = diagnostic. If single with 1 indicator classified "not malnourished": INCORRECT. If serial/long <2 indicators but z<-2: check WHO justifies. If serial/long <2 indicators and z≥-2: should be "not malnourished".

5. **VALIDATE TEMPORAL:** Confirm trends correct. Validate z-signs match %iles.

6. **CORRECT:** Misclassifications with citations. Temporal calculations. Threshold application. Align with ground truth. Fix vague to specific criterion. Fix threshold errors.

7. **ENHANCE:** Add specific criterion where generic. Add temporal detail with dates. Calculate missing trends (if serial/long). Identify type if missing. Add forward-thinking: labs with schedule, care with intervals, insights with timeline. Add lab/clinical correlation for single.

8. **FILL GAPS:** Specify severity with criterion. Calculate trends if data + serial/long. Add recommendations with timeframes. Transform "not documented" → recommendations with guidelines + schedules. Single: add lab/clinical correlation if missing.

9. **CONSISTENCY:** Diagnosis matches ground truth with temporal evidence. Temporal consistency. ASPEN count accurate. Correct threshold applied (1 single, 2+ serial/long). Care plan has timeline. Single has lab/clinical correlation.

10. **HANDLE MISSING:** Labs with adequate: explain appropriateness. Labs with malnutrition: recommend panel + schedule. Anthropometrics: recommend. Intake: recommend quantification. Incomplete temporal (single): recommend serial. Single without lab correlation: add recommendation.

**PRINCIPLES:**
- Preserve fidelity
- Specific criteria: transform vague to "per ASPEN z-score -2 to -2.9 (z-score -2.3)"
- Verify type drives interpretation
- Verify threshold: 1 single, 2+ serial/long
- Verify single has lab/clinical correlation
- Quote guidelines (cite ASPEN, WHO, CDC)
- Flag discrepancies
- Embed forward-thinking with timelines
- Enhance temporal: dates, trends, type, significance
- Validate count: explicitly "X/4", verify threshold
- Correct z-signs (%ile <50th = negative)
- For "PERCENTILE z VALUE": %ile <50th needs negative z
- Ground truth absolute
- Single more liberal: any 1 diagnostic

**Z-SCORE/PERCENTILE:**
- %ile <50th: negative z (3rd=-1.88, 5th=-1.64, 10th=-1.28, 25th=-0.67)
- %ile >50th: positive z (75th=+0.67, 90th=+1.28, 95th=+1.64)
- Align with clinical ("short"=below avg; "well-nourished"=normal/above)
- Flag discrepancies
- Verify WHO/ASPEN: z<-3 Severe, -3 to -2 Moderate, -2 to -1 Mild

**SYNTHESIS:**
Use guidelines. Present with temporal context. Care with schedules. Insights with timelines. Final matches ground truth. Single correlates labs/clinical.

[END]

ORIGINAL TEXT:
{clinical_text}

GROUND TRUTH:
{label_context}

INITIAL EXTRACTION:
{stage3_json_output}

EVIDENCE BASE:
{retrieved_evidence_chunks}

{json_schema_instructions}

Return ONLY JSON: Start {{ end }}. No markdown. Refine using guideline evidence."""


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
        "version": "1.0.0",
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
        "version": "1.0.0",
        "schema": {
            "malnutrition_status": {
                "type": "string",
                "description": "'Malnutrition Present' or 'Malnutrition Absent'",
                "required": True
            },
            "assessment_type": {
                "type": "string",
                "description": "Single-point (≥1 indicator) | Serial same-encounter (≥2) | Longitudinal (≥2). State with justification and dates.",
                "required": True
            },
            "case_presentation": {
                "type": "string",
                "description": "Assessment type first with justification, then setting, concern, timeline, family perspective.",
                "required": True
            },
            "clinical_symptoms_and_signs": {
                "type": "string",
                "description": "Temporal symptoms with onset dates and progression. GI/Systemic/Feeding/Functional. Trajectory and impact. If not documented: state and recommend review.",
                "required": False
            },
            "growth_and_anthropometrics": {
                "type": "string",
                "description": "Temporal with ALL measurements and dates. Validate z-score signs. Serial/Long: trends, velocity. Single: note limitation, correlate labs/clinical. WHO/ASPEN criteria with specific values.",
                "required": True
            },
            "physical_exam": {
                "type": "string",
                "description": "Temporal. Serial: progression with dates. Single: correlate with anthropometric/labs. Quote exact findings.",
                "required": True
            },
            "nutrition_and_intake": {
                "type": "string",
                "description": "Temporal patterns with dates and timeframes. Quantify percentage of needs. Document ASPEN inadequate intake if met.",
                "required": True
            },
            "labs_and_screening": {
                "type": "string",
                "description": "Temporal: ALL labs with dates and trends. Single: corroborate anthropometrics. If missing: recommend panel/schedule or explain appropriateness.",
                "required": False
            },
            "aspen_indicator_count": {
                "type": "string",
                "description": "Required count: 'ASPEN indicators: X/4 met'. List each. Verify threshold: single ≥1, serial/long ≥2. State meets/exceeds.",
                "required": True
            },
            "diagnosis_and_reasoning": {
                "type": "string",
                "description": "Align with ground truth. State type + threshold. Specific criteria with exact values (not vague). Include indicator count. Temporal synthesis. Single: lab/clinical correlation. Severity with criterion reference.",
                "required": True
            },
            "care_plan": {
                "type": "string",
                "description": "Temporal monitoring: goals, interventions, schedules (Week 1-4, Months 2-3), labs schedule, follow-up dates, escalation timepoints, trajectory timeline. Single: recommend serial.",
                "required": True
            },
            "social_context": {
                "type": "string",
                "description": "Temporal: food security, resources, barriers. Changes with dates. If missing: recommend assessment.",
                "required": False
            },
            "clinical_insights": {
                "type": "string",
                "description": "Temporal synthesis: summary with integration + guidelines. Prognosis with timeline. Decision points with dates. Risk factors with timeframes. Single: serial follow-up importance.",
                "required": True
            }
        }
    },
    "diabetes": {
        "main": DIABETES_MAIN_PROMPT,
        "minimal": DIABETES_MINIMAL_PROMPT,
        "rag_prompt": DEFAULT_RAG_REFINEMENT_PROMPT,
        "description": "📚 EXAMPLE: Diabetes assessment with labs and meds (adapt for your task!)",
        "version": "1.0.0",
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

    CRITICAL FIX: Now includes FAILED tool results with error analysis to help LLM
    learn from mistakes and correct parameters in next iteration.

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

    # Track failed tools for error section
    failed_functions = []
    failed_rag = []
    failed_extras = []

    for result in tool_results:
        tool_type = result.get('type', '').lower()
        success = result.get('success', False)

        # CRITICAL FIX: Track failed tools instead of skipping them
        if not success:
            if tool_type == 'function' and include_functions:
                failed_functions.append(result)
            elif tool_type == 'rag' and include_rag:
                failed_rag.append(result)
            elif tool_type == 'extras' and include_extras:
                failed_extras.append(result)
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

    # NEW: Add failed tools section with error analysis and correction guidance
    error_output = ""

    if failed_functions or failed_rag or failed_extras:
        error_output += "\n[⚠️  TOOL ERRORS - ANALYZE AND CORRECT]\n"
        error_output += "━" * 60 + "\n"
        error_output += "🔴 CRITICAL: The following tools FAILED. You MUST learn from these errors.\n\n"

    # Failed functions with parameter correction guidance
    if failed_functions:
        error_output += "❌ FAILED FUNCTIONS:\n"
        error_output += "=" * 60 + "\n\n"

        for i, fail in enumerate(failed_functions, 1):
            func_name = fail.get('name', 'unknown')
            error_message = fail.get('message', 'Unknown error')
            attempted_params = fail.get('parameters', {})

            error_output += f"FUNCTION #{i}: {func_name}\n"
            error_output += f"{'─' * 50}\n"
            error_output += f"ATTEMPTED PARAMETERS:\n"
            for key, value in attempted_params.items():
                error_output += f"  • {key} = {value}\n"
            error_output += f"\nERROR MESSAGE:\n  {error_message}\n\n"

            # Intelligent error analysis
            error_output += "ERROR ANALYSIS & FIX:\n"

            if "missing" in error_message.lower() and "required" in error_message.lower():
                # Missing required parameter
                missing_param = _extract_missing_parameter(error_message)
                if missing_param:
                    error_output += f"  ⚠️  MISSING REQUIRED PARAMETER: '{missing_param}'\n"
                    error_output += f"  ✓ FIX: Add '{missing_param}' parameter with appropriate value\n"

                    # Check if they used wrong parameter name
                    similar_params = [p for p in attempted_params.keys()
                                     if missing_param in p or p in missing_param]
                    if similar_params:
                        error_output += f"  💡 NOTE: You used '{similar_params[0]}' but function needs '{missing_param}'\n"
                        error_output += f"     Example: Change {similar_params[0]}=value to {missing_param}=value\n"

                    # Provide correct function signature hint
                    error_output += f"\n  📋 CHECK FUNCTION SIGNATURE:\n"
                    error_output += f"     Review the function definition to see required parameters\n"
                    error_output += f"     Required parameters MUST be provided\n"

            elif "unexpected keyword argument" in error_message.lower():
                # Wrong parameter name
                wrong_param = _extract_unexpected_parameter(error_message)
                if wrong_param:
                    error_output += f"  ⚠️  INVALID PARAMETER: '{wrong_param}'\n"
                    error_output += f"  ✓ FIX: This parameter doesn't exist in the function\n"
                    error_output += f"     Check the function signature for correct parameter names\n"
                    error_output += f"     Remove '{wrong_param}' or rename to correct parameter\n"

            elif "invalid" in error_message.lower() or "type" in error_message.lower():
                # Type/value error
                error_output += f"  ⚠️  INVALID PARAMETER VALUE\n"
                error_output += f"  ✓ FIX: Check parameter types and value formats\n"
                error_output += f"     Ensure values match expected types (string, number, etc.)\n"

            else:
                # General error
                error_output += f"  ⚠️  FUNCTION EXECUTION FAILED\n"
                error_output += f"  ✓ FIX: Review error message and adjust parameters accordingly\n"

            error_output += f"\n{'─' * 50}\n\n"

        error_output += "🔧 NEXT STEPS FOR FUNCTIONS:\n"
        error_output += "  1. ANALYZE the error messages above\n"
        error_output += "  2. IDENTIFY incorrect or missing parameters\n"
        error_output += "  3. CORRECT parameter names and values\n"
        error_output += "  4. If in ADAPTIVE mode: RETRY with corrected parameters\n"
        error_output += "  5. If in STRUCTURED mode: Use corrected understanding for extraction\n\n"

    # Failed RAG queries
    if failed_rag:
        error_output += "❌ FAILED RAG QUERIES:\n"
        error_output += "=" * 60 + "\n\n"

        for i, fail in enumerate(failed_rag, 1):
            query = fail.get('query', 'unknown')
            error_message = fail.get('message', 'Unknown error')

            error_output += f"QUERY #{i}: \"{query}\"\n"
            error_output += f"ERROR: {error_message}\n\n"

        error_output += "🔧 NEXT STEPS FOR RAG:\n"
        error_output += "  1. If RAG not configured: Continue without RAG evidence\n"
        error_output += "  2. If query failed: Try different query or proceed without\n"
        error_output += "  3. ADAPTIVE mode: Can request different RAG query\n\n"

    # Failed extras
    if failed_extras:
        error_output += "❌ FAILED EXTRAS:\n"
        error_output += "=" * 60 + "\n\n"

        for i, fail in enumerate(failed_extras, 1):
            keywords = fail.get('keywords', [])
            error_message = fail.get('message', 'Unknown error')

            error_output += f"KEYWORDS #{i}: {', '.join(keywords)}\n"
            error_output += f"ERROR: {error_message}\n\n"

        error_output += "🔧 NEXT STEPS FOR EXTRAS:\n"
        error_output += "  1. Continue extraction without these hints\n"
        error_output += "  2. ADAPTIVE mode: Can try different keywords\n\n"

    if error_output:
        error_output += "━" * 60 + "\n"
        error_output += "⚡ IMPORTANT: Learn from these errors and DO NOT repeat the same mistakes!\n"
        error_output += "━" * 60 + "\n"

    # Append errors to function output (most relevant location)
    if error_output and function_output:
        function_output += "\n" + error_output
    elif error_output:
        function_output = error_output

    return {
        'rag_outputs': rag_output,
        'function_outputs': function_output,
        'extras_outputs': extras_output
    }


def _extract_missing_parameter(error_message: str) -> str:
    """Extract missing parameter name from error message"""
    import re
    # Pattern: "missing 1 required positional argument: 'param_name'"
    match = re.search(r"argument:\s*'([^']+)'", error_message)
    if match:
        return match.group(1)
    # Pattern: "Required parameter param_name missing"
    match = re.search(r"parameter\s+(\w+)\s+missing", error_message, re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def _extract_unexpected_parameter(error_message: str) -> str:
    """Extract unexpected parameter name from error message"""
    import re
    # Pattern: "unexpected keyword argument 'param_name'"
    match = re.search(r"argument\s*'([^']+)'", error_message)
    if match:
        return match.group(1)
    return ""


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

**TASK-DRIVEN EXECUTION WORKFLOW:**

**PHASE 1 - IDENTIFY TASK REQUIREMENTS:**
1. Read the task description above → Identify exactly which schema fields are required
2. Read the clinical text → Extract values that can be directly copied (no tools needed)
3. Identify which schema fields require tool calls:
   - Fields needing calculations (e.g., schema field "bmi_zscore" → call calculate_bmi_zscore function)
   - Fields referencing guidelines (e.g., schema field description "classify per ASPEN criteria" → call query_rag for ASPEN)
   - DO NOT call tools for fields that can be extracted directly from text

**PHASE 2 - EXECUTE REQUIRED TOOLS:**
4. Call ONLY the tools identified as required by the task schema in Phase 1
5. DO NOT call tools speculatively or for exploration purposes
6. Each tool call must fulfill a specific schema field requirement

**PHASE 3 - COMPLETE EXTRACTION:**
7. Use tool results to fill schema fields that required calculations/retrieval
8. Extract remaining schema fields directly from clinical text
9. Output final JSON matching the exact schema structure

🔴 CRITICAL: Follow the task description and schema requirements exactly.
DO NOT make autonomous decisions about which tools to call.

**CRITICAL PRINCIPLES:**

- **Follow Task Definition Above**: The task description above specifies your requirements - follow them exactly
- **Task-Required Tools Only**: Call tools ONLY when the task explicitly requires them
- **No Exploration**: DO NOT call tools to "gather context" or "explore" - only to fulfill task requirements
- **Multiple Tool Calls**: Call the same tool multiple times if the task requires multiple calculations
- **Parallel Execution**: Tools execute in parallel (async) for performance - request multiple at once when possible
- **Support Ground Truth**: Ensure your extraction supports the ground truth diagnosis
- **Complete Extraction**: Output JSON with all required schema fields filled

**EXAMPLE TASK-DRIVEN EXECUTION:**

```
[PHASE 1 - Identify Task Requirements]
"Reading task description: Task requires 'growth_and_anthropometrics' field"
"Task says: 'Validate z-score signs. WHO/ASPEN criteria with specific values.'"
"Task says: 'Call tools ONLY to: (1) calculate missing z-scores'"
"Clinical text shows: 3-year-old, weight 12.5kg (10th percentile), height 92cm (25th percentile)"

"Required tools based on task:"
"- Task requires z-scores → need to convert percentiles"
"- Task mentions 'WHO/ASPEN criteria' → need to retrieve ASPEN guidelines"

→ Call: call_percentile_to_zscore({{"percentile": 10}})
→ Call: call_percentile_to_zscore({{"percentile": 25}})
→ Call: query_rag("ASPEN pediatric malnutrition classification criteria", "Task requires ASPEN criteria")

[Tools return: 10th percentile = -1.28 z-score, 25th percentile = -0.67 z-score, ASPEN criteria]

[PHASE 2 - Execute Required Tools]
"Tool results received. All task-required tools have been called."

[PHASE 3 - Complete Extraction]
"Mapping results to schema fields:"
"- growth_and_anthropometrics: Using z-scores (-1.28, -0.67) and ASPEN criteria from RAG"
"- Extracting other fields directly from clinical text"
"- All schema fields filled per task requirements"

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
