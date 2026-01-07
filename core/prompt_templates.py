#!/usr/bin/env python3
"""
Prompt Templates Module - Central repository for all prompt templates
Professional Quality v1.0.0 - Natural clinical language with guideline-based interpretation

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Lab: ClinicalNLP Lab, Biomedical Informatics Center
Version: 1.0.0 - Natural conversation with guideline-based evidence synthesis

═══════════════════════════════════════════════════════════════════════════
 TRULY UNIVERSAL SYSTEM - Works for ANY Clinical Task
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

DEFAULT_MAIN_PROMPT = """
╔════════════════════════════════════════════════════════════════════════════╗
║                          TASK DESCRIPTION                               ║
╚════════════════════════════════════════════════════════════════════════════╝

You are a clinical expert analyzing medical records for structured information extraction.

YOUR TASK:
Extract and synthesize clinical information from the provided text, following the JSON schema exactly.

EXTRACTION GUIDELINES:
- Be precise and accurate
- Use null for unknown values
- Maintain clinical terminology
- Extract only factual information from the text
- ANONYMIZE: Use "the patient", "the [age]-year-old", or "the family" (NEVER use names)

FUNCTION CALLING:
Ensure parameters are in correct units. Convert units using conversion functions before calling primary functions.

════════════════════════════════════════════════════════════════════════════


╔════════════════════════════════════════════════════════════════════════════╗
║                    CLINICAL TEXT (PRIMARY SOURCE DATA)                  ║
╚════════════════════════════════════════════════════════════════════════════╝

 CRITICAL: This is the PRIMARY CLINICAL NOTE you must extract from.
    All other sections below provide SUPPORTING INFORMATION ONLY.

CLINICAL NOTE CONTENT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{clinical_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


╔════════════════════════════════════════════════════════════════════════════╗
║                          CLASSIFICATION CONTEXT                         ║
╚════════════════════════════════════════════════════════════════════════════╝

{label_context}


{rag_outputs}

{function_outputs}

{extras_outputs}

{json_schema_instructions}"""

DEFAULT_MINIMAL_PROMPT = """
╔════════════════════════════════════════════════════════════════════════════╗
║                    TASK (MINIMAL MODE - CONCISE)                        ║
╚════════════════════════════════════════════════════════════════════════════╝

Extract clinical information in JSON format, following the schema exactly.

CRITICAL RULES:
- Extract factual information only
- Use null for unknown values
- ANONYMIZE patient/family names
- Convert units before calling functions

════════════════════════════════════════════════════════════════════════════


╔════════════════════════════════════════════════════════════════════════════╗
║                    CLINICAL TEXT (PRIMARY SOURCE)                       ║
╚════════════════════════════════════════════════════════════════════════════╝

 CRITICAL: Extract from THIS clinical note.
    Supporting info provided below.

{clinical_text}

════════════════════════════════════════════════════════════════════════════


CLASSIFICATION:
{label_context}

{rag_outputs}

{function_outputs}

{extras_outputs}

{json_schema_instructions}"""

DEFAULT_RAG_REFINEMENT_PROMPT = """[RAG REFINEMENT TASK]

You are refining a preliminary extraction using evidence from authoritative sources.

CRITICAL ANONYMIZATION:
- NEVER use patient or family names
- ALWAYS use: "the patient", "the [age]-year-old", "the family"

REFINEMENT OBJECTIVES:

1. VALIDATE: Confirm interpretations against guideline criteria, verify accuracy

2. CORRECT: Adjust misclassifications with citations, align with evidence-based practice

3. ENHANCE: Add guideline interpretations, prognostic information, diagnostic criteria

4. FILL GAPS: Complete missing but important details supported by evidence

5. ENSURE CONSISTENCY: Verify assessments align with findings and support classification

6. HANDLE MISSING DATA: For null/not documented fields, provide appropriate clinical reasoning

CRITICAL PRINCIPLES:
- Preserve fidelity: Never remove correct data or fabricate information
- Quote sources when correcting: "Per [guideline/source]..."
- Flag discrepancies clearly
- Add value only when evidence clearly applies
- Maintain expert clinical tone
- ANONYMIZE always

[END RAG REFINEMENT TASK]

ORIGINAL TEXT:
{clinical_text}

CLASSIFICATION CONTEXT:
{label_context}

INITIAL EXTRACTION:
{stage3_json_output}

EVIDENCE BASE:
{retrieved_evidence_chunks}

{json_schema_instructions}

Return ONLY JSON in the exact schema format. No markdown. Use evidence to refine extraction."""

# ============================================================================
# EXAMPLE TEMPLATE 1: MALNUTRITION ASSESSMENT
# ============================================================================
#  NOTE: This is an EXAMPLE template for malnutrition tasks.
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
 "Moderate per ASPEN anthropometric z-score -2 to -2.9 (BMI z-score -2.3 on 3/15/25)"
 "Velocity: decline 2 z-scores (from -0.5 on 1/15 to -2.5 on 3/15, 59 days)"
 "Based on ASPEN criteria", "Meets WHO guidelines"

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
 "Moderate per ASPEN z-score -2 to -2.9 (z-score -2.3 on 3/15)"
 "Based on ASPEN" - TOO VAGUE

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
 "Moderate per ASPEN z-score -2 to -2.9 (BMI z-score -2.3 on 3/15/25)"
 "Based on ASPEN"
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
#  NOTE: This is an EXAMPLE template for diabetes tasks.
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
You are an intelligent task analyst for clinical data extraction. Your job: understand the extraction task, analyze available data, and determine which tools would help complete the task.

[EXTRACTION TASK]
{task_description}

OUTPUT SCHEMA:
{json_schema}

[AVAILABLE TOOLS]
{available_tools_description}

[YOUR ANALYSIS TASK]

**STEP 1 - UNDERSTAND REQUIREMENTS:**
- Review task description → understand WHAT to extract and HOW
- Review output schema → understand required fields and structure

**STEP 2 - ANALYZE AVAILABLE DATA:**
- Read clinical text → identify what information is currently available
- Identify available values, measurements, and mentions

**STEP 3 - GAP ANALYSIS:**
Determine what's missing or needs transformation:
- Calculations needed? (e.g., available values need to be computed/converted)
- Guidelines/criteria needed? (e.g., task mentions standards to apply)
- Context/hints needed? (e.g., domain knowledge would clarify interpretation)

**STEP 4 - SELECT TOOLS:**
Based on gaps, determine which tools are REQUIRED:
- **Functions**: Call when calculations, conversions, or computations are needed
- **RAG**: Call when guidelines, criteria, or evidence-based standards would help
- **Extras**: Call when supplementary context or domain knowledge would assist

[CLINICAL TEXT TO ANALYZE]
{clinical_text}

[CLASSIFICATION/LABEL CONTEXT]
{label_context}

[OUTPUT FORMAT]
Return JSON with this EXACT structure:
{{
  "analysis": "Brief analysis of required information and tools that will help",
  "tool_requests": [
    {{
      "tool": "function",
      "name": "<function_name>",
      "arguments": {{"param1": value1, "param2": value2}},
      "reasoning": "Why this function is needed to complete the task"
    }},
    {{
      "tool": "rag",
      "keywords": ["<keyword1>", "<keyword2>", "<keyword3>"],
      "reasoning": "What guidelines/evidence this will retrieve and why needed"
    }},
    {{
      "tool": "extras",
      "keywords": ["<keyword1>", "<keyword2>", "<keyword3>"],
      "reasoning": "What contextual information this will provide"
    }}
  ]
}}

CRITICAL PRINCIPLES:
- Extract parameters from clinical text (ages, weights, measurements, dates, etc.)
- Build queries from task requirements and schema fields
- Select tools that DIRECTLY support completing the required output
- Tools must serve the task requirements, not unrelated exploration
- For RAG/Extras: Use RELEVANT keywords that will find NEW information to fill gaps
- DO NOT repeat the same keywords that were used before - vary terms to get diverse results"""

# ============================================================================
# TEMPLATE REGISTRY - Your Starting Points
# ============================================================================
#  UNIVERSAL SYSTEM: Pick a template below or create your own!
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
        "description": " EXAMPLE: Pediatric malnutrition with temporal reasoning (adapt for your task!)",
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
        "description": " EXAMPLE: Diabetes assessment with labs and meds (adapt for your task!)",
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

    CRITICAL IMPROVEMENT: Uses ULTRA-CLEAR section markers to prevent LLM confusion
    - Clinical notes vs tool outputs are CLEARLY distinguished
    - Each tool type has distinct visual formatting
    - Purpose of each section is explicitly stated

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
            # Format RAG chunks with ULTRA-CLEAR visual separation
            results_list = result.get('results', [])
            if results_list:
                if not rag_output:
                    rag_output = "\n\n" + "╔" + "═"*78 + "╗\n"
                    rag_output += "║" + " "*20 + "EVIDENCE FROM GUIDELINES & LITERATURE" + " "*18 + "║\n"
                    rag_output += "╚" + "═"*78 + "╝\n\n"
                    rag_output += " CRITICAL: This section contains AUTHORITATIVE EVIDENCE retrieved from\n"
                    rag_output += "    clinical guidelines and medical literature.\n\n"
                    rag_output += "PURPOSE: Use this evidence to:\n"
                    rag_output += "    - SUPPORT your clinical interpretations\n"
                    rag_output += "    - APPLY diagnostic criteria and guidelines\n"
                    rag_output += "    - CITE specific sources in your extraction\n"
                    rag_output += "    - REFERENCE evidence when making decisions\n\n"
                    rag_output += " DO NOT CONFUSE WITH CLINICAL NOTES - This is REFERENCE MATERIAL!\n"
                    rag_output += "━"*80 + "\n\n"

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

                        rag_output += f"┌─ EVIDENCE #{i} ─ RELEVANCE: {score:.2f} " + "─"*40 + "\n"
                        rag_output += f"│ SOURCE: {source}\n"
                        if source_filename and source_filename != source:
                            rag_output += f"│ FILE: {source_filename}\n"
                        rag_output += "│\n"
                        rag_output += f"│ CONTENT:\n"
                        for line in text[:1500].split('\n'):
                            rag_output += f"│   {line}\n"
                        rag_output += "└" + "─"*78 + "\n\n"

        elif tool_type == 'function' and include_functions:
            func_name = result.get('name', 'unknown')
            func_result = result.get('result', {})
            date_context = result.get('date_context', '')

            if not function_output:
                function_output = "\n\n" + "╔" + "═"*78 + "╗\n"
                function_output += "║" + " "*22 + "CALCULATED VALUES & FUNCTION OUTPUTS" + " "*17 + "║\n"
                function_output += "╚" + "═"*78 + "╝\n\n"
                function_output += " CRITICAL: This section contains COMPUTED RESULTS from mathematical\n"
                function_output += "    functions applied to data from the clinical notes.\n\n"
                function_output += "PURPOSE OF EACH CALCULATION:\n"
                function_output += "    - Each function was called to FILL A GAP in the clinical notes\n"
                function_output += "    - The calculation addresses a SPECIFIC missing data point\n"
                function_output += "    - USE these exact calculated values in your JSON output\n"
                function_output += "    - DO NOT recalculate - the computation is already done!\n\n"
                function_output += " DO NOT CONFUSE WITH CLINICAL NOTES - These are DERIVED VALUES!\n"
                function_output += "━"*80 + "\n\n"

            # Include date context for serial measurements
            function_output += "┌─ FUNCTION RESULT " + "─"*60 + "\n"
            function_output += f"│  FUNCTION NAME: {func_name}\n"
            if date_context:
                function_output += f"│ DATE CONTEXT: {date_context}\n"
            function_output += "│\n"
            function_output += "│ WHAT THIS CALCULATION FILLS:\n"
            function_output += "│   This function was called because the clinical notes were\n"
            function_output += "│   missing computed/derived data. Use this result to populate\n"
            function_output += "│   the corresponding field in your JSON output.\n"
            function_output += "│\n"
            function_output += "│ COMPUTED RESULT:\n"

            if isinstance(func_result, dict):
                for key, value in func_result.items():
                    function_output += f"│   - {key}: {value}\n"
            else:
                function_output += f"│   - Result: {func_result}\n"

            function_output += "│\n"
            function_output += "│  CRITICAL: Include this exact value in your extraction!\n"
            function_output += "└" + "─"*78 + "\n\n"

        elif tool_type == 'extras' and include_extras:
            # ENHANCED: Format extras with ULTRA-CLEAR visual separation
            items = result.get('results', [])
            keywords = result.get('keywords', [])

            if items:
                if not extras_output:
                    extras_output = "\n\n" + "╔" + "═"*78 + "╗\n"
                    extras_output += "║" + " "*25 + "CLINICAL HINTS & GUIDELINES" + " "*24 + "║\n"
                    extras_output += "╚" + "═"*78 + "╝\n\n"
                    extras_output += " CRITICAL: This section contains DOMAIN KNOWLEDGE & CLINICAL PATTERNS\n"
                    extras_output += f"    Retrieved based on task keywords: {', '.join(keywords)}\n\n"
                    extras_output += "PURPOSE: These hints help you:\n"
                    extras_output += "    - UNDERSTAND clinical patterns relevant to this task\n"
                    extras_output += "    - APPLY best practices for this type of extraction\n"
                    extras_output += "    - FOLLOW domain-specific guidelines\n"
                    extras_output += "    - INTERPRET clinical findings correctly\n\n"
                    extras_output += " DO NOT CONFUSE WITH CLINICAL NOTES - These are GUIDANCE HINTS!\n"
                    extras_output += "━"*80 + "\n\n"

                for i, item in enumerate(items, 1):
                    content = item.get('content', '')
                    item_type = item.get('type', 'hint')
                    relevance = item.get('relevance_score', 0)
                    matched_kw = item.get('matched_keywords', [])

                    if content:
                        extras_output += f"┌─ HINT #{i} ({item_type.upper()}) " + "─"*50 + "\n"
                        if matched_kw:
                            extras_output += f"│ MATCHED: {', '.join(matched_kw)}\n│\n"
                        extras_output += "│ GUIDANCE:\n"
                        for line in content.split('\n'):
                            extras_output += f"│   {line}\n"
                        extras_output += "└" + "─"*78 + "\n\n"

    # NEW: Add failed tools section with error analysis and correction guidance
    error_output = ""

    if failed_functions or failed_rag or failed_extras:
        error_output += "\n[️  TOOL ERRORS - ANALYZE AND CORRECT]\n"
        error_output += "━" * 60 + "\n"
        error_output += " CRITICAL: The following tools FAILED. You MUST learn from these errors.\n\n"

    # Failed functions with parameter correction guidance
    if failed_functions:
        error_output += " FAILED FUNCTIONS:\n"
        error_output += "=" * 60 + "\n\n"

        for i, fail in enumerate(failed_functions, 1):
            func_name = fail.get('name', 'unknown')
            error_message = fail.get('message', 'Unknown error')
            attempted_params = fail.get('parameters', {})

            error_output += f"FUNCTION #{i}: {func_name}\n"
            error_output += f"{'─' * 50}\n"
            error_output += f"ATTEMPTED PARAMETERS:\n"
            for key, value in attempted_params.items():
                error_output += f"  - {key} = {value}\n"
            error_output += f"\nERROR MESSAGE:\n  {error_message}\n\n"

            # Intelligent error analysis
            error_output += "ERROR ANALYSIS & FIX:\n"

            if "missing" in error_message.lower() and "required" in error_message.lower():
                # Missing required parameter
                missing_param = _extract_missing_parameter(error_message)
                if missing_param:
                    error_output += f"  ️  MISSING REQUIRED PARAMETER: '{missing_param}'\n"
                    error_output += f"   FIX: Add '{missing_param}' parameter with appropriate value\n"

                    # Check if they used wrong parameter name
                    similar_params = [p for p in attempted_params.keys()
                                     if missing_param in p or p in missing_param]
                    if similar_params:
                        error_output += f"   NOTE: You used '{similar_params[0]}' but function needs '{missing_param}'\n"
                        error_output += f"     Example: Change {similar_params[0]}=value to {missing_param}=value\n"

                    # Provide correct function signature hint
                    error_output += f"\n   CHECK FUNCTION SIGNATURE:\n"
                    error_output += f"     Review the function definition to see required parameters\n"
                    error_output += f"     Required parameters MUST be provided\n"

            elif "unexpected keyword argument" in error_message.lower():
                # Wrong parameter name
                wrong_param = _extract_unexpected_parameter(error_message)
                if wrong_param:
                    error_output += f"  ️  INVALID PARAMETER: '{wrong_param}'\n"
                    error_output += f"   FIX: This parameter doesn't exist in the function\n"
                    error_output += f"     Check the function signature for correct parameter names\n"
                    error_output += f"     Remove '{wrong_param}' or rename to correct parameter\n"

            elif "invalid" in error_message.lower() or "type" in error_message.lower():
                # Type/value error
                error_output += f"  ️  INVALID PARAMETER VALUE\n"
                error_output += f"   FIX: Check parameter types and value formats\n"
                error_output += f"     Ensure values match expected types (string, number, etc.)\n"

            else:
                # General error
                error_output += f"  ️  FUNCTION EXECUTION FAILED\n"
                error_output += f"   FIX: Review error message and adjust parameters accordingly\n"

            error_output += f"\n{'─' * 50}\n\n"

        error_output += " NEXT STEPS FOR FUNCTIONS:\n"
        error_output += "  1. ANALYZE the error messages above\n"
        error_output += "  2. IDENTIFY incorrect or missing parameters\n"
        error_output += "  3. CORRECT parameter names and values\n"
        error_output += "  4. If in ADAPTIVE mode: RETRY with corrected parameters\n"
        error_output += "  5. If in STRUCTURED mode: Use corrected understanding for extraction\n\n"

    # Failed RAG queries
    if failed_rag:
        error_output += " FAILED RAG QUERIES:\n"
        error_output += "=" * 60 + "\n\n"

        for i, fail in enumerate(failed_rag, 1):
            query = fail.get('query', 'unknown')
            error_message = fail.get('message', 'Unknown error')

            error_output += f"QUERY #{i}: \"{query}\"\n"
            error_output += f"ERROR: {error_message}\n\n"

        error_output += " NEXT STEPS FOR RAG:\n"
        error_output += "  1. If RAG not configured: Continue without RAG evidence\n"
        error_output += "  2. If query failed: Try different query or proceed without\n"
        error_output += "  3. ADAPTIVE mode: Can request different RAG query\n\n"

    # Failed extras
    if failed_extras:
        error_output += " FAILED EXTRAS:\n"
        error_output += "=" * 60 + "\n\n"

        for i, fail in enumerate(failed_extras, 1):
            keywords = fail.get('keywords', [])
            error_message = fail.get('message', 'Unknown error')

            error_output += f"KEYWORDS #{i}: {', '.join(keywords)}\n"
            error_output += f"ERROR: {error_message}\n\n"

        error_output += " NEXT STEPS FOR EXTRAS:\n"
        error_output += "  1. Continue extraction without these hints\n"
        error_output += "  2. ADAPTIVE mode: Can try different keywords\n\n"

    if error_output:
        error_output += "━" * 60 + "\n"
        error_output += " IMPORTANT: Learn from these errors and DO NOT repeat the same mistakes!\n"
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
        except (KeyError, IndexError) as e:
            # If there are other placeholders we don't know about, leave them
            # IndexError occurs when template has {0}, {1} but only keyword args are provided
            import re
            user_prompt_filled = user_task_prompt
            # Fill known placeholders using regex to avoid KeyError/IndexError
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

1. **query_rag(query, purpose)** - Retrieve clinical guidelines with SEARCH STRATEGY
   - Use TERM VARIATIONS for better recall:
     • Include: synonyms, abbreviations, related terms, alternative phrasings
     • Example: "malnutrition" → add "undernutrition", "PEM", "wasting", "nutritional deficiency"
   - Build multi-term queries: "ASPEN pediatric malnutrition undernutrition criteria assessment"
   - Call MULTIPLE times with DIFFERENT terminology angles
   - Example: query_rag("ASPEN pediatric malnutrition undernutrition PEM criteria", "classification with varied terms")

2. **call_[function_name](parameters)**
   - Perform medical calculations: z-scores, BMI, percentiles, growth calculations, lab interpretations
   - Call same function multiple times for serial measurements at different time points
   - Available functions are dynamically listed based on your registry
   - Example: call_percentile_to_zscore({{"percentile": 3}})

3. **query_extras(keywords)** - Get hints with TERM EXPANSION
   - Expand each concept with variations:
     • Core + synonyms + abbreviations + related terms
     • Add qualifiers: age group, specialty, system
   - Example: ["malnutrition", "undernutrition", "PEM", "pediatric malnutrition", "nutritional assessment"]
   - Use VARIED terminology for better recall

**AUTONOMOUS TASK-DRIVEN EXECUTION WORKFLOW:**

**PHASE 1 - UNDERSTAND REQUIREMENTS & ANALYZE DATA:**
1. Read the task description above → Understand WHAT needs to be extracted and HOW
2. Read the clinical text → Identify WHAT data is currently available
3. Perform gap analysis between available data and required output:
   - Available format vs. required format (e.g., percentile available, z-score required)
   - Guidelines mentioned in task vs. knowledge needed (e.g., "ASPEN criteria" mentioned)
   - Calculations needed to complete schema fields

**PHASE 2 - EXECUTE TOOLS WITH SEARCH STRATEGY:**
4. Based on gap analysis, determine required tools and USE SEARCH STRATEGY:
   - Functions: Call for calculations/conversions as usual
   - RAG: Use MULTI-TERM queries with variations
     • Build: "primary keywords + synonyms + abbreviations + related terms"
     • Example: "ASPEN malnutrition undernutrition PEM criteria pediatric assessment"
     • Leniency: Include broader/narrower terms, alternative phrasings
   - Extras: Expand keywords with variations
     • Core concept + medical synonyms + abbreviations + qualifiers
     • Example: ["malnutrition", "undernutrition", "PEM", "wasting", "nutritional deficiency"]
5. Execute all tool calls (tools run in parallel for performance)

**PHASE 3 - ASSESS & REFINE (ITERATIVE):**
6. Review tool results and assess current extraction state
7. Determine if additional tools would improve extraction:
   - Would clarify ambiguous findings?
   - Would fix inconsistencies or errors?
   - Would ascertain missing but important details?
   - Would improve completeness or quality?
8. If yes: Call additional tools with DIFFERENT term variations
   - Use NEW terminology angles (synonyms, related concepts not yet tried)
   - Expand search with broader/narrower terms
   - Return to Phase 3
9. If no: Proceed to Phase 4

**PHASE 4 - COMPLETE EXTRACTION:**
10. Use all tool results to fill schema fields
11. Extract remaining fields directly from clinical text
12. Output final JSON matching the exact schema structure

 CRITICAL: You autonomously determine which tools are REQUIRED to fulfill the task.
Tools must serve the TASK requirements, not unrelated exploration.

**CRITICAL PRINCIPLES:**

- **Understand the Task**: Read task description to understand WHAT to extract, not just schema structure
- **Autonomous Gap Analysis**: Analyze available data vs. required output, determine tools needed to close gaps
- **Tools Serve the Task**: Call tools that are REQUIRED to fulfill task, not for unrelated exploration
- **Iterative Refinement**: After initial tools, assess if additional tools would clarify/fix/improve extraction
- **Multiple Tool Calls**: Call same function multiple times for serial measurements or different parameters
- **Parallel Execution**: Tools execute in parallel (async) for performance - request multiple at once when possible
- **Support Ground Truth**: Ensure extraction supports the ground truth diagnosis with evidence
- **Complete Extraction**: Output JSON with all required schema fields filled

**EXAMPLE AUTONOMOUS TASK-DRIVEN EXECUTION:**

```
[PHASE 1 - Understand Requirements & Analyze Data]
"Reading task: Extract malnutrition assessment with growth anthropometrics per ASPEN criteria"
"Schema requires: growth_and_anthropometrics field with z-scores and ASPEN classification"
"Clinical text shows: 3-year-old, weight 12.5kg (10th percentile), height 92cm (25th percentile)"

"Gap analysis:"
"- Available: percentiles (10th, 25th)"
"- Required: z-scores"
"- Gap: Need to convert percentile → z-score"
"- Task mentions 'ASPEN criteria' but I need the actual criteria to apply"
"- Gap: Need to retrieve ASPEN classification guidelines"

[PHASE 2 - Autonomously Determine & Execute Tools]
"Based on gap analysis, I need these tools:"
→ Call: call_percentile_to_zscore({{"percentile": 10}})  # Convert weight percentile
→ Call: call_percentile_to_zscore({{"percentile": 25}})  # Convert height percentile
→ Call: query_rag("ASPEN pediatric malnutrition classification criteria z-scores", "Need ASPEN criteria to classify")

[Tools return: 10th %ile = -1.28 z-score, 25th %ile = -0.67 z-score, ASPEN criteria retrieved]

[PHASE 3 - Assess & Refine]
"Reviewing results:"
"- Have z-scores: weight -1.28, height -0.67"
"- Have ASPEN criteria from RAG"
"- Can now classify per ASPEN (mild risk based on z-scores)"

"Do I need additional tools to improve extraction?"
"- Text mentions 'poor intake' but no quantification → Could query RAG for intake assessment criteria"
"- Text has exam findings that support malnutrition → Sufficient for classification"
"Decision: Have sufficient information to complete task"

[PHASE 4 - Complete Extraction]
"Mapping results to schema:"
"- growth_and_anthropometrics: Weight z-score -1.28 (mild risk per ASPEN), Height z-score -0.67 (normal)"
"- diagnosis: Mild malnutrition risk per ASPEN anthropometric criteria"
"- All required fields complete"

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
