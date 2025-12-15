"""
Malnutrition Classification SDK - Complete Pipeline with Batch Processing
===========================================================================

Complete ready-to-run example showing:
- STRUCTURED pipeline (4-stage predefined workflow)
- ADAPTIVE pipeline (autonomous dynamic workflow)
- BATCH PROCESSING with multi-GPU support (H100)
- Malnutrition-specific functions, patterns, and extras
- Binary classification: "Malnutrition" or "No-Malnutrition"
- Based on ASPEN pediatric criteria and WHO z-score classification

Usage:
    # Single note
    python malnutrition_classification_sdk.py --mode structured --note positive

    # Batch processing
    python malnutrition_classification_sdk.py --mode structured --batch

    # Multi-GPU batch (2 H100s)
    python malnutrition_classification_sdk.py --mode structured --batch --gpus 2
"""

import os
import sys
import yaml
import argparse
import torch
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.llm_manager import LLMManager
from core.agent_system import AgentSystem
from core.agentic_agent import AgenticAgent
from core.function_registry import FunctionRegistry
from core.regex_preprocessor import RegexPreprocessor
from core.extras_manager import ExtrasManager


# ============================================================================
# MALNUTRITION FUNCTIONS (YAML Format)
# ============================================================================

MALNUTRITION_FUNCTIONS_YAML = """
- name: calculate_zscore
  description: "Calculate WHO/CDC z-score for weight, height, or BMI"
  enabled: true
  code: |
    def calculate_zscore(observed_value, mean, std_dev):
        '''Calculate z-score from observed value, population mean, and standard deviation'''
        if not all([observed_value, mean, std_dev]) or std_dev == 0:
            return None

        z_score = (observed_value - mean) / std_dev
        return round(z_score, 2)

  parameters:
    observed_value:
      type: number
      description: "Observed measurement (weight, height, or BMI)"
      required: true
    mean:
      type: number
      description: "Population mean for age/sex"
      required: true
    std_dev:
      type: number
      description: "Population standard deviation"
      required: true

  returns: "Z-score value (rounded to 2 decimals)"

- name: interpret_zscore_malnutrition
  description: "Interpret z-score according to WHO malnutrition classification"
  enabled: true
  code: |
    def interpret_zscore_malnutrition(z_score, measurement_type="weight_for_age"):
        '''Interpret z-score for malnutrition severity (WHO/ASPEN criteria)'''
        if z_score is None:
            return {"category": "Unknown", "malnutrition_status": "UNKNOWN"}

        # WHO Classification for weight-for-age, height-for-age, BMI-for-age
        if z_score >= -1:
            return {
                "category": "Normal",
                "severity": "None",
                "malnutrition_status": "NO-MALNUTRITION",
                "z_score": z_score
            }
        elif -2 <= z_score < -1:
            return {
                "category": "At Risk",
                "severity": "Mild",
                "malnutrition_status": "RISK",
                "z_score": z_score
            }
        elif -3 <= z_score < -2:
            return {
                "category": "Moderate Malnutrition",
                "severity": "Moderate",
                "malnutrition_status": "MALNUTRITION",
                "z_score": z_score
            }
        else:  # z_score < -3
            return {
                "category": "Severe Malnutrition",
                "severity": "Severe",
                "malnutrition_status": "MALNUTRITION",
                "z_score": z_score
            }

  parameters:
    z_score:
      type: number
      description: "Z-score value"
      required: true
    measurement_type:
      type: string
      description: "Type of measurement (weight_for_age, height_for_age, bmi_for_age)"
      required: false

  returns: "Dictionary with category, severity, and malnutrition status"

- name: calculate_pediatric_nutrition_status
  description: "Assess ASPEN pediatric malnutrition criteria (requires ≥2 indicators)"
  enabled: true
  code: |
    def calculate_pediatric_nutrition_status(weight_zscore=None, height_zscore=None,
                                            bmi_zscore=None, insufficient_intake=False,
                                            weight_loss=False, growth_deceleration=False,
                                            muscle_fat_loss=False, diminished_function=False):
        '''
        Assess ASPEN pediatric malnutrition criteria
        Requires ≥2 of the following indicators:
        1. Insufficient energy/protein intake
        2. Weight loss or inadequate weight gain
        3. Growth deceleration (length/height)
        4. Muscle/fat mass loss
        5. Diminished functional status

        Z-score criteria:
        - Moderate: -2 to -3 SD
        - Severe: < -3 SD
        '''
        indicators_met = 0
        indicators_list = []

        # Check clinical indicators
        if insufficient_intake:
            indicators_met += 1
            indicators_list.append("Insufficient energy/protein intake")

        if weight_loss or (weight_zscore and weight_zscore < -1):
            indicators_met += 1
            indicators_list.append("Weight loss or inadequate gain")

        if growth_deceleration or (height_zscore and height_zscore < -2):
            indicators_met += 1
            indicators_list.append("Growth deceleration")

        if muscle_fat_loss:
            indicators_met += 1
            indicators_list.append("Muscle/fat mass loss")

        if diminished_function:
            indicators_met += 1
            indicators_list.append("Diminished functional status")

        # Determine severity based on z-scores
        severity = "None"
        worst_zscore = None

        z_scores = [z for z in [weight_zscore, height_zscore, bmi_zscore] if z is not None]
        if z_scores:
            worst_zscore = min(z_scores)
            if worst_zscore < -3:
                severity = "Severe"
            elif worst_zscore < -2:
                severity = "Moderate"
            elif worst_zscore < -1:
                severity = "Mild"

        # ASPEN diagnosis requires ≥2 indicators
        if indicators_met >= 2:
            malnutrition_status = "MALNUTRITION"
        else:
            malnutrition_status = "NO-MALNUTRITION"

        return {
            "malnutrition_status": malnutrition_status,
            "severity": severity,
            "indicators_met": indicators_met,
            "indicators": indicators_list,
            "worst_zscore": worst_zscore,
            "aspen_criteria_met": indicators_met >= 2
        }

  parameters:
    weight_zscore:
      type: number
      description: "Weight-for-age z-score"
      required: false
    height_zscore:
      type: number
      description: "Height-for-age z-score"
      required: false
    bmi_zscore:
      type: number
      description: "BMI-for-age z-score"
      required: false
    insufficient_intake:
      type: boolean
      description: "Documented insufficient energy/protein intake"
      required: false
    weight_loss:
      type: boolean
      description: "Weight loss or inadequate weight gain"
      required: false
    growth_deceleration:
      type: boolean
      description: "Growth deceleration documented"
      required: false
    muscle_fat_loss:
      type: boolean
      description: "Loss of muscle or fat mass"
      required: false
    diminished_function:
      type: boolean
      description: "Diminished functional status"
      required: false

  returns: "Comprehensive malnutrition assessment with ASPEN criteria"

- name: calculate_bmi
  description: "Calculate Body Mass Index"
  enabled: true
  code: |
    def calculate_bmi(weight_kg, height_m):
        '''Calculate BMI from weight and height'''
        if not weight_kg or not height_m or height_m <= 0:
            return None

        bmi = weight_kg / (height_m ** 2)
        return round(bmi, 2)

  parameters:
    weight_kg:
      type: number
      description: "Weight in kilograms"
      required: true
    height_m:
      type: number
      description: "Height in meters"
      required: true

  returns: "BMI value (kg/m²)"

- name: calculate_growth_velocity
  description: "Calculate growth velocity (weight or height gain per unit time)"
  enabled: true
  code: |
    def calculate_growth_velocity(current_value, previous_value, time_interval_months):
        '''
        Calculate growth velocity
        Returns change per month
        '''
        if not all([current_value, previous_value, time_interval_months]) or time_interval_months <= 0:
            return None

        change = current_value - previous_value
        velocity = change / time_interval_months

        return {
            "velocity_per_month": round(velocity, 2),
            "total_change": round(change, 2),
            "time_interval_months": time_interval_months,
            "adequate": velocity > 0  # Positive growth expected in children
        }

  parameters:
    current_value:
      type: number
      description: "Current measurement (weight or height)"
      required: true
    previous_value:
      type: number
      description: "Previous measurement"
      required: true
    time_interval_months:
      type: number
      description: "Time between measurements in months"
      required: true

  returns: "Growth velocity analysis"
"""


# ============================================================================
# MALNUTRITION PATTERNS (YAML Format)
# ============================================================================

MALNUTRITION_PATTERNS_YAML = """
- name: normalize_zscore
  pattern: '\\b[zZ][-\\s]?score[s]?[:\\s]+(-?\\d+\\.?\\d*)\\b'
  replacement: 'Z-score: \\1'
  description: "Standardize z-score notation"
  enabled: true

- name: normalize_weight
  pattern: '\\b(weight|wt)[:\\s]+(\\d+\\.?\\d*)\\s*(kg|kilograms?)\\b'
  replacement: 'Weight: \\2 kg'
  description: "Standardize weight notation"
  enabled: true

- name: normalize_height
  pattern: '\\b(height|ht|length)[:\\s]+(\\d+\\.?\\d*)\\s*(cm|centimeters?)\\b'
  replacement: 'Height: \\2 cm'
  description: "Standardize height notation"
  enabled: true

- name: normalize_bmi
  pattern: '\\bBMI[:\\s]+(\\d+\\.?\\d*)\\b'
  replacement: 'BMI: \\1'
  description: "Standardize BMI notation"
  enabled: true

- name: normalize_malnutrition_severity
  pattern: '\\b(severe|moderate|mild)\\s+malnutrition\\b'
  replacement: '\\1 malnutrition'
  description: "Standardize malnutrition severity terms"
  enabled: true

- name: normalize_intake
  pattern: '\\b(poor|decreased|inadequate|insufficient)\\s+(oral\\s+)?(intake|feeding|nutrition)\\b'
  replacement: 'insufficient intake'
  description: "Normalize intake descriptions"
  enabled: true

- name: normalize_growth
  pattern: '\\b(poor|inadequate|delayed|slow)\\s+growth\\b'
  replacement: 'growth deceleration'
  description: "Standardize growth terminology"
  enabled: true

- name: normalize_weight_loss
  pattern: '\\b(weight\\s+loss|losing\\s+weight|wt\\s+loss)\\b'
  replacement: 'weight loss'
  description: "Standardize weight loss terminology"
  enabled: true
"""


# ============================================================================
# MALNUTRITION EXTRAS (YAML Format)
# ============================================================================

MALNUTRITION_EXTRAS_YAML = """
- id: aspen_pediatric_criteria
  name: "ASPEN Pediatric Malnutrition Criteria"
  type: criteria
  content: |
    ASPEN Pediatric Malnutrition Diagnostic Criteria

    Diagnosis requires ≥2 of the following indicators:

    1. Insufficient energy intake
       - Documented inadequate energy or protein intake
       - Persistent poor oral intake

    2. Weight loss or inadequate weight gain
       - Weight-for-age z-score decline
       - Inadequate weight gain for age
       - Weight loss documented

    3. Growth deceleration (linear growth)
       - Height-for-age z-score < -2 SD
       - Documented growth deceleration
       - Length/height gain inadequate for age

    4. Loss of muscle and/or subcutaneous fat mass
       - Clinical assessment of muscle wasting
       - Loss of subcutaneous fat
       - Mid-upper arm circumference decline

    5. Diminished functional status
       - Decreased activity level
       - Developmental delay or regression
       - Reduced endurance

    Severity Classification:
    - Moderate malnutrition: -2 to -3 SD z-score
    - Severe malnutrition: < -3 SD z-score

    Reference: Mehta et al. (2013) ASPEN Pediatric Malnutrition Criteria

  keywords:
    - malnutrition
    - ASPEN
    - pediatric
    - criteria
    - diagnosis

  metadata:
    category: malnutrition
    priority: high
    source: "ASPEN 2013"

- id: who_zscore_classification
  name: "WHO Z-Score Classification for Malnutrition"
  type: reference
  content: |
    WHO Z-Score Classification Standards

    Weight-for-Age, Height-for-Age, BMI-for-Age:

    Z-score ≥ -1 SD:
    - Category: Normal/Adequate
    - Status: No malnutrition

    Z-score -1 to -2 SD:
    - Category: At Risk
    - Status: Mild malnutrition risk
    - Action: Monitor closely, nutrition counseling

    Z-score -2 to -3 SD:
    - Category: Moderate Malnutrition
    - Status: Moderate acute malnutrition (MAM)
    - Action: Therapeutic feeding, medical evaluation

    Z-score < -3 SD:
    - Category: Severe Malnutrition
    - Status: Severe acute malnutrition (SAM)
    - Action: Immediate medical intervention

    Special Indicators:
    - Weight-for-height < -3 SD: Severe wasting (SAM)
    - Height-for-age < -2 SD: Stunting (chronic malnutrition)
    - Weight-for-age < -2 SD: Underweight

    Reference: WHO Child Growth Standards (2006)

  keywords:
    - WHO
    - z-score
    - malnutrition
    - classification
    - growth

  metadata:
    category: malnutrition
    priority: high
    source: "WHO 2006"

- id: malnutrition_indicators
  name: "Clinical Indicators of Pediatric Malnutrition"
  type: knowledge
  content: |
    Clinical Indicators of Pediatric Malnutrition

    Anthropometric Indicators:
    - Weight-for-age z-score < -2
    - Height-for-age z-score < -2
    - BMI-for-age z-score < -2
    - Mid-upper arm circumference (MUAC) < 11.5 cm (6-59 months)
    - Triceps skinfold thickness decline

    Dietary Indicators:
    - Inadequate caloric intake (< 75% estimated needs)
    - Protein deficiency
    - Poor appetite or feeding refusal
    - Prolonged NPO status
    - Restrictive diet without supplementation

    Growth Indicators:
    - No weight gain over 1-3 months
    - Growth velocity < 5th percentile
    - Crossing downward ≥2 major percentile lines
    - Plateauing growth curve

    Physical Examination Findings:
    - Temporal wasting
    - Rib prominence
    - Decreased muscle bulk
    - Loss of subcutaneous fat
    - Edema (kwashiorkor)
    - Skin/hair changes

    Functional Indicators:
    - Developmental delay
    - Decreased activity level
    - Frequent infections
    - Poor wound healing
    - Decreased endurance

  keywords:
    - malnutrition
    - indicators
    - clinical
    - assessment
    - pediatric

  metadata:
    category: malnutrition
    priority: medium
"""


# ============================================================================
# SAMPLE CLINICAL NOTES
# ============================================================================

SAMPLE_NOTES = {
    "positive": """
PEDIATRIC NUTRITION CONSULTATION NOTE

Patient: 4-year-old male
Chief Complaint: Poor weight gain, feeding difficulties

ANTHROPOMETRICS:
- Current weight: 12.8 kg (z-score: -2.5)
- Height: 95 cm (z-score: -2.2)
- BMI: 14.2 (z-score: -2.1)
- Weight 3 months ago: 13.2 kg
- Weight loss: 0.4 kg over 3 months

DIETARY HISTORY:
- Reported intake: 60-70% of estimated caloric needs
- Poor appetite, refuses most solid foods
- Primarily drinking milk, limited variety
- Parent reports "very picky eater"

CLINICAL ASSESSMENT:
- Visible rib prominence noted
- Decreased subcutaneous fat in extremities
- Temporal wasting present
- Muscle bulk appears decreased compared to age norms
- Activity level: Parents report child tires easily, less active than peers

GROWTH PATTERN:
- Weight-for-age has crossed 2 major percentile lines downward
- Height velocity adequate but slowing
- Growth deceleration documented over past 6 months

FUNCTIONAL STATUS:
- Developmental milestones: Meeting most, some mild delays in gross motor
- Frequent upper respiratory infections (4 episodes in past 6 months)
- Parents report decreased endurance during play

ASSESSMENT:
Multiple indicators of malnutrition present:
1. Insufficient energy intake (60-70% of needs)
2. Weight loss and inadequate weight gain
3. Growth deceleration with downward percentile crossing
4. Clinical evidence of muscle and fat loss
5. Diminished functional status

PLAN:
- Pediatric malnutrition diagnosis: MODERATE
- Initiate high-calorie supplementation
- Nutrition counseling for parents
- Follow-up in 2 weeks for weight check
- Consider GI evaluation if no improvement
""",

    "negative": """
PEDIATRIC WELL-CHILD VISIT

Patient: 5-year-old female
Chief Complaint: Annual check-up

ANTHROPOMETRICS:
- Current weight: 18.5 kg (z-score: 0.2)
- Height: 110 cm (z-score: 0.5)
- BMI: 15.3 (z-score: -0.1)
- Growth pattern: Consistent along 50th percentile

DIETARY HISTORY:
- Eating well-balanced diet
- Good appetite
- Variety of foods including fruits, vegetables, proteins
- Age-appropriate portion sizes
- No feeding concerns

CLINICAL ASSESSMENT:
- Well-nourished appearance
- Appropriate muscle bulk for age
- Adequate subcutaneous fat
- No abnormal physical findings
- Active and energetic during visit

GROWTH PATTERN:
- Weight-for-age tracking along growth curve consistently
- Height velocity appropriate for age
- BMI stable and within normal range
- No concerning changes in growth pattern

FUNCTIONAL STATUS:
- Meeting all developmental milestones
- Active in preschool activities
- Good endurance during play
- No frequent illnesses reported
- Parents report normal activity level for age

ASSESSMENT:
- Normal growth and development
- Adequate nutrition
- No concerns for malnutrition

PLAN:
- Continue current diet
- Return for next annual visit
- No interventions needed
""",

    "borderline": """
PEDIATRIC NUTRITION ASSESSMENT

Patient: 3-year-old male
Chief Complaint: Parental concern about eating habits and growth

ANTHROPOMETRICS:
- Current weight: 13.2 kg (z-score: -1.5)
- Height: 92 cm (z-score: -0.8)
- BMI: 15.6 (z-score: -1.2)
- Weight 2 months ago: 13.0 kg
- Minimal weight gain noted

DIETARY HISTORY:
- Reported intake: 80-85% of estimated caloric needs
- Selective eater, prefers carbohydrates
- Limited protein and vegetable intake
- Parents describe as "somewhat picky"
- Drinks adequate fluids

CLINICAL ASSESSMENT:
- Appears slightly thin but not wasted
- Muscle bulk appears borderline low
- Subcutaneous fat present but reduced
- No obvious temporal wasting
- Energy level appears adequate during visit

GROWTH PATTERN:
- Weight-for-age tracking between 5th-10th percentile
- Has crossed 1 percentile line downward in past 6 months
- Height velocity currently adequate
- BMI tracking below average but stable

FUNCTIONAL STATUS:
- Meeting all developmental milestones appropriately
- Activity level reported as normal by parents
- No concerning increase in infections
- Participates normally in daycare activities

ASSESSMENT:
Some concerning features present:
1. Suboptimal caloric intake (80-85% of needs)
2. Mild weight gain inadequacy
3. Z-scores in borderline range (-1 to -2)
4. One indicator clearly met, second indicator borderline

Currently does not meet full ASPEN criteria (needs ≥2 indicators)
However, warrants close monitoring given trajectory

PLAN:
- At risk for malnutrition - not currently diagnosed
- Nutrition counseling for parents
- Dietary modifications to increase caloric density
- Close follow-up in 4 weeks
- Recheck anthropometrics
- Will reassess if ≥2 indicators develop
"""
}


# ============================================================================
# STRUCTURED PIPELINE IMPLEMENTATION
# ============================================================================

def run_structured_pipeline(llm, func_registry, preprocessor, extras_mgr, clinical_note):
    """
    Run STRUCTURED pipeline for malnutrition classification

    STRUCTURED = 4-stage predefined workflow:
    1. Text preprocessing
    2. Information extraction
    3. Analysis
    4. Final synthesis
    """
    print("\n" + "="*70)
    print("STRUCTURED PIPELINE - Malnutrition Classification")
    print("="*70)

    # Create agent with all tools
    agent = AgentSystem(
        llm_interface=llm,
        function_registry=func_registry,
        regex_preprocessor=preprocessor,
        extras_manager=extras_mgr
    )

    # Define classification task
    task_prompt = """
    Classify this pediatric clinical note as MALNUTRITION or NO-MALNUTRITION.

    Use ASPEN Pediatric Malnutrition Criteria:
    - Diagnosis requires ≥2 of these indicators:
      1. Insufficient energy/protein intake
      2. Weight loss or inadequate weight gain
      3. Growth deceleration (linear growth)
      4. Loss of muscle/fat mass
      5. Diminished functional status

    Use WHO Z-Score Classification:
    - Z-score < -2: Malnutrition likely
    - Z-score -1 to -2: At risk, monitor
    - Z-score ≥ -1: Normal

    Analyze:
    - Anthropometric measurements and z-scores
    - Dietary intake adequacy
    - Growth pattern and velocity
    - Physical examination findings
    - Functional status

    Provide binary classification with confidence and clear reasoning.
    """

    schema = {
        "classification": "MALNUTRITION or NO-MALNUTRITION (binary)",
        "confidence": "High, Medium, or Low",
        "aspen_indicators_met": "Number of ASPEN criteria met (0-5)",
        "severity": "None, Mild, Moderate, or Severe",
        "key_findings": "List of critical findings supporting classification",
        "reasoning": "Brief clinical reasoning for classification"
    }

    # Run extraction with all tools enabled
    print("\n🔄 Running 4-stage STRUCTURED pipeline...")
    print("   Stage 1: Preprocessing clinical text")
    print("   Stage 2: Extracting structured data")
    print("   Stage 3: Analyzing malnutrition indicators")
    print("   Stage 4: Synthesizing final classification")

    result = agent.extract(
        text=clinical_note,
        task_prompt=task_prompt,
        schema=schema,
        use_functions=True,
        use_extras=True,
        use_patterns=True
    )

    return result


# ============================================================================
# ADAPTIVE PIPELINE IMPLEMENTATION
# ============================================================================

def run_adaptive_pipeline(llm, func_registry, clinical_note):
    """
    Run ADAPTIVE pipeline for malnutrition classification

    ADAPTIVE = Dynamic autonomous workflow:
    - LLM decides which tools to use and when
    - Iterative refinement based on findings
    - No predefined stages
    """
    print("\n" + "="*70)
    print("ADAPTIVE PIPELINE - Malnutrition Classification")
    print("="*70)

    # Create adaptive agent
    agent = AgenticAgent(
        llm_interface=llm,
        function_registry=func_registry,
        max_iterations=10,
        confidence_threshold=0.8
    )

    # Define classification task
    task_prompt = """
    Autonomously classify this pediatric clinical note as MALNUTRITION or NO-MALNUTRITION.

    You have access to malnutrition assessment functions. Use them as needed to:
    - Calculate and interpret z-scores
    - Assess ASPEN pediatric criteria
    - Evaluate growth patterns
    - Determine malnutrition severity

    Adapt your analysis strategy based on available data:
    - If z-scores present, interpret them
    - If growth data available, calculate velocity
    - Count ASPEN indicators (need ≥2 for diagnosis)
    - Consider clinical and functional findings

    Provide binary classification with supporting evidence.
    """

    schema = {
        "classification": "MALNUTRITION or NO-MALNUTRITION",
        "confidence": "High, Medium, or Low",
        "aspen_indicators_met": "Number of ASPEN indicators identified",
        "severity": "None, Mild, Moderate, or Severe if malnutrition present",
        "key_findings": "Critical findings supporting classification",
        "reasoning": "Clinical reasoning and decision pathway"
    }

    # Run adaptive extraction
    print("\n🤖 Running ADAPTIVE autonomous pipeline...")
    print("   LLM will dynamically decide which assessments to perform")
    print("   Iterative refinement based on intermediate findings")

    result = agent.extract(
        text=clinical_note,
        task_prompt=task_prompt,
        schema=schema
    )

    return result


# ============================================================================
# BATCH PROCESSING DATA
# ============================================================================

BATCH_NOTES = [
    {
        "id": 1,
        "patient": "Patient A",
        "note": SAMPLE_NOTES["positive"]
    },
    {
        "id": 2,
        "patient": "Patient B",
        "note": SAMPLE_NOTES["negative"]
    },
    {
        "id": 3,
        "patient": "Patient C",
        "note": SAMPLE_NOTES["borderline"]
    },
    {
        "id": 4,
        "patient": "Patient D",
        "note": """PEDIATRIC NOTE: 3yo M, Weight z-score -2.8, Height z-score -2.5. Poor intake x2mo. Assessment: Moderate malnutrition."""
    },
    {
        "id": 5,
        "patient": "Patient E",
        "note": """WELL-CHILD: 5yo F, All growth parameters normal. Weight z-score 0.3, Height z-score 0.5. Good appetite. Assessment: No malnutrition."""
    }
]


# ============================================================================
# BATCH PROCESSING WITH MULTI-GPU SUPPORT
# ============================================================================

def process_single_note_worker(note_data, config_dict, mode):
    """Worker function for processing single note in parallel"""

    # Initialize managers in worker
    func_registry = FunctionRegistry()
    preprocessor = RegexPreprocessor()
    extras_mgr = ExtrasManager()

    # Load YAML configs
    func_registry.import_functions(MALNUTRITION_FUNCTIONS_YAML)
    preprocessor.import_patterns(MALNUTRITION_PATTERNS_YAML)
    extras_mgr.import_extras(MALNUTRITION_EXTRAS_YAML)

    # Initialize LLM
    llm = LLMManager(config=config_dict)

    # Process based on mode
    if mode == "structured":
        result = run_structured_pipeline(
            llm, func_registry, preprocessor, extras_mgr, note_data["note"]
        )
    else:
        result = run_adaptive_pipeline(llm, func_registry, note_data["note"])

    return {
        "id": note_data["id"],
        "patient": note_data["patient"],
        "result": result
    }


def run_batch_processing(llm_config, mode="structured", num_gpus=1):
    """Run batch processing across multiple notes with GPU parallelization"""

    print("\n" + "="*70)
    print(f"BATCH PROCESSING - {mode.upper()} Pipeline")
    print(f"Processing {len(BATCH_NOTES)} notes with {num_gpus} GPU(s)")
    print("="*70)

    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"\n🎮 GPUs Available: {gpu_count}")
        for i in range(gpu_count):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

        if num_gpus > gpu_count:
            print(f"   ⚠️  Requested {num_gpus} GPUs but only {gpu_count} available")
            num_gpus = gpu_count
    else:
        print("\n⚠️  CUDA not available - running on CPU")
        num_gpus = 1

    # Process in parallel
    results = []

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {}

        for i, note_data in enumerate(BATCH_NOTES):
            # Assign GPU for multi-GPU setup
            if torch.cuda.is_available() and num_gpus > 1:
                gpu_id = i % num_gpus
                os.environ[f'CUDA_VISIBLE_DEVICES_{i}'] = str(gpu_id)

            future = executor.submit(
                process_single_note_worker,
                note_data,
                llm_config,
                mode
            )
            futures[future] = note_data["id"]

        # Collect results
        completed = 0
        for future in as_completed(futures):
            note_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                print(f"\n✓ Completed {completed}/{len(BATCH_NOTES)}: {result['patient']}")
            except Exception as e:
                print(f"\n✗ Failed {note_id}: {e}")
                results.append({
                    "id": note_id,
                    "error": str(e)
                })

    return results


# ============================================================================
# SETUP FUNCTIONS
# ============================================================================

def setup_managers():
    """Initialize and configure all managers with YAML configs"""

    print("\n📋 Setting up managers with YAML configurations...")

    # Initialize managers
    func_registry = FunctionRegistry()
    preprocessor = RegexPreprocessor()
    extras_mgr = ExtrasManager()

    # Import malnutrition functions
    print("   ✓ Loading malnutrition functions...")
    success, count, msg = func_registry.import_functions(MALNUTRITION_FUNCTIONS_YAML)
    if success:
        print(f"     Loaded {count} functions")
    else:
        print(f"     ERROR: {msg}")

    # Import malnutrition patterns
    print("   ✓ Loading malnutrition patterns...")
    success, count, msg = preprocessor.import_patterns(MALNUTRITION_PATTERNS_YAML)
    if success:
        print(f"     Loaded {count} patterns")
    else:
        print(f"     ERROR: {msg}")

    # Import malnutrition extras
    print("   ✓ Loading malnutrition extras...")
    success, count, msg = extras_mgr.import_extras(MALNUTRITION_EXTRAS_YAML)
    if success:
        print(f"     Loaded {count} extras")
    else:
        print(f"     ERROR: {msg}")

    return func_registry, preprocessor, extras_mgr


def setup_llm():
    """Initialize LLM Manager"""

    print("\n🔧 Initializing LLM Manager...")

    # Try to get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("   ⚠️  OPENAI_API_KEY not found")
        print("   ℹ️  Set with: export OPENAI_API_KEY='your-key'")
        api_key = "your-api-key-here"

    config = {
        'provider': 'openai',
        'model_name': 'gpt-4',
        'api_key': api_key,
        'temperature': 0.1,
        'max_tokens': 2048
    }

    llm = LLMManager(config=config)
    print(f"   ✓ LLM configured: OpenAI GPT-4")

    return llm, config


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description="Malnutrition Classification SDK with Batch Processing"
    )
    parser.add_argument(
        "--mode",
        choices=["structured", "adaptive", "both"],
        default="structured",
        help="Pipeline mode"
    )
    parser.add_argument(
        "--note",
        choices=["positive", "negative", "borderline"],
        default="positive",
        help="Sample clinical note (for single processing)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch processing on multiple notes"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for batch processing (default: 1)"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print(" MALNUTRITION CLASSIFICATION - SDK EXAMPLE")
    print("="*70)

    # Setup
    llm, llm_config = setup_llm()

    if args.batch:
        # Batch processing mode
        results = run_batch_processing(llm_config, args.mode, args.gpus)

        print("\n" + "="*70)
        print("BATCH PROCESSING RESULTS:")
        print("="*70)

        for result in results:
            print(f"\n--- {result['patient']} (ID: {result['id']}) ---")
            if 'error' in result:
                print(f"ERROR: {result['error']}")
            else:
                print(yaml.dump(result['result'], default_flow_style=False))

    else:
        # Single note processing
        func_registry, preprocessor, extras_mgr = setup_managers()
        clinical_note = SAMPLE_NOTES[args.note]

        print(f"\n📝 Clinical Note: {args.note.upper()}")
        print(f"🔄 Pipeline Mode: {args.mode.upper()}")

        print("\n" + "-"*70)
        print("CLINICAL NOTE PREVIEW:")
        print("-"*70)
        preview = clinical_note[:500] + "..." if len(clinical_note) > 500 else clinical_note
        print(preview)

        # Run pipeline(s)
        if args.mode in ["structured", "both"]:
            result = run_structured_pipeline(
                llm, func_registry, preprocessor, extras_mgr, clinical_note
            )

            print("\n" + "="*70)
            print("STRUCTURED PIPELINE RESULTS:")
            print("="*70)
            print(f"\n{yaml.dump(result, default_flow_style=False)}")

        if args.mode in ["adaptive", "both"]:
            result = run_adaptive_pipeline(llm, func_registry, clinical_note)

            print("\n" + "="*70)
            print("ADAPTIVE PIPELINE RESULTS:")
            print("="*70)
            print(f"\n{yaml.dump(result, default_flow_style=False)}")

    print("\n" + "="*70)
    print("✅ Classification Complete")
    print("="*70)


if __name__ == "__main__":
    main()
