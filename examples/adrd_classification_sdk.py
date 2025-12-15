#!/usr/bin/env python3
"""
ADRD Classification SDK - Complete Pipeline with Batch Processing
===================================================================

Complete ready-to-run example showing:
- STRUCTURED pipeline (4-stage predefined workflow)
- ADAPTIVE pipeline (autonomous dynamic workflow)
- BATCH PROCESSING with multi-GPU support (H100)
- ADRD-specific functions, patterns, and extras
- Binary classification: "ADRD" or "NO-ADRD"

Usage:
    # Single note
    python adrd_classification_sdk.py --mode structured --note positive

    # Batch processing
    python adrd_classification_sdk.py --mode structured --batch

    # Multi-GPU batch (2 H100s)
    python adrd_classification_sdk.py --mode structured --batch --gpus 2
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
# ADRD FUNCTIONS (YAML Format)
# ============================================================================

ADRD_FUNCTIONS_YAML = """
- name: calculate_cdr_severity
  description: "Interpret Clinical Dementia Rating (CDR) global score"
  enabled: true
  code: |
    def calculate_cdr_severity(cdr_global_score):
        '''Interpret CDR global score'''
        valid_scores = [0, 0.5, 1, 2, 3]

        if cdr_global_score not in valid_scores:
            return {"category": "Invalid", "adrd_status": "UNKNOWN"}

        interpretations = {
            0: {
                "category": "No Cognitive Impairment",
                "adrd_status": "NO-ADRD"
            },
            0.5: {
                "category": "Questionable/MCI",
                "adrd_status": "POSSIBLE-ADRD"
            },
            1: {
                "category": "Mild Dementia",
                "adrd_status": "ADRD"
            },
            2: {
                "category": "Moderate Dementia",
                "adrd_status": "ADRD"
            },
            3: {
                "category": "Severe Dementia",
                "adrd_status": "ADRD"
            }
        }

        return interpretations[cdr_global_score]

  parameters:
    cdr_global_score:
      type: number
      description: "CDR global score (0, 0.5, 1, 2, or 3)"
      required: true

  returns: "Dictionary with category and ADRD status"

- name: calculate_mmse_severity
  description: "Interpret MMSE score for cognitive impairment"
  enabled: true
  code: |
    def calculate_mmse_severity(mmse_score):
        '''Interpret MMSE score (0-30)'''
        if mmse_score < 0 or mmse_score > 30:
            return {"category": "Invalid", "adrd_status": "UNKNOWN"}

        if mmse_score >= 24:
            category = "Normal Cognition"
            adrd_status = "NO-ADRD"
        elif mmse_score >= 19:
            category = "Mild Impairment"
            adrd_status = "POSSIBLE-ADRD"
        elif mmse_score >= 10:
            category = "Moderate Impairment"
            adrd_status = "ADRD"
        else:
            category = "Severe Impairment"
            adrd_status = "ADRD"

        return {
            "category": category,
            "adrd_status": adrd_status,
            "score": mmse_score
        }

  parameters:
    mmse_score:
      type: number
      description: "MMSE score (0-30)"
      required: true

  returns: "Dictionary with category, ADRD status, and score"

- name: assess_functional_independence
  description: "Assess functional independence based on ADL/IADL status"
  enabled: true
  code: |
    def assess_functional_independence(adl_impaired, iadl_impaired):
        '''
        Assess functional independence
        ADLs: Basic activities (bathing, dressing, eating)
        IADLs: Instrumental activities (cooking, finances, medications)
        '''
        if not adl_impaired and not iadl_impaired:
            return {
                "status": "Fully Independent",
                "functional_decline": False,
                "supports_adrd": False
            }
        elif not adl_impaired and iadl_impaired:
            return {
                "status": "IADL Impairment Only",
                "functional_decline": True,
                "supports_adrd": True,
                "note": "IADL impairment common in MCI/early dementia"
            }
        elif adl_impaired:
            return {
                "status": "ADL Impairment Present",
                "functional_decline": True,
                "supports_adrd": True,
                "note": "ADL impairment indicates moderate-severe impairment"
            }

        return {"status": "Unknown", "functional_decline": False, "supports_adrd": False}

  parameters:
    adl_impaired:
      type: boolean
      description: "Whether ADLs are impaired"
      required: true
    iadl_impaired:
      type: boolean
      description: "Whether IADLs are impaired"
      required: true

  returns: "Dictionary with functional status and ADRD support"
"""


# ============================================================================
# ADRD PATTERNS (YAML Format)
# ============================================================================

ADRD_PATTERNS_YAML = """
- name: normalize_mmse
  pattern: '\\bMMSE[:\\s]+(\\d+)/30\\b'
  replacement: 'MMSE: \\1'
  description: "Normalize MMSE score notation"
  enabled: true

- name: normalize_cdr
  pattern: '\\bCDR[:\\s]+(\\d+\\.?\\d*)\\b'
  replacement: 'CDR: \\1'
  description: "Normalize CDR notation"
  enabled: true

- name: normalize_dementia_terms
  pattern: '\\b(Alzheimer\\'s disease|AD|dementia|cognitive impairment)\\b'
  replacement: 'dementia'
  description: "Standardize dementia terminology"
  enabled: true

- name: normalize_adl
  pattern: '\\b(ADLs?|activities of daily living)\\b'
  replacement: 'ADL'
  description: "Standardize ADL terminology"
  enabled: true

- name: normalize_iadl
  pattern: '\\b(IADLs?|instrumental activities)\\b'
  replacement: 'IADL'
  description: "Standardize IADL terminology"
  enabled: true
"""


# ============================================================================
# ADRD EXTRAS (YAML Format)
# ============================================================================

ADRD_EXTRAS_YAML = """
- id: adrd_diagnostic_criteria
  name: "ADRD Diagnostic Criteria Summary"
  type: criteria
  content: |
    ADRD Diagnostic Criteria:

    1. Cognitive Assessment:
       - MMSE < 24: Cognitive impairment likely
       - CDR ≥ 1: Dementia diagnosis
       - CDR 0.5: MCI/questionable dementia

    2. Functional Impairment:
       - IADL impairment: Early-stage dementia/MCI
       - ADL impairment: Moderate-severe dementia

    3. Progressive Decline:
       - Documented worsening over time
       - Decline in multiple cognitive domains

    4. Rule Out Other Causes:
       - Delirium excluded
       - Depression screened
       - Reversible causes investigated

  keywords:
    - ADRD
    - diagnosis
    - criteria
    - dementia

  metadata:
    category: adrd
    priority: high

- id: nia_aa_criteria
  name: "NIA-AA Diagnostic Criteria for Alzheimer's Disease"
  type: criteria
  content: |
    NIA-AA Criteria for Probable Alzheimer's Disease:

    Core Clinical Criteria (all required):
    1. Dementia diagnosis by clinical examination
    2. Insidious onset (gradual, not sudden)
    3. Clear history of worsening cognition
    4. Initial and prominent cognitive deficits in one of:
       - Amnestic presentation (memory impairment)
       - Non-amnestic presentations (language, visuospatial, executive)

    Supportive Features:
    - Family history of AD
    - Biomarker evidence (amyloid, tau)
    - Absence of other neurological/medical causes

    Probable AD-Dementia vs MCI:
    - Dementia: Interferes with daily function
    - MCI: Minimal functional impact

  keywords:
    - NIA-AA
    - Alzheimer
    - criteria
    - diagnosis

  metadata:
    category: adrd
    priority: high
    source: "NIA-AA 2011"

- id: cognitive_assessment_reference
  name: "Cognitive Assessment Tools Reference"
  type: reference
  content: |
    Common Cognitive Assessment Tools:

    MMSE (Mini-Mental State Examination):
    - Score: 0-30
    - 24-30: Normal
    - 19-23: Mild impairment
    - 10-18: Moderate impairment
    - 0-9: Severe impairment

    CDR (Clinical Dementia Rating):
    - 0: No impairment
    - 0.5: Questionable/very mild (MCI)
    - 1: Mild dementia
    - 2: Moderate dementia
    - 3: Severe dementia

    MoCA (Montreal Cognitive Assessment):
    - Score: 0-30
    - ≥26: Normal
    - <26: Cognitive impairment
    - More sensitive than MMSE for MCI

  keywords:
    - MMSE
    - CDR
    - MoCA
    - assessment
    - cognitive

  metadata:
    category: adrd
    priority: medium
"""


# ============================================================================
# SAMPLE CLINICAL NOTES
# ============================================================================

SAMPLE_NOTES = {
    "positive": """
NEUROLOGY CONSULTATION NOTE

Patient: 78-year-old female
Chief Complaint: Progressive memory decline

COGNITIVE ASSESSMENT:
- MMSE: 18/30 (moderate impairment)
  - Orientation: 3/10
  - Recall: 0/3
  - Language: Adequate
  - Visuospatial: Impaired
- CDR: 2.0 (moderate dementia)
  - Memory: Moderate impairment
  - Orientation: Moderate impairment
  - Judgment: Moderate impairment

FUNCTIONAL STATUS:
- ADLs: Requires assistance with bathing and dressing
- IADLs: Unable to manage finances, medications, or cooking independently
- Daughter reports significant decline over past 2 years

HISTORY:
- Onset: Gradual, insidious over 3-4 years
- Progression: Steady worsening
- Initial symptoms: Short-term memory loss, word-finding difficulties
- Current: Disorientation, wandering, safety concerns

PHYSICAL EXAM:
- No focal neurological deficits
- Gait: Slightly unsteady but mobile
- No parkinsonian features

WORKUP:
- MRI brain: Cortical atrophy, hippocampal volume loss
- Labs: B12, TSH, RPR normal
- No evidence of stroke or mass

ASSESSMENT:
- Probable Alzheimer's Disease Dementia
- Meets NIA-AA criteria for probable AD
- CDR 2.0 indicates moderate stage
- Functional impairment in both ADLs and IADLs
- Progressive cognitive decline documented

PLAN:
- Start cholinesterase inhibitor
- Safety evaluation at home
- Caregiver support resources
- Follow-up in 3 months
""",

    "negative": """
ANNUAL WELLNESS VISIT

Patient: 72-year-old male
Chief Complaint: Annual check-up

COGNITIVE SCREENING:
- MMSE: 29/30 (normal)
  - Orientation: 10/10
  - Recall: 3/3
  - Language: Intact
  - Calculation: Intact
- No subjective memory complaints
- Patient reports: "Memory is fine, no issues"

FUNCTIONAL STATUS:
- ADLs: Fully independent
- IADLs: Fully independent
  - Managing finances independently
  - Driving without issues
  - Cooking, shopping, medications all independent
- Active lifestyle, volunteers twice weekly

REVIEW:
- No memory concerns reported by patient or family
- No confusion or disorientation
- No difficulty with complex tasks
- Continues working part-time as consultant

PHYSICAL EXAM:
- Alert and oriented x3
- Follows complex commands
- Normal neurological examination
- Gait: Normal, steady

ASSESSMENT:
- Normal cognitive function for age
- No evidence of cognitive impairment
- Functionally independent
- No dementia symptoms

PLAN:
- Continue routine health maintenance
- Return for annual visit
- No cognitive interventions needed
""",

    "borderline": """
MEMORY CLINIC EVALUATION

Patient: 68-year-old female
Chief Complaint: Family reports memory concerns

COGNITIVE ASSESSMENT:
- MMSE: 26/30 (borderline)
  - Lost points on recall (1/3) and delayed recall
  - Orientation: Intact
  - Language: Intact
  - Attention: Intact
- CDR: 0.5 (questionable dementia/MCI)

SUBJECTIVE COMPLAINTS:
- Patient: "Sometimes forget names and appointments"
- Spouse: "More forgetful than before, but managing"
- No disorientation or getting lost

FUNCTIONAL STATUS:
- ADLs: Fully independent
- IADLs: Mostly independent
  - Manages finances with minimal help
  - Uses calendar/reminders for appointments
  - Driving without issues
  - Cooking and household tasks: Independent

HISTORY:
- Onset: Gradual over 1-2 years
- Progression: Slowly progressive but subtle
- Mainly affects episodic memory
- No language or visuospatial problems

PHYSICAL EXAM:
- Normal neurological examination
- No motor deficits
- Gait normal

WORKUP:
- MRI: Mild age-appropriate atrophy
- Labs: Normal

ASSESSMENT:
- Mild Cognitive Impairment (MCI)
- CDR 0.5, MMSE 26 - borderline range
- Minimal functional impact
- May progress to dementia but not meeting criteria currently
- Monitor for progression

PLAN:
- Lifestyle modifications (exercise, cognitive activities)
- Repeat cognitive testing in 6 months
- Close monitoring for progression
- Not initiating medications at this time
"""
}


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
        "note": """NEUROLOGY NOTE: 75yo M with CDR 1.5, MMSE 15/30. ADL impaired. Progressive decline. Diagnosis: Moderate AD."""
    },
    {
        "id": 5,
        "patient": "Patient E",
        "note": """ANNUAL VISIT: 70yo F, MMSE 30/30, fully independent. No cognitive complaints. Assessment: Normal cognition."""
    }
]


# ============================================================================
# STRUCTURED PIPELINE IMPLEMENTATION
# ============================================================================

def run_structured_pipeline(llm, func_registry, preprocessor, extras_mgr, clinical_note):
    """Run STRUCTURED pipeline for ADRD classification"""

    print("\n" + "="*70)
    print("STRUCTURED PIPELINE - ADRD Classification")
    print("="*70)

    agent = AgentSystem(
        llm_interface=llm,
        function_registry=func_registry,
        regex_preprocessor=preprocessor,
        extras_manager=extras_mgr
    )

    task_prompt = """
    Classify this clinical note as ADRD or NO-ADRD based on:
    1. Cognitive assessment scores (MMSE, CDR)
    2. Functional impairment (ADL/IADL status)
    3. Progressive cognitive decline
    4. NIA-AA diagnostic criteria

    Provide binary classification with supporting evidence.
    """

    schema = {
        "classification": "ADRD or NO-ADRD",
        "confidence": "High, Medium, or Low",
        "key_findings": "Critical findings supporting classification",
        "reasoning": "Brief clinical reasoning"
    }

    print("\n🔄 Running 4-stage STRUCTURED pipeline...")

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
    """Run ADAPTIVE pipeline for ADRD classification"""

    print("\n" + "="*70)
    print("ADAPTIVE PIPELINE - ADRD Classification")
    print("="*70)

    agent = AgenticAgent(
        llm_interface=llm,
        function_registry=func_registry,
        max_iterations=10,
        confidence_threshold=0.8
    )

    task_prompt = """
    Autonomously classify this clinical note as ADRD or NO-ADRD.

    Use available ADRD assessment functions as needed.
    Adapt your analysis based on available data.
    """

    schema = {
        "classification": "ADRD or NO-ADRD",
        "confidence": "High, Medium, or Low",
        "key_findings": "Critical findings",
        "reasoning": "Clinical reasoning"
    }

    print("\n🤖 Running ADAPTIVE autonomous pipeline...")

    result = agent.extract(
        text=clinical_note,
        task_prompt=task_prompt,
        schema=schema
    )

    return result


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
    func_registry.import_functions(ADRD_FUNCTIONS_YAML)
    preprocessor.import_patterns(ADRD_PATTERNS_YAML)
    extras_mgr.import_extras(ADRD_EXTRAS_YAML)

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

    func_registry = FunctionRegistry()
    preprocessor = RegexPreprocessor()
    extras_mgr = ExtrasManager()

    # Import ADRD configurations
    print("   ✓ Loading ADRD functions...")
    success, count, msg = func_registry.import_functions(ADRD_FUNCTIONS_YAML)
    if success:
        print(f"     Loaded {count} functions")

    print("   ✓ Loading ADRD patterns...")
    success, count, msg = preprocessor.import_patterns(ADRD_PATTERNS_YAML)
    if success:
        print(f"     Loaded {count} patterns")

    print("   ✓ Loading ADRD extras...")
    success, count, msg = extras_mgr.import_extras(ADRD_EXTRAS_YAML)
    if success:
        print(f"     Loaded {count} extras")

    return func_registry, preprocessor, extras_mgr


def setup_llm():
    """Initialize LLM Manager"""

    print("\n🔧 Initializing LLM Manager...")

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
    parser = argparse.ArgumentParser(
        description="ADRD Classification SDK with Batch Processing"
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
    print(" ADRD CLASSIFICATION - SDK EXAMPLE")
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
