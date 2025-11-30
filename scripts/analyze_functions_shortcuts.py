#!/usr/bin/env python3
"""
Analyze all function files for patterns that could cause shortcut decisions by LLM
"""
import json
import os
from pathlib import Path

def analyze_functions_for_shortcuts():
    """Find diagnostic language and shortcut patterns in function outputs"""
    functions_dir = Path("/home/user/clinorchestra/functions")

    # Problematic patterns to look for
    diagnostic_keywords = [
        "malnutrition", "dementia", "ADRD", "Alzheimer", "diagnosis",
        "severe", "moderate", "mild", "disorder", "disease",
        "immediate intervention", "requires", "indicates"
    ]

    shortcut_patterns = {
        'direct_diagnosis': [],
        'strong_language': [],
        'boolean_flags': [],
        'categorical_labels': [],
        'interpretation_bias': []
    }

    all_functions = []

    print("="*80)
    print("FUNCTION FILES SHORTCUT ANALYSIS")
    print("="*80)

    # Read all function files
    for func_file in sorted(functions_dir.glob("*.json")):
        try:
            with open(func_file, 'r') as f:
                data = json.load(f)
                func_name = data.get('name', '')
                code = data.get('code', '')
                description = data.get('description', '')

                all_functions.append({
                    'name': func_name,
                    'file': func_file.name,
                    'code': code,
                    'description': description
                })

                # Check for diagnostic language in code
                code_lower = code.lower()
                for keyword in diagnostic_keywords:
                    if keyword.lower() in code_lower:
                        # Check if it's in a return statement or string literal
                        if any(pattern in code for pattern in [
                            f'"{keyword}', f"'{keyword}",
                            f'"{keyword.title()}', f"'{keyword.title()}",
                            f'"{keyword.upper()}', f"'{keyword.upper()}"
                        ]):
                            shortcut_patterns['strong_language'].append({
                                'function': func_name,
                                'keyword': keyword,
                                'file': func_file.name
                            })

                # Check for boolean diagnostic flags
                if any(pattern in code for pattern in [
                    'dementia_present', 'dementia_suggested', 'supports_dementia',
                    'malnutrition_indicator', 'malnutrition_present'
                ]):
                    shortcut_patterns['boolean_flags'].append({
                        'function': func_name,
                        'file': func_file.name
                    })

                # Check for categorical classification returns
                if any(pattern in code for pattern in [
                    '"Severe ', '"Moderate ', '"Mild ',
                    "'Severe ", "'Moderate ", "'Mild ",
                    'severity =', 'classification =', 'diagnosis ='
                ]):
                    shortcut_patterns['categorical_labels'].append({
                        'function': func_name,
                        'file': func_file.name
                    })

        except Exception as e:
            print(f"Error reading {func_file.name}: {e}")

    # Print results
    print(f"\nTotal functions analyzed: {len(all_functions)}\n")

    print("="*80)
    print("SHORTCUT PATTERNS FOUND")
    print("="*80)

    print("\n1. BOOLEAN DIAGNOSTIC FLAGS (High Risk for Shortcuts)")
    print("-" * 80)
    if shortcut_patterns['boolean_flags']:
        for item in shortcut_patterns['boolean_flags']:
            print(f"  • {item['function']} ({item['file']})")
    else:
        print("  None found")

    print("\n2. CATEGORICAL DIAGNOSTIC LABELS (High Risk for Shortcuts)")
    print("-" * 80)
    if shortcut_patterns['categorical_labels']:
        seen = set()
        for item in shortcut_patterns['categorical_labels']:
            key = f"{item['function']}"
            if key not in seen:
                print(f"  • {item['function']} ({item['file']})")
                seen.add(key)
    else:
        print("  None found")

    print("\n3. STRONG DIAGNOSTIC LANGUAGE (Medium Risk for Bias)")
    print("-" * 80)
    if shortcut_patterns['strong_language']:
        seen = set()
        for item in shortcut_patterns['strong_language']:
            key = f"{item['function']}:{item['keyword']}"
            if key not in seen:
                print(f"  • {item['function']} uses '{item['keyword']}' ({item['file']})")
                seen.add(key)
    else:
        print("  None found")

    # Specific function analysis
    print("\n" + "="*80)
    print("HIGH-RISK FUNCTIONS FOR MISCLASSIFICATION")
    print("="*80)

    high_risk_functions = [
        'interpret_zscore_malnutrition',
        'interpret_albumin_malnutrition',
        'calculate_pediatric_nutrition_status',
        'assess_functional_independence',
        'calculate_cdr_severity',
        'calculate_mmse_severity',
        'calculate_moca_severity',
        'calculate_vascular_risk_score'
    ]

    for func_name in high_risk_functions:
        func_data = next((f for f in all_functions if f['name'] == func_name), None)
        if func_data:
            print(f"\n{func_name}:")
            print(f"  File: {func_data['file']}")

            # Check for specific patterns
            code = func_data['code']
            issues = []

            if 'dementia_present' in code or 'dementia_suggested' in code:
                issues.append("Returns boolean dementia flag")

            if 'malnutrition_indicator' in code:
                issues.append("Returns boolean malnutrition flag")

            if '"Severe ' in code or "'Severe " in code:
                issues.append("Returns severity labels")

            if 'supports_dementia' in code:
                issues.append("Returns diagnostic support flag")

            if 'SEVERE' in code or 'IMMEDIATE' in code or 'URGENT' in code:
                issues.append("Uses urgent/severe language in caps")

            if 'interpretation' in code.lower() and ('malnutrition' in code or 'dementia' in code):
                issues.append("Includes diagnostic interpretation text")

            if issues:
                for issue in issues:
                    print(f"    ⚠ {issue}")
            else:
                print("    ✓ No major issues found")

    return shortcut_patterns, all_functions

if __name__ == "__main__":
    analyze_functions_for_shortcuts()
