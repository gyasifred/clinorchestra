#!/usr/bin/env python3
"""
Comprehensive audit of ALL 44 functions - read full contents and identify ALL issues
"""
import json
from pathlib import Path

def audit_all_functions():
    """Read and analyze every function file completely"""
    functions_dir = Path("/home/user/clinorchestra/functions")

    # Categories of issues
    issues = {
        'boolean_flags': [],  # dementia_present, malnutrition_indicator, etc.
        'diagnostic_labels': [],  # "Severe Malnutrition", "Dementia", etc.
        'directive_language': [],  # "immediate intervention", "requires", "must"
        'severity_classifications': [],  # returns severity directly
        'interpretation_text': [],  # includes diagnostic interpretation
        'clean_functions': []  # functions that only return objective data
    }

    all_functions = []

    for func_file in sorted(functions_dir.glob("*.json")):
        with open(func_file, 'r') as f:
            data = json.load(f)
            func_name = data.get('name', '')
            code = data.get('code', '')
            description = data.get('description', '')

            func_info = {
                'name': func_name,
                'file': func_file.name,
                'code': code,
                'description': description,
                'issues_found': []
            }

            # Check for boolean diagnostic flags
            boolean_patterns = [
                'dementia_present', 'dementia_suggested', 'supports_dementia',
                'malnutrition_indicator', 'malnutrition_present',
                'impairment_suggested', 'disorder_present'
            ]
            for pattern in boolean_patterns:
                if pattern in code:
                    func_info['issues_found'].append(f'Boolean flag: {pattern}')
                    issues['boolean_flags'].append(func_name)
                    break

            # Check for diagnostic labels in return statements
            diagnostic_terms = [
                '"Severe ', '"Moderate ', '"Mild ',
                '"Dementia"', '"Malnutrition"', '"Wasting"', '"Stunting"',
                "'Severe ", "'Moderate ", "'Mild ",
                "'Dementia'", "'Malnutrition'", "'Wasting'", "'Stunting'"
            ]
            for term in diagnostic_terms:
                if term in code:
                    func_info['issues_found'].append(f'Diagnostic label: {term}')
                    if func_name not in issues['diagnostic_labels']:
                        issues['diagnostic_labels'].append(func_name)
                    break

            # Check for directive/urgent language
            directive_patterns = [
                'immediate intervention', 'urgent', 'requires',
                'IMMEDIATE', 'URGENT', 'CRITICAL',
                'must be', 'should be', 'needs to'
            ]
            for pattern in directive_patterns:
                if pattern in code:
                    func_info['issues_found'].append(f'Directive language: {pattern}')
                    if func_name not in issues['directive_language']:
                        issues['directive_language'].append(func_name)
                    break

            # Check for severity classification returns
            if 'severity' in code.lower() and ('return' in code or '=' in code):
                func_info['issues_found'].append('Returns severity classification')
                if func_name not in issues['severity_classifications']:
                    issues['severity_classifications'].append(func_name)

            # Check for interpretation fields
            if '"interpretation"' in code or "'interpretation'" in code:
                func_info['issues_found'].append('Includes interpretation field')
                if func_name not in issues['interpretation_text']:
                    issues['interpretation_text'].append(func_name)

            # Mark as clean if no issues
            if not func_info['issues_found']:
                issues['clean_functions'].append(func_name)

            all_functions.append(func_info)

    # Print comprehensive report
    print("=" * 80)
    print("COMPREHENSIVE FUNCTION AUDIT - ALL 44 FUNCTIONS")
    print("=" * 80)
    print(f"\nTotal functions analyzed: {len(all_functions)}")

    print("\n" + "=" * 80)
    print("ISSUE SUMMARY")
    print("=" * 80)
    print(f"Functions with boolean flags: {len(set(issues['boolean_flags']))}")
    print(f"Functions with diagnostic labels: {len(set(issues['diagnostic_labels']))}")
    print(f"Functions with directive language: {len(set(issues['directive_language']))}")
    print(f"Functions with severity classifications: {len(set(issues['severity_classifications']))}")
    print(f"Functions with interpretation text: {len(set(issues['interpretation_text']))}")
    print(f"Clean functions (objective data only): {len(issues['clean_functions'])}")

    # List all problematic functions
    print("\n" + "=" * 80)
    print("FUNCTIONS REQUIRING FIXES")
    print("=" * 80)

    problematic_functions = set()
    for category in ['boolean_flags', 'diagnostic_labels', 'directive_language',
                     'severity_classifications', 'interpretation_text']:
        problematic_functions.update(issues[category])

    print(f"\nTotal functions needing fixes: {len(problematic_functions)}")
    print("\nDetailed breakdown:")

    for func in sorted(all_functions, key=lambda x: x['name']):
        if func['issues_found']:
            print(f"\n{func['name']} ({func['file']}):")
            for issue in func['issues_found']:
                print(f"  ⚠ {issue}")

    # List clean functions
    print("\n" + "=" * 80)
    print("CLEAN FUNCTIONS (No changes needed)")
    print("=" * 80)
    for func_name in sorted(issues['clean_functions']):
        print(f"  ✓ {func_name}")

    return issues, all_functions

if __name__ == "__main__":
    audit_all_functions()
