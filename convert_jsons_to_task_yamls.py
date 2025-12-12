#!/usr/bin/env python3
"""
Convert all JSON files to task-specific YAML files
Groups by task: malnutrition, ADRD, MIMIC-IV, and shared/common
"""

import json
import yaml
from pathlib import Path
from typing import List, Dict, Any


def categorize_function(name: str) -> str:
    """Determine task category for a function"""
    name_lower = name.lower()

    # Malnutrition keywords
    if any(kw in name_lower for kw in ['malnutrition', 'nutrition', 'zscore', 'z_score', 'growth',
                                         'percentile', 'anthropometric', 'bmi', 'weight', 'height']):
        return 'malnutrition'

    # ADRD/Cognitive keywords
    if any(kw in name_lower for kw in ['cdr', 'mmse', 'moca', 'cognitive', 'dementia', 'alzheimer',
                                         'adrd', 'memory', 'orientation']):
        return 'adrd'

    # MIMIC-IV/Critical care keywords
    if any(kw in name_lower for kw in ['mimic', 'icu', 'sofa', 'apache', 'sepsis', 'kdigo',
                                         'aki', 'mechanical', 'ventilator']):
        return 'mimic_iv'

    # Common/Shared utilities
    if any(kw in name_lower for kw in ['calculate_age', 'calculate_days', 'kg_to', 'lbs_to',
                                         'cm_to', 'inches_to', 'calculate_bmi']):
        return 'shared'

    return 'shared'


def categorize_pattern(name: str) -> str:
    """Determine task category for a pattern"""
    name_lower = name.lower()

    # Malnutrition keywords
    if any(kw in name_lower for kw in ['malnutrition', 'nutrition', 'zscore', 'z_score', 'growth',
                                         'percentile', 'anthropometric', 'stunting', 'wasting']):
        return 'malnutrition'

    # ADRD keywords
    if any(kw in name_lower for kw in ['cdr', 'mmse', 'moca', 'cognitive', 'dementia', 'alzheimer',
                                         'adrd', 'memory', 'orientation', 'boston_naming', 'clock_drawing']):
        return 'adrd'

    # MIMIC-IV keywords
    if any(kw in name_lower for kw in ['mimic', 'icu', 'sofa', 'apache', 'sepsis', 'kdigo',
                                         'aki', 'mechanical', 'ventilator']):
        return 'mimic_iv'

    return 'shared'


def categorize_extra(extra: Dict[str, Any]) -> str:
    """Determine task category for an extra"""
    name = extra.get('name', '').lower()
    content = extra.get('content', '').lower()
    metadata = extra.get('metadata', {})
    category = metadata.get('category', '').lower()

    # Check all fields for keywords
    text = f"{name} {content} {category}"

    # Malnutrition keywords
    if any(kw in text for kw in ['malnutrition', 'nutrition', 'zscore', 'z-score', 'aspen',
                                   'who growth', 'growth standard', 'stunting', 'wasting',
                                   'anthropometric', 'muac', 'percentile']):
        return 'malnutrition'

    # ADRD keywords
    if any(kw in text for kw in ['adrd', 'alzheimer', 'dementia', 'cognitive', 'cdr', 'mmse',
                                   'moca', 'memory', 'nia-aa', 'biomarker', 'amyloid', 'tau']):
        return 'adrd'

    # MIMIC-IV keywords
    if any(kw in text for kw in ['mimic', 'icu', 'critical care', 'sofa', 'apache', 'sepsis',
                                   'kdigo', 'aki', 'mechanical ventilation']):
        return 'mimic_iv'

    return 'shared'


def convert_functions():
    """Convert function JSONs to task-specific YAMLs"""
    functions_dir = Path('functions')
    if not functions_dir.exists():
        print("No functions directory found")
        return

    # Group functions by task
    tasks = {
        'malnutrition': [],
        'adrd': [],
        'mimic_iv': [],
        'shared': []
    }

    for json_file in functions_dir.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                func_data = json.load(f)

            task = categorize_function(func_data.get('name', ''))
            tasks[task].append(func_data)

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    # Save each task's functions to YAML
    for task_name, functions in tasks.items():
        if functions:
            output_file = Path('yaml_configs') / f'{task_name}_functions.yaml'
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(functions, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            print(f"✓ Created {output_file} ({len(functions)} functions)")


def convert_patterns():
    """Convert pattern JSONs to task-specific YAMLs"""
    patterns_dir = Path('patterns')
    if not patterns_dir.exists():
        print("No patterns directory found")
        return

    # Group patterns by task
    tasks = {
        'malnutrition': [],
        'adrd': [],
        'mimic_iv': [],
        'shared': []
    }

    for json_file in patterns_dir.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                pattern_data = json.load(f)

            task = categorize_pattern(pattern_data.get('name', ''))
            tasks[task].append(pattern_data)

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    # Save each task's patterns to YAML
    for task_name, patterns in tasks.items():
        if patterns:
            output_file = Path('yaml_configs') / f'{task_name}_patterns.yaml'
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(patterns, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            print(f"✓ Created {output_file} ({len(patterns)} patterns)")


def convert_extras():
    """Convert extras JSONs to task-specific YAMLs"""
    extras_dir = Path('extras')
    if not extras_dir.exists():
        print("No extras directory found")
        return

    # Group extras by task
    tasks = {
        'malnutrition': [],
        'adrd': [],
        'mimic_iv': [],
        'shared': []
    }

    for json_file in extras_dir.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                extra_data = json.load(f)

            task = categorize_extra(extra_data)
            tasks[task].append(extra_data)

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    # Save each task's extras to YAML
    for task_name, extras in tasks.items():
        if extras:
            output_file = Path('yaml_configs') / f'{task_name}_extras.yaml'
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(extras, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
            print(f"✓ Created {output_file} ({len(extras)} extras)")


def main():
    """Main conversion process"""
    print("=" * 70)
    print("Converting JSON files to task-specific YAMLs")
    print("=" * 70)

    print("\n1. Converting functions...")
    convert_functions()

    print("\n2. Converting patterns...")
    convert_patterns()

    print("\n3. Converting extras...")
    convert_extras()

    print("\n" + "=" * 70)
    print("✓ Conversion complete!")
    print("=" * 70)
    print("\nTask-specific YAML files created in yaml_configs/:")
    print("  - malnutrition_functions.yaml")
    print("  - malnutrition_patterns.yaml")
    print("  - malnutrition_extras.yaml")
    print("  - adrd_functions.yaml")
    print("  - adrd_patterns.yaml")
    print("  - adrd_extras.yaml")
    print("  - mimic_iv_functions.yaml")
    print("  - mimic_iv_patterns.yaml")
    print("  - mimic_iv_extras.yaml")
    print("  - shared_functions.yaml")
    print("  - shared_patterns.yaml")
    print("  - shared_extras.yaml")
    print("\nUsers can import these via UI to populate their registries.")


if __name__ == '__main__':
    main()
