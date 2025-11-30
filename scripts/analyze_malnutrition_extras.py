#!/usr/bin/env python3
"""
Comprehensive analysis of all malnutrition-related extras for contradictions and duplicates
"""
import json
from pathlib import Path
from collections import defaultdict

def analyze_malnutrition_extras():
    """Read and analyze all malnutrition, z-score, and percentile extras"""
    extras_dir = Path("/home/user/clinorchestra/extras")

    # Get all relevant files
    malnutrition_files = []
    for pattern in ['*malnutrition*.json', '*z*score*.json', '*percentile*.json']:
        malnutrition_files.extend(extras_dir.glob(pattern))

    # Remove duplicates
    malnutrition_files = list(set(malnutrition_files))

    print("=" * 80)
    print("MALNUTRITION EXTRAS COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal malnutrition-related extras: {len(malnutrition_files)}\n")

    # Read all files
    extras_data = []
    for file_path in sorted(malnutrition_files):
        with open(file_path, 'r') as f:
            data = json.load(f)
            extras_data.append({
                'file': file_path.name,
                'id': data.get('id', ''),
                'name': data.get('name', ''),
                'type': data.get('type', ''),
                'content': data.get('content', ''),
                'metadata': data.get('metadata', {})
            })

    # Check for content duplicates
    print("=" * 80)
    print("1. CHECKING FOR DUPLICATE CONTENT")
    print("=" * 80)

    content_map = defaultdict(list)
    for extra in extras_data:
        content = extra['content']
        if content:  # Only check non-empty content
            content_map[content].append(extra['file'])

    duplicates = [(content, files) for content, files in content_map.items() if len(files) > 1]

    if duplicates:
        print(f"\nFound {len(duplicates)} sets of duplicate content:\n")
        for i, (content, files) in enumerate(duplicates, 1):
            print(f"Duplicate Set {i}:")
            for f in sorted(files):
                print(f"  - {f}")
            print(f"  Content preview: {content[:100]}...")
            print()
    else:
        print("\n✓ No duplicate content found")

    # Check for z-score threshold definitions
    print("\n" + "=" * 80)
    print("2. Z-SCORE THRESHOLD DEFINITIONS")
    print("=" * 80)

    threshold_definitions = {}
    for extra in extras_data:
        content = extra['content'].lower()
        file_name = extra['file']

        # Look for severity definitions
        if 'mild' in content and 'z-score' in content:
            if '-1' in content or '-2' in content:
                threshold_definitions[file_name] = {
                    'content': extra['content'],
                    'type': 'defines_thresholds'
                }

    print(f"\nFound {len(threshold_definitions)} files defining z-score thresholds:\n")
    for file_name, info in sorted(threshold_definitions.items()):
        print(f"{file_name}:")
        # Extract threshold info
        content = info['content']
        if 'Mild' in content:
            mild_start = content.find('Mild')
            mild_section = content[mild_start:mild_start+200]
            print(f"  {mild_section[:150]}...")
        print()

    # Check for contradictions in thresholds
    print("\n" + "=" * 80)
    print("3. CHECKING FOR CONTRADICTORY THRESHOLDS")
    print("=" * 80)

    # Extract specific threshold mentions
    threshold_patterns = {
        'mild_aspen': [],
        'moderate_aspen': [],
        'severe_aspen': [],
        'mild_who': [],
        'moderate_who': [],
        'severe_who': []
    }

    for extra in extras_data:
        content = extra['content']
        file_name = extra['file']

        # ASPEN patterns
        if 'ASPEN' in content and 'Mild' in content:
            if '-1.0 to -1.9' in content or '-1 to -1.9' in content:
                threshold_patterns['mild_aspen'].append((file_name, 'Mild: -1.0 to -1.9'))
            elif '1-2 SD' in content:
                threshold_patterns['mild_aspen'].append((file_name, 'Mild: 1-2 SD (AMBIGUOUS)'))

        if 'ASPEN' in content and 'Moderate' in content:
            if '-2.0 to -2.9' in content or '-2 to -2.9' in content:
                threshold_patterns['moderate_aspen'].append((file_name, 'Moderate: -2.0 to -2.9'))
            elif '2-3 SD' in content:
                threshold_patterns['moderate_aspen'].append((file_name, 'Moderate: 2-3 SD (AMBIGUOUS)'))

        if 'ASPEN' in content and 'Severe' in content:
            if '≤-3.0' in content or '<-3' in content or '≤ -3' in content:
                threshold_patterns['severe_aspen'].append((file_name, 'Severe: ≤-3.0'))
            elif '>3 SD' in content:
                threshold_patterns['severe_aspen'].append((file_name, 'Severe: >3 SD (AMBIGUOUS)'))

        # WHO patterns
        if 'WHO' in content:
            if 'z < -3' in content or 'z<-3' in content:
                threshold_patterns['severe_who'].append((file_name, 'WHO Severe: z < -3'))
            if '-3 ≤ z < -2' in content or 'z < -2' in content:
                threshold_patterns['moderate_who'].append((file_name, 'WHO Moderate: -3 ≤ z < -2'))

    print("\nASPEN Threshold Definitions:")
    for category in ['mild_aspen', 'moderate_aspen', 'severe_aspen']:
        if threshold_patterns[category]:
            print(f"\n{category.upper()}:")
            for file_name, definition in threshold_patterns[category]:
                print(f"  {file_name}: {definition}")

    print("\nWHO Threshold Definitions:")
    for category in ['mild_who', 'moderate_who', 'severe_who']:
        if threshold_patterns[category]:
            print(f"\n{category.upper()}:")
            for file_name, definition in threshold_patterns[category]:
                print(f"  {file_name}: {definition}")

    # Check for similar purposes (duplicates)
    print("\n" + "=" * 80)
    print("4. FILES WITH SIMILAR PURPOSES (Potential Duplicates)")
    print("=" * 80)

    # Group by purpose
    purpose_groups = defaultdict(list)
    for extra in extras_data:
        if 'z-score' in extra['name'].lower() and 'conversion' in extra['name'].lower():
            purpose_groups['percentile_conversion'].append(extra['file'])
        elif 'aspen' in extra['name'].lower() and 'criteria' in extra['name'].lower():
            purpose_groups['aspen_criteria'].append(extra['file'])
        elif 'who' in extra['name'].lower() and 'z-score' in extra['name'].lower():
            purpose_groups['who_criteria'].append(extra['file'])
        elif 'interpretation' in extra['name'].lower() and 'z-score' in extra['name'].lower():
            purpose_groups['z_interpretation'].append(extra['file'])

    for purpose, files in sorted(purpose_groups.items()):
        if len(files) > 1:
            print(f"\n{purpose.upper().replace('_', ' ')} ({len(files)} files):")
            for f in sorted(files):
                print(f"  - {f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF ISSUES")
    print("=" * 80)
    print(f"Content duplicates: {len(duplicates)} sets")
    print(f"Files defining thresholds: {len(threshold_definitions)}")

    # Count ambiguous definitions
    ambiguous = sum(1 for category in threshold_patterns.values()
                    for _, definition in category if 'AMBIGUOUS' in definition)
    print(f"Ambiguous threshold definitions: {ambiguous}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    if duplicates:
        print("1. Remove duplicate content files - keeping most comprehensive version")
    if ambiguous > 0:
        print("2. Fix ambiguous threshold definitions (e.g., '1-2 SD' → 'z-score -1.0 to -1.9')")
    if len(purpose_groups.get('aspen_criteria', [])) > 1:
        print("3. Consolidate ASPEN criteria files into single comprehensive file")
    if len(purpose_groups.get('percentile_conversion', [])) > 1:
        print("4. Keep only one percentile conversion reference")

    return extras_data, duplicates, threshold_patterns

if __name__ == "__main__":
    analyze_malnutrition_extras()
