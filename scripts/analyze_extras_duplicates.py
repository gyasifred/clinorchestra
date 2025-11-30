#!/usr/bin/env python3
"""
Analyze extras files for duplicates and contradictions
"""
import json
import os
from collections import defaultdict
from pathlib import Path

def analyze_extras_duplicates():
    """Find duplicate and contradictory extras"""
    extras_dir = Path("/home/user/clinorchestra/extras")

    # Storage
    content_map = defaultdict(list)  # content hash -> list of files
    name_map = defaultdict(list)     # name -> list of files
    id_map = defaultdict(list)       # id -> list of files

    files_data = []

    # Read all files
    for file_path in sorted(extras_dir.glob("*.json")):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                files_data.append({
                    'path': file_path,
                    'filename': file_path.name,
                    'data': data
                })

                # Track by content
                content = data.get('content', '')
                content_map[content].append(file_path.name)

                # Track by name
                name = data.get('name', '')
                if name:
                    name_map[name].append(file_path.name)

                # Track by id
                id_val = data.get('id', '')
                if id_val:
                    id_map[id_val].append(file_path.name)

        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    print("=" * 80)
    print("EXTRAS FILES DUPLICATE ANALYSIS")
    print("=" * 80)
    print(f"\nTotal files analyzed: {len(files_data)}\n")

    # Find content duplicates
    print("\n" + "=" * 80)
    print("1. EXACT CONTENT DUPLICATES (same content field)")
    print("=" * 80)
    content_duplicates = [(content, files) for content, files in content_map.items()
                          if len(files) > 1 and content]

    if content_duplicates:
        for i, (content, files) in enumerate(content_duplicates, 1):
            print(f"\nDuplicate Group {i}: {len(files)} files")
            for f in sorted(files):
                print(f"  - {f}")
            print(f"  Content preview: {content[:100]}...")
    else:
        print("No exact content duplicates found.")

    # Find name duplicates
    print("\n" + "=" * 80)
    print("2. NAME DUPLICATES (same 'name' field)")
    print("=" * 80)
    name_duplicates = [(name, files) for name, files in name_map.items()
                       if len(files) > 1 and name]

    if name_duplicates:
        for i, (name, files) in enumerate(name_duplicates, 1):
            print(f"\nDuplicate Name {i}: '{name}'")
            print(f"  Found in {len(files)} files:")
            for f in sorted(files):
                print(f"    - {f}")
    else:
        print("No name duplicates found.")

    # Find ID duplicates
    print("\n" + "=" * 80)
    print("3. ID DUPLICATES (same 'id' field)")
    print("=" * 80)
    id_duplicates = [(id_val, files) for id_val, files in id_map.items()
                     if len(files) > 1 and id_val]

    if id_duplicates:
        for i, (id_val, files) in enumerate(id_duplicates, 1):
            print(f"\nDuplicate ID {i}: '{id_val}'")
            print(f"  Found in {len(files)} files:")
            for f in sorted(files):
                print(f"    - {f}")
    else:
        print("No ID duplicates found.")

    # Find filename pattern duplicates (_2, _3 suffixes)
    print("\n" + "=" * 80)
    print("4. FILENAME PATTERN DUPLICATES (e.g., file.json, file_2.json)")
    print("=" * 80)
    filename_patterns = defaultdict(list)
    for item in files_data:
        base_name = item['filename'].replace('_2.json', '.json').replace('_3.json', '.json')
        filename_patterns[base_name].append(item['filename'])

    pattern_dupes = [(base, files) for base, files in filename_patterns.items() if len(files) > 1]

    if pattern_dupes:
        for i, (base, files) in enumerate(pattern_dupes, 1):
            print(f"\nPattern {i}: {base}")
            for f in sorted(files):
                print(f"  - {f}")
    else:
        print("No filename pattern duplicates found.")

    # Analyze malnutrition and ADRD specific files
    print("\n" + "=" * 80)
    print("5. MALNUTRITION-SPECIFIC FILES")
    print("=" * 80)
    malnut_files = [item for item in files_data if 'malnutrition' in item['filename'].lower()
                    or 'malnutrition' in str(item['data'].get('content', '')).lower()
                    or 'aspen' in item['filename'].lower()
                    or 'who' in item['filename'].lower() and 'growth' in item['filename'].lower()]

    print(f"Found {len(malnut_files)} malnutrition-related files:")
    for item in sorted(malnut_files, key=lambda x: x['filename']):
        print(f"  - {item['filename']}")

    print("\n" + "=" * 80)
    print("6. ADRD-SPECIFIC FILES")
    print("=" * 80)
    adrd_files = [item for item in files_data if 'adrd' in item['filename'].lower()]

    print(f"Found {len(adrd_files)} ADRD-related files:")
    for item in sorted(adrd_files, key=lambda x: x['filename']):
        print(f"  - {item['filename']}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(files_data)}")
    print(f"Content duplicates: {len(content_duplicates)} groups")
    print(f"Name duplicates: {len(name_duplicates)} groups")
    print(f"ID duplicates: {len(id_duplicates)} groups")
    print(f"Filename pattern duplicates: {len(pattern_dupes)} groups")
    print(f"Malnutrition files: {len(malnut_files)}")
    print(f"ADRD files: {len(adrd_files)}")

    return {
        'content_duplicates': content_duplicates,
        'name_duplicates': name_duplicates,
        'id_duplicates': id_duplicates,
        'pattern_duplicates': pattern_dupes,
        'malnut_files': malnut_files,
        'adrd_files': adrd_files
    }

if __name__ == "__main__":
    analyze_extras_duplicates()
