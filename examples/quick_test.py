#!/usr/bin/env python3
"""
Quick test script to verify ClinAnnotate is working
Tests functions, patterns, and extras without requiring full UI
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_functions():
    """Test medical calculation functions"""
    print("\n" + "="*60)
    print("TESTING MEDICAL FUNCTIONS")
    print("="*60)

    from core.function_registry import FunctionRegistry

    registry = FunctionRegistry()
    functions = registry.list_functions()

    print(f"\nRegistered functions: {len(functions)}")
    for func_name in functions[:5]:  # Show first 5
        print(f"  - {func_name}")
    if len(functions) > 5:
        print(f"  ... and {len(functions) - 5} more")

    # Test BMI calculation
    print("\nTest: calculate_bmi(70, 1.75)")
    success, result, msg = registry.execute_function(
        "calculate_bmi",
        weight_kg=70,
        height_m=1.75
    )
    if success:
        print(f"  ✅ Result: {result} kg/m²")
    else:
        print(f"  ❌ Failed: {msg}")

    # Test unit conversion
    print("\nTest: lbs_to_kg(154)")
    success, result, msg = registry.execute_function("lbs_to_kg", lbs=154)
    if success:
        print(f"  ✅ Result: {result} kg")
    else:
        print(f"  ❌ Failed: {msg}")

def test_patterns():
    """Test text normalization patterns"""
    print("\n" + "="*60)
    print("TESTING NORMALIZATION PATTERNS")
    print("="*60)

    from core.regex_preprocessor import RegexPreprocessor

    preprocessor = RegexPreprocessor()
    patterns = preprocessor.list_patterns()

    print(f"\nRegistered patterns: {len(patterns)}")
    for pattern_name in list(patterns.keys())[:5]:  # Show first 5
        print(f"  - {pattern_name}")
    if len(patterns) > 5:
        print(f"  ... and {len(patterns) - 5} more")

    # Test normalization
    test_text = "Patient has DM with HbA1c 8.2%. BP 145/92. Current meds: Metformin 1000mg BID."
    print(f"\nOriginal text:")
    print(f"  {test_text}")

    normalized = preprocessor.preprocess(test_text)
    print(f"\nNormalized text:")
    print(f"  {normalized}")

def test_extras():
    """Test extras/hints"""
    print("\n" + "="*60)
    print("TESTING EXTRAS (HINTS)")
    print("="*60)

    from core.extras_manager import ExtrasManager

    manager = ExtrasManager()
    extras = manager.list_extras()

    print(f"\nRegistered extras: {len(extras)}")
    for extra in extras[:3]:  # Show first 3
        print(f"  - {extra.get('name', extra['id'])}")
    if len(extras) > 3:
        print(f"  ... and {len(extras) - 3} more")

    # Test keyword matching
    print("\nTest: match_extras_by_keywords(['diabetes', 'HbA1c', 'diagnostic'])")
    matched = manager.match_extras_by_keywords(['diabetes', 'HbA1c', 'diagnostic'])

    if matched:
        print(f"  Found {len(matched)} matches:")
        for match in matched[:2]:  # Show first 2
            print(f"    - {match.get('name', match['id'])} (relevance: {match.get('relevance_score', 0):.2f})")
    else:
        print("  No matches found")

def test_sample_data():
    """Verify sample data exists"""
    print("\n" + "="*60)
    print("VERIFYING SAMPLE DATA")
    print("="*60)

    import pandas as pd

    csv_path = Path(__file__).parent / "sample_clinical_notes.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\n✅ Sample data loaded: {len(df)} clinical notes")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Diagnoses: {df['diagnosis_label'].unique().tolist()}")
    else:
        print(f"\n❌ Sample data not found at {csv_path}")

def main():
    print("\n" + "="*70)
    print("CLINANNOTATE - QUICK TEST")
    print("="*70)

    try:
        test_functions()
    except Exception as e:
        print(f"\n❌ Function test failed: {e}")

    try:
        test_patterns()
    except Exception as e:
        print(f"\n❌ Pattern test failed: {e}")

    try:
        test_extras()
    except Exception as e:
        print(f"\n❌ Extras test failed: {e}")

    try:
        test_sample_data()
    except Exception as e:
        print(f"\n❌ Sample data test failed: {e}")

    print("\n" + "="*70)
    print("✅ QUICK TEST COMPLETED")
    print("="*70)
    print("\nTo run full application:")
    print("  python annotate.py")
    print("\nOr if installed via pip:")
    print("  clinannotate")
    print()

if __name__ == "__main__":
    main()
