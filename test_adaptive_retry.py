#!/usr/bin/env python3
"""
Test script for adaptive retry system and tool deduplication

This script validates:
1. Adaptive retry system works correctly
2. Tool deduplication preventer blocks duplicates
3. LLM Manager integrates retry properly

Author: Frederick Gyasi (gyasi@musc.edu)
"""

import sys
from core.adaptive_retry import AdaptiveRetryManager, create_retry_context, RetryStrategy
from core.tool_dedup_preventer import create_tool_dedup_preventer
from core.logging_config import get_logger

logger = get_logger(__name__)


def test_adaptive_retry():
    """Test adaptive retry with simulated failures"""
    print("\n" + "=" * 80)
    print("TEST 1: Adaptive Retry System")
    print("=" * 80)

    retry_manager = AdaptiveRetryManager(max_retries=5)
    retry_context = create_retry_context(
        clinical_text="Patient is a 5-year-old with weight 15 kg and height 105 cm...",
        max_attempts=5
    )

    attempt_count = [0]

    def failing_generation():
        """Simulated generation that fails first 2 times"""
        attempt_count[0] += 1
        print(f"\n  Attempt {attempt_count[0]}: ", end="")

        if attempt_count[0] < 3:
            print(f"FAIL (simulated error)")
            raise ValueError(f"Simulated failure #{attempt_count[0]}")

        print(f"SUCCESS")
        return "Generated response"

    try:
        result = retry_manager.execute_with_retry(
            failing_generation,
            retry_context
        )
        print(f"\n✓ Retry test PASSED: Got result after {attempt_count[0]} attempts")
        print(f"  Final clinical text length: {len(retry_context.clinical_text)} chars")
        print(f"  (reduced from original {retry_context.original_clinical_text_length} chars)")
        return True
    except Exception as e:
        print(f"\n✗ Retry test FAILED: {e}")
        return False


def test_tool_deduplication():
    """Test tool deduplication preventer"""
    print("\n" + "=" * 80)
    print("TEST 2: Tool Deduplication Preventer")
    print("=" * 80)

    preventer = create_tool_dedup_preventer(max_tool_calls=100)

    # Simulate tool calls
    tool_calls = [
        {'type': 'function', 'name': 'calculate_bmi', 'parameters': {'weight_kg': 15, 'height_cm': 105}},
        {'type': 'function', 'name': 'calculate_age', 'parameters': {'birth_date': '2019-01-15'}},
        {'type': 'function', 'name': 'calculate_bmi', 'parameters': {'weight_kg': 15, 'height_cm': 105}},  # Duplicate!
        {'type': 'rag', 'name': 'query_rag', 'parameters': {'query': 'pediatric growth standards'}},
        {'type': 'rag', 'name': 'query_rag', 'parameters': {'query': 'pediatric growth standards'}},  # Duplicate!
    ]

    print(f"\nOriginal tool calls: {len(tool_calls)}")
    for i, tc in enumerate(tool_calls):
        print(f"  {i+1}. {tc['type']}.{tc['name']}")

    unique_calls, num_duplicates = preventer.filter_duplicates(tool_calls)

    print(f"\nAfter deduplication: {len(unique_calls)} unique calls")
    print(f"Duplicates prevented: {num_duplicates}")
    print(f"\nBudget status: {preventer.get_budget_status()}")

    # Verify
    if len(unique_calls) == 3 and num_duplicates == 2:
        print(f"\n✓ Deduplication test PASSED")
        return True
    else:
        print(f"\n✗ Deduplication test FAILED")
        print(f"  Expected: 3 unique, 2 duplicates")
        print(f"  Got: {len(unique_calls)} unique, {num_duplicates} duplicates")
        return False


def test_prevention_prompt():
    """Test prevention prompt generation"""
    print("\n" + "=" * 80)
    print("TEST 3: Prevention Prompt Generation")
    print("=" * 80)

    preventer = create_tool_dedup_preventer(max_tool_calls=100)

    # Record some calls
    tool_calls = [
        {'type': 'function', 'name': 'calculate_bmi', 'parameters': {'weight_kg': 15, 'height_cm': 105}},
        {'type': 'rag', 'name': 'query_rag', 'parameters': {'query': 'pediatric growth standards'}},
    ]

    preventer.filter_duplicates(tool_calls)

    # Generate prevention prompt
    prompt = preventer.generate_prevention_prompt()

    if prompt and "ALREADY been executed" in prompt:
        print("\n✓ Prevention prompt test PASSED")
        print("\nGenerated prompt preview:")
        print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        return True
    else:
        print("\n✗ Prevention prompt test FAILED")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("ADAPTIVE RETRY SYSTEM - TEST SUITE")
    print("=" * 80)

    results = []

    # Test 1: Adaptive retry
    results.append(("Adaptive Retry", test_adaptive_retry()))

    # Test 2: Tool deduplication
    results.append(("Tool Deduplication", test_tool_deduplication()))

    # Test 3: Prevention prompt
    results.append(("Prevention Prompt", test_prevention_prompt()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 80)

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Adaptive retry system is working correctly.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
