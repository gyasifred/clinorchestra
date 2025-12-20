#!/usr/bin/env python3
"""
JSON Content Validator

Validates that extracted JSON has meaningful content, not just empty fields.
Prevents accepting JSON responses that have all null/empty/unknown values.

Author: Frederick Gyasi (gyasi@musc.edu)
Institution: Medical University of South Carolina, Biomedical Informatics Center
Version: 1.0.0
"""

from typing import Dict, Any, List, Tuple, Optional
from core.logging_config import get_logger

logger = get_logger(__name__)


class JSONValidationResult:
    """Result of JSON content validation"""
    def __init__(self, is_valid: bool, reason: str = "", filled_field_count: int = 0,
                 total_field_count: int = 0, filled_fields: List[str] = None,
                 empty_fields: List[str] = None):
        self.is_valid = is_valid
        self.reason = reason
        self.filled_field_count = filled_field_count
        self.total_field_count = total_field_count
        self.filled_fields = filled_fields or []
        self.empty_fields = empty_fields or []
        self.fill_percentage = (filled_field_count / total_field_count * 100) if total_field_count > 0 else 0


class JSONContentValidator:
    """
    Validates that JSON extraction has meaningful content

    Checks:
    1. JSON is not None or empty dict
    2. Fields contain actual values (not null, empty string, "unknown", "N/A", etc.)
    3. Minimum percentage of fields are filled
    4. At least minimum number of fields have content
    """

    # Values that are considered "empty" or "no information"
    EMPTY_VALUES = {
        None,
        "",
        "null",
        "none",
        "unknown",
        "n/a",
        "na",
        "not available",
        "not specified",
        "not mentioned",
        "not found",
        "not documented",
        "not stated",
        "unspecified",
        "unclear",
        "pending",
        "see notes",
        "[]",
        "{}",
    }

    def __init__(self,
                 min_filled_fields: int = 1,
                 min_fill_percentage: float = 10.0,
                 strict_mode: bool = False):
        """
        Initialize validator

        Args:
            min_filled_fields: Minimum number of fields that must have content
            min_fill_percentage: Minimum percentage of fields that must be filled (0-100)
            strict_mode: If True, requires higher thresholds for validation
        """
        self.min_filled_fields = min_filled_fields
        self.min_fill_percentage = min_fill_percentage
        self.strict_mode = strict_mode

        if strict_mode:
            self.min_filled_fields = max(min_filled_fields, 3)
            self.min_fill_percentage = max(min_fill_percentage, 25.0)

    def validate(self, json_data: Optional[Dict[str, Any]],
                 schema: Optional[Dict[str, Any]] = None) -> JSONValidationResult:
        """
        Validate that JSON has meaningful content

        Args:
            json_data: Extracted JSON data to validate
            schema: Optional JSON schema to validate against

        Returns:
            JSONValidationResult with validation status and details
        """
        # Check 1: Not None
        if json_data is None:
            return JSONValidationResult(
                is_valid=False,
                reason="JSON is None"
            )

        # Check 2: Is dictionary
        if not isinstance(json_data, dict):
            return JSONValidationResult(
                is_valid=False,
                reason=f"JSON is not a dictionary (type: {type(json_data).__name__})"
            )

        # Check 3: Not empty dict
        if not json_data:
            return JSONValidationResult(
                is_valid=False,
                reason="JSON is empty dictionary"
            )

        # Analyze field content
        filled_fields = []
        empty_fields = []

        for field_name, field_value in json_data.items():
            if self._is_field_filled(field_value):
                filled_fields.append(field_name)
            else:
                empty_fields.append(field_name)

        total_fields = len(json_data)
        filled_count = len(filled_fields)
        fill_percentage = (filled_count / total_fields * 100) if total_fields > 0 else 0

        # Check 4: Minimum filled fields
        if filled_count < self.min_filled_fields:
            return JSONValidationResult(
                is_valid=False,
                reason=f"Only {filled_count}/{total_fields} fields filled (minimum: {self.min_filled_fields})",
                filled_field_count=filled_count,
                total_field_count=total_fields,
                filled_fields=filled_fields,
                empty_fields=empty_fields
            )

        # Check 5: Minimum fill percentage
        if fill_percentage < self.min_fill_percentage:
            return JSONValidationResult(
                is_valid=False,
                reason=f"Only {fill_percentage:.1f}% fields filled (minimum: {self.min_fill_percentage}%)",
                filled_field_count=filled_count,
                total_field_count=total_fields,
                filled_fields=filled_fields,
                empty_fields=empty_fields
            )

        # Check 6: Schema validation (if provided)
        if schema:
            schema_validation = self._validate_against_schema(json_data, schema)
            if not schema_validation[0]:
                return JSONValidationResult(
                    is_valid=False,
                    reason=schema_validation[1],
                    filled_field_count=filled_count,
                    total_field_count=total_fields,
                    filled_fields=filled_fields,
                    empty_fields=empty_fields
                )

        # All checks passed
        return JSONValidationResult(
            is_valid=True,
            reason=f"Valid JSON with {filled_count}/{total_fields} fields filled ({fill_percentage:.1f}%)",
            filled_field_count=filled_count,
            total_field_count=total_fields,
            filled_fields=filled_fields,
            empty_fields=empty_fields
        )

    def _is_field_filled(self, value: Any) -> bool:
        """
        Check if a field value is considered "filled" with meaningful content

        Args:
            value: Field value to check

        Returns:
            True if field has meaningful content, False otherwise
        """
        # None is empty
        if value is None:
            return False

        # Convert to string for comparison
        if isinstance(value, str):
            value_str = value.strip().lower()
        elif isinstance(value, (list, dict)):
            # Empty lists/dicts are empty
            if not value:
                return False
            # Non-empty lists/dicts are considered filled
            return True
        elif isinstance(value, bool):
            # Booleans are always considered filled (True or False are both meaningful)
            return True
        elif isinstance(value, (int, float)):
            # Numbers are considered filled (including 0)
            return True
        else:
            value_str = str(value).strip().lower()

        # Check against empty values
        if value_str in self.EMPTY_VALUES:
            return False

        # Empty string
        if not value_str:
            return False

        # Looks filled
        return True

    def _validate_against_schema(self, json_data: Dict[str, Any],
                                 schema: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate JSON data against schema

        Args:
            json_data: JSON data to validate
            schema: JSON schema

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for required fields (if schema specifies)
        if isinstance(schema, dict):
            for field_name, field_spec in schema.items():
                if isinstance(field_spec, dict):
                    # Check if field is required
                    is_required = field_spec.get('required', False)
                    if is_required and field_name not in json_data:
                        return False, f"Required field '{field_name}' is missing"

                    # Check if required field has content
                    if is_required and field_name in json_data:
                        if not self._is_field_filled(json_data[field_name]):
                            return False, f"Required field '{field_name}' is empty"

        return True, ""

    def get_validation_summary(self, result: JSONValidationResult) -> str:
        """
        Get human-readable validation summary

        Args:
            result: JSONValidationResult to summarize

        Returns:
            Human-readable summary string
        """
        if result.is_valid:
            return f"✅ Valid: {result.filled_field_count}/{result.total_field_count} fields filled ({result.fill_percentage:.1f}%)"
        else:
            return f"❌ Invalid: {result.reason}"


def create_json_validator(min_filled_fields: int = 1,
                          min_fill_percentage: float = 10.0,
                          strict_mode: bool = False) -> JSONContentValidator:
    """
    Factory function to create JSON validator

    Args:
        min_filled_fields: Minimum number of fields that must have content
        min_fill_percentage: Minimum percentage of fields that must be filled
        strict_mode: If True, uses stricter validation thresholds

    Returns:
        JSONContentValidator instance
    """
    return JSONContentValidator(
        min_filled_fields=min_filled_fields,
        min_fill_percentage=min_fill_percentage,
        strict_mode=strict_mode
    )


# Example usage
if __name__ == "__main__":
    validator = create_json_validator(min_filled_fields=2, min_fill_percentage=20.0)

    # Test cases
    test_cases = [
        {"name": "John Doe", "age": 45, "diagnosis": "Diabetes"},
        {"name": "unknown", "age": None, "diagnosis": ""},
        {"name": "", "age": "", "diagnosis": ""},
        {"name": "Jane", "age": None, "diagnosis": None},
        None,
        {},
    ]

    for i, test_data in enumerate(test_cases):
        result = validator.validate(test_data)
        print(f"\nTest {i+1}: {test_data}")
        print(f"  {validator.get_validation_summary(result)}")
        if not result.is_valid:
            print(f"  Empty fields: {result.empty_fields}")
