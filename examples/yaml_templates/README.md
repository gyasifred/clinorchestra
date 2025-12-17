# YAML Templates for Clinorchestra

These YAML templates provide blueprints for creating your own functions, patterns, and extras for clinical data extraction tasks.

## Overview

Clinorchestra accepts YAML imports for user-friendly configuration, which are automatically converted to JSON for internal storage and execution. These templates show you how to structure your own YAMLs for any clinical task.

## Files

- **functions_example.yaml** - Complete guide for creating custom functions
- **patterns_example.yaml** - Guide for text normalization patterns
- **extras_example.yaml** - Guide for clinical knowledge and guidelines

## Quick Start

### 1. Choose a Template

Copy the appropriate template file and modify it for your needs.

### 2. Modify the YAML

Edit the copied YAML file with your specific clinical task requirements.

### 3. Upload via UI

Upload your YAML file through the Clinorchestra UI:
- **Functions Tab** → "Import Functions" → Select your `functions.yaml`
- **Patterns Tab** → "Import Patterns" → Select your `patterns.yaml`
- **Extras Tab** → "Import Extras" → Select your `extras.yaml`

### 4. Files are Automatically Converted and Saved

When you register via UI, Clinorchestra:
1. Accepts your YAML imports (user-friendly format)
2. Converts YAML to JSON (required for execution)
3. Saves as JSON files in:
   - `functions/*.json`
   - `patterns/*.json`
   - `extras/*.json`

## YAML Format Requirements

### Functions

```yaml
name: calculate_bmi
description: "Calculate Body Mass Index"
enabled: true

code: |
  def calculate_bmi(weight_kg, height_m):
      if height_m <= 0:
          return None
      return round(weight_kg / (height_m ** 2), 2)

parameters:
  weight_kg:
    type: number
    description: "Weight in kilograms"
    required: true

returns: "BMI value (kg/m²)"
```

### Patterns

```yaml
name: normalize_blood_pressure
pattern: '\b(BP|bp)\s*[=:]?\s*(\d+)\s*/\s*(\d+)\b'
replacement: 'BP \2/\3'
description: "Standardize BP notation"
enabled: true
```

### Extras

```yaml
id: aspen_criteria
name: "ASPEN Pediatric Malnutrition Criteria"
type: criteria
enabled: true

content: |
  ASPEN Pediatric Malnutrition requires ≥2 of:
  1. Insufficient energy intake
  2. Weight loss or deceleration
  3. Loss of muscle/subcutaneous fat
  4. Edema

keywords:
  - malnutrition
  - ASPEN
  - pediatric

metadata:
  category: malnutrition
  priority: high
```

## Important Notes

### YAML Import, JSON Storage

Clinorchestra accepts **YAML imports** for ease of use, but stores configurations as **JSON** internally:
- Import files: Use `.yaml` extension for your import files
- Storage files: System automatically creates `.json` files for internal use
- Use proper YAML syntax (indentation matters!)
- Multi-line content uses `|` after the key
- System handles conversion automatically

### Multi-Document YAML

You can define multiple items in one YAML file using `---` separator:

```yaml
name: function1
code: |
  def function1():
      return "hello"
---
name: function2
code: |
  def function2():
      return "world"
```

### Testing Your YAML

Before uploading, validate your YAML syntax:
```bash
python3 -c "import yaml; yaml.safe_load(open('your_file.yaml'))"
```

## Common Clinical Tasks

### Malnutrition Assessment

See `functions_example.yaml` for:
- `calculate_pediatric_nutrition_status`
- `calculate_zscore`
- `interpret_zscore_malnutrition`

See `extras_example.yaml` for:
- ASPEN Pediatric Malnutrition Criteria
- WHO Z-Score Classification
- Growth Velocity Assessment

### Cognitive Assessment

See `functions_example.yaml` for:
- Clinical Dementia Rating (CDR) calculation
- MMSE scoring

See `extras_example.yaml` for:
- CDR Assessment Methodology
- MMSE Scoring and Interpretation

### Cardiovascular

See `functions_example.yaml` for:
- Mean Arterial Pressure (MAP) calculation

See `patterns_example.yaml` for:
- Blood pressure normalization
- Heart rate standardization

## Creating Your Own Task

1. **Identify your clinical domain** (e.g., nephrology, oncology)

2. **Define required calculations** → Create functions YAML
   - What calculations does the LLM struggle with?
   - What validations are needed?
   - What conversions between units?

3. **Define text normalization needs** → Create patterns YAML
   - What text formats need standardizing?
   - What abbreviations need expanding?
   - What units need normalizing?

4. **Gather clinical knowledge** → Create extras YAML
   - What diagnostic criteria apply?
   - What reference ranges are needed?
   - What guidelines should LLM know?

5. **Test incrementally**
   - Upload one YAML at a time
   - Test extraction with sample clinical notes
   - Refine based on results

## Best Practices

### Functions
- Keep functions focused on single tasks
- Include comprehensive parameter validation
- Return structured data (dicts) for complex results
- Add docstrings explaining purpose and usage

### Patterns
- Test regex patterns thoroughly before deploying
- Order patterns from specific to general
- Use word boundaries (`\b`) to avoid partial matches
- Document what each pattern does

### Extras
- Include authoritative references and citations
- Use clear hierarchical structure
- Tag with comprehensive keywords
- Set appropriate priority levels

## Troubleshooting

### YAML Syntax Errors
- Check indentation (use spaces, not tabs)
- Ensure proper quoting of strings with special chars
- Use `|` for multi-line blocks
- Validate with `yamllint` or online validator

### Functions Not Appearing
- Check `enabled: true` is set
- Verify function name matches code definition
- Check for syntax errors in code block
- Review logs for loading errors

### Patterns Not Matching
- Test regex with online tool (regex101.com)
- Check escape sequences (use raw strings)
- Verify enabled status
- Check pattern application order

### Extras Not Triggered
- Verify keywords match schema/task terms
- Check enabled status
- Ensure metadata category is set
- Test keyword matching logic

## Support

For more examples and documentation, see:
- `functions_example.yaml` - Comprehensive function examples
- `patterns_example.yaml` - Pattern creation guide
- `extras_example.yaml` - Clinical knowledge examples

These templates cover the most common clinical scenarios and provide a foundation for creating task-specific configurations for any clinical domain.
