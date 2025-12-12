# Task-Specific YAML Configurations

This directory contains pre-configured YAML files for common clinical tasks, converted from the original JSON configurations.

## Available Tasks

### Malnutrition Assessment
- **malnutrition_functions.yaml** (13 functions) - Growth calculations, z-scores, percentiles
- **malnutrition_patterns.yaml** (10 patterns) - Malnutrition text normalization
- **malnutrition_extras.yaml** (90 extras) - ASPEN criteria, WHO standards, growth guidelines

### ADRD/Cognitive Assessment
- **adrd_functions.yaml** (3 functions) - CDR, MMSE scoring and interpretation
- **adrd_patterns.yaml** (12 patterns) - Cognitive assessment text normalization
- **adrd_extras.yaml** (22 extras) - NIA-AA criteria, biomarker guidelines, diagnostic criteria

### MIMIC-IV/Critical Care
- **mimic_iv_functions.yaml** (3 functions) - SOFA, APACHE, AKI calculations
- **mimic_iv_extras.yaml** (20 extras) - Critical care criteria and guidelines

### Shared/Common
- **shared_functions.yaml** (25 functions) - Common utilities (BMI, age, unit conversions)
- **shared_patterns.yaml** (149 patterns) - Common text normalization (vitals, labs, dates)
- **shared_extras.yaml** (53 extras) - Common medical knowledge and abbreviations

## How to Use

### 1. Import via UI

**For Functions:**
1. Go to **Functions Tab** in Clinorchestra UI
2. Click **"Import Functions"**
3. Select the YAML file for your task (e.g., `malnutrition_functions.yaml`)
4. Click Import
5. Functions will be saved to `functions/` directory as individual `.yaml` files

**For Patterns:**
1. Go to **Patterns Tab**
2. Click **"Import Patterns"**
3. Select the YAML file (e.g., `shared_patterns.yaml`)
4. Click Import
5. Patterns will be saved to `patterns/` directory as individual `.yaml` files

**For Extras:**
1. Go to **Extras Tab**
2. Click **"Import Extras"**
3. Select the YAML file (e.g., `adrd_extras.yaml`)
4. Click Import
5. Extras will be saved to `extras/` directory as individual `.yaml` files

### 2. Recommended Import Order

For each clinical task, import in this order:

1. **Shared resources first** (if applicable):
   ```
   shared_functions.yaml
   shared_patterns.yaml
   shared_extras.yaml
   ```

2. **Task-specific resources**:
   ```
   <task>_functions.yaml
   <task>_patterns.yaml
   <task>_extras.yaml
   ```

### 3. Example: Setting up Malnutrition Task

```bash
# Import shared utilities (optional but recommended)
1. Import: shared_functions.yaml
2. Import: shared_patterns.yaml
3. Import: shared_extras.yaml

# Import malnutrition-specific configs
4. Import: malnutrition_functions.yaml
5. Import: malnutrition_patterns.yaml
6. Import: malnutrition_extras.yaml
```

Now you have:
- 13 malnutrition functions + 25 shared functions = 38 total functions
- 10 malnutrition patterns + 149 shared patterns = 159 total patterns
- 90 malnutrition extras + 53 shared extras = 143 total extras

### 4. Example: Setting up ADRD Task

```bash
# Import shared utilities
1. Import: shared_functions.yaml
2. Import: shared_patterns.yaml
3. Import: shared_extras.yaml

# Import ADRD-specific configs
4. Import: adrd_functions.yaml
5. Import: adrd_patterns.yaml
6. Import: adrd_extras.yaml
```

## What Happens When You Import

1. **UI validates the YAML** - Checks syntax and structure
2. **Registers each item** - Adds to the appropriate manager (FunctionRegistry, RegexPreprocessor, ExtrasManager)
3. **Saves individual YAML files** - Each item gets its own `.yaml` file:
   - Functions → `functions/<function_name>.yaml`
   - Patterns → `patterns/<pattern_name>.yaml`
   - Extras → `extras/<extra_id>.yaml`
4. **Ready to use** - Items are immediately available for extraction tasks

## File Format

All YAML files contain a list of items:

```yaml
# Functions
- name: calculate_bmi
  code: |
    def calculate_bmi(weight_kg, height_m):
        ...
  parameters:
    weight_kg:
      type: number
      ...

# Patterns
- name: normalize_bp
  pattern: '\b(BP)\s*(\d+)/(\d+)\b'
  replacement: 'BP \2/\3'
  description: "..."
  enabled: true

# Extras
- id: aspen_criteria
  name: "ASPEN Criteria"
  type: criteria
  content: |
    ...
  keywords:
    - malnutrition
    - aspen
  metadata:
    category: malnutrition
```

## Customization

You can:
1. **Edit these files** before importing (add/remove items, modify code)
2. **Create your own** task-specific YAML using these as templates
3. **Mix and match** - Import only the items you need
4. **Re-import** - Importing again will update existing items

## Task Breakdown

### Malnutrition (13 functions)
- `calculate_pediatric_nutrition_status` - Main assessment
- `calculate_zscore` - CDC/WHO z-score calculations
- `interpret_zscore_malnutrition` - WHO/ASPEN interpretation
- `zscore_to_percentile` - Convert z-score to percentile
- `percentile_to_zscore` - Convert percentile to z-score
- `interpret_albumin_malnutrition` - Albumin assessment
- `calculate_growth_velocity` - Growth tracking
- `calculate_growth_percentile` - Percentile calculation
- `kg_to_lbs`, `lbs_to_kg` - Weight conversions
- `calculate_bmi` - BMI calculation
- `cm_to_inches` - Height conversion
- `calculate_age_months` - Age calculation

### ADRD (3 functions)
- `calculate_cdr_severity` - CDR global score interpretation
- `assess_functional_independence` - Functional status assessment
- `calculate_vascular_risk_score` - Vascular risk calculation

### MIMIC-IV (3 functions)
- `calculate_sofa_score` - Sequential Organ Failure Assessment
- `calculate_kdigo_aki_stage` - AKI staging
- `calculate_curb65` - Pneumonia severity

### Shared (25 functions)
- Age calculations (years, months, days between)
- Unit conversions (weight, height, temperature)
- Vital sign calculations (MAP, anion gap, corrected calcium)
- Body metrics (BMI, BSA, ideal body weight)
- Cardiovascular scores (CHADS-VASc, HEART)
- And more...

## Next Steps

After importing:
1. Go to **Processing Tab**
2. Load a clinical note
3. Select your extraction schema
4. Run extraction - your imported functions/patterns/extras will be used automatically!

## Troubleshooting

**Import fails with "Invalid YAML":**
- Check YAML syntax with a validator
- Ensure proper indentation (use spaces, not tabs)
- Verify all required fields are present

**Functions not appearing:**
- Check that `enabled: true` is set
- Refresh the functions list in UI
- Check logs for registration errors

**Items already exist:**
- Importing will update existing items with the same name/id
- Or use UI to delete existing items first

## Support

For creating your own task-specific YAMLs, see:
- `examples/yaml_templates/` - Comprehensive templates and guides
- `examples/yaml_templates/README.md` - Detailed documentation

These pre-configured YAMLs give you a complete starting point for malnutrition, ADRD, and critical care tasks!
