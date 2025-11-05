#!/usr/bin/env python3
"""
Create malnutrition-specific extras for z-score and percentile interpretation
Author: Frederick Gyasi (gyasi@musc.edu)
"""

import json
from pathlib import Path
from datetime import datetime

# Create extras directory if it doesn't exist
extras_dir = Path("./extras")
extras_dir.mkdir(parents=True, exist_ok=True)

# Malnutrition-specific extras with z-score knowledge
malnutrition_extras = [
    {
        "name": "Z-Score and Percentile Relationship",
        "type": "reference",
        "content": """Z-SCORE AND PERCENTILE RELATIONSHIP FOR GROWTH ASSESSMENT:

CRITICAL: Lower percentiles correspond to NEGATIVE z-scores

Standard Conversions:
- 50th percentile = z-score 0 (mean/average)
- 25th percentile = z-score -0.67
- 10th percentile = z-score -1.28
- 5th percentile = z-score -1.64 (malnutrition risk threshold)
- 3rd percentile = z-score -1.88 (moderate malnutrition)
- 2nd percentile = z-score -2.0 (severe malnutrition threshold)
- 1st percentile = z-score -2.33
- <1st percentile = z-score < -2.5 (very severe)

Above Mean:
- 75th percentile = z-score +0.67
- 90th percentile = z-score +1.28
- 95th percentile = z-score +1.64 (overweight threshold)
- 97th percentile = z-score +1.88
- 99th percentile = z-score +2.33

INTERPRETATION RULE: If a patient is at a low percentile (e.g., 3rd), the z-score MUST be negative (e.g., -1.88). Never report positive z-scores for low percentiles."""
    },
    {
        "name": "WHO Malnutrition Z-Score Criteria",
        "type": "guideline",
        "content": """WHO MALNUTRITION CLASSIFICATION BY Z-SCORE:

Weight-for-Height or BMI-for-Age:
- z < -3 SD: SEVERE ACUTE MALNUTRITION (SAM)
  * Requires immediate intervention
  * High mortality risk
  * <1st percentile

- -3 ≤ z < -2 SD: MODERATE ACUTE MALNUTRITION (MAM)
  * Nutritional rehabilitation needed
  * 2nd-3rd percentile
  * Risk of progression to SAM

- -2 ≤ z < -1 SD: MILD MALNUTRITION RISK
  * Close monitoring required
  * 3rd-15th percentile
  * Early intervention may prevent progression

- -1 ≤ z ≤ +1 SD: NORMAL RANGE
  * 15th-85th percentile
  * No malnutrition

- z > +2 SD: OVERWEIGHT/OBESITY
  * >97th percentile

Height-for-Age (Stunting - Chronic Malnutrition):
- z < -3 SD: SEVERELY STUNTED
- z < -2 SD: STUNTED (chronic undernutrition)
- z ≥ -2 SD: Normal height

Weight-for-Age (Combined Indicator):
- z < -3 SD: SEVERELY UNDERWEIGHT
- z < -2 SD: UNDERWEIGHT
- z ≥ -2 SD: Normal weight"""
    },
    {
        "name": "ASPEN Pediatric Malnutrition Criteria with Z-Scores",
        "type": "guideline",
        "content": """ASPEN PEDIATRIC MALNUTRITION CRITERIA (Requires 2+ indicators):

Z-Score Based Indicators:

1. WEIGHT-FOR-HEIGHT or BMI-FOR-AGE:
   Mild: z-score -1 to -1.9 (approximately 3rd-15th percentile)
   Moderate: z-score -2 to -2.9 (approximately 0.5th-3rd percentile)
   Severe: z-score ≤ -3 (<0.5th percentile)

2. LENGTH/HEIGHT-FOR-AGE (if chronically malnourished):
   Mild: z-score -1 to -1.9
   Moderate: z-score -2 to -2.9
   Severe: z-score ≤ -3

3. WEIGHT LOSS (compared to baseline):
   Mild: 5% weight loss
   Moderate: 7.5% weight loss
   Severe: 10% weight loss

4. DECELERATION IN WEIGHT GAIN VELOCITY:
   Mild: Decline of 1 z-score
   Moderate: Decline of 2 z-scores
   Severe: Decline of 3 z-scores

5. INADEQUATE NUTRIENT INTAKE:
   Mild: <75% estimated needs for >7 days
   Moderate: <50% estimated needs for >7 days
   Severe: <25% estimated needs for >7 days

Additional Indicators (support diagnosis):
- Physical exam: muscle wasting, subcutaneous fat loss
- Laboratory: low albumin, prealbumin
- Functional: reduced activity, developmental delay

IMPORTANT: Z-score below -2 is a PRIMARY indicator. Document trends over time."""
    },
    {
        "name": "Z-Score Documentation Best Practices",
        "type": "pattern",
        "content": """BEST PRACTICES FOR DOCUMENTING Z-SCORES IN MALNUTRITION ASSESSMENT:

Always Document:
1. The actual z-score value (with sign)
2. The corresponding percentile
3. The measurement type (weight-for-age, height-for-age, BMI-for-age, weight-for-height)
4. Date of measurement
5. Comparison to previous measurements (trend)

Correct Documentation Examples:
✓ "Weight 12.5 kg, weight-for-age z-score -1.88 (3rd percentile)"
✓ "BMI 14.8 kg/m², BMI-for-age z-score -2.1 (2nd percentile), indicating moderate acute malnutrition"
✓ "Height 92 cm, height-for-age z-score -0.67 (25th percentile), within normal range"

Incorrect Documentation (DO NOT USE):
✗ "3rd percentile, z-score +1.88" (wrong sign - should be negative)
✗ "Below average growth, z-score positive" (contradiction)
✗ "Z-score -2, 97th percentile" (inconsistent - -2 ≈ 2nd percentile, not 97th)

Trend Documentation:
- "Weight z-score declined from -1.2 (6 months ago) to -2.1 (today), indicating progression of malnutrition"
- "Height-for-age z-score improved from -2.5 to -1.8 over 3 months with nutritional intervention"

Remember: Negative z-scores = below average = lower percentiles"""
    },
    {
        "name": "Common Z-Score Interpretation Errors",
        "type": "guideline",
        "content": """COMMON ERRORS IN Z-SCORE INTERPRETATION (AVOID THESE):

ERROR 1: Wrong Sign
✗ "Patient at 3rd percentile with z-score +1.88"
✓ "Patient at 3rd percentile with z-score -1.88"
RULE: Low percentiles = negative z-scores

ERROR 2: Inverted Relationship
✗ "Low percentile indicates good growth"
✓ "Low percentile indicates poor growth/malnutrition risk"
RULE: Lower percentile = worse growth

ERROR 3: Inconsistent Pairs
✗ "Z-score -2.5, 95th percentile"
✓ "Z-score -2.5, <1st percentile (severe malnutrition)"
RULE: Negative z-scores pair with low percentiles

ERROR 4: Missing Context
✗ "Z-score -2"
✓ "Weight-for-height z-score -2 (2nd percentile), indicating moderate acute malnutrition per WHO criteria"
RULE: Always specify measurement type and interpretation

ERROR 5: Ignoring Trends
✗ "Single z-score measurement without context"
✓ "Z-score decreased from -1.0 to -2.5 over 3 months, indicating rapid weight loss and nutritional decline"
RULE: Document trajectory and velocity

ERROR 6: Confusing Weight-for-Age with Weight-for-Height
- Weight-for-age: Can be low due to stunting OR wasting
- Weight-for-height or BMI-for-age: Better indicator of ACUTE malnutrition
- Height-for-age: Indicator of CHRONIC malnutrition (stunting)
RULE: Use all three indicators for comprehensive assessment"""
    },
    {
        "name": "Quick Reference - Z-Score Cutoffs",
        "type": "reference",
        "content": """QUICK REFERENCE: CRITICAL Z-SCORE CUTOFFS FOR MALNUTRITION

Immediate Red Flags (Require Urgent Action):
- z < -3 SD: SEVERE malnutrition (<1st percentile)
- z < -2 SD: MODERATE malnutrition (2nd-3rd percentile)

Monitoring Needed:
- z < -1.64 SD: Below 5th percentile (at-risk)
- z < -1 SD: Below 15th percentile (monitor)

Normal Range:
- -1 to +1 SD: 15th-85th percentile (normal)

Overweight Thresholds:
- z > +1.64 SD: Above 95th percentile (overweight)
- z > +2 SD: Above 97th percentile (obesity)

Velocity/Decline Thresholds (ASPEN):
- Decline of 1 z-score: Mild concern
- Decline of 2 z-scores: Moderate concern
- Decline of 3 z-scores: Severe concern

Memory Aid: "2-2-2 Rule"
- Below -2: Moderate problem
- Below -2 in 2 indicators: Malnutrition diagnosis (ASPEN)
- Decline of 2: Moderate severity"""
    }
]

# Write each extra to a separate file
for extra in malnutrition_extras:
    extra_id = f"extra_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    extra_data = {
        'id': extra_id,
        'name': extra['name'],
        'type': extra['type'],
        'content': extra['content'],
        'metadata': {
            'category': 'malnutrition',
            'keywords': ['z-score', 'percentile', 'malnutrition', 'growth', 'WHO', 'ASPEN', 'nutrition'],
            'clinical_area': 'pediatric nutrition'
        },
        'created_at': datetime.now().isoformat()
    }

    # Write to file
    filename = f"{extra_id}_{extra['name'].lower().replace(' ', '_').replace('-', '_')}.json"
    filepath = extras_dir / filename
    with open(filepath, 'w') as f:
        json.dump(extra_data, f, indent=2)

    print(f"✓ Created: {extra['name']}")

print(f"\n✅ Created {len(malnutrition_extras)} malnutrition-specific extras with z-score guidance!")
print("\nExtras cover:")
print("  - Z-score and percentile relationships")
print("  - WHO malnutrition classification")
print("  - ASPEN pediatric criteria")
print("  - Documentation best practices")
print("  - Common interpretation errors")
print("  - Quick reference cutoffs")
