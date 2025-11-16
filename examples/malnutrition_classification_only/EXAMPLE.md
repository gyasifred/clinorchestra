# Malnutrition Classification Examples

## Example 1: Single-Point Assessment

### Clinical Note
```
3/15/2025 - Emergency Department Visit

Chief Complaint: Vomiting and poor feeding

6-year-old male presents with 2 weeks of vomiting 4-5 times daily and decreased oral intake. Parents report child eating only 30-40% of usual meals. No prior medical records available - family recently moved from another state.

Vital Signs:
- Weight: 16.2 kg
- Height: 110 cm
- BMI: 13.4 kg/m²

Physical Exam: Appears thin, alert. No prior measurements documented for comparison. Parents report child has always been "small for age" but seems to have lost weight recently based on clothing fit.

Assessment: Acute gastroenteritis with dehydration and poor intake
```

### Classification Output (Main Prompt)

```json
{
  "assessment_type": "Single-point",
  "assessment_justification": "Only one encounter documented (3/15/25). No prior growth measurements available despite 6 years of age. Family recently relocated, prior records not obtained.",
  "criteria_satisfied": [
    {
      "criterion": "WHO BMI-for-age z-score classification",
      "specific_documentation": "Moderate acute malnutrition per WHO criterion z-score -3 to -2 (BMI-for-age z-score -2.1 for 6-year-old male on 3/15/25, calculated from BMI 13.4 kg/m², <3rd percentile)",
      "severity": "Moderate"
    },
    {
      "criterion": "ASPEN anthropometric deficit",
      "specific_documentation": "Moderate malnutrition per ASPEN anthropometric criterion z-score -2 to -2.9 (BMI-for-age z-score -2.1)",
      "severity": "Moderate"
    },
    {
      "criterion": "ASPEN inadequate intake",
      "specific_documentation": "Inadequate intake per ASPEN criterion <50% estimated needs for ≥1 week (documented intake 30-40% of usual meals for 2 weeks per parent report)",
      "severity": "N/A"
    }
  ],
  "criteria_not_met": [
    "ASPEN physical findings - exam notes 'appears thin' but lacks specific documentation of temporal wasting, muscle atrophy, rib prominence, or fat loss"
  ],
  "missing_data": [
    "ASPEN growth velocity criterion cannot be assessed - only single measurement available. CRITICAL: Obtain prior growth records from previous provider to establish baseline and calculate velocity. If unavailable, establish serial measurements at 2-week intervals to monitor trajectory during nutritional rehabilitation.",
    "Detailed physical exam needed documenting presence/absence of muscle wasting (temporal area, extremities), subcutaneous fat loss (orbital, triceps, ribs), and edema to complete ASPEN assessment.",
    "Laboratory assessment recommended: CBC, CMP, albumin, prealbumin to assess nutritional status and severity. Not required for ASPEN diagnosis but helpful for treatment planning.",
    "Dietary history and 24-hour recall needed to quantify caloric/protein intake more precisely than parent estimate of '30-40%'."
  ],
  "malnutrition_status": "Present",
  "severity": "Moderate",
  "aspen_indicator_count": "3/4 indicators met (exceeds threshold of 2+ for ASPEN malnutrition diagnosis). Indicators present: (1) Anthropometric deficit z-score -2.1, (2) Inadequate intake <50% for 2 weeks, (3) Likely velocity decline based on parent report of weight loss though cannot calculate without baseline. Physical findings not specifically documented.",
  "confidence": "Moderate",
  "clinical_reasoning": "Single-point assessment with significant limitations due to lack of prior measurements. Current anthropometrics clearly abnormal with BMI-for-age z-score -2.1, meeting both WHO moderate malnutrition criterion (z-score -3 to -2) and ASPEN moderate anthropometric deficit criterion (z-score -2 to -2.9). Inadequate intake over 2 weeks meets ASPEN criterion. Parent report of recent weight loss and clothing fitting more loosely suggests acute decline, but growth velocity cannot be calculated without baseline measurements - this is a critical limitation of single-point assessment. While parents report child has 'always been small,' current z-score of -2.1 indicates malnutrition by WHO/ASPEN standards regardless of chronic vs. acute etiology. ASPEN diagnosis established with 3/4 indicators (would be 2/4 if velocity cannot be inferred). Confidence limited by: (1) inability to assess actual velocity/trajectory, (2) incomplete physical exam documentation, (3) lack of laboratory correlation. URGENT: Obtain prior records to differentiate chronic vs. acute malnutrition, as this significantly impacts treatment approach and expected recovery timeline. If chronic, may represent failure to thrive requiring comprehensive evaluation. If acute, expect more rapid response to nutritional rehabilitation once vomiting controlled."
}
```

---

## Example 2: Longitudinal Assessment (Same Patient - Follow-up)

### Clinical Notes (Serial)

```
1/15/2025 - Well Child Visit
6-year-old male, routine check-up
Weight: 18.5 kg (25th percentile, BMI-for-age z-score -0.67)
Height: 112 cm (20th percentile)
BMI: 14.8 kg/m²
Physical: Well-nourished appearance, active, playful
Intake: Parents report good appetite, eating variety of foods

2/14/2025 - Urgent Care Visit
Chief Complaint: Vomiting x 1 week
Weight: 17.3 kg (10th percentile, BMI-for-age z-score -1.28)
Height: 112 cm
BMI: 13.8 kg/m²
Intake: Reduced to 60-70% due to nausea
Diagnosis: Viral gastroenteritis
Plan: Supportive care, follow-up if not improving

3/15/2025 - Emergency Department Visit
Chief Complaint: Persistent vomiting, severe dehydration
Weight: 16.2 kg (3rd percentile, BMI-for-age z-score -2.1)
Height: 112 cm (stable)
BMI: 12.9 kg/m²
Intake: 30-40% of baseline for past 2 weeks
Physical: Appears thin, temporal wasting noted, prominent ribs, decreased muscle mass in extremities
Labs: Albumin 3.1 g/dL (low), prealbumin 15 mg/dL (low)
```

### Classification Output (Main Prompt)

```json
{
  "assessment_type": "Longitudinal",
  "assessment_justification": "Three documented encounters over 2 months (1/15/25, 2/14/25, 3/15/25) with complete anthropometric measurements at each visit, allowing calculation of growth velocity and percentile trajectory",
  "criteria_satisfied": [
    {
      "criterion": "WHO BMI-for-age z-score classification",
      "specific_documentation": "Moderate acute malnutrition per WHO criterion z-score -3 to -2 (current BMI-for-age z-score -2.1 on 3/15/25, representing acute decline from normal range z-score -0.67 on 1/15/25)",
      "severity": "Moderate"
    },
    {
      "criterion": "ASPEN anthropometric deficit",
      "specific_documentation": "Moderate malnutrition per ASPEN anthropometric criterion z-score -2 to -2.9 (current BMI-for-age z-score -2.1 on 3/15/25)",
      "severity": "Moderate"
    },
    {
      "criterion": "ASPEN growth velocity deceleration",
      "specific_documentation": "Mild-to-moderate malnutrition per ASPEN velocity criterion decline approaching 2 z-scores (BMI-for-age z-score declined from -0.67 on 1/15/25 to -1.28 on 2/14/25 to -2.1 on 3/15/25, total decline of 1.43 z-scores over 59 days at rate of 0.024 z-scores/day, with percentile crossing from 25th→10th→3rd percentile). Weight loss 2.3 kg (12.4% of baseline) over 59 days represents severe acute malnutrition velocity by rate criteria.",
      "severity": "Moderate"
    },
    {
      "criterion": "ASPEN inadequate intake",
      "specific_documentation": "Inadequate intake per ASPEN criterion <50% estimated needs for ≥1 week (intake declined progressively: 100% at baseline 1/15, 60-70% at 2/14, 30-40% at 3/15, with intake <50% documented for final 2 weeks)",
      "severity": "N/A"
    },
    {
      "criterion": "ASPEN physical findings",
      "specific_documentation": "Muscle wasting and fat loss present per ASPEN criterion (temporal wasting, prominent ribs, decreased muscle mass in extremities documented on exam 3/15/25, contrasting with 'well-nourished appearance' on 1/15/25)",
      "severity": "N/A"
    }
  ],
  "criteria_not_met": [],
  "missing_data": [
    "Mid-upper arm circumference (MUAC) would provide additional anthropometric data to quantify muscle/fat loss objectively",
    "Triceps skinfold thickness could quantify subcutaneous fat depletion",
    "Functional assessment (grip strength, activity level, school performance) would help assess impact of malnutrition on daily functioning"
  ],
  "malnutrition_status": "Present",
  "severity": "Moderate",
  "aspen_indicator_count": "5/4 indicators met (far exceeds threshold of 2+ for ASPEN malnutrition diagnosis). All indicators present with multiple anthropometric measures: (1) Anthropometric deficit z-score -2.1, (2) Growth velocity decline 1.43 z-scores over 59 days, (3) Weight loss 12.4% baseline, (4) Inadequate intake <50% for 2 weeks, (5) Physical findings of muscle/fat wasting. Low albumin (3.1) and prealbumin (15) provide laboratory correlation.",
  "confidence": "High",
  "clinical_reasoning": "Longitudinal assessment with excellent serial data demonstrates clear acute malnutrition with rapid nutritional decline over 2 months. Patient started at 25th percentile with normal BMI-for-age z-score -0.67 on 1/15/25, representing adequate baseline nutrition. Progressive deterioration: 10th percentile (z-score -1.28) by 2/14/25, then 3rd percentile (z-score -2.1) by 3/15/25, crossing two major percentile bands (25th→10th→3rd) and declining 1.43 z-scores. This rate of decline (0.024 z-scores/day) with 12.4% weight loss over 59 days represents acute severe malnutrition velocity even though absolute z-score is in moderate range. ALL ASPEN indicators comprehensively met: (1) Moderate anthropometric deficit by current z-score, (2) Velocity decline approaching ASPEN moderate threshold of 2 z-scores, (3) Severe inadequate intake <50% for 2 weeks with progressive decline from baseline 100%, (4) Clinical evidence of muscle/fat wasting on exam, (5) Laboratory evidence with low albumin 3.1 g/dL and prealbumin 15 mg/dL. Etiology clearly acute illness-related (gastroenteritis) with ongoing symptoms despite initial supportive care, indicating need for escalated nutritional intervention. High confidence based on: complete serial anthropometric data allowing precise velocity calculation, documented clinical progression with physical exam changes, laboratory correlation, and clear temporal relationship to illness onset. This demonstrates CRITICAL difference between single-point and longitudinal assessment: single measurement on 3/15 alone shows moderate malnutrition, but serial data reveals ACUTE severe velocity of decline requiring urgent aggressive nutritional rehabilitation and investigation for persistent gastrointestinal pathology (not expected with simple viral gastroenteritis). The velocity and severity of decline warrant inpatient admission for NG/NJ nutritional support if oral intake cannot be immediately restored, plus evaluation for underlying GI disorders (celiac, IBD, malabsorption) given atypical prolonged course."
}
```

### Classification Output (Minimal Prompt)

```json
{
  "malnutrition_status": "Malnourished",
  "reasoning": "Assessment type: Longitudinal (3 visits over 59 days with serial measurements). Criteria met: (1) WHO moderate malnutrition z-score -3 to -2 (current BMI-for-age -2.1), (2) ASPEN moderate anthropometric deficit z-score -2 to -2.9 (-2.1), (3) ASPEN velocity decline 1.43 z-scores from baseline -0.67 to -2.1 over 2 months, (4) ASPEN inadequate intake <50% for 2 weeks, (5) ASPEN physical findings muscle/fat wasting. ASPEN indicators: 5/4 met (exceeds 2+ threshold). Severity: Moderate by absolute z-score, but acute severe velocity (12.4% weight loss, percentile crossing 25th→3rd). High confidence - complete serial data."
}
```

---

## Key Differences Demonstrated

### Single-Point Assessment:
- ✓ Can classify current status (z-score -2.1 = moderate malnutrition)
- ✓ Can identify inadequate intake if documented
- ✗ **Cannot assess velocity** - major limitation noted
- ✗ Cannot determine if chronic vs. acute
- ✗ Cannot calculate percentile trajectory
- → **Moderate confidence** due to limitations
- → **Urgent need for prior records or serial measurements**

### Longitudinal Assessment:
- ✓ Can classify current status (same z-score -2.1)
- ✓ Can assess velocity (1.43 z-score decline over 59 days)
- ✓ Can determine acute onset (started at normal nutrition)
- ✓ Can calculate precise trajectory (25th→10th→3rd percentile)
- ✓ Can assess rate of decline (0.024 z-scores/day)
- → **High confidence** with complete data
- → **Changes clinical urgency** - acute severe velocity requires immediate intervention

## Documentation Specificity Examples

### ❌ Generic (Incorrect)
```
"Patient has malnutrition based on ASPEN criteria"
"Growth velocity is declining"
"BMI is low"
```

### ✅ Specific (Correct)
```
"Moderate malnutrition per ASPEN anthropometric criterion z-score -2 to -2.9 (BMI-for-age z-score -2.1 measured on 3/15/25)"

"Moderate malnutrition per ASPEN velocity criterion decline approaching 2 z-scores (declined from -0.67 to -2.1 over 59 days, decline of 1.43 z-scores)"

"Moderate acute malnutrition per WHO criterion z-score -3 to -2 (current z-score -2.1, <3rd percentile)"
```

## ASPEN Indicator Counting

### Example: Clear Documentation
```
ASPEN Indicator Count: 5/4 (exceeds threshold of 2+ for diagnosis)

Indicators Present:
1. ✓ Anthropometric deficit: BMI-for-age z-score -2.1 (moderate per criterion -2 to -2.9)
2. ✓ Growth velocity: Decline 1.43 z-scores over 59 days (moderate per criterion 1-2 z-score decline)
3. ✓ Weight loss: 12.4% over 2 months (severe per criterion >10% in 6 months)
4. ✓ Inadequate intake: <50% for 2 weeks (meets criterion <50% for ≥1 week)
5. ✓ Physical findings: Temporal wasting, rib prominence, decreased muscle mass

Indicators Not Present:
- N/A - all standard indicators documented and met

Diagnostic Threshold: Met (2+ indicators required, 5 present)
```

This comprehensive documentation allows clear audit trail of which specific criteria justified the classification.
