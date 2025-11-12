# Malnutrition Classification Prompts

**Purpose**: Binary classification of malnutrition status from clinical notes with evidence-based reasoning.

## Overview

This prompt set guides the LLM to:
1. Extract clinical evidence of malnutrition from patient notes
2. Apply evidence-based diagnostic criteria (ASPEN/AND guidelines)
3. Classify patients as "Malnourished" or "Not Malnourished"
4. Provide structured reasoning with evidence summary and justification

## Files Included

### 1. `main_prompt.txt`
**Comprehensive extraction prompt** with detailed instructions:
- Complete malnutrition diagnostic criteria
- Step-by-step extraction guidance
- Clinical examples and edge cases
- Evidence-based assessment framework

**Use for**: Primary extraction with full guidance (recommended for complex cases)

### 2. `minimal_prompt.txt`
**Concise fallback prompt** for retry scenarios:
- Key diagnostic criteria only
- Direct classification instructions
- Minimal guidance

**Use for**:
- Retry attempts (when main prompt fails)
- High-volume processing (faster, less tokens)
- Simple/clear-cut cases

### 3. `refinement_prompt.txt`
**RAG-enhanced refinement prompt**:
- Validates classification against retrieved guidelines
- Incorporates evidence-based standards
- Refines reasoning with clinical context

**Use for**:
- Stage 4 RAG refinement (STRUCTURED mode)
- Improving classification accuracy with guidelines
- Adding guideline citations to reasoning

### 4. `schema.json`
**Structured output schema**:
```json
{
  "reasoning": {
    "evidence_summary": "Clinical findings relevant to malnutrition",
    "justification": "How evidence supports the classification"
  },
  "malnutrition_status": "Malnourished" or "Not Malnourished"
}
```

## Clinical Criteria Used

Based on ASPEN/AND malnutrition diagnostic criteria:

1. **Weight Loss**
   - ≥5% in 1 month
   - ≥7.5% in 3 months
   - ≥10% in 6 months

2. **Low BMI**
   - Adults: <18.5 kg/m²
   - Pediatrics: <5th percentile

3. **Muscle Wasting**
   - Physical exam findings
   - Functional decline

4. **Reduced Fat Stores**
   - Visual assessment
   - Physical signs

5. **Fluid Accumulation**
   - Edema, ascites masking weight loss

6. **Reduced Food Intake**
   - ≤50% intake for ≥1 week

7. **Disease/Inflammation**
   - Acute or chronic illness
   - Inflammatory conditions

## How to Use in ClinOrchestra

### Option 1: Load Prompts in UI

1. Navigate to **Prompt Configuration** tab
2. Load `main_prompt.txt` → Main Prompt
3. Load `minimal_prompt.txt` → Minimal Prompt (optional)
4. Load `refinement_prompt.txt` → RAG Refinement Prompt (if using RAG)
5. Load `schema.json` → JSON Schema

### Option 2: Create Configuration File

See `example_config_malnutrition_classification.json` for a complete configuration example.

### Option 3: Command Line

```bash
# Create config from these prompts
python3 -m clinorchestra.scripts.create_config \
    --main-prompt examples/malnutrition_classification/main_prompt.txt \
    --minimal-prompt examples/malnutrition_classification/minimal_prompt.txt \
    --schema examples/malnutrition_classification/schema.json \
    --output malnutrition_config.json
```

## Recommended Setup

### For STRUCTURED Mode (Default)
```
✓ Use main_prompt.txt for Stage 1 & 3
✓ Use minimal_prompt.txt as fallback (retry)
✓ Use refinement_prompt.txt for Stage 4 (if RAG enabled)
✓ Load malnutrition diagnostic guidelines as RAG documents
```

### For ADAPTIVE Mode
```
✓ Use main_prompt.txt as system context
✓ Enable function calling for BMI/z-score calculations if needed
✓ Max iterations: 5-10 (for complex reasoning chains)
```

## RAG Documents (Optional but Recommended)

Enhance accuracy by loading these as RAG documents:
- ASPEN/AND Consensus Malnutrition Diagnostic Criteria
- WHO Malnutrition Guidelines
- Academy of Nutrition and Dietetics Practice Guidelines
- Clinical Nutrition Guidelines (relevant specialty)

## Expected Performance

**Accuracy Considerations**:
- Clear cases (obvious malnutrition or well-nourished): >95% accuracy
- Borderline cases: Requires clinical judgment, benefit from RAG
- Complex cases (fluid overload, sarcopenic obesity): Reasoning quality is critical

**Processing Speed**:
- Main prompt: ~2000 tokens (moderate speed)
- Minimal prompt: ~500 tokens (fast)
- With RAG refinement: +30-40% processing time but higher accuracy

## Example Use Cases

### Use Case 1: Cancer Patients
**Input**: Oncology notes with weight loss, poor intake
**Expected**: High sensitivity for malnutrition detection
**Recommendation**: Use with RAG (cancer-specific nutrition guidelines)

### Use Case 2: General Medicine
**Input**: Hospital admission notes, primary care visits
**Expected**: Balanced sensitivity/specificity
**Recommendation**: Main prompt sufficient, RAG optional

### Use Case 3: Pediatrics
**Input**: Growth charts, pediatric assessments
**Expected**: Age-appropriate criteria (BMI percentiles, z-scores)
**Recommendation**: Use with calculate_zscore function for accurate assessment

### Use Case 4: Critical Care
**Input**: ICU notes with fluid shifts, acute illness
**Expected**: Complex cases, fluid masking weight loss
**Recommendation**: Use with RAG (critical care nutrition guidelines)

## Output Interpretation

### "Malnourished"
Patient meets ≥2 clinical criteria with strong evidence:
- Actionable: Consider nutrition consult, intervention
- Document: Evidence summary provides clinical justification

### "Not Malnourished"
Insufficient evidence OR explicitly well-nourished:
- Actionable: Continue monitoring if at risk
- Document: Reasoning explains why criteria not met

## Limitations

1. **Document Quality**: Accuracy depends on note completeness
2. **Temporal Context**: Weight changes need timeframes
3. **Fluid Status**: May miss fluid-masked malnutrition
4. **Clinical Judgment**: Edge cases may need human review

## Validation Recommendations

1. **Test on Labeled Dataset**: Validate accuracy on your data
2. **Review Edge Cases**: Human review for borderline classifications
3. **Monitor Reasoning**: Check evidence_summary quality
4. **Iterate Prompts**: Refine based on error analysis

## Version History

- **v1.0.0**: Initial release
  - Main, minimal, and refinement prompts
  - Evidence-based ASPEN/AND criteria
  - Structured reasoning output

## Author

Frederick Gyasi (gyasi@musc.edu)
Medical University of South Carolina
Biomedical Informatics Center

## License

Part of ClinOrchestra v1.0.0
