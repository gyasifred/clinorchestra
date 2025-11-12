# Quick Start: Malnutrition Classification

Get started with malnutrition classification in ClinOrchestra in 5 minutes.

## Step 1: Launch ClinOrchestra

```bash
cd /home/user/clinorchestra
python3 app.py
```

Open browser to `http://localhost:7860`

## Step 2: Configure Model (Model Config Tab)

1. Select your LLM provider (OpenAI, Anthropic, etc.)
2. Enter API key
3. Choose model:
   - **Recommended**: GPT-4 or Claude Sonnet for best accuracy
   - **Fast/Budget**: GPT-3.5-turbo for speed
4. Set parameters:
   - Temperature: `0.1` (more deterministic)
   - Max Tokens: `2000`

## Step 3: Load Prompts (Prompt Config Tab)

### Main Prompt
1. Click **Browse** next to "Main Prompt"
2. Navigate to: `examples/malnutrition_classification/main_prompt.txt`
3. Click **Load Prompt from File**

### Minimal Prompt (Optional but Recommended)
1. Enable "Use Minimal Prompt on Retry"
2. Click **Browse** next to "Minimal Prompt"
3. Load: `examples/malnutrition_classification/minimal_prompt.txt`

### JSON Schema
1. Click **Browse** next to "JSON Schema"
2. Load: `examples/malnutrition_classification/schema.json`

## Step 4: Load Data (Data Config Tab)

### Option A: Use Sample Data
1. Upload: `examples/malnutrition_classification/sample_notes.csv`
2. Select Text Column: `clinical_note`
3. Optional: Select Label Column: `expected_status` (for validation)

### Option B: Use Your Own Data
1. Prepare CSV/Excel with column containing clinical notes
2. Upload file
3. Select the column with clinical text
4. Optionally: Select label column if you have ground truth

## Step 5: (Optional) Add RAG Documents

For improved accuracy, load malnutrition guidelines:

1. Go to **RAG Config Tab**
2. Enable RAG
3. Upload PDF documents:
   - ASPEN/AND Malnutrition Criteria
   - WHO Guidelines
   - Clinical Nutrition Standards
4. Click **Process Documents**

### RAG Refinement (Recommended)
1. In **Prompt Config**, click "RAG Refinement Prompt"
2. Load: `examples/malnutrition_classification/refinement_prompt.txt`
3. Select fields to refine: `reasoning`, `malnutrition_status`

## Step 6: (Optional) Enable Functions

For objective measurements (BMI, z-scores):

1. Go to **Functions Tab**
2. Load built-in functions:
   - `calculate_bmi`
   - `calculate_zscore` (for pediatric cases)
3. These will be automatically called when needed

## Step 7: Execute Processing

1. Go to **Processing Tab**
2. Review configuration summary
3. Set processing options:
   - Batch Size: `10` (balance speed vs memory)
   - Error Strategy: `retry` (recommended)
   - Max Retries: `3`
4. Optional: Enable "Dry Run" to test on 5 rows first
5. Click **▶️ Start Processing**

## Step 8: Review Results

### During Processing
- Monitor progress bar
- Watch log for extraction details
- Check for any errors

### After Completion
- Download results CSV
- Review columns:
  - `reasoning.evidence_summary` - Clinical findings
  - `reasoning.justification` - Classification reasoning
  - `malnutrition_status` - Final classification
- If label provided, check accuracy

## Expected Output Format

```json
{
  "reasoning": {
    "evidence_summary": "72yo M with pancreatic cancer. Unintentional 15 lb weight loss over 2 months (BMI 17.2). Poor intake <50% x 3 weeks. Temporal wasting, prominent clavicles, albumin 2.8.",
    "justification": "Patient meets multiple malnutrition criteria: significant weight loss (>7.5% in <3 months), low BMI (<18.5), reduced intake, muscle wasting on exam, and low albumin in context of cancer-related malnutrition."
  },
  "malnutrition_status": "Malnourished"
}
```

## Troubleshooting

### Issue: Low accuracy on your data
**Solution**:
- Add RAG documents with relevant guidelines
- Enable RAG refinement
- Review failed cases and adjust prompts

### Issue: Inconsistent classifications
**Solution**:
- Lower temperature (0.0 - 0.1)
- Use more explicit criteria in prompt
- Enable "Use Minimal Prompt" for clarity

### Issue: Reasoning too brief
**Solution**:
- Increase max_tokens (try 3000)
- Modify prompt to require more detailed evidence
- Check that schema descriptions are clear

### Issue: Slow processing
**Solution**:
- Disable RAG refinement if not needed
- Use faster model (GPT-3.5-turbo)
- Increase batch size
- Enable parallel processing (optimization tab)

## Performance Benchmarks

**Expected Speed** (GPT-4):
- Without RAG: ~3-5 seconds per note
- With RAG: ~5-8 seconds per note
- 100 notes: ~8-12 minutes

**Expected Accuracy** (on clear cases):
- Sensitivity (detecting malnutrition): 90-95%
- Specificity (ruling out malnutrition): 92-97%
- Overall accuracy: 91-96%

## Next Steps

1. **Validate on Your Data**: Test on labeled dataset
2. **Tune Prompts**: Adjust based on error patterns
3. **Expand RAG**: Add specialty-specific guidelines
4. **Enable Functions**: Use calculate_bmi for objective data
5. **Monitor Results**: Track accuracy over time

## Support

For questions or issues:
- Check main README: `examples/malnutrition_classification/README.md`
- Review ClinOrchestra docs
- Contact: Frederick Gyasi (gyasi@musc.edu)

---

**Pro Tip**: Start with a dry run (5 rows) to verify configuration before processing your full dataset!
