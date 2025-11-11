# Exploratory Data Analysis (EDA) for Publication

## Overview

The `eda_train_test_publication.py` script generates comprehensive, publication-ready descriptive statistics, tables, and visualizations for your MIMIC-IV train and test datasets.

## What It Generates

### 📊 Publication-Ready Tables

#### Table 1: Baseline Characteristics
- Comprehensive demographic and clinical characteristics
- Train vs Test comparison with statistical tests
- Formatted for journal submission
- **Formats**: CSV, HTML, LaTeX

**Contents**:
- Total sample sizes
- Age (mean, SD, median, IQR, range)
- Gender distribution with percentages
- Race/ethnicity distribution with percentages
- Diagnosis category distribution
- Clinical note characteristics (length, word count)
- **P-values** for all comparisons (t-test for continuous, chi-square for categorical)

#### Table 2: Diagnosis Distribution
- Detailed breakdown of all 20 diagnoses
- ICD codes with full diagnosis names
- Category groupings (Cardiovascular, Infectious, etc.)
- Train, test, and total counts with percentages
- **Formats**: CSV, HTML, LaTeX

### 📈 High-Resolution Figures

#### Figure 1: Diagnosis Distribution (PNG + PDF)
- **Panel A**: Diagnosis category comparison (bar chart)
- **Panel B**: Top 10 specific diagnoses (horizontal bar chart)
- Train vs Test side-by-side comparison

#### Figure 2: Demographics (PNG + PDF)
- **Panel A**: Age distribution (histogram overlay)
- **Panel B**: Gender distribution (grouped bar chart)
- **Panel C**: Race/ethnicity distribution (grouped bar chart)
- **Panel D**: Clinical note length distribution (log-scale histogram)

#### Figure 3: Text Complexity (PNG + PDF)
- **Panel A**: Text length by diagnosis category (box plots)
- **Panel B**: Word count distribution (histogram)
- **Panel C**: Text quartile distribution (grouped bar chart)
- **Panel D**: Text length vs age scatter plot

### 📋 Supplementary Materials

#### supplementary_missing_data.csv
- Column-by-column missing data analysis
- Train and test missing counts and percentages
- Only includes columns with missing data

#### supplementary_diagnosis_details.csv
- Detailed statistics for each diagnosis
- Age means by diagnosis (train/test)
- Text length means by diagnosis (train/test)
- Sample sizes for each diagnosis

### 📝 Manuscript Content

#### methods_section_manuscript.txt
- Complete methods section ready for copy/paste
- Includes:
  - Dataset description
  - Study population
  - Sampling strategy
  - Train/test split methodology
  - Baseline characteristics summary
  - Statistical methods
  - Ethical considerations
  - Data availability statement

#### eda_summary_report.txt
- Executive summary of all analyses
- List of all generated files
- Key findings
- Usage instructions
- Reproducibility information

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scipy
```

## Configuration

Edit these paths in the script:

```python
TRAIN_FILE = r"C:\Users\gyasi\Documents\mimic-iv-3.1\outputs\train_dataset_4000.csv"
TEST_FILE = r"C:\Users\gyasi\Documents\mimic-iv-3.1\outputs\test_dataset_1000.csv"
OUTPUT_DIR = r"C:\Users\gyasi\Documents\mimic-iv-3.1\outputs\eda_results"
```

## Usage

### Step 1: Run the Script

```bash
python eda_train_test_publication.py
```

**Runtime**: ~1-2 minutes

### Step 2: Review Outputs

All files will be saved in the `eda_results/` directory:

```
eda_results/
├── table1_baseline_characteristics.csv
├── table1_baseline_characteristics.html
├── table1_baseline_characteristics.tex
├── table2_diagnosis_distribution.csv
├── table2_diagnosis_distribution.html
├── table2_diagnosis_distribution.tex
├── figure1_diagnosis_distribution.png
├── figure1_diagnosis_distribution.pdf
├── figure2_demographics.png
├── figure2_demographics.pdf
├── figure3_text_complexity.png
├── figure3_text_complexity.pdf
├── supplementary_missing_data.csv
├── supplementary_diagnosis_details.csv
├── methods_section_manuscript.txt
└── eda_summary_report.txt
```

## Output Examples

### Table 1 Format

```
Characteristic                Train (n=4,000)         Test (n=1,000)          P-value
Total, N                      4,000                   1,000                   -
Age, years
  Mean (SD)                   64.2 (17.3)             64.5 (17.1)             0.645
  Median [IQR]                66 [53-78]              67 [53-78]
  Range                       18-99                   18-98
Gender, n (%)                                                                 0.823
  F                           1820 (45.5%)            452 (45.2%)
  M                           2150 (53.8%)            542 (54.2%)
  U                           30 (0.8%)               6 (0.6%)
Race/Ethnicity, n (%)                                                         0.912
  White                       2400 (60.0%)            605 (60.5%)
  Black/African American      800 (20.0%)             198 (19.8%)
  Asian                       320 (8.0%)              79 (7.9%)
  ...
```

### Figure Quality

- **Resolution**: 300 DPI (publication quality)
- **Formats**:
  - PNG for manuscripts and presentations
  - PDF for vector graphics (scalable, no pixelation)
- **Style**: Clean, professional, publication-ready
- **Color palette**: Colorblind-friendly

### Methods Section Preview

```
DATA PREPARATION AND STUDY POPULATION

Dataset Description

We utilized the Medical Information Mart for Intensive Care IV (MIMIC-IV) database,
a freely accessible critical care database containing de-identified health data from
patients admitted to the Beth Israel Deaconess Medical Center between 2008 and 2019.

Study Population

From the complete MIMIC-IV dataset (N=5,000), we selected 5,000 cases representing
the 20 most common primary diagnoses. Cases were selected using stratified sampling
to ensure balanced representation across multiple dimensions...
```

## Statistical Tests Performed

### Continuous Variables
- **Independent samples t-test**: Age, text length, word count
- Reports: mean, SD, median, IQR, p-value

### Categorical Variables
- **Chi-square test**: Gender, race, diagnosis category
- Reports: counts, percentages, p-value

### Non-Parametric Tests
- **Mann-Whitney U test**: For non-normal distributions (e.g., text length)

## Using Outputs in Your Manuscript

### For Main Text

1. **Table 1** → Insert as "Table 1: Baseline Characteristics"
   - Use LaTeX version for LaTeX manuscripts
   - Use HTML version for Word manuscripts (paste from browser)

2. **Figures 1-2** → Main figures showing data distribution
   - Use PDF versions for LaTeX
   - Use PNG versions for Word/PowerPoint

3. **Methods Section** → Copy from `methods_section_manuscript.txt`
   - Customize as needed for your specific analysis
   - Add any additional details about your model

### For Supplementary Materials

1. **Table 2** → "Supplementary Table 1: Diagnosis Distribution"
2. **Figure 3** → "Supplementary Figure 1: Text Complexity Analysis"
3. **supplementary_missing_data.csv** → Data quality report
4. **supplementary_diagnosis_details.csv** → Detailed per-diagnosis statistics

### For Presentations

1. Use PNG figures (300 DPI ensures quality on projectors)
2. Extract key numbers from Table 1 for slides
3. Show Figure 1 for quick overview of dataset

### For Reviewers

1. Provide CSV versions of all tables for verification
2. Include supplementary_diagnosis_details.csv for transparency
3. Share eda_summary_report.txt for reproducibility

## Interpreting Results

### Good Signs (Well-Balanced Dataset)

✅ **All p-values > 0.05**: No significant differences between train and test
✅ **Similar percentages**: Train/test proportions match closely
✅ **Overlapping distributions**: Histogram overlays show similar shapes
✅ **Balanced quartiles**: Text complexity evenly distributed

### Warning Signs (May Need Rebalancing)

⚠️ **P-values < 0.05**: Significant differences between train/test
⚠️ **Large percentage differences**: >5% difference in any category
⚠️ **Non-overlapping distributions**: Train and test look very different
⚠️ **Skewed quartiles**: One quartile dominates

### What to Report in Manuscript

**Example paragraph**:

> "Table 1 presents baseline characteristics of the training and test sets.
> The training set comprised 4,000 cases (mean age 64.2 ± 17.3 years, 45.5% female),
> while the test set comprised 1,000 cases (mean age 64.5 ± 17.1 years, 45.2% female).
> No significant differences were observed between training and test sets in any
> demographic or clinical characteristics (all p > 0.05), confirming successful
> stratification. The five diagnosis categories were well-represented: Cardiovascular
> (38.4%), Infectious (19.5%), Renal (8.1%), Psychiatric (19.9%), and Oncology (8.5%)."

## Customization

### Change Figure Colors

Edit the color palette:

```python
sns.set_palette("husl")  # Change to: "deep", "muted", "bright", "pastel", "dark", "colorblind"
```

### Add More Statistical Tests

Add your own comparisons:

```python
# Effect size (Cohen's d)
from scipy.stats import cohen_d

cohens_d = (train_age.mean() - test_age.mean()) / np.sqrt((train_age.std()**2 + test_age.std()**2) / 2)
```

### Modify Table Layout

The LaTeX tables can be customized:

```python
column_format='l|c|c|c'  # Change to adjust column alignment
```

### Add More Figures

Create additional visualizations:

```python
# Example: Diagnosis by gender
fig, ax = plt.subplots(figsize=(10, 6))
pd.crosstab(train['diagnosis'], train['gender']).plot(kind='bar', ax=ax)
plt.savefig('figure4_diagnosis_by_gender.png')
```

## Troubleshooting

### "File not found"
Check that `TRAIN_FILE` and `TEST_FILE` paths are correct for your system

### "Column not found"
The script auto-detects column names. If it fails, check your CSV column names match expected patterns

### "ImportError: No module named..."
Install missing packages:
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### Figures don't display
The script saves figures to files automatically. To view during execution, add:
```python
plt.show()  # Before plt.close()
```

### LaTeX tables have formatting issues
- Check special characters are escaped
- Adjust `column_format` parameter
- Use `\resizebox` for wide tables

### P-values show warnings
Small sample sizes in some groups may cause chi-square warnings. This is expected and the script handles it.

## Advanced Usage

### Compare Multiple Datasets

Modify the script to load and compare multiple train/test splits:

```python
train1, test1 = load_datasets("split1")
train2, test2 = load_datasets("split2")
# Compare distributions across splits
```

### Generate Confidence Intervals

Add bootstrap confidence intervals:

```python
from scipy.stats import bootstrap

# 95% CI for mean age
result = bootstrap((train['age'],), np.mean, n_resamples=10000)
ci = result.confidence_interval
```

### Export to Excel

For stakeholders who prefer Excel:

```python
with pd.ExcelWriter('eda_results.xlsx') as writer:
    table1_df.to_excel(writer, sheet_name='Table1')
    table2_df.to_excel(writer, sheet_name='Table2')
```

## Best Practices

### For Publication

1. **Always include p-values** for train/test comparisons
2. **Report both mean/SD and median/IQR** for continuous variables
3. **Use consistent rounding** (1 decimal for percentages, 0 for counts)
4. **Provide both PNG and PDF** figures to journal
5. **Include methods section** describing your approach

### For Data Quality

1. **Review missing data table** - address any systematic missingness
2. **Check all p-values** - all should be > 0.05 for good balance
3. **Visually inspect figures** - look for obvious imbalances
4. **Verify sample sizes** - ensure groups aren't too small

### For Reproducibility

1. **Version control** all scripts
2. **Document random seeds** (script uses 42)
3. **Save environment** requirements.txt
4. **Include data dictionary** explaining all columns

## Citation

If you use this EDA script in your publication, consider acknowledging:

```
Data analysis and visualization performed using custom Python scripts
based on pandas, numpy, scipy, matplotlib, and seaborn libraries.
```

## Version History

- **v1.0** (2025-11-10): Initial release
  - Comprehensive baseline characteristics table
  - Diagnosis distribution table
  - Three main figure panels
  - Supplementary statistics
  - Manuscript methods section
  - Publication-ready outputs

---

**Author**: Claude (Anthropic)
**Date**: 2025-11-10
**For**: MIMIC-IV Clinical Consultant AI Training Project
**License**: Open source - modify as needed for your analysis
