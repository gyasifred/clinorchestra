# ICD Code Consistency Check Report
# MIMIC-IV Top 20 Diagnoses

**Date**: 2025-11-11
**Purpose**: Verify that diagnoses with different ICD codes (ICD-9 vs ICD-10) representing the same condition are handled consistently across all files

---

## EXECUTIVE SUMMARY

**Total Diagnosis Pairs Checked**: 6 pairs (12 individual ICD codes)

**Issues Found**: 2

1. ❌ **Sepsis extras file missing ICD-9 codes** in metadata
2. ⚠️ **AKI has two separate extras files** (may cause confusion or duplication)

**Recommendations**:
- Update sepsis_diagnostic_criteria.json metadata to include ICD-9 codes
- Consolidate AKI extras files OR document that both should be loaded
- Standardize ICD-10 code format (dotted vs non-dotted)

---

## DETAILED FINDINGS

### 1. CHEST PAIN (3 ICD codes)

**ICD Codes**:
- 78650 (ICD-9): "Chest pain, unspecified" - 7,297 cases
- 78659 (ICD-9): "Other chest pain" - 5,212 cases
- R079 (ICD-10): "Chest pain, unspecified" - 2,906 cases

**Total Cases**: 15,415 (20.1% of dataset)

**Status**: ✅ **CONSISTENT**

**Extras File**: `chest_pain_evaluation.json`
- **Content**: Unified clinical approach covering all chest pain presentations
- **Metadata**:
  ```json
  "icd9_codes": ["78650", "78659"],
  "icd10_codes": ["R079"]
  ```
- **Clinical Content**: Single comprehensive evaluation protocol appropriate for all 3 codes
- **Key Points**: RED FLAGS, cardiac vs non-cardiac, HEART score, diagnostic workup

**Classification Prompt**: Lists all 3 as separate diagnoses (#1, #3, #4)

**Recommendation**: ✅ No changes needed - well structured

---

### 2. SEPSIS (2 ICD codes)

**ICD Codes**:
- A419 (ICD-10): "Sepsis, unspecified organism" - 5,095 cases
- 0389 (ICD-9): "Unspecified septicemia" - 3,144 cases

**Total Cases**: 8,239 (10.8% of dataset)

**Status**: ❌ **INCONSISTENT - NEEDS FIX**

**Extras File**: `sepsis_diagnostic_criteria.json`
- **Content**: Sepsis-3 definition, SOFA criteria, septic shock
- **Metadata** (CURRENT):
  ```json
  "icd10_codes": ["A41.9", "A41.89"]
  ```
- **Problem**:
  - ❌ Missing ICD-9 codes: ["0389", "389"]
  - ❌ Uses dotted format "A41.9" instead of "A419" (inconsistent with classification prompt)
  - ❌ Includes "A41.89" which is NOT in top 20 list

**Classification Prompt**: Lists both codes as separate diagnoses (#9, #10)

**Recommendation**: 🔧 **UPDATE REQUIRED**
- Add ICD-9 codes to metadata
- Use non-dotted format for consistency
- Remove "A41.89" if not in top 20

**Proposed Metadata Fix**:
```json
"icd9_codes": ["0389", "389"],
"icd10_codes": ["A419"]
```

---

### 3. DEPRESSION (2 ICD codes)

**ICD Codes**:
- F329 (ICD-10): "Major depressive disorder, single episode, unspecified" - 3,703 cases
- 311 (ICD-9): "Depressive disorder, not elsewhere classified" - 3,499 cases

**Total Cases**: 7,202 (9.4% of dataset)

**Status**: ✅ **CONSISTENT**

**Extras File**: `depression_criteria.json`
- **Content**: DSM-5 criteria, PHQ-9 scoring, suicide assessment
- **Metadata**:
  ```json
  "icd9_codes": ["311"],
  "icd10_codes": ["F329"]
  ```
- **Clinical Content**: Unified approach appropriate for both ICD-9 and ICD-10 codes

**Classification Prompt**: Lists both codes as separate diagnoses (#15, #16)

**Recommendation**: ✅ No changes needed

---

### 4. ALCOHOL USE DISORDER (2 ICD codes)

**ICD Codes**:
- 30500 (ICD-9): "Alcohol abuse, unspecified" - 4,591 cases
- F10129 (ICD-10): "Alcohol abuse with intoxication, unspecified" - 3,447 cases

**Total Cases**: 8,038 (10.5% of dataset)

**Status**: ✅ **CONSISTENT**

**Extras File**: `alcohol_use_disorder.json`
- **Content**: DSM-5 criteria, withdrawal, CIWA scoring, complications
- **Metadata**:
  ```json
  "icd9_codes": ["30500"],
  "icd10_codes": ["F10129", "F1010"]
  ```
- **Clinical Content**: Unified approach for alcohol-related conditions
- **Note**: Includes additional ICD-10 code "F1010" (not in top 20 but related)

**Classification Prompt**: Lists both codes as separate diagnoses (#17, #18)

**Recommendation**: ✅ No changes needed

---

### 5. CHEMOTHERAPY ENCOUNTER (2 ICD codes)

**ICD Codes**:
- Z5111 (ICD-10): "Encounter for antineoplastic chemotherapy" - 3,557 cases
- V5811 (ICD-9): "Encounter for antineoplastic chemotherapy" - 2,973 cases

**Total Cases**: 6,530 (8.5% of dataset)

**Status**: ✅ **CONSISTENT**

**Extras File**: `chemotherapy_encounter.json`
- **Content**: NOT A DIAGNOSIS - encounter code, complications, supportive care
- **Metadata**:
  ```json
  "icd9_codes": ["V5811"],
  "icd10_codes": ["Z5111"]
  ```
- **Clinical Content**: Appropriate guidance for both code systems

**Classification Prompt**: Lists both codes as separate diagnoses (#19, #20)

**Recommendation**: ✅ No changes needed

---

### 6. ACUTE KIDNEY INJURY (2 ICD codes)

**ICD Codes**:
- 5849 (ICD-9): "Acute kidney failure, unspecified" - 3,532 cases
- N179 (ICD-10): "Acute kidney failure, unspecified" - 2,653 cases

**Total Cases**: 6,185 (8.1% of dataset)

**Status**: ⚠️ **TWO SEPARATE FILES - POTENTIAL DUPLICATION**

**Extras Files**:

#### File 1: `aki_staging.json`
- **Content**: KDIGO staging criteria (basic)
- **Metadata**:
  ```json
  "icd10_codes": ["N17.9", "N17.0", "N17.1", "N17.2"]
  ```
- **Problem**:
  - ❌ Missing ICD-9 code "5849"
  - ❌ Uses dotted format (N17.9)
  - ✅ Includes additional stage-specific codes (N17.0, N17.1, N17.2)

#### File 2: `aki_detailed.json`
- **Content**: KDIGO staging + etiology (prerenal/intrinsic/postrenal) + workup
- **Metadata**:
  ```json
  "icd9_codes": ["5849"],
  "icd10_codes": ["N179"]
  ```
- **More comprehensive** than aki_staging.json

**Classification Prompt**: Lists both codes as separate diagnoses (#13, #14)

**Recommendation**: ⚠️ **CONSOLIDATION NEEDED**

**Option A (Recommended)**: Consolidate into single file
- Keep `aki_detailed.json` (more comprehensive)
- Delete `aki_staging.json`
- Update metadata to include all relevant codes:
  ```json
  "icd9_codes": ["5849"],
  "icd10_codes": ["N179", "N170", "N171", "N172"]
  ```

**Option B**: Keep both files but document clearly
- Rename files to indicate purpose:
  - `aki_kdigo_staging_basic.json`
  - `aki_comprehensive_with_etiology.json`
- Ensure metadata is consistent across both
- Document in SETUP_COMPLETE.md that BOTH should be loaded

**Option C**: Current state (not recommended)
- Risk: LLM may get confused with duplicate/conflicting information
- Risk: One file may be loaded but not the other

---

## SUMMARY OF ISSUES

### Critical Issues (Must Fix)

1. **Sepsis extras file metadata** (`sepsis_diagnostic_criteria.json`)
   - Add ICD-9 codes: ["0389", "389"]
   - Change "A41.9" → "A419" for consistency
   - Remove "A41.89" (not in top 20)

### Warnings (Should Address)

2. **AKI duplicate extras files**
   - Two files for same condition may cause confusion
   - Consolidate OR clearly document both are needed

### Minor Inconsistencies (Low Priority)

3. **ICD code format inconsistency**
   - Some files use dotted format: "A41.9", "N17.9"
   - Classification prompt uses non-dotted: "A419", "N179"
   - Standardize to non-dotted format for consistency

---

## FILES CHECKED

### Extras Files (7 files checked)
✅ chest_pain_evaluation.json
❌ sepsis_diagnostic_criteria.json (missing ICD-9)
✅ depression_criteria.json
✅ alcohol_use_disorder.json
✅ chemotherapy_encounter.json
⚠️ aki_staging.json (missing ICD-9)
⚠️ aki_detailed.json (duplicate with above)

### Prompts (1 file checked)
✅ task2_classification_prompt_FINAL.txt (lists all 20 diagnoses correctly)

### Documentation (4 files checked)
✅ SETUP_COMPLETE.md (lists all 20 diagnoses)
✅ COMPLETE_SYSTEM_OVERVIEW.md (comprehensive overview)
✅ RAG_RESOURCES_COMPREHENSIVE.md (resources by diagnosis)
✅ CLINICAL_RESOURCES.md (guidelines)

### Scripts (2 files checked)
✅ eda_train_test_publication.py (ICD_NAMES dictionary)
✅ README_train_test_split.md (top 20 list)

---

## RECOMMENDED ACTIONS

### Immediate Actions (Priority 1)

1. **Fix Sepsis Metadata**
   ```bash
   File: mimic-iv/extras/sepsis_diagnostic_criteria.json
   Change metadata from:
     "icd10_codes": ["A41.9", "A41.89"]
   To:
     "icd9_codes": ["0389", "389"],
     "icd10_codes": ["A419"]
   ```

2. **Consolidate AKI Files**
   ```bash
   Option A: Keep aki_detailed.json, delete aki_staging.json
   Option B: Keep both, update metadata in aki_staging.json to add ICD-9
   ```

### Follow-up Actions (Priority 2)

3. **Standardize ICD-10 Code Format**
   - Use non-dotted format throughout: "A419" not "A41.9"
   - Update any remaining dotted codes in extras files

4. **Update SETUP_COMPLETE.md**
   - Document which extras files should be loaded
   - Note that some diagnoses share the same extras file

5. **Validation**
   - Re-run consistency check after fixes
   - Test ClinOrchestra loading of all extras files

---

## CLINICAL CORRECTNESS

All extras files contain **clinically appropriate** content for the diagnoses they represent:

✅ **Chest pain**: Appropriate evaluation for all 3 codes (same condition, different specificity)
✅ **Sepsis**: Sepsis-3 criteria apply to both ICD-9 and ICD-10 codes
✅ **Depression**: DSM-5 criteria are code-agnostic
✅ **Alcohol**: DSM-5 and withdrawal management are universal
✅ **Chemotherapy**: Encounter code guidance applies to both systems
✅ **AKI**: KDIGO criteria are the same for ICD-9 and ICD-10

**No clinical content changes needed** - only metadata/organizational fixes.

---

## CONCLUSION

The MIMIC-IV top 20 diagnoses configuration is **mostly consistent** with only **2 actionable issues**:

1. Sepsis metadata missing ICD-9 codes (easy fix)
2. AKI has duplicate files (needs consolidation decision)

All clinical content is appropriate and evidence-based. The inconsistencies are primarily organizational/metadata issues that should be addressed to ensure:
- ClinOrchestra loads the correct extras for each diagnosis
- No confusion from duplicate or missing metadata
- Standardized format across all configuration files

**Estimated time to fix**: 15-30 minutes

---

**Report Generated**: 2025-11-11
**Files Analyzed**: 14 files across extras, prompts, documentation, and scripts
**Total ICD Codes Checked**: 12 codes (6 diagnosis pairs)
