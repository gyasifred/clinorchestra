"""
Create Balanced Train/Test Split from MIMIC-IV Dataset

This script:
1. Reads large annotation/classification dataset
2. Intelligently samples 5,000 cases balanced across:
   - Top 20 diagnoses (proportional to original distribution)
   - Gender (M/F)
   - Race/ethnicity
   - Clinical note complexity (text length quartiles)
3. Splits into train (4,000) and test (1,000) sets
4. Maintains balance across all dimensions

Author: Claude
Date: 2025-11-10
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import os
import sys

# Configuration
INPUT_FILE = r"C:\Users\gyasi\Documents\mimic-iv-3.1\outputs\classification_dataset.csv"
OUTPUT_DIR = r"C:\Users\gyasi\Documents\mimic-iv-3.1\outputs"
TOTAL_SAMPLES = 5000
TRAIN_SIZE = 4000
TEST_SIZE = 1000
RANDOM_SEED = 42

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

def calculate_text_complexity(text):
    """
    Calculate text complexity based on length.
    Returns quartile: 'Q1_short', 'Q2_medium', 'Q3_long', 'Q4_very_long'
    """
    if pd.isna(text):
        return 'Q1_short'

    length = len(str(text))
    return length

def assign_complexity_quartile(lengths):
    """
    Assign complexity quartile based on text length distribution.
    """
    q1, q2, q3 = np.percentile(lengths, [25, 50, 75])

    def categorize(length):
        if length <= q1:
            return 'Q1_short'
        elif length <= q2:
            return 'Q2_medium'
        elif length <= q3:
            return 'Q3_long'
        else:
            return 'Q4_very_long'

    return [categorize(l) for l in lengths]

def standardize_race(race):
    """
    Standardize race/ethnicity categories.
    """
    if pd.isna(race):
        return 'UNKNOWN'

    race = str(race).upper()

    # Map to standard categories
    if 'WHITE' in race:
        return 'WHITE'
    elif 'BLACK' in race or 'AFRICAN' in race:
        return 'BLACK'
    elif 'ASIAN' in race:
        return 'ASIAN'
    elif 'HISPANIC' in race or 'LATINO' in race:
        return 'HISPANIC'
    elif 'OTHER' in race:
        return 'OTHER'
    elif 'UNKNOWN' in race or 'UNABLE' in race or 'DECLINED' in race:
        return 'UNKNOWN'
    else:
        return 'OTHER'

def create_stratification_key(row):
    """
    Create composite stratification key combining diagnosis, gender, race, and text complexity.
    """
    diagnosis = row.get('icd_code', row.get('diagnosis_code', 'UNKNOWN'))
    gender = row.get('gender', 'U')
    race = row.get('race_category', 'UNKNOWN')
    complexity = row.get('text_complexity', 'Q2_medium')

    return f"{diagnosis}_{gender}_{race}_{complexity}"

def balanced_sample(df, n_samples, stratify_columns):
    """
    Perform stratified sampling to maintain balance across multiple dimensions.

    Strategy:
    1. Calculate target samples per diagnosis (proportional to original)
    2. Within each diagnosis, balance by gender, race, and text complexity
    3. If some strata are under-represented, redistribute to other strata
    """

    # Calculate target samples per diagnosis (proportional)
    diagnosis_col = 'icd_code' if 'icd_code' in df.columns else 'diagnosis_code'
    diagnosis_counts = df[diagnosis_col].value_counts()
    diagnosis_proportions = diagnosis_counts / len(df)

    target_per_diagnosis = (diagnosis_proportions * n_samples).round().astype(int)

    # Adjust to ensure exactly n_samples
    while target_per_diagnosis.sum() != n_samples:
        if target_per_diagnosis.sum() < n_samples:
            # Add to largest group
            target_per_diagnosis[target_per_diagnosis.idxmax()] += 1
        else:
            # Subtract from largest group
            target_per_diagnosis[target_per_diagnosis.idxmax()] -= 1

    sampled_dfs = []

    # Sample from each diagnosis
    for diagnosis, target_n in target_per_diagnosis.items():
        diagnosis_df = df[df[diagnosis_col] == diagnosis].copy()

        if len(diagnosis_df) <= target_n:
            # Take all if not enough samples
            sampled_dfs.append(diagnosis_df)
            print(f"  {diagnosis}: Taking all {len(diagnosis_df)} samples (target: {target_n})")
        else:
            # Stratified sampling within diagnosis
            # Create stratification key (gender + race + complexity)
            diagnosis_df['stratum'] = (
                diagnosis_df['gender'].astype(str) + '_' +
                diagnosis_df['race_category'].astype(str) + '_' +
                diagnosis_df['text_complexity'].astype(str)
            )

            # Calculate samples per stratum (proportional within diagnosis)
            stratum_counts = diagnosis_df['stratum'].value_counts()
            stratum_proportions = stratum_counts / len(diagnosis_df)
            target_per_stratum = (stratum_proportions * target_n).round().astype(int)

            # Adjust to ensure exactly target_n
            while target_per_stratum.sum() != target_n:
                if target_per_stratum.sum() < target_n:
                    target_per_stratum[target_per_stratum.idxmax()] += 1
                else:
                    target_per_stratum[target_per_stratum.idxmax()] -= 1

            # Sample from each stratum
            stratum_samples = []
            for stratum, stratum_target in target_per_stratum.items():
                stratum_df = diagnosis_df[diagnosis_df['stratum'] == stratum]
                if len(stratum_df) <= stratum_target:
                    stratum_samples.append(stratum_df)
                else:
                    stratum_samples.append(stratum_df.sample(n=stratum_target, random_state=RANDOM_SEED))

            sampled_diagnosis = pd.concat(stratum_samples, ignore_index=True)
            sampled_dfs.append(sampled_diagnosis.drop('stratum', axis=1))
            print(f"  {diagnosis}: Sampled {len(sampled_diagnosis)} cases (target: {target_n})")

    return pd.concat(sampled_dfs, ignore_index=True)

def print_distribution(df, name):
    """
    Print distribution statistics for the dataset.
    """
    print(f"\n{'='*60}")
    print(f"{name} - Distribution Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}")

    # Diagnosis distribution
    diagnosis_col = 'icd_code' if 'icd_code' in df.columns else 'diagnosis_code'
    print(f"\n{'-'*60}")
    print("Diagnosis Distribution:")
    print(f"{'-'*60}")
    diagnosis_dist = df[diagnosis_col].value_counts()
    for diagnosis, count in diagnosis_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {diagnosis}: {count:4d} ({percentage:5.2f}%)")

    # Gender distribution
    print(f"\n{'-'*60}")
    print("Gender Distribution:")
    print(f"{'-'*60}")
    gender_dist = df['gender'].value_counts()
    for gender, count in gender_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {gender}: {count:4d} ({percentage:5.2f}%)")

    # Race distribution
    print(f"\n{'-'*60}")
    print("Race Distribution:")
    print(f"{'-'*60}")
    race_dist = df['race_category'].value_counts()
    for race, count in race_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {race}: {count:4d} ({percentage:5.2f}%)")

    # Text complexity distribution
    print(f"\n{'-'*60}")
    print("Text Complexity Distribution:")
    print(f"{'-'*60}")
    complexity_dist = df['text_complexity'].value_counts()
    for complexity, count in complexity_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  {complexity}: {count:4d} ({percentage:5.2f}%)")

    # Text length statistics
    print(f"\n{'-'*60}")
    print("Text Length Statistics:")
    print(f"{'-'*60}")
    print(f"  Mean:   {df['text_length'].mean():,.0f} characters")
    print(f"  Median: {df['text_length'].median():,.0f} characters")
    print(f"  Min:    {df['text_length'].min():,.0f} characters")
    print(f"  Max:    {df['text_length'].max():,.0f} characters")
    print(f"  Q1:     {df['text_length'].quantile(0.25):,.0f} characters")
    print(f"  Q3:     {df['text_length'].quantile(0.75):,.0f} characters")

def main():
    print("="*70)
    print("MIMIC-IV Balanced Train/Test Split Generator")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Input file: {INPUT_FILE}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Total samples to select: {TOTAL_SAMPLES:,}")
    print(f"  Train size: {TRAIN_SIZE:,}")
    print(f"  Test size: {TEST_SIZE:,}")
    print(f"  Random seed: {RANDOM_SEED}")

    # Check if input file exists
    if not os.path.exists(INPUT_FILE):
        print(f"\nERROR: Input file not found: {INPUT_FILE}")
        print(f"Please check the path and try again.")
        sys.exit(1)

    # Step 1: Read dataset
    print(f"\n{'='*70}")
    print("Step 1: Reading dataset...")
    print(f"{'='*70}")

    # Read in chunks to avoid memory issues
    chunk_size = 10000
    chunks = []
    total_rows = 0

    try:
        for i, chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=chunk_size)):
            chunks.append(chunk)
            total_rows += len(chunk)
            if (i + 1) % 10 == 0:
                print(f"  Read {total_rows:,} rows...")
    except Exception as e:
        print(f"\nERROR reading file: {e}")
        sys.exit(1)

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Total rows read: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # Step 2: Data preprocessing
    print(f"\n{'='*70}")
    print("Step 2: Preprocessing data...")
    print(f"{'='*70}")

    # Identify clinical text column
    text_columns = ['clinical_text', 'text', 'note_text', 'discharge_text']
    text_col = None
    for col in text_columns:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        print(f"ERROR: Could not find clinical text column. Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"  Clinical text column: {text_col}")

    # Calculate text lengths
    print(f"  Calculating text complexity...")
    df['text_length'] = df[text_col].apply(calculate_text_complexity)

    # Assign complexity quartiles
    df['text_complexity'] = assign_complexity_quartile(df['text_length'].values)

    # Standardize race categories
    race_col = 'race' if 'race' in df.columns else 'ethnicity'
    if race_col in df.columns:
        print(f"  Standardizing race/ethnicity...")
        df['race_category'] = df[race_col].apply(standardize_race)
    else:
        print(f"  WARNING: No race/ethnicity column found. Using 'UNKNOWN'.")
        df['race_category'] = 'UNKNOWN'

    # Ensure gender is clean
    if 'gender' not in df.columns:
        print(f"  WARNING: No gender column found. Using 'U' (unknown).")
        df['gender'] = 'U'
    else:
        df['gender'] = df['gender'].fillna('U')

    # Identify diagnosis column (support both old and new formats)
    diagnosis_col = None
    for col in ['consolidated_diagnosis_name', 'consolidated_diagnosis_id', 'icd_code', 'diagnosis_code', 'primary_diagnosis']:
        if col in df.columns:
            diagnosis_col = col
            break

    if diagnosis_col is None:
        print(f"ERROR: Could not find diagnosis column. Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"  Diagnosis column: {diagnosis_col}")

    # Print original distribution
    print_distribution(df, "ORIGINAL DATASET")

    # Step 3: Balanced sampling
    print(f"\n{'='*70}")
    print(f"Step 3: Performing balanced sampling ({TOTAL_SAMPLES:,} cases)...")
    print(f"{'='*70}")

    stratify_columns = [diagnosis_col, 'gender', 'race_category', 'text_complexity']
    sampled_df = balanced_sample(df, TOTAL_SAMPLES, stratify_columns)

    print(f"\n  Successfully sampled {len(sampled_df):,} cases")

    # Print sampled distribution
    print_distribution(sampled_df, "SAMPLED DATASET (5000 cases)")

    # Step 4: Train/Test split
    print(f"\n{'='*70}")
    print(f"Step 4: Splitting into train ({TRAIN_SIZE:,}) and test ({TEST_SIZE:,})...")
    print(f"{'='*70}")

    # Create composite stratification key for train/test split
    sampled_df['stratify_key'] = sampled_df.apply(create_stratification_key, axis=1)

    # Perform stratified split
    # If some strata have only 1 sample, they can't be split, so we'll handle that
    try:
        train_df, test_df = train_test_split(
            sampled_df,
            test_size=TEST_SIZE / TOTAL_SAMPLES,
            stratify=sampled_df['stratify_key'],
            random_state=RANDOM_SEED
        )
    except ValueError as e:
        print(f"  Warning: Some strata too small for perfect stratification. Using diagnosis-only stratification.")
        train_df, test_df = train_test_split(
            sampled_df,
            test_size=TEST_SIZE / TOTAL_SAMPLES,
            stratify=sampled_df[diagnosis_col],
            random_state=RANDOM_SEED
        )

    # Remove temporary columns
    train_df = train_df.drop(['stratify_key', 'text_length', 'text_complexity', 'race_category'],
                              axis=1, errors='ignore')
    test_df = test_df.drop(['stratify_key', 'text_length', 'text_complexity', 'race_category'],
                            axis=1, errors='ignore')

    print(f"\n  Train set: {len(train_df):,} cases")
    print(f"  Test set:  {len(test_df):,} cases")

    # Add temporary columns back for distribution display
    train_df_display = train_df.copy()
    test_df_display = test_df.copy()

    train_df_display['text_length'] = train_df_display[text_col].apply(calculate_text_complexity)
    train_df_display['text_complexity'] = assign_complexity_quartile(train_df_display['text_length'].values)
    train_df_display['race_category'] = train_df_display[race_col].apply(standardize_race) if race_col in train_df_display.columns else 'UNKNOWN'

    test_df_display['text_length'] = test_df_display[text_col].apply(calculate_text_complexity)
    test_df_display['text_complexity'] = assign_complexity_quartile(test_df_display['text_length'].values)
    test_df_display['race_category'] = test_df_display[race_col].apply(standardize_race) if race_col in test_df_display.columns else 'UNKNOWN'

    # Print distributions
    print_distribution(train_df_display, "TRAIN SET (4000 cases)")
    print_distribution(test_df_display, "TEST SET (1000 cases)")

    # Step 5: Save datasets
    print(f"\n{'='*70}")
    print("Step 5: Saving train and test sets...")
    print(f"{'='*70}")

    train_file = os.path.join(OUTPUT_DIR, "train_dataset_4000.csv")
    test_file = os.path.join(OUTPUT_DIR, "test_dataset_1000.csv")

    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"  Train set saved: {train_file}")
    print(f"  Test set saved:  {test_file}")

    # Save metadata
    metadata_file = os.path.join(OUTPUT_DIR, "train_test_split_metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write("MIMIC-IV Train/Test Split Metadata\n")
        f.write("="*70 + "\n\n")
        f.write(f"Creation date: {pd.Timestamp.now()}\n")
        f.write(f"Random seed: {RANDOM_SEED}\n")
        f.write(f"Input file: {INPUT_FILE}\n")
        f.write(f"Total samples selected: {TOTAL_SAMPLES:,}\n")
        f.write(f"Train samples: {TRAIN_SIZE:,}\n")
        f.write(f"Test samples: {TEST_SIZE:,}\n\n")

        f.write("Balancing dimensions:\n")
        f.write("  - Top 20 diagnoses (proportional to original distribution)\n")
        f.write("  - Gender (M/F/U)\n")
        f.write("  - Race/ethnicity categories\n")
        f.write("  - Text complexity quartiles (based on character length)\n\n")

        f.write("Files created:\n")
        f.write(f"  - {train_file}\n")
        f.write(f"  - {test_file}\n")
        f.write(f"  - {metadata_file}\n")

    print(f"  Metadata saved: {metadata_file}")

    # Final summary
    print(f"\n{'='*70}")
    print("SUCCESS! Train/Test split completed.")
    print(f"{'='*70}")
    print(f"\nFiles created:")
    print(f"  1. {train_file}")
    print(f"  2. {test_file}")
    print(f"  3. {metadata_file}")
    print(f"\nNext steps:")
    print(f"  1. Review the distribution statistics above")
    print(f"  2. Use train_dataset_4000.csv for Task 1 (Annotation) or Task 2 (Classification)")
    print(f"  3. Use test_dataset_1000.csv for final evaluation")
    print(f"  4. Keep the datasets separate throughout training and evaluation")
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nScript interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
