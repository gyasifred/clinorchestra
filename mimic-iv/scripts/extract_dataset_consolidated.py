"""
MIMIC-IV Dataset Extraction Script with Diagnosis Consolidation

This script creates comprehensive datasets for:
1. Annotation Task: Extract evidence supporting a given diagnosis
2. Classification Task: Predict diagnosis from clinical evidence

CONSOLIDATION: Same clinical conditions with different ICD codes (ICD-9 vs ICD-10)
are consolidated into single diagnoses.

Example: "Chest pain (78650)", "Other chest pain (78659)", "Chest pain (R079)"
         → Single "Chest pain" diagnosis
"""

import pandas as pd
import os
from pathlib import Path
import logging
from typing import Optional, List
import json
from datetime import datetime
import sys

# Add parent directory to path to import diagnosis_mapping
sys.path.insert(0, str(Path(__file__).parent.parent))
from diagnosis_mapping import (
    DIAGNOSIS_CONSOLIDATION,
    CONSOLIDATED_DIAGNOSES,
    get_consolidated_diagnosis,
    get_diagnosis_info
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MIMICDatasetBuilderConsolidated:
    """Build comprehensive MIMIC-IV datasets with consolidated diagnoses"""

    def __init__(self, mimic_path: str):
        """
        Initialize the dataset builder

        Args:
            mimic_path: Path to MIMIC-IV root directory
        """
        self.mimic_path = Path(mimic_path)
        self.hosp_path = self.mimic_path / "hosp"
        self.icu_path = self.mimic_path / "icu"
        self.note_path = self.mimic_path / "note"

        # Get all ICD codes from consolidation mapping
        self.target_icd_codes = set(DIAGNOSIS_CONSOLIDATION.keys())
        logger.info(f"Targeting {len(self.target_icd_codes)} ICD codes across {len(CONSOLIDATED_DIAGNOSES)} consolidated diagnoses")

    def load_core_tables(self):
        """Load core tables needed for dataset construction"""
        logger.info("="*80)
        logger.info("LOADING CORE MIMIC-IV TABLES")
        logger.info("="*80)

        # Patients
        logger.info("Loading patients.csv...")
        self.patients = pd.read_csv(self.hosp_path / "patients.csv")
        logger.info(f"  ✓ Loaded {len(self.patients):,} patients")

        # Admissions
        logger.info("Loading admissions.csv...")
        self.admissions = pd.read_csv(self.hosp_path / "admissions.csv")
        logger.info(f"  ✓ Loaded {len(self.admissions):,} admissions")

        # Diagnoses
        logger.info("Loading diagnoses_icd.csv...")
        self.diagnoses = pd.read_csv(self.hosp_path / "diagnoses_icd.csv")
        logger.info(f"  ✓ Loaded {len(self.diagnoses):,} diagnoses")

        # ICD code descriptions
        logger.info("Loading d_icd_diagnoses.csv...")
        self.icd_descriptions = pd.read_csv(self.hosp_path / "d_icd_diagnoses.csv")
        logger.info(f"  ✓ Loaded {len(self.icd_descriptions):,} ICD descriptions")

        # Clinical notes
        logger.info("Loading discharge.csv...")
        self.discharge_notes = pd.read_csv(self.note_path / "discharge.csv")
        logger.info(f"  ✓ Loaded {len(self.discharge_notes):,} discharge notes")

        logger.info("Loading radiology.csv...")
        self.radiology_notes = pd.read_csv(self.note_path / "radiology.csv")
        logger.info(f"  ✓ Loaded {len(self.radiology_notes):,} radiology notes")

        logger.info("="*80 + "\n")

    def build_base_dataset(self) -> pd.DataFrame:
        """
        Build base dataset with patient, admission, and consolidated diagnosis

        Returns:
            DataFrame with base patient information including consolidated diagnosis
        """
        logger.info("\n" + "="*80)
        logger.info("BUILDING BASE DATASET WITH CONSOLIDATED DIAGNOSES")
        logger.info("="*80)

        # Get primary diagnoses only
        primary_dx = self.diagnoses[self.diagnoses['seq_num'] == 1].copy()

        # Filter for target ICD codes
        primary_dx = primary_dx[primary_dx['icd_code'].isin(self.target_icd_codes)]
        logger.info(f"Found {len(primary_dx):,} admissions with target primary diagnoses")

        # Add consolidated diagnosis mapping
        primary_dx['consolidated_diagnosis_id'] = primary_dx['icd_code'].apply(
            lambda x: get_consolidated_diagnosis(x)[2] if get_consolidated_diagnosis(x) else None
        )
        primary_dx['consolidated_diagnosis_name'] = primary_dx['icd_code'].apply(
            lambda x: get_consolidated_diagnosis(x)[0] if get_consolidated_diagnosis(x) else None
        )
        primary_dx['consolidated_category'] = primary_dx['icd_code'].apply(
            lambda x: get_consolidated_diagnosis(x)[1] if get_consolidated_diagnosis(x) else None
        )

        # Filter out any records that didn't map (shouldn't happen if mapping is complete)
        unmapped = primary_dx['consolidated_diagnosis_id'].isna().sum()
        if unmapped > 0:
            logger.warning(f"  ⚠ {unmapped} records could not be mapped to consolidated diagnoses (dropping)")
            primary_dx = primary_dx[primary_dx['consolidated_diagnosis_id'].notna()]

        # Merge with admissions
        dataset = primary_dx.merge(
            self.admissions,
            on=['subject_id', 'hadm_id'],
            how='inner'
        )
        logger.info(f"After admission merge: {len(dataset):,} records")

        # Merge with patients
        dataset = dataset.merge(
            self.patients,
            on='subject_id',
            how='inner'
        )
        logger.info(f"After patient merge: {len(dataset):,} records")

        # Merge with ICD descriptions (original ICD code description)
        dataset = dataset.merge(
            self.icd_descriptions,
            on=['icd_code', 'icd_version'],
            how='left'
        )
        logger.info(f"After ICD description merge: {len(dataset):,} records")

        # Rename for clarity
        dataset = dataset.rename(columns={
            'long_title': 'original_icd_description'
        })

        return dataset

    def add_clinical_notes(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Add discharge and radiology notes to dataset"""
        logger.info("\nAdding clinical notes...")

        # Aggregate discharge notes by hadm_id
        discharge_agg = self.discharge_notes.groupby('hadm_id').agg({
            'text': lambda x: '\n\n---\n\n'.join(x),
            'note_seq': 'count'
        }).reset_index()
        discharge_agg.columns = ['hadm_id', 'discharge_note', 'discharge_note_count']

        dataset = dataset.merge(discharge_agg, on='hadm_id', how='left')
        logger.info(f"  ✓ Added discharge notes: {dataset['discharge_note'].notna().sum():,} records")

        # Aggregate radiology notes by hadm_id
        radiology_agg = self.radiology_notes.groupby('hadm_id').agg({
            'text': lambda x: '\n\n---\n\n'.join(x),
            'note_seq': 'count'
        }).reset_index()
        radiology_agg.columns = ['hadm_id', 'radiology_note', 'radiology_note_count']

        dataset = dataset.merge(radiology_agg, on='hadm_id', how='left')
        logger.info(f"  ✓ Added radiology notes: {dataset['radiology_note'].notna().sum():,} records")

        # Combine all clinical text
        dataset['clinical_text'] = dataset.apply(
            lambda row: self._combine_clinical_text(row), axis=1
        )

        return dataset

    def _combine_clinical_text(self, row) -> str:
        """Combine all clinical text for a patient"""
        sections = []

        if pd.notna(row.get('discharge_note')):
            sections.append(f"=== DISCHARGE SUMMARY ===\n{row['discharge_note']}")

        if pd.notna(row.get('radiology_note')):
            sections.append(f"=== RADIOLOGY REPORTS ===\n{row['radiology_note']}")

        return '\n\n'.join(sections)

    def create_annotation_dataset(self, output_path: str, sample_size: Optional[int] = None):
        """
        Create dataset for TASK 1: Annotation (Evidence Extraction)

        Format: Each row has consolidated diagnosis + clinical text, model extracts evidence

        Args:
            output_path: Where to save the dataset
            sample_size: Limit number of records (for testing)
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING ANNOTATION DATASET (TASK 1) - CONSOLIDATED DIAGNOSES")
        logger.info("="*80)

        # Load tables
        self.load_core_tables()

        # Build base dataset
        dataset = self.build_base_dataset()

        # Add clinical notes
        dataset = self.add_clinical_notes(dataset)

        # Filter records with clinical text
        dataset = dataset[dataset['clinical_text'].str.len() > 100]
        logger.info(f"Records with sufficient clinical text: {len(dataset):,}")

        # Sample if requested
        if sample_size:
            dataset = dataset.sample(n=min(sample_size, len(dataset)), random_state=42)
            logger.info(f"Sampled to {len(dataset):,} records")

        # Select and order columns for annotation task
        annotation_columns = [
            'subject_id',
            'hadm_id',
            # Consolidated diagnosis (what the model should annotate)
            'consolidated_diagnosis_id',
            'consolidated_diagnosis_name',
            'consolidated_category',
            # Original ICD code (for reference)
            'icd_code',
            'icd_version',
            'original_icd_description',
            # Clinical data
            'clinical_text',
            # Demographics
            'admission_type',
            'gender',
            'anchor_age',
            'race',
            'insurance',
            'admittime',
            'dischtime',
            'hospital_expire_flag'
        ]

        annotation_dataset = dataset[annotation_columns].copy()

        # Save
        annotation_dataset.to_csv(output_path, index=False)
        logger.info(f"\n✓ Annotation dataset saved to: {output_path}")
        logger.info(f"✓ Total records: {len(annotation_dataset):,}")
        logger.info(f"✓ Columns: {list(annotation_dataset.columns)}")

        # Print statistics
        self._print_consolidated_diagnosis_distribution(annotation_dataset)

        return annotation_dataset

    def create_classification_dataset(self, output_path: str, sample_size: Optional[int] = None):
        """
        Create dataset for TASK 2: Classification (Diagnosis Prediction)

        Format: Each row has clinical text only, model predicts consolidated diagnosis

        Args:
            output_path: Where to save the dataset
            sample_size: Limit number of records (for testing)
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING CLASSIFICATION DATASET (TASK 2) - CONSOLIDATED DIAGNOSES")
        logger.info("="*80)

        # Load tables if not already loaded
        if not hasattr(self, 'patients'):
            self.load_core_tables()

        # Build base dataset
        dataset = self.build_base_dataset()

        # Add clinical notes
        dataset = self.add_clinical_notes(dataset)

        # Filter records with clinical text
        dataset = dataset[dataset['clinical_text'].str.len() > 100]

        # Sample if requested
        if sample_size:
            dataset = dataset.sample(n=min(sample_size, len(dataset)), random_state=42)

        # Select columns for classification task
        classification_columns = [
            'subject_id',
            'hadm_id',
            'clinical_text',
            'admission_type',
            'gender',
            'anchor_age',
            'race',
            'insurance',
            'admittime',
            # Ground truth (for evaluation) - CONSOLIDATED DIAGNOSIS
            'consolidated_diagnosis_id',
            'consolidated_diagnosis_name',
            'consolidated_category',
            # Original ICD code (for reference/analysis)
            'icd_code',
            'icd_version',
            'original_icd_description'
        ]

        classification_dataset = dataset[classification_columns].copy()

        # Save
        classification_dataset.to_csv(output_path, index=False)
        logger.info(f"\n✓ Classification dataset saved to: {output_path}")
        logger.info(f"✓ Total records: {len(classification_dataset):,}")
        logger.info(f"✓ Columns: {list(classification_dataset.columns)}")

        # Print statistics
        self._print_consolidated_diagnosis_distribution(classification_dataset)

        return classification_dataset

    def _print_consolidated_diagnosis_distribution(self, dataset: pd.DataFrame):
        """Print distribution of CONSOLIDATED diagnoses in dataset"""
        logger.info("\n" + "-"*80)
        logger.info("CONSOLIDATED DIAGNOSIS DISTRIBUTION IN DATASET")
        logger.info("-"*80)

        dist = dataset.groupby(['consolidated_diagnosis_id', 'consolidated_diagnosis_name', 'consolidated_category']).size().reset_index(name='count')
        dist = dist.sort_values('count', ascending=False)

        for idx, row in dist.iterrows():
            pct = (row['count'] / len(dataset)) * 100
            logger.info(f"  {row['consolidated_diagnosis_id']:2d}. {row['consolidated_diagnosis_name']:35s} ({row['consolidated_category']:15s}) - {row['count']:6,d} cases ({pct:5.1f}%)")

        # Show original ICD code breakdown for consolidated diagnoses
        logger.info("\n" + "-"*80)
        logger.info("ORIGINAL ICD CODE BREAKDOWN (within consolidated diagnoses)")
        logger.info("-"*80)

        icd_dist = dataset.groupby(['consolidated_diagnosis_name', 'icd_code', 'icd_version']).size().reset_index(name='count')
        icd_dist = icd_dist.sort_values(['consolidated_diagnosis_name', 'count'], ascending=[True, False])

        current_diagnosis = None
        for idx, row in icd_dist.iterrows():
            if row['consolidated_diagnosis_name'] != current_diagnosis:
                current_diagnosis = row['consolidated_diagnosis_name']
                logger.info(f"\n  {current_diagnosis}:")
            logger.info(f"    - {row['icd_code']:10s} (ICD-{row['icd_version']}) - {row['count']:6,d} cases")

        logger.info("-"*80 + "\n")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("MIMIC-IV DATASET EXTRACTION WITH CONSOLIDATED DIAGNOSES")
    print("Consolidates same conditions with different ICD codes (ICD-9 vs ICD-10)")
    print("="*80)

    # Get path from user
    mimic_path = input("\nEnter path to MIMIC-IV directory: ").strip()

    if not mimic_path:
        print("❌ No path provided")
        sys.exit(1)

    sample_size = input("Enter sample size (or press Enter for all records): ").strip()
    sample_size = int(sample_size) if sample_size else None

    try:
        # Initialize builder
        builder = MIMICDatasetBuilderConsolidated(mimic_path)

        # Create annotation dataset
        annotation_output = "mimic-iv/annotation_dataset_consolidated.csv"
        builder.create_annotation_dataset(annotation_output, sample_size)

        # Create classification dataset
        classification_output = "mimic-iv/classification_dataset_consolidated.csv"
        builder.create_classification_dataset(classification_output, sample_size)

        print("\n" + "="*80)
        print("✓ DATASET EXTRACTION COMPLETE (CONSOLIDATED DIAGNOSES)!")
        print("="*80)
        print(f"✓ Annotation dataset: {annotation_output}")
        print(f"✓ Classification dataset: {classification_output}")
        print(f"\n✓ Key changes:")
        print(f"  - {len(CONSOLIDATED_DIAGNOSES)} consolidated diagnoses (instead of 20 individual ICD codes)")
        print(f"  - Same conditions with different ICD codes are now unified")
        print(f"  - Original ICD codes preserved for reference")
        print(f"\n✓ Next steps:")
        print(f"  1. Review the datasets")
        print(f"  2. Use UPDATED prompts for consolidated diagnoses")
        print(f"  3. Use UPDATED schemas for 13 diagnoses")
        print("="*80 + "\n")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
