"""
MIMIC-IV Dataset Extraction Script

This script creates comprehensive datasets for:
1. Annotation Task: Extract evidence supporting a given diagnosis
2. Classification Task: Predict diagnosis from clinical evidence

For each patient admission, it joins:
- Patient demographics (patients.csv)
- Admission info (admissions.csv)
- Primary diagnosis (diagnoses_icd.csv)
- Clinical notes (discharge.csv, radiology.csv)
- Lab results (labevents.csv)
- Medications (prescriptions.csv)
- Vital signs (chartevents.csv)
"""

import pandas as pd
import os
from pathlib import Path
import logging
from typing import Optional, List
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MIMICDatasetBuilder:
    """Build comprehensive MIMIC-IV datasets for annotation and classification"""

    def __init__(self, mimic_path: str, top_diagnoses_path: str):
        """
        Initialize the dataset builder

        Args:
            mimic_path: Path to MIMIC-IV root directory
            top_diagnoses_path: Path to CSV with top 20 diagnoses
        """
        self.mimic_path = Path(mimic_path)
        self.hosp_path = self.mimic_path / "hosp"
        self.icu_path = self.mimic_path / "icu"
        self.note_path = self.mimic_path / "note"

        # Load top diagnoses
        logger.info(f"Loading top diagnoses from: {top_diagnoses_path}")
        self.top_diagnoses = pd.read_csv(top_diagnoses_path)
        self.top_icd_codes = set(self.top_diagnoses['icd_code'].values)

        logger.info(f"Loaded {len(self.top_diagnoses)} top diagnoses")

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

    def load_additional_tables(self, include_labs: bool = True,
                               include_meds: bool = True,
                               include_vitals: bool = False):
        """
        Load additional tables (optional due to size)

        Args:
            include_labs: Load lab results
            include_meds: Load medications
            include_vitals: Load vital signs from ICU
        """
        logger.info("Loading additional clinical data...")

        if include_labs:
            logger.info("  Loading labevents.csv (this may take a while)...")
            # Note: labevents.csv is VERY large - consider sampling or filtering
            try:
                self.labevents = pd.read_csv(
                    self.hosp_path / "labevents.csv",
                    usecols=['subject_id', 'hadm_id', 'itemid', 'charttime',
                            'value', 'valuenum', 'valueuom', 'flag']
                )
                logger.info(f"  ✓ Loaded {len(self.labevents):,} lab events")
            except Exception as e:
                logger.warning(f"  ⚠ Could not load labevents: {e}")
                self.labevents = None

        if include_meds:
            logger.info("  Loading prescriptions.csv...")
            self.prescriptions = pd.read_csv(self.hosp_path / "prescriptions.csv")
            logger.info(f"  ✓ Loaded {len(self.prescriptions):,} prescriptions")

        if include_vitals:
            logger.info("  Loading chartevents.csv (this may take a while)...")
            # Note: chartevents.csv is VERY large
            try:
                self.chartevents = pd.read_csv(
                    self.icu_path / "chartevents.csv",
                    nrows=1000000  # Limit for memory
                )
                logger.info(f"  ✓ Loaded {len(self.chartevents):,} chart events (sampled)")
            except Exception as e:
                logger.warning(f"  ⚠ Could not load chartevents: {e}")
                self.chartevents = None

    def build_base_dataset(self) -> pd.DataFrame:
        """
        Build base dataset with patient, admission, and primary diagnosis

        Returns:
            DataFrame with base patient information
        """
        logger.info("\n" + "="*80)
        logger.info("BUILDING BASE DATASET")
        logger.info("="*80)

        # Get primary diagnoses only
        primary_dx = self.diagnoses[self.diagnoses['seq_num'] == 1].copy()

        # Filter for top 20 diagnoses
        primary_dx = primary_dx[primary_dx['icd_code'].isin(self.top_icd_codes)]
        logger.info(f"Found {len(primary_dx):,} admissions with top 20 primary diagnoses")

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

        # Merge with ICD descriptions
        dataset = dataset.merge(
            self.icd_descriptions,
            on=['icd_code', 'icd_version'],
            how='left'
        )
        logger.info(f"After ICD description merge: {len(dataset):,} records")

        # Rename for clarity
        dataset = dataset.rename(columns={
            'long_title': 'primary_diagnosis_name'
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

    def add_lab_summary(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Add aggregated lab results"""
        if self.labevents is None:
            logger.warning("  ⚠ Skipping labs (not loaded)")
            return dataset

        logger.info("\nAdding lab results...")

        # Get lab dictionary
        lab_items = pd.read_csv(self.hosp_path / "d_labitems.csv")

        # For each admission, get key lab values
        # This is a simplified version - you may want to aggregate differently
        lab_summary = self.labevents.merge(lab_items, on='itemid', how='left')

        # Pivot to get one row per admission with lab values
        # (This is memory intensive - consider sampling)
        lab_pivot = lab_summary.groupby(['hadm_id', 'label']).agg({
            'valuenum': ['mean', 'min', 'max', 'count']
        }).reset_index()

        # Convert to JSON for storage
        lab_json = lab_pivot.groupby('hadm_id').apply(
            lambda x: x.to_dict('records')
        ).reset_index()
        lab_json.columns = ['hadm_id', 'lab_results']

        dataset = dataset.merge(lab_json, on='hadm_id', how='left')
        logger.info(f"  ✓ Added lab summaries: {dataset['lab_results'].notna().sum():,} records")

        return dataset

    def create_annotation_dataset(self, output_path: str, sample_size: Optional[int] = None):
        """
        Create dataset for TASK 1: Annotation (Evidence Extraction)

        Format: Each row has diagnosis + clinical text, model extracts evidence

        Args:
            output_path: Where to save the dataset
            sample_size: Limit number of records (for testing)
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING ANNOTATION DATASET (TASK 1)")
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
            'icd_code',
            'icd_version',
            'primary_diagnosis_name',
            'clinical_text',
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
        self._print_diagnosis_distribution(annotation_dataset)

        return annotation_dataset

    def create_classification_dataset(self, output_path: str, sample_size: Optional[int] = None):
        """
        Create dataset for TASK 2: Classification (Diagnosis Prediction)

        Format: Each row has clinical text only, model predicts diagnosis

        Args:
            output_path: Where to save the dataset
            sample_size: Limit number of records (for testing)
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING CLASSIFICATION DATASET (TASK 2)")
        logger.info("="*80)

        # This uses the same data as annotation but formatted differently
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
            # Ground truth (for evaluation):
            'icd_code',
            'icd_version',
            'primary_diagnosis_name'
        ]

        classification_dataset = dataset[classification_columns].copy()

        # Save
        classification_dataset.to_csv(output_path, index=False)
        logger.info(f"\n✓ Classification dataset saved to: {output_path}")
        logger.info(f"✓ Total records: {len(classification_dataset):,}")
        logger.info(f"✓ Columns: {list(classification_dataset.columns)}")

        return classification_dataset

    def _print_diagnosis_distribution(self, dataset: pd.DataFrame):
        """Print distribution of diagnoses in dataset"""
        logger.info("\n" + "-"*80)
        logger.info("DIAGNOSIS DISTRIBUTION IN DATASET")
        logger.info("-"*80)

        dist = dataset.groupby(['icd_code', 'primary_diagnosis_name']).size().reset_index(name='count')
        dist = dist.sort_values('count', ascending=False)

        for idx, row in dist.iterrows():
            pct = (row['count'] / len(dataset)) * 100
            logger.info(f"  {row['icd_code']:10s} - {row['count']:5d} ({pct:5.1f}%) - {row['primary_diagnosis_name']}")

        logger.info("-"*80 + "\n")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("MIMIC-IV DATASET EXTRACTION")
    print("="*80)

    # Get paths from user
    mimic_path = input("\nEnter path to MIMIC-IV directory: ").strip()
    top_diagnoses_path = input("Enter path to top_20_primary_diagnoses.csv [mimic-iv/top_20_primary_diagnoses.csv]: ").strip()

    if not top_diagnoses_path:
        top_diagnoses_path = "mimic-iv/top_20_primary_diagnoses.csv"

    sample_size = input("Enter sample size (or press Enter for all records): ").strip()
    sample_size = int(sample_size) if sample_size else None

    try:
        # Initialize builder
        builder = MIMICDatasetBuilder(mimic_path, top_diagnoses_path)

        # Create annotation dataset
        annotation_output = "mimic-iv/annotation_dataset.csv"
        builder.create_annotation_dataset(annotation_output, sample_size)

        # Create classification dataset
        classification_output = "mimic-iv/classification_dataset.csv"
        builder.create_classification_dataset(classification_output, sample_size)

        print("\n" + "="*80)
        print("✓ DATASET EXTRACTION COMPLETE!")
        print("="*80)
        print(f"✓ Annotation dataset: {annotation_output}")
        print(f"✓ Classification dataset: {classification_output}")
        print(f"\n✓ Next steps:")
        print(f"  1. Review the datasets")
        print(f"  2. Use the prompts in mimic-iv/prompts/ directory")
        print(f"  3. Load JSON schemas from mimic-iv/schemas/ directory")
        print("="*80 + "\n")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
