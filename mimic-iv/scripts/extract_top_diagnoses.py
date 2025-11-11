"""
MIMIC-IV Top 20 Primary Diagnoses Extraction Script

This script identifies the top 20 most common primary diagnoses in MIMIC-IV
based on the seq_num = 1 (primary diagnosis) in diagnoses_icd.csv
"""

import pandas as pd
import os
from pathlib import Path
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MIMICDiagnosisExtractor:
    """Extract top diagnoses from MIMIC-IV dataset"""

    def __init__(self, mimic_path: str):
        """
        Initialize the extractor

        Args:
            mimic_path: Path to MIMIC-IV root directory (containing hosp, icu, note folders)
        """
        self.mimic_path = Path(mimic_path)
        self.hosp_path = self.mimic_path / "hosp"
        self.icu_path = self.mimic_path / "icu"
        self.note_path = self.mimic_path / "note"

        # Verify paths exist
        self._verify_paths()

    def _verify_paths(self):
        """Verify that MIMIC-IV folders exist"""
        if not self.mimic_path.exists():
            raise FileNotFoundError(f"MIMIC-IV path not found: {self.mimic_path}")

        if not self.hosp_path.exists():
            raise FileNotFoundError(f"HOSP folder not found: {self.hosp_path}")

        logger.info(f"✓ MIMIC-IV path verified: {self.mimic_path}")

    def get_top_primary_diagnoses(self, n: int = 20) -> pd.DataFrame:
        """
        Get top N primary diagnoses

        Args:
            n: Number of top diagnoses to return

        Returns:
            DataFrame with columns: icd_code, icd_version, long_title, count
        """
        logger.info("Loading diagnoses_icd.csv...")
        diagnoses_df = pd.read_csv(self.hosp_path / "diagnoses_icd.csv")

        logger.info("Loading d_icd_diagnoses.csv...")
        icd_descriptions = pd.read_csv(self.hosp_path / "d_icd_diagnoses.csv")

        # Filter for primary diagnoses only (seq_num = 1)
        logger.info("Filtering for primary diagnoses (seq_num = 1)...")
        primary_diagnoses = diagnoses_df[diagnoses_df['seq_num'] == 1].copy()

        logger.info(f"Found {len(primary_diagnoses)} primary diagnoses")

        # Count occurrences of each ICD code
        diagnosis_counts = primary_diagnoses.groupby(['icd_code', 'icd_version']).size().reset_index(name='count')

        # Sort by count descending
        diagnosis_counts = diagnosis_counts.sort_values('count', ascending=False)

        # Get top N
        top_diagnoses = diagnosis_counts.head(n)

        # Merge with descriptions
        top_diagnoses = top_diagnoses.merge(
            icd_descriptions,
            on=['icd_code', 'icd_version'],
            how='left'
        )

        logger.info(f"\n{'='*80}")
        logger.info(f"TOP {n} PRIMARY DIAGNOSES IN MIMIC-IV")
        logger.info(f"{'='*80}")
        for idx, row in top_diagnoses.iterrows():
            logger.info(f"{idx+1:2d}. {row['icd_code']:10s} (ICD-{row['icd_version']}) - {row['count']:5d} cases - {row['long_title']}")
        logger.info(f"{'='*80}\n")

        return top_diagnoses

    def save_top_diagnoses(self, output_path: str, n: int = 20):
        """Save top diagnoses to CSV"""
        top_diagnoses = self.get_top_primary_diagnoses(n)
        top_diagnoses.to_csv(output_path, index=False)
        logger.info(f"Top {n} diagnoses saved to: {output_path}")
        return top_diagnoses


def main():
    """Main execution"""
    # YOU NEED TO SET THIS PATH TO YOUR MIMIC-IV DIRECTORY
    # Example: "C:\\Users\\gyasi\\Documents\\mimic-iv-3.1" (Windows)
    # Example: "/path/to/mimic-iv-3.1" (Linux/Mac)

    mimic_path = input("Enter path to MIMIC-IV directory: ").strip()

    if not mimic_path:
        logger.error("No path provided. Exiting.")
        return

    try:
        extractor = MIMICDiagnosisExtractor(mimic_path)

        # Extract and save top 20 diagnoses
        output_path = "mimic-iv/top_20_primary_diagnoses.csv"
        top_diagnoses = extractor.save_top_diagnoses(output_path, n=20)

        logger.info(f"\n✓ Success! Top 20 primary diagnoses identified and saved.")
        logger.info(f"✓ Next step: Run extract_dataset.py to create the full dataset")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
