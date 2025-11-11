"""
MIMIC-IV Clinical Notes Analysis Script (Post-Extraction)

This script analyzes the clinical notes from the extracted datasets
(annotation_dataset.csv or classification_dataset.csv) created by extract_dataset.py.

It consolidates ICD-9 and ICD-10 codes for the same diagnosis and provides:
- Average, median, min, max text length (characters) for each diagnosis
- Word count statistics
- Text length distributions
- Visualizations

This helps understand text complexity across the top 20 diagnoses before
processing with LLMs.

Author: Claude
Date: 2025-11-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import logging
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


class ClinicalNotesAnalyzer:
    """Analyze clinical notes from extracted MIMIC-IV datasets"""

    def __init__(self, dataset_path: str):
        """
        Initialize analyzer

        Args:
            dataset_path: Path to annotation_dataset.csv or classification_dataset.csv
        """
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        logger.info(f"✓ Dataset path verified: {self.dataset_path}")

    def consolidate_diagnosis_names(self, df):
        """
        Consolidate ICD-9 and ICD-10 codes for the same diagnosis

        Groups diagnoses by similar names (e.g., "Sepsis" ICD-9 and ICD-10)

        Args:
            df: DataFrame with icd_code and primary_diagnosis_name columns

        Returns:
            DataFrame with added consolidated_diagnosis column
        """
        logger.info("Consolidating ICD-9 and ICD-10 codes for same diagnoses...")

        # Create a mapping based on diagnosis names
        # This groups similar diagnoses together
        diagnosis_mapping = {}

        for _, row in df[['icd_code', 'primary_diagnosis_name']].drop_duplicates().iterrows():
            icd_code = row['icd_code']
            diagnosis_name = row['primary_diagnosis_name']

            # Normalize diagnosis name for grouping
            # Remove common suffixes that differ between ICD versions
            normalized = diagnosis_name.lower()

            # Remove ICD-specific qualifiers
            normalized = normalized.replace('unspecified', '').strip()
            normalized = normalized.replace(', unspecified', '').strip()

            # Group similar diagnoses
            if 'sepsis' in normalized or 'septicemia' in normalized:
                consolidated = 'Sepsis'
            elif 'chest pain' in normalized:
                consolidated = 'Chest Pain'
            elif 'pneumonia' in normalized:
                consolidated = 'Pneumonia'
            elif 'coronary' in normalized or 'atherosclerosis' in normalized:
                consolidated = 'Coronary Artery Disease'
            elif 'myocardial infarction' in normalized or 'nstemi' in normalized or 'subendocardial' in normalized:
                consolidated = 'Myocardial Infarction'
            elif 'atrial fibrillation' in normalized or 'atrial flutter' in normalized:
                consolidated = 'Atrial Fibrillation'
            elif 'heart failure' in normalized or 'chf' in normalized:
                consolidated = 'Heart Failure'
            elif 'hypertensive' in normalized and ('heart' in normalized or 'kidney' in normalized or 'ckd' in normalized):
                consolidated = 'Hypertensive Heart/Kidney Disease'
            elif 'kidney' in normalized or 'renal' in normalized or 'acute kidney' in normalized:
                consolidated = 'Acute Kidney Injury'
            elif 'respiratory failure' in normalized:
                consolidated = 'Respiratory Failure'
            elif 'copd' in normalized or 'obstructive' in normalized:
                consolidated = 'COPD'
            elif 'urinary' in normalized or 'uti' in normalized:
                consolidated = 'Urinary Tract Infection'
            elif 'depression' in normalized or 'depressive' in normalized:
                consolidated = 'Depression'
            elif 'alcohol' in normalized:
                consolidated = 'Alcohol Use Disorder'
            elif 'diabetes' in normalized:
                consolidated = 'Diabetes'
            elif 'stroke' in normalized or 'cerebral' in normalized:
                consolidated = 'Stroke'
            elif 'gastrointestinal' in normalized or 'gi bleed' in normalized:
                consolidated = 'Gastrointestinal Bleeding'
            elif 'chemotherapy' in normalized or 'antineoplastic' in normalized:
                consolidated = 'Chemotherapy Encounter'
            elif 'dehydration' in normalized:
                consolidated = 'Dehydration'
            else:
                # Use first 40 characters of original name
                consolidated = diagnosis_name[:40].strip()

            diagnosis_mapping[icd_code] = {
                'consolidated_diagnosis': consolidated,
                'original_diagnosis': diagnosis_name
            }

        # Apply mapping
        df['consolidated_diagnosis'] = df['icd_code'].map(
            lambda x: diagnosis_mapping.get(x, {}).get('consolidated_diagnosis', 'Other')
        )

        # Log consolidation results
        logger.info(f"\nConsolidation Results:")
        logger.info(f"  Original ICD codes: {df['icd_code'].nunique()}")
        logger.info(f"  Consolidated diagnoses: {df['consolidated_diagnosis'].nunique()}")

        consolidation_summary = df.groupby('consolidated_diagnosis').agg({
            'icd_code': lambda x: ', '.join(sorted(set(x))),
            'primary_diagnosis_name': lambda x: list(set(x))[0]
        }).reset_index()

        logger.info("\nConsolidation Mapping:")
        for _, row in consolidation_summary.iterrows():
            logger.info(f"  {row['consolidated_diagnosis']}: ICD codes {row['icd_code']}")

        return df

    def calculate_text_statistics(self, df):
        """
        Calculate text statistics for each record

        Args:
            df: DataFrame with clinical_text column

        Returns:
            DataFrame with added text statistics columns
        """
        logger.info("\nCalculating text statistics...")

        # Calculate character count
        df['char_count'] = df['clinical_text'].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )

        # Calculate word count
        df['word_count'] = df['clinical_text'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )

        # Calculate line count
        df['line_count'] = df['clinical_text'].apply(
            lambda x: str(x).count('\n') + 1 if pd.notna(x) else 0
        )

        logger.info(f"  ✓ Calculated statistics for {len(df):,} records")

        return df

    def analyze_by_diagnosis(self, df):
        """
        Analyze text length by consolidated diagnosis

        Args:
            df: DataFrame with consolidated_diagnosis and text statistics

        Returns:
            DataFrame with per-diagnosis statistics
        """
        logger.info("\n" + "="*80)
        logger.info("ANALYZING TEXT LENGTH BY DIAGNOSIS (TOP 20)")
        logger.info("="*80)

        # Group by consolidated diagnosis
        diagnosis_stats = df.groupby('consolidated_diagnosis').agg({
            'char_count': ['count', 'mean', 'median', 'min', 'max', 'std'],
            'word_count': ['mean', 'median', 'min', 'max'],
            'line_count': ['mean', 'median'],
            'icd_code': lambda x: ', '.join(sorted(set(x)))
        }).reset_index()

        # Flatten column names
        diagnosis_stats.columns = [
            'diagnosis',
            'n_cases', 'char_mean', 'char_median', 'char_min', 'char_max', 'char_std',
            'word_mean', 'word_median', 'word_min', 'word_max',
            'line_mean', 'line_median',
            'icd_codes'
        ]

        # Sort by number of cases
        diagnosis_stats = diagnosis_stats.sort_values('n_cases', ascending=False)

        # Get top 20
        diagnosis_stats = diagnosis_stats.head(20)

        # Print results
        logger.info(f"\nTop 20 Diagnoses (Consolidated):")
        logger.info("-"*80)

        for idx, row in diagnosis_stats.iterrows():
            logger.info(f"\n{row['diagnosis']}")
            logger.info(f"  ICD Codes: {row['icd_codes']}")
            logger.info(f"  Cases: {row['n_cases']:,}")
            logger.info(f"  Characters: avg={row['char_mean']:,.0f}, "
                       f"median={row['char_median']:,.0f}, "
                       f"min={row['char_min']:,}, "
                       f"max={row['char_max']:,}, "
                       f"std={row['char_std']:,.0f}")
            logger.info(f"  Words: avg={row['word_mean']:,.0f}, "
                       f"median={row['word_median']:,.0f}, "
                       f"min={row['word_min']:,}, "
                       f"max={row['word_max']:,}")
            logger.info(f"  Lines: avg={row['line_mean']:.1f}, "
                       f"median={row['line_median']:.0f}")

        logger.info("-"*80)

        return diagnosis_stats

    def analyze_overall_statistics(self, df):
        """
        Analyze overall dataset statistics

        Args:
            df: DataFrame with text statistics
        """
        logger.info("\n" + "="*80)
        logger.info("OVERALL DATASET STATISTICS")
        logger.info("="*80)

        logger.info(f"\nTotal Records: {len(df):,}")

        logger.info(f"\nCharacter Count:")
        logger.info(f"  Mean:   {df['char_count'].mean():,.0f}")
        logger.info(f"  Median: {df['char_count'].median():,.0f}")
        logger.info(f"  Min:    {df['char_count'].min():,}")
        logger.info(f"  Max:    {df['char_count'].max():,}")
        logger.info(f"  Std:    {df['char_count'].std():,.0f}")
        logger.info(f"  Q1:     {df['char_count'].quantile(0.25):,.0f}")
        logger.info(f"  Q3:     {df['char_count'].quantile(0.75):,.0f}")

        logger.info(f"\nWord Count:")
        logger.info(f"  Mean:   {df['word_count'].mean():,.0f}")
        logger.info(f"  Median: {df['word_count'].median():,.0f}")
        logger.info(f"  Min:    {df['word_count'].min():,}")
        logger.info(f"  Max:    {df['word_count'].max():,}")

        logger.info(f"\nLine Count:")
        logger.info(f"  Mean:   {df['line_count'].mean():.1f}")
        logger.info(f"  Median: {df['line_count'].median():.0f}")

        logger.info("="*80)

    def create_visualizations(self, df, diagnosis_stats, output_dir="mimic-iv"):
        """
        Create visualizations of text length distributions

        Args:
            df: DataFrame with text statistics
            diagnosis_stats: DataFrame with per-diagnosis statistics
            output_dir: Directory to save visualizations
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING VISUALIZATIONS")
        logger.info("="*80)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Figure 1: Overall distribution and top diagnoses
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Overall character count distribution
        ax = axes[0, 0]
        ax.hist(df['char_count'], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Character Count')
        ax.set_ylabel('Frequency')
        ax.set_title('Overall Character Count Distribution')
        ax.axvline(df['char_count'].mean(), color='red',
                   linestyle='--', label=f"Mean: {df['char_count'].mean():,.0f}")
        ax.axvline(df['char_count'].median(), color='green',
                   linestyle='--', label=f"Median: {df['char_count'].median():,.0f}")
        ax.legend()
        ax.grid(alpha=0.3)

        # Top 20 diagnoses - average character count
        ax = axes[0, 1]
        top_10_for_plot = diagnosis_stats.head(10)
        y_pos = np.arange(len(top_10_for_plot))
        ax.barh(y_pos, top_10_for_plot['char_mean'], alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([d[:25] + '...' if len(d) > 25 else d
                            for d in top_10_for_plot['diagnosis']], fontsize=8)
        ax.set_xlabel('Average Character Count')
        ax.set_title('Top 10 Diagnoses - Average Text Length')
        ax.grid(axis='x', alpha=0.3)

        # Box plot - character count by top 10 diagnoses
        ax = axes[1, 0]
        top_10_diagnoses = diagnosis_stats.head(10)['diagnosis'].values
        data_to_plot = []
        labels = []
        for diag in top_10_diagnoses:
            subset = df[df['consolidated_diagnosis'] == diag]['char_count']
            if len(subset) > 0:
                data_to_plot.append(subset)
                label = diag[:20] + '...' if len(diag) > 20 else diag
                labels.append(label)

        if data_to_plot:
            ax.boxplot(data_to_plot, labels=labels)
            ax.set_ylabel('Character Count')
            ax.set_title('Character Count Distribution - Top 10 Diagnoses')
            ax.tick_params(axis='x', rotation=45, labelsize=7)
            ax.grid(axis='y', alpha=0.3)

        # Word count vs character count scatter
        ax = axes[1, 1]
        # Sample for performance
        sample_df = df.sample(min(5000, len(df)), random_state=42)
        ax.scatter(sample_df['word_count'], sample_df['char_count'],
                  alpha=0.3, s=10)
        ax.set_xlabel('Word Count')
        ax.set_ylabel('Character Count')
        ax.set_title('Word Count vs Character Count')
        ax.grid(alpha=0.3)

        plt.tight_layout()

        fig_path = output_path / "clinical_notes_analysis_consolidated.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved visualization: {fig_path}")
        plt.close()

        # Figure 2: Top 20 diagnoses comparison
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # All top 20 - average character count
        ax = axes[0]
        y_pos = np.arange(len(diagnosis_stats))
        colors = plt.cm.viridis(np.linspace(0, 1, len(diagnosis_stats)))
        ax.barh(y_pos, diagnosis_stats['char_mean'], alpha=0.8, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{d[:30]}..." if len(d) > 30 else d
                            for d in diagnosis_stats['diagnosis']], fontsize=8)
        ax.set_xlabel('Average Character Count')
        ax.set_title('Top 20 Diagnoses - Average Text Length (Characters)')
        ax.grid(axis='x', alpha=0.3)

        # All top 20 - case counts
        ax = axes[1]
        ax.barh(y_pos, diagnosis_stats['n_cases'], alpha=0.8, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{d[:30]}..." if len(d) > 30 else d
                            for d in diagnosis_stats['diagnosis']], fontsize=8)
        ax.set_xlabel('Number of Cases')
        ax.set_title('Top 20 Diagnoses - Case Counts')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        fig_path = output_path / "clinical_notes_top20_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved visualization: {fig_path}")
        plt.close()

        logger.info("="*80)

    def save_results(self, df, diagnosis_stats, output_dir="mimic-iv"):
        """
        Save analysis results to CSV

        Args:
            df: DataFrame with text statistics
            diagnosis_stats: DataFrame with per-diagnosis statistics
            output_dir: Directory to save results
        """
        logger.info("\nSaving results...")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Save per-diagnosis statistics (top 20)
        diag_path = output_path / "clinical_notes_by_diagnosis_top20.csv"
        diagnosis_stats.to_csv(diag_path, index=False)
        logger.info(f"  ✓ Saved diagnosis statistics: {diag_path}")

        # Save summary
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "dataset_path": str(self.dataset_path),
            "total_records": len(df),
            "unique_diagnoses": df['consolidated_diagnosis'].nunique(),
            "top_20_diagnoses": diagnosis_stats['diagnosis'].tolist(),
            "overall_statistics": {
                "char_count": {
                    "mean": float(df['char_count'].mean()),
                    "median": float(df['char_count'].median()),
                    "min": int(df['char_count'].min()),
                    "max": int(df['char_count'].max()),
                    "std": float(df['char_count'].std()),
                    "q1": float(df['char_count'].quantile(0.25)),
                    "q3": float(df['char_count'].quantile(0.75))
                },
                "word_count": {
                    "mean": float(df['word_count'].mean()),
                    "median": float(df['word_count'].median()),
                    "min": int(df['word_count'].min()),
                    "max": int(df['word_count'].max())
                },
                "line_count": {
                    "mean": float(df['line_count'].mean()),
                    "median": float(df['line_count'].median())
                }
            }
        }

        import json
        summary_path = output_path / "clinical_notes_analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  ✓ Saved summary: {summary_path}")

    def run_analysis(self, output_dir="mimic-iv"):
        """
        Run complete analysis pipeline

        Args:
            output_dir: Directory to save outputs
        """
        logger.info("\n" + "="*80)
        logger.info("LOADING DATASET")
        logger.info("="*80)

        # Load dataset
        logger.info(f"Reading: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        logger.info(f"  ✓ Loaded {len(df):,} records")
        logger.info(f"  ✓ Columns: {list(df.columns)}")

        # Verify required columns
        required_cols = ['clinical_text', 'icd_code', 'primary_diagnosis_name']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Consolidate diagnosis names
        df = self.consolidate_diagnosis_names(df)

        # Calculate text statistics
        df = self.calculate_text_statistics(df)

        # Overall statistics
        self.analyze_overall_statistics(df)

        # Analyze by diagnosis (top 20)
        diagnosis_stats = self.analyze_by_diagnosis(df)

        # Create visualizations
        self.create_visualizations(df, diagnosis_stats, output_dir)

        # Save results
        self.save_results(df, diagnosis_stats, output_dir)

        return df, diagnosis_stats


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("MIMIC-IV CLINICAL NOTES ANALYSIS (POST-EXTRACTION)")
    print("Analyzes clinical text from extracted datasets with consolidated diagnoses")
    print("="*80)

    # Get dataset path
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        print("\nThis script analyzes datasets created by extract_dataset.py")
        print("It consolidates ICD-9/ICD-10 codes for the same diagnosis.")
        dataset_path = input("\nEnter path to dataset CSV (annotation_dataset.csv or classification_dataset.csv): ").strip()

    if not dataset_path:
        print("❌ No path provided")
        sys.exit(1)

    try:
        # Run analysis
        analyzer = ClinicalNotesAnalyzer(dataset_path)
        df, diagnosis_stats = analyzer.run_analysis()

        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE!")
        print("="*80)
        print("\nFiles created:")
        print("  - clinical_notes_by_diagnosis_top20.csv (per-diagnosis statistics)")
        print("  - clinical_notes_analysis_summary.json (overall summary)")
        print("  - clinical_notes_analysis_consolidated.png (visualizations)")
        print("  - clinical_notes_top20_comparison.png (top 20 comparison)")

        print("\nKey Findings:")
        print(f"  Total records: {len(df):,}")
        print(f"  Consolidated diagnoses: {df['consolidated_diagnosis'].nunique()}")
        print(f"  Top 20 diagnoses analyzed")
        print(f"  Overall avg length: {df['char_count'].mean():,.0f} chars, "
              f"median: {df['char_count'].median():,.0f} chars")

        print("\nTop 5 Diagnoses by Case Count:")
        for idx, row in diagnosis_stats.head(5).iterrows():
            print(f"  {row['diagnosis']}: {row['n_cases']:,} cases, "
                  f"avg {row['char_mean']:,.0f} chars")

        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
