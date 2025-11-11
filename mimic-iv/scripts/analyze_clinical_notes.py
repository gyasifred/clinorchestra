"""
MIMIC-IV Clinical Notes Analysis Script

Analyzes clinical notes to understand text characteristics:
- Average, median, min, max text length (characters)
- Word count and line count statistics
- Statistics by note type (discharge, radiology)
- Per-diagnosis text length analysis
- Visualizations of text length distributions

This helps understand the complexity and variability of clinical notes
before processing them with LLMs.

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10


class ClinicalNotesAnalyzer:
    """Analyze clinical notes from MIMIC-IV"""

    def __init__(self, mimic_path: str):
        """
        Initialize analyzer

        Args:
            mimic_path: Path to MIMIC-IV root directory
        """
        self.mimic_path = Path(mimic_path)
        self.note_path = self.mimic_path / "note"
        self.hosp_path = self.mimic_path / "hosp"

        # Verify paths
        if not self.note_path.exists():
            raise FileNotFoundError(f"Note directory not found: {self.note_path}")

        logger.info(f"✓ MIMIC-IV path verified: {self.mimic_path}")

    def load_notes(self):
        """Load discharge and radiology notes"""
        logger.info("="*80)
        logger.info("LOADING CLINICAL NOTES")
        logger.info("="*80)

        # Load discharge notes
        logger.info("Loading discharge.csv...")
        self.discharge_notes = pd.read_csv(self.note_path / "discharge.csv")
        logger.info(f"  ✓ Loaded {len(self.discharge_notes):,} discharge notes")

        # Load radiology notes
        logger.info("Loading radiology.csv...")
        self.radiology_notes = pd.read_csv(self.note_path / "radiology.csv")
        logger.info(f"  ✓ Loaded {len(self.radiology_notes):,} radiology notes")

        logger.info("="*80 + "\n")

    def calculate_text_statistics(self, text):
        """
        Calculate statistics for a text string

        Returns:
            dict with char_count, word_count, line_count
        """
        if pd.isna(text):
            return {
                'char_count': 0,
                'word_count': 0,
                'line_count': 0
            }

        text_str = str(text)
        return {
            'char_count': len(text_str),
            'word_count': len(text_str.split()),
            'line_count': text_str.count('\n') + 1
        }

    def analyze_note_type(self, df, note_type):
        """
        Analyze notes of a specific type

        Args:
            df: DataFrame with notes
            note_type: Type of note ('discharge' or 'radiology')

        Returns:
            DataFrame with statistics
        """
        logger.info(f"\nAnalyzing {note_type} notes...")

        # Calculate statistics for each note
        stats_list = []
        for idx, row in df.iterrows():
            if idx % 10000 == 0 and idx > 0:
                logger.info(f"  Processed {idx:,} notes...")

            text_stats = self.calculate_text_statistics(row['text'])
            stats_list.append({
                'subject_id': row.get('subject_id', None),
                'hadm_id': row.get('hadm_id', None),
                'note_id': row.get('note_id', None),
                'note_type': note_type,
                **text_stats
            })

        stats_df = pd.DataFrame(stats_list)

        # Print summary statistics
        logger.info(f"\n{note_type.upper()} NOTES STATISTICS")
        logger.info("-"*80)
        logger.info(f"Total notes: {len(stats_df):,}")
        logger.info(f"\nCharacter Count:")
        logger.info(f"  Mean:   {stats_df['char_count'].mean():,.0f}")
        logger.info(f"  Median: {stats_df['char_count'].median():,.0f}")
        logger.info(f"  Min:    {stats_df['char_count'].min():,}")
        logger.info(f"  Max:    {stats_df['char_count'].max():,}")
        logger.info(f"  Q1:     {stats_df['char_count'].quantile(0.25):,.0f}")
        logger.info(f"  Q3:     {stats_df['char_count'].quantile(0.75):,.0f}")

        logger.info(f"\nWord Count:")
        logger.info(f"  Mean:   {stats_df['word_count'].mean():,.0f}")
        logger.info(f"  Median: {stats_df['word_count'].median():,.0f}")
        logger.info(f"  Min:    {stats_df['word_count'].min():,}")
        logger.info(f"  Max:    {stats_df['word_count'].max():,}")

        logger.info(f"\nLine Count:")
        logger.info(f"  Mean:   {stats_df['line_count'].mean():.1f}")
        logger.info(f"  Median: {stats_df['line_count'].median():.0f}")
        logger.info(f"  Min:    {stats_df['line_count'].min()}")
        logger.info(f"  Max:    {stats_df['line_count'].max()}")

        return stats_df

    def analyze_by_diagnosis(self, top_diagnoses_path=None):
        """
        Analyze text length by diagnosis

        Args:
            top_diagnoses_path: Path to top diagnoses CSV (optional)
        """
        if top_diagnoses_path is None:
            logger.info("\nSkipping per-diagnosis analysis (no top_diagnoses file provided)")
            return None

        if not Path(top_diagnoses_path).exists():
            logger.warning(f"\nTop diagnoses file not found: {top_diagnoses_path}")
            return None

        logger.info("\n" + "="*80)
        logger.info("ANALYZING TEXT LENGTH BY DIAGNOSIS")
        logger.info("="*80)

        # Load diagnoses and top diagnoses
        logger.info("Loading diagnoses...")
        diagnoses = pd.read_csv(self.hosp_path / "diagnoses_icd.csv")
        top_diagnoses = pd.read_csv(top_diagnoses_path)
        icd_descriptions = pd.read_csv(self.hosp_path / "d_icd_diagnoses.csv")

        # Filter for primary diagnoses and top codes
        primary_dx = diagnoses[diagnoses['seq_num'] == 1]
        top_icd_codes = set(top_diagnoses['icd_code'].values)
        primary_dx = primary_dx[primary_dx['icd_code'].isin(top_icd_codes)]

        # Merge with descriptions
        primary_dx = primary_dx.merge(icd_descriptions, on=['icd_code', 'icd_version'], how='left')

        # Combine discharge and radiology notes per admission
        logger.info("Aggregating notes per admission...")

        # Discharge notes
        discharge_agg = self.discharge_notes.groupby('hadm_id')['text'].apply(
            lambda x: '\n\n---\n\n'.join(x)
        ).reset_index()
        discharge_agg.columns = ['hadm_id', 'combined_text']

        # Merge with diagnoses
        diagnosis_notes = primary_dx.merge(discharge_agg, on='hadm_id', how='inner')

        # Calculate text statistics
        logger.info("Calculating text statistics per diagnosis...")
        diagnosis_notes['char_count'] = diagnosis_notes['combined_text'].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )
        diagnosis_notes['word_count'] = diagnosis_notes['combined_text'].apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )

        # Group by diagnosis
        diagnosis_stats = diagnosis_notes.groupby(['icd_code', 'long_title']).agg({
            'char_count': ['count', 'mean', 'median', 'min', 'max', 'std'],
            'word_count': ['mean', 'median']
        }).reset_index()

        # Flatten column names
        diagnosis_stats.columns = [
            'icd_code', 'diagnosis',
            'n_cases', 'char_mean', 'char_median', 'char_min', 'char_max', 'char_std',
            'word_mean', 'word_median'
        ]

        # Sort by number of cases
        diagnosis_stats = diagnosis_stats.sort_values('n_cases', ascending=False)

        # Print results
        logger.info("\n" + "-"*80)
        logger.info("TEXT LENGTH BY DIAGNOSIS")
        logger.info("-"*80)
        for idx, row in diagnosis_stats.iterrows():
            logger.info(f"\n{row['icd_code']} - {row['diagnosis']}")
            logger.info(f"  Cases: {row['n_cases']:,}")
            logger.info(f"  Characters: {row['char_mean']:,.0f} (±{row['char_std']:,.0f}), "
                       f"median: {row['char_median']:,.0f}, range: {row['char_min']:,}-{row['char_max']:,}")
            logger.info(f"  Words: {row['word_mean']:,.0f}, median: {row['word_median']:,.0f}")

        return diagnosis_stats

    def create_visualizations(self, discharge_stats, radiology_stats, output_dir="mimic-iv"):
        """
        Create visualizations of text length distributions

        Args:
            discharge_stats: DataFrame with discharge note statistics
            radiology_stats: DataFrame with radiology note statistics
            output_dir: Directory to save visualizations
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING VISUALIZATIONS")
        logger.info("="*80)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Figure 1: Character count distributions
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Discharge notes - histogram
        ax = axes[0, 0]
        ax.hist(discharge_stats['char_count'], bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Character Count')
        ax.set_ylabel('Frequency')
        ax.set_title('Discharge Notes - Character Count Distribution')
        ax.axvline(discharge_stats['char_count'].mean(), color='red',
                   linestyle='--', label=f"Mean: {discharge_stats['char_count'].mean():,.0f}")
        ax.axvline(discharge_stats['char_count'].median(), color='green',
                   linestyle='--', label=f"Median: {discharge_stats['char_count'].median():,.0f}")
        ax.legend()
        ax.grid(alpha=0.3)

        # Radiology notes - histogram
        ax = axes[0, 1]
        ax.hist(radiology_stats['char_count'], bins=50, alpha=0.7, edgecolor='black', color='orange')
        ax.set_xlabel('Character Count')
        ax.set_ylabel('Frequency')
        ax.set_title('Radiology Notes - Character Count Distribution')
        ax.axvline(radiology_stats['char_count'].mean(), color='red',
                   linestyle='--', label=f"Mean: {radiology_stats['char_count'].mean():,.0f}")
        ax.axvline(radiology_stats['char_count'].median(), color='green',
                   linestyle='--', label=f"Median: {radiology_stats['char_count'].median():,.0f}")
        ax.legend()
        ax.grid(alpha=0.3)

        # Box plot comparison
        ax = axes[1, 0]
        data_to_plot = [discharge_stats['char_count'], radiology_stats['char_count']]
        ax.boxplot(data_to_plot, labels=['Discharge', 'Radiology'])
        ax.set_ylabel('Character Count')
        ax.set_title('Character Count Comparison')
        ax.grid(alpha=0.3)

        # Word count comparison
        ax = axes[1, 1]
        data_to_plot = [discharge_stats['word_count'], radiology_stats['word_count']]
        ax.boxplot(data_to_plot, labels=['Discharge', 'Radiology'])
        ax.set_ylabel('Word Count')
        ax.set_title('Word Count Comparison')
        ax.grid(alpha=0.3)

        plt.tight_layout()

        # Save figure
        fig_path = output_path / "clinical_notes_length_analysis.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved visualization: {fig_path}")
        plt.close()

        # Figure 2: Log scale for better visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Log scale histogram - discharge
        ax = axes[0]
        discharge_log = np.log10(discharge_stats['char_count'] + 1)
        ax.hist(discharge_log, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Log10(Character Count + 1)')
        ax.set_ylabel('Frequency')
        ax.set_title('Discharge Notes - Log Scale')
        ax.grid(alpha=0.3)

        # Log scale histogram - radiology
        ax = axes[1]
        radiology_log = np.log10(radiology_stats['char_count'] + 1)
        ax.hist(radiology_log, bins=50, alpha=0.7, edgecolor='black', color='orange')
        ax.set_xlabel('Log10(Character Count + 1)')
        ax.set_ylabel('Frequency')
        ax.set_title('Radiology Notes - Log Scale')
        ax.grid(alpha=0.3)

        plt.tight_layout()

        fig_path = output_path / "clinical_notes_length_analysis_log.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved visualization: {fig_path}")
        plt.close()

        logger.info("="*80 + "\n")

    def save_results(self, discharge_stats, radiology_stats, diagnosis_stats=None,
                     output_dir="mimic-iv"):
        """
        Save analysis results to CSV

        Args:
            discharge_stats: DataFrame with discharge note statistics
            radiology_stats: DataFrame with radiology note statistics
            diagnosis_stats: DataFrame with per-diagnosis statistics (optional)
            output_dir: Directory to save results
        """
        logger.info("Saving results...")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        # Save individual note statistics
        all_stats = pd.concat([discharge_stats, radiology_stats], ignore_index=True)
        stats_path = output_path / "clinical_notes_analysis.csv"
        all_stats.to_csv(stats_path, index=False)
        logger.info(f"  ✓ Saved note statistics: {stats_path}")

        # Save diagnosis statistics if available
        if diagnosis_stats is not None:
            diag_path = output_path / "clinical_notes_by_diagnosis.csv"
            diagnosis_stats.to_csv(diag_path, index=False)
            logger.info(f"  ✓ Saved diagnosis statistics: {diag_path}")

        # Create summary report
        summary = {
            "analysis_date": datetime.now().isoformat(),
            "discharge_notes": {
                "total_notes": len(discharge_stats),
                "char_count": {
                    "mean": float(discharge_stats['char_count'].mean()),
                    "median": float(discharge_stats['char_count'].median()),
                    "min": int(discharge_stats['char_count'].min()),
                    "max": int(discharge_stats['char_count'].max()),
                    "std": float(discharge_stats['char_count'].std()),
                    "q1": float(discharge_stats['char_count'].quantile(0.25)),
                    "q3": float(discharge_stats['char_count'].quantile(0.75))
                },
                "word_count": {
                    "mean": float(discharge_stats['word_count'].mean()),
                    "median": float(discharge_stats['word_count'].median()),
                    "min": int(discharge_stats['word_count'].min()),
                    "max": int(discharge_stats['word_count'].max())
                },
                "line_count": {
                    "mean": float(discharge_stats['line_count'].mean()),
                    "median": float(discharge_stats['line_count'].median())
                }
            },
            "radiology_notes": {
                "total_notes": len(radiology_stats),
                "char_count": {
                    "mean": float(radiology_stats['char_count'].mean()),
                    "median": float(radiology_stats['char_count'].median()),
                    "min": int(radiology_stats['char_count'].min()),
                    "max": int(radiology_stats['char_count'].max()),
                    "std": float(radiology_stats['char_count'].std()),
                    "q1": float(radiology_stats['char_count'].quantile(0.25)),
                    "q3": float(radiology_stats['char_count'].quantile(0.75))
                },
                "word_count": {
                    "mean": float(radiology_stats['word_count'].mean()),
                    "median": float(radiology_stats['word_count'].median()),
                    "min": int(radiology_stats['word_count'].min()),
                    "max": int(radiology_stats['word_count'].max())
                },
                "line_count": {
                    "mean": float(radiology_stats['line_count'].mean()),
                    "median": float(radiology_stats['line_count'].median())
                }
            }
        }

        import json
        summary_path = output_path / "clinical_notes_analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"  ✓ Saved summary: {summary_path}")

    def run_full_analysis(self, top_diagnoses_path=None, output_dir="mimic-iv"):
        """
        Run complete analysis pipeline

        Args:
            top_diagnoses_path: Optional path to top diagnoses CSV
            output_dir: Directory to save outputs
        """
        # Load notes
        self.load_notes()

        # Analyze discharge notes
        discharge_stats = self.analyze_note_type(self.discharge_notes, 'discharge')

        # Analyze radiology notes
        radiology_stats = self.analyze_note_type(self.radiology_notes, 'radiology')

        # Analyze by diagnosis (if top diagnoses provided)
        diagnosis_stats = None
        if top_diagnoses_path:
            diagnosis_stats = self.analyze_by_diagnosis(top_diagnoses_path)

        # Create visualizations
        self.create_visualizations(discharge_stats, radiology_stats, output_dir)

        # Save results
        self.save_results(discharge_stats, radiology_stats, diagnosis_stats, output_dir)

        return discharge_stats, radiology_stats, diagnosis_stats


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("MIMIC-IV CLINICAL NOTES ANALYSIS")
    print("Analyzes text length, word count, and complexity of clinical notes")
    print("="*80)

    # Get paths
    if len(sys.argv) > 1:
        mimic_path = sys.argv[1]
    else:
        mimic_path = input("\nEnter path to MIMIC-IV directory: ").strip()

    if not mimic_path:
        print("❌ No path provided")
        sys.exit(1)

    # Optional: top diagnoses for per-diagnosis analysis
    top_diagnoses_path = input("Path to top_20_primary_diagnoses.csv (or press Enter to skip): ").strip()
    if not top_diagnoses_path:
        top_diagnoses_path = None

    try:
        # Run analysis
        analyzer = ClinicalNotesAnalyzer(mimic_path)
        discharge_stats, radiology_stats, diagnosis_stats = analyzer.run_full_analysis(
            top_diagnoses_path=top_diagnoses_path
        )

        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE!")
        print("="*80)
        print("\nFiles created:")
        print("  - clinical_notes_analysis.csv (all note statistics)")
        print("  - clinical_notes_analysis_summary.json (summary statistics)")
        print("  - clinical_notes_length_analysis.png (visualizations)")
        print("  - clinical_notes_length_analysis_log.png (log scale)")
        if diagnosis_stats is not None:
            print("  - clinical_notes_by_diagnosis.csv (per-diagnosis statistics)")

        print("\nKey Findings:")
        print(f"  Discharge notes: {len(discharge_stats):,} notes, "
              f"avg {discharge_stats['char_count'].mean():,.0f} chars, "
              f"median {discharge_stats['char_count'].median():,.0f} chars")
        print(f"  Radiology notes: {len(radiology_stats):,} notes, "
              f"avg {radiology_stats['char_count'].mean():,.0f} chars, "
              f"median {radiology_stats['char_count'].median():,.0f} chars")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
