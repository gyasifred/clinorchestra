"""
Simple script to get top 20 primary diagnoses from MIMIC-IV

Just provide your MIMIC-IV path and this will output the top 20 diagnoses
"""

import pandas as pd
import sys

def get_top_20_diagnoses(mimic_path):
    """Get top 20 primary diagnoses from MIMIC-IV"""

    print(f"\nLoading data from: {mimic_path}")
    print("="*80)

    try:
        # Load diagnoses
        print("Loading diagnoses_icd.csv...")
        diagnoses = pd.read_csv(f"{mimic_path}/hosp/diagnoses_icd.csv")

        # Load ICD descriptions
        print("Loading d_icd_diagnoses.csv...")
        icd_desc = pd.read_csv(f"{mimic_path}/hosp/d_icd_diagnoses.csv")

        # Get primary diagnoses only (seq_num = 1)
        print("\nFiltering for primary diagnoses (seq_num = 1)...")
        primary = diagnoses[diagnoses['seq_num'] == 1]

        # Count by ICD code
        print("Counting diagnoses...")
        counts = primary.groupby(['icd_code', 'icd_version']).size().reset_index(name='count')

        # Sort by count
        counts = counts.sort_values('count', ascending=False)

        # Get top 20
        top_20 = counts.head(20)

        # Merge with descriptions
        top_20 = top_20.merge(icd_desc, on=['icd_code', 'icd_version'], how='left')

        # Save
        output_file = "mimic-iv/top_20_primary_diagnoses.csv"
        top_20.to_csv(output_file, index=False)

        print("\n" + "="*80)
        print("TOP 20 PRIMARY DIAGNOSES")
        print("="*80)

        for idx, row in top_20.iterrows():
            print(f"{idx+1:2d}. {row['icd_code']:10s} (ICD-{row['icd_version']}) - {row['count']:6,d} cases - {row['long_title']}")

        print("="*80)
        print(f"\n✓ Saved to: {output_file}")
        print(f"✓ Total primary diagnoses found: {len(primary):,}")
        print(f"✓ Top 20 account for: {top_20['count'].sum():,} cases")

        return top_20

    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find file")
        print(f"   {e}")
        print(f"\n   Make sure your path is correct and includes hosp/ folder")
        print(f"   Example: C:\\Users\\gyasi\\Documents\\mimic-iv-3.1")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("MIMIC-IV TOP 20 PRIMARY DIAGNOSES EXTRACTOR")
    print("="*80)

    if len(sys.argv) > 1:
        mimic_path = sys.argv[1]
    else:
        mimic_path = input("\nEnter path to MIMIC-IV directory: ").strip()

    if not mimic_path:
        print("❌ No path provided")
        sys.exit(1)

    get_top_20_diagnoses(mimic_path)

    print("\n✓ Done! You can now use this list to create datasets.\n")
