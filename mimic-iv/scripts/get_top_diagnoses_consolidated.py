"""
MIMIC-IV Top Diagnoses Extraction with Consolidation

This script identifies the top primary diagnoses in MIMIC-IV and consolidates
diagnoses that are the same clinical condition but have different ICD codes
(ICD-9 vs ICD-10).

Example: "Chest pain (78650)", "Other chest pain (78659)", "Chest pain (R079)"
         are consolidated into a single "Chest pain" diagnosis.
"""

import pandas as pd
import sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from diagnosis_mapping import (
    DIAGNOSIS_CONSOLIDATION,
    CONSOLIDATED_DIAGNOSES,
    get_consolidated_diagnosis
)


def get_consolidated_top_diagnoses(mimic_path, top_n_codes=60):
    """
    Get top primary diagnoses with consolidation.

    Args:
        mimic_path: Path to MIMIC-IV root directory
        top_n_codes: Number of individual ICD codes to consider before consolidation
                     (default 25 to ensure we capture all relevant codes)

    Returns:
        DataFrame with consolidated diagnoses sorted by total case count
    """

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

        # Get top N codes (before consolidation)
        top_codes = counts.head(top_n_codes)

        # Merge with descriptions
        top_codes = top_codes.merge(icd_desc, on=['icd_code', 'icd_version'], how='left')

        print(f"\n✓ Found {len(top_codes)} top individual ICD codes")
        print("✓ Now consolidating by clinical condition...")

        # Consolidate diagnoses
        consolidated_data = []

        for diagnosis_id, info in sorted(CONSOLIDATED_DIAGNOSES.items()):
            # Get counts for all ICD codes belonging to this diagnosis
            diagnosis_codes = info['all_codes']
            diagnosis_rows = top_codes[top_codes['icd_code'].isin(diagnosis_codes)]

            if len(diagnosis_rows) > 0:
                total_cases = diagnosis_rows['count'].sum()

                # Create consolidated entry
                consolidated_entry = {
                    'diagnosis_id': diagnosis_id,
                    'diagnosis_name': info['name'],
                    'category': info['category'],
                    'total_cases': total_cases,
                    'icd9_codes': ', '.join(info['icd9_codes']) if info['icd9_codes'] else 'N/A',
                    'icd10_codes': ', '.join(info['icd10_codes']) if info['icd10_codes'] else 'N/A',
                    'num_codes': len(diagnosis_codes),
                    'description': info['description'],
                    # Individual code details
                    'code_details': []
                }

                for _, row in diagnosis_rows.iterrows():
                    consolidated_entry['code_details'].append({
                        'icd_code': row['icd_code'],
                        'icd_version': row['icd_version'],
                        'count': row['count'],
                        'long_title': row['long_title']
                    })

                consolidated_data.append(consolidated_entry)

        # Sort by total cases
        consolidated_data.sort(key=lambda x: x['total_cases'], reverse=True)

        # Create DataFrame
        consolidated_df = pd.DataFrame([
            {
                'diagnosis_id': d['diagnosis_id'],
                'diagnosis_name': d['diagnosis_name'],
                'category': d['category'],
                'total_cases': d['total_cases'],
                'num_codes': d['num_codes'],
                'icd9_codes': d['icd9_codes'],
                'icd10_codes': d['icd10_codes'],
                'description': d['description']
            }
            for d in consolidated_data
        ])

        # Create output directory if it doesn't exist
        output_dir = Path("mimic-iv")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save consolidated results
        output_file = output_dir / "top_diagnoses_consolidated.csv"
        consolidated_df.to_csv(output_file, index=False)

        # Save detailed breakdown
        detailed_output = output_dir / "top_diagnoses_detailed_breakdown.csv"
        detailed_rows = []
        for d in consolidated_data:
            for code_detail in d['code_details']:
                detailed_rows.append({
                    'diagnosis_id': d['diagnosis_id'],
                    'diagnosis_name': d['diagnosis_name'],
                    'category': d['category'],
                    'icd_code': code_detail['icd_code'],
                    'icd_version': code_detail['icd_version'],
                    'count': code_detail['count'],
                    'long_title': code_detail['long_title'],
                    'total_cases_consolidated': d['total_cases']
                })
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_df.to_csv(detailed_output, index=False)

        # Print results
        print("\n" + "="*80)
        print("CONSOLIDATED TOP DIAGNOSES (BY CLINICAL CONDITION)")
        print("="*80)

        total_cases_all = 0
        for idx, d in enumerate(consolidated_data, 1):
            total_cases_all += d['total_cases']
            print(f"\n{idx:2d}. {d['diagnosis_name']} ({d['category'].upper()})")
            print(f"    Total Cases: {d['total_cases']:,}")
            print(f"    ICD-9: {d['icd9_codes']}")
            print(f"    ICD-10: {d['icd10_codes']}")
            print(f"    Code Breakdown:")
            for code_detail in d['code_details']:
                print(f"      - {code_detail['icd_code']} (ICD-{code_detail['icd_version']}): {code_detail['count']:,} cases - {code_detail['long_title']}")

        print("\n" + "="*80)
        print(f"✓ Total consolidated diagnoses: {len(consolidated_data)}")
        print(f"✓ Total cases: {total_cases_all:,}")
        print(f"✓ Original ICD codes: {sum(d['num_codes'] for d in consolidated_data)}")
        print(f"✓ Saved consolidated list to: {output_file}")
        print(f"✓ Saved detailed breakdown to: {detailed_output}")
        print("="*80)

        return consolidated_df, detailed_df

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
    print("MIMIC-IV CONSOLIDATED DIAGNOSIS EXTRACTOR")
    print("Consolidates same conditions with different ICD codes (ICD-9 vs ICD-10)")
    print("="*80)

    if len(sys.argv) > 1:
        mimic_path = sys.argv[1]
    else:
        mimic_path = input("\nEnter path to MIMIC-IV directory: ").strip()

    if not mimic_path:
        print("❌ No path provided")
        sys.exit(1)

    get_consolidated_top_diagnoses(mimic_path)

    print("\n✓ Done! You now have 13 consolidated diagnoses.")
    print("✓ Next: Update your datasets using the consolidation mapping.\n")
