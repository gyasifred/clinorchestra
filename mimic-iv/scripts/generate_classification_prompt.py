"""
Generate Classification Prompt with Top 20 Diagnoses

This script reads the top_20_primary_diagnoses.csv and inserts the diagnosis list
into the classification prompt template.
"""

import pandas as pd
from pathlib import Path


def generate_classification_prompt_with_diagnoses(
    top_diagnoses_path: str,
    template_path: str,
    output_path: str
):
    """
    Generate classification prompt with top 20 diagnoses inserted

    Args:
        top_diagnoses_path: Path to top_20_primary_diagnoses.csv
        template_path: Path to task2_classification_prompt.txt
        output_path: Where to save the generated prompt
    """
    # Load top diagnoses
    print(f"Loading top diagnoses from: {top_diagnoses_path}")
    top_diagnoses = pd.read_csv(top_diagnoses_path)

    # Generate diagnosis list text
    diagnosis_list_text = "\n".join([
        f"{idx+1}. {row['long_title']} (ICD-{row['icd_version']}: {row['icd_code']}) - {row['count']:,} cases in dataset"
        for idx, row in top_diagnoses.iterrows()
    ])

    # Read template
    print(f"Reading template from: {template_path}")
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    # Replace placeholder
    # The template has: [This list will be dynamically generated based on your top_20_primary_diagnoses.csv]
    # We'll replace the entire section
    placeholder_start = "TOP 20 POSSIBLE PRIMARY DIAGNOSES:"
    placeholder_end = "YOUR TASK:"

    # Find the section to replace
    start_idx = template.find(placeholder_start)
    end_idx = template.find(placeholder_end)

    if start_idx == -1 or end_idx == -1:
        raise ValueError("Could not find placeholder section in template")

    # Construct new section
    new_section = f"""TOP 20 POSSIBLE PRIMARY DIAGNOSES:
Your prediction MUST be one of these diagnoses. These are the most common primary diagnoses in this dataset:

{diagnosis_list_text}

"""

    # Replace
    generated_prompt = template[:start_idx] + new_section + template[end_idx:]

    # Save
    print(f"Saving generated prompt to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(generated_prompt)

    print(f"\n✓ Classification prompt generated successfully!")
    print(f"✓ You can now copy this prompt into ClinOrchestra")

    # Also save a version with diagnosis codes in JSON format for reference
    diagnosis_json_path = output_path.replace('.txt', '_diagnoses.json')
    top_diagnoses.to_json(diagnosis_json_path, orient='records', indent=2)
    print(f"✓ Diagnosis list also saved as JSON: {diagnosis_json_path}")


def main():
    """Main execution"""
    print("="*80)
    print("CLASSIFICATION PROMPT GENERATOR")
    print("="*80)

    # Default paths
    top_diagnoses_path = "mimic-iv/top_20_primary_diagnoses.csv"
    template_path = "mimic-iv/prompts/task2_classification_prompt.txt"
    output_path = "mimic-iv/prompts/task2_classification_prompt_generated.txt"

    # Allow user override
    custom_path = input(f"\nPath to top_20_primary_diagnoses.csv [{top_diagnoses_path}]: ").strip()
    if custom_path:
        top_diagnoses_path = custom_path

    try:
        generate_classification_prompt_with_diagnoses(
            top_diagnoses_path,
            template_path,
            output_path
        )

        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print(f"1. Open: {output_path}")
        print(f"2. Copy the entire contents")
        print(f"3. In ClinOrchestra UI, go to Prompt tab")
        print(f"4. Paste the prompt")
        print(f"5. Load your classification_dataset.csv")
        print(f"6. Upload the JSON schema from schemas/task2_classification_schema.json")
        print(f"7. Start processing!")
        print("="*80)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\nMake sure you've run extract_top_diagnoses.py first!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
