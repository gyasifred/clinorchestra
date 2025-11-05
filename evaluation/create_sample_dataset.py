"""
Generate sample clinical datasets for testing clinAnnotate

Creates realistic synthetic clinical cases for:
- Malnutrition detection
- AKI (Acute Kidney Injury) detection
- Diabetes screening
- Serial measurements

Usage:
    python create_sample_dataset.py --task malnutrition --samples 50
"""

import pandas as pd
import random
import argparse
from datetime import datetime, timedelta
from typing import List, Dict


class SampleDatasetGenerator:
    """Generate synthetic clinical test cases"""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate_malnutrition_cases(self, n: int = 50) -> pd.DataFrame:
        """
        Generate malnutrition detection cases based on BMI and albumin

        Labels:
        0 = No malnutrition
        1 = Malnutrition risk
        2 = Moderate-severe malnutrition
        """
        cases = []

        for i in range(n):
            case_type = random.choice(['bmi_only', 'albumin_only', 'both', 'percentile', 'zscore'])

            if case_type == 'bmi_only':
                bmi = random.uniform(14.0, 30.0)
                if bmi < 16:
                    label = 2
                    reasoning = f"BMI {bmi:.1f} indicates severe malnutrition (WHO criteria: BMI < 16)"
                elif bmi < 18.5:
                    label = 1
                    reasoning = f"BMI {bmi:.1f} indicates malnutrition risk (BMI < 18.5)"
                else:
                    label = 0
                    reasoning = f"BMI {bmi:.1f} is within normal range (≥18.5)"

                text = f"Patient presents with BMI of {bmi:.1f} kg/m2."
                extracted_value = bmi

            elif case_type == 'albumin_only':
                albumin = random.uniform(2.0, 4.5)
                if albumin < 2.5:
                    label = 2
                    reasoning = f"Albumin {albumin:.1f} g/dL indicates severe malnutrition (< 2.5)"
                elif albumin < 3.5:
                    label = 1
                    reasoning = f"Albumin {albumin:.1f} g/dL indicates mild-moderate malnutrition (< 3.5)"
                else:
                    label = 0
                    reasoning = f"Albumin {albumin:.1f} g/dL is normal (≥3.5)"

                text = f"Labs show serum albumin of {albumin:.1f} g/dL."
                extracted_value = albumin

            elif case_type == 'both':
                bmi = random.uniform(14.0, 30.0)
                albumin = random.uniform(2.0, 4.5)

                bmi_severe = bmi < 16
                bmi_risk = bmi < 18.5
                alb_severe = albumin < 2.5
                alb_risk = albumin < 3.5

                if bmi_severe or alb_severe:
                    label = 2
                    reasoning = f"BMI {bmi:.1f} and albumin {albumin:.1f} indicate severe malnutrition"
                elif bmi_risk or alb_risk:
                    label = 1
                    reasoning = f"BMI {bmi:.1f} and/or albumin {albumin:.1f} indicate malnutrition risk"
                else:
                    label = 0
                    reasoning = f"BMI {bmi:.1f} and albumin {albumin:.1f} are within normal limits"

                text = f"Patient has BMI {bmi:.1f} kg/m2 and serum albumin {albumin:.1f} g/dL."
                extracted_value = f"{bmi:.1f},{albumin:.1f}"

            elif case_type == 'percentile':
                age_months = random.randint(24, 180)  # 2-15 years
                percentile = random.uniform(1, 99)

                if percentile < 3:
                    label = 2
                    reasoning = f"Growth at {percentile:.1f}th percentile (< 3rd) indicates severe malnutrition"
                elif percentile < 5:
                    label = 1
                    reasoning = f"Growth at {percentile:.1f}th percentile (< 5th) indicates malnutrition risk"
                else:
                    label = 0
                    reasoning = f"Growth at {percentile:.1f}th percentile is adequate"

                text = f"Child age {age_months} months with weight at {percentile:.1f}th percentile."
                extracted_value = percentile

            else:  # zscore
                zscore = random.uniform(-3.0, 2.0)

                if zscore < -2.0:
                    label = 2
                    reasoning = f"Z-score {zscore:.2f} (< -2) indicates moderate-severe malnutrition"
                elif zscore < -1.64:
                    label = 1
                    reasoning = f"Z-score {zscore:.2f} (< -1.64, below 5th percentile) indicates malnutrition risk"
                else:
                    label = 0
                    reasoning = f"Z-score {zscore:.2f} is adequate"

                text = f"Growth assessment shows weight-for-age z-score of {zscore:.2f}."
                extracted_value = zscore

            cases.append({
                'id': f'MAL_{i+1:03d}',
                'text': text,
                'label': label,
                'extracted_value': extracted_value,
                'reasoning': reasoning,
                'task': 'malnutrition'
            })

        return pd.DataFrame(cases)

    def generate_aki_cases(self, n: int = 50) -> pd.DataFrame:
        """
        Generate AKI detection cases with serial creatinine

        KDIGO Criteria:
        - Stage 1: 1.5-1.9x baseline OR ≥0.3 mg/dL increase
        - Stage 2: 2.0-2.9x baseline
        - Stage 3: ≥3.0x baseline OR Cr ≥4.0
        """
        cases = []

        for i in range(n):
            baseline_cr = random.uniform(0.6, 1.2)
            case_type = random.choice(['no_aki', 'stage1', 'stage2', 'stage3'])

            if case_type == 'no_aki':
                current_cr = baseline_cr + random.uniform(-0.2, 0.2)
                current_cr = max(0.5, current_cr)  # Keep realistic
                label = 0
                reasoning = f"Creatinine {current_cr:.1f} (baseline {baseline_cr:.1f}) shows no significant change"

            elif case_type == 'stage1':
                if random.choice([True, False]):
                    # Absolute increase ≥0.3
                    current_cr = baseline_cr + random.uniform(0.3, 0.5)
                else:
                    # 1.5-1.9x baseline
                    current_cr = baseline_cr * random.uniform(1.5, 1.9)

                label = 1
                reasoning = f"Creatinine {current_cr:.1f} (baseline {baseline_cr:.1f}, {current_cr/baseline_cr:.1f}x) indicates AKI Stage 1"

            elif case_type == 'stage2':
                # 2.0-2.9x baseline
                current_cr = baseline_cr * random.uniform(2.0, 2.9)
                label = 2
                reasoning = f"Creatinine {current_cr:.1f} (baseline {baseline_cr:.1f}, {current_cr/baseline_cr:.1f}x) indicates AKI Stage 2"

            else:  # stage3
                if random.choice([True, False]) and baseline_cr < 1.5:
                    # ≥4.0 mg/dL
                    current_cr = random.uniform(4.0, 6.0)
                else:
                    # ≥3.0x baseline
                    current_cr = baseline_cr * random.uniform(3.0, 4.5)

                label = 3
                reasoning = f"Creatinine {current_cr:.1f} (baseline {baseline_cr:.1f}, {current_cr/baseline_cr:.1f}x) indicates AKI Stage 3"

            # Generate dates
            baseline_date = (datetime.now() - timedelta(days=random.randint(30, 90))).strftime('%m/%d/%Y')
            current_date = (datetime.now() - timedelta(days=random.randint(1, 7))).strftime('%m/%d/%Y')

            text = f"Baseline creatinine {baseline_cr:.1f} mg/dL on {baseline_date}. Current creatinine {current_cr:.1f} mg/dL on {current_date}."

            cases.append({
                'id': f'AKI_{i+1:03d}',
                'text': text,
                'label': label,
                'extracted_value': f"{baseline_cr:.1f},{current_cr:.1f}",
                'reasoning': reasoning,
                'task': 'aki'
            })

        return pd.DataFrame(cases)

    def generate_diabetes_cases(self, n: int = 50) -> pd.DataFrame:
        """
        Generate diabetes screening cases

        ADA Criteria:
        - HbA1c ≥6.5% = Diabetes
        - HbA1c 5.7-6.4% = Prediabetes
        - Fasting glucose ≥126 mg/dL = Diabetes
        - Fasting glucose 100-125 mg/dL = Prediabetes
        """
        cases = []

        for i in range(n):
            case_type = random.choice(['hba1c', 'glucose', 'both'])

            if case_type == 'hba1c':
                hba1c = random.uniform(4.5, 9.0)

                if hba1c >= 6.5:
                    label = 2
                    reasoning = f"HbA1c {hba1c:.1f}% (≥6.5%) confirms diabetes"
                elif hba1c >= 5.7:
                    label = 1
                    reasoning = f"HbA1c {hba1c:.1f}% (5.7-6.4%) indicates prediabetes"
                else:
                    label = 0
                    reasoning = f"HbA1c {hba1c:.1f}% is normal (< 5.7%)"

                text = f"Lab results show HbA1c of {hba1c:.1f}%."
                extracted_value = hba1c

            elif case_type == 'glucose':
                glucose = random.uniform(70, 200)

                if glucose >= 126:
                    label = 2
                    reasoning = f"Fasting glucose {glucose:.0f} mg/dL (≥126) confirms diabetes"
                elif glucose >= 100:
                    label = 1
                    reasoning = f"Fasting glucose {glucose:.0f} mg/dL (100-125) indicates prediabetes"
                else:
                    label = 0
                    reasoning = f"Fasting glucose {glucose:.0f} mg/dL is normal (< 100)"

                text = f"Fasting blood glucose measured at {glucose:.0f} mg/dL."
                extracted_value = glucose

            else:  # both
                hba1c = random.uniform(4.5, 9.0)
                glucose = random.uniform(70, 200)

                hba1c_dm = hba1c >= 6.5
                hba1c_pre = hba1c >= 5.7
                glucose_dm = glucose >= 126
                glucose_pre = glucose >= 100

                if hba1c_dm or glucose_dm:
                    label = 2
                    reasoning = f"HbA1c {hba1c:.1f}% and/or glucose {glucose:.0f} mg/dL confirm diabetes"
                elif hba1c_pre or glucose_pre:
                    label = 1
                    reasoning = f"HbA1c {hba1c:.1f}% and/or glucose {glucose:.0f} mg/dL indicate prediabetes"
                else:
                    label = 0
                    reasoning = f"HbA1c {hba1c:.1f}% and glucose {glucose:.0f} mg/dL are normal"

                text = f"Patient has HbA1c {hba1c:.1f}% and fasting glucose {glucose:.0f} mg/dL."
                extracted_value = f"{hba1c:.1f},{glucose:.0f}"

            cases.append({
                'id': f'DM_{i+1:03d}',
                'text': text,
                'label': label,
                'extracted_value': extracted_value,
                'reasoning': reasoning,
                'task': 'diabetes'
            })

        return pd.DataFrame(cases)

    def generate_serial_measurement_cases(self, n: int = 20) -> pd.DataFrame:
        """
        Generate cases with multiple serial measurements

        Tests ability to handle temporal data
        """
        cases = []

        for i in range(n):
            measurement_type = random.choice(['creatinine', 'weight', 'blood_pressure'])

            if measurement_type == 'creatinine':
                # Generate 3-5 measurements
                num_measurements = random.randint(3, 5)
                baseline = random.uniform(0.8, 1.2)
                trend = random.choice(['stable', 'increasing', 'decreasing'])

                measurements = [baseline]
                for j in range(num_measurements - 1):
                    if trend == 'stable':
                        measurements.append(measurements[-1] + random.uniform(-0.1, 0.1))
                    elif trend == 'increasing':
                        measurements.append(measurements[-1] + random.uniform(0.2, 0.5))
                    else:
                        measurements.append(measurements[-1] - random.uniform(0.1, 0.3))

                # Generate text with dates
                text_parts = []
                for j, cr in enumerate(measurements):
                    date = (datetime.now() - timedelta(days=(num_measurements-j)*7)).strftime('%m/%d')
                    text_parts.append(f"Cr {cr:.1f} on {date}")

                text = "Serial creatinine measurements: " + ", ".join(text_parts) + "."

                if trend == 'increasing' and measurements[-1] > baseline * 1.5:
                    label = 1
                    reasoning = f"Rising creatinine trend from {baseline:.1f} to {measurements[-1]:.1f} indicates renal deterioration"
                elif trend == 'stable':
                    label = 0
                    reasoning = f"Stable creatinine around {baseline:.1f} indicates stable renal function"
                else:
                    label = 0
                    reasoning = f"Improving creatinine trend from {baseline:.1f} to {measurements[-1]:.1f}"

                extracted_value = ",".join([f"{m:.1f}" for m in measurements])

            cases.append({
                'id': f'SERIAL_{i+1:03d}',
                'text': text,
                'label': label,
                'extracted_value': extracted_value,
                'reasoning': reasoning,
                'task': 'serial_measurements'
            })

        return pd.DataFrame(cases)


def main():
    parser = argparse.ArgumentParser(description='Generate sample clinical datasets')
    parser.add_argument('--task', choices=['malnutrition', 'aki', 'diabetes', 'serial', 'all'],
                       default='all', help='Type of cases to generate')
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of samples per task')
    parser.add_argument('--output-dir', default='sample_datasets',
                       help='Output directory for datasets')

    args = parser.parse_args()

    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    generator = SampleDatasetGenerator()

    if args.task == 'malnutrition' or args.task == 'all':
        print(f"Generating {args.samples} malnutrition cases...")
        mal_df = generator.generate_malnutrition_cases(args.samples)
        output_path = f"{args.output_dir}/malnutrition_gold_standard.csv"
        mal_df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")
        print(f"  Label distribution: {mal_df['label'].value_counts().to_dict()}")

    if args.task == 'aki' or args.task == 'all':
        print(f"\nGenerating {args.samples} AKI cases...")
        aki_df = generator.generate_aki_cases(args.samples)
        output_path = f"{args.output_dir}/aki_gold_standard.csv"
        aki_df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")
        print(f"  Label distribution: {aki_df['label'].value_counts().to_dict()}")

    if args.task == 'diabetes' or args.task == 'all':
        print(f"\nGenerating {args.samples} diabetes screening cases...")
        dm_df = generator.generate_diabetes_cases(args.samples)
        output_path = f"{args.output_dir}/diabetes_gold_standard.csv"
        dm_df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")
        print(f"  Label distribution: {dm_df['label'].value_counts().to_dict()}")

    if args.task == 'serial' or args.task == 'all':
        print(f"\nGenerating {args.samples//2} serial measurement cases...")
        serial_df = generator.generate_serial_measurement_cases(args.samples // 2)
        output_path = f"{args.output_dir}/serial_measurements_gold_standard.csv"
        serial_df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")
        print(f"  Label distribution: {serial_df['label'].value_counts().to_dict()}")

    print("\n" + "="*60)
    print("Sample datasets created successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Process these datasets through clinAnnotate")
    print("2. Run evaluation:")
    print(f"   python evaluate_system.py \\")
    print(f"       --gold {args.output_dir}/malnutrition_gold_standard.csv \\")
    print(f"       --system <your_system_output>.csv \\")
    print(f"       --task classification")


if __name__ == '__main__':
    main()
