"""
ClinOrchestra System Evaluation Script

This script provides comprehensive evaluation metrics for your annotation system.
Compare system outputs against gold standard annotations.

Usage:
    python evaluate_system.py --gold gold_standard.csv --system system_output.csv --task classification
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
from typing import Dict, List, Tuple
import argparse
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinOrchestraEvaluator:
    """Evaluate ClinOrchestra system performance"""

    def __init__(self, gold_standard_path: str, system_output_path: str):
        """
        Args:
            gold_standard_path: Path to gold standard CSV
            system_output_path: Path to system output CSV
        """
        self.gold_df = pd.read_csv(gold_standard_path)
        self.system_df = pd.read_csv(system_output_path)

        logger.info(f"Loaded {len(self.gold_df)} gold standard annotations")
        logger.info(f"Loaded {len(self.system_df)} system predictions")

    def evaluate_classification(self,
                               id_col: str = 'id',
                               label_col: str = 'label') -> Dict:
        """
        Evaluate classification performance

        Returns:
            Dictionary with metrics: accuracy, precision, recall, f1, kappa
        """
        # Merge on ID
        merged = pd.merge(
            self.gold_df[[id_col, label_col]],
            self.system_df[[id_col, label_col]],
            on=id_col,
            suffixes=('_gold', '_system')
        )

        if len(merged) == 0:
            raise ValueError("No matching IDs between gold standard and system output!")

        y_true = merged[f'{label_col}_gold']
        y_pred = merged[f'{label_col}_system']

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        kappa = cohen_kappa_score(y_true, y_pred)

        # Per-class metrics
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)

        results = {
            'overall': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'cohen_kappa': float(kappa),
                'n_samples': len(merged)
            },
            'per_class': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'class_labels': sorted(y_true.unique().tolist())
        }

        return results

    def evaluate_extraction(self,
                           id_col: str = 'id',
                           value_col: str = 'extracted_value',
                           tolerance: float = 0.01) -> Dict:
        """
        Evaluate information extraction performance

        Args:
            tolerance: Relative tolerance for numeric values (e.g., 0.01 = 1%)

        Returns:
            Dictionary with extraction metrics
        """
        merged = pd.merge(
            self.gold_df[[id_col, value_col]],
            self.system_df[[id_col, value_col]],
            on=id_col,
            suffixes=('_gold', '_system')
        )

        y_true = merged[f'{value_col}_gold']
        y_pred = merged[f'{value_col}_system']

        exact_matches = 0
        partial_matches = 0
        false_positives = 0
        false_negatives = 0

        for true_val, pred_val in zip(y_true, y_pred):
            # Both missing
            if pd.isna(true_val) and pd.isna(pred_val):
                exact_matches += 1
            # False negative (should extract but didn't)
            elif not pd.isna(true_val) and pd.isna(pred_val):
                false_negatives += 1
            # False positive (extracted but shouldn't)
            elif pd.isna(true_val) and not pd.isna(pred_val):
                false_positives += 1
            # Both present - check match
            else:
                # Try numeric comparison
                try:
                    true_num = float(true_val)
                    pred_num = float(pred_val)

                    if abs(true_num - pred_num) < tolerance * abs(true_num):
                        exact_matches += 1
                    elif abs(true_num - pred_num) < 2 * tolerance * abs(true_num):
                        partial_matches += 1
                except (ValueError, TypeError):
                    # String comparison
                    if str(true_val).strip().lower() == str(pred_val).strip().lower():
                        exact_matches += 1
                    elif str(true_val).strip().lower() in str(pred_val).strip().lower() or \
                         str(pred_val).strip().lower() in str(true_val).strip().lower():
                        partial_matches += 1

        total = len(merged)
        exact_accuracy = exact_matches / total if total > 0 else 0
        partial_accuracy = (exact_matches + partial_matches) / total if total > 0 else 0

        results = {
            'exact_matches': exact_matches,
            'partial_matches': partial_matches,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'total_samples': total,
            'exact_accuracy': exact_accuracy,
            'partial_accuracy': partial_accuracy,
            'extraction_precision': exact_matches / (exact_matches + false_positives) if (exact_matches + false_positives) > 0 else 0,
            'extraction_recall': exact_matches / (exact_matches + false_negatives) if (exact_matches + false_negatives) > 0 else 0
        }

        return results

    def evaluate_reasoning(self,
                          id_col: str = 'id',
                          reasoning_col: str = 'reasoning',
                          label_col: str = 'label') -> Dict:
        """
        Evaluate quality of reasoning/explanation

        Checks if reasoning mentions key clinical concepts for correct predictions
        """
        merged = pd.merge(
            self.gold_df[[id_col, label_col, reasoning_col]],
            self.system_df[[id_col, label_col, reasoning_col]],
            on=id_col,
            suffixes=('_gold', '_system')
        )

        correct_with_reasoning = 0
        correct_without_reasoning = 0
        incorrect_with_reasoning = 0
        incorrect_without_reasoning = 0

        for _, row in merged.iterrows():
            correct = row[f'{label_col}_gold'] == row[f'{label_col}_system']
            has_reasoning = not pd.isna(row[f'{reasoning_col}_system']) and \
                          len(str(row[f'{reasoning_col}_system']).strip()) > 10

            if correct and has_reasoning:
                correct_with_reasoning += 1
            elif correct and not has_reasoning:
                correct_without_reasoning += 1
            elif not correct and has_reasoning:
                incorrect_with_reasoning += 1
            else:
                incorrect_without_reasoning += 1

        total = len(merged)

        results = {
            'correct_with_reasoning': correct_with_reasoning,
            'correct_without_reasoning': correct_without_reasoning,
            'incorrect_with_reasoning': incorrect_with_reasoning,
            'incorrect_without_reasoning': incorrect_without_reasoning,
            'reasoning_coverage': (correct_with_reasoning + incorrect_with_reasoning) / total if total > 0 else 0,
            'total_samples': total
        }

        return results

    def error_analysis(self,
                      id_col: str = 'id',
                      label_col: str = 'label',
                      text_col: str = 'text') -> pd.DataFrame:
        """
        Identify and categorize errors

        Returns:
            DataFrame with error cases and categories
        """
        merged = pd.merge(
            self.gold_df,
            self.system_df,
            on=id_col,
            suffixes=('_gold', '_system')
        )

        # Filter to errors only
        errors = merged[merged[f'{label_col}_gold'] != merged[f'{label_col}_system']].copy()

        logger.info(f"Found {len(errors)} errors out of {len(merged)} samples ({len(errors)/len(merged)*100:.1f}%)")

        # Add error type categorization
        def categorize_error(row):
            """Categorize type of error"""
            # You can customize this based on your labels
            true_label = row[f'{label_col}_gold']
            pred_label = row[f'{label_col}_system']

            if pd.isna(pred_label):
                return "system_no_prediction"
            elif true_label == 0 and pred_label == 1:
                return "false_positive"
            elif true_label == 1 and pred_label == 0:
                return "false_negative"
            else:
                return "misclassification"

        errors['error_type'] = errors.apply(categorize_error, axis=1)

        return errors

    def generate_report(self,
                       output_path: str = 'evaluation_report.json',
                       task_type: str = 'classification') -> None:
        """
        Generate comprehensive evaluation report

        Args:
            output_path: Where to save JSON report
            task_type: 'classification', 'extraction', or 'both'
        """
        report = {
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'gold_standard_samples': len(self.gold_df),
            'system_output_samples': len(self.system_df)
        }

        try:
            if task_type in ['classification', 'both']:
                logger.info("Evaluating classification performance...")
                report['classification'] = self.evaluate_classification()

                # Print summary
                print("\n" + "="*60)
                print("CLASSIFICATION RESULTS")
                print("="*60)
                print(f"Accuracy:  {report['classification']['overall']['accuracy']:.3f}")
                print(f"Precision: {report['classification']['overall']['precision']:.3f}")
                print(f"Recall:    {report['classification']['overall']['recall']:.3f}")
                print(f"F1 Score:  {report['classification']['overall']['f1']:.3f}")
                print(f"Cohen's Kappa: {report['classification']['overall']['cohen_kappa']:.3f}")
                print(f"Samples:   {report['classification']['overall']['n_samples']}")

                print("\nPer-Class Performance:")
                for label, metrics in report['classification']['per_class'].items():
                    if label not in ['accuracy', 'macro avg', 'weighted avg']:
                        print(f"  {label}: P={metrics.get('precision', 0):.3f}, R={metrics.get('recall', 0):.3f}, F1={metrics.get('f1-score', 0):.3f}")

                print("\nConfusion Matrix:")
                print(np.array(report['classification']['confusion_matrix']))

            if task_type in ['extraction', 'both']:
                logger.info("Evaluating extraction performance...")
                report['extraction'] = self.evaluate_extraction()

                print("\n" + "="*60)
                print("EXTRACTION RESULTS")
                print("="*60)
                print(f"Exact Accuracy:   {report['extraction']['exact_accuracy']:.3f}")
                print(f"Partial Accuracy: {report['extraction']['partial_accuracy']:.3f}")
                print(f"Precision:        {report['extraction']['extraction_precision']:.3f}")
                print(f"Recall:           {report['extraction']['extraction_recall']:.3f}")
                print(f"False Positives:  {report['extraction']['false_positives']}")
                print(f"False Negatives:  {report['extraction']['false_negatives']}")

            # Error analysis
            logger.info("Performing error analysis...")
            errors_df = self.error_analysis()
            report['error_analysis'] = {
                'total_errors': len(errors_df),
                'error_types': errors_df['error_type'].value_counts().to_dict()
            }

            print("\n" + "="*60)
            print("ERROR ANALYSIS")
            print("="*60)
            print(f"Total Errors: {report['error_analysis']['total_errors']}")
            print("Error Types:")
            for error_type, count in report['error_analysis']['error_types'].items():
                print(f"  {error_type}: {count}")

            # Save errors to CSV
            errors_csv = output_path.replace('.json', '_errors.csv')
            errors_df.to_csv(errors_csv, index=False)
            logger.info(f"Saved error details to {errors_csv}")

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()

        # Save full report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Saved evaluation report to {output_path}")

        print("\n" + "="*60)
        print(f"Full report saved to: {output_path}")
        print(f"Error details saved to: {errors_csv}")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ClinOrchestra system performance')
    parser.add_argument('--gold', required=True, help='Path to gold standard CSV')
    parser.add_argument('--system', required=True, help='Path to system output CSV')
    parser.add_argument('--task', choices=['classification', 'extraction', 'both'],
                       default='classification', help='Task type to evaluate')
    parser.add_argument('--output', default='evaluation_report.json',
                       help='Path to save evaluation report')

    args = parser.parse_args()

    # Create evaluator
    evaluator = ClinOrchestraEvaluator(args.gold, args.system)

    # Generate report
    evaluator.generate_report(output_path=args.output, task_type=args.task)


if __name__ == '__main__':
    main()
