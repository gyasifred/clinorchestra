"""
Classification Metrics Evaluation Script

This script evaluates the multiclass diagnosis prediction results against ground truth.

Metrics computed:
- Top-1 Accuracy: Correct diagnosis is #1 prediction
- Top-3 Accuracy: Correct diagnosis in top 3
- Top-5 Accuracy: Correct diagnosis in top 5
- Cross-Entropy Loss: Log-loss for probability calibration
- Brier Score: Mean squared error of probabilities
- Per-Class Precision, Recall, F1-Score
- Confusion Matrix
- Calibration metrics
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    log_loss,
    brier_score_loss,
    classification_report
)


class DiagnosisClassificationEvaluator:
    """Evaluate multiclass diagnosis predictions"""

    def __init__(self, predictions_file: str, ground_truth_file: str,
                 diagnosis_mapping_file: str):
        """
        Initialize evaluator

        Args:
            predictions_file: JSON file with model predictions
            ground_truth_file: CSV with ground truth diagnoses
            diagnosis_mapping_file: CSV mapping ICD codes to diagnosis names
        """
        print("Loading data...")
        self.predictions = self._load_predictions(predictions_file)
        self.ground_truth = pd.read_csv(ground_truth_file)
        self.diagnosis_mapping = pd.read_csv(diagnosis_mapping_file)

        # Create ICD code to index mapping
        self.icd_to_idx = {
            row['icd_code']: idx
            for idx, row in self.diagnosis_mapping.iterrows()
        }
        self.idx_to_icd = {v: k for k, v in self.icd_to_idx.items()}
        self.n_classes = len(self.diagnosis_mapping)

        print(f"Loaded {len(self.predictions)} predictions")
        print(f"Loaded {len(self.ground_truth)} ground truth records")
        print(f"{self.n_classes} diagnosis classes")

    def _load_predictions(self, predictions_file: str) -> List[Dict]:
        """Load predictions from JSON file"""
        with open(predictions_file, 'r') as f:
            return json.load(f)

    def extract_prediction_arrays(self):
        """
        Extract prediction probabilities and true labels as numpy arrays

        Returns:
            y_true: Array of true class indices
            y_pred_proba: Array of predicted probabilities (n_samples x n_classes)
            y_pred: Array of predicted class indices (highest probability)
        """
        y_true = []
        y_pred_proba = []
        y_pred = []

        for pred in self.predictions:
            # Get ground truth
            hadm_id = pred['patient_info']['hadm_id']
            gt_row = self.ground_truth[self.ground_truth['hadm_id'] == hadm_id]

            if len(gt_row) == 0:
                print(f"Warning: No ground truth for hadm_id {hadm_id}")
                continue

            true_icd = gt_row.iloc[0]['icd_code']
            true_idx = self.icd_to_idx[true_icd]
            y_true.append(true_idx)

            # Get predictions
            proba_vector = np.zeros(self.n_classes)

            for diag_pred in pred['multiclass_prediction']['predictions']:
                icd_code = diag_pred['icd_code']
                probability = diag_pred['probability']
                idx = self.icd_to_idx[icd_code]
                proba_vector[idx] = probability

            y_pred_proba.append(proba_vector)
            y_pred.append(np.argmax(proba_vector))

        return (
            np.array(y_true),
            np.array(y_pred_proba),
            np.array(y_pred)
        )

    def compute_top_k_accuracy(self, y_true, y_pred_proba, k=1):
        """
        Compute top-k accuracy

        Args:
            y_true: True class indices
            y_pred_proba: Predicted probabilities
            k: Top-k to consider

        Returns:
            Top-k accuracy (0-1)
        """
        n_samples = len(y_true)
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]

        correct = 0
        for i in range(n_samples):
            if y_true[i] in top_k_preds[i]:
                correct += 1

        return correct / n_samples

    def compute_cross_entropy(self, y_true, y_pred_proba):
        """
        Compute cross-entropy loss (log loss)

        Lower is better. Measures probability calibration.

        Args:
            y_true: True class indices
            y_pred_proba: Predicted probabilities

        Returns:
            Cross-entropy loss
        """
        return log_loss(y_true, y_pred_proba)

    def compute_brier_score(self, y_true, y_pred_proba):
        """
        Compute Brier score (mean squared error of probabilities)

        Lower is better. Range: 0-2 (0 = perfect, 2 = worst)

        Args:
            y_true: True class indices
            y_pred_proba: Predicted probabilities

        Returns:
            Brier score
        """
        # Convert to one-hot encoding
        n_samples = len(y_true)
        n_classes = y_pred_proba.shape[1]

        y_true_one_hot = np.zeros((n_samples, n_classes))
        y_true_one_hot[np.arange(n_samples), y_true] = 1

        # Compute MSE
        brier = np.mean((y_pred_proba - y_true_one_hot) ** 2)
        return brier

    def compute_per_class_metrics(self, y_true, y_pred):
        """
        Compute precision, recall, F1 for each diagnosis class

        Args:
            y_true: True class indices
            y_pred: Predicted class indices

        Returns:
            DataFrame with per-class metrics
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        results = []
        for idx in range(len(precision)):
            icd_code = self.idx_to_icd[idx]
            diagnosis_name = self.diagnosis_mapping[
                self.diagnosis_mapping['icd_code'] == icd_code
            ]['long_title'].iloc[0]

            results.append({
                'icd_code': icd_code,
                'diagnosis': diagnosis_name,
                'precision': precision[idx],
                'recall': recall[idx],
                'f1_score': f1[idx],
                'support': support[idx]
            })

        return pd.DataFrame(results)

    def compute_confusion_matrix(self, y_true, y_pred):
        """
        Compute confusion matrix

        Args:
            y_true: True class indices
            y_pred: Predicted class indices

        Returns:
            Confusion matrix (n_classes x n_classes)
        """
        return confusion_matrix(y_true, y_pred)

    def plot_confusion_matrix(self, cm, output_path: str):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[self.idx_to_icd[i] for i in range(self.n_classes)],
            yticklabels=[self.idx_to_icd[i] for i in range(self.n_classes)]
        )
        plt.title('Diagnosis Prediction Confusion Matrix', fontsize=16)
        plt.ylabel('True Diagnosis', fontsize=12)
        plt.xlabel('Predicted Diagnosis', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {output_path}")
        plt.close()

    def evaluate_all(self, output_dir: str = "mimic-iv/evaluation_results"):
        """
        Run all evaluation metrics and save results

        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("MULTICLASS DIAGNOSIS CLASSIFICATION EVALUATION")
        print("="*80)

        # Extract arrays
        y_true, y_pred_proba, y_pred = self.extract_prediction_arrays()

        print(f"\nEvaluating {len(y_true)} predictions...")

        # Overall accuracy metrics
        print("\n" + "-"*80)
        print("ACCURACY METRICS")
        print("-"*80)

        top1_acc = self.compute_top_k_accuracy(y_true, y_pred_proba, k=1)
        top3_acc = self.compute_top_k_accuracy(y_true, y_pred_proba, k=3)
        top5_acc = self.compute_top_k_accuracy(y_true, y_pred_proba, k=5)

        print(f"Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
        print(f"Top-3 Accuracy: {top3_acc:.4f} ({top3_acc*100:.2f}%)")
        print(f"Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")

        # Calibration metrics
        print("\n" + "-"*80)
        print("CALIBRATION METRICS")
        print("-"*80)

        cross_entropy = self.compute_cross_entropy(y_true, y_pred_proba)
        brier_score = self.compute_brier_score(y_true, y_pred_proba)

        print(f"Cross-Entropy Loss: {cross_entropy:.4f} (lower is better)")
        print(f"Brier Score: {brier_score:.4f} (lower is better, range 0-2)")

        # Per-class metrics
        print("\n" + "-"*80)
        print("PER-CLASS METRICS")
        print("-"*80)

        per_class_df = self.compute_per_class_metrics(y_true, y_pred)
        print(per_class_df.to_string(index=False))

        # Save per-class metrics
        per_class_path = output_path / "per_class_metrics.csv"
        per_class_df.to_csv(per_class_path, index=False)
        print(f"\nPer-class metrics saved to: {per_class_path}")

        # Confusion matrix
        cm = self.compute_confusion_matrix(y_true, y_pred)
        cm_path = output_path / "confusion_matrix.png"
        self.plot_confusion_matrix(cm, str(cm_path))

        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        print("\n" + "-"*80)
        print("AGGREGATE METRICS")
        print("-"*80)
        print(f"Macro-Averaged Precision: {precision_macro:.4f}")
        print(f"Macro-Averaged Recall: {recall_macro:.4f}")
        print(f"Macro-Averaged F1-Score: {f1_macro:.4f}")
        print()
        print(f"Weighted-Averaged Precision: {precision_weighted:.4f}")
        print(f"Weighted-Averaged Recall: {recall_weighted:.4f}")
        print(f"Weighted-Averaged F1-Score: {f1_weighted:.4f}")

        # Save summary
        summary = {
            "evaluation_summary": {
                "total_predictions": len(y_true),
                "n_classes": self.n_classes,
                "accuracy_metrics": {
                    "top_1_accuracy": float(top1_acc),
                    "top_3_accuracy": float(top3_acc),
                    "top_5_accuracy": float(top5_acc)
                },
                "calibration_metrics": {
                    "cross_entropy_loss": float(cross_entropy),
                    "brier_score": float(brier_score)
                },
                "aggregate_metrics": {
                    "macro_precision": float(precision_macro),
                    "macro_recall": float(recall_macro),
                    "macro_f1": float(f1_macro),
                    "weighted_precision": float(precision_weighted),
                    "weighted_recall": float(recall_weighted),
                    "weighted_f1": float(f1_weighted)
                }
            }
        }

        summary_path = output_path / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nEvaluation summary saved to: {summary_path}")

        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"\nAll results saved to: {output_path}/")

        return summary


def main():
    """Main execution"""
    import sys

    print("\n" + "="*80)
    print("DIAGNOSIS CLASSIFICATION EVALUATION")
    print("="*80)

    # Get file paths
    predictions_file = input("\nPath to predictions JSON file: ").strip()
    if not predictions_file:
        print("Error: Predictions file required")
        return

    ground_truth_file = input("Path to ground truth CSV file: ").strip()
    if not ground_truth_file:
        print("Error: Ground truth file required")
        return

    diagnosis_mapping_file = input("Path to top_20_primary_diagnoses.csv [mimic-iv/top_20_primary_diagnoses.csv]: ").strip()
    if not diagnosis_mapping_file:
        diagnosis_mapping_file = "mimic-iv/top_20_primary_diagnoses.csv"

    try:
        evaluator = DiagnosisClassificationEvaluator(
            predictions_file,
            ground_truth_file,
            diagnosis_mapping_file
        )

        results = evaluator.evaluate_all()

        print("\n✓ Evaluation complete!")
        print("\nKey Metrics:")
        print(f"  Top-1 Accuracy: {results['evaluation_summary']['accuracy_metrics']['top_1_accuracy']:.2%}")
        print(f"  Top-5 Accuracy: {results['evaluation_summary']['accuracy_metrics']['top_5_accuracy']:.2%}")
        print(f"  Macro F1-Score: {results['evaluation_summary']['aggregate_metrics']['macro_f1']:.4f}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
