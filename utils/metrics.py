"""
Shared evaluation metrics functions.

Used by:
  - models/random_forest/03_validation.py
  - models/random_forest/05_extrapolation_validation.py
  - models/xgboost/03_validation.py
  - models/xgboost/05_extrapolation_validation.py

Author: Aman Arora
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)


def compute_all_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels (0 or 1).
    y_pred_prob : array-like
        Predicted probabilities.
    threshold : float
        Classification threshold for binarizing predictions.

    Returns
    -------
    dict
        Dictionary containing:
        - auc, accuracy, sensitivity, specificity
        - precision, f1, tss
        - confusion_matrix, fpr, tpr
        - y_pred_binary

    Example
    -------
    >>> metrics = compute_all_metrics(y_true, y_pred, threshold=0.5)
    >>> print(f"AUC: {metrics['auc']:.4f}")
    """
    y_pred_binary = (np.array(y_pred_prob) >= threshold).astype(int)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    cm = confusion_matrix(y_true, y_pred_binary)

    # Extract TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'auc': roc_auc_score(y_true, y_pred_prob),
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
        'tss': sensitivity + specificity - 1,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'y_pred_binary': y_pred_binary,
    }


def print_metrics(metrics, model_name="Model"):
    """
    Pretty-print metrics to console.

    Parameters
    ----------
    metrics : dict
        Output from compute_all_metrics().
    model_name : str
        Name for display.
    """
    print(f"\n  {model_name} Results:")
    print(f"  ┌──────────────────────────────┐")
    print(f"  │ ROC AUC:     {metrics['auc']:.4f}          │")
    print(f"  │ Accuracy:    {metrics['accuracy']:.4f}          │")
    print(f"  │ Sensitivity: {metrics['sensitivity']:.4f}          │")
    print(f"  │ Specificity: {metrics['specificity']:.4f}          │")
    print(f"  │ Precision:   {metrics['precision']:.4f}          │")
    print(f"  │ F1 Score:    {metrics['f1']:.4f}          │")
    print(f"  │ TSS:         {metrics['tss']:.4f}          │")
    print(f"  └──────────────────────────────┘")
    print(f"\n  Confusion Matrix:\n{metrics['confusion_matrix']}")


def save_metrics(metrics, y_true, y_pred_prob, output_dir,
                 prefix="model", stage="validation"):
    """
    Save metrics summary and predictions to files.

    Parameters
    ----------
    metrics : dict
        Output from compute_all_metrics().
    y_true : array
        Ground truth labels.
    y_pred_prob : array
        Predicted probabilities.
    output_dir : Path
        Directory to save files.
    prefix : str
        Model prefix (e.g., 'rf', 'xgb').
    stage : str
        Pipeline stage (e.g., 'validation', 'extrapolation').

    Returns
    -------
    tuple
        (summary_path, csv_path)
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save text summary
    summary_path = output_dir / f"{prefix}_{stage}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"{prefix.upper()} {stage.title()} Summary\n")
        f.write("=" * 50 + "\n\n")
        for key in ['auc', 'accuracy', 'sensitivity', 'specificity',
                     'precision', 'f1', 'tss']:
            f.write(f"{key:15s}: {metrics[key]:.4f}\n")
        f.write(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}\n")
    print(f"  → Saved: {summary_path}")

    # Save predictions CSV
    csv_path = output_dir / f"{prefix}_{stage}_predictions.csv"
    pd.DataFrame({
        'y_true': y_true,
        'y_pred_prob': y_pred_prob,
        'y_pred_binary': metrics['y_pred_binary']
    }).to_csv(csv_path, index=False)
    print(f"  → Saved: {csv_path}")

    return summary_path, csv_path
