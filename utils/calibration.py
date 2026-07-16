"""
Probability calibration: Brier score and reliability diagrams.

Assesses whether the model probability outputs are well calibrated on the
unseen Gapeau basin. Reproduces the reliability-diagram figure and the
Brier / AUROC values reported in the manuscript (extrapolation setting).

Inputs
------
Balanced calibration samples under ``data/calibration/``:
  Gapeau_<MODEL>_calibration_sample_seed42.csv
with columns: pixel_index, row, col, x_utm, y_utm, flood_label, pred_prob.
Each file holds 10,000 flood + 10,000 non-flood pixels drawn with numpy
seed=42 from the reference grid (flood_label = reference depth > 0).

Outputs
-------
  - Printed Brier score and AUROC per model.
  - figures/Fig_Reliability_Diagrams.pdf : 1x3 reliability diagram panel.

Author: Aman Arora
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

try:
    from config import DATA_DIR, FIGURES_DIR, FIGURE_DPI
except Exception:  # allow running as a stand-alone script
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "data"
    FIGURES_DIR = ROOT / "figures"
    FIGURE_DPI = 300

CALIBRATION_DIR = DATA_DIR / "calibration"

# Display name -> file token
MODELS = {
    "Random Forest": "RF",
    "XGBoost": "XGB",
    "Neural Network": "Nnet",
}


def brier_score(y_true, y_prob):
    """Mean squared error between predicted probability and outcome."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_bins(y_true, y_prob, n_bins=10):
    """
    Bin predicted probabilities and return, per non-empty bin, the mean
    predicted probability and the observed flood frequency.

    Returns
    -------
    mean_pred : ndarray
        Mean predicted probability in each populated bin.
    obs_freq : ndarray
        Observed frequency of flood_label == 1 in each populated bin.
    counts : ndarray
        Number of samples in each populated bin.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)

    mean_pred, obs_freq, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        mean_pred.append(y_prob[m].mean())
        obs_freq.append(y_true[m].mean())
        counts.append(int(m.sum()))
    return np.array(mean_pred), np.array(obs_freq), np.array(counts)


def evaluate_model(token):
    """Load one model's calibration sample and compute metrics + bins."""
    path = CALIBRATION_DIR / f"Gapeau_{token}_calibration_sample_seed42.csv"
    df = pd.read_csv(path)
    y_true = df["flood_label"].to_numpy()
    y_prob = df["pred_prob"].to_numpy()
    return {
        "n": len(df),
        "brier": brier_score(y_true, y_prob),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "bins": reliability_bins(y_true, y_prob, n_bins=10),
    }


def plot_reliability(results, out_path):
    """Draw a 1x3 reliability diagram panel and save to out_path."""
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 4),
                             sharex=True, sharey=True)
    if len(results) == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        mean_pred, obs_freq, _ = res["bins"]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.plot(mean_pred, obs_freq, "o-", color="#1f77b4", lw=1.5,
                label="Model")
        ax.set_title(f"{name}\nBrier = {res['brier']:.4f}  "
                     f"AUROC = {res['auroc']:.4f}", fontsize=10)
        ax.set_xlabel("Mean predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Observed flood frequency")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    results = {}
    print(f"{'Model':16s} {'n':>7s} {'AUROC':>8s} {'Brier':>8s}")
    for name, token in MODELS.items():
        res = evaluate_model(token)
        results[name] = res
        print(f"{name:16s} {res['n']:7d} {res['auroc']:8.4f} {res['brier']:8.4f}")

    out = plot_reliability(results, FIGURES_DIR / "Fig_Reliability_Diagrams.pdf")
    print(f"\nReliability diagram written to: {out}")


if __name__ == "__main__":
    main()
