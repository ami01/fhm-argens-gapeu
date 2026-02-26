"""
XGBoost — Training & Hyperparameter Tuning

Steps:
  1. Load training data (70k flood/no-flood points)
  2. Select features and check multicollinearity (VIF)
  3. Hyperparameter tuning via RandomizedSearchCV
  4. Save best model to disk

Author: Aman Arora
"""
import sys
from pathlib import Path

# Add repo root to path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import (
    RAW_DATA_DIR, RESULTS_DIR,
    FIGURES_DIR, RANDOM_SEED
)

# ── Paths (relative to repo root) ──────────────────────
TRAIN_DATA = RAW_DATA_DIR / "inventory" / "train_data_70k.csv"
MODEL_DIR = RESULTS_DIR / "xgboost"
MODEL_PATH = MODEL_DIR / "xgb_model_4dtsts.pkl"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Features to use ────────────────────────────────────
SELECTED_FEATURES = ['Discharge_N', 'Dist2stream_N', 'HAND', 'River_Slope']
TARGET_COL = 'CID'


def load_and_prepare_data(filepath, features, target):
    """
    Load training CSV and prepare feature matrix and target.

    Parameters
    ----------
    filepath : Path
        Path to training CSV.
    features : list
        List of feature column names.
    target : str
        Target column name.

    Returns
    -------
    X : pd.DataFrame
        Cleaned feature matrix.
    y : pd.Series
        Target variable.
    """
    df = pd.read_csv(filepath)

    # Select features
    X = df[features].copy()

    # Remove constant columns
    X = X.loc[:, X.nunique() > 1]

    # Replace infinite values with NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Impute missing values with median
    X = X.fillna(X.median())

    y = df[target]

    return X, y


def compute_vif(X):
    """
    Compute Variance Inflation Factor and Tolerance for features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.

    Returns
    -------
    pd.DataFrame
        VIF and Tolerance per feature.
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    vif_data["Tolerance"] = 1 / vif_data["VIF"]
    return vif_data


def plot_correlation_matrix(X, save_path=None):
    """
    Plot and optionally save correlation matrix heatmap.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    save_path : Path, optional
        Path to save figure.
    """
    corr_matrix = X.corr().abs()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("Correlation Matrix — XGBoost Features")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  → Saved correlation matrix: {save_path}")

    plt.show()


def train_xgboost(X, y, param_grid, seed=42):
    """
    Train XGBoost with RandomizedSearchCV.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    param_grid : dict
        Hyperparameter search space.
    seed : int
        Random seed.

    Returns
    -------
    RandomizedSearchCV
        Fitted search object with best model.
    """
    xgb_model = XGBClassifier(
        random_state=seed,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='auc'
    )

    xgb_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=50,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        random_state=seed,
        n_jobs=-1
    )

    xgb_search.fit(X, y)
    return xgb_search


def main():
    print("=" * 60)
    print("XGBOOST — TRAINING")
    print("=" * 60)

    # 1. Load data
    print("\n1. Loading training data...")
    X_train, y_train = load_and_prepare_data(
        TRAIN_DATA, SELECTED_FEATURES, TARGET_COL
    )
    print(f"   Samples: {len(X_train)}, Features: {X_train.shape[1]}")
    print(f"   Features: {X_train.columns.tolist()}")

    # 2. Check multicollinearity
    print("\n2. Computing VIF...")
    vif = compute_vif(X_train)
    print(vif.to_string(index=False))

    # 3. Correlation matrix
    print("\n3. Plotting correlation matrix...")
    corr_fig_path = FIGURES_DIR / "xgb_correlation_matrix.png"
    plot_correlation_matrix(X_train, save_path=corr_fig_path)

    # 4. Train model
    print("\n4. Training XGBoost...")
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 6, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2],
    }

    xgb_search = train_xgboost(X_train, y_train, param_grid, RANDOM_SEED)

    print(f"\n   Best parameters: {xgb_search.best_params_}")
    print(f"   Best AUC: {xgb_search.best_score_:.4f}")

    # 5. Save model
    joblib.dump(xgb_search.best_estimator_, MODEL_PATH)
    print(f"\n   ✅ Model saved to: {MODEL_PATH}")

    # 6. Save best parameters to text file
    params_path = MODEL_DIR / "xgb_best_params.txt"
    with open(params_path, 'w') as f:
        f.write(f"Best AUC: {xgb_search.best_score_:.4f}\n\n")
        f.write("Best Parameters:\n")
        for param, value in xgb_search.best_params_.items():
            f.write(f"  {param}: {value}\n")
    print(f"   ✅ Parameters saved to: {params_path}")


if __name__ == "__main__":
    main()
