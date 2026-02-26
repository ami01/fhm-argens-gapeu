"""
Forward Feature Selection optimized on CSI (Critical Success Index).

Shared utility used by all models (RF, XGB, NNet) to identify
the best subset of predictor features.

Methods:
  - best_csi_score(): Compute best CSI over threshold range
  - load_predictor_rasters(): Find all .tif predictors
  - sample_rasters(): Create sampled dataset from rasters
  - forward_feature_selection(): Iterative feature selection
  - plot_feature_selection(): Plot CSI vs steps
  - save_feature_selection_results(): Save all outputs

Author: Aman Arora
"""

import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def best_csi_score(y_true, y_pred_prob, n_thresholds=101):
    """
    Compute the best CSI over a range of thresholds.

    CSI = TP / (TP + FP + FN)
    Also known as Threat Score or Jaccard Index.

    Parameters
    ----------
    y_true : array-like
        Ground truth binary labels (0 or 1).
    y_pred_prob : array-like
        Predicted probabilities for positive class.
    n_thresholds : int
        Number of thresholds to evaluate between 0 and 1.

    Returns
    -------
    best_csi : float
        Best CSI score found.
    best_threshold : float
        Threshold that produced best CSI.

    Example
    -------
    >>> csi, threshold = best_csi_score(y_true, y_pred_prob)
    >>> print(f"CSI: {csi:.4f} at threshold {threshold:.2f}")
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    best_csi = 0.0
    best_threshold = 0.5

    for th in thresholds:
        y_pred_bin = (y_pred_prob >= th).astype(int)
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
        except ValueError:
            tp = fp = fn = 0
            tn = len(y_true)

        csi = tp / (tp + fp + fn + 1e-9)

        if csi > best_csi:
            best_csi = csi
            best_threshold = th

    return best_csi, best_threshold


def load_predictor_rasters(predictor_dir):
    """
    Find all .tif predictor rasters in a directory.

    Parameters
    ----------
    predictor_dir : str or Path
        Directory containing predictor .tif files.

    Returns
    -------
    files : list of str
        Sorted list of full file paths.
    names : list of str
        Predictor names (filename without extension).

    Raises
    ------
    FileNotFoundError
        If no .tif files found.

    Example
    -------
    >>> files, names = load_predictor_rasters("data/raw/rasters")
    >>> print(f"Found {len(files)} predictors")
    """
    predictor_dir = str(predictor_dir)
    files = sorted([
        os.path.join(predictor_dir, f)
        for f in os.listdir(predictor_dir)
        if f.endswith(".tif")
    ])

    if not files:
        raise FileNotFoundError(
            f"No .tif files found in {predictor_dir}"
        )

    names = [
        os.path.splitext(os.path.basename(f))[0]
        for f in files
    ]

    return files, names


def sample_rasters(predictor_files, predictor_names, target_path,
                   sample_fraction=0.1, seed=42):
    """
    Load rasters and create a sampled dataset for feature selection.

    Randomly samples a fraction of pixels from all rasters
    to create a manageable dataset for iterative feature selection.

    Parameters
    ----------
    predictor_files : list of str
        Paths to predictor raster files.
    predictor_names : list of str
        Names of predictors (column names).
    target_path : str or Path
        Path to binary target raster (flood=1, no flood=0).
    sample_fraction : float
        Fraction of total pixels to sample (0 to 1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    df : pd.DataFrame
        Sampled data with predictor columns and 'target' column.

    Example
    -------
    >>> df = sample_rasters(files, names, "target.tif", 0.1, 42)
    >>> print(df.shape)  # (n_samples, n_predictors + 1)
    """
    np.random.seed(seed)

    # Get raster dimensions
    with rasterio.open(predictor_files[0]) as src:
        n_pixels = src.width * src.height
        print(f"  Total pixels per raster: {n_pixels:,}")

    # Compute sample size
    sample_size = int(n_pixels * sample_fraction)
    print(f"  Sample size ({sample_fraction*100:.0f}%): {sample_size:,}")

    # Random pixel indices
    indices = np.random.choice(n_pixels, size=sample_size, replace=False)

    # Load predictors
    print(f"  Loading {len(predictor_files)} predictor rasters...")
    X_list = []
    for i, filepath in enumerate(predictor_files):
        with rasterio.open(filepath) as src:
            arr = src.read(1).astype(np.float32).flatten()
            X_list.append(arr[indices])
        if (i + 1) % 5 == 0:
            print(f"    Loaded {i + 1}/{len(predictor_files)}")

    X_sample = np.stack(X_list, axis=1)

    # Load target
    print(f"  Loading target raster...")
    with rasterio.open(str(target_path)) as src:
        target_arr = src.read(1).astype(np.float32).flatten()
    y_sample = (target_arr[indices] > 0.5).astype(int)

    # Create DataFrame
    df = pd.DataFrame(X_sample, columns=predictor_names)
    df['target'] = y_sample

    print(f"  Class distribution:")
    print(f"    No flood (0): {(y_sample == 0).sum():,}")
    print(f"    Flood    (1): {(y_sample == 1).sum():,}")

    return df


def forward_feature_selection(df, predictor_cols, desired_csi=0.9,
                               n_estimators=100, seed=42,
                               n_thresholds=101):
    """
    Forward feature selection optimized on CSI.

    Iteratively adds the feature that maximizes CSI
    when combined with already-selected features.
    Uses Random Forest as the base classifier.

    Parameters
    ----------
    df : pd.DataFrame
        Sampled data with predictor columns and 'target'.
    predictor_cols : list of str
        All available predictor names.
    desired_csi : float
        Stop early if CSI reaches this value.
    n_estimators : int
        Number of RF trees per evaluation.
    seed : int
        Random seed.
    n_thresholds : int
        Number of thresholds to test for CSI.

    Returns
    -------
    results : list of dict
        Step-by-step results with keys:
        step, added_feature, n_features, features, csi, threshold.
    best_features : list of str
        Final selected feature set in order of selection.
    best_csi : float
        Best CSI achieved.

    Example
    -------
    >>> results, features, csi = forward_feature_selection(
    ...     df, predictor_names, desired_csi=0.9)
    >>> print(f"Best features: {features}, CSI: {csi:.4f}")
    """
    current_features = []
    remaining_features = predictor_cols.copy()
    best_csi = 0.0
    results = []

    print(f"\n  Total candidate features: {len(remaining_features)}")
    print(f"  Target CSI: {desired_csi}\n")

    step = 0
    while remaining_features and best_csi < desired_csi:
        step += 1
        best_candidate = None
        best_candidate_csi = best_csi
        best_candidate_threshold = 0.5

        print(f"  ── Step {step} ──────────────────────────────")
        print(f"     Current ({len(current_features)}): {current_features}")
        print(f"     Testing {len(remaining_features)} candidates...")

        for feature in remaining_features:
            temp_features = current_features + [feature]
            X_temp = df[temp_features].values

            # Train RF
            clf = RandomForestClassifier(
                random_state=seed,
                n_estimators=n_estimators,
                n_jobs=-1
            )
            clf.fit(X_temp, df['target'].values)

            # Predict probabilities
            y_pred_prob = clf.predict_proba(X_temp)[:, 1]

            # Compute CSI
            csi, threshold = best_csi_score(
                df['target'].values, y_pred_prob, n_thresholds
            )

            print(f"       + {feature:25s} → CSI: {csi:.4f} "
                  f"(threshold: {threshold:.2f})")

            if csi > best_candidate_csi:
                best_candidate_csi = csi
                best_candidate = feature
                best_candidate_threshold = threshold

        # Check improvement
        if best_candidate is not None and best_candidate_csi > best_csi:
            current_features.append(best_candidate)
            remaining_features.remove(best_candidate)
            best_csi = best_candidate_csi

            results.append({
                'step': step,
                'added_feature': best_candidate,
                'n_features': len(current_features),
                'features': current_features.copy(),
                'csi': best_csi,
                'threshold': best_candidate_threshold
            })

            print(f"\n     ✅ Added '{best_candidate}'")
            print(f"     CSI: {best_csi:.4f}\n")
        else:
            print("\n     ❌ No further improvement. Stopping.")
            break

    return results, current_features, best_csi


def plot_feature_selection(results, save_path=None, dpi=300):
    """
    Plot CSI improvement across feature selection steps.

    Parameters
    ----------
    results : list of dict
        Output from forward_feature_selection().
    save_path : str or Path, optional
        Path to save figure.
    dpi : int
        Figure resolution.

    Example
    -------
    >>> plot_feature_selection(results, "figures/feature_selection.png")
    """
    steps = [r['step'] for r in results]
    csis = [r['csi'] for r in results]
    labels = [r['added_feature'] for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, csis, 'bo-', linewidth=2, markersize=8)

    for s, c, l in zip(steps, csis, labels):
        ax.annotate(
            l, (s, c),
            textcoords="offset points",
            xytext=(0, 12),
            ha='center',
            fontsize=8,
            rotation=45
        )

    ax.set_xlabel('Feature Selection Step', fontsize=12)
    ax.set_ylabel('CSI Score', fontsize=12)
    ax.set_title('Forward Feature Selection — CSI Optimization', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps)
    plt.tight_layout()

    if save_path:
        plt.savefig(str(save_path), dpi=dpi, bbox_inches='tight')
        print(f"  → Saved: {save_path}")

    plt.show()


def save_feature_selection_results(results, best_features, best_csi,
                                    output_dir):
    """
    Save feature selection results to files.

    Saves:
      - feature_selection_results.csv (step-by-step)
      - feature_selection_summary.txt (human-readable)
      - selected_features.txt (one feature per line)

    Parameters
    ----------
    results : list of dict
        Step-by-step results.
    best_features : list of str
        Final selected features.
    best_csi : float
        Best CSI achieved.
    output_dir : str or Path
        Output directory.

    Example
    -------
    >>> save_feature_selection_results(
    ...     results, features, csi, "results/feature_selection")
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step-by-step CSV
    df_results = pd.DataFrame(results)
    csv_path = output_dir / "feature_selection_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"  → Saved: {csv_path}")

    # Human-readable summary
    txt_path = output_dir / "feature_selection_summary.txt"
    with open(txt_path, 'w') as f:
        f.write("Forward Feature Selection Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total features selected: {len(best_features)}\n")
        f.write(f"Best CSI: {best_csi:.4f}\n\n")
        f.write("Selected Features (in order):\n")
        for i, feat in enumerate(best_features, 1):
            info = results[i - 1]
            f.write(f"  {i}. {feat:25s} "
                    f"(CSI: {info['csi']:.4f}, "
                    f"threshold: {info['threshold']:.2f})\n")
        f.write(f"\nFinal feature list:\n")
        f.write(f"  {best_features}\n")
    print(f"  → Saved: {txt_path}")

    # One feature per line (easy to parse)
    feat_path = output_dir / "selected_features.txt"
    with open(feat_path, 'w') as f:
        for feat in best_features:
            f.write(f"{feat}\n")
    print(f"  → Saved: {feat_path}")
