#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adult XGBoost — Patient-Grouped 10-Fold CV

Standalone, Spyder-friendly journal code.

Method
------
- Binary channel classification: Non-EZ (0) versus EZ (1)
- Adult and pediatric cohorts are analyzed in separate files
- 10-fold StratifiedGroupKFold using Subject_ID
- All channels from one patient remain in one fold
- Imputation/scaling are fitted only on each training fold
- No SMOTE is used
- Test folds are never resampled
- Class imbalance is handled only within training folds

Expected columns
----------------
Subject_ID, Channel, is_soz, Hq Value, evec avg, deltaH, frac_Avg

The legacy column name ``is_soz`` is accepted, but all figures use
``EZ`` and ``Non-EZ``.

Spyder use
----------
1. Place this script beside adult_features.xlsx or .csv.
2. Open the script in Spyder.
3. Press Run.
4. Results are saved in outputs/adult_xgboost.

Install
-------
pip install numpy pandas matplotlib scikit-learn openpyxl xgboost
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "Install XGBoost first: pip install xgboost"
    ) from exc

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# USER SETTINGS
# ============================================================

COHORT = "adult"
DATA_BASENAME = "adult_features"
FEATURE_COLUMNS = ["Hq Value", "evec avg", "deltaH", "frac_Avg"]
GROUP_COLUMN = "Subject_ID"
CHANNEL_COLUMN = "Channel"
TARGET_CANDIDATES = ["is_soz", "is_ez", "EZ", "label"]

N_SPLITS = 10
RANDOM_STATE = 42
DECISION_THRESHOLD = 0.50
PLOT_DPI = 600

# Optional automation/testing variables:
# EZ_DATA_FILE=/full/path/to/file.xlsx
# EZ_OUTPUT_DIR=/full/path/to/output
# EZ_QUICK_TEST=1  -> 2 folds, fewer trees, 150-DPI test figure
QUICK_TEST = os.environ.get("EZ_QUICK_TEST", "0") == "1"
if QUICK_TEST:
    N_SPLITS = 2
    PLOT_DPI = 150

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = Path(
    os.environ.get(
        "EZ_OUTPUT_DIR",
        str(SCRIPT_DIR / "outputs" / "adult_xgboost"),
    )
)

MODEL_NAME = "XGBoost"
MODEL_SHORT = "xgb"
MODEL_HYPERPARAMETERS = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
}


# ============================================================
# INPUT AND STYLE
# ============================================================

def find_data_file() -> Path:
    env_path = os.environ.get("EZ_DATA_FILE", "").strip()
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"EZ_DATA_FILE does not exist: {path}")
        return path

    names = [
        f"{DATA_BASENAME}.xlsx",
        f"{DATA_BASENAME}.xls",
        f"{DATA_BASENAME}.csv",
        f"{DATA_BASENAME}.csv.gz",
    ]
    search_dirs = [
        SCRIPT_DIR,
        SCRIPT_DIR / "data",
        SCRIPT_DIR.parent,
        Path.cwd(),
        Path.cwd() / "data",
    ]
    for directory in search_dirs:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return candidate.resolve()

    raise FileNotFoundError(
        f"Could not find {DATA_BASENAME}.xlsx/.csv. Put the data beside "
        "the script or set EZ_DATA_FILE."
    )


def read_table(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def configure_journal_style() -> str:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = "DejaVu Sans"
    for candidate in ("Arial", "Arimo", "Liberation Sans", "DejaVu Sans"):
        if candidate in installed:
            selected = candidate
            break

    plt.rcParams.update({
        "font.family": selected,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.0,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return selected


# ============================================================
# DATA
# ============================================================

def prepare_data(df: pd.DataFrame):
    target_column = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target_column is None:
        raise ValueError(
            "Target column not found. Expected one of: "
            + ", ".join(TARGET_CANDIDATES)
        )

    required = [GROUP_COLUMN, target_column] + FEATURE_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
            + "\nAvailable columns: " + ", ".join(map(str, df.columns))
        )

    work = df.copy()
    work[GROUP_COLUMN] = work[GROUP_COLUMN].astype(str).str.strip()
    work = work[
        work[GROUP_COLUMN].ne("") & work[GROUP_COLUMN].ne("nan")
    ].copy()

    for column in FEATURE_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    work = work[work[target_column].isin([0, 1])].copy()
    work[target_column] = work[target_column].astype(int)

    if CHANNEL_COLUMN not in work.columns:
        work[CHANNEL_COLUMN] = np.arange(1, len(work) + 1).astype(str)
    else:
        work[CHANNEL_COLUMN] = work[CHANNEL_COLUMN].astype(str)

    X = work[FEATURE_COLUMNS].copy()
    y = work[target_column].to_numpy(dtype=int)
    groups = work[GROUP_COLUMN].to_numpy(dtype=str)

    if np.unique(groups).size < N_SPLITS:
        raise ValueError(
            f"Only {np.unique(groups).size} patients are available, "
            f"but N_SPLITS={N_SPLITS}."
        )
    if np.unique(y).size != 2:
        raise ValueError("The complete dataset must contain both classes.")

    return work, X, y, groups, target_column


# ============================================================
# MODEL
# ============================================================

def build_model(y_train: np.ndarray, fold_seed: int):
    positive = max(1, int((y_train == 1).sum()))
    negative = max(1, int((y_train == 0).sum()))
    estimator = XGBClassifier(
        n_estimators=20 if QUICK_TEST else 500,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        scale_pos_weight=negative / positive,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
        random_state=fold_seed,
        n_jobs=1,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])


# ============================================================
# METRICS
# ============================================================

def safe_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probability))


def safe_auprc(y_true: np.ndarray, probability: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, probability))


def calculate_metrics(y_true, y_pred, probability) -> dict:
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    balanced_accuracy = (
        float(np.nanmean([sensitivity, specificity]))
        if np.isfinite(sensitivity) or np.isfinite(specificity)
        else float("nan")
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_sensitivity": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": balanced_accuracy,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auroc": safe_auc(y_true, probability),
        "auprc": safe_auprc(y_true, probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# JOURNAL DASHBOARD
# ============================================================

def save_dashboard(
    y_true,
    y_pred,
    probability,
    fold_metrics,
    fold_roc_data,
    output_base,
    selected_font,
):
    fig = plt.figure(figsize=(6.69, 6.25), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.88])

    # A. Confusion matrix
    ax_cm = fig.add_subplot(grid[0, 0])
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    image = ax_cm.imshow(matrix, cmap="coolwarm", interpolation="nearest")
    fig.colorbar(image, ax=ax_cm, fraction=0.046, pad=0.04)
    ax_cm.set_xticks([0, 1], labels=["Non-EZ", "EZ"])
    ax_cm.set_yticks([0, 1], labels=["Non-EZ", "EZ"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title("A  Confusion matrix", loc="left", fontweight="bold")

    midpoint = (matrix.max() + matrix.min()) / 2.0
    for row in range(2):
        for col in range(2):
            value = int(matrix[row, col])
            text_color = "white" if value > midpoint else "black"
            ax_cm.text(
                col,
                row,
                f"{value:,}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold",
            )

    # B. ROC curves
    ax_roc = fig.add_subplot(grid[0, 1])
    mean_fpr = np.linspace(0.0, 1.0, 201)
    interpolated = []
    valid_aucs = []

    for fpr, tpr, auc_value, fold_number in fold_roc_data:
        ax_roc.plot(
            fpr, tpr, color="0.75", linewidth=0.7, alpha=0.65
        )
        fold_tpr = np.interp(mean_fpr, fpr, tpr)
        fold_tpr[0] = 0.0
        interpolated.append(fold_tpr)
        valid_aucs.append(auc_value)

    if interpolated:
        roc_array = np.vstack(interpolated)
        mean_tpr = np.nanmean(roc_array, axis=0)
        std_tpr = np.nanstd(roc_array, axis=0)
        mean_tpr[-1] = 1.0
        lower = np.maximum(mean_tpr - std_tpr, 0.0)
        upper = np.minimum(mean_tpr + std_tpr, 1.0)
        mean_auc = float(np.nanmean(valid_aucs))
        std_auc = (
            float(np.nanstd(valid_aucs, ddof=1))
            if len(valid_aucs) > 1
            else 0.0
        )
        ax_roc.plot(
            mean_fpr,
            mean_tpr,
            color="#159447",
            linewidth=1.6,
            label=f"Mean AUROC = {mean_auc:.3f} ± {std_auc:.3f}",
        )
        ax_roc.fill_between(
            mean_fpr,
            lower,
            upper,
            color="#7acb95",
            alpha=0.25,
            linewidth=0,
        )

    pooled_auc = safe_auc(y_true, probability)
    if np.isfinite(pooled_auc):
        pooled_fpr, pooled_tpr, _ = roc_curve(y_true, probability)
        ax_roc.plot(
            pooled_fpr,
            pooled_tpr,
            color="#006d2c",
            linewidth=1.0,
            linestyle="--",
            label=f"Pooled AUROC = {pooled_auc:.3f}",
        )

    ax_roc.plot(
        [0, 1], [0, 1],
        linestyle="--", color="0.45", linewidth=0.8
    )
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("B  ROC curve", loc="left", fontweight="bold")
    ax_roc.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
    ax_roc.legend(loc="lower right", frameon=True)

    # C. Fold-level metric distributions
    ax_box = fig.add_subplot(grid[1, :])
    metric_columns = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall_sensitivity", "Recall"),
        ("f1", "F1"),
        ("auroc", "AUROC"),
    ]
    values = [
        fold_metrics[column].dropna().to_numpy(dtype=float)
        for column, label in metric_columns
    ]
    labels = [label for column, label in metric_columns]

    box_kwargs = {
        "patch_artist": True,
        "widths": 0.55,
        "showfliers": True,
        "medianprops": {"color": "black", "linewidth": 0.9},
        "whiskerprops": {"linewidth": 0.8},
        "capprops": {"linewidth": 0.8},
        "boxprops": {"linewidth": 0.8},
        "flierprops": {
            "marker": "o", "markersize": 2.5, "alpha": 0.45
        },
    }
    try:
        box = ax_box.boxplot(
            values, tick_labels=labels, **box_kwargs
        )
    except TypeError:
        box = ax_box.boxplot(
            values, labels=labels, **box_kwargs
        )

    colors = ["#2b83ba", "#f28e2b", "#3a9d3f", "#d64541", "#8e63b0"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.90)

    finite_groups = [
        value[np.isfinite(value)] for value in values
        if np.isfinite(value).any()
    ]
    if finite_groups:
        all_finite = np.concatenate(finite_groups)
        lower_limit = max(0.0, float(np.min(all_finite)) - 0.05)
        upper_limit = min(1.0, float(np.max(all_finite)) + 0.05)
        if upper_limit - lower_limit < 0.20:
            center = (upper_limit + lower_limit) / 2.0
            lower_limit = max(0.0, center - 0.10)
            upper_limit = min(1.0, center + 0.10)
        ax_box.set_ylim(lower_limit, upper_limit)

    ax_box.set_ylabel("Score")
    ax_box.set_title(
        "C  Ten-fold patient-grouped metrics",
        loc="left",
        fontweight="bold",
    )
    ax_box.grid(
        axis="y", linestyle="--", linewidth=0.45, alpha=0.45
    )

    fig.suptitle(
        f"{COHORT.capitalize()} cohort — {MODEL_NAME}",
        y=1.01,
        fontweight="bold",
    )

    fig.savefig(
        output_base.with_suffix(".png"),
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
    )
    if not QUICK_TEST:
        fig.savefig(
            output_base.with_suffix(".tiff"),
            dpi=PLOT_DPI,
            pil_kwargs={"compression": "tiff_lzw"},
            bbox_inches="tight",
        )
    plt.close(fig)

    audit = {
        "font": selected_font,
        "png_dpi": PLOT_DPI,
        "figure_width_inches": 6.69,
        "pdf_fonttype": 42,
        "class_labels": ["Non-EZ", "EZ"],
    }
    output_base.with_name(
        output_base.name + "_font_audit.json"
    ).write_text(json.dumps(audit, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_font = configure_journal_style()

    data_path = find_data_file()
    dataframe = read_table(data_path)
    work, X, y, groups, target_column = prepare_data(dataframe)

    print("=" * 72)
    print(f"Model:       {MODEL_NAME}")
    print(f"Cohort:      {COHORT}")
    print(f"Input:       {data_path}")
    print(f"Rows:        {len(work):,}")
    print(f"Patients:    {work[GROUP_COLUMN].nunique():,}")
    print(f"Non-EZ:      {int((y == 0).sum()):,}")
    print(f"EZ:          {int((y == 1).sum()):,}")
    print(f"Features:    {FEATURE_COLUMNS}")
    print(f"CV:          {N_SPLITS}-fold StratifiedGroupKFold")
    print(f"Threshold:   {DECISION_THRESHOLD:.2f}")
    print(f"Output:      {OUTPUT_DIR}")
    print("=" * 72)

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    prediction_rows = []
    fold_rows = []
    split_rows = []
    fold_roc_data = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups),
        start=1,
    ):
        train_patients = sorted(
            np.unique(groups[train_idx]).tolist()
        )
        test_patients = sorted(
            np.unique(groups[test_idx]).tolist()
        )
        overlap = sorted(
            set(train_patients).intersection(test_patients)
        )
        if overlap:
            raise RuntimeError(
                f"Patient leakage detected in fold {fold}: {overlap}"
            )

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = build_model(y_train, RANDOM_STATE + fold)
        model.fit(X_train, y_train)

        probability = model.predict_proba(X_test)[:, 1]
        prediction = (
            probability >= DECISION_THRESHOLD
        ).astype(int)

        metrics = calculate_metrics(
            y_test, prediction, probability
        )
        metrics.update({
            "fold": fold,
            "n_train_rows": len(train_idx),
            "n_test_rows": len(test_idx),
            "n_train_patients": len(train_patients),
            "n_test_patients": len(test_patients),
        })
        fold_rows.append(metrics)

        if np.unique(y_test).size == 2:
            fpr, tpr, _ = roc_curve(y_test, probability)
            fold_roc_data.append(
                (fpr, tpr, metrics["auroc"], fold)
            )
        else:
            print(
                f"Fold {fold:02d}: AUROC/AUPRC are undefined "
                f"because y_true contains only "
                f"{np.unique(y_test).tolist()}."
            )

        for local_position, row_index in enumerate(test_idx):
            prediction_rows.append({
                "row_index": int(row_index),
                GROUP_COLUMN: str(
                    work.iloc[row_index][GROUP_COLUMN]
                ),
                CHANNEL_COLUMN: str(
                    work.iloc[row_index][CHANNEL_COLUMN]
                ),
                "y_true": int(y_test[local_position]),
                "probability_ez": float(
                    probability[local_position]
                ),
                "y_pred": int(prediction[local_position]),
                "fold": int(fold),
            })

        split_rows.append({
            "fold": fold,
            "train_patients": "|".join(train_patients),
            "test_patients": "|".join(test_patients),
            "n_train_patients": len(train_patients),
            "n_test_patients": len(test_patients),
            "patient_overlap_count": len(overlap),
        })

        print(
            f"Fold {fold:02d}/{N_SPLITS} | "
            f"Acc={metrics['accuracy']:.3f} | "
            f"Sens={metrics['sensitivity']:.3f} | "
            f"Spec={metrics['specificity']:.3f} | "
            f"AUROC={metrics['auroc']:.3f} | "
            f"AUPRC={metrics['auprc']:.3f}"
        )

    predictions = pd.DataFrame(
        prediction_rows
    ).sort_values("row_index")
    fold_metrics = pd.DataFrame(fold_rows)
    split_audit = pd.DataFrame(split_rows)

    pooled = calculate_metrics(
        predictions["y_true"].to_numpy(dtype=int),
        predictions["y_pred"].to_numpy(dtype=int),
        predictions["probability_ez"].to_numpy(dtype=float),
    )
    pooled.update({
        "cohort": COHORT,
        "model": MODEL_NAME,
        "n_rows": len(predictions),
        "n_patients": int(work[GROUP_COLUMN].nunique()),
        "n_splits": N_SPLITS,
        "decision_threshold": DECISION_THRESHOLD,
        "target_column_read": target_column,
    })

    predictions.to_csv(
        OUTPUT_DIR / f"{COHORT}_{MODEL_SHORT}_oof_predictions.csv",
        index=False,
    )
    fold_metrics.to_csv(
        OUTPUT_DIR / f"{COHORT}_{MODEL_SHORT}_fold_metrics.csv",
        index=False,
    )
    split_audit.to_csv(
        OUTPUT_DIR / f"{COHORT}_{MODEL_SHORT}_split_audit.csv",
        index=False,
    )
    pd.DataFrame([pooled]).to_csv(
        OUTPUT_DIR / f"{COHORT}_{MODEL_SHORT}_pooled_summary.csv",
        index=False,
    )

    hyperparameter_record = {
        "cohort": COHORT,
        "model": MODEL_NAME,
        "features": FEATURE_COLUMNS,
        "group_column": GROUP_COLUMN,
        "target_column": target_column,
        "n_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "decision_threshold": DECISION_THRESHOLD,
        "no_smote": True,
        "test_folds_resampled": False,
        "hyperparameters": MODEL_HYPERPARAMETERS,
    }
    (
        OUTPUT_DIR
        / f"{COHORT}_{MODEL_SHORT}_hyperparameters.json"
    ).write_text(
        json.dumps(hyperparameter_record, indent=2),
        encoding="utf-8",
    )

    save_dashboard(
        predictions["y_true"].to_numpy(dtype=int),
        predictions["y_pred"].to_numpy(dtype=int),
        predictions["probability_ez"].to_numpy(dtype=float),
        fold_metrics,
        fold_roc_data,
        OUTPUT_DIR
        / f"{COHORT}_{MODEL_SHORT}_performance_dashboard",
        selected_font,
    )

    print("\nPooled out-of-fold performance")
    print("-" * 72)
    for key in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "auroc",
        "auprc",
    ):
        print(f"{key:20s}: {pooled[key]:.4f}")
    print(
        "Confusion matrix     : "
        f"TN={pooled['tn']}, FP={pooled['fp']}, "
        f"FN={pooled['fn']}, TP={pooled['tp']}"
    )
    print(f"\nDone. Results saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
