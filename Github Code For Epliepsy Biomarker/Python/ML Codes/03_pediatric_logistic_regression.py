from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ID_COLUMN = "Subject_ID"
CHANNEL_COLUMN = "Channel"
TARGET_COLUMN = "is_soz"
FEATURE_COLUMNS = ["Hq Value", "evec avg", "deltaH", "frac_Avg"]
DATASET_NAME = "Pediatric"
MODEL_NAME = "Logistic Regression"
OUTPUT_DIR_NAME = "results_03_pediatric_logistic"
N_SPLITS = 10
RANDOM_STATE = 42
THRESHOLD = 0.50




def select_csv() -> Path:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    filename = filedialog.askopenfilename(
        title=f"Select {DATASET_NAME} mean-feature CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    if not filename:
        raise SystemExit("No CSV file selected.")
    return Path(filename)


def show_message(title: str, message: str, error: bool = False) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if error:
            messagebox.showerror(title, message)
        else:
            messagebox.showinfo(title, message)
        root.destroy()
    except Exception:
        print(f"{title}: {message}")


def load_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    data = pd.read_csv(path)
    required = [ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.copy()
    data[ID_COLUMN] = data[ID_COLUMN].astype(str).str.strip()
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=[ID_COLUMN, TARGET_COLUMN]).reset_index(drop=False)
    data = data.rename(columns={"index": "source_row"})
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

    labels = set(data[TARGET_COLUMN].unique())
    if labels != {0, 1}:
        raise ValueError(f"{TARGET_COLUMN} must contain binary labels 0 and 1; found {sorted(labels)}")
    if data[ID_COLUMN].nunique() < N_SPLITS:
        raise ValueError(f"At least {N_SPLITS} patients are required.")

    return data


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fold_auc = roc_auc_score(y_true, y_prob) if np.unique(y_true).size == 2 else np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Sensitivity": safe_ratio(tp, tp + fn),
        "Specificity": safe_ratio(tn, tn + fp),
        "AUC": fold_auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def build_pipeline(random_state: int, k_neighbors: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=random_state, k_neighbors=k_neighbors)),
            (
                "model",
                LogisticRegression(
                    C=np.inf,
                    solver="lbfgs",
                    max_iter=10000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def create_splits(data: pd.DataFrame):
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN].to_numpy()
    groups = data[ID_COLUMN].to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    yield from splitter.split(X, y, groups)


def run_cross_validation(data: pd.DataFrame):
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN].to_numpy()
    groups = data[ID_COLUMN].to_numpy()

    oof_probability = np.full(len(data), np.nan)
    oof_prediction = np.full(len(data), -1, dtype=int)
    oof_fold = np.full(len(data), -1, dtype=int)
    fold_metric_rows = []
    fold_roc_rows = []
    assignment_lines = []

    for fold_number, (train_idx, test_idx) in enumerate(create_splits(data), start=1):
        train_patients = sorted(np.unique(groups[train_idx]).tolist())
        test_patients = sorted(np.unique(groups[test_idx]).tolist())
        overlap = set(train_patients).intersection(test_patients)
        if overlap:
            raise RuntimeError(f"Patient leakage in fold {fold_number}: {sorted(overlap)}")

        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        class_counts = np.bincount(y_train, minlength=2)
        minority_count = int(class_counts.min())
        if minority_count < 2:
            raise ValueError(f"Fold {fold_number} has fewer than two minority-class training rows.")
        k_neighbors = min(5, minority_count - 1)

        pipeline = build_pipeline(RANDOM_STATE + fold_number, k_neighbors)
        pipeline.fit(X_train, y_train)

        probability = pipeline.predict_proba(X_test)[:, 1]
        prediction = (probability >= THRESHOLD).astype(int)

        oof_probability[test_idx] = probability
        oof_prediction[test_idx] = prediction
        oof_fold[test_idx] = fold_number

        metrics = calculate_metrics(y_test, prediction, probability)
        metrics.update({
            "Fold": fold_number,
            "Train patients": len(train_patients),
            "Test patients": len(test_patients),
            "Train rows": len(train_idx),
            "Test rows": len(test_idx),
            "Test patient IDs": ", ".join(test_patients),
        })
        fold_metric_rows.append(metrics)

        if np.unique(y_test).size == 2:
            fpr, tpr, _ = roc_curve(y_test, probability)
            fold_roc_rows.append((fold_number, fpr, tpr))
        else:
            warnings.warn(f"Fold {fold_number} has one test class; fold ROC omitted.", RuntimeWarning)

        assignment_lines.extend([
            f"Fold {fold_number}:",
            f"Train {len(train_patients)} patients",
            f"Test {len(test_patients)} patients",
            ", ".join(test_patients),
            "",
        ])

        print(
            f"Fold {fold_number:02d} | Train patients={len(train_patients)} | "
            f"Test patients={len(test_patients)} | Accuracy={metrics['Accuracy']:.4f} | "
            f"Sensitivity={metrics['Sensitivity']:.4f} | "
            f"Specificity={metrics['Specificity']:.4f} | AUC={metrics['AUC']:.4f}"
        )

    if np.isnan(oof_probability).any() or (oof_prediction < 0).any():
        raise RuntimeError("Some rows did not receive an out-of-fold prediction.")

    fold_metrics = pd.DataFrame(fold_metric_rows).sort_values("Fold")
    keep_columns = ["source_row", ID_COLUMN]
    if CHANNEL_COLUMN in data.columns:
        keep_columns.append(CHANNEL_COLUMN)
    keep_columns.append(TARGET_COLUMN)

    predictions = data[keep_columns].copy()
    predictions["Fold"] = oof_fold
    predictions["Probability_EZ"] = oof_probability
    predictions["Predicted_class"] = oof_prediction
    predictions["Predicted_label"] = np.where(oof_prediction == 1, "EZ", "Non-EZ")

    pooled_metrics = calculate_metrics(y, oof_prediction, oof_probability)
    pooled_metrics.update({
        "Dataset": DATASET_NAME,
        "Model": MODEL_NAME,
        "Patients": data[ID_COLUMN].nunique(),
        "Rows": len(data),
        "Threshold": THRESHOLD,
        "Mean fold AUC": fold_metrics["AUC"].mean(skipna=True),
        "SD fold AUC": fold_metrics["AUC"].std(skipna=True, ddof=1),
    })

    return predictions, fold_metrics, pooled_metrics, fold_roc_rows, assignment_lines


def plot_performance(
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    pooled_metrics: dict,
    fold_roc_rows: list,
    output_path: Path,
) -> None:
    y_true = predictions[TARGET_COLUMN].to_numpy()
    y_pred = predictions["Predicted_class"].to_numpy()
    y_prob = predictions["Probability_EZ"].to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig = plt.figure(figsize=(12, 11), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.05, 0.95],
        hspace=0.42,
        wspace=0.35,
        left=0.07,
        right=0.97,
        bottom=0.07,
        top=0.79,
    )
    fig.suptitle(
        f"{DATASET_NAME} {MODEL_NAME} Performance Summary\n"
        "Patient-wise testing; all fitting operations use training patients only",
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )

    ax_cm = fig.add_subplot(grid[0, 0])
    image = ax_cm.imshow(cm, interpolation="nearest", cmap="coolwarm")
    colorbar = fig.colorbar(image, ax=ax_cm, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=9)
    text_threshold = (cm.max() + cm.min()) / 2
    for row in range(2):
        for column in range(2):
            ax_cm.text(
                column,
                row,
                f"{cm[row, column]:d}",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="white" if cm[row, column] < text_threshold else "black",
            )
    ax_cm.set_xticks([0, 1], labels=["Non-EZ", "EZ"])
    ax_cm.set_yticks([0, 1], labels=["Non-EZ", "EZ"])
    ax_cm.set_xlabel("Predicted class")
    ax_cm.set_ylabel("Actual class")
    ax_cm.set_title(
        f"A  Confusion Matrix\n{DATASET_NAME} – {MODEL_NAME}\n"
        "(pooled unseen-patient predictions)",
        fontweight="bold",
    )

    ax_roc = fig.add_subplot(grid[0, 1])
    base_fpr = np.linspace(0, 1, 400)
    interpolated_tprs = []
    for _, fpr, tpr in fold_roc_rows:
        ax_roc.plot(fpr, tpr, linewidth=0.8, alpha=0.22)
        interpolated = np.interp(base_fpr, fpr, tpr)
        interpolated[0] = 0
        interpolated_tprs.append(interpolated)

    if interpolated_tprs:
        tpr_array = np.vstack(interpolated_tprs)
        mean_tpr = tpr_array.mean(axis=0)
        mean_tpr[-1] = 1
        sd_tpr = tpr_array.std(axis=0)
        ax_roc.fill_between(
            base_fpr,
            np.maximum(mean_tpr - sd_tpr, 0),
            np.minimum(mean_tpr + sd_tpr, 1),
            alpha=0.16,
        )
        ax_roc.plot(
            base_fpr,
            mean_tpr,
            linewidth=2.2,
            label=f"Mean fold AUROC = {pooled_metrics['Mean fold AUC']:.2f} "
            f"± {pooled_metrics['SD fold AUC']:.2f}",
        )

    pooled_fpr, pooled_tpr, _ = roc_curve(y_true, y_prob)
    pooled_auc = auc(pooled_fpr, pooled_tpr)
    ax_roc.plot(
        pooled_fpr,
        pooled_tpr,
        linestyle=":",
        linewidth=2,
        label=f"Pooled AUROC = {pooled_auc:.2f}",
    )
    ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Chance")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.set_xlabel("False-positive rate")
    ax_roc.set_ylabel("True-positive rate")
    ax_roc.grid(True, linestyle="--", alpha=0.28)
    ax_roc.legend(loc="lower right", fontsize=9)
    ax_roc.set_title("B  ROC Curve\nPatient-grouped outer-test folds", fontweight="bold")

    ax_box = fig.add_subplot(grid[1, :])
    metric_names = ["Accuracy", "Sensitivity", "Specificity", "AUC"]
    values = [fold_metrics[name].dropna().to_numpy() for name in metric_names]
    box = ax_box.boxplot(
        values,
        labels=metric_names,
        patch_artist=True,
        widths=0.58,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, color in zip(box["boxes"], ["#6f92b7", "#f28e2b", "#69a95e", "#b07aa1"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    ax_box.set_ylim(0, 1.02)
    ax_box.set_ylabel("Score")
    ax_box.grid(axis="y", linestyle="--", alpha=0.28)
    ax_box.set_title(
        "C  Outer-Fold Metrics\nAccuracy, Sensitivity, Specificity, and AUC",
        fontweight="bold",
    )

    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_results(
    output_dir: Path,
    predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    pooled_metrics: dict,
    assignment_lines: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame([pooled_metrics]).to_csv(output_dir / "pooled_summary.csv", index=False)
    (output_dir / "patient_fold_assignments.txt").write_text(
        "\n".join(assignment_lines), encoding="utf-8"
    )


def run_analysis(input_path: Path) -> Path:
    output_dir = input_path.parent / OUTPUT_DIR_NAME
    data = load_data(input_path)
    print(
        f"Loaded {len(data)} rows from {data[ID_COLUMN].nunique()} patients | "
        f"Non-EZ={(data[TARGET_COLUMN] == 0).sum()} | EZ={(data[TARGET_COLUMN] == 1).sum()}"
    )

    predictions, fold_metrics, pooled_metrics, fold_roc_rows, assignment_lines = (
        run_cross_validation(data)
    )
    save_results(output_dir, predictions, fold_metrics, pooled_metrics, assignment_lines)
    figure_path = output_dir / "performance_summary_600dpi.png"
    plot_performance(predictions, fold_metrics, pooled_metrics, fold_roc_rows, figure_path)

    print("\nPooled unseen-patient results")
    print(f"Accuracy: {pooled_metrics['Accuracy']:.4f}")
    print(f"Sensitivity: {pooled_metrics['Sensitivity']:.4f}")
    print(f"Specificity: {pooled_metrics['Specificity']:.4f}")
    print(f"Pooled AUC: {pooled_metrics['AUC']:.4f}")
    print(
        f"Mean fold AUC: {pooled_metrics['Mean fold AUC']:.4f} "
        f"± {pooled_metrics['SD fold AUC']:.4f}"
    )
    print(f"Saved results: {output_dir}")
    return output_dir


def main() -> None:
    try:
        input_path = select_csv()
        output_dir = run_analysis(input_path)
        show_message(
            "Analysis complete",
            f"{DATASET_NAME} {MODEL_NAME} analysis completed.\n\nResults saved in:\n{output_dir}",
        )
    except SystemExit:
        return
    except Exception as exc:
        show_message("Analysis failed", str(exc), error=True)
        raise


if __name__ == "__main__":
    main()
