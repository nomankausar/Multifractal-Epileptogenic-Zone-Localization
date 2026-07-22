
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, learning_curve
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

try:
    from imblearn.combine import SMOTETomek
    IMBLEARN_OK = True
except Exception:
    IMBLEARN_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False


# =========================
# Hard-locked settings
# =========================
K_BEST = 4
CV_FOLDS = 10
GRID_CV_FOLDS = 10


# ------------------------- Utilities -------------------------

def _safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _savefig(path: Path, dpi: int = 300):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()

def _detect_gpu_kwargs() -> Dict[str, Any]:
    # XGBoost >=2.x: device="cuda" + tree_method="hist"
    return {"tree_method": "hist", "device": "cuda"}

def _cpu_kwargs() -> Dict[str, Any]:
    return {"tree_method": "hist", "device": "cpu"}

def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file: {path}")

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\u00a0", " ") for c in df.columns]
    return df

def _pick_target(df: pd.DataFrame, target: str) -> str:
    if target in df.columns:
        return target
    candidates = [c for c in df.columns if c.lower() == target.lower()]
    if candidates:
        return candidates[0]
    raise ValueError(f"Target column '{target}' not found. Available: {list(df.columns)[:30]} ...")

def _split_X_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    return X, y

def _drop_non_numeric_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    Xn = X[numeric_cols].copy()
    return Xn, numeric_cols

def _maybe_balance(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, str]:
    if IMBLEARN_OK:
        try:
            smt = SMOTETomek(random_state=random_state)
            Xb, yb = smt.fit_resample(X, y)
            return Xb, yb, "SMOTETomek"
        except Exception:
            return X, y, "NoResample (SMOTETomek failed)"
    return X, y, "NoResample (imblearn not installed)"

def _coerce_required_features_numeric(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Prediction-time safe:
    - Ensure all required feature_cols exist
    - Keep exact order
    - Coerce to numeric (bad strings -> NaN)
    """
    X = df.reindex(columns=feature_cols).copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


# ------------------------- Plot helpers -------------------------

def plot_confusion_matrix_seaborn(cm: np.ndarray, out_path: Path, title: str):
    """
    Match your screenshot style:
    - coolwarm
    - big title
    - white annotations
    - square cells
    - no gridlines
    """
    sns.set_theme(style="white", context="talk")
    plt.figure(figsize=(8, 7))

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="coolwarm",
        vmin=0,
        vmax=int(np.max(cm)),
        square=True,
        linewidths=0,
        cbar=True,
        annot_kws={"color": "white", "fontsize": 22},
    )

    ax.set_title(title, fontsize=30, pad=18)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(["Non-SOZ", "SOZ"], fontsize=20)
    ax.set_yticklabels(["Non-SOZ", "SOZ"], fontsize=20, rotation=90, va="center")

    ax.tick_params(axis="both", which="both", length=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_val: float, out_path: Path, title: str):
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}", color="green")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    _savefig(out_path, dpi=300)

def plot_pr_curve(rec: np.ndarray, prec: np.ndarray, ap: float, out_path: Path, title: str):
    plt.figure(figsize=(8, 6))
    plt.plot(rec, prec, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    _savefig(out_path, dpi=300)

def plot_metrics_boxplot(metrics_df: pd.DataFrame, out_path: Path, title: str):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=metrics_df)
    plt.title(title)
    plt.grid(True)
    _savefig(out_path, dpi=300)

def plot_learning_curve_auc(estimator, X, y, out_path: Path, title: str, cv: int = 5):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=1,  # safer on Windows
        shuffle=True,
        random_state=42,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, "o-", label="Train AUC", color="blue")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="blue")
    plt.plot(train_sizes, val_mean, "o-", label="Validation AUC", color="orange")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color="orange")
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("AUC Score")
    plt.legend()
    plt.grid(True)

    _savefig(out_path, dpi=300)


# ------------------------- Core training -------------------------

def train_and_export(
    data_path: Path,
    target: str = "is_soz",
    out_dir: Path = Path("artifacts_xgb"),
    test_size: float = 0.2,
    topk_imp: int = 25,
    random_state: int = 42,
) -> Path:
    if not XGB_OK:
        raise RuntimeError("xgboost is not installed. Install: pip install xgboost")

    out_dir = _safe_mkdir(out_dir)

    df = _clean_columns(_read_table(data_path))
    target_col = _pick_target(df, target)
    X_raw, y = _split_X_y(df, target_col)

    id_cols = [c for c in X_raw.columns if c.lower() in ("channel", "channel_name", "chan", "name", "electrode")]
    id_col = id_cols[0] if id_cols else None

    # Numeric features only
    X_num, _ = _drop_non_numeric_features(X_raw)
    if X_num.shape[1] == 0:
        raise ValueError("No numeric feature columns found. Dataset must contain numeric features.")

    kbest = int(max(1, min(int(K_BEST), X_num.shape[1])))

    # Holdout split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_num, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Preprocess pipeline
    pre = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("kbest", SelectKBest(score_func=f_classif, k=kbest)),
    ])

    # Fit preprocessing on training only
    X_train_p = pre.fit_transform(X_train_df, y_train)
    X_test_p = pre.transform(X_test_df)

    # Optional balance (train only)
    X_train_bal, y_train_bal, bal_note = _maybe_balance(X_train_p, y_train.to_numpy(), random_state=random_state)

    # Base model params
    base_params = dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        min_child_weight=1.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=random_state,
        n_jobs=max(11, os.cpu_count() or 1),
    )

    gpu_kwargs = _detect_gpu_kwargs()
    cpu_kwargs = _cpu_kwargs()

    # Light grid
    param_grid = {
        "max_depth": [4, 5, 6],
        "subsample": [0.8, 0.9],
        "colsample_bytree": [0.8, 0.9],
        "min_child_weight": [1, 3],
        "reg_lambda": [1.0, 2.0],
    }

    skf_grid = StratifiedKFold(n_splits=GRID_CV_FOLDS, shuffle=True, random_state=random_state)

    def _run_grid(model) -> GridSearchCV:
        gs = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="roc_auc",
            n_jobs=11,
            cv=skf_grid,
            verbose=1,
        )
        gs.fit(X_train_bal, y_train_bal)
        return gs

    used_device = "cuda"
    try:
        gs = _run_grid(xgb.XGBClassifier(**base_params, **gpu_kwargs))
    except Exception:
        used_device = "cpu"
        gs = _run_grid(xgb.XGBClassifier(**base_params, **cpu_kwargs))

    best_model = gs.best_estimator_

    # Switch to CPU for stable predict_proba
    try:
        if used_device == "cuda":
            best_model.set_params(device="cpu")
    except Exception:
        pass

    # -------------------------
    # Holdout evaluation
    # -------------------------
    y_prob = best_model.predict_proba(X_test_p)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, digits=4)
    cm_holdout = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    plot_confusion_matrix_seaborn(cm_holdout, out_dir / "confusion_matrix_holdout.png",
                                  title="Confusion Matrix - XGBoost (Holdout)")
    plot_roc_curve(fpr, tpr, roc_auc, out_dir / "roc_curve_holdout.png",
                   title="ROC Curve - XGBoost (Holdout)")
    plot_pr_curve(pr_rec, pr_prec, ap, out_dir / "precision_recall_curve_holdout.png",
                  title="Precision-Recall - XGBoost (Holdout)")

    # -------------------------
    # 10-Fold CV metrics
    # -------------------------
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=random_state)

    mean_fpr = np.linspace(0, 1, 200)
    tprs, aucs = [], []
    cms_total = np.zeros((2, 2), dtype=int)
    metrics_rows = []

    final_params = base_params.copy()
    final_params.update(gs.best_params_)

    cv_used_device = used_device

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_num, y), start=1):
        X_tr_df = X_num.iloc[tr_idx]
        y_tr = y.iloc[tr_idx].to_numpy()
        X_te_df = X_num.iloc[te_idx]
        y_te = y.iloc[te_idx].to_numpy()

        # Fold preprocess
        pre_fold = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("kbest", SelectKBest(score_func=f_classif, k=kbest)),
        ])
        X_tr_p = pre_fold.fit_transform(X_tr_df, y_tr)
        X_te_p = pre_fold.transform(X_te_df)

        # Balance on fold train only
        X_tr_b, y_tr_b, _ = _maybe_balance(X_tr_p, y_tr, random_state=random_state)

        # Fit model fold (GPU-first, CPU fallback)
        try:
            model_fold = xgb.XGBClassifier(**final_params, **gpu_kwargs)
            model_fold.fit(X_tr_b, y_tr_b)
            cv_used_device = "cuda"
        except Exception:
            model_fold = xgb.XGBClassifier(**final_params, **cpu_kwargs)
            model_fold.fit(X_tr_b, y_tr_b)
            cv_used_device = "cpu"

        try:
            if cv_used_device == "cuda":
                model_fold.set_params(device="cpu")
        except Exception:
            pass

        prob = model_fold.predict_proba(X_te_p)[:, 1]
        pred = (prob >= 0.5).astype(int)

        acc = accuracy_score(y_te, pred)
        prc = precision_score(y_te, pred, zero_division=0)
        rcl = recall_score(y_te, pred, zero_division=0)
        f1 = f1_score(y_te, pred, zero_division=0)

        fpr_i, tpr_i, _ = roc_curve(y_te, prob)
        auc_i = auc(fpr_i, tpr_i)

        cm_i = confusion_matrix(y_te, pred)
        cms_total += cm_i

        tpr_interp = np.interp(mean_fpr, fpr_i, tpr_i)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        aucs.append(auc_i)

        metrics_rows.append({
            "Fold": fold,
            "Accuracy": acc,
            "Precision": prc,
            "Recall": rcl,
            "F1": f1,
            "AUC": auc_i,
        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "metrics_10fold.csv", index=False)

    metrics_only = metrics_df.drop(columns=["Fold"], errors="ignore")
    plot_metrics_boxplot(metrics_only, out_dir / "boxplot_metrics_10fold.png",
                         title=f"XGBoost Metrics - {CV_FOLDS}-Fold")

    plot_confusion_matrix_seaborn(cms_total, out_dir / "confusion_matrix_total_10fold.png",
                                  title=f"Confusion Matrix (Summed) - {CV_FOLDS}-Fold")

    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))

    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", lw=1)

    # Optional: show per-fold ROC curves faintly
    for tpr_i in tprs:
        plt.plot(mean_fpr, tpr_i, alpha=0.15, lw=1)

    # Mean ROC + shaded band (±1 std)
    plt.plot(
        mean_fpr,
        mean_tpr,
        color="green",
        lw=2,
        label=f"Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})",
    )
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.20)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Mean ± 1 SD) - {CV_FOLDS}-Fold")
    plt.legend(loc="lower right")
    _savefig(out_dir / "roc_curve_mean_10fold.png", dpi=300)

    # Learning curve on preprocessed+balanced training data for stability
    plot_learning_curve_auc(best_model, X_train_bal, y_train_bal,
                            out_dir / "learning_curve_auc.png",
                            title="Learning Curve (AUC)",
                            cv=5)

    # Feature importance (selected space)
    selector: SelectKBest = pre.named_steps["kbest"]
    support_mask = selector.get_support()
    train_cols_order = list(X_train_df.columns)
    selected_features = [c for c, keep in zip(train_cols_order, support_mask) if keep]

    importances = getattr(best_model, "feature_importances_", None)
    if importances is not None and len(selected_features) == len(importances):
        idx = np.argsort(importances)[::-1][: max(1, min(int(topk_imp), len(importances)))]
        plt.figure(figsize=(8, max(3, 0.25 * len(idx))))
        plt.barh([selected_features[i] for i in idx][::-1], importances[idx][::-1])
        plt.xlabel("Importance")
        plt.title(f"Top {len(idx)} Feature Importances")
        _savefig(out_dir / "feature_importance_topk.png", dpi=300)

        pd.DataFrame({
            "feature": [selected_features[i] for i in idx],
            "importance": importances[idx],
        }).to_csv(out_dir / "feature_importance_topk.csv", index=False)

    # Export model bundle for software
    bundle = {
        "preprocess": pre,
        "model": best_model,
        "feature_columns": list(X_train_df.columns),  # exact order used
        "selected_columns": selected_features,
        "threshold": 0.5,
    }
    joblib.dump(bundle, out_dir / "model_bundle.joblib", compress=3)

    try:
        best_model.save_model(str(out_dir / "xgb_model.json"))
    except Exception:
        pass

    meta = {
        "data_path": str(data_path),
        "target": target_col,
        "n_rows": int(df.shape[0]),
        "n_features_total_numeric": int(X_num.shape[1]),
        "kbest": int(kbest),
        "cv_folds_metrics": int(CV_FOLDS),
        "grid_cv_folds": int(GRID_CV_FOLDS),
        "test_size": float(test_size),
        "best_params": gs.best_params_,
        "best_grid_cv_auc": float(gs.best_score_),
        "holdout_roc_auc": float(roc_auc),
        "holdout_average_precision": float(ap),
        "balance_method_train": bal_note,
        "used_device_grid": used_device,
        "used_device_cv": cv_used_device,
        "id_col": id_col,
        "metrics_10fold_mean": metrics_only.mean(numeric_only=True).to_dict(),
        "metrics_10fold_std": metrics_only.std(numeric_only=True).to_dict(),
        "mean_auc_10fold": mean_auc,
        "std_auc_10fold": std_auc,
    }
    (out_dir / "bundle_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary = []
    summary.append("=== EZ XGBoost Training Summary (FIXED) ===")
    summary.append(f"Data: {data_path}")
    summary.append(f"Target: {target_col}")
    summary.append(f"Rows: {df.shape[0]}  Numeric features: {X_num.shape[1]}  KBest: {kbest}")
    summary.append(f"Balance (train split): {bal_note}")
    summary.append(f"Grid device: {used_device} | CV device (final): {cv_used_device}")
    summary.append(f"Grid CV folds: {GRID_CV_FOLDS}")
    summary.append(f"Metrics CV folds: {CV_FOLDS}")
    summary.append(f"Grid best CV AUC: {gs.best_score_:.4f}")
    summary.append(f"Holdout AUC: {roc_auc:.4f}   Holdout AP: {ap:.4f}")
    summary.append("")
    summary.append(f"{CV_FOLDS}-Fold Metrics (mean ± std):")
    for col in metrics_only.columns:
        arr = metrics_only[col].to_numpy(dtype=float)
        summary.append(f"  {col}: {arr.mean():.4f} ± {arr.std():.4f}")
    summary.append("")
    summary.append("Holdout classification report:")
    summary.append(report)

    (out_dir / "run_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    return out_dir


def predict_table(input_path: Path, bundle_path: Path, out_csv: Optional[Path] = None) -> Path:
    """
    Prediction-only helper (for GUI):
    - Loads model_bundle.joblib
    - Reads input table (csv/xlsx) WITHOUT is_soz
    - Outputs CSV with prob_soz and pred_soz
    """
    bundle = joblib.load(bundle_path)
    pre: Pipeline = bundle["preprocess"]
    model = bundle["model"]
    feature_cols: List[str] = list(bundle["feature_columns"])
    threshold: float = float(bundle.get("threshold", 0.5))

    df = _clean_columns(_read_table(input_path))

    id_cols = [c for c in df.columns if c.lower() in ("channel", "channel_name", "chan", "name", "electrode")]
    id_series = df[id_cols[0]] if id_cols else pd.Series(range(len(df)), name="row")

    X = _coerce_required_features_numeric(df, feature_cols)

    Xp = pre.transform(X)
    prob = model.predict_proba(Xp)[:, 1]
    pred = (prob >= threshold).astype(int)

    out = pd.DataFrame({"id": id_series.values, "prob_soz": prob, "pred_soz": pred})

    if out_csv is None:
        out_csv = input_path.with_suffix("").with_name(input_path.stem + "_predictions.csv")
    out.to_csv(out_csv, index=False)
    return out_csv


# ------------------------- CLI -------------------------

def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(prog="EZ_XGBoost_Full_Train_Export_Plots_Seaborn_10Fold_FIXED.py")
    parser.add_argument("--data", type=str, default=None, help="Training data path (.xlsx or .csv).")
    parser.add_argument("--target", type=str, default="is_soz", help="Target column (default: is_soz).")
    parser.add_argument("--out", type=str, default=None, help="Output directory for artifacts.")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    # Auto data selection
    if args.data is None:
        candidate = script_dir / "merged v2.xlsx"
        if candidate.exists():
            data_path = candidate
        else:
            picks = list(script_dir.glob("*.xlsx")) + list(script_dir.glob("*.csv"))
            if not picks:
                raise SystemExit("No --data provided and no .xlsx/.csv found next to the script.")
            data_path = picks[0]
    else:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = (script_dir / data_path).resolve()

    out_dir = Path(args.out) if args.out else (script_dir / "artifacts_xgb")
    if not out_dir.is_absolute():
        out_dir = (script_dir / out_dir).resolve()

    print("Script folder :", script_dir)
    print("Training data :", data_path)
    print("Output folder :", out_dir)
    print("Target column :", args.target)
    print("Holdout test  :", args.test_size)
    print("KBest (fixed) :", K_BEST)
    print("CV (fixed)    :", CV_FOLDS)
    print("Grid CV fixed :", GRID_CV_FOLDS)
    print("imblearn      :", "OK" if IMBLEARN_OK else "NOT INSTALLED (training will skip SMOTE/Tomek)")
    print("xgboost       :", "OK" if XGB_OK else "NOT INSTALLED")
    print("seaborn       :", "OK")

    train_and_export(
        data_path=data_path,
        target=args.target,
        out_dir=out_dir,
        test_size=args.test_size,
    )

    print("\n Done. Artifacts saved to:", out_dir)


if __name__ == "__main__":
    main()
