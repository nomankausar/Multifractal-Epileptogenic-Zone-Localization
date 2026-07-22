import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, kurtosis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek

# ================================
#  Load and Prepare Dataset
# ================================
file_path = r"F:\Research After Conference\task 78 ml on adult 5min  cleaned\merged v2.xlsx"
df = pd.read_excel(file_path)
df.drop(columns=["Subject_ID", "Channel"], inplace=True)

X = df.drop(columns=["is_soz"])
y = df["is_soz"]

# ================================
#  Imputation + Feature Engineering
# ================================
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_df = X.copy()
# X_df["Entropy"] = np.apply_along_axis(lambda x: entropy(np.abs(x) + 1e-10), 1, X)
# X_df["Kurtosis"] = np.apply_along_axis(kurtosis, 1, X)

# ================================
#  Scaling, Feature Selection, Resampling
# ================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_df)
selector = SelectKBest(mutual_info_classif, k=4)
X_selected = selector.fit_transform(X_scaled, y)
X_resampled, y_resampled = SMOTETomek(random_state=42).fit_resample(X_selected, y)

# ================================
#  XGBoost with GridSearchCV (GPU)
# ================================
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.01, 0.05],
    "max_depth": [2, 3],
    "subsample": [0.7, 0.85],
    "colsample_bytree": [0.7, 0.85],
    "reg_alpha": [0.1, 0.3, 1.0],
    "reg_lambda": [1.0, 2.0, 5.0],
    "gamma": [0.1, 0.3, 0.5]
}

xgb_base = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    verbosity=0,
    tree_method='hist', device='cuda'
)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='f1',
    cv=kf,
    n_jobs=-1,
    verbose=0
)
grid.fit(X_resampled, y_resampled)
best_model = grid.best_estimator_

# ================================
# 10-Fold CV Evaluation + Overfit Plot
# ================================
metrics = {m: [] for m in ["Accuracy", "Precision", "Recall", "F1", "AUC"]}
conf_matrix_total = np.zeros((2, 2), dtype=int)
mean_fpr = np.linspace(0, 1, 200)
# Store interpolated TPRs so we can compute mean ± std ROC
tprs = []
roc_aucs = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_resampled, y_resampled)):
    X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
    y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fold_auc = auc(fpr, tpr)
    roc_aucs.append(fold_auc)
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    tprs.append(tpr_interp)
    if fold == 0:
        y_train_prob = best_model.predict_proba(X_train)[:, 1]
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
        auc_train = auc(fpr_train, tpr_train)
        auc_test = fold_auc

        plt.figure(figsize=(8, 6))
        plt.plot(fpr_train, tpr_train, label=f"Train ROC (AUC = {auc_train:.2f})", color="blue")
        plt.plot(fpr, tpr, label=f"Test ROC (AUC = {auc_test:.2f})", color="orange")
        plt.plot([0, 1], [0, 1], 'k--', label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Train vs Test ROC - XGBoost (GPU) - Fold 0")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("results_xgb_gpu_final", exist_ok=True)
        plt.savefig("results_xgb_gpu_final/train_vs_test_roc_xgb_gpu.png", dpi=300)
        plt.show()

    conf_matrix_total += confusion_matrix(y_test, y_pred)
    metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
    metrics["Precision"].append(precision_score(y_test, y_pred))
    metrics["Recall"].append(recall_score(y_test, y_pred))
    metrics["F1"].append(f1_score(y_test, y_pred))
    metrics["AUC"].append(fold_auc)

# ================================
# Save Outputs
# ================================
output_dir = "results_xgb_gpu_final"
os.makedirs(output_dir, exist_ok=True)
pd.DataFrame(metrics).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
pd.DataFrame([grid.best_params_]).to_csv(os.path.join(output_dir, "best_params.csv"), index=False)

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_total, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=["Non-SOZ", "SOZ"], yticklabels=["Non-SOZ", "SOZ"])
plt.title("Confusion Matrix - XGBoost (GPU)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_xgb_gpu.png"), dpi=300)
plt.show()

# ROC Curve (Mean ± Std across folds)
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', lw=1)

tprs_arr = np.array(tprs)
mean_tpr = tprs_arr.mean(axis=0)
std_tpr  = tprs_arr.std(axis=0)
mean_tpr[-1] = 1.0

# Optional: show each fold as a faint curve
for tpr_i in tprs_arr:
    plt.plot(mean_fpr, tpr_i, alpha=0.15, lw=1)

# Mean ROC + shaded band
mean_auc = float(np.mean(roc_aucs))
std_auc  = float(np.std(roc_aucs))
plt.plot(mean_fpr, mean_tpr, color='green', lw=2,
         label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})")

upper = np.minimum(mean_tpr + std_tpr, 1)
lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, lower, upper, alpha=0.20)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost (GPU) - 10-Fold")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve_xgb_gpu.png"), dpi=300)
plt.show()

# Boxplot of Metrics
plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame(metrics))
plt.title("XGBoost (GPU) Metrics - 10-Fold")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "boxplot_xgb_gpu.png"), dpi=300)
plt.show()

# ================================
#  Learning Curve (Improved)
# ================================
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', label="Train AUC", color='blue')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', label="Validation AUC", color='orange')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='orange')
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("AUC Score")
    plt.xticks(train_sizes, [f"{int(s / len(X) * 100)}%" for s in train_sizes])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png"), dpi=300)
    plt.show()

plot_learning_curve(best_model, X_resampled, y_resampled, "XGBoost GPU Learning Curve")


# ================================
#  Export Model Bundle (Software)
# ================================
# This saves EVERYTHING needed for your software inference:
# - imputer, scaler, selector, trained XGBoost model, and feature columns.
# Your GUI should load this .joblib file and run predict_bundle(...).
import joblib

# Fit best_model on ALL resampled data for final deployment
best_model.fit(X_resampled, y_resampled)

bundle = {
    "imputer": imputer,
    "scaler": scaler,
    "selector": selector,
    "model": best_model,
    "feature_columns_before_engineering": list(X.columns),
    "feature_columns_after_engineering": list(X_df.columns),  # includes Entropy, Kurtosis
    "threshold": 0.5,
    "notes": {
        "resampling": "SMOTETomek (training only)",
        "feature_engineering": "Entropy, Kurtosis computed per-row from absolute features",
    }
}

joblib.dump(bundle, os.path.join(output_dir, "xgb_software_bundle.joblib"), compress=3)

# Also save native xgboost model (optional)
try:
    best_model.save_model(os.path.join(output_dir, "xgb_model.json"))
except Exception:
    pass

# Save a small meta file (human readable)
meta = {
    "best_params": grid.best_params_,
    "n_rows_original": int(df.shape[0]),
    "n_rows_resampled": int(len(y_resampled)),
    "n_features_after_engineering": int(X_df.shape[1]),
    "n_features_selected": int(X_resampled.shape[1]),
}
with open(os.path.join(output_dir, "bundle_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

def predict_bundle(input_csv_or_xlsx, bundle_path=os.path.join(output_dir, "xgb_software_bundle.joblib"),
                   output_csv=None):
    """
    Prediction helper for your software.

    Input: CSV or XLSX with the SAME raw feature columns used in training.
           It MAY include Subject_ID and Channel; those will be ignored.
           It should NOT require is_soz.

    Output: CSV with prob_soz and pred_soz. If Subject_ID/Channel exist, they are kept in output.
    """
    b = joblib.load(bundle_path)
    imputer_ = b["imputer"]
    scaler_ = b["scaler"]
    selector_ = b["selector"]
    model_ = b["model"]
    thr = float(b.get("threshold", 0.5))

    # Read
    p = str(input_csv_or_xlsx).lower()
    if p.endswith(".xlsx") or p.endswith(".xls"):
        df_in = pd.read_excel(input_csv_or_xlsx)
    else:
        df_in = pd.read_csv(input_csv_or_xlsx)

    # Keep IDs if present
    keep_cols = []
    for c in ["Subject_ID", "Channel"]:
        if c in df_in.columns:
            keep_cols.append(c)
    ids_df = df_in[keep_cols].copy() if keep_cols else None

    # Drop non-features
    df2 = df_in.copy()
    for c in ["Subject_ID", "Channel", "is_soz"]:
        if c in df2.columns:
            df2.drop(columns=[c], inplace=True)

    # Apply same preprocessing + engineering
    X0 = pd.DataFrame(imputer_.transform(df2), columns=df2.columns)
    X_df0 = X0.copy()
    X_df0["Entropy"]  = np.apply_along_axis(lambda x: entropy(np.abs(x) + 1e-10), 1, X0.values)
    X_df0["Kurtosis"] = np.apply_along_axis(kurtosis, 1, X0.values)

    X_scaled0 = scaler_.transform(X_df0)
    X_sel0 = selector_.transform(X_scaled0)

    prob = model_.predict_proba(X_sel0)[:, 1]
    pred = (prob >= thr).astype(int)

    out = pd.DataFrame({"prob_soz": prob, "pred_soz": pred})
    if ids_df is not None:
        out = pd.concat([ids_df.reset_index(drop=True), out], axis=1)

    if output_csv is None:
        base = os.path.splitext(str(input_csv_or_xlsx))[0]
        output_csv = base + "_xgb_predictions.csv"
    out.to_csv(output_csv, index=False)
    return output_csv

print("\n Saved software bundle:", os.path.join(output_dir, "xgb_software_bundle.joblib"))
print(" (Optional) Saved native model:", os.path.join(output_dir, "xgb_model.json"))


# ================================
#  Print Summary
# ================================
print("\n Best Parameters:")
print(grid.best_params_)

print("\n 10-Fold Average Metrics:")
for m in metrics:
    print(f"{m}: {np.mean(metrics[m]):.4f} ± {np.std(metrics[m]):.4f}")
