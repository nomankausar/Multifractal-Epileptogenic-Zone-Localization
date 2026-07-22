import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.stats import entropy, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
from imblearn.combine import SMOTETomek


# ========== Load Dataset ==========
file_path = r"F:\Research After Conference\task 78 ml on adult 5min  cleaned\merged v2.xlsx"
df = pd.read_excel(file_path)
df.drop(columns=["Subject_ID", "Channel"], inplace=True)
X = df.drop(columns=["is_soz"])
y = df["is_soz"]

X_df = X.copy()

# ========== Scale + Feature Selection + Resample ==========
X_scaled = StandardScaler().fit_transform(X_df)
X_selected = SelectKBest(score_func=mutual_info_classif, k=4).fit_transform(X_scaled, y)
X_res, y_res = SMOTETomek(random_state=42).fit_resample(X_selected, y)

# ========== Models ==========
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=4,
        max_leaf_nodes=100,
        max_features='sqrt',
        class_weight='balanced',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    "Logistic Regression": LogisticRegression(max_iter=1000000, solver='lbfgs', random_state=42)
}

# ========== Output Folder ==========
output_dir = "results_rf_logistic"
os.makedirs(output_dir, exist_ok=True)

# ========== Overfit AUC Plot Function ==========
def plot_mean_overfit_roc(train_aucs, test_aucs, model_name, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(train_aucs, label='Train AUC', marker='o', color='blue')
    plt.plot(test_aucs, label='Test AUC', marker='o', color='orange')
    plt.fill_between(range(len(train_aucs)), train_aucs, test_aucs, color='red', alpha=0.1)
    plt.title(f"Mean Overfit ROC Gap - {model_name}")
    plt.xlabel("Fold")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.show()

# ========== Evaluation Loop ==========
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n Evaluating: {name}")
    metrics = {m: [] for m in ["Accuracy", "Precision", "Recall", "F1", "AUC"]}
    conf_matrix_total = np.zeros((2, 2), dtype=int)

    # For ROC mean + shaded std across folds
    mean_fpr = np.linspace(0, 1, 200)
    tprs = []
    roc_aucs = []

    train_auc_all, test_auc_all = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_res, y_res)):
        X_train, X_test = X_res[train_idx], X_res[test_idx]
        y_train, y_test = y_res[train_idx], y_res[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f" Skipping fold {fold} for {name} due to class imbalance.")
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        y_train_prob = model.predict_proba(X_train)[:, 1]
        auc_train = auc(*roc_curve(y_train, y_train_prob)[:2])
        auc_test = auc(*roc_curve(y_test, y_prob)[:2])
        train_auc_all.append(auc_train)
        test_auc_all.append(auc_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        # Collect ROC for mean + shaded band
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        roc_aucs.append(auc_test)

        conf_matrix_total += confusion_matrix(y_test, y_pred)
        metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["Precision"].append(precision_score(y_test, y_pred))
        metrics["Recall"].append(recall_score(y_test, y_pred))
        metrics["F1"].append(f1_score(y_test, y_pred))
        metrics["AUC"].append(auc_test)

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(os.path.join(output_dir, f"metrics_{name.replace(' ', '_')}.csv"), index=False)

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix_total, annot=True, fmt='d', cmap='coolwarm',
                xticklabels=["Non-SOZ", "SOZ"], yticklabels=["Non-SOZ", "SOZ"])
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{name.replace(' ', '_')}.png"), dpi=300)
    plt.show()

    # ROC Curve (mean ROC + shaded std across folds)
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', lw=1)

    if len(tprs) == 0:
        print(f" No valid folds to plot ROC for {name}.")
    else:
        tprs_arr = np.array(tprs)
        mean_tpr = tprs_arr.mean(axis=0)
        std_tpr = tprs_arr.std(axis=0)
        mean_tpr[-1] = 1.0

        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

        # Optional: plot each fold faintly
        for tpr_i in tprs_arr:
            plt.plot(mean_fpr, tpr_i, alpha=0.15, lw=1)

        mean_auc = float(np.mean(roc_aucs))
        std_auc = float(np.std(roc_aucs))

        plt.plot(mean_fpr, mean_tpr, color='green', lw=2,
                 label=f"Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})")
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper, alpha=0.20)
        plt.legend(loc='lower right')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name} (10-fold)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"roc_curve_{name.replace(' ', '_')}.png"), dpi=300)
    plt.show()

    # Boxplot of Metrics
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_metrics)
    plt.title(f"{name} Metrics - 10-Fold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"boxplot_{name.replace(' ', '_')}.png"), dpi=300)
    plt.show()

    # Train vs Test AUC
    plt.figure(figsize=(8, 6))
    plt.plot(train_auc_all, label="Train AUC", marker='o', color='blue')
    plt.plot(test_auc_all, label="Test AUC", marker='o', color='orange')
    plt.fill_between(range(len(train_auc_all)), train_auc_all, test_auc_all, color='red', alpha=0.1)
    plt.title(f"Train vs Test AUC per Fold - {name}")
    plt.xlabel("Fold")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"overfit_auc_gap_{name.replace(' ', '_')}.png"), dpi=300)
    plt.show()

    # Mean Overfit ROC Plot
    plot_mean_overfit_roc(train_auc_all, test_auc_all, name, f"mean_overfit_roc_{name.replace(' ', '_')}.png")

    # Final Results Print
    auc_gap = np.mean(train_auc_all) - np.mean(test_auc_all)
    print(f"\n 10-Fold Average Metrics ({name}):")
    for m in metrics:
        print(f"{m}: {np.mean(metrics[m]):.4f} ± {np.std(metrics[m]):.4f}")
    print(f"\n Overfit check ({name}):")
    print(f"Mean Train AUC: {np.mean(train_auc_all):.4f}")
    print(f"Mean Test  AUC: {np.mean(test_auc_all):.4f}")
    print(f"AUC Gap       : {auc_gap:.4f}")
    if auc_gap > 0.05:
        print(" Likely Overfitting!")
    else:
        print(" No strong overfitting detected.")


# ============================================================
#  EXPORT TRAINED MODEL FILES (FOR SOFTWARE DEPLOYMENT)
#    (Keeps your core training logic the same)
# ============================================================

export_dir = os.path.join(output_dir, "trained_models")
os.makedirs(export_dir, exist_ok=True)

# Re-fit scaler + selector on FULL original data (X_df, y)
scaler = StandardScaler()
X_scaled_full = scaler.fit_transform(X_df)

selector = SelectKBest(score_func=mutual_info_classif, k=4)
X_selected_full = selector.fit_transform(X_scaled_full, y)

# Re-fit SMOTETomek on full selected data (training-only)
resampler = SMOTETomek(random_state=42)
X_res_full, y_res_full = resampler.fit_resample(X_selected_full, y)

# Train final versions of each model on full resampled training set
for name, model in models.items():
    model.fit(X_res_full, y_res_full)

    bundle = {
        "model_name": name,
        "scaler": scaler,
        "selector": selector,
        "kbest_k": 4,
        "feature_names_original": list(X_df.columns),
        "model": model,
        "threshold": 0.5,  # you can tune later (ROC-based threshold)
        "notes": "Inference: apply scaler -> selector -> model.predict_proba"
    }

    safe_name = name.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(export_dir, f"EZ_{safe_name}.joblib")
    joblib.dump(bundle, out_path)

    print(f" Saved trained file: {out_path}")

