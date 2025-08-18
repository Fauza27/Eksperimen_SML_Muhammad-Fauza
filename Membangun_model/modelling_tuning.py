# modelling_tuning.py (ADVANCED)
import os
import io
import json
import time
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import dagshub

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

# =========================
# Load data (hasil preprocessing)
# =========================
df = pd.read_csv("Membangun_model/landmine_preprocessing.csv")

TARGET_COL = "Mine type"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# Setup DagsHub + MLflow (ONLINE)
# =========================
dagshub.init(
    repo_owner="Fauza27",
    repo_name="Eksperimen_SML_Muhammad-Fauza",
    mlflow=True
)
mlflow.set_tracking_uri(
    "https://dagshub.com/Fauza27/Eksperimen_SML_Muhammad-Fauza.mlflow"
)
mlflow.set_experiment("Mine Classification with RF")

# =========================
# Hyperparameter Tuning
# =========================
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    refit=True,
    return_train_score=True
)

# =========================
# Training + Manual Logging
# =========================
with mlflow.start_run(run_name="rf_gridsearch_advanced") as run:
    start_time = time.time()
    grid.fit(X_train, y_train)
    train_time = time.time() - start_time

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # ===== Metrics (manual) =====
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)               # ekstra
    test_size_ratio = len(X_test) / len(X)                # ekstra

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_macro", f1)
    mlflow.log_metric("precision_macro", precision)
    mlflow.log_metric("recall_macro", recall)
    mlflow.log_metric("matthews_corrcoef", mcc)           # ekstra
    mlflow.log_metric("train_time_sec", train_time)       # ekstra
    mlflow.log_metric("test_size_ratio", test_size_ratio) # ekstra
    mlflow.set_tags({
        "stage": "advanced",
        "framework": "scikit-learn",
        "task": "classification",
        "data_rows": len(df),
        "data_features": X.shape[1]
    })

    # ===== Artifacts (manual, ≥2 di luar cakupan autolog) =====
    # 1) Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax_cm, values_format="d", colorbar=False)
    plt.tight_layout()
    mlflow.log_figure(fig_cm, "confusion_matrix.png")
    plt.close(fig_cm)

    # 2) Feature Importances plot
    importances = getattr(best_model, "feature_importances_", None)
    if importances is not None:
        fi = pd.Series(importances, index=X.columns).sort_values(ascending=True)
        fig_fi, ax_fi = plt.subplots(figsize=(7, max(4, len(fi) * 0.25)))
        fi.plot(kind="barh", ax=ax_fi)
        ax_fi.set_title("RandomForest Feature Importances")
        ax_fi.set_xlabel("Importance")
        plt.tight_layout()
        mlflow.log_figure(fig_fi, "feature_importances.png")
        plt.close(fig_fi)

        # Simpan tabel importances juga (opsional)
        fi.to_csv("feature_importances.csv")
        mlflow.log_artifact("feature_importances.csv")
        os.remove("feature_importances.csv")

    # 3) Semua hasil GridSearch (cv_results_) ke CSV
    cv_df = pd.DataFrame(grid.cv_results_)
    cv_path = "grid_search_results.csv"
    cv_df.to_csv(cv_path, index=False)
    mlflow.log_artifact(cv_path)
    os.remove(cv_path)

    # 4) Classification report (txt)
    cls_report = classification_report(y_test, y_pred, zero_division=0)
    mlflow.log_text(cls_report, "classification_report.txt")

    # 5) (opsional) Ringkasan run ke JSON
    run_summary = {
        "best_params": grid.best_params_,
        "metrics": {
            "accuracy": acc,
            "f1_macro": f1,
            "precision_macro": precision,
            "recall_macro": recall,
            "matthews_corrcoef": mcc,
            "train_time_sec": train_time,
            "test_size_ratio": test_size_ratio
        },
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "labels": [str(v) for v in sorted(pd.Series(y).unique())]
    }
    mlflow.log_dict(run_summary, "run_summary.json")

    # ===== Log model (manual) + signature =====
    signature = infer_signature(X_test, best_model.predict(X_test))
    input_example = X_test.iloc[:5]
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="random_forest_model",
        signature=signature,
        input_example=input_example
    )

print("Best params:", grid.best_params_)
print("Accuracy:", acc)
