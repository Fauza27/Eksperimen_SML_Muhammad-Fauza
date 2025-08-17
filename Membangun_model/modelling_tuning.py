import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
import time

# ========================================
# Load data preprocessing
# ========================================
df = pd.read_csv("Membangun_model/landmine_preprocessing.csv")

X = df.drop("Mine type", axis=1)
y = df["Mine type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================================
# Setup MLflow lokal
# ========================================
import dagshub
dagshub.init(repo_owner='Fauza27', repo_name='Eksperimen_SML_Muhammad-Fauza', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Fauza27/Eksperimen_SML_Muhammad-Fauza.mlflow")
mlflow.set_experiment("Mine Classification with RF")

# ========================================
# Hyperparameter Tuning
# ========================================
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

# ========================================
# Training + Logging Manual
# ========================================
with mlflow.start_run():
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    train_time = time.time() - start_time

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    mcc = matthews_corrcoef(y_test, y_pred)   # metric tambahan
    test_size_ratio = len(X_test) / len(X)    # metric tambahan

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_macro", f1)
    mlflow.log_metric("precision_macro", precision)
    mlflow.log_metric("recall_macro", recall)
    mlflow.log_metric("matthews_corrcoef", mcc)
    mlflow.log_metric("train_time_sec", train_time)
    mlflow.log_metric("test_size_ratio", test_size_ratio)

    mlflow.sklearn.log_model(best_model, "random_forest_model")

print("Best params:", grid_search.best_params_)
print("Accuracy:", acc)
