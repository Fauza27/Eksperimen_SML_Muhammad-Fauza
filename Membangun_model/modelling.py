# modelling.py untuk basic
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
mlflow.set_tracking_uri("file:./mlruns") 
mlflow.set_experiment("Mine Classification RF (local)")

# Aktifkan autolog
mlflow.sklearn.autolog()

# ========================================
# Training + Logging otomatis
# ========================================
with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
    model.fit(X_train, y_train)
