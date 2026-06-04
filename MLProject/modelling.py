import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Set MLflow Tracking URI Github Actions
mlflow.set_tracking_uri("file:./mlruns")

# Nama experiment
mlflow.set_experiment("Telco Customer Churn Basic Model")

# Mengaktifkan autolog sesuai kriteria basic
mlflow.sklearn.autolog()

# Load dataset hasil preprocessing
data_path = "telco_preprocessing/telco_customer_churn_preprocessed.csv"
df = pd.read_csv(data_path)

# Memisahkan fitur dan target
X = df.drop(columns=["Churn"])
y = df["Churn"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Membuat dan melatih model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

with mlflow.start_run(run_name="RandomForest_Basic_Autolog"):
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi tambahan
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Training selesai.")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")