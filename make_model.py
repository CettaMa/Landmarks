import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset
data = pd.read_csv("output/2025-05-09 08-50-57/2025-05-09 08-50-57.csv")  # Replace with your dataset
X = data.drop(["State","Time"], axis=1)  # Replace 'label' with your target column
# Encode the target column
y = data["State"]  # Replace 'State' with your target column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="linear", probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
}

# Train and evaluate models
results = []
for name, model in models.items():
    print(f"Training {name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Inference time
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")

    print(f"{name} Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Training Time: {training_time:.4f} seconds")
    print(f"  Inference Time: {inference_time:.6f} seconds per sample")
    print(classification_report(y_test, y_pred))

    # Save results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Recall": recall,
        "Precision": precision,
        "Training Time (s)": training_time,
        "Inference Time (s/sample)": inference_time,
    })

    # Export the model to a .pkl file
    model_filename = f"exported_model/{name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, model_filename)
    print(f"{name} model saved as {model_filename}")

# Compare model performance
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)

# Export comparison results to a CSV file
results_df.to_csv("model_comparison.csv", index=False)
# Create the folder if it doesn't exist
export_folder = "exported_model"
os.makedirs(export_folder, exist_ok=True)

# Save models