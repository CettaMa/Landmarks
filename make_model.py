import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

# Load your dataset
# Replace 'your_dataset.csv' with the path to your dataset
data = pd.read_csv("output/combined_output.csv")  # Replace with your dataset
X = data.drop(["State","Time","Video"], axis=1)  # Replace 'label' with your target column
# Encode the target column
y = data["State"]  # Replace 'State' with your target column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),
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

    # Create a directory for confusion matrices if it doesn't exist
    os.makedirs("confusion_matrices", exist_ok=True)

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(y.unique()), 
                yticklabels=sorted(y.unique()))
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f"confusion_matrices/{name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()
    # Save the model
    os.makedirs("saved_models", exist_ok=True)
    model_filename = f"saved_models/{name.replace(' ', '_')}.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"  Model saved to {model_filename}")

# Compare model performance
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)

# Export comparison results to a CSV file
results_df.to_csv("model_comparison.csv", index=False)
