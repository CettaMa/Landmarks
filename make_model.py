import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # Import additional metrics
import time  # Import time module for measuring inference time

# Traditional ML Models
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, SimpleRNN
from tensorflow.keras.utils import to_categorical

# Load dataset
data = pd.read_csv("output/csv/combined_output.csv")

# Separate features and target
X = data.drop(columns=["State", "Time"])
y = data["State"]

# Preprocessing
le = LabelEncoder()
y = le.fit_transform(y)
X = StandardScaler().fit_transform(X)  # Convert to numpy array

# Train-test split (now working with numpy arrays)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Proper reshaping for neural networks
timesteps = 1  # Adjust based on your time series requirements
n_features = X_train.shape[1]

# Convert to 3D format for LSTM/CNN
X_train_reshaped = X_train.reshape((X_train.shape[0], timesteps, n_features))
X_test_reshaped = X_test.reshape((X_test.shape[0], timesteps, n_features))

# For categorical crossentropy
num_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_test_cat = to_categorical(y_test, num_classes=num_classes)

# Dictionary to store results
results = {}
# Dictionary to store inference times
inference_times = {}

def evaluate_model(model, model_name, X_test, y_test, is_nn=False):
    # Warm-up run to initialize model
    if is_nn:
        model.predict(X_test, verbose=0)
    else:
        model.predict(X_test)
    
    # Time measurement with multiple repetitions
    num_repeats = 100  # Increase for faster models, decrease for slower ones
    start_time = time.perf_counter()  # High-resolution timer
    
    for _ in range(num_repeats):
        if is_nn:
            y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
        else:
            y_pred = model.predict(X_test)
    
    end_time = time.perf_counter()
    avg_inference_time = (end_time - start_time) / num_repeats
    
    # Get final predictions for metrics
    if is_nn:
        y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    else:
        y_pred = model.predict(X_test)
    
    y_true = y_test
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    results[model_name] = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Inference Time": avg_inference_time
    }
    
    print(f"{model_name} Accuracy: {acc:.4f}")
    print(f"{model_name} Precision: {precision:.4f}")
    print(f"{model_name} Recall: {recall:.4f}")
    print(f"{model_name} F1-Score: {f1:.4f}")
    print(f"{model_name} Avg Inference Time: {avg_inference_time:.6f} seconds")
    print("-" * 50)
 

# Traditional ML Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    evaluate_model(model, name, X_test, y_test)

# Neural Networks
nn_models = {
    "CNN": Sequential([
        Conv1D(64, 1, activation='relu', input_shape=(timesteps, n_features)),
        MaxPooling1D(1),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(num_classes, activation='softmax')
    ]),
    "LSTM": Sequential([
        LSTM(50, activation='relu', input_shape=(timesteps, n_features)),
        Dense(num_classes, activation='softmax')
    ]),
    "RNN": Sequential([
        SimpleRNN(50, activation='relu', input_shape=(timesteps, n_features)),
        Dense(num_classes, activation='softmax')
    ])
}

for name, model in nn_models.items():
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_reshaped, y_train_cat, epochs=50, verbose=0)
    evaluate_model(model, name, X_test_reshaped, y_test, is_nn=True)

# Display all metrics
print("\nModel Comparison (Detailed):")
print(f"{'Model':20} {'Accuracy':10} {'Precision':10} {'Recall':10} {'F1-Score':10} {'Inf.Time (s)':12}")

for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['Accuracy'], reverse=True):
    print(f"{model_name:20} {metrics['Accuracy']:.4f}     {metrics['Precision']:.4f}     {metrics['Recall']:.4f}     {metrics['F1-Score']:.4f}     {metrics['Inference Time']:.6f}")
