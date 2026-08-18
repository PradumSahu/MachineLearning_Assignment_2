from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

import pandas as pd
import joblib
import os

# ----------------------------
# Load Dataset
# ----------------------------

data = load_breast_cancer(as_frame=True)

X = data.data
y = data.target

# ----------------------------
# Train-Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save test data
test_data = X_test.copy()
test_data["target"] = y_test
test_data.to_csv("test_data.csv", index=False)

# ----------------------------
# Create Models Folder
# ----------------------------

os.makedirs("model", exist_ok=True)

# ----------------------------
# Define Models
# ----------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Naive Bayes":
        GaussianNB(),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
}

# ----------------------------
# Train & Evaluate
# ----------------------------

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": round(
            accuracy_score(y_test, y_pred), 4
        ),
        "AUC": round(
            roc_auc_score(y_test, y_prob), 4
        ),
        "Precision": round(
            precision_score(y_test, y_pred), 4
        ),
        "Recall": round(
            recall_score(y_test, y_pred), 4
        ),
        "F1": round(
            f1_score(y_test, y_pred), 4
        ),
        "MCC": round(
            matthews_corrcoef(y_test, y_pred), 4
        )
    })

    file_name = (
        name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    joblib.dump(
        model,
        f"model/{file_name}"
    )

# ----------------------------
# Save Results
# ----------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    "model_results.csv",
    index=False
)

print(results_df)