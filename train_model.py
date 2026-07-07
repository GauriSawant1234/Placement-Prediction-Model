import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib

# ==========================================
# DAY 1 - LOAD & UNDERSTAND DATA
# ==========================================

df = pd.read_csv("data/Placement.csv")

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATASET INFO")
print(df.info())

print("\nSTATISTICAL SUMMARY")
print(df.describe())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATES")
print(df.duplicated().sum())

print("\nCOLUMN TYPES")
print(df.dtypes)

# ==========================================
# DAY 2 - VISUALIZATION
# ==========================================

# Placement Status
df["status"].value_counts().plot(kind="bar")
plt.title("Placement Status")
plt.xlabel("Status")
plt.ylabel("Count")
plt.show()

# Gender Distribution
df["gender"].value_counts().plot(kind="bar")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# SSC Distribution
plt.hist(df["ssc_p"], bins=10)
plt.title("SSC Percentage Distribution")
plt.xlabel("SSC Percentage")
plt.ylabel("Students")
plt.show()

# HSC Distribution
plt.hist(df["hsc_p"], bins=10)
plt.title("HSC Percentage Distribution")
plt.xlabel("HSC Percentage")
plt.ylabel("Students")
plt.show()

# Degree Distribution
plt.hist(df["degree_p"], bins=10)
plt.title("Degree Percentage Distribution")
plt.xlabel("Degree Percentage")
plt.ylabel("Students")
plt.show()

# Placement vs SSC
placed = df[df["status"] == "Placed"]
not_placed = df[df["status"] == "Not Placed"]

plt.hist(placed["ssc_p"], alpha=0.5, label="Placed")
plt.hist(not_placed["ssc_p"], alpha=0.5, label="Not Placed")
plt.legend()
plt.title("SSC Percentage vs Placement")
plt.show()

# Placement vs Degree
plt.hist(placed["degree_p"], alpha=0.5, label="Placed")
plt.hist(not_placed["degree_p"], alpha=0.5, label="Not Placed")
plt.legend()
plt.title("Degree Percentage vs Placement")
plt.show()

numeric_df = df.select_dtypes(include=["number"])

print("\nCORRELATION MATRIX")
print(numeric_df.corr())

# ==========================================
# DAY 3 - DATA PREPROCESSING
# ==========================================

# Remove serial number
df = df.drop("sl_no", axis=1)

# Missing values
print("\nMISSING VALUES BEFORE")
print(df.isnull().sum())

# Fill salary nulls
df["salary"] = df["salary"].fillna(0)

print("\nMISSING VALUES AFTER")
print(df.isnull().sum())

# Encode categorical data
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

categorical_cols = [
    "gender",
    "ssc_b",
    "hsc_b",
    "hsc_s",
    "degree_t",
    "workex",
    "specialisation",
    "status"
]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop("status", axis=1)
y = df["status"]

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTRAIN SHAPE:", X_train.shape)
print("TEST SHAPE:", X_test.shape)

# Save processed data
df.to_csv("placement_prepared.csv", index=False)

# ==========================================
# DAY 4 - MODEL TRAINING
# ==========================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

results = []

for name, model in models.items():

    start = time.time()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    end = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1,
        end - start
    ])

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

# Comparison Table
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Training Time"
    ]
)

print("\nMODEL COMPARISON")
print(results_df)

# Accuracy Plot
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.show()

# Best Model
best_model = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\nBEST MODEL")
print(best_model.head(1))

# Random Forest Analysis
rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_rf)

print("\nCONFUSION MATRIX")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["Not Placed", "Placed"],
    yticklabels=["Not Placed", "Placed"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(feature_importance)

plt.figure(figsize=(10, 6))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xticks(rotation=90)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# Save Model
joblib.dump(rf_model, "placement_model.pkl")

print("\nPROJECT COMPLETE")
print("Generated Files:")
print("1. placement_prepared.csv")
print("2. placement_model.pkl")