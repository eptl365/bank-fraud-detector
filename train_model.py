# ============================================================
# BANK FRAUD DETECTOR — Training Script
# Run this in Google Colab, cell by cell, or all at once
# ============================================================

# ── CELL 1: Imports ──────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, ConfusionMatrixDisplay
)
from sklearn.utils import resample
import joblib, json, warnings
warnings.filterwarnings('ignore')
print("✅ Libraries loaded")


# ── CELL 2: Upload & Load Data ───────────────────────────────
from google.colab import files
uploaded = files.upload()          # <-- click and select creditcard_2023.csv

df = pd.read_csv("creditcard_2023.csv")
print(f"✅ Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(df.head())


# ── CELL 3: Explore ──────────────────────────────────────────
print("=== Missing values ===")
print(df.isnull().sum().sum(), "missing")

print("\n=== Class distribution ===")
print(df["Class"].value_counts())
print(f"Fraud rate: {df['Class'].mean()*100:.4f}%")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["Class"].value_counts().plot(kind="bar", ax=axes[0], color=["#2196F3","#F44336"])
axes[0].set_xticklabels(["Legitimate","Fraud"], rotation=0)
axes[0].set_title("Class Distribution")
df[df["Class"]==0]["Amount"].hist(bins=50, ax=axes[1], alpha=0.6, label="Legit")
df[df["Class"]==1]["Amount"].hist(bins=50, ax=axes[1], alpha=0.6, label="Fraud", color="red")
axes[1].set_title("Amount by Class")
axes[1].legend()
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
plt.show()
print("✅ Plot saved")


# ── CELL 4: Clean ────────────────────────────────────────────
# Drop id column
if "id" in df.columns:
    df = df.drop(columns=["id"])
    print("Dropped 'id' column")

# Remove duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicates")

print(f"✅ Clean shape: {df.shape}")


# ── CELL 5: Features & Scaling ───────────────────────────────
X = df.drop(columns=["Class"]).copy()
y = df["Class"]

feature_names = X.columns.tolist()
print(f"Features: {feature_names}")

# Scale Amount
scaler = StandardScaler()
X["Amount"] = scaler.fit_transform(X[["Amount"]])
print("✅ Amount scaled")


# ── CELL 6: Balance Classes ──────────────────────────────────
X_combined = X.copy()
X_combined["Class"] = y.values

fraud = X_combined[X_combined["Class"] == 1]
legit = X_combined[X_combined["Class"] == 0]

n_fraud = len(fraud)
legit_down = resample(legit, replace=False,
                      n_samples=min(n_fraud * 5, len(legit)),
                      random_state=42)

balanced = pd.concat([fraud, legit_down]).sample(frac=1, random_state=42)
X_bal = balanced.drop(columns=["Class"])
y_bal = balanced["Class"]

print(f"✅ Balanced: {y_bal.value_counts().to_dict()}")
print(f"Total training samples: {len(balanced):,}")


# ── CELL 7: Train/Test Split ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42, stratify=y_bal
)
print(f"✅ Train: {len(X_train):,}  |  Test: {len(X_test):,}")


# ── CELL 8: Train Model ──────────────────────────────────────
print("⏳ Training Random Forest... (takes 1-3 minutes)")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("✅ Training complete!")


# ── CELL 9: Evaluate ─────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Legitimate","Fraud"]))

roc_auc  = roc_auc_score(y_test, y_prob)
avg_prec = average_precision_score(y_test, y_prob)
print(f"ROC-AUC:           {roc_auc:.4f}")
print(f"Average Precision: {avg_prec:.4f}")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Legit","Fraud"]).plot(ax=axes[0], cmap="Blues")
axes[0].set_title("Confusion Matrix")

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color="#F44336", lw=2, label=f"AUC={roc_auc:.3f}")
axes[1].plot([0,1],[0,1],"k--", alpha=0.4)
axes[1].set_title("ROC Curve"); axes[1].legend()

prec, rec, _ = precision_recall_curve(y_test, y_prob)
axes[2].plot(rec, prec, color="#2196F3", lw=2, label=f"AP={avg_prec:.3f}")
axes[2].set_title("Precision-Recall Curve"); axes[2].legend()

plt.tight_layout()
plt.savefig("evaluation_plots.png", dpi=150)
plt.show()
print("✅ Eval plots saved")


# ── CELL 10: Feature Importance ──────────────────────────────
importances = pd.Series(model.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(12, 5))
importances.head(15).plot(kind="bar", color="#2196F3")
plt.title("Top 15 Most Important Features")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("Top 5 features:")
print(importances.head())


# ── CELL 11: Save Files ──────────────────────────────────────
joblib.dump(model, "fraud_model.joblib")
joblib.dump(scaler, "scaler.joblib")
with open("feature_names.json", "w") as f:
    json.dump(feature_names, f)

print("✅ Saved: fraud_model.joblib, scaler.joblib, feature_names.json")


# ── CELL 12: Download Files ──────────────────────────────────
from google.colab import files
files.download("fraud_model.joblib")
files.download("scaler.joblib")
files.download("feature_names.json")
files.download("class_distribution.png")
files.download("evaluation_plots.png")
files.download("feature_importance.png")
print("✅ Downloads started — check your Downloads folder!")


# ── CELL 13: Sanity Check ────────────────────────────────────
sample      = X_test.head(5)
true_labels = y_test.head(5).values
preds       = model.predict(sample)
probs       = model.predict_proba(sample)[:, 1]

print("\n=== Sample Predictions ===")
for i in range(5):
    verdict = "FRAUD" if preds[i] == 1 else "Legit"
    actual  = "FRAUD" if true_labels[i] == 1 else "Legit"
    print(f"Row {i+1}: Predicted={verdict} ({probs[i]*100:.1f}% fraud)  |  Actual={actual}")
