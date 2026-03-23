"""
churn_model.py — Modular churn prediction pipeline
Telco Customer Churn dataset (7,043 records)
Author: Mayank Joshi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)


# ── Data Loading ──────────────────────────────────────────────

def load_data(filepath: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
      """Load the Telco churn dataset and perform initial cleaning."""
      df = pd.read_csv(filepath)
      df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
      df.dropna(subset=["TotalCharges"], inplace=True)
      df.drop(columns=["customerID"], inplace=True)
      return df


# ── Feature Engineering ───────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
      """Encode categorical variables and scale numeric features."""
      target = (df["Churn"] == "Yes").astype(int).values
      features = df.drop(columns=["Churn"])

    # Label-encode binary columns
      binary_cols = features.select_dtypes(include="object").columns
      le = LabelEncoder()
      for col in binary_cols:
                features[col] = le.fit_transform(features[col])

      # Scale numeric columns
      scaler = StandardScaler()
      numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
      features[numeric_cols] = scaler.fit_transform(features[numeric_cols])

    return features, target


# ── Model Training ────────────────────────────────────────────

def train_models(X_train, X_test, y_train, y_test) -> dict:
      """Train logistic regression and random forest; return results dict."""
      results = {}

    # Logistic Regression
      lr = LogisticRegression(max_iter=1000, random_state=42)
      lr.fit(X_train, y_train)
      lr_preds = lr.predict(X_test)
      lr_proba = lr.predict_proba(X_test)[:, 1]
      results["logistic_regression"] = {
          "model": lr,
          "accuracy": accuracy_score(y_test, lr_preds),
          "auc_roc": roc_auc_score(y_test, lr_proba),
          "report": classification_report(y_test, lr_preds),
      }

    # Random Forest
      rf = RandomForestClassifier(n_estimators=200, random_state=42)
      rf.fit(X_train, y_train)
      rf_preds = rf.predict(X_test)
      rf_proba = rf.predict_proba(X_test)[:, 1]
      results["random_forest"] = {
          "model": rf,
          "accuracy": accuracy_score(y_test, rf_preds),
          "auc_roc": roc_auc_score(y_test, rf_proba),
          "report": classification_report(y_test, rf_preds),
      }

    return results


# ── Main Pipeline ─────────────────────────────────────────────

def main():
      print("Loading data...")
      df = load_data()

    print("Engineering features...")
    X, y = engineer_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
              X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training models...")
    results = train_models(X_train, X_test, y_train, y_test)

    for name, res in results.items():
              print(f"\n{'='*50}")
              print(f"  {name.replace('_', ' ').title()}")
              print(f"{'='*50}")
              print(f"  Accuracy : {res['accuracy']:.2%}")
              print(f"  AUC-ROC  : {res['auc_roc']:.4f}")
              print(f"\n{res['report']}")


if __name__ == "__main__":
      main()
