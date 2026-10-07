"""
churn_model.py - Customer churn prediction pipeline (same method as churn_pipeline.ipynb)
Dataset: IBM Telco Customer Churn sample (Kaggle), 7,043 customers, US dollar charges.
Author: Mayank Joshi

Run:          python churn_model.py
Export data:  python churn_model.py --export   (writes model_results.json used by index.html)

The scaler is fitted on the training split only. Accuracy should be read against the
"always predict no churn" baseline (about 73%), so AUC is the more useful metric here.
"""

import json
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_STATE = 42


def load_and_engineer(filepath: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """Load the data, fill the 11 blank TotalCharges values, add engineered features."""
    df = pd.read_csv(filepath)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["churn_binary"] = (df["Churn"] == "Yes").astype(int)

    df["charge_per_month_ratio"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    df["high_monthly_charge"] = (df["MonthlyCharges"] > 70).astype(int)

    cat_cols = [c for c in df.columns
                if not pd.api.types.is_numeric_dtype(df[c]) and c not in ("customerID", "Churn")]
    for col in cat_cols:
        df[col + "_enc"] = LabelEncoder().fit_transform(df[col].astype(str))

    feature_cols = (["tenure", "MonthlyCharges", "TotalCharges", "charge_per_month_ratio",
                     "is_new_customer", "high_monthly_charge"] + [c + "_enc" for c in cat_cols])
    return df, df[feature_cols], df["churn_binary"]


def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_te, proba)
    return {
        "name": name,
        "accuracy": accuracy_score(y_te, preds),
        "auc_roc": roc_auc_score(y_te, proba),
        "precision_churn": precision_score(y_te, preds),
        "recall_churn": recall_score(y_te, preds),
        "confusion_matrix": confusion_matrix(y_te, preds).tolist(),
        "roc_fpr": [round(float(v), 4) for v in fpr],
        "roc_tpr": [round(float(v), 4) for v in tpr],
    }


def main(export: bool = False):
    df, X, y = load_and_engineer()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    results = [evaluate("Logistic Regression", lr, X_tr, X_te, y_tr, y_te),
               evaluate("Random Forest", rf, X_tr, X_te, y_tr, y_te)]
    for res, model in zip(results, (lr, rf)):
        cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring="roc_auc")
        res["cv_auc_mean"], res["cv_auc_std"] = float(cv.mean()), float(cv.std())

    baseline = 1 - y_te.mean()
    print(f"Rows: {len(df):,}  test rows: {len(y_te):,}  baseline accuracy (always 'no churn'): {baseline:.2%}\n")
    for r in results:
        print(f"{r['name']:20} accuracy {r['accuracy']:.2%} | AUC {r['auc_roc']:.3f} | "
              f"5-fold CV AUC {r['cv_auc_mean']:.3f} ± {r['cv_auc_std']:.3f} | "
              f"precision {r['precision_churn']:.2f} | recall {r['recall_churn']:.2f}")

    imp = (pd.Series(rf.feature_importances_, index=X.columns)
             .rename(lambda c: c.replace("_enc", "")).sort_values(ascending=False))
    print("\nTop 10 Random Forest feature importances:")
    print(imp.head(10).round(3).to_string())

    if export:
        out = {"rows": len(df), "test_rows": int(len(y_te)), "baseline_accuracy": float(baseline),
               "models": results,
               "rf_feature_importance": {k: round(float(v), 4) for k, v in imp.head(10).items()}}
        with open("model_results.json", "w") as f:
            json.dump(out, f, indent=1)
        print("\nWrote model_results.json")


if __name__ == "__main__":
    main(export="--export" in sys.argv)
