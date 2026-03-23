# Customer Churn Prediction Pipeline

**Tools:** Python (pandas, scikit-learn, matplotlib, seaborn) · SQL (sqlite3) · Plotly.js · GitHub Pages  
**Dataset:** Telco Customer Churn — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 records)  
**Model accuracy:** 82% | AUC-ROC: 0.86  
**Live Dashboard:** [🔗 View Live Interactive Dashboard](https://mayankjoshiii.github.io/customer-churn-prediction/)

---

## Problem Statement

A telecom company loses ~27% of its customers annually. Can we predict which customers are about to churn — and translate that prediction into a concrete retention strategy?

---

## Approach

| Step | Description |
|------|-------------|
| 1. SQL Exploration | Segmentation queries via `sqlite3` — tenure, contract type, monthly charges |
| 2. Data Cleaning | Handle missing values, encode categoricals, scale features |
| 3. Feature Engineering | Create tenure bands, charge-per-service ratio, engagement score |
| 4. Modelling | Logistic Regression + Random Forest — comparison with cross-validation |
| 5. Evaluation | Confusion matrix, ROC curve, precision/recall, feature importance |
| 6. Business Output | Translate model to actionable retention recommendation |

---

## Key Results

| Model | Accuracy | AUC-ROC | Precision (Churn) | Recall (Churn) |
|-------|----------|---------|-------------------|----------------|
| Logistic Regression | 80% | 0.84 | 0.67 | 0.58 |
| Random Forest | **82%** | **0.86** | **0.71** | **0.63** |

---

## Business Recommendation

> **Targeting the 3 highest-risk segments (Month-to-month contracts + Fibre Optic + High monthly charges) with a proactive retention offer — a 12-month contract discount — is projected to reduce overall churn by 12–15%, saving approximately £340,000 in annual revenue per 10,000 customers.**

Top 3 churn drivers identified:

1. **Contract type** — Month-to-month customers churn at 3.4× the rate of annual contract holders
2. **Tenure** — Customers in their first 6 months are at highest risk (42% churn rate)
3. **Internet service** — Fibre Optic subscribers churn at 2.1× the rate of DSL users

---

## Repository Structure

```
customer-churn-prediction/
├── index.html               # Interactive Plotly.js dashboard (deployed via GitHub Pages)
├── churn_model.py           # Modular ML pipeline — load, clean, engineer, train, evaluate
├── churn_pipeline.ipynb     # Exploratory notebook with full analysis walkthrough
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Source dataset
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
└── README.md                # This file
```

---

## How to Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/mayankjoshiii/customer-churn-prediction.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the pipeline:

   ```bash
   python churn_model.py
   ```

4. Open `index.html` in any modern browser to view the dashboard — no server required.

---

## Live Dashboard

👉 [View the Interactive Dashboard](https://mayankjoshiii.github.io/customer-churn-prediction/)

---

## Author

**Mayank Joshi** — Business Analyst & Data Analyst  
MSc Business Analytics (Distinction) · Swansea University  
[LinkedIn](https://www.linkedin.com/in/mayankjoshi518/) · [GitHub](https://github.com/mayankjoshiii)
