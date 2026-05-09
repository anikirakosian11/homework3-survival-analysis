# Homework 3 – Survival Analysis
**Marketing Analytics | Karen Hovhannisyan**

## Overview
This project models **customer churn** using Accelerated Failure Time (AFT) survival models on the Telco subscriber dataset (1,000 customers). It covers model comparison, feature selection, and Customer Lifetime Value (CLV) calculation per subscriber.

## Structure
```
├── Homework3_SurvivalAnalysis.ipynb   # Main notebook (code + report)
├── survival_analysis.py               # Standalone script version
├── telco.csv                          # Dataset
├── aft_model_comparison.png           # Survival curves (all models)
├── clv_by_segment.png                 # CLV segment analysis
├── requirements.txt
└── README.md
```

## Key Findings
- **Best model:** Log-Normal AFT (AIC = 2966, C-index = 0.786)
- **Top churn drivers:** Customer category, internet add-on, marital status, age, address stability
- **Median CLV:** ~$28,000 | **Mean CLV:** ~$51,500
- **Most valuable segment:** Plus & Total service, age 45–65, married
- **Suggested retention budget:** ~$34,500/year

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook Homework3_SurvivalAnalysis.ipynb
# or:
python survival_analysis.py
```

## Methods
- AFT models: Weibull, Log-Normal, Log-Logistic, Generalised Gamma (lifelines)
- Model comparison: AIC, log-likelihood, concordance index
- CLV formula: `CLV = MM × Σ p_i / (1 + r/12)^(i-1)` (60-month horizon, 10% discount)
