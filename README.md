# 📉 Telecom Customer Churn Prediction – Streamlit App

**Live app:** https://YOUR-APP-NAME.streamlit.app  *(replace after deployment)*
**Colab notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/churn-prediction-app/blob/main/notebooks/churn_analysis.ipynb)
**GitHub repo:** https://github.com/YOUR-USERNAME/churn-prediction-app

## 1. Business Problem
Telecom companies lose revenue when customers cancel. Retaining a customer costs much less than acquiring a new one,
so the company needs to know **which customers are likely to churn** and **what to do about it**.

## 2. Dataset
IBM *Telco Customer Churn* – 7,043 customers, 20 features (demographics, services, contract, billing) and the target `Churn`.
Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) · [IBM GitHub copy](https://github.com/IBM/telco-customer-churn-on-icp4d)

## 3. Methodology
1. **Cleaning:** converted `TotalCharges` to numeric (11 blanks = new customers → 0), dropped `customerID`, removed duplicates.
2. **EDA:** churn share (~27%), churn by contract / internet service / payment method, tenure and charges distributions.
3. **Preparation:** `StandardScaler` for numeric columns, `OneHotEncoder` for categorical columns inside a scikit-learn `Pipeline`.
4. **Modelling:** Logistic Regression, Random Forest and Gradient Boosting; 80/20 stratified train–test split.
5. **Selection:** best model by ROC-AUC, saved with `joblib` (`model/churn_model.joblib`).
6. **App:** Streamlit form → churn probability → risk level → business recommendations.

## 4. Model Performance (20% hold-out set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (selected)** | 0.742 | 0.508 | 0.780 | 0.615 | **0.840** |
| Random Forest | 0.787 | 0.586 | 0.667 | 0.624 | 0.836 |
| Gradient Boosting | 0.800 | 0.659 | 0.508 | 0.574 | 0.839 |

Recall is prioritised: missing a customer who will churn costs more than contacting one who would have stayed.

## 5. Business Insights & Recommendations
* **Month-to-month contracts** churn the most → offer discounts for 1–2 year plans.
* **First-year customers** are most at risk → onboarding and early loyalty offers.
* **No Tech Support / Online Security** → bundle these free for at-risk customers.
* **Electronic-check payers** churn more → incentivise automatic payment.
* **High fibre-optic bills** → review price/value.

## 6. Run the App Locally
```bash
git clone https://github.com/YOUR-USERNAME/churn-prediction-app.git
cd churn-prediction-app
pip install -r requirements.txt
streamlit run app.py
```
To retrain the model: `python train_model.py`

## 7. Project Structure
```
├── app.py                       # Streamlit application
├── train_model.py               # Training + evaluation script
├── requirements.txt
├── model/
│   ├── churn_model.joblib       # Trained pipeline
│   └── metrics.json             # Evaluation results
├── notebooks/churn_analysis.ipynb   # Colab notebook (cleaning, EDA, model, evaluation)
└── README.md
```

## 8. Team
| Name | Roll No. | Contribution |
|---|---|---|
| | | |
