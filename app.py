import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("model/churn_model.joblib")


@st.cache_data
def load_metrics():
    with open("model/metrics.json") as f:
        return json.load(f)


model = load_model()
metrics = load_metrics()

st.title("📉 Telecom Customer Churn Predictor")
st.caption("Predict whether a customer is likely to leave, and get a retention recommendation.")

tab_pred, tab_perf, tab_about = st.tabs(["🔮 Prediction", "📊 Model performance", "ℹ️ About"])

with tab_pred:
    st.subheader("Enter customer details")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Has partner", ["Yes", "No"])
        dependents = st.selectbox("Has dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone service", ["Yes", "No"])
    with c2:
        multi = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
    with c3:
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, step=1.0)
    total = st.number_input("Total charges ($)", 0.0, 10000.0,
                            float(round(monthly * tenure, 2)), step=10.0)

    if st.button("Predict churn risk", type="primary"):
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
            "OnlineSecurity": security, "OnlineBackup": backup,
            "DeviceProtection": protection, "TechSupport": support,
            "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly, "TotalCharges": total}])
        prob = float(model.predict_proba(row)[0, 1])

        if prob >= 0.65:
            level, icon = "HIGH", "🔴"
        elif prob >= 0.35:
            level, icon = "MEDIUM", "🟠"
        else:
            level, icon = "LOW", "🟢"

        m1, m2 = st.columns(2)
        m1.metric("Churn probability", f"{prob:.1%}")
        m2.metric("Risk level", f"{icon} {level}")
        st.progress(min(prob, 1.0))

        st.subheader("💡 Business recommendation")
        tips = []
        if contract == "Month-to-month":
            tips.append("Offer a discount to move the customer to a 1- or 2-year contract (the biggest churn driver).")
        if tenure < 12:
            tips.append("New customer: run an onboarding / first-year loyalty programme.")
        if internet == "Fiber optic" and monthly > 80:
            tips.append("High fibre bill: review pricing or bundle value-added services.")
        if internet != "No" and support == "No":
            tips.append("Offer free or discounted Tech Support and Online Security for 3 months.")
        if payment == "Electronic check":
            tips.append("Encourage automatic payment (bank transfer / card) with a small incentive.")
        if senior == "Yes":
            tips.append("Provide a dedicated support line for senior customers.")
        if level == "HIGH":
            st.error("Immediate retention action needed. Contact this customer with a personalised offer.")
        elif level == "MEDIUM":
            st.warning("Monitor this customer and consider a proactive engagement offer.")
        else:
            st.success("Customer looks stable. Maintain service quality and consider upsell opportunities.")
        for t in tips:
            st.write("• " + t)

with tab_perf:
    st.subheader(f"Selected model: {metrics['best_model']}")
    st.dataframe(pd.DataFrame(metrics["results"]).T, width='stretch')
    st.caption("Metrics computed on a 20% stratified hold-out test set. The model was selected by ROC-AUC; "
               "recall is prioritised because missing a churner is costlier than a false alarm.")

with tab_about:
    st.markdown("""
**Business problem:** Telecom companies lose revenue when customers churn. Acquiring a new customer costs
much more than retaining one, so identifying at-risk customers early lets the business act first.

**Dataset:** IBM Telco Customer Churn (7,043 customers, 20 features).

**Pipeline:** data cleaning → one-hot encoding + scaling → Logistic Regression / Random Forest /
Gradient Boosting → evaluation → best model saved with `joblib` → this Streamlit app.
""")
