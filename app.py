import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ---------- Page setup ----------
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
    </style>
""", unsafe_allow_html=True)

model = joblib.load('src/churn_model.pkl')
model_columns = joblib.load('src/model_columns.pkl')

st.title("📊 Customer Churn Prediction Tool")
st.caption("Predict whether a telecom customer is likely to leave, and understand exactly why.")

st.divider()

# ---------- Default values (used for both initial state and Reset) ----------
DEFAULTS = {
    "gender": "Male", "senior": "No", "partner": "No", "dependents": "No",
    "tenure": 12, "contract": "Month-to-month", "paperless_billing": "No",
    "payment_method": "Electronic check", "phone_service": "Yes",
    "internet_service": "DSL", "online_security": "No", "tech_support": "No",
    "monthly_charges": 70.0, "total_charges": 840.0,
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

def reset_form():
    for key, val in DEFAULTS.items():
        st.session_state[key] = val

# ---------- Input form (sidebar keeps main area clean) ----------
with st.sidebar:
    st.header("Customer Profile")

    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    senior = st.selectbox("Senior Citizen", ["No", "Yes"], key="senior")
    partner = st.selectbox("Has Partner", ["No", "Yes"], key="partner")
    dependents = st.selectbox("Has Dependents", ["No", "Yes"], key="dependents")

    st.subheader("Account")
    tenure = st.slider("Tenure (months)", 0, 72, key="tenure")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], key="contract")
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], key="paperless_billing")
    payment_method = st.selectbox("Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        key="payment_method")

    st.subheader("Services")
    phone_service = st.selectbox("Phone Service", ["No", "Yes"], key="phone_service")

    if phone_service == "No":
        multiple_lines = "No phone service"
        st.selectbox("Multiple Lines", ["No phone service"], disabled=True, key="multiple_lines_locked")
    else:
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"], key="multiple_lines")

    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], key="internet_service")

    if internet_service == "No":
        online_security = "No internet service"
        tech_support = "No internet service"
        online_backup = "No internet service"
        device_protection = "No internet service"
        streaming_tv = "No internet service"
        streaming_movies = "No internet service"
        st.selectbox("Online Security", ["No internet service"], disabled=True, key="online_security_locked")
        st.selectbox("Tech Support", ["No internet service"], disabled=True, key="tech_support_locked")
        st.caption("ℹ️ No internet service selected — all internet add-ons are locked to unavailable.")
    else:
        online_security = st.selectbox("Online Security", ["No", "Yes"], key="online_security")
        tech_support = st.selectbox("Tech Support", ["No", "Yes"], key="tech_support")
        online_backup = "No"
        device_protection = "No"
        streaming_tv = "No"
        streaming_movies = "No"

    st.subheader("Billing")
    # Realistic price ranges observed in the training data per internet service type
    price_ranges = {
        "Fiber optic": (67.75, 118.75),
        "DSL": (23.45, 94.80),
        "No": (18.25, 26.90),
    }
    min_price, max_price = price_ranges[internet_service]
    # Keep the slider's current value within the valid range for this service type
    if st.session_state["monthly_charges"] < min_price or st.session_state["monthly_charges"] > max_price:
        st.session_state["monthly_charges"] = (min_price + max_price) / 2

    monthly_charges = st.slider("Monthly Charges ($)", min_price, max_price, key="monthly_charges")
    st.caption(f"ℹ️ Range reflects real {internet_service} pricing in the training data.")

    auto_total = st.checkbox("Auto-calculate Total Charges from tenure × monthly", value=True)
    if auto_total:
        st.session_state["total_charges"] = round(tenure * monthly_charges, 2)
        total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, key="total_charges", disabled=True)
    else:
        total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, key="total_charges")
        expected_total = tenure * monthly_charges
        if tenure > 0 and abs(total_charges - expected_total) > max(0.25 * expected_total, 50):
            st.caption(f"⚠️ Total charges look off — expect roughly **${expected_total:,.0f}** for {tenure} months at ${monthly_charges}/mo.")

    col_a, col_b = st.columns(2)
    predict_btn = col_a.button("🔮 Predict", use_container_width=True, type="primary")
    col_b.button("↺ Reset", use_container_width=True, on_click=reset_form)

# ---------- Prediction + display ----------
if predict_btn:
    input_dict = {
        'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection, 'TechSupport': tech_support,
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless_billing, 'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }

    with st.spinner("Analyzing customer profile..."):
        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
        input_encoded = input_encoded.astype(float)  # ensures no bool dtype issues anywhere

        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_encoded)

    # ---------- Result cards ----------
    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Probability", f"{probability:.1%}")
    col2.metric("Prediction", "Will Churn" if prediction == 1 else "Will Stay")
    col3.metric("Confidence", "High" if abs(probability - 0.5) > 0.3 else "Moderate")

    st.progress(float(probability))

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Churn** — this customer has a {probability:.1%} chance of leaving.")
    else:
        st.success(f"✅ **Low Risk of Churn** — this customer has only a {probability:.1%} chance of leaving.")

    st.divider()

    # ---------- SHAP explanation (must run BEFORE the recommendation block below) ----------
    st.subheader("🔍 Why this prediction?")
    st.caption("Each bar shows how much that factor pushed the prediction toward or away from churn.")

    shap_df = pd.DataFrame({
        'feature': input_encoded.columns,
        'value': input_encoded.iloc[0].values,
        'shap_value': shap_values[0]
    })
    shap_df = shap_df.reindex(shap_df.shap_value.abs().sort_values(ascending=False).index)
    top_features = shap_df.head(8).copy()
    top_features['label'] = top_features['feature'] + " = " + top_features['value'].round(2).astype(str)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ['#ff4b4b' if v > 0 else '#4b7bff' for v in top_features['shap_value']]
    ax.barh(top_features['label'], top_features['shap_value'], color=colors)
    ax.axvline(0, color='#888', linewidth=0.8)
    ax.set_xlabel("Impact on churn risk  (red = increases risk, blue = decreases risk)")
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

    # ---------- Actionable recommendation (uses shap_df created above) ----------
    st.subheader("💡 Recommended Action")

    if prediction == 1:
        # Look at the top risk-increasing features (positive SHAP values) for this customer
        risk_factors = shap_df[shap_df['shap_value'] > 0].head(3)['feature'].tolist()

        tips = []
        if any('Month-to-month' in f for f in risk_factors):
            tips.append("offer a discount to switch to a **1-year or 2-year contract**")
        if any('tenure' in f for f in risk_factors) and tenure < 6:
            tips.append("this is a **new customer** — a welcome check-in call or onboarding offer can build loyalty early")
        if any('MonthlyCharges' in f for f in risk_factors):
            tips.append("consider a **loyalty discount or bundle offer** to reduce their monthly cost")
        if any('Fiber optic' in f for f in risk_factors):
            tips.append("check in about **service quality/reliability**, fiber customers churn more often")
        if any('Electronic check' in f for f in risk_factors):
            tips.append("encourage **switching to automatic payment** (bank transfer/credit card), it's linked to lower churn")

        if not tips:
            tips.append("reach out proactively to understand their concerns before they cancel")

        st.warning(f"📞 **Call to action:** For this customer, {tips[0]}.")

    else:
        st.info("✅ This customer looks stable — no urgent action needed. Standard loyalty touchpoints are enough.")

else:
    st.info("👈 Fill in the customer profile in the sidebar and click **Predict Churn Risk** to get started.")