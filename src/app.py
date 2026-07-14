import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier

# Page config
st.set_page_config(
    page_title="ChurnSphere Analytics - Predictive Customer Insights",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Global Font Override */
html, body, [data-testid="stAppViewContainer"], [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #0d0f1d;
    color: #f1f3f9;
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: #080911;
    border-right: 1px solid #1f2235;
}

/* Custom Header with Gradient Background */
.hero-container {
    background: linear-gradient(135deg, #6c5ce7 0%, #a8a5e6 50%, #00cec9 100%);
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 25px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 16px;
    font-weight: 400;
    color: rgba(255, 255, 255, 0.9);
    margin-top: 5px;
}

/* Card Styling */
.metric-card {
    background-color: #151828;
    border: 1px solid #232742;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    transition: transform 0.3s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    border-color: #6c5ce7;
}

.metric-title {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #8f94a5;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 32px;
    font-weight: 700;
    color: #00cec9;
}

/* Metric color variations */
.metric-value-bank {
    color: #0984e3;
}
.metric-value-telco {
    color: #fd79a8;
}

/* Prediction Output Cards */
.result-card-retained {
    background: linear-gradient(135deg, rgba(38, 222, 129, 0.15) 0%, rgba(38, 222, 129, 0.03) 100%);
    border: 1px solid #26de81;
    border-radius: 12px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(38, 222, 129, 0.2);
}

.result-card-churn {
    background: linear-gradient(135deg, rgba(252, 92, 101, 0.15) 0%, rgba(252, 92, 101, 0.03) 100%);
    border: 1px solid #fc5c65;
    border-radius: 12px;
    padding: 25px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(252, 92, 101, 0.2);
}

/* Styled Submit Button */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #6c5ce7 0%, #00cec9 100%);
    color: white;
    border: none;
    padding: 10px 30px;
    border-radius: 8px;
    font-weight: 600;
    font-size: 16px;
    width: 100%;
    box-shadow: 0 4px 15px rgba(108, 92, 231, 0.4);
    transition: all 0.3s ease;
}
div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 206, 201, 0.6);
    color: #ffffff;
}

/* Tab borders & alignment */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    background-color: #151828;
    border: 1px solid #232742;
    border-radius: 6px 6px 0 0;
    padding: 8px 16px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: #6c5ce7 !important;
    border-color: #6c5ce7 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_catboost_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    
    bank_path = os.path.join(models_dir, "bank_churn_prediction_model.cbm")
    telco_path = os.path.join(models_dir, "telco_churn_prediction_model.cbm")
    
    bank_model = None
    telco_model = None
    
    if os.path.exists(bank_path):
        bank_model = CatBoostClassifier()
        bank_model.load_model(bank_path)
        
    if os.path.exists(telco_path):
        telco_model = CatBoostClassifier()
        telco_model.load_model(telco_path)
        
    return bank_model, telco_model

bank_model, telco_model = load_catboost_models()

# Init session state
if "nav_selection" not in st.session_state:
    st.session_state.nav_selection = "📊 Executive Dashboard"

def set_nav_page(page_name):
    st.session_state.nav_selection = page_name

# Sidebar
with st.sidebar:
    st.markdown("<div style='text-align: center; padding-bottom: 20px;'><h1 style='font-size: 26px; font-weight: 800; color: #00cec9; margin:0;'>🔄 ChurnSphere</h1><p style='color: #8f94a5; font-size:12px; margin:0;'>Predictive Loyalty Analytics</p></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["📊 Executive Dashboard", "🏦 Bank Customer Churn", "📞 Telco Customer Churn", "📈 Model Insights"],
        key="nav_selection"
    )
    st.markdown("---")
    
    # Model status
    st.markdown("### System Status")
    if bank_model:
        st.markdown("🟢 **Bank Churn Model:** Active")
    else:
        st.markdown("🔴 **Bank Churn Model:** Offline")
        
    if telco_model:
        st.markdown("🟢 **Telco Churn Model:** Active")
    else:
        st.markdown("🔴 **Telco Churn Model:** Offline")
        
    st.markdown("---")
    st.markdown("<p style='color:#5a607c; font-size:11px; text-align:center;'>Version 1.0.0<br>© 2026 ChurnSphere Inc.</p>", unsafe_allow_html=True)

# Dashboard page
if page == "📊 Executive Dashboard":
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Executive Churn Analytics Dashboard</h1>
        <p class="hero-subtitle">Interactive customer churn risk profiling using advanced tuned CatBoost models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics grid
    st.markdown("<h3 style='margin-bottom:15px; color:#f1f3f9; border-bottom:1px solid #232742; padding-bottom:5px;'>🏦 Bank Customer Churn Model Metrics</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Accuracy Score</div>
            <div class="metric-value metric-value-bank">81.0%</div>
            <p style='margin:0; font-size:12px; color:#8f94a5;'>Overall correct predictions</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Exited Recall</div>
            <div class="metric-value metric-value-bank">75.0%</div>
            <p style='margin:0; font-size:12px; color:#8f94a5;'>Percent of actual churn identified</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Exited Precision</div>
            <div class="metric-value metric-value-bank">52.0%</div>
            <p style='margin:0; font-size:12px; color:#8f94a5;'>Percent of predicted churn correct</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Precision-Recall AUC</div>
            <div class="metric-value metric-value-bank">0.7294</div>
            <p style='margin:0; font-size:12px; color:#8f94a5;'>Model area under PR curve</p>
        </div>
        """, unsafe_allow_html=True)

    # Bank button
    col_btn_b1, col_btn_b2 = st.columns([1.2, 2.8])
    with col_btn_b1:
        st.button("🏦 Open Bank Predictor ➡️", key="btn_go_bank", on_click=set_nav_page, args=("🏦 Bank Customer Churn",))

    st.markdown("<br><h3 style='margin-bottom:15px; color:#f1f3f9; border-bottom:1px solid #232742; padding-bottom:5px;'>📞 Telco Subscriber Churn Model Metrics</h3>", unsafe_allow_html=True)
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Accuracy Score</div>
            <div class="metric-value metric-value-telco">77.0%</div>
            <p style='margin:0; font-size:12px; color:#8f94a5;'>Overall correct predictions</p>
        </div>
        """, unsafe_allow_html=True)
    with col6:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Churned Recall</div>
            <div class="metric-value metric-value-telco">77.0%</div>
            <p style='margin:0; font-size:12px; color:#8f94a5;'>Percent of actual churn identified</p>
        </div>
        """, unsafe_allow_html=True)
    with col7:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Churned Precision</div>
            <div class="metric-value metric-value-telco">54.0%</div>
            <p style='margin:0; font-size:12px; color:#8f94a5;'>Percent of predicted churn correct</p>
        </div>
        """, unsafe_allow_html=True)
    with col8:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Precision-Recall AUC</div>
            <div class="metric-value metric-value-telco">0.6643</div>
            <p style='margin:0; font-size:12px; color:#8f94a5;'>Model area under PR curve</p>
        </div>
        """, unsafe_allow_html=True)

    # Telco button
    col_btn_t1, col_btn_t2 = st.columns([1.2, 2.8])
    with col_btn_t1:
        st.button("📞 Open Telco Predictor ➡️", key="btn_go_telco", on_click=set_nav_page, args=("📞 Telco Customer Churn",))

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Overview
    st.markdown("## Overview & Objectives")
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("""
        ### Why Churn Prediction?
        Acquiring a new customer costs **5x to 25x** more than retaining an existing one. In both financial services (banking) and telecommunications:
        * **Banking Churn**: Customers leaving impacts deposit volume, transaction fee revenues, and credit products.
        * **Telco Churn**: High customer attrition (churn) directly depresses monthly recurring revenue (MRR) and decreases Customer Lifetime Value (CLV).
        
        ### Prediction Engine
        This application utilizes **CatBoost Classifier** models, trained on historically cleaned datasets. CatBoost is uniquely suited for these datasets because it natively optimizes categorical features (like Geography, Gender, Contract type, and Internet services) without losing valuable correlations.
        """)
    with right_col:
        st.markdown("""
        ### Key Churn Indicators (Based on Model Training)
        * **Banking**: **Age**, **Number of Products**, and **IsActiveMember** are highly critical. Older, inactive bank customers with multiple products are statistically more likely to exit.
        * **Telco**: **Contract duration** (Month-to-month contracts have high risk), **Tenure Months** (new customers churn at higher rates), and **Internet Service type** (Fiber Optic users have higher churn rates) dominate.
        
        ### Use Case Actionability
        Use the side tabs to input a customer's profile details. The model will instantly calculate their churn probability and provide actionable retention suggestions.
        """)

# Bank page
elif page == "🏦 Bank Customer Churn":
    st.markdown("""
    <div class="hero-container" style="background: linear-gradient(135deg, #0984e3 0%, #74b9ff 100%);">
        <h1 class="hero-title">Bank Customer Churn Predictor</h1>
        <p class="hero-subtitle">Assess retention risk of financial customers using CatBoost classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not bank_model:
        st.error("Bank Churn CatBoost model not found in /models directory. Please check file placement.")
    else:
        # Form layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Demographics & Basic Info")
            age = st.slider("Age", min_value=18, max_value=92, value=38, help="Customer's age in years")
            gender = st.selectbox("Gender", options=["Female", "Male"], help="Customer's gender")
            geography = st.selectbox("Geography", options=["France", "Spain", "Germany"], help="Customer's country of residence")
            tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5, help="Number of years customer has been with the bank")
            card_type = st.selectbox("Card Type", options=["DIAMOND", "GOLD", "SILVER", "PLATINUM"], help="Tier of credit/debit card held")
            
        with col2:
            st.markdown("### Financial & Account Profile")
            credit_score = st.slider("Credit Score", min_value=350, max_value=850, value=650, help="Customer's credit score rating")
            balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=250000.0, value=76000.0, step=1000.0, help="Current account savings balance")
            num_products = st.selectbox("Number of Products", options=[1, 2, 3, 4], index=0, help="Number of bank products customer uses (loans, cards, accounts)")
            estimated_salary = st.number_input("Estimated Salary ($)", min_value=10.0, max_value=200000.0, value=100000.0, step=1000.0, help="Customer's estimated annual salary")
            
            # Binary fields
            has_cr_card = st.selectbox("Has Credit Card?", options=["No", "Yes"], index=1, help="Does the customer hold an active credit card?")
            is_active_member = st.selectbox("Is Active Member?", options=["No", "Yes"], index=1, help="Has the customer made transactions recently?")
            
        # Map binary inputs
        has_cr_card_bin = 1 if has_cr_card == "Yes" else 0
        is_active_member_bin = 1 if is_active_member == "Yes" else 0
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("Predict Bank Customer Retention Risk")
        
        if predict_button:
            # Match model feature order
            features = [
                credit_score,
                geography,
                gender,
                age,
                tenure,
                balance,
                num_products,
                has_cr_card_bin,
                is_active_member_bin,
                estimated_salary,
                card_type
            ]
            
            # Predict
            input_df = pd.DataFrame([features], columns=bank_model.feature_names_)
            # Cast numeric columns
            input_df['CreditScore'] = input_df['CreditScore'].astype(int)
            input_df['Age'] = input_df['Age'].astype(int)
            input_df['Tenure'] = input_df['Tenure'].astype(int)
            input_df['Balance'] = input_df['Balance'].astype(float)
            input_df['NumOfProducts'] = input_df['NumOfProducts'].astype(int)
            input_df['HasCrCard'] = input_df['HasCrCard'].astype(int)
            input_df['IsActiveMember'] = input_df['IsActiveMember'].astype(int)
            input_df['EstimatedSalary'] = input_df['EstimatedSalary'].astype(float)
            
            prob = bank_model.predict_proba(input_df)[0][1]
            churn_risk_percentage = round(prob * 100, 2)
            
            st.markdown("---")
            st.markdown("## Prediction Analysis")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                if churn_risk_percentage > 50.0:
                    st.markdown(f"""
                    <div class="result-card-churn">
                        <h2 style='color:#fc5c65; margin:0;'>HIGH CHURN RISK</h2>
                        <h1 style='font-size: 52px; font-weight:800; color:#fc5c65; margin:10px 0;'>{churn_risk_percentage}%</h1>
                        <p style='color:#f1f3f9; margin:0;'>Customer shows strong probability of leaving the bank.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card-retained">
                        <h2 style='color:#26de81; margin:0;'>LOW RISK (RETAINED)</h2>
                        <h1 style='font-size: 52px; font-weight:800; color:#26de81; margin:10px 0;'>{churn_risk_percentage}%</h1>
                        <p style='color:#f1f3f9; margin:0;'>Customer exhibits stable behavior and high loyalty indicators.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with col_res2:
                st.markdown("### Risk Analysis & Actions")
                
                # Check warning flags
                warnings = []
                recommendations = []
                
                if age > 50:
                    warnings.append("• **Age Flag**: Customer is over 50 years old, a demographic group showing significantly elevated churn risk in historical data.")
                    recommendations.append("• Provide personalized wealth management consults or retirement planning offers.")
                if num_products >= 3:
                    warnings.append("• **Product Overload**: Customer has 3 or more products. Statistically, customers with 3 or 4 products have a very high exit rate, potentially indicating system friction or overlapping product fees.")
                    recommendations.append("• Offer account fee waivers or bundle products to consolidate costs.")
                if is_active_member_bin == 0:
                    warnings.append("• **Inactivity**: Customer is not currently an active member.")
                    recommendations.append("• Launch re-engagement emails with transaction bonuses or fee rebates.")
                if balance == 0:
                    warnings.append("• **Zero Savings Balance**: Savings accounts with empty funds are a primary indicator of transactional departure.")
                    recommendations.append("• Offer incentive deposit rates (e.g. higher savings APY) to encourage inflows.")
                
                if churn_risk_percentage > 50.0:
                    if not warnings:
                        st.markdown("⚠️ **Complex Risk Profile**: The predictive engine identifies a high exit probability based on complex combinations of features (such as region, card tier, credit score, or salary) not captured by individual alerts.")
                        st.markdown("#### Suggested Retention Actions:")
                        st.markdown("• Conduct a proactive relationship manager consult.")
                        st.markdown("• Check customer fee structures and regional competitor campaigns.")
                    else:
                        st.markdown("#### Primary Churn Drivers Identified:")
                        for w in warnings:
                            st.markdown(w)
                        st.markdown("#### Targeted Retention Interventions:")
                        for r in recommendations:
                            st.markdown(r)
                else:
                    if not warnings:
                        st.markdown("✅ **No primary warning flags.** Customer account metrics are in a safe configuration.")
                        st.markdown("#### Suggested Maintenance:")
                        st.markdown("• Maintain active relationship communications.")
                        st.markdown("• Propose standard diamond/platinum card benefit upgrades during routine cycles.")
                    else:
                        st.markdown("⚠️ **Minor Risk Flags**: Customer is in a low-risk category, but exhibits some potential friction points:")
                        for w in warnings:
                            st.markdown(w)
                        st.markdown("#### Recommended Optimizations:")
                        for r in recommendations:
                            st.markdown(r)

# Telco page
elif page == "📞 Telco Customer Churn":
    st.markdown("""
    <div class="hero-container" style="background: linear-gradient(135deg, #fd79a8 0%, #e84393 100%);">
        <h1 class="hero-title">Telecom Customer Churn Predictor</h1>
        <p class="hero-subtitle">Assess attrition probability and calculate subscriber loyalty risk</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not telco_model:
        st.error("Telco Churn CatBoost model not found in /models directory. Please check file placement.")
    else:
        # Form layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Subscriber Demographics")
            gender = st.selectbox("Gender", options=["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen?", options=["No", "Yes"])
            partner = st.selectbox("Has Partner?", options=["No", "Yes"])
            dependents = st.selectbox("Has Dependents?", options=["No", "Yes"])
            tenure_months = st.slider("Tenure (Months)", min_value=0, max_value=72, value=32, help="Number of months subscribed to the network")
            
        with col2:
            st.markdown("### Contract & Billing")
            contract = st.selectbox("Contract Type", options=["Month-to-month", "One year", "Two year"], index=0, help="Monthly vs multi-year contract commitments")
            paperless_billing = st.selectbox("Paperless Billing?", options=["Yes", "No"], index=0)
            payment_method = st.selectbox("Payment Method", options=[
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ], index=1)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=64.0, step=1.0)
            calculated_total = round(tenure_months * monthly_charges, 2)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=8700.0, value=float(calculated_total) if tenure_months > 0 else 0.0, step=10.0, help="Expected total accumulated charges. Suggested value based on inputs is shown.")
            
            # Validate total charges
            if tenure_months > 0:
                expected_total = tenure_months * monthly_charges
                diff_pct = abs(total_charges - expected_total) / expected_total if expected_total > 0 else 0
                if diff_pct > 0.3:
                    st.info(f"💡 **Validation Tip**: Based on tenure ({tenure_months} months) and monthly charges (${monthly_charges:.2f}), the expected Total Charges is around **${expected_total:.2f}**. Large differences may skew prediction accuracy.")
            
        with col3:
            st.markdown("### Subscribed Services")
            phone_service = st.selectbox("Phone Service?", options=["Yes", "No"], index=0)
            multiple_lines = st.selectbox("Multiple Lines?", options=["No", "Yes", "No phone service"], index=0)
            internet_service = st.selectbox("Internet Service?", options=["Fiber optic", "DSL", "No"], index=0)
            online_security = st.selectbox("Online Security Addon", options=["No", "Yes", "No internet service"], index=0)
            online_backup = st.selectbox("Online Backup Addon", options=["No", "Yes", "No internet service"], index=0)
            device_protection = st.selectbox("Device Protection Addon", options=["No", "Yes", "No internet service"], index=0)
            tech_support = st.selectbox("Tech Support Addon", options=["No", "Yes", "No internet service"], index=0)
            streaming_tv = st.selectbox("Streaming TV", options=["No", "Yes", "No internet service"], index=0)
            streaming_movies = st.selectbox("Streaming Movies", options=["No", "Yes", "No internet service"], index=0)
            
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("Predict Telco Subscriber Churn Risk")
        
        if predict_button:
            # Model features
            features = [
                gender,
                senior_citizen,
                partner,
                dependents,
                tenure_months,
                phone_service,
                multiple_lines,
                internet_service,
                online_security,
                online_backup,
                device_protection,
                tech_support,
                streaming_tv,
                streaming_movies,
                contract,
                paperless_billing,
                payment_method,
                monthly_charges,
                total_charges
            ]
            
            # Run prediction
            input_df = pd.DataFrame([features], columns=telco_model.feature_names_)
            # Cast numeric columns
            input_df['Tenure Months'] = input_df['Tenure Months'].astype(int)
            input_df['Monthly Charges'] = input_df['Monthly Charges'].astype(float)
            input_df['Total Charges'] = input_df['Total Charges'].astype(float)
            
            prob = telco_model.predict_proba(input_df)[0][1]
            churn_risk_percentage = round(prob * 100, 2)
            
            st.markdown("---")
            st.markdown("## Prediction Analysis")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                if churn_risk_percentage > 50.0:
                    st.markdown(f"""
                    <div class="result-card-churn">
                        <h2 style='color:#fc5c65; margin:0;'>HIGH CHURN RISK</h2>
                        <h1 style='font-size: 52px; font-weight:800; color:#fc5c65; margin:10px 0;'>{churn_risk_percentage}%</h1>
                        <p style='color:#f1f3f9; margin:0;'>Subscriber shows high likelihood of cancelling services.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card-retained">
                        <h2 style='color:#26de81; margin:0;'>LOW RISK (RETAINED)</h2>
                        <h1 style='font-size: 52px; font-weight:800; color:#26de81; margin:10px 0;'>{churn_risk_percentage}%</h1>
                        <p style='color:#f1f3f9; margin:0;'>Subscriber exhibits reliable retention behavior.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with col_res2:
                st.markdown("### Risk Analysis & Actions")
                
                # Check top warning flags for telco churn
                warnings = []
                recommendations = []
                
                if contract == "Month-to-month":
                    warnings.append("• **Month-to-Month Contract**: Rolling monthly accounts have negligible exit friction and present the highest churn rates in telcos.")
                    recommendations.append("• Offer a limited discount (e.g. $5/month off) to switch to an annual contract commitment.")
                if tenure_months < 6:
                    warnings.append("• **Early Lifecycle**: Customer has under 6 months tenure. Early stage subscribers have statistically high volatility.")
                    recommendations.append("• Establish a customer check-in call/support email at month 3 to ensure satisfaction.")
                if internet_service == "Fiber optic":
                    warnings.append("• **Fiber Optic Attrition**: Fiber optic subscribers represent the highest churn volume in historical data. This usually points to service instability or value/pricing grievances.")
                    recommendations.append("• Have tech support run line diagnostic tests, or offer high-speed internet bundle discounts.")
                if tech_support == "No" and internet_service != "No":
                    warnings.append("• **Missing Tech Support Addon**: Internet subscribers without active tech support exit quickly when technical faults occur.")
                    recommendations.append("• Offer 3 months of complimentary technical support service to build safety nets.")
                
                if churn_risk_percentage > 50.0:
                    if not warnings:
                        st.markdown("⚠️ **Complex Risk Profile**: The predictive engine identifies a high churn risk based on complex service usage combinations (such as billing method, specific addons, or charges) not captured by standard flags.")
                        st.markdown("#### Suggested Retention Actions:")
                        st.markdown("• Contact the subscriber to conduct a loyalty check-in survey.")
                        st.markdown("• Review customer billing plans and compare them with standard loyalty packages.")
                    else:
                        st.markdown("#### Primary Churn Drivers Identified:")
                        for w in warnings:
                            st.markdown(w)
                        st.markdown("#### Targeted Retention Interventions:")
                        for r in recommendations:
                            st.markdown(r)
                else:
                    if not warnings:
                        st.markdown("✅ **No primary warning flags.** Subscriber has steady long-term values (e.g. annual contracts, automatic payments).")
                        st.markdown("#### Suggested Maintenance:")
                        st.markdown("• Push seasonal upgrade opportunities (e.g. streaming packs, additional lines).")
                    else:
                        st.markdown("⚠️ **Minor Service Flags**: Subscriber is in a low-risk category, but has some areas for service optimization:")
                        for w in warnings:
                            st.markdown(w)
                        st.markdown("#### Recommended Optimizations:")
                        for r in recommendations:
                            st.markdown(r)

# ----------------- Model Insights -----------------
elif page == "📈 Model Insights":
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">Model Insights & Feature Importance</h1>
        <p class="hero-subtitle">Understand what drives customer churn in both models based on CatBoost calculations</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🏦 Bank Churn Features", "📞 Telco Churn Features"])
    
    with tab1:
        if not bank_model:
            st.error("Bank model offline. Cannot load feature importances.")
        else:
            st.markdown("### Top Drivers of Bank Churn")
            importances = bank_model.get_feature_importance()
            features = bank_model.feature_names_
            
            df_imp = pd.DataFrame({
                'Feature': features,
                'Importance (%)': importances
            }).sort_values('Importance (%)', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor('#0d0f1d')
            ax.set_facecolor('#151828')
            
            colors = sns.color_palette("Blues_d", n_colors=len(df_imp))
            bars = ax.barh(df_imp['Feature'], df_imp['Importance (%)'], color=colors, edgecolor='#1f2235')
            
            # Styling Matplotlib chart for dark mode aesthetics
            ax.tick_params(colors='#f1f3f9', labelsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#1f2235')
            ax.spines['bottom'].set_color('#1f2235')
            ax.xaxis.label.set_color('#f1f3f9')
            ax.yaxis.label.set_color('#f1f3f9')
            ax.set_xlabel("Importance Percentage (%)")
            ax.grid(color='#1f2235', linestyle='--', linewidth=0.5)
            
            # Value labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', 
                        va='center', ha='left', color='#00cec9', fontsize=9, fontweight='bold')
            
            st.pyplot(fig)
            
            st.markdown("""
            * **Age** is by far the strongest indicator of churn (over 22% importance). This highlights a critical need to adjust banking offers for older, retired cohorts who may withdraw their balances.
            * **Estimated Salary & Balance** reflect that wealth distribution and transactional profiles are heavily tied to account retention.
            * **Number of Products** indicates that customers with concentrated, complex product holdings may feel higher friction and decide to leave.
            """)
            
    with tab2:
        if not telco_model:
            st.error("Telco model offline. Cannot load feature importances.")
        else:
            st.markdown("### Top Drivers of Telco Subscriber Churn")
            importances = telco_model.get_feature_importance()
            features = telco_model.feature_names_
            
            df_imp = pd.DataFrame({
                'Feature': features,
                'Importance (%)': importances
            }).sort_values('Importance (%)', ascending=True)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor('#0d0f1d')
            ax.set_facecolor('#151828')
            
            colors = sns.color_palette("Purples_d", n_colors=len(df_imp))
            bars = ax.barh(df_imp['Feature'], df_imp['Importance (%)'], color=colors, edgecolor='#1f2235')
            
            # Styling Matplotlib chart for dark mode aesthetics
            ax.tick_params(colors='#f1f3f9', labelsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#1f2235')
            ax.spines['bottom'].set_color('#1f2235')
            ax.xaxis.label.set_color('#f1f3f9')
            ax.yaxis.label.set_color('#f1f3f9')
            ax.set_xlabel("Importance Percentage (%)")
            ax.grid(color='#1f2235', linestyle='--', linewidth=0.5)
            
            # Value labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.3, bar.get_y() + bar.get_height()/2, f'{width:.2f}%', 
                        va='center', ha='left', color='#fd79a8', fontsize=9, fontweight='bold')
            
            st.pyplot(fig)
            
            st.markdown("""
            * **Contract Type** (Month-to-month) and **Tenure Months** are the most dominant loyalty predictors for Telco subscribers. 
            * **Internet Service** and **Monthly Charges** represent customer pricing sensitivity. Fiber Optic plans, in particular, are drivers of departures, indicating that pricing or reliability of fiber services needs review.
            * Service addons like **Online Security** and **Tech Support** show that support utilities have a strong preventative correlation against subscriber attrition.
            """)
