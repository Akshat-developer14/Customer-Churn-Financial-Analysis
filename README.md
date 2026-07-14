# Customer Churn Financial Analysis & Dashboard

This repository contains machine learning models and an interactive Streamlit dashboard for predicting customer churn in both financial (Banking) and telecommunications (Telco) sectors.

The models are trained using **CatBoost Classifier** which natively handles categorical features, delivering high recall and predictive precision.

## Project Structure

```text
├── data/
│   └── cleaned/
│       ├── Bank_Customer_Churn_Cleaned.csv
│       └── Telco_customer_churn_cleaned.csv
├── models/
│   ├── bank_churn_prediction_model.cbm       # CatBoost model for Bank churn
│   └── telco_churn_prediction_model.cbm      # CatBoost model for Telco churn
├── notebooks/
│   ├── training/
│   │   ├── Bank_customer_churn_modeltraining.ipynb
│   │   └── Telco_customer_churn_modeltraining.ipynb
│   └── EDA_bank_customer.ipynb
│   └── EDA_telco.ipynb
├── src/
│   └── app.py                                # Streamlit dashboard application
├── pyproject.toml                            # Dependencies declaration
└── README.md
```

## Dashboard Features

1. **Executive Dashboard**: Provides a high-level view of model metrics (Accuracy, Recall, Precision-Recall AUC) and the business objectives of churn prevention.
2. **Bank Customer Churn**: Interactive form to input customer details (Age, Credit Score, Balance, Products, Activity status) and get immediate churn risk scoring (%), warning flags, and personalized retention strategies.
3. **Telco Customer Churn**: Interactive profile entry for subscribers (Contract type, Internet service, Tenure, Services, Monthly charges) to assess loyalty risk and discover recommended retention actions.
4. **Model Insights**: Interactive feature importance visualizations explaining what factors drive churn most (e.g., Age for banking, Contract/Tenure for telecom).

## Running the Application

### 1. Prerequisites
Ensure you have the virtual environment set up and active. If you are using `uv`, dependencies can be managed automatically.

### 2. Install Dependencies
If not already installed, install the required packages (Streamlit, CatBoost, Pandas, Matplotlib, Seaborn, etc.):
```bash
uv sync
```
Or with standard pip:
```bash
pip install -r requirements.txt
```
*(Streamlit and CatBoost are required)*

### 3. Run the Dashboard
Start the Streamlit application by running the following command from the project root:
```bash
streamlit run src/app.py
```

Open the local URL displayed in your terminal (usually `http://localhost:8501`) to interact with the app.
