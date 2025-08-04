import streamlit as st
import pandas as pd
import numpy as np
from models.model import train_model, predict
from preprocessing.preprocessing import fill_missing_values, encode_categorical
from config.logger import get_logger
from utils.helpers import map_prediction

logger = get_logger(__name__)

@st.cache_data
def load_sample_data():
    df = pd.read_csv("data/credit.csv")
    df = fill_missing_values(df)
    df, _ = encode_categorical(df, ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents'])
    return df

@st.cache_resource
def get_trained_model(df):
    X = df.drop(columns=['Loan_ID', 'Loan_Approved'])
    y = df['Loan_Approved'].map({'Y': 1, 'N': 0})
    model, accuracy = train_model(X, y)
    return model, accuracy, X.columns

def main():
    st.set_page_config(page_title="Loan Eligibility Predictor", layout="centered")
    st.title("🏦 Loan Eligibility Predictor")
    st.write("Fill in your information below to check loan approval status.")

    df = load_sample_data()
    model, accuracy, feature_cols = get_trained_model(df)

    # UI Input Fields
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
    loan_term = st.selectbox("Loan Term (in days)", [360.0, 180.0, 300.0, 240.0, 120.0, 84.0, 60.0, 36.0, 12.0])
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

    if st.button("Predict"):
        try:
            user_input = pd.DataFrame([[
                1 if gender == "Male" else 0,
                1 if married == "Yes" else 0,
                int(dependents.replace("3+", "3")),
                0 if education == "Graduate" else 1,
                1 if self_employed == "Yes" else 0,
                applicant_income,
                coapplicant_income,
                loan_amount,
                loan_term,
                credit_history,
                {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]
            ]], columns=feature_cols)

            prediction = predict(model, user_input)
            result = map_prediction(prediction)[0]

            st.success(f"Result: Loan is **{result}**")
            st.info(f"Model Accuracy: {accuracy:.2%}")

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            st.error("An error occurred during prediction. Check logs.")

if __name__ == "__main__":
    main()