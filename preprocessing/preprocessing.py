"""
Preprocessing functions for the Loan Eligibility Model.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def encode_categorical(df, columns):
    try:
        label_encoders = {}
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        return df, label_encoders
    except Exception as e:
        raise Exception(f"Error in encoding: {e}")

def fill_missing_values(df):
    try:
        df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
        df['Married'].fillna(df['Married'].mode()[0], inplace=True)
        df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
        df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
        df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
        df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
        df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Error in filling missing values: {e}")