"""
Main script to test the Loan Eligibility Model pipeline.
"""

import os
import pandas as pd
from config.logger import get_logger
from preprocessing.preprocessing import load_data, fill_missing_values, encode_categorical
from models.model import train_model
from utils.helpers import map_prediction

logger = get_logger(__name__)

def main():
    try:
        logger.info("Loading data...")
        df = pd.read_csv("data/loan_data.csv")

        logger.info("Filling missing values...")
        df = fill_missing_values(df)

        logger.info("🔡 Encoding categorical columns...")
        categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 
                            'Property_Area', 'Dependents']
        df, encoders = encode_categorical(df, categorical_cols)

        logger.info("Preparing features and target...")
        X = df.drop(columns=['Loan_ID', 'Loan_Approved'])
        y = df['Loan_Approved'].map({'Y': 1, 'N': 0})

        logger.info("Training model...")
        model, accuracy = train_model(X, y)

        logger.info(f"Model training complete. Accuracy: {accuracy:.2%}")

        logger.info("🔍 Predicting a sample input...")
        sample = X.iloc[[0]]
        pred = model.predict(sample)
        pred_label = map_prediction(pred)
        logger.info(f"Sample Prediction: {pred_label[0]}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()