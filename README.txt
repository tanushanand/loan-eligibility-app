# Loan Eligibility Prediction App

This project predicts whether a loan application will be approved based on applicant and financial details using a logistic regression model.

Developed as part of the CST2216 (Machine Learning 2) Individual Term Project at Algonquin College.

---

## Features

- Logistic Regression model with feature scaling
- Streamlit web app for user-friendly predictions
- Modular code structure (VS Code project)
- Logging and error handling implemented
- Deployable on Streamlit Cloud

---

## Project Structure

loan_eligibility_app/
├── app.py # Streamlit frontend
├── main.py # Backend testing script
├── config/
│ └── logger.py
├── data/
│ └── credit.csv # input dataset
├── models/
│ └── model.py
├── preprocessing/
│ └── preprocessing.py
├── utils/
│ └── helpers.py
└── README.txt