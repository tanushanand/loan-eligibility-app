"""
Model training and prediction functions for the Loan Eligibility Model.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return pipeline, accuracy
    except Exception as e:
        raise Exception(f"Error during model training: {e}")

def predict(model, input_df):
    try:
        return model.predict(input_df)
    except Exception as e:
        raise Exception(f"Error during prediction: {e}")
