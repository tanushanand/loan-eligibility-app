"""
Helper functions for the Loan Eligibility App.
"""

def map_prediction(prediction):
    try:
        return ["Approved" if p == 1 else "Rejected" for p in prediction]
    except Exception as e:
        raise Exception(f"Error in mapping prediction: {e}")
