import streamlit as st
import joblib
import numpy as np

# Load the model safely
@st.cache_resource
def load_model():
    return joblib.load("Credit_Card_model.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()  

st.title("Credit Card Default Prediction App")

# Input fields
credit_limit = st.number_input("Credit Limit", min_value=1000)
bill_amount = st.number_input("Bill Amount", min_value=0)
payment_amount = st.number_input("Payment Amount", min_value=0)
utilization = st.number_input("Credit Utilization Ratio", 0.0, 1.0, 0.3)
late_payments = st.number_input("Late Payments", min_value=0)
avg_spend = st.number_input("Average Monthly Spend", min_value=0)

if st.button("Predict"):
    try:
        # Prepare input data (2D array)
        input_data = np.array([[credit_limit,
                                bill_amount,
                                payment_amount,
                                utilization,
                                late_payments,
                                avg_spend]])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.error("High Risk: Likely to Default")
        else:
            st.success("Low Risk: Not Likely to Default")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
