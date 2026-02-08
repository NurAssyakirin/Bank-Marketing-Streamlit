import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Bank Marketing Prediction")
st.write("Enter client details to predict if they will subscribe to a term deposit.")

# User Inputs
job = st.selectbox('Job', ['Select Job', 'admin', 'student', 'self-employed', 'unemployed'])
marital = st.selectbox('Marital Status', ['Select Marital Status', 'married', 'single', 'divorced'])
age = st.number_input('Age', min_value=18, max_value=100, value=18, step=1)
education = st.selectbox('Education', ['Select Education', 'primary', 'secondary', 'tertiary', 'unknown'])
balance = st.number_input('Balance', value=0, step=100)

# Results Placeholder
result_placeholder = st.empty()

# Predict Buttons
if st.button('Predict'):
    # Validate inputs
    if job.startswith('Select') or marital.startswith('Select') or education.startswith('Select'):
        result_placeholder.warning("Please fill all fields before predicting.")
    else:
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'age': age,
            'job': job,
            'marital': marital,
            'education': education,
            'balance': balance,
            'default': 'no',
            'housing': 'no',
            'loan': 'no',
            'contact': 'cellular',
            'month': 'may',
            'day_of_week': 'mon',
            'poutcome': 'unknown'
        }])

        # One Hot Encoding User Inputs
        input_encoded = pd.get_dummies(input_data)

        # Missing Columns that exists in the Training Data 
        for col in model_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        input_encoded = input_encoded[model_columns]

        # Make Prediction
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]

        # Display Result
        result_placeholder.success(f"Prediction: {prediction}")
        result_placeholder.info(f"Probability of Subscribing: {probability: .2%}")
else:
        result_placeholder.info("Prediction will appear after clicking 'Predict'.")