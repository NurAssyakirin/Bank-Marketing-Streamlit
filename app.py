import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Bank Marketing Prediction")
st.write("Enter client details to predict if they will subscribe to a term deposit.")

# User Inputs
job = st.selectbox('Job', ['admin', 'student', 'self-employed', 'unemployed'])
marital = st.selectbox('Marital Status', ['married', 'single', 'divorced'])
age = st.number_input('Age', min_value=18, max_value=100, value=30)
education = st.selectbox('Education', ['primary', 'secondary', 'tertiary', 'unknown'])
balance = st.number_input('Balance', value=1000)

# Input Dataframe
input_data = pd.DataFrame({
    "age": [age],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "balance": [balance],
})

# Predict Buttons
if st.button('Predict'):
    # create input DataFrame with categorical columns
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

st.write(f"Prediction: {prediction}")
st.write(f"Probability of Subscribing: {probability: .2%}")