import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('rf_model.pkl')

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
    prediction = model.predict(input_data)[0] # Yes or No
    probability = model.predict_proba(input_data)[0][1] # Probability of Subscribing
    st.write(f"Prediction: {prediction}")
    st.write(f"Probability of Subscribing: {probability: .2%}")