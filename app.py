import streamlit as st
from pathlib import Path
import pandas as pd
import joblib
import base64


def set_bg_image():
    image_path = Path("BankBackground.jpeg")
    if not image_path.exists():
        st.warning("Background image not found")
        return

    encoded = base64.b64encode(image_path.read_bytes()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpeg;base64,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}

        section[data-testid="stSidebar"] {{
            background-color: rgba(255, 255, 255, 0.9);
        }}

        .block-container {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_bg_image()

st.markdown(
    """
    <style>
    /* Titles */
    h1, h2, h3 {
        color: black !important;
    }

    /* Normal text */
    .stMarkdown, .stMarkdown p {
        color: black !important;
    }

    /* Input labels */
    label {
        color: black !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the trained model
model = joblib.load("rf_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# App Title
st.title("Bank Marketing Prediction")
st.write(
    "Enter client details to predict if they will subscribe to a term deposit."
)

# Session State Defaults
if "job" not in st.session_state:
    st.session_state.job = "Select Job"
if "marital" not in st.session_state:
    st.session_state.marital = "Select Marital Status"
if "age" not in st.session_state:
    st.session_state.age = 18
if "education" not in st.session_state:
    st.session_state.education = "Select Education"
if "balance" not in st.session_state:
    st.session_state.balance = 0

# User Inputs
job = st.selectbox(
    "Job",
    ["Select Job", "admin", "student", "self-employed", "unemployed"],
    key="job",
)
marital = st.selectbox(
    "Marital Status",
    ["Select Marital Status", "married", "single", "divorced"],
    key="marital",
)
age = st.number_input(
    "Age", min_value=18, max_value=100, value=st.session_state.age, key="age"
)
education = st.selectbox(
    "Education",
    ["Select Education", "primary", "secondary", "tertiary", "unknown"],
    key="education",
)
balance = st.number_input(
    "Balance", value=st.session_state.balance, step=100, key="balance"
)

# Results Placeholder
result_placeholder = st.empty()

# Predict Buttons
if st.button("Predict"):
    # Validate inputs
    if (
        job.startswith("Select")
        or marital.startswith("Select")
        or education.startswith("Select")
    ):
        result_placeholder.warning("Please fill all fields before predicting.")
    else:
        # Create input DataFrame
        input_data = pd.DataFrame(
            [
                {
                    "age": age,
                    "job": job,
                    "marital": marital,
                    "education": education,
                    "balance": balance,
                    "default": "no",
                    "housing": "no",
                    "loan": "no",
                    "contact": "cellular",
                    "month": "may",
                    "day_of_week": "mon",
                    "poutcome": "unknown",
                }
            ]
        )

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
        result_placeholder.info(
            f"Probability of Subscribing: {probability: .2%}"
        )

        # Reset Button
        if st.button("Reset"):
            st.session_state.job = "Select Job"
            st.session_state.marital = "Select Marital Status"
            st.session_state.age = 18
            st.session_state.education = "Select Education"
            st.session_state.balance = 0
            result_placeholder.empty()  # Clear Prediction Output
            st.experimental_rerun()  # Refresh App with defaultsconda conda ac