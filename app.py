from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="SpO₂ Forecasting",
    page_icon="🫁",
    layout="wide"
)


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "final_gradient_boosting_model.joblib"
)

FEATURES_FILE = (
    PROJECT_ROOT
    / "models"
    / "final_features.json"
)


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_FILE)

    with open(FEATURES_FILE, "r") as f:
        feature_names = json.load(f)

    return model, feature_names


model, final_features = load_model()


# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------

st.title("🫁 Oxygen Saturation Forecasting")

st.write(
    """
    This machine-learning application estimates a patient's
    **SpO₂ five minutes into the future** using recent
    oxygen-saturation and heart-rate measurements.
    """
)

st.warning(
    "Educational machine-learning demonstration only. "
    "This application is not intended for diagnosis, treatment, "
    "clinical monitoring, or medical decision-making."
)

st.divider()


# ---------------------------------------------------------
# SPO2 INPUTS
# ---------------------------------------------------------

st.header("1. Recent SpO₂ Measurements")

st.write(
    "Enter the patient's recent oxygen-saturation measurements."
)

col1, col2, col3 = st.columns(3)

with col1:

    spo2_5 = st.number_input(
        "SpO₂ 5 minutes ago (%)",
        min_value=70.0,
        max_value=100.0,
        value=98.0,
        step=0.1
    )

    spo2_4 = st.number_input(
        "SpO₂ 4 minutes ago (%)",
        min_value=70.0,
        max_value=100.0,
        value=98.0,
        step=0.1
    )


with col2:

    spo2_3 = st.number_input(
        "SpO₂ 3 minutes ago (%)",
        min_value=70.0,
        max_value=100.0,
        value=98.0,
        step=0.1
    )

    spo2_2 = st.number_input(
        "SpO₂ 2 minutes ago (%)",
        min_value=70.0,
        max_value=100.0,
        value=98.0,
        step=0.1
    )


with col3:

    spo2_1 = st.number_input(
        "SpO₂ 1 minute ago (%)",
        min_value=70.0,
        max_value=100.0,
        value=98.0,
        step=0.1
    )

    spo2_now = st.number_input(
        "Current SpO₂ (%)",
        min_value=70.0,
        max_value=100.0,
        value=98.0,
        step=0.1
    )


# ---------------------------------------------------------
# HEART RATE INPUTS
# ---------------------------------------------------------

st.header("2. Recent Heart Rate Measurements")

hr1, hr2, hr3, hr4, hr5 = st.columns(5)

with hr1:
    heart_rate_4 = st.number_input(
        "HR 4 min ago",
        min_value=30.0,
        max_value=220.0,
        value=75.0
    )

with hr2:
    heart_rate_3 = st.number_input(
        "HR 3 min ago",
        min_value=30.0,
        max_value=220.0,
        value=75.0
    )

with hr3:
    heart_rate_2 = st.number_input(
        "HR 2 min ago",
        min_value=30.0,
        max_value=220.0,
        value=75.0
    )

with hr4:
    heart_rate_1 = st.number_input(
        "HR 1 min ago",
        min_value=30.0,
        max_value=220.0,
        value=75.0
    )

with hr5:
    heart_rate_now = st.number_input(
        "Current HR",
        min_value=30.0,
        max_value=220.0,
        value=75.0
    )


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

spo2_recent_5 = np.array([
    spo2_4,
    spo2_3,
    spo2_2,
    spo2_1,
    spo2_now
])

heart_rate_recent_5 = np.array([
    heart_rate_4,
    heart_rate_3,
    heart_rate_2,
    heart_rate_1,
    heart_rate_now
])

feature_values = {

    "spo2": spo2_now,

    "spo2_lag_1": spo2_1,

    "spo2_lag_3": spo2_3,

    "spo2_lag_5": spo2_5,

    "spo2_change_1min": (
        spo2_now - spo2_1
    ),

    "spo2_change_3min": (
        spo2_now - spo2_3
    ),

    "spo2_rolling_mean_5": (
        spo2_recent_5.mean()
    ),

    "spo2_rolling_std_5": (
        pd.Series(spo2_recent_5).std()
    ),

    "heart_rate": heart_rate_now,

    "heart_rate_lag_1": heart_rate_1,

    "heart_rate_rolling_mean_5": (
        heart_rate_recent_5.mean()
    )
}


input_df = pd.DataFrame(
    [feature_values]
)

input_df = input_df[
    final_features
]


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

st.divider()

st.header("3. Generate Forecast")

if st.button(
    "Predict SpO₂ in 5 Minutes",
    type="primary",
    width="stretch"
):

    prediction = model.predict(
        input_df
    )[0]

    # Keep displayed SpO₂ within the physiological scale
    prediction = float(
    np.clip(prediction, 0, 100)
)

    persistence_prediction = spo2_now

    predicted_change = (
        prediction - spo2_now
    )


    st.subheader("Prediction Results")

    result1, result2, result3 = st.columns(3)


    with result1:

        st.metric(
            "Current SpO₂",
            f"{spo2_now:.1f}%"
        )


    with result2:

        st.metric(
            "Predicted SpO₂ in 5 min",
            f"{prediction:.2f}%",
            delta=f"{predicted_change:+.2f}"
        )


    with result3:

        st.metric(
            "Persistence Baseline",
            f"{persistence_prediction:.1f}%"
        )


    # -----------------------------------------------------
    # TREND CHART
    # -----------------------------------------------------

    chart_df = pd.DataFrame({

        "Minute": [
            -5,
            -4,
            -3,
            -2,
            -1,
            0,
            5
        ],

        "SpO₂": [
            spo2_5,
            spo2_4,
            spo2_3,
            spo2_2,
            spo2_1,
            spo2_now,
            prediction
        ]
    })

    chart_df = chart_df.set_index(
        "Minute"
    )

    st.subheader(
        "Recent SpO₂ Trend and Forecast"
    )

    st.line_chart(
        chart_df
    )


    # -----------------------------------------------------
    # FEATURE TABLE
    # -----------------------------------------------------

    with st.expander(
        "View engineered model features"
    ):

        st.dataframe(
            input_df,
            width="stretch"
        )


# ---------------------------------------------------------
# MODEL INFORMATION
# ---------------------------------------------------------

st.divider()

st.subheader("About the Final Model")

st.write(
    """
    The deployed model is a tuned **Gradient Boosting Regressor**
    trained using patient-level physiological time-series data.

    The model uses recent SpO₂ history, short-term SpO₂ changes,
    rolling statistics, and heart-rate information to forecast
    oxygen saturation five minutes ahead.
    """
)


metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Test RMSE",
        "1.1818"
    )

with metric2:
    st.metric(
        "Test R²",
        "0.2976"
    )

with metric3:
    st.metric(
        "Test MAE",
        "0.5375"
    )


st.info(
    """
    On unseen test patients, the persistence baseline achieved
    lower overall MAE because most five-minute intervals were stable.

    The Gradient Boosting model achieved lower RMSE and higher R²,
    and performed better during moderate and large SpO₂ changes.
    """
)