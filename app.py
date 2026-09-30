import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Page settings
st.set_page_config(
    page_title="Hospital Readmission Prediction",
    page_icon="🏥",
    layout="wide"
)


# Title
st.title("🏥 Hospital Readmission Prediction")

st.write(
    "Interactive Machine Learning Dashboard "
    "for 30-Day Hospital Readmission Prediction"
)


# Load dashboard data
df = pd.read_csv(
    "data/hospital_readmission_dashboard.csv"
)

# Load trained ML model
model = joblib.load(
    "models/hospital_readmission_model.pkl"
)


# Basic information
st.subheader("📊 Dataset Overview")

st.write(
    f"Dataset contains **{len(df):,} patient records**."
)


# KPI calculations
total_patients = len(df)

readmitted_patients = df[
    "readmission_30"
].sum()

readmission_rate = (
    readmitted_patients / total_patients
) * 100


# KPI cards
col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "Total Patients",
        f"{total_patients:,}"
    )


with col2:
    st.metric(
        "30-Day Readmissions",
        f"{readmitted_patients:,}"
    )


with col3:
    st.metric(
        "Readmission Rate",
        f"{readmission_rate:.2f}%"
    )


# Data preview
st.subheader("📋 Patient Data Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# ==========================================
# INTERACTIVE EDA
# ==========================================

import plotly.express as px


st.subheader("📈 Interactive Analysis")


# ------------------------------------------
# 1. Age-wise Readmission Rate
# ------------------------------------------

age_rate = (
    df.groupby("age")["readmission_30"]
    .mean()
    .reset_index()
)

age_rate["readmission_rate"] = (
    age_rate["readmission_30"] * 100
)

fig_age = px.bar(
    age_rate,
    x="age",
    y="readmission_rate",
    title="30-Day Readmission Rate by Age Group",
    labels={
        "age": "Age Group",
        "readmission_rate": "Readmission Rate (%)"
    }
)

st.plotly_chart(
    fig_age,
    use_container_width=True
)


# ------------------------------------------
# 2. Hospital Stay vs Readmission
# ------------------------------------------

stay_rate = (
    df.groupby("time_in_hospital")["readmission_30"]
    .mean()
    .reset_index()
)

stay_rate["readmission_rate"] = (
    stay_rate["readmission_30"] * 100
)

fig_stay = px.bar(
    stay_rate,
    x="time_in_hospital",
    y="readmission_rate",
    title="30-Day Readmission Rate by Hospital Stay",
    labels={
        "time_in_hospital": "Days in Hospital",
        "readmission_rate": "Readmission Rate (%)"
    }
)

st.plotly_chart(
    fig_stay,
    use_container_width=True
)


# ------------------------------------------
# 3. Gender-wise Readmission Rate
# ------------------------------------------

gender_rate = (
    df.groupby("gender")["readmission_30"]
    .mean()
    .reset_index()
)

gender_rate["readmission_rate"] = (
    gender_rate["readmission_30"] * 100
)

fig_gender = px.bar(
    gender_rate,
    x="gender",
    y="readmission_rate",
    title="30-Day Readmission Rate by Gender",
    labels={
        "gender": "Gender",
        "readmission_rate": "Readmission Rate (%)"
    }
)

st.plotly_chart(
    fig_gender,
    use_container_width=True
)

# ==========================================
# ML PREDICTION
# ==========================================

st.divider()

st.header("🤖 30-Day Readmission Prediction")

st.write(
    "Enter patient information below to generate "
    "a machine learning prediction."
)


# ------------------------------------------
# Prediction Form
# ------------------------------------------

with st.form("prediction_form"):

    st.subheader("👤 Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        race = st.selectbox(
            "Race",
            [
                "Caucasian",
                "AfricanAmerican",
                "Asian",
                "Hispanic",
                "Other"
            ]
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Unknown/Invalid"
            ]
        )

        age = st.selectbox(
            "Age Group",
            [
                "[0-10)",
                "[10-20)",
                "[20-30)",
                "[30-40)",
                "[40-50)",
                "[50-60)",
                "[60-70)",
                "[70-80)",
                "[80-90)",
                "[90-100)"
            ]
        )

    with col2:

        time_in_hospital = st.number_input(
            "Days in Hospital",
            min_value=1,
            max_value=14,
            value=3
        )

        num_lab_procedures = st.number_input(
            "Number of Lab Procedures",
            min_value=0,
            max_value=200,
            value=40
        )

        num_procedures = st.number_input(
            "Number of Procedures",
            min_value=0,
            max_value=10,
            value=1
        )

        num_medications = st.number_input(
            "Number of Medications",
            min_value=0,
            max_value=100,
            value=10
        )

    with col3:

        number_outpatient = st.number_input(
            "Previous Outpatient Visits",
            min_value=0,
            max_value=100,
            value=0
        )

        number_emergency = st.number_input(
            "Previous Emergency Visits",
            min_value=0,
            max_value=100,
            value=0
        )

        number_inpatient = st.number_input(
            "Previous Inpatient Visits",
            min_value=0,
            max_value=100,
            value=0
        )

        number_diagnoses = st.number_input(
            "Number of Diagnoses",
            min_value=1,
            max_value=20,
            value=5
        )


    # --------------------------------------
    # Diabetes Information
    # --------------------------------------

    st.subheader("💊 Diabetes & Medication Information")

    col4, col5 = st.columns(2)

    with col4:

        diabetesMed = st.selectbox(
            "Diabetes Medication",
            ["Yes", "No"]
        )

        change = st.selectbox(
            "Medication Changed",
            ["No", "Ch"]
        )

    with col5:

        max_glu_serum = st.selectbox(
            "Max Glucose Serum",
            [
                "None",
                "Norm",
                ">200",
                ">300"
            ]
        )

        A1Cresult = st.selectbox(
            "A1C Result",
            [
                "None",
                "Norm",
                ">7",
                ">8"
            ]
        )


    # --------------------------------------
    # Diagnosis Information
    # --------------------------------------

    st.subheader("🩺 Diagnosis Information")

    col6, col7, col8 = st.columns(3)

    with col6:

        diag_1_category = st.selectbox(
            "Primary Diagnosis Category",
            ["Circulatory", "Respiratory", "Diabetes",
             "Digestive", "Injury", "Other"]
        )

    with col7:

        diag_2_category = st.selectbox(
            "Secondary Diagnosis Category",
            ["Circulatory", "Respiratory", "Diabetes",
             "Digestive", "Injury", "Other"]
        )

    with col8:

        diag_3_category = st.selectbox(
            "Additional Diagnosis Category",
            ["Circulatory", "Respiratory", "Diabetes",
             "Digestive", "Injury", "Other"]
        )


    # --------------------------------------
    # Previous Visits
    # --------------------------------------

    total_previous_visits = (
        number_outpatient
        + number_emergency
        + number_inpatient
    )

    num_medications_used = (
        1 if diabetesMed == "Yes" else 0
    )

    medication_changed = (
        1 if change == "Ch" else 0
    )


    # --------------------------------------
    # Prediction Button
    # --------------------------------------

    predict_button = st.form_submit_button(
        "🔮 Predict Readmission",
        use_container_width=True
    )


# ==========================================
# MAKE PREDICTION
# ==========================================

if predict_button:

    input_data = pd.DataFrame({

        "race": [race],

        "gender": [gender],

        "age": [age],

        "admission_type_id": [1],

        "discharge_disposition_id": [1],

        "admission_source_id": [1],

        "time_in_hospital": [time_in_hospital],

        "num_lab_procedures": [num_lab_procedures],

        "num_procedures": [num_procedures],

        "num_medications": [num_medications],

        "number_outpatient": [number_outpatient],

        "number_emergency": [number_emergency],

        "number_inpatient": [number_inpatient],

        "number_diagnoses": [number_diagnoses],

        "max_glu_serum": [max_glu_serum],

        "A1Cresult": [A1Cresult],

        "metformin": ["No"],

        "repaglinide": ["No"],

        "nateglinide": ["No"],

        "chlorpropamide": ["No"],

        "glimepiride": ["No"],

        "acetohexamide": ["No"],

        "glipizide": ["No"],

        "glyburide": ["No"],

        "tolbutamide": ["No"],

        "pioglitazone": ["No"],

        "rosiglitazone": ["No"],

        "acarbose": ["No"],

        "miglitol": ["No"],

        "troglitazone": ["No"],

        "tolazamide": ["No"],

        "examide": ["No"],

        "citoglipton": ["No"],

        "insulin": ["No"],

        "glyburide-metformin": ["No"],

        "glipizide-metformin": ["No"],

        "glimepiride-pioglitazone": ["No"],

        "metformin-rosiglitazone": ["No"],

        "metformin-pioglitazone": ["No"],

        "diabetesMed": [diabetesMed],

        "diag_1_category": [diag_1_category],

        "diag_2_category": [diag_2_category],

        "diag_3_category": [diag_3_category],

        "total_previous_visits": [
            total_previous_visits
        ],

        "num_medications_used": [
            num_medications_used
        ],

        "medication_changed": [
            medication_changed
        ]
    })


    # --------------------------------------
    # Prediction
    # --------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    probability = model.predict_proba(
        input_data
    )[0][1]


    probability_percent = probability * 100


    # --------------------------------------
    # Result
    # --------------------------------------

    st.divider()

    st.subheader("🎯 Prediction Result")


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        if prediction == 1:

            st.error(
                "⚠️ Higher Predicted Risk"
            )

            st.write(
                "The model predicts that the "
                "patient may be readmitted "
                "within 30 days."
            )

        else:

            st.success(
                "✅ Lower Predicted Risk"
            )

            st.write(
                "The model predicts that the "
                "patient is less likely to be "
                "readmitted within 30 days."
            )


    with result_col2:

        st.metric(
            "Estimated 30-Day Readmission Probability",
            f"{probability_percent:.2f}%"
        )

        st.progress(
            float(probability)
        )


    st.info(
        "This prediction is generated by a "
        "machine learning model and should not "
        "be treated as a medical diagnosis."
    )