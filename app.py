import streamlit as st
import pandas as pd
import pickle
# Load your model (replace with correct path if needed)
model = pickle.load(open("model.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))  # Save X.columns using pickle

def preprocess_and_predict(raw_input, model, feature_names):
    import pandas as pd

    # Mapping for binary features
    binary_map = {
        "Attrition": {"Yes": 1, "No": 0},
        "Gender": {"Male": 1, "Female": 0},
        "MaritalStatus": {"Married": 1, "Single": 0, "Divorced": 0},
        "OverTime": {"Yes": 1, "No": 0},
        "Over18": {"Y": 1}
    }

    # Apply binary encoding
    for key in binary_map:
        if key in raw_input:
            raw_input[key] = binary_map[key].get(raw_input[key], 0)

    # One-hot fields
    one_hot_fields = {
        "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"],
        "Department": ["Sales", "Research & Development", "Human Resources"],
        "EducationField": ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"],
        "JobRole": ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
                    'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources']
    }

    # Handle one-hot encoding
    for field, categories in one_hot_fields.items():
        value = raw_input.get(field)
        for cat in categories:
            key = f"{field}_{cat}".replace(" ", "_")
            raw_input[key] = 1 if value == cat else 0
        raw_input.pop(field, None)

    # Create input DataFrame with required columns
    input_df = pd.DataFrame([raw_input])

    # Ensure all model columns are present
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_names]

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return prediction, probability


st.title("Employee Attrition Prediction")

# Collect user inputs
age = st.slider("Age", 18, 60, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
overtime = st.selectbox("OverTime", ["Yes", "No"])
income = st.number_input("Monthly Income", 1000, 20000, 5000)
job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
job_role = st.selectbox("Job Role", ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
                                     'Manufacturing Director', 'Healthcare Representative', 'Manager',
                                     'Sales Representative', 'Research Director', 'Human Resources'])

# Create input dictionary
user_input = {
    "Age": age,
    "Gender": gender,
    "MaritalStatus": marital_status,
    "OverTime": overtime,
    "MonthlyIncome": income,
    "JobLevel": job_level,
    "JobSatisfaction": job_satisfaction,
    "BusinessTravel": business_travel,
    "Department": department,
    "EducationField": education_field,
    "JobRole": job_role
}

# Prediction
if st.button("Predict"):
    prediction, confidence = preprocess_and_predict(user_input, model, feature_names)
    result = "Yes" if prediction == 1 else "No"
    st.write(f"**Predicted Attrition:** {result}")
    st.write(f"**Confidence:** {round(confidence * 100, 2)}%")
