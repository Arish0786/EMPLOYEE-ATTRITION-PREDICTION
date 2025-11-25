from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)

# Load the model and feature names
model = pickle.load(open("model.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

def preprocess_and_predict(raw_input, model, feature_names):
    """
    Preprocess user input and make prediction
    """
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

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        
        # Prepare input data
        user_input = {
            "Age": int(data.get('age', 30)),
            "Gender": data.get('gender', 'Male'),
            "MaritalStatus": data.get('maritalStatus', 'Single'),
            "OverTime": data.get('overtime', 'No'),
            "MonthlyIncome": int(data.get('monthlyIncome', 5000)),
            "JobLevel": int(data.get('jobLevel', 1)),
            "JobSatisfaction": int(data.get('jobSatisfaction', 3)),
            "BusinessTravel": data.get('businessTravel', 'Travel_Rarely'),
            "Department": data.get('department', 'Sales'),
            "EducationField": data.get('educationField', 'Life Sciences'),
            "JobRole": data.get('jobRole', 'Sales Executive')
        }
        
        # Get prediction
        prediction, confidence = preprocess_and_predict(user_input, model, feature_names)
        
        # Prepare response
        result = {
            'prediction': 'Yes' if prediction == 1 else 'No',
            'confidence': round(confidence * 100, 2),
            'risk_level': 'High' if confidence > 0.7 else 'Medium' if confidence > 0.4 else 'Low'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
