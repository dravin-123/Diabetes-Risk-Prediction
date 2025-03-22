import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
                "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pd.read_csv(url, names=column_names)

# Drop unnecessary columns
df.drop(columns=["SkinThickness", "Insulin", "DiabetesPedigreeFunction"], inplace=True)

# Split data
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "diabetes_model.pkl")
print("Model saved successfully!")

# Load model
model = joblib.load("diabetes_model.pkl")

# Streamlit App
st.title("Diabetes Risk Prediction")

st.write("Enter the following details to predict your diabetes risk percentage:")

# User Input
pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
age = st.number_input("Age", min_value=0, step=1)

if st.button("Predict Risk"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, bmi, age]])
    
    # Predict Probability
    risk_percentage = model.predict_proba(input_data)[0][1] * 100

    # Risk Category Function
    def categorize_risk(risk_percentage):
        if risk_percentage < 30:
            return "Low Risk"
        elif 30 <= risk_percentage < 70:
            return "Moderate Risk"
        else:
            return "High Risk"

    risk_category = categorize_risk(risk_percentage)

    # Display Results
    st.write(f"### Risk Percentage: {risk_percentage:.2f}%")
    st.write(f"### Risk Category: **{risk_category}**")

    # Recommendation Messages
    if risk_category == "Low Risk":
        st.success("You have a low risk of diabetes.")
    elif risk_category == "Moderate Risk":
        st.warning("You have a moderate risk of diabetes. Consider lifestyle changes.")
    else:
        st.error("You have a high risk of diabetes. Consult a doctor immediately.")
