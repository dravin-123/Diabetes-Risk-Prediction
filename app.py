import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", 
                "DiabetesPedigreeFunction", "Age", "Outcome"]

df = pd.read_csv(url, names=column_names)

# Drop unwanted columns
df.drop(columns=["SkinThickness", "Insulin", "DiabetesPedigreeFunction"], inplace=True)

# Split into features (X) and target (y)
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Split into train and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Show dataset info
print("Dataset preview:")
print(df.head())

print("\nShape of Training Data:", X_train.shape)
print("Shape of Testing Data:", X_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, "diabetes_model.pkl")
print("Model saved successfully!")


import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("diabetes_model.pkl")

# Streamlit UI
st.title("Diabetes Risk Prediction")

st.write("Enter the following details to predict your diabetes risk percentage:")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose Level", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
age = st.number_input("Age", min_value=0, step=1)

# Predict button
if st.button("Predict Risk"):
    # Convert input into a numpy array
    input_data = np.array([[pregnancies, glucose, blood_pressure, bmi, age]])
    
    # Get probability prediction
    risk_percentage = model.predict_proba(input_data)[0][1] * 100

    # Display result
    st.success(f"Your predicted diabetes risk is **{risk_percentage:.2f}%**")

# Define risk categories
def categorize_risk(risk_percentage):
    if risk_percentage < 30:
        return "Low Risk"
    elif 30 <= risk_percentage < 70:
        return "Moderate Risk"
    else:
        return "High Risk"

# Get risk category
risk_category = categorize_risk(risk_percentage)

# Display result
st.write(f"### Risk Percentage: {risk_percentage:.2f}%")
st.write(f"### Risk Category: **{risk_category}**")

# Customize output colors
if risk_category == "Low Risk":
    st.success("You have a low risk of diabetes.")
elif risk_category == "Moderate Risk":
    st.warning("You have a moderate risk of diabetes. Consider lifestyle changes.")
else:
    st.error("You have a high risk of diabetes. Consult a doctor immediately.")


