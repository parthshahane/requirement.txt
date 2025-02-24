import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit UI
st.title("Titanic Survival Prediction")
st.write("Enter the passenger details to predict survival.")

# User input fields
pclass = st.selectbox("Passenger Class (1 = First, 2 = Second, 3 = Third)", [1, 2, 3])
sex = st.selectbox("Sex (0 = Male, 1 = Female)", [0, 1])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=32.0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
embarked_q = st.selectbox("Embarked at Queenstown? (0 = No, 1 = Yes)", [0, 1])
embarked_s = st.selectbox("Embarked at Southampton? (0 = No, 1 = Yes)", [0, 1])

# Prepare the input data
input_data = np.array([[pclass, sex, age, fare, sibsp, parch, embarked_q, embarked_s]])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make a prediction
if st.button("Predict Survival"):
    prediction = model.predict(input_data_scaled)[0]
    probability = model.predict_proba(input_data_scaled)[0][1]

    if prediction == 1:
        st.success(f"The passenger **survives** with a probability of {probability:.2f}")
    else:
        st.error(f"The passenger **does not survive** with a probability of {1 - probability:.2f}")
