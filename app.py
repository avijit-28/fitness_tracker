import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained SVM model
with open("svm_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset for insights
df = pd.read_csv("fitness.csv")

# Streamlit UI Configuration
st.set_page_config(page_title="Personal Fitness Tracker", layout="centered")

st.title("🏋️‍♂️ Personal Fitness Tracker")
st.subheader("Predict Calories Burned Based on Your Fitness Data")

# Sidebar for user inputs
st.sidebar.header("Enter Your Details")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
duration = st.sidebar.number_input("Exercise Duration (min)", min_value=1, max_value=300, value=30)
heart_rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=50, max_value=200, value=120)
body_temp = st.sidebar.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)

# Convert gender to numeric
gender_numeric = 1 if gender == "Male" else 0

# Predict button
if st.sidebar.button("Predict Calories Burned"):
    input_data = np.array([[age, height, weight, duration, heart_rate, body_temp, gender_numeric]])
    prediction = model.predict(input_data)[0]

    # Insights based on dataset statistics
    age_percentile = (df["Age"] < age).mean() * 100
    duration_percentile = (df["Duration"] < duration).mean() * 100
    heart_rate_percentile = (df["Heart_Rate"] < heart_rate).mean() * 100
    body_temp_percentile = (df["Body_Temp"] < body_temp).mean() * 100

    # Display prediction in a styled table
    result_df = pd.DataFrame({
        "Parameter": ["Age", "Height (cm)", "Weight (kg)", "Duration (min)", "Heart Rate (bpm)", "Body Temperature (°C)", "Gender", "Predicted Calories Burned"],
        "Value": [age, height, weight, duration, heart_rate, body_temp, gender, f"{prediction:.2f} kcal"]
    })

    st.markdown("### 🔥 Prediction Results")
    st.table(result_df.style.set_table_styles([{'selector': 'thead th', 'props': [('background-color', '#4CAF50'), ('color', 'white')]}]))

    # Display insights
    st.markdown("### 📊 Personalized Insights")
    insights = [
        f"📌 You are older than **{age_percentile:.1f}%** of other people.",
        f"📌 Your exercise duration is higher than **{duration_percentile:.1f}%** of other people.",
        f"📌 Your heart rate is higher than **{heart_rate_percentile:.1f}%** of other people during exercise.",
        f"📌 Your body temperature is higher than **{body_temp_percentile:.1f}%** of other people during exercise."
    ]
    
    for insight in insights:
        st.write(insight)

# Footer
st.markdown("---")
st.markdown("🚀 **Developed by Avijit Pakhira**")