import streamlit as st
import joblib
import numpy as np

# Set page configuration
st.set_page_config(page_title="Titanic Predictor", page_icon="🚢", layout="wide")

# Load model
model = joblib.load("titanic_model.pkl")

# Custom CSS styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f4f8;
        }
        .title {
            text-align: center;
            font-size: 3em;
            color: #003366;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2em;
            color: #555;
            margin-bottom: 30px;
        }
        .stButton > button {
            background-color: #007acc;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton > button:hover {
            background-color: #005b99;
        }
        .card {
            background-color: white;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>🚢 Titanic Survival Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Enter the passenger's details to estimate their chance of survival.</div>", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("🧾 Passenger Information")

pclass = st.sidebar.selectbox("🎟️ Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("⚧️ Sex", ["Male", "Female"])
age = st.sidebar.slider("🎂 Age", 0, 80, 25)
sibsp = st.sidebar.number_input("👨‍👩‍👧 Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.sidebar.number_input("👶 Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.sidebar.number_input("💰 Fare Paid", min_value=0.0, max_value=600.0, value=30.0)
embarked = st.sidebar.selectbox("🛳️ Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"])

# Convert inputs
sex_val = 1 if sex == "Male" else 0
embarked_val = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}[embarked]

# Main layout
col1, col2 = st.columns([1, 2])

with col1:
   st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)


with col2:
    st.subheader("🔍 Prediction")

    if st.button("🚀 Predict Survival"):
        input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])
        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)[0]

        result_color = "green" if prediction[0] == 1 else "red"
        result_text = "🎉 This person would have SURVIVED!" if prediction[0] == 1 else "💀 Unfortunately, this person would NOT have survived."

        st.markdown(f"""
        <div class='card' style='background-color: {result_color};'>
            <h3>{result_text}</h3>
            <p><strong>🔢 Survival Probability:</strong> {proba[1]*100:.2f}%</p>
            <p><strong>📉 Non-survival Probability:</strong> {proba[0]*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: small;'>Made with ❤️ using Streamlit | Titanic ML Model</p>", unsafe_allow_html=True)
