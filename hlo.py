import streamlit as st
import pickle
import numpy as np

# Load saved model
model = pickle.load(open("disease_model.pkl", "rb"))

# Title
st.title("Disease Prediction System")

# User inputs
itching = st.selectbox("Itching", [0,1])
skin_rash = st.selectbox("Skin Rash", [0,1])
vomiting = st.selectbox("Vomiting", [0,1])

# Prediction button
if st.button("Predict"):

    input_data = np.array([[itching, skin_rash, vomiting]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Disease: {prediction[0]}")


