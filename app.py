import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('random_forest_model.joblib')

# Streamlit app
st.title('Random Forest Model Prediction')

# Load the dataset
file_path = './coffee shop.csv'
data = pd.read_csv(file_path)

# Get feature columns
features = data.drop(columns=['Service Rating']).columns

# User input
st.sidebar.header('Input Features')
user_input = {col: st.sidebar.text_input(col) for col in features}

# Convert user input to DataFrame
user_input_df = pd.DataFrame([user_input])

# One-hot encode the user input
user_input_encoded = pd.get_dummies(user_input_df)
user_input_encoded = user_input_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

# Predict
if st.button('Predict'):
    prediction = model.predict(user_input_encoded)
    st.write(f'Predicted Service Rating: {prediction[0]}')

# Show dataset summary
st.header('Dataset Summary')
st.write(data.describe())
