import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

## Load the trained model
import os
model_path = os.path.join(os.path.dirname(__file__), 'regression_model.h5')
model = tf.keras.models.load_model(model_path)

## Load the encoder and scaler





# Load label encoder
with open(os.path.join(os.path.dirname(__file__), 'label_encoder_gender.pkl'), 'rb') as f:
    label_encoder_gender = pickle.load(f)

# Load one-hot encoder for geography
with open(os.path.join(os.path.dirname(__file__), 'onehot_encoder_geo.pkl'), 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

# Load scaler
with open(os.path.join(os.path.dirname(__file__), 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Load model
model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), 'regression_model.h5'))

## streamlit app
st.title("Estimated Salary Prediction")


#User Input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
exited = st.selectbox('Exited', [0,1])
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

# Prepare the input data
input_data = pd.DataFrame({
  'CreditScore' : [credit_score],
  'Gender' : [label_encoder_gender.transform([gender])[0]],
  'Age' : [age],
  'Tenure' : [tenure],
  'Balance' : [balance],
  'NumOfProducts' : [num_of_products],
  'HasCrCard' : [has_cr_card],
  'IsActiveMember' : [is_active_member],
  'Exited' : [exited]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict estimated salary
prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]

st.write(f"Predicted Estimated Salary: ${predicted_salary:.2f}")


    