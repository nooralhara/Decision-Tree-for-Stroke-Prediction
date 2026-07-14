import streamlit as st 
import joblib 
import numpy as np 

@st.cache_resource
def load_assets():
    model = joblib.load('my_model.pkl')
    scaler = joblib .load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()
    

st.title('Cerebral Stroke Prediction Using Decision Tree')
st.write("This model predicts whether a patient has a possibility of getting a cerebral stroke using a decision tree model.")

gender = st.selectbox('Gender', options=[0, 1, 2],format_func=lambda x: ['Female', 'Male', 'Other'][x])
age = st.number_input('Age', value=30)
hypertension = st.selectbox('Hypertension', [0, 1])
heart_disease = st.selectbox('Heart Disease', [0, 1])
ever_married = st.selectbox('Ever Married', [0, 1])
work_type = st.selectbox('Work Type', [0, 1, 2, 3, 4])
residence = st.selectbox('Residence Type', [0, 1])
glucose = st.number_input('Avg Glucose Level', value=100.0)
bmi = st.number_input('BMI', value=25.0)
smoking = st.selectbox('Smoking Status', [0, 1, 2])

if st.button('Predict'):
    input_data = np.array([[gender,age,hypertension,heart_disease,ever_married,work_type,residence,glucose,bmi,smoking]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    st.success(f"The model predicted: {prediction[0]}")