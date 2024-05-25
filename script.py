import streamlit as st
import pandas as pd
import numpy as np
import warnings
import joblib


warnings.filterwarnings("ignore")

st.set_page_config(page_title="Deposit Prediction App", layout="wide", page_icon="🏦")

# Header section
st.title('Deposit Prediction Web App 🏦')
st.markdown("## Please fill out the details below to predict deposit subscription:")

# Divide the page into three columns for better alignment of inputs
col1, col2, col3 = st.columns([1, 1.5, 1])

with col1:
    age = st.number_input('Age', min_value=16, max_value=100, step=1, help="Enter your age")
    job = st.selectbox('Job', ('admin','blue-collar','entrepreneur', 'housemaid','management','retired','self-employed', 'services','student','technician','unemployed','unknown'))
    marital = st.selectbox('Marital Status', ('married', 'single', 'divorced'))
    education = st.selectbox('Education', ('basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'))

with col2:
    default = st.selectbox('Default', ('yes', 'no', 'unknown'))
    housing = st.selectbox('Housing', ('yes', 'no', 'unknown'))
    loan = st.selectbox('Loan', ('yes', 'no', 'unknown'))
    contact = st.selectbox('Contact', ('cellular', 'telephone'))
    month = st.selectbox('Month', ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'))
    day_of_week = st.selectbox('Day of Week', ('mon', 'tue', 'wed', 'thu', 'fri'))

with col3:
    duration = st.number_input('Duration', min_value=0, step=1)
    campaign = st.number_input('Campaign', min_value=0, step=1)
    pdays = st.number_input('PDays', min_value=0, step=1)
    previous = st.number_input('Previous', min_value=0, step=1)
    poutcome = st.selectbox('POutcome', ('success', 'failure', 'nonexistent'))
    emp_var_rate = st.number_input('Employment variation rate', format="%.2f")
    cons_price_idx = st.number_input('Consumer price index', format="%.2f")
    cons_conf_idx = st.number_input('Consumer confidence index', format="%.2f")
    euribo3m = st.number_input('Euribor 3 month rate', format="%.2f")
    nr_employed = st.number_input('Number of employees', format="%.2f")

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'default': [default],
    'housing': [housing],
    'loan': [loan],
    'contact': [contact],
    'month': [month],
    'day_of_week': [day_of_week],
    'duration': [duration],
    'campaign': [campaign],
    'pdays': [pdays],
    'previous': [previous],
    'poutcome': [poutcome],
    'emp.var.rate': [emp_var_rate],
    'cons.price.idx': [cons_price_idx],
    'cons.conf.idx': [cons_conf_idx],
    'euribor3m': [euribo3m],
    'nr.employed': [nr_employed]
})

# Load the model using Joblib

# Load the model using Joblib
try:
    model = joblib.load('tuned_best_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please ensure the 'tuned_best_model.pkl' file is in the correct directory.")
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")


def predict_display():
    input_predictions = model.predict(input_data)
    if input_predictions == "no":
        st.error('No! Client is NOT subscribed to a term deposit. ❌')
    else:
        st.success('YES! Client is subscribed to a term deposit. ✅')

# Prediction button
if st.button('Predict Subscription'):
    predict_display()
