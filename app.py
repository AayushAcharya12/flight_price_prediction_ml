import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="Flight Price Prediction ", layout="centered")
st.title("Flight Price Prediction ")
st.subheader("Enter flight details and get the predicted price!")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


models = {
    "Linear Regression": joblib.load(os.path.join(BASE_DIR, "models", "linear_regression.pkl")),
    "Random Forest": joblib.load(os.path.join(BASE_DIR, "models", "random_forest.pkl")),
    "XGBoost": joblib.load(os.path.join(BASE_DIR, "models", "xgboost.pkl"))
}


columns = joblib.load(os.path.join(BASE_DIR, "models", "preprocessed_columns.pkl"))


selected_model_name = st.selectbox("Choose a model:", list(models.keys()))
selected_model = models[selected_model_name]


st.header("Flight Details")


total_stops = st.number_input("Total Stops", min_value=0, max_value=4, value=1)
journey_day = st.number_input("Journey Day", min_value=1, max_value=31, value=15)
journey_month = st.number_input("Journey Month", min_value=1, max_value=12, value=6)

dep_hour = st.number_input("Departure Hour", min_value=0, max_value=23, value=10)
dep_min = st.number_input("Departure Minute", min_value=0, max_value=59, value=30)

arr_hour = st.number_input("Arrival Hour", min_value=0, max_value=23, value=13)
arr_min = st.number_input("Arrival Minute", min_value=0, max_value=59, value=45)

duration_hour = st.number_input("Duration Hours", min_value=0, max_value=30, value=3)
duration_min = st.number_input("Duration Minutes", min_value=0, max_value=59, value=15)


airline_options = ["Air India", "GoAir", "IndiGo", "Jet Airways", "Jet Airways Business", "SpiceJet", "Vistara", "Multiple carriers"]
selected_airline = st.selectbox("Select Airline", airline_options)

source_options = ["Delhi", "Kolkata", "Mumbai", "Chennai", "Banglore"]
selected_source = st.selectbox("Select Source", source_options)

destination_options = ["Cochin", "Banglore", "Delhi", "New Delhi", "Hyderabad", "Kolkata", "Mumbai"]
selected_destination = st.selectbox("Select Destination", destination_options)


if st.button("Predict Price"):

    
    user_input = {
        "Total_Stops": total_stops,
        "Journey_day": journey_day,
        "Journey_month": journey_month,
        "Dep_hour": dep_hour,
        "Dep_min": dep_min,
        "Arrival_hour": arr_hour,
        "Arrival_min": arr_min,
        "Duration_hours": duration_hour,
        "Duration_mins": duration_min
    }

  
    for airline in airline_options:
        user_input[f"Airline_{airline}"] = 1 if selected_airline == airline else 0


    for src in source_options:
        user_input[f"Source_{src}"] = 1 if selected_source == src else 0

    for dst in destination_options:
        user_input[f"Destination_{dst}"] = 1 if selected_destination == dst else 0

    for col in columns:
        if col not in user_input:
            user_input[col] = 0

    input_df = pd.DataFrame([user_input])
    input_df = input_df[columns]
    if selected_model_name == "Linear Regression":
     log_pred = selected_model.predict(input_df)[0]
     price = np.expm1(log_pred)  
    else:
        price = selected_model.predict(input_df)[0]

    st.success(f"Predicted Price using {selected_model_name}:  {round(price, 2)}")
