import streamlit as st
import pickle
import numpy as np

# Load K-Means model and scaler
with open("kmeans_model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("Bus Passenger Clustering")

route_id = st.number_input("Enter Route ID", min_value=0, step=1)
hour = st.number_input("Enter Hour", min_value=0, max_value=23, step=1)

if st.button("Predict Cluster"):
    scaled_input = scaler.transform(np.array([[route_id, hour]]))
    cluster = kmeans.predict(scaled_input)[0]
    st.success(f"The predicted cluster is: {cluster}")
