import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Parkinson's Diagnostic Dashboard", layout="wide", page_icon="🎙️")

# --- Load Models ---
@st.cache_resource
def load_models():
    scaler = joblib.load('ML_Models/scaler.pkl')
    model = joblib.load('ML_Models/svm.pkl')
    return scaler, model

try:
    scaler, model = load_models()
except Exception as e:
    st.error(f"Error loading models. Ensure 'ML_Models' contains 'scaler.pkl' and 'svm.pkl'. {e}")
    st.stop()

# --- Main App ---
st.title("🎙️ Parkinson's Disease Diagnostic Dashboard")
st.markdown("[![GitHub](https://img.shields.io/badge/Developed_by-piyushkadam96k-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/piyushkadam96k)")

tab1, tab2 = st.tabs(["🩺 Patient Diagnosis Suite", "📊 System Analytics & Graphs"])

# --- TAB 1: Patient Diagnosis ---
with tab1:
    st.markdown("Enter vocal biomarker measurements to predict the likelihood of Parkinson's Disease via the optimized Support Vector Machine.")
    
    features = [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
        'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
        'spread1', 'spread2', 'D2', 'PPE'
    ]
    
    default_values = [
        119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554, 0.01109,
        0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033,
        0.414783, 0.815285, -4.813031, 0.266482, 2.301442, 0.284654
    ]
    
    with st.form("prediction_form"):
        cols = st.columns(4) # Tighter layout for better aesthetics
        input_data = {}
        for i, feature in enumerate(features):
            col = cols[i % 4]
            input_data[feature] = col.number_input(f"{feature}", value=float(default_values[i]), format="%.5f")
            
        submit_button = st.form_submit_button(label="Analyze Vocal Pipeline", use_container_width=True)

    if submit_button:
        input_df = pd.DataFrame([input_data])
        scaled_input = scaler.transform(input_df)
        
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0]
        
        st.markdown("---")
        st.subheader("Diagnostic Engine Result:")
        
        # Display Prediction Confidence Graphically
        if prediction == 1:
            st.error("### ⚠️ Parkinson's Disease Profile Detected")
            st.metric(label="Confidence Level", value=f"{probability[1]*100:.2f}%")
            st.progress(probability[1]) # Visual Progress Bar
            st.info("Vocal biomarkers strongly correlate with known Parkinson's pathology. Clinical follow-up recommended.")
        else:
            st.success("### ✅ Healthy Profile (No Parkinson's Detected)")
            st.metric(label="Confidence Level", value=f"{probability[0]*100:.2f}%")
            st.progress(probability[0]) # Visual Progress Bar
            st.info("The provided vocal metrics fall into the healthy baseline spectrum.")

# --- TAB 2: Model Analytics (Graphs) ---
with tab2:
    st.header("Comparative Model Performance Graphs")
    st.markdown("Review how our Support Vector Machine stacks up against other algorithms tested during development.")
    
    try:
        metrics_df = pd.read_csv('model_metrics_comparison.csv')
        
        # We need Model as index for Streamlit's native bar chart
        chart_data = metrics_df.set_index('Model')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Accuracy Comparison")
            # Interactive Bar Chart
            st.bar_chart(chart_data[['Accuracy']], color="#ff4b4b")
            
        with col2:
            st.subheader("F1-Score Medical Validation")
            # Interactive Bar Chart
            st.bar_chart(chart_data[['F1-Score']], color="#0068c9")
            
        st.subheader("Raw Metric Leaderboard")
        st.dataframe(metrics_df, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Metrics file not found. Please run `main.py` first to generate model performance graphs.")
