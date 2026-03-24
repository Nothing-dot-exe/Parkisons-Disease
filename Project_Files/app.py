import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from fpdf import FPDF
import tempfile
from datetime import datetime

def generate_pdf_report(name, age, gender, input_data, prediction, confidence):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Parkinson's Disease Assessment Report", ln=1, align='C')
    pdf.ln(5)
    
    # Date
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align='R')
    pdf.ln(5)
    
    # Patient Info
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Patient Information", ln=1)
    pdf.set_font("Arial", '', 11)
    pdf.cell(200, 8, txt=f"Name: {name}", ln=1)
    pdf.cell(200, 8, txt=f"Age: {str(age)}", ln=1)
    pdf.cell(200, 8, txt=f"Gender: {gender}", ln=1)
    pdf.ln(5)
    
    # Diagnosis Result
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Diagnostic Result", ln=1)
    pdf.set_font("Arial", '', 11)
    result_text = "Parkinson's Disease Profile Detected" if prediction == 1 else "Healthy Profile (No Parkinson's Detected)"
    pdf.cell(200, 8, txt=f"Diagnosis: {result_text}", ln=1)
    pdf.cell(200, 8, txt=f"Confidence Level: {confidence:.2f}%", ln=1)
    pdf.ln(5)
    
    # Input Biomarkers
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Vocal Biomarker Measurements", ln=1)
    pdf.set_font("Arial", '', 10)
    for k, v in input_data.items():
        pdf.cell(90, 6, txt=f"{k}:", ln=0)
        pdf.cell(100, 6, txt=f"{v}", ln=1)
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        tmp.seek(0)
        pdf_bytes = tmp.read()
    
    # Ensure file handle is closed in Windows before delete
    try:
        os.remove(tmp.name)
    except:
        pass
    
    return pdf_bytes

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


tab1, tab2 = st.tabs(["🩺 Patient Diagnosis Suite", "📊 System Analytics & Graphs"])

# --- TAB 1: Patient Diagnosis ---
with tab1:
    st.markdown("Enter the patient's demographics and vocal biomarker measurements.")
    
    st.subheader("👤 Patient Information")
    col_p1, col_p2, col_p3 = st.columns(3)
    patient_name = col_p1.text_input("Patient Name", "John Doe")
    patient_age = col_p2.number_input("Age", min_value=1, max_value=120, value=65)
    patient_gender = col_p3.selectbox("Gender", ["Male", "Female", "Other"])
    
    st.markdown("---")
    st.subheader("🎙️ Vocal Biomarkers")
    
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
        confidence = probability[prediction] * 100
        
        st.markdown("---")
        st.subheader("Diagnostic Engine Result:")
        
        # Display Prediction Confidence Graphically
        if prediction == 1:
            st.error("### ⚠️ Parkinson's Disease Profile Detected")
            st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
            st.progress(probability[1]) # Visual Progress Bar
            st.info("Vocal biomarkers strongly correlate with known Parkinson's pathology. Clinical follow-up recommended.")
        else:
            st.success("### ✅ Healthy Profile (No Parkinson's Detected)")
            st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
            st.progress(probability[0]) # Visual Progress Bar
            st.info("The provided vocal metrics fall into the healthy baseline spectrum.")
            
        # PDF Generation and Download
        pdf_bytes = generate_pdf_report(patient_name, patient_age, patient_gender, input_data, prediction, confidence)
        st.download_button(
            label="📄 Download Assessment Report (PDF)",
            data=pdf_bytes,
            file_name=f"Parkinsons_Report_{patient_name.replace(' ', '_')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

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
        
        st.markdown("---")
        st.header("Exploratory Data Analysis")
        st.markdown("Below are additional visual insights extracted directly from the Parkinson's dataset:")
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if os.path.exists("graphs/target_distribution.png"):
                st.image("graphs/target_distribution.png", caption="Target Variable Distribution")
        with col_g2:
            if os.path.exists("graphs/correlation_heatmap.png"):
                st.image("graphs/correlation_heatmap.png", caption="Highly Correlated Features")
        
        if os.path.exists("graphs/model_performance_comparison.png"):
            st.subheader("Model Performance Comparison (Alternative View)")
            st.image("graphs/model_performance_comparison.png", caption="Algorithm F1-Score Comparison")
    except FileNotFoundError:
        st.warning("Metrics file not found. Please run `main.py` first to generate model performance graphs.")
