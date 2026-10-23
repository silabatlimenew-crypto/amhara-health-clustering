import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Amhara Region Healthcare Clustering",
    page_icon="🏥",
    layout="wide"
)

# Load Saved Artifacts securely
@st.cache_resource
def load_artifacts():
    model = joblib.load('kmeans_health_model.pkl')
    scaler = joblib.load('health_scaler.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}. Make sure the .pkl files are uploaded to Colab.")
    st.stop()

# App Header
st.title("🏥 Amhara Region Zonal Healthcare Capacity & Priority Predictor")
st.markdown("""
This application uses a trained **K-Means Machine Learning model** to classify any zone or town in the Amhara Region based on its malaria burden, health facility infrastructure, and key health indicators.
""")

st.sidebar.header("📊 Enter Zone Healthcare Indicators")

# Dynamically generate input fields based on the training features
input_data = {}
for col in feature_columns:
    input_data[col] = st.sidebar.number_input(
        label=f"{col.replace('_', ' ')}", 
        value=0.0, 
        step=1.0,
        format="%.2f"
    )

# Prediction Button
if st.sidebar.button("Predict Zonal Cluster", type="primary"):
    input_df = pd.DataFrame([input_data])[feature_columns]
    input_scaled = scaler.transform(input_df)
    cluster_pred = int(model.predict(input_scaled)[0])
    
    st.divider()
    st.subheader(f"🎯 Prediction Result: Cluster {cluster_pred}")
    
    if cluster_pred == 0:
        st.success("### **Cluster 0: Moderate Activity & Good Infrastructure**")
        st.write("Moderate levels of malaria activity and hospitalizations, supported by a solid foundation of health infrastructure.")
    elif cluster_pred == 1:
        st.info("### **Cluster 1: Lower Incidence & Reduced Facility Load**")
        st.write("Regions showing lower malaria incidence and proportionally lighter health facility infrastructure.")
    elif cluster_pred == 2:
        st.warning("### **Cluster 2: High Disease Burden & Heavy Facility Strain**")
        st.write("Areas severely impacted by significantly higher malaria cases and hospitalizations. **High Priority Intervention.**")
    elif cluster_pred == 3:
        st.info("### **Cluster 3: Localized High-Risk Profile (e.g., Awi-style profile)**")
        st.write("A distinct profile characterized by heavy disease incidence metrics alongside critical maternal health indicators.")
        
    with st.expander("View Submitted Zone Data Profile"):
        st.dataframe(input_df)
else:
    st.info("👈 Enter the indicators in the sidebar and click **'Predict Zonal Cluster'** to see the classification.")
