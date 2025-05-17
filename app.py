# app.py

import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.title("💳 Financial Transaction Anomaly Detection")

uploaded_file = st.file_uploader("Upload your CSV transaction file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("📊 Uploaded Data Sample:")
    st.dataframe(data.head())

    if 'Class' in data.columns:
        X = data.drop(['Class'], axis=1)
    else:
        X = data

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(contamination=0.001)
    predictions = model.fit_predict(X_scaled)
    data['Anomaly'] = ['Fraud' if x == -1 else 'Normal' for x in predictions]

    st.write("✅ Prediction Results:")
    st.dataframe(data)

    frauds = data[data['Anomaly'] == 'Fraud']
    st.write(f"🚨 Number of Fraudulent Transactions Detected: {len(frauds)}")

    st.download_button("Download Results", data.to_csv(index=False), "fraud_results.csv")
