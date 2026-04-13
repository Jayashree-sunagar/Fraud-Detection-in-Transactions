# =====================================================
# FRAUD DETECTION STREAMLIT APP (FINAL)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="Fraud Detection App", layout="wide")

# -----------------------------------------------------
# TITLE + SLOGAN
# -----------------------------------------------------
st.title("🔍 Fraud Detection System")
st.markdown("### Detect Fraud Before It Detects You 🚨")

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------
df = pd.read_csv("bank_transactions_data.csv")

# -----------------------------------------------------
# FIX DATE FORMAT
# -----------------------------------------------------
df['TransactionDate'] = pd.to_datetime(
    df['TransactionDate'],
    dayfirst=True,
    errors='coerce'
)

df.dropna(subset=['TransactionDate'], inplace=True)

# -----------------------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------------------
df['Hour'] = df['TransactionDate'].dt.hour

df['Amount_Range'] = pd.cut(df['TransactionAmount'],
                           bins=[0, 10, 50, 1000],
                           labels=['Low', 'Medium', 'High'])

df['High_Amount'] = (df['TransactionAmount'] > 500).astype(int)

df['FraudFlag'] = (
    (df['TransactionAmount'] > 400) |
    (df['Hour'] < 5) |
    (df['LoginAttempts'] > 3)
).astype(int)

# -----------------------------------------------------
# MODEL TRAINING
# -----------------------------------------------------
X = df[['TransactionAmount', 'Hour', 'High_Amount']]
y = df['FraudFlag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
menu = st.sidebar.radio("Menu", ["Dashboard", "Fraud Prediction", "Data View"])

# =====================================================
# DASHBOARD
# =====================================================
if menu == "Dashboard":
    st.header("📊 Dashboard Overview")

    # KPIs
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", len(df))
    col2.metric("Total Fraud", int(df['FraudFlag'].sum()))
    col3.metric("Fraud %", round(df['FraudFlag'].mean()*100, 2))

    # -------------------------------
    # SYNCHRONOUS FRAUD BAR
    # -------------------------------
    fraud_percent = df['FraudFlag'].mean() * 100

    st.subheader("🚨 Fraud Risk Indicator")

    if fraud_percent < 20:
        st.success(f"Low Risk: {fraud_percent:.2f}%")
    elif fraud_percent < 50:
        st.warning(f"Medium Risk: {fraud_percent:.2f}%")
    else:
        st.error(f"High Risk: {fraud_percent:.2f}%")

    progress_bar = st.progress(0)

    for i in range(int(fraud_percent)):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    # -------------------------------
    # FRAUD DISTRIBUTION
    # -------------------------------
    st.subheader("Fraud vs Normal Transactions")

    fig, ax = plt.subplots()
    df['FraudFlag'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel("FraudFlag")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # -------------------------------
    # AMOUNT ANALYSIS
    # -------------------------------
    st.subheader("Transaction Amount by Fraud")

    fig2, ax2 = plt.subplots()
    sns.boxplot(x='FraudFlag', y='TransactionAmount', data=df, ax=ax2)
    st.pyplot(fig2)

# =====================================================
# FRAUD PREDICTION
# =====================================================
elif menu == "Fraud Prediction":
    st.header("🔍 Predict Fraud Transaction")

    amount = st.number_input("Transaction Amount", min_value=0.0)
    hour = st.slider("Transaction Hour", 0, 23)
    high_amount = st.selectbox("High Amount (>500)", [0, 1])

    if st.button("Predict"):
        input_data = np.array([[amount, hour, high_amount]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Normal Transaction")

# =====================================================
# DATA VIEW
# =====================================================
else:
    st.header("📄 Dataset Preview")
    st.dataframe(df.head(100))