import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="wide")

st.title("📉 Customer Churn Prediction System")
st.markdown("**Sqrock IT Solutions Internship — Project 2** | Built with Random Forest Classifier")
st.markdown("---")

uploaded_file = st.file_uploader("Upload Telco Customer Churn CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset loaded! {df.shape[0]} rows, {df.shape[1]} columns")

    # Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{df.shape[0]:,}")
    col2.metric("Churned", f"{df['Churn'].sum():,}")
    col3.metric("Churn Rate", f"{df['Churn'].mean()*100:.1f}%")

    # EDA
    st.subheader("📊 Exploratory Data Analysis")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    df['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%',
        labels=['No Churn','Churned'], colors=['steelblue','coral'], ax=axes[0])
    axes[0].set_title('Churn Distribution')

    df.groupby('Churn')['tenure'].mean().plot(kind='bar',
        color=['steelblue','coral'], ax=axes[1])
    axes[1].set_title('Avg Tenure vs Churn')
    axes[1].set_xlabel('Churn')
    axes[1].set_ylabel('Avg Tenure (months)')

    df.groupby('Churn')['MonthlyCharges'].mean().plot(kind='bar',
        color=['steelblue','coral'], ax=axes[2])
    axes[2].set_title('Avg Monthly Charges vs Churn')
    axes[2].set_xlabel('Churn')
    axes[2].set_ylabel('Monthly Charges ($)')

    plt.tight_layout()
    st.pyplot(fig)

    # Heatmap
    st.subheader("🔥 Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax2)
    plt.tight_layout()
    st.pyplot(fig2)

    # Model
    st.subheader("🤖 Model Training & Evaluation")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = round(accuracy_score(y_test, preds) * 100, 2)
    report = classification_report(y_test, preds, output_dict=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{acc}%")
    c2.metric("Precision", f"{round(report['weighted avg']['precision']*100,1)}%")
    c3.metric("Recall",    f"{round(report['weighted avg']['recall']*100,1)}%")
    c4.metric("F1 Score",  f"{round(report['weighted avg']['f1-score']*100,1)}%")

    # Confusion Matrix
    st.subheader("📋 Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn','Churned'],
                yticklabels=['No Churn','Churned'], ax=ax3)
    ax3.set_ylabel('Actual')
    ax3.set_xlabel('Predicted')
    plt.tight_layout()
    st.pyplot(fig3)

    # Feature Importance
    st.subheader("⭐ Top 10 Features Affecting Churn")
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    feat_imp.plot(kind='bar', color='steelblue', ax=ax4)
    ax4.set_ylabel('Importance Score')
    plt.tight_layout()
    st.pyplot(fig4)

    # Predict single customer
    st.subheader("🔮 Predict Single Customer Churn")
    st.markdown("Adjust the sliders to predict if a customer will churn:")

    col_a, col_b, col_c = st.columns(3)
    tenure = col_a.slider("Tenure (months)", 0, 72, 12)
    monthly = col_b.slider("Monthly Charges ($)", 18, 120, 65)
    total = col_c.slider("Total Charges ($)", 0, 9000, 1500)

    if st.button("Predict Churn 🚀"):
        sample = X_test.iloc[0].copy()
        sample['tenure'] = tenure
        sample['MonthlyCharges'] = monthly
        sample['TotalCharges'] = total
        result = model.predict([sample])[0]
        prob = model.predict_proba([sample])[0][1]
        if result == 1:
            st.error(f"⚠️ This customer is likely to CHURN! (Probability: {prob*100:.1f}%)")
        else:
            st.success(f"✅ This customer is likely to STAY! (Churn Probability: {prob*100:.1f}%)")

else:
    st.info("👆 Please upload the Telco Customer Churn CSV file to get started.")
    st.markdown("""
    **How to use:**
    1. Download dataset from Kaggle: *Telco Customer Churn*
    2. Upload the CSV file above
    3. App will train the model and show all results
    """)
