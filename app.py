import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📊",
)

# Title with Dark Orange Color
st.markdown(
    "<h1 style='color:darkorange; text-align:center;'>Customer Churn Prediction App</h1>",
    unsafe_allow_html=True
)

# Upload the dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    # Handle missing values
    if data.isnull().sum().sum() > 0:
        st.warning("Dataset contains missing values. These will be dropped.")
        data = data.dropna()

    # Select only important columns
    st.subheader("Selecting Important Columns")
    important_columns = ["gender", "SeniorCitizen", "Partner", "tenure", "MonthlyCharges", "Churn"]  # Example columns
    data = data[important_columns]

    st.write("Dataset with Important Columns:")
    st.write(data.head())

    # Encode categorical columns
    st.subheader("Data Preprocessing")
    st.write("Encoding categorical variables...")

    le = LabelEncoder()

    # List of categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

    # Ensure target column 'Churn' exists
    if 'Churn' not in data.columns:
        st.error("The dataset must include a 'Churn' column as the target variable.")
    else:
        # Encode categorical variables
        for col in categorical_columns:
            data[col] = le.fit_transform(data[col].astype(str))

        st.write("Encoded Dataset:")
        st.write(data.head())

        # Define features (X) and target (y)
        X = data.drop("Churn", axis=1)  # Features
        y = data["Churn"]  # Target variable

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        st.subheader("Training the Model")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save the trained model
        with open("churn_model.pkl", "wb") as file:
            pickle.dump(model, file)

        st.success("Model trained and saved successfully!")

        # Prediction section
        st.subheader("Make a Prediction")
        st.write("Enter customer details below to predict churn:")

        # Create input fields dynamically based on features
        user_inputs = {}
        for col in X.columns:
            user_inputs[col] = st.text_input(f"Enter value for {col}:")

        # Predict churn based on user input
        if st.button("Predict Churn"):
            try:
                # Convert user inputs into a dataframe
                input_data = pd.DataFrame([user_inputs])
                input_data = input_data.astype(float)  # Ensure all inputs are numerical
                prediction = model.predict(input_data)
                churn_result = "Yes" if prediction[0] == 1 else "No"
                st.write(f"Prediction: Customer will churn: {churn_result}")
            except Exception as e:
                st.error(f"Error in prediction: {e}")
