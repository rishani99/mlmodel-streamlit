import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("boston_house_prices_dataset.csv")

# Define the 8 features your model expects
features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS']

# Prepare data with only these features
X = data[features]
y = data["MEDV"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model if not exists
model_filename = "boston_model.pkl"

try:
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
    st.write("Loaded model from disk.")
except FileNotFoundError:
    st.write("Model not found, training a new Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    st.write("Model trained and saved.")

st.title("Boston Housing Price Predictor")

# 1. Data Visualisation Section
st.header("Data Visualisation")

if st.checkbox("Show Raw Data"):
    st.write(data.head())

selected_column = st.selectbox("Select a column for histogram", data.columns)
fig1 = px.histogram(data, x=selected_column)
st.plotly_chart(fig1)

if st.checkbox("Show Correlation Heatmap"):
    fig2, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig2)

x_col = st.selectbox("X-axis column", data.columns, index=0)
y_col = st.selectbox("Y-axis column", data.columns, index=1)
fig3 = px.scatter(data, x=x_col, y=y_col)
st.plotly_chart(fig3)

# 2. Predict House Price Section
st.header("Predict House Price")

# Input fields for the 8 features
CRIM = st.number_input("CRIM (per capita crime rate)", value=0.1, step=0.01)
ZN = st.number_input("ZN (proportion of residential land zoned)", value=18.0, step=0.1)
INDUS = st.number_input("INDUS (proportion of non-retail business acres)", value=2.31, step=0.01)
CHAS = st.number_input("CHAS (Charles River dummy variable, 0 or 1)", min_value=0, max_value=1, step=1, value=0)
NOX = st.number_input("NOX (nitric oxides concentration)", value=0.54, step=0.01)
RM = st.number_input("RM (average number of rooms per dwelling)", value=6.5, step=0.01)
AGE = st.number_input("AGE (proportion of owner-occupied units built prior to 1940)", value=65.0, step=1.0)
DIS = st.number_input("DIS (weighted distances to employment centres)", value=4.0, step=0.1)

if st.button("Predict Price"):
    input_data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS]])
    prediction = model.predict(input_data)
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

# 3. Model Performance Section
st.header("Model Performance")

# Check shapes and feature counts before prediction
st.write(f"Model expects {model.n_features_in_} features.")
st.write(f"Test set shape: {X_test.shape}")

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

st.metric("R² Score", f"{r2:.2f}")
st.metric("Mean Squared Error", f"{mse:.2f}")

comparison_df = pd.DataFrame({"Actual": y_test[:25].values, "Predicted": y_pred[:25]})
st.line_chart(comparison_df)
st.subheader("Actual vs Predicted Prices")
st.write(comparison_df)
st.bar_chart(comparison_df)
st.subheader("Actual vs Predicted Prices (Bar Chart)")
# 4. Model Information Section
st.header("Model Information")
st.write("This model predicts the median value of owner-occupied homes in $1000s based on 8 features from the Boston Housing dataset.")
st.write("The model was trained using a Linear Regression algorithm and is saved as `boston_model.pkl`.")

# 5. About Section
st.header("About")
st.write("This Streamlit app allows users to visualize the Boston Housing dataset, predict house prices, and evaluate model performance."       )
st.write("Developed by [Rishani Gunasekara].")