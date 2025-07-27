import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Config
st.set_page_config(page_title="Boston Housing App", layout="wide")

# Load dataset
data = pd.read_csv("boston_house_prices_dataset.csv")
features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS']
X = data[features]
y = data["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load or train model
model_filename = "boston_model.pkl"
try:
    with open(model_filename, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = LinearRegression()
    model.fit(X_train, y_train)
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)

# Navigation bar
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["🏠 Home", "📊 Visualize", "💡 Predict", "📉 Performance", "ℹ️ About"],
        icons=["house", "bar-chart", "lightbulb", "graph-up", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "10px", "background-color": "#f8f9fa"},
            "icon": {"color": "#4b8bbe", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#e0f0ff",
            },
            "nav-link-selected": {"background-color": "#4b8bbe", "color": "white"},
        }
    )

# === Pages ===
if selected == "🏠 Home":
    st.title("🏠 Boston Housing Price Predictor")
    st.markdown("""
    Welcome to the **Boston Housing App**!  
    Navigate through the tabs to explore data, predict housing prices, and view model performance.
    """)

elif selected == "📊 Visualize":
    st.header("📊 Data Visualization")

    with st.expander("View Raw Data"):
        st.dataframe(data.head())

    col1, col2 = st.columns(2)
    with col1:
        selected_column = st.selectbox("Select a column for histogram", data.columns)
        fig1 = px.histogram(data, x=selected_column)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        if st.checkbox("Show Correlation Heatmap"):
            fig2, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig2)

    st.subheader("Scatter Plot")
    x_col = st.selectbox("X-axis", data.columns)
    y_col = st.selectbox("Y-axis", data.columns, index=1)
    fig3 = px.scatter(data, x=x_col, y=y_col)
    st.plotly_chart(fig3, use_container_width=True)

elif selected == "💡 Predict":
    st.header("💡 Predict House Price")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            CRIM = st.number_input("CRIM", value=0.1, step=0.01)
            ZN = st.number_input("ZN", value=18.0, step=0.1)
            INDUS = st.number_input("INDUS", value=2.31, step=0.01)
            CHAS = st.selectbox("CHAS", [0, 1])

        with col2:
            NOX = st.number_input("NOX", value=0.54, step=0.01)
            RM = st.number_input("RM", value=6.5, step=0.01)
            AGE = st.number_input("AGE", value=65.0, step=1.0)
            DIS = st.number_input("DIS", value=4.0, step=0.1)

        submit = st.form_submit_button("Predict")

        if submit:
            input_data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS]])
            prediction = model.predict(input_data)
            st.success(f"🏡 Predicted House Price: **${prediction[0]*1000:,.2f}**")

elif selected == "📉 Performance":
    st.header("📉 Model Performance")

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    st.metric("R² Score", f"{r2:.2f}")
    st.metric("Mean Squared Error", f"{mse:.2f}")

    comparison_df = pd.DataFrame({
        "Actual": y_test.values[:25],
        "Predicted": y_pred[:25]
    })

    st.subheader("Actual vs Predicted Prices")
    st.line_chart(comparison_df)
    st.bar_chart(comparison_df)

elif selected == "ℹ️ About":
    st.header("ℹ️ About This App")
    st.write("""
    This app predicts **Boston housing prices** based on 8 features using a **Linear Regression** model.

    **Technologies used**:
    - Python
    - Streamlit
    - Scikit-learn
    - Plotly & Seaborn

    **Developer**: Rishani Gunasekara  
    Email: rishhasintha@gmail.com
    """)

