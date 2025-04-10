import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load pivoted salary data
@st.cache_data
def load_salary_data():
    return pd.read_csv("/Users/arberxhauri/PycharmProjects/Test/salary/salaries_data.csv", index_col="Period", parse_dates=True)

# Load prediction CSVs
@st.cache_data
def load_predictions(profession, model):
    safe_name = profession.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace(".", "")
    filename = f"{safe_name}_{model}.csv"
    filepath = os.path.join("/Users/arberxhauri/PycharmProjects/Test/salary/salary_predictions", filename)
    if os.path.exists(filepath):
        return pd.read_csv(filepath, parse_dates=["Period"])
    return None

# Streamlit UI
st.set_page_config(layout="wide")
st.title("💼 Average Salary Forecast by Profession")

# Load and filter data
df = load_salary_data()
profession_options = sorted(df.columns)
selected_profession = st.sidebar.selectbox("Select Profession", profession_options)
model_options = ["LinearRegression", "RandomForest", "SVR"]
selected_model = st.sidebar.selectbox("Select Forecasting Model", model_options)

# Plot historical data
st.subheader(f"📊 Historical Salaries for: {selected_profession}")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df[selected_profession], label="Historical", marker='o')
ax.set_xlabel("Period")
ax.set_ylabel("Salary (Lekë)")
ax.set_title(f"Salary Trend for {selected_profession}")
ax.legend()
st.pyplot(fig)

# Load and plot predictions
prediction_df = load_predictions(selected_profession, selected_model)
if prediction_df is not None:
    st.subheader(f"📈 Forecast using {selected_model}")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df.index, df[selected_profession], label="Historical", marker='o')
    ax2.plot(prediction_df["Period"], prediction_df["Prediction"], label="Forecast", linestyle="--", marker='x')
    ax2.set_xlabel("Period")
    ax2.set_ylabel("Predicted Salary (Lekë)")
    ax2.set_title(f"Forecast for {selected_profession} ({selected_model})")
    ax2.legend()
    st.pyplot(fig2)
else:
    st.warning("⚠️ Forecast file not found for the selected profession and model.")