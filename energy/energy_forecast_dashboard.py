import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt

# Set layout
st.set_page_config(layout="wide")

# Load energy dataset
@st.cache_data
def load_data():
    csv_path = "/Users/arberxhauri/PycharmProjects/Test/energy/energy_data.csv"
    if not os.path.exists(csv_path):
        st.error("Missing file: energy_data.csv")
        st.stop()
    return pd.read_csv(csv_path, index_col="Year")

df = load_data()

# Sidebar
st.sidebar.title("Energy Explorer")
item_options = sorted(df.columns)
selected_item = st.sidebar.selectbox("Select Energy Metric", item_options)
model_types = ["LinearRegression", "RandomForest", "SVR"]
selected_model = st.sidebar.selectbox("Select Prediction CSV", model_types)
predict_year = st.sidebar.number_input("Predict Year", min_value=2025, max_value=2030, value=2030)

# Title
st.title("Energy Forecast Dashboard")
st.write(f"### Forecasting '{selected_item}' using {selected_model}")

# Build CSV path
prediction_filename = f"{selected_item.replace(' ', '_').replace('.', '')}_{selected_model}.csv"
prediction_path = os.path.join("/Users/arberxhauri/PycharmProjects/Test/energy/model_predictions", prediction_filename)

# Load prediction data
if os.path.exists(prediction_path):
    pred_df = pd.read_csv(prediction_path)

    if predict_year in pred_df["Year"].values:
        predicted_value = pred_df.loc[pred_df["Year"] == predict_year, "Prediction"].values[0]
        st.success(f"Predicted value for {predict_year}: {predicted_value:,.2f} MWh")
    else:
        st.warning(f"No prediction available for year {predict_year}")

    # Plot actual + predicted
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df[selected_item], label="Actual Data", marker='o')
    ax.plot(pred_df["Year"], pred_df["Prediction"], label=f"{selected_model} Forecast", linestyle="--", marker='x')
    ax.axvline(predict_year, color='gray', linestyle='--')
    if 'predicted_value' in locals():
        ax.scatter([predict_year], [predicted_value], color='red', label='Selected Prediction')
    ax.set_title(f"{selected_item} Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("Energy (MWh)")
    ax.legend()
    st.pyplot(fig)
else:
    st.error(f"Prediction CSV not found: {prediction_path}")