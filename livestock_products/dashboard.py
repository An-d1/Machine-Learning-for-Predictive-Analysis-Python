import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(layout="wide")
st.title("📈 Prodhimet Blegtorale Dashboard")

# Auto-generate data if missing
if not os.path.exists("cleaned_production_data.csv") or not os.path.exists("forecast_all.csv") or not os.path.exists("clusters_pca.csv"):
    import generate_data

# Load data
df = pd.read_csv("cleaned_production_data.csv")
forecast_df = pd.read_csv("forecast_all.csv")
clusters_df = pd.read_csv("clusters_pca.csv")

# Sidebar filters
with st.sidebar:
    st.header("🔍 Filters")
    selected_product = st.selectbox("Select Product", sorted(df["Product"].unique()))
    selected_region = st.selectbox("Select Region", sorted(df["Region"].unique()))
    year_range = st.slider("Year Range", min_value=int(df["Year"].min()), max_value=int(df["Year"].max()), value=(2010, 2023))

# Filter data
df_filtered = df[
    (df["Product"] == selected_product) &
    (df["Region"] == selected_region) &
    (df["Year"].between(year_range[0], year_range[1]))
]

# Line chart
fig = px.line(df_filtered, x="Year", y="Production_tons", title=f"Production Trend: {selected_product} in {selected_region}", markers=True)
st.plotly_chart(fig, use_container_width=True)

# General Forecast Display
future_data = forecast_df[
    (forecast_df["Product"] == selected_product) &
    (forecast_df["Region"] == selected_region)
]
if not future_data.empty:
    st.subheader(f"📉 Forecast (2024–2028) for {selected_product} in {selected_region}")
    forecast_fig = px.line(future_data, x="Year", y="Forecast_tons", title="Forecasted Production")
    st.plotly_chart(forecast_fig, use_container_width=True)

# Cluster visualization
st.subheader("🧬 Regional Clusters Based on Production")
cluster_fig = px.scatter(
    clusters_df, x="PC1", y="PC2", color=clusters_df["Cluster"].astype(str), text="Region",
    title="Cluster Map of Regions (PCA-Reduced)", height=600
)
cluster_fig.update_traces(textposition='top center')
st.plotly_chart(cluster_fig, use_container_width=True)

# Data table
with st.expander("📄 View Filtered Data"):
    st.dataframe(df_filtered.reset_index(drop=True))