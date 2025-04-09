# Mixed Model for Prodhimet Blegtorale

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import streamlit as st
import json
from tqdm import tqdm

# Load and flatten JSON data
with open("/Users/arberxhauri/PycharmProjects/Test/livestock_products/prodhimet_blegtorale.json", encoding="utf-8") as f:
    data = json.load(f)["dataset"]

# Extract dimension labels
products = data["dimension"]["Prod"]["category"]["label"]
regions = data["dimension"]["Pref"]["category"]["label"]
years = data["dimension"]["Year"]["category"]["label"]

# Get sizes
prod_size = data["dimension"]["size"][0]
pref_size = data["dimension"]["size"][1]
year_size = data["dimension"]["size"][3]

# Create full index list
records = []
values = data["value"]
idx = 0
for prod_id, prod_label in products.items():
    for pref_id, pref_label in regions.items():
        for year_id, year_label in years.items():
            records.append({
                "Product": prod_label,
                "Region": pref_label,
                "Year": int(year_label),
                "Production_tons": values[idx]
            })
            idx += 1

# Create DataFrame
df = pd.DataFrame(records)

# Save cleaned data for use in Streamlit app
df.to_csv("cleaned_production_data.csv", index=False)

# Simple trend visualization
plt.figure(figsize=(12,6))
sns.lineplot(data=df[df["Product"] == "Qum\u00ebsht lope"], x="Year", y="Production_tons", hue="Region")
plt.title("Trend for Qum\u00ebsht lope by Region")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("trend_qumesht_lope.png")
plt.close()

# Forecasting example (ARIMA)
df_forecast = df[(df["Product"] == "Qum\u00ebsht lope") & (df["Region"] == "Tiran\u00eb")].sort_values("Year")
model = ARIMA(df_forecast["Production_tons"], order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=5)
forecast_years = list(range(df_forecast["Year"].max() + 1, df_forecast["Year"].max() + 6))

# Save forecast to file
forecast_df = pd.DataFrame({"Year": forecast_years, "Forecast_tons": forecast})
forecast_df.to_csv("forecast_qumesht_tirane.csv", index=False)

# Create future forecast for all (Product, Region)
forecast_list = []

grouped = df.groupby(["Product", "Region"])
for (product, region), group in tqdm(grouped, desc="Forecasting"):
    ts = group.sort_values("Year")[["Year", "Production_tons"]]
    if len(ts) < 5 or ts["Production_tons"].isnull().any():
        continue  # skip if not enough data

    try:
        model = ARIMA(ts["Production_tons"], order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=5)
        future_years = range(ts["Year"].max() + 1, ts["Year"].max() + 6)
        for year, pred in zip(future_years, forecast):
            forecast_list.append({
                "Product": product,
                "Region": region,
                "Year": year,
                "Forecast_tons": int(pred)
            })
    except Exception as e:
        print(f"Failed for {product} - {region}: {e}")

# Save to CSV
forecast_df_all = pd.DataFrame(forecast_list)
forecast_df_all.to_csv("forecast_all.csv", index=False)

# Clustering regions
pivot = df.pivot_table(index="Region", columns="Year", values="Production_tons", aggfunc="sum")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pivot.fillna(0))
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
pivot["Cluster"] = kmeans.labels_

# PCA for 2D visualization
pca = PCA(n_components=2)
components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
pca_df["Region"] = pivot.index
pca_df["Cluster"] = kmeans.labels_
pca_df.to_csv("clusters_pca.csv", index=False)

# Regression model
features = pd.get_dummies(df[["Product", "Region", "Year"]])
target = df["Production_tons"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(features, target)
importances = pd.Series(model.feature_importances_, index=features.columns)
importances.sort_values(ascending=False).head(10).to_csv("top_features.csv")
