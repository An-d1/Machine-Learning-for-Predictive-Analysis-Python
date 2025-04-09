import os
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Load JSON file
with open("energy_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Parse values and metadata
values = data['dataset']['value']
items = data['dataset']['dimension']['Items']['category']['label']
years = data['dataset']['dimension']['Viti']['category']['label']

item_keys = list(items.keys())
year_keys = list(years.keys())

# Build long-form DataFrame
rows = []
i = 0
for item_key in item_keys:
    item_name = items[item_key]
    for year_key in year_keys:
        year = int(year_key)
        if i < len(values):
            rows.append((year, item_name, values[i]))
            i += 1

df = pd.DataFrame(rows, columns=["Year", "Item", "Value"])

# Pivot to wide format
df_pivot = df.pivot(index="Year", columns="Item", values="Value")
df_pivot = df_pivot.fillna(method="ffill").fillna(method="bfill")

# Prepare features and output directory
X = df_pivot.index.values.reshape(-1, 1)
pred_years = np.arange(2025, 2031).reshape(-1, 1)
os.makedirs("model_predictions", exist_ok=True)

# Train and save prediction CSVs for each target feature
for col in df_pivot.columns:
    y = df_pivot[col].values

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": make_pipeline(StandardScaler(), SVR())
    }

    for name, model in models.items():
        model.fit(X, y)
        predictions = model.predict(pred_years)
        pred_df = pd.DataFrame({"Year": pred_years.flatten(), "Prediction": predictions})

        filename = f"{col.replace(' ', '_').replace('.', '')}_{name}.csv"
        path = os.path.join("model_predictions", filename)
        pred_df.to_csv(path, index=False)
        print(f"Saved predictions to: {path}")
