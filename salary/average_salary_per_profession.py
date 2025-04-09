import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import json

with open("Instat1_20250409-214151.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Recreate df_pivot from previously parsed JSON content

# Extract values again (in case of variable reset)
values = data['dataset']['value']
professions = data['dataset']['dimension']['GrupeProfesionet']['category']['label']
periods = data['dataset']['dimension']['Year']['category']['label']

profession_keys = list(professions.keys())
period_keys = list(periods.keys())

# Build structured DataFrame
rows = []
index = 0
for prof_key in profession_keys:
    prof_name = professions[prof_key]
    for period_key in period_keys:
        period = periods[period_key]
        if index < len(values):
            value = values[index]
            rows.append((period, prof_name, value))
            index += 1

df = pd.DataFrame(rows, columns=["Period", "Profession", "Value"])

# Convert 'Period' to datetime
def quarter_to_date(qstr):
    year, quarter = qstr.split("-")
    month = {"1": "01", "2": "04", "3": "07", "4": "10"}[quarter]
    return pd.to_datetime(f"{year}-{month}-01")

df["Period"] = df["Period"].apply(quarter_to_date)

# Pivot and fill missing data
df_pivot = df.pivot(index="Period", columns="Profession", values="Value")
df_pivot = df_pivot.ffill().bfill()
df_pivot.head()

# Save pivoted data to CSV
salaries_csv_path = "salaries_data.csv"
df_pivot.to_csv(salaries_csv_path)

# Prepare features and output folder
X = np.array([d.toordinal() for d in df_pivot.index]).reshape(-1, 1)
future_dates = pd.date_range(start=df_pivot.index[-1] + pd.offsets.QuarterEnd(), periods=8, freq='Q')
X_future = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)

# Directory to store model predictions
output_dir = "salary_predictions"
os.makedirs(output_dir, exist_ok=True)

# Train and save predictions for each profession
for profession in df_pivot.columns:
    y = df_pivot[profession].values

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVR": make_pipeline(StandardScaler(), SVR())
    }

    for name, model in models.items():
        model.fit(X, y)
        predictions = model.predict(X_future)
        df_predictions = pd.DataFrame({
            "Period": future_dates.strftime("%Y-%m-%d"),
            "Prediction": predictions
        })
        filename = f"{profession.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('.', '')}_{name}.csv"
        filepath = os.path.join(output_dir, filename)
        df_predictions.to_csv(filepath, index=False)

print("Saved to:", salaries_csv_path)
print("Sample prediction files:", os.listdir(output_dir)[:5])