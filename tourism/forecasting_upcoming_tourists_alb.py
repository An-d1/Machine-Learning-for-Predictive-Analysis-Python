import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load JSON data
file_path = "shtetas_te_huaj_ne_territorin_shqiptar.json"
df = pd.read_json(file_path)

# Convert 'Viti' to integer
df["Viti"] = df["Viti"].astype(int)

# Reshape Data: Convert to a time series format
df_melted = df.melt(id_vars=["Viti", "Totali"], var_name="Month", value_name="Tourists")
month_order = ["Janar", "Shkurt", "Mars", "Prill", "Maj", "Qershor", "Korrik", "Gusht", "Shtator", "Tetor", "Nentor", "Dhjetor"]
df_melted["Month"] = pd.Categorical(df_melted["Month"], categories=month_order, ordered=True)
df_melted.sort_values(["Viti", "Month"], inplace=True)

df_melted = df_melted[df_melted["Viti"] != 2020]

# Create a DateTime index
df_melted["Date"] = pd.to_datetime(df_melted["Viti"].astype(str) + "-" + (df_melted["Month"].cat.codes + 1).astype(str))
df_melted.set_index("Date", inplace=True)

# Keep only the tourists column
df_ts = df_melted["Tourists"]

df_ts.plot(figsize=(12,6), title="Monthly Tourists in Albania", ylabel="Number of Tourists")
plt.show()

# Define SARIMA model (SARIMA(p,d,q)(P,D,Q,s))
model = SARIMAX(df_ts,
                order=(1,1,1),  # (p,d,q)
                seasonal_order=(1,1,1,12),  # (P,D,Q,s) where s=12 (monthly seasonality)
                enforce_stationarity=False,
                enforce_invertibility=False)

# Fit the model
sarima_result = model.fit()

# Print model summary
print(sarima_result.summary())

# Forecast next 12 months
forecast_steps = 24
future_dates = pd.date_range(start=df_ts.index[-1], periods=forecast_steps+1, freq='M')[1:]
forecast = sarima_result.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()

# Plot results
plt.figure(figsize=(12,6))
plt.plot(df_ts, label="Observed", color="blue")
plt.plot(future_dates, forecast.predicted_mean, label="Forecast", color="red")
plt.fill_between(future_dates, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1], color='pink', alpha=0.3)
plt.legend()
plt.title(f"Tourist Forecast for Next {forecast_steps} Months")
plt.show()

# pie chart for the forecasted year (2025)
year = 2025
# Create month names for the forecast period
albanian_months = ["Janar", "Shkurt", "Mars", "Prill", "Maj", "Qershor",
                   "Korrik", "Gusht", "Shtator", "Tetor", "Nentor", "Dhjetor"]

# Get the 2025 forecast data (first 12 months of forecast)
forecast_2025 = forecast.predicted_mean[:12]
month_indices = [d.month - 1 for d in future_dates[:12]]  # Convert to 0-based month indices
month_names = [albanian_months[i] for i in month_indices]

plt.figure(figsize=(10, 10))
plt.pie(forecast_2025, labels=month_names, autopct='%1.1f%%', startangle=90)
plt.title(f"Predicted Monthly Tourist Share ({year}) (Ndarja sipas muajve e turisteve qe priten ne Shqiperi per vitin {year})")
plt.show()

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Tourists": forecast.predicted_mean.values,
    "Lower_Bound": forecast_ci.iloc[:, 0].values,
    "Upper_Bound": forecast_ci.iloc[:, 1].values
})
pd.options.display.float_format = '{:,.0f}'.format  # No decimals, comma separator
print(forecast_df)
