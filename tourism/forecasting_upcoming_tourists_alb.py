import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load JSON data
file_path = "C:\\Users\\Andis\\PycharmProjects\\PythonProject\\Machine-Learning-for-Predictive-Analysis-Python\\Data\\Raw-Data\\shtetas_te_huaj_ne_territorin_shqiptar.json"
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

# Plot diagnostics
sarima_result.plot_diagnostics(figsize=(12,6))
plt.show()

# Forecast next 12 months
forecast_steps = 12
future_dates = pd.date_range(start=df_ts.index[-1], periods=forecast_steps+1, freq='M')[1:]
forecast = sarima_result.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()

# Plot results
plt.figure(figsize=(12,6))
plt.plot(df_ts, label="Observed", color="blue")
plt.plot(future_dates, forecast.predicted_mean, label="Forecast", color="red")
plt.fill_between(future_dates, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1], color='pink', alpha=0.3)
plt.legend()
plt.title("Tourist Forecast for Next 12 Months")
plt.show()

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Tourists": forecast.predicted_mean.values,
    "Lower_Bound": forecast_ci.iloc[:, 0].values,
    "Upper_Bound": forecast_ci.iloc[:, 1].values
})
pd.options.display.float_format = '{:,.0f}'.format  # No decimals, comma separator
print(forecast_df)
sum = 0;
for numbers in forecast_df["Predicted_Tourists"]:
    sum += numbers

print(sum)