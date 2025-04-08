import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the JSON data
file_path = "numri_emigranteve.json"
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract year and values (emigrants)
years = list(data['dataset']['dimension']['Year']['category']['index'].keys())
values = data['dataset']['value']

# Convert to DataFrame
df = pd.DataFrame({
    'Year': list(map(int, years)),
    'Emigrants': values
})

# Remove the year 2020
df = df[df['Year'] != 2020]

# Create a simple linear regression model
X = df[['Year']]
y = df['Emigrants']
model = LinearRegression()
model.fit(X, y)

# Predict future values for the next 3 years
future_years = pd.DataFrame({'Year': np.arange(2025, 2028)})
future_predictions = model.predict(future_years)

# Plotting the actual and predicted data
plt.figure(figsize=(10, 6))
plt.plot(df['Year'], df['Emigrants'], marker='o', label='Actual Data')
plt.plot(future_years['Year'], future_predictions, marker='x', linestyle='-', color='red', label='Predicted')
plt.title('Numri i emigranteve ne Shqiperi \n'
          'Number of emigrants in Albania (Actual and Predicted)')
plt.xlabel('Year')
plt.ylabel('Number of Emigrants')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("Predicted Emigrants (2025–2027):")
print(future_years.assign(Predicted_Emigrants=future_predictions.astype(int)))
