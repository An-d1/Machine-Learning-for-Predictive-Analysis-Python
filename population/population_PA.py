import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the JSON data
file_path = "Instat1_20250408-205211.json"
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract years and population values
years = list(data['dataset']['dimension']['Year']['category']['index'].keys())
population_values = data['dataset']['value']

# Convert to DataFrame
df_pop = pd.DataFrame({
    'Year': list(map(int, years)),
    'Population': population_values
})

# Train a linear regression model
X_pop = df_pop[['Year']]
y_pop = df_pop['Population']
pop_model = LinearRegression()
pop_model.fit(X_pop, y_pop)

# Predict population for the next 4 years
future_years_pop = pd.DataFrame({'Year': np.arange(2025, 2029)})
future_predictions_pop = pop_model.predict(future_years_pop)

# Plot actual and predicted data
plt.figure(figsize=(10, 6))
plt.plot(df_pop['Year'], df_pop['Population'], marker='o', label='Actual Population')
plt.plot(future_years_pop['Year'], future_predictions_pop, marker='x', linestyle='--', color='green', label='Predicted Population')
plt.title('Popullsia e Shqipërisë (Aktuale dhe e Parashikuar)\nPopulation of Albania (Actual and Predicted)')
plt.xlabel('Year')
plt.ylabel('Population')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

print("Predicted Population (2024–2028):")
print(future_years_pop.assign(Predicted_Population=future_predictions_pop.astype(int)))
