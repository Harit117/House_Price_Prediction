import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Bengaluru_House_Data.csv").fillna(0.0)

features = ['total_sqft', 'bath', 'balcony', 'harit']
target = 'price'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

titles = ['Total Sqft vs Price', 'Bath vs Price', 'Balcony vs Price', 'Size vs Price']
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for i, ax in enumerate(axs.flat):
    x = df[features[i]].values
    y_plot = df['price'].values

    # Scatter plot
    ax.scatter(x, y_plot, alpha=0.5, color='teal')

    mask = ~np.isnan(x) & ~np.isnan(y_plot)
    m, b = np.polyfit(x[mask], y_plot[mask], 1)
    ax.plot(x, m * x + b, color='red', linewidth=2)

    ax.set_title(titles[i])
    ax.set_xlabel(features[i])
    ax.set_ylabel('Price')

plt.tight_layout()
plt.show()
