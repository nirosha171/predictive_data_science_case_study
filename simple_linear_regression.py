import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample Data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Create Model
model = LinearRegression()

# Train Model
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Print Results
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Plot Graph
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", label="Regression Line")
plt.title("Simple Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()