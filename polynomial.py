import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate sample data with a quadratic relationship
np.random.seed(0)
X = 2 * np.random.rand(200, 1)
y = 0.5 * X**2 + X + 2 + np.random.randn(200, 1)

# Apply polynomial features with degree 2
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Fit linear regression model to the polynomial features
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

# Plot the original data
plt.scatter(X, y, color='blue', label='Original Data')

# Plot the quadratic regression curve
X_new = np.linspace(0, 2, 200).reshape(200, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
plt.plot(X_new, y_new, color='red', linewidth=2, label='Polynomial Regression (degree=2)')

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Polynomial Regression Example')
plt.show()
