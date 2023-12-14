import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Generate some example data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Independent variable
lambda_ = 2  # Poisson parameter
y_poisson = np.random.poisson(lambda_, size=(100,)).astype(float)  # Convert to float
y_noise = np.random.normal(0, 0.5, size=(100,))
y = y_poisson + y_noise

# Fit Poisson regression model
X = sm.add_constant(X)  # Add a constant term for the intercept
poisson_model = sm.Poisson(y, X).fit()

# Predict using the model
X_pred = np.linspace(0, 10, 100)
X_pred = sm.add_constant(X_pred)
y_pred = poisson_model.predict(X_pred)

# Plot the data and regression line
plt.scatter(X[:, 1], y, label='Actual Data', alpha=0.7)
plt.plot(X_pred[:, 1], y_pred, label='Poisson Regression', color='red', linewidth=2)
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable (Counts)')
plt.title('Poisson Regression Example')
plt.legend()
plt.show()

