import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic time series data
np.random.seed(42)
date_rng = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
time_series_data = np.random.randn(len(date_rng))  # Random values for demonstration

# Create a DataFrame with the generated time series data
df = pd.DataFrame(data={'Date': date_rng, 'Value': time_series_data})

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Visualize the time series data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Value'], label='Time Series Data', color='blue')
plt.title('Sample Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
