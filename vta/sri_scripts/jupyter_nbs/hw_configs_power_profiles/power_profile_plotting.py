import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft

# Load the CSV files
file_4x8x8 = 'power_readings_pmbus_4x8x8.csv'
file_1x16x16 = 'power_readings_pmbus_1x16x16.csv'

data_4x8x8 = pd.read_csv(file_4x8x8)
data_1x16x16 = pd.read_csv(file_1x16x16)
data_4x8x8.columns.values[0] = 'Sample'
data_1x16x16.columns.values[0] = 'Sample'

# Plot the power consumption data
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(data_4x8x8['Sample'], data_4x8x8['12V_power'], label='4x8x8')
plt.title('Power Profile: 4x8x8')
plt.xlabel('Sample')
plt.ylabel('Power')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(data_1x16x16['Sample'], data_1x16x16['12V_power'], label='1x16x16', color='orange')
plt.title('Power Profile: 1x16x16')
plt.xlabel('Sample')
plt.ylabel('Power')
plt.legend()

plt.tight_layout()
plt.show()

# Compute summary statistics
summary_4x8x8 = data_4x8x8['12V_power'].describe()
summary_1x16x16 = data_1x16x16['12V_power'].describe()

# Compute additional statistics
variance_4x8x8 = data_4x8x8['12V_power'].var()
variance_1x16x16 = data_1x16x16['12V_power'].var()

skewness_4x8x8 = skew(data_4x8x8['12V_power'])
skewness_1x16x16 = skew(data_1x16x16['12V_power'])

kurtosis_4x8x8 = kurtosis(data_4x8x8['12V_power'])
kurtosis_1x16x16 = kurtosis(data_1x16x16['12V_power'])

# Perform FFT
fft_4x8x8 = fft(data_4x8x8['12V_power'].to_numpy())
fft_1x16x16 = fft(data_1x16x16['12V_power'].to_numpy())

# Frequency domain features
freq_4x8x8 = np.abs(fft_4x8x8)
freq_1x16x16 = np.abs(fft_1x16x16)

# Print summary statistics and additional features
print("Summary Statistics for 4x8x8 Power Profile:")
print(summary_4x8x8)
print(f"Variance: {variance_4x8x8}")
print(f"Skewness: {skewness_4x8x8}")
print(f"Kurtosis: {kurtosis_4x8x8}")

print("\nSummary Statistics for 1x16x16 Power Profile:")
print(summary_1x16x16)
print(f"Variance: {variance_1x16x16}")
print(f"Skewness: {skewness_1x16x16}")
print(f"Kurtosis: {kurtosis_1x16x16}")

# Plot frequency domain features
plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(freq_4x8x8, label='4x8x8')
plt.title('Frequency Domain: 4x8x8')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(freq_1x16x16, label='1x16x16', color='orange')
plt.title('Frequency Domain: 1x16x16')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()