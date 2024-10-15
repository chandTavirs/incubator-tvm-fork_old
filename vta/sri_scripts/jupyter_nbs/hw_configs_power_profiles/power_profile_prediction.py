import os
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, DMatrix
from sklearn.metrics import accuracy_score

def extract_features(file_path):
    data = pd.read_csv(file_path)
    power = data['12V_power'].to_numpy()

    # Time domain features
    mean_power = np.mean(power)
    std_power = np.std(power)
    var_power = np.var(power)
    skewness_power = skew(power)
    kurtosis_power = kurtosis(power)

    # Frequency domain features
    fft_power = fft(power)
    freq_power = np.abs(fft_power)
    mean_freq_power = np.mean(freq_power)
    std_freq_power = np.std(freq_power)

    return [mean_power, std_power, var_power, skewness_power, kurtosis_power, mean_freq_power, std_freq_power]

def load_dataset(folder_path, label):
    features = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            feature_vector = extract_features(file_path)
            feature_vector.append(label)
            features.append(feature_vector)
    return features

# Load datasets
features_1x16x16 = load_dataset('1x16x16_rcg', 0)
features_2x16x16 = load_dataset('2x16x16_rcg', 1)
features_4x8x8 = load_dataset('4x8x8_rcg', 2)


# Combine datasets
dataset = features_1x16x16 + features_4x8x8 + features_2x16x16
columns = ['mean_power', 'std_power', 'var_power', 'skewness_power', 'kurtosis_power', 'mean_freq_power', 'std_freq_power', 'label']
df = pd.DataFrame(dataset, columns=columns)

# Split dataset into training and testing sets
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create DMatrix for training
# dtrain = DMatrix(X_train, label=y_train, feature_names=columns[:-1])

# Train an XGBoost model, random intialization

# model = XGBClassifier(n_estimators=100, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Load the test files
file_4x8x8 = 'power_readings_pmbus_4x8x8.csv'
file_1x16x16 = 'power_readings_pmbus_1x16x16.csv'
file_2x16x16 = 'resnet18_pmbus_2x16x16.csv'

# Extract features from the test files
features_4x8x8 = [extract_features(file_4x8x8)]
features_1x16x16 = [extract_features(file_1x16x16)]
features_2x16x16 = [extract_features(file_2x16x16)]

columns = ['mean_power', 'std_power', 'var_power', 'skewness_power', 'kurtosis_power', 'mean_freq_power', 'std_freq_power']
df_4x8x8 = pd.DataFrame(features_4x8x8, columns=columns)
df_1x16x16 = pd.DataFrame(features_1x16x16, columns=columns)
df_2x16x16 = pd.DataFrame(features_2x16x16, columns=columns)


# Predict the configuration for the test files
prediction_4x8x8 = model.predict(df_4x8x8)
prediction_1x16x16 = model.predict(df_1x16x16)
prediction_2x16x16 = model.predict(df_2x16x16)

# Print the results
# print(f'Prediction for {file_4x8x8}: {"4x8x8" if prediction_4x8x8[0] == 1 else "1x16x16"}')
# print(f'Prediction for {file_1x16x16}: {"4x8x8" if prediction_1x16x16[0] == 1 else "1x16x16"}')
# print(f'Prediction for {file_2x16x16}: {"4x8x8" if prediction_2x16x16[0] == 1 else "1x16x16"}')

# Print the prediction results
print(f'Prediction for {file_4x8x8}: {prediction_4x8x8[0]}')
print(f'Prediction for {file_1x16x16}: {prediction_1x16x16[0]}')
print(f'Prediction for {file_2x16x16}: {prediction_2x16x16[0]}')