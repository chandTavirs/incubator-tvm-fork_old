import argparse
import csv
import os
import pandas as pd
import numpy as np
import pywt
from scipy.stats import median_absolute_deviation
from sklearn.cluster import KMeans
from scipy import stats


broken_networks = []
broken_files = []

def get_cleaned_data_from_csv(csv_file_path):
    df = pd.read_csv(csv_file_path)
    z_scores = stats.zscore(df['RO_VALUE'])
    threshold = 2.5

    df = df[(np.abs(z_scores) <= threshold)]
    df.reset_index(drop=True, inplace=True)

    return df


def extract_features(window, only_stat=False):
    features = {}

    features['mean'] = window.mean()
    features['std'] = window.std()
    features['median'] = np.median(window)

    if only_stat == True:
        return pd.Series(features)

    # Discrete Wavelet Transform (DWT) using pywt library
    coeffs = pywt.dwt(window, 'db4')
    features['cA'] = np.mean(coeffs[0])  # Approximation coefficients
    features['cD'] = np.mean(coeffs[1])  # Detail coefficients

    return pd.Series(features)

def get_windows_from_input_data(df, column_name='RO_VALUE', window_size=1000, stride=1):
    return [df[column_name].iloc[i:i + window_size] for i in
                   range(0, len(df) - window_size + 1, stride)]


def get_normalized_features(features_df):
    medians = features_df['median']

    normalized_features = (features_df - features_df.mean()) / features_df.std()

    normalized_features['median'] = medians

    return normalized_features

def get_cluster_labels(normalized_features, num_clusters=2):
    kmeans = KMeans(n_clusters=num_clusters)
    return kmeans.fit_predict(normalized_features)

def get_exec_cluster_indices(cluster_labels, window_size=1000):
    middle_index = len(cluster_labels) // 2
    middle_cluster = cluster_labels[middle_index - len(cluster_labels) // 8:middle_index + len(cluster_labels) // 3]
    exec_cluster = 0
    if np.sum(middle_cluster == 1) > np.sum(middle_cluster == 0):
        exec_cluster = 1

    cluster_exec_indices = np.where(cluster_labels == exec_cluster)[0]
    cluster_exec_indices = np.concatenate(
        (cluster_exec_indices, range(cluster_exec_indices[-1], cluster_exec_indices[-1] + window_size)))

    return cluster_exec_indices
def generate_reduced_csv_files(full_csv_dir, output_reduced_csv_dir, window_size=1000, stride=1):
    for csv_file in os.listdir(full_csv_dir):
        if not csv_file.endswith('csv'):
            continue

        df = get_cleaned_data_from_csv(os.path.join(full_csv_dir, csv_file))

        windows = get_windows_from_input_data(df, 'RO_VALUE', window_size, stride)

        features_df = pd.DataFrame([extract_features(window) for window in windows])

        normalized_features = get_normalized_features(features_df)

        cluster_labels = get_cluster_labels(normalized_features, 2)

        exec_cluster_indices = get_exec_cluster_indices(cluster_labels, window_size)

        exec_data = df.iloc[exec_cluster_indices]
        exec_data.reset_index(drop=True, inplace=True)

        exec_data.to_csv(os.path.join(output_reduced_csv_dir, csv_file))

        # with open(os.path.join(output_reduced_csv_dir,'_'.join(csv_file.split('_')[:-1])+'.csv'),'w') as myfile:
        #     writer = csv.writer(myfile)
        #     writer.writerow(["RO_VALUE", "LAYER_OR_BOUNDARY"])\
        #     # myfile.write('\t'.join(["RO_VALUE", "LAYER_OR_BOUNDARY"])+'\n')
        #
        #     # Write the data from the lists into two columns
        #     for item1, item2 in zip(data_record_file_lines, layer_labels):
        #         # line = '\t'.join([item1.strip(), item2.strip()])
        #         # myfile.write(line+'\n')
        #         writer.writerow([item1.strip(), item2.strip()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='RO UART random compute graphs dataset preparation script')
    parser.add_argument('--full_csv_dir', type=str, default="/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/dataset/ro_uart/rcg/csvs",
                        help='csv data set files obtained from previous step')
    parser.add_argument('--output_reduced_csv_dir', type=str, default="/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/dataset/ro_uart/rcg/csvs/reduced",
                        help='output dataset directory')
    parser.add_argument('--window_size', type=int, default=1000, help='window size for performing k means clustering')
    parser.add_argument('--stride', type=int, default=1, help='stride for windowing for performing k means clustering')

    args = parser.parse_args()

    full_csv_dir = args.full_csv_dir

    output_reduced_csv_dir = args.output_reduced_csv_dir

    generate_reduced_csv_files(full_csv_dir, output_reduced_csv_dir, args.window_size, args.stride)
