import argparse
import csv
import os
import pandas as pd
import numpy as np
import pywt
from scipy.stats import median_absolute_deviation
from sklearn.cluster import KMeans
from scipy import stats



def generate_windowed_labels(reduced_csv_dir, output_csv_dir, window_size=50, stride=1):
    for csv_file in os.listdir(reduced_csv_dir):
        if not csv_file.endswith('csv'):
            continue

        df = pd.read_csv(os.path.join(reduced_csv_dir, csv_file))
        df = df.drop(columns=['Unnamed: 0'])
        boundary_indices = []
        for index, row in df.iterrows():
            if row['LAYER_OR_BOUNDARY'] == 'boundary':
                boundary_indices.append(index)

        for index in boundary_indices:
            start_index = max(0, index - window_size//2)
            end_index = min(len(df), index + window_size//2-1)
            df.loc[start_index:end_index, 'LAYER_OR_BOUNDARY'] = 'boundary'

        df.to_csv(os.path.join(output_csv_dir, csv_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='RO UART random compute graphs dataset preparation script')
    parser.add_argument('--reduced_csv_dir', type=str, default="/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/dataset/ro_uart/rcg/csvs/reduced",
                        help='csv data set files obtained from previous step')
    parser.add_argument('--output_csv_dir', type=str, default="/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/dataset/ro_uart/rcg/csvs/reduced/voting",
                        help='output dataset directory')
    parser.add_argument('--window_size', type=int, default=20, help='window size for boundary label')
    parser.add_argument('--stride', type=int, default=1, help='stride for windowing')

    args = parser.parse_args()

    reduced_csv_dir = args.reduced_csv_dir

    output_csv_dir = args.output_csv_dir

    generate_windowed_labels(reduced_csv_dir, output_csv_dir, args.window_size, args.stride)
