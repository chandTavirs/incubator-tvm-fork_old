import pandas as pd
import numpy as np
import pywt
from scipy.stats import median_absolute_deviation
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import stats
from tsfresh import extract_features as ef
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
import os

label_index_ro = {"ofm_dim": 1,
               "ifm_dim": 2,
               "kernel_dim": 3,
               "stride": 4,
               "pad": 5}


def get_final_df():
    data = []
    labels_file = '/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/dataset/ro_uart/conv_hyp/labels/labels.txt'
    dataset_path = '/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/ro_uart_logs/profiling_data/convs/csvs'
    with open(labels_file, 'r') as myfile:
        labels_lines = myfile.readlines()[1:]

    label_file_data = []

    for line in labels_lines:
        line = line.rstrip().split('\t')
        for file_name in os.listdir(dataset_path):
            if line[0] in file_name:
                df = pd.read_csv(os.path.join(dataset_path, file_name))
                z_scores = stats.zscore(df['RO_VALUES'])
                threshold = 20
                df = df[(np.abs(z_scores) <= threshold)]
                df.reset_index(drop=True, inplace=True)
                ifm_dim = line[label_index_ro['ifm_dim']].strip('()').split(',')[1:]
                kernel_dim = line[label_index_ro['kernel_dim']].strip('()').split(',')[0]
                stride = line[label_index_ro['stride']].strip('()').split(',')[0]
                pad = line[label_index_ro['pad']].strip('()').split(',')[0]
                df['id'] = line[0]
                label_file_tmp = {}
                label_file_tmp['ifm_dim'] = [float(dim) for dim in ifm_dim]
                label_file_tmp['kernel_dim'] = float(kernel_dim)
                label_file_tmp['stride'] = float(stride)
                label_file_tmp['pad'] = float(pad)
                label_file_data.append(label_file_tmp)

                data.append(df)

    return pd.concat(data)


if __name__ == "__main__":
    final_df = get_final_df()

    features = ef(final_df, column_id='id')

    print(features)