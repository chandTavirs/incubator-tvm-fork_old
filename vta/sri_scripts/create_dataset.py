import argparse
import json
import os
import pandas as pd
import numpy as np

READ_BYTES = "read_bytes"
READ_BW = "read_bw"
WRITE_BYTES = "write_bytes"
WRITE_BW = "write_bw"


def check_max_num_samples(file_data):
    overall_bytes = []
    for trial in file_data:
        overall_bytes.append(trial["total_samples"])
    if len(overall_bytes) > 0:
        return max(overall_bytes)
    return 0


def create_dataset_file(file_data, file_name, output_dataset_dir):
    overall_bytes = []
    for trial in file_data:
        overall_bytes.append(trial['0']["overall"]["write_bytes"])

    index_max = max(range(len(overall_bytes)), key=overall_bytes.__getitem__)

    data_list = []
    data_read_load = file_data[index_max]['2']["samples"]
    data_write_store = file_data[index_max]['0']["samples"]
    data_read_fetch = file_data[index_max]['1']["samples"]
    data_read_uop = file_data[index_max]['3']["samples"]
    data_read_accum = file_data[index_max]['4']["samples"]

    assert len(data_read_load) == len(data_write_store) == len(data_read_fetch) == len(data_read_uop) == len(
        data_read_accum)

    for i, (load, store, fetch, uop, accum) in enumerate(zip(data_read_load, data_write_store, data_read_fetch,
                                                             data_read_uop, data_read_accum)):
        data_list.append(({"load_bytes": load[READ_BYTES], "load_bw": load[READ_BW],
                           "store_bytes": store[WRITE_BYTES], "store_bw": store[WRITE_BW],
                           "fetch_bytes": fetch[READ_BYTES], "fetch_bw": fetch[READ_BW],
                           "uop_bytes": uop[READ_BYTES], "uop_bw": uop[READ_BW],
                           "accum_bytes": accum[READ_BYTES], "accum_bw": accum[READ_BW]}))

    df = pd.DataFrame.from_dict(data_list)

    np.savetxt(os.path.join(output_dataset_dir, file_name.split('.')[0] + '.txt'), df.values, fmt='%.3f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NER dataset preparation script')
    parser.add_argument('--log_file_dir', type=str, default="profiling_results/random_graphs/graphs_500",
                        help='apm log files dir')
    parser.add_argument('--output_dataset_dir', type=str, default="dataset/ner/ner_min_20",
                        help='output dataset directory')

    args = parser.parse_args()

    log_file_dir = args.log_file_dir

    apm_log_files = [pos_json for pos_json in os.listdir(log_file_dir) if pos_json.endswith('.json')]

    for file in apm_log_files:
        with open(os.path.join(log_file_dir, file), 'r') as myfile:
            file_data = json.load(myfile)
            data = file_data["results"]
            if check_max_num_samples(data) < 20:
                continue
            create_dataset_file(data, file, args.output_dataset_dir)

    # data = []
    # with open(log_file, 'r') as myfile:
    #     file_data = json.load(myfile)
    #     if args.plot_model_time_series or args.plot_model_time_series_simul or args.plot_model_time_series_all_slots:
    #         data = file_data["results"]
    #     else:
    #         for wkl_idx in workloads:
    #             data.append(file_data["workloads"][wkl_idx])
