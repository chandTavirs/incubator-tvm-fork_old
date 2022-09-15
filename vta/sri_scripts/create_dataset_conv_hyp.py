import argparse
import json
import os
from wkl_configs import Workload
import numpy as np
from get_output_sizes import calc_conv_output_size
import pandas as pd

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
    data_list = []

    output_file_name = file_name.split('.')[0].split('/')[1]+'.txt'
    with open(os.path.join(output_dataset_dir, output_file_name), 'w+') as myfile:
        myfile.write('\t'.join(["load_bytes", "store_bytes", "fetch_bytes", "uop_bytes", "accum_bytes", "ofm_dim",
                      "ifm_dim", "kernel_dim", "stride", "pad"])+'\n')

    for wkl in file_data:
        conv_wkl = Workload(1, wkl['height'], wkl['width'], wkl['in_filter'], wkl['out_filter'], wkl['hkernel'],
                            wkl['wkernel'], wkl['hpad'], wkl['wpad'], wkl['hstride'], wkl['wstride'])

        out_height, out_width, _ = calc_conv_output_size(conv_wkl)

        ofm_dim_label = (out_height, out_width, conv_wkl.out_filter)

        ifm_dim_label = (conv_wkl.height, conv_wkl.width, conv_wkl.in_filter)

        kernel_dim_label = (conv_wkl.hkernel, conv_wkl.wkernel)

        stride_label = (conv_wkl.hstride, conv_wkl.wstride)

        pad_label = (conv_wkl.hpad, conv_wkl.wpad)

        overall_bytes = []
        for sample in wkl['results']:
            overall_bytes.append(sample['0']["overall"]["write_bytes"])

        index_max = max(range(len(overall_bytes)), key=overall_bytes.__getitem__)

        data_read_load = wkl['results'][index_max]['2']["overall"]['read_bytes']
        data_write_store = wkl['results'][index_max]['0']["overall"]['write_bytes']
        data_read_fetch = wkl['results'][index_max]['1']["overall"]['read_bytes']
        data_read_uop = wkl['results'][index_max]['3']["overall"]['read_bytes']
        data_read_accum = wkl['results'][index_max]['4']["overall"]['read_bytes']

        data_list.append(({"load_bytes": data_read_load,
                           "store_bytes": data_write_store,
                           "fetch_bytes": data_read_fetch,
                           "uop_bytes": data_read_uop,
                           "accum_bytes": data_read_accum,
                           "ofm_dim": ofm_dim_label,
                           "ifm_dim": ifm_dim_label,
                           "kernel_dim": kernel_dim_label,
                           "stride": stride_label,
                           "pad": pad_label
                           }))

        with open(os.path.join(output_dataset_dir, output_file_name), 'a') as myfile:
            myfile.writelines('\t'.join([str(data_read_load), str(data_write_store), str(data_read_fetch), str(data_read_uop),
                                    str(data_read_accum), str(ofm_dim_label), str(ifm_dim_label), str(kernel_dim_label),
                                    str(stride_label), str(pad_label)])+'\n')

            #myfile.writelines("{.3f}\t{.3f}\t{.3f}\t{.3f}\t{.3f}\t{}\t{}\t{}\t{}\t{}")

    # df = pd.DataFrame.from_dict(data_list)
    #
    # np.savetxt(os.path.join(output_dataset_dir, file_name.split('.')[0].split('/')[1] + '.txt'), df.values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convolution hyperparameters estimation dataset preparation script')
    parser.add_argument('--log_file', type=str, default="profiling_results/conv2d_50_tuned_wkls_5_samples.json",
                        help='apm log files dir')
    parser.add_argument('--output_dataset_dir', type=str, default="dataset/trial",
                        help='output dataset directory')

    args = parser.parse_args()

    log_file = args.log_file


    with open(log_file, 'r') as myfile:
        file_data = json.load(myfile)
        data = file_data["workloads"]

        create_dataset_file(data, log_file, args.output_dataset_dir)

    # data = []
    # with open(log_file, 'r') as myfile:
    #     file_data = json.load(myfile)
    #     if args.plot_model_time_series or args.plot_model_time_series_simul or args.plot_model_time_series_all_slots:
    #         data = file_data["results"]
    #     else:
    #         for wkl_idx in workloads:
    #             data.append(file_data["workloads"][wkl_idx])
