import argparse
import json
import os
from wkl_configs import Workload
import numpy as np
from get_output_sizes import calc_conv_output_size, calc_maxpool_output_size
import pandas as pd
import re
from Wkls import MAX_POOL_WKLS

data_pattern = re.compile("([\d]+):([\d]+):([\d]+):([\d]+):([\d]+)")

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


def create_dataset_file(data, output_dataset_file):
    data_list = []

    with open(output_dataset_file, 'w+') as myfile:
        myfile.write('\t'.join(["load_bytes", "store_bytes", "fetch_bytes", "uop_bytes", "accum_bytes", "ofm_dim",
                                "ifm_dim", "kernel_dim", "stride", "pad"]) + '\n')

    for wkl in data.keys():
        conv_wkl_hyp = [int(item) for item in wkl.split('_')[:-2]]
        conv_wkl = Workload(1, conv_wkl_hyp[0], conv_wkl_hyp[1], conv_wkl_hyp[2], conv_wkl_hyp[3], conv_wkl_hyp[4],
                            conv_wkl_hyp[5], conv_wkl_hyp[6], conv_wkl_hyp[7], conv_wkl_hyp[8], conv_wkl_hyp[9])

        mp_config = MAX_POOL_WKLS[int(re.findall("Config_([124])",wkl)[0]) -1 ][1]


        conv_out_height, conv_out_width, _ = calc_conv_output_size(conv_wkl)

        ifm_dim_label = (conv_out_height, conv_out_width, conv_wkl.out_filter)

        mp_out_height, mp_out_width, _ = calc_maxpool_output_size(conv_wkl, mp_config)

        ofm_dim_label = (mp_out_height, mp_out_height, conv_wkl.out_filter)

        kernel_dim_label = (mp_config.hkernel, mp_config.wkernel)

        stride_label = (mp_config.hstride, mp_config.wstride)

        pad_label = (mp_config.hpad, mp_config.wpad)

        index = 0
        if len(data[wkl]) < 3:
            continue
        elif data[wkl][0]['fetch'] == data[wkl][1]['fetch']:
            index = 0
        elif data[wkl][1]['fetch'] == data[wkl][2]['fetch']:
            index = 1
        elif data[wkl][0]['fetch'] == data[wkl][2]['fetch']:
            index = 0
        elif data[wkl][0]['store'] == data[wkl][1]['store']:
            index = 0
        elif data[wkl][1]['store'] == data[wkl][2]['store']:
            index = 1
        elif data[wkl][0]['store'] == data[wkl][2]['store']:
            index = 0
        elif data[wkl][0]['load'] == data[wkl][1]['load']:
            index = 0
        elif data[wkl][1]['load'] == data[wkl][2]['load']:
            index = 1
        elif data[wkl][0]['load'] == data[wkl][2]['load']:
            index = 0
        else:
            continue

        data_read_load = data[wkl][index]['load']
        data_write_store = data[wkl][index]['store']
        data_read_fetch = data[wkl][index]['fetch']
        data_read_uop = data[wkl][index]['uop']
        data_read_accum = data[wkl][index]['data']

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

        with open(output_dataset_file, 'a') as myfile:
            myfile.writelines(
                '\t'.join([str(data_read_load), str(data_write_store), str(data_read_fetch), str(data_read_uop),
                           str(data_read_accum), str(ofm_dim_label), str(ifm_dim_label), str(kernel_dim_label),
                           str(stride_label), str(pad_label)]) + '\n')

            # myfile.writelines("{.3f}\t{.3f}\t{.3f}\t{.3f}\t{.3f}\t{}\t{}\t{}\t{}\t{}")

    # df = pd.DataFrame.from_dict(data_list)
    #
    # np.savetxt(os.path.join(output_dataset_dir, file_name.split('.')[0].split('/')[1] + '.txt'), df.values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='UART Sniffer maxpool hyperparameters estimation dataset preparation script')
    parser.add_argument('--log_files_dir', type=str, default="uart_sniffer_data/maxpools",
                        help='apm log files dir')
    parser.add_argument('--output_dataset_file', type=str, default="dataset/uart_sniffer/maxpool_hyp/dataset.txt",
                        help='output dataset directory')

    args = parser.parse_args()

    log_files_dir = args.log_files_dir

    wkl_data_dict = {}

    wkl_re = re.compile("\(([\d,]+)\)")
    for i, filename in enumerate(os.listdir(log_files_dir)):
        arr_file_name = filename.split('_')
        wkl_str = arr_file_name[4]
        wkl_str = re.findall(wkl_re, wkl_str)[0].split(',')
        wkl_str = '_'.join(wkl_str)+'_'+'_'.join(arr_file_name[5:7])
        if wkl_str not in wkl_data_dict.keys() and 'Config_3' not in wkl_str:
            wkl_data_dict[wkl_str] = []
        if os.path.isfile(os.path.join(log_files_dir, filename)) and 'Config_3' not in filename:
            with open(os.path.join(log_files_dir, filename), 'r') as myfile:
                data_lines = myfile.readlines()
                if len(data_lines) >= 3:
                    data_line = data_lines[-2]
                    matches_all = re.findall(data_pattern, data_line)
                    if matches_all:
                        matches = matches_all[0]
                        wkl_data_dict[wkl_str].append(
                            {'fetch': int(matches[0]), 'store': int(matches[1]), 'load': int(matches[2]),
                             'uop': int(matches[3]), 'data': int(matches[4])})


    create_dataset_file(wkl_data_dict, args.output_dataset_file)
    # data = []
    # with open(log_file, 'r') as myfile:
    #     file_data = json.load(myfile)
    #     if args.plot_model_time_series or args.plot_model_time_series_simul or args.plot_model_time_series_all_slots:
    #         data = file_data["results"]
    #     else:
    #         for wkl_idx in workloads:
    #             data.append(file_data["workloads"][wkl_idx])
