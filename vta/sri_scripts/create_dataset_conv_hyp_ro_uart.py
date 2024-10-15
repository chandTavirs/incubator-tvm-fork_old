import argparse
import json
import os
from wkl_configs import Workload
import numpy as np
from get_output_sizes import calc_conv_output_size
import pandas as pd
import re

def create_labels_file(data, output_labels_file):
    data_list = []

    with open(output_labels_file, 'w+') as myfile:
        myfile.write('\t'.join(["wkl_name", "ofm_dim",
                      "ifm_dim", "kernel_dim", "stride", "pad"])+'\n')

    for wkl in data:
        wkl_hyp = [int(item) for item in wkl.split('_')]
        conv_wkl = Workload(1, wkl_hyp[0],wkl_hyp[1],wkl_hyp[2],wkl_hyp[3],wkl_hyp[4],wkl_hyp[5],wkl_hyp[6],wkl_hyp[7],
                            wkl_hyp[8],wkl_hyp[9])

        out_height, out_width, _ = calc_conv_output_size(conv_wkl)

        ofm_dim_label = (out_height, out_width, conv_wkl.out_filter)

        ifm_dim_label = (conv_wkl.height, conv_wkl.width, conv_wkl.in_filter)

        kernel_dim_label = (conv_wkl.hkernel, conv_wkl.wkernel)

        stride_label = (conv_wkl.hstride, conv_wkl.wstride)

        pad_label = (conv_wkl.hpad, conv_wkl.wpad)


        data_list.append(({"wkl_name": wkl,
                           "ofm_dim": ofm_dim_label,
                           "ifm_dim": ifm_dim_label,
                           "kernel_dim": kernel_dim_label,
                           "stride": stride_label,
                           "pad": pad_label
                           }))

        with open(output_labels_file, 'a') as myfile:
            myfile.writelines('\t'.join([wkl, str(ofm_dim_label), str(ifm_dim_label), str(kernel_dim_label),
                                    str(stride_label), str(pad_label)])+'\n')

            #myfile.writelines("{.3f}\t{.3f}\t{.3f}\t{.3f}\t{.3f}\t{}\t{}\t{}\t{}\t{}")

    # df = pd.DataFrame.from_dict(data_list)
    #
    # np.savetxt(os.path.join(output_dataset_dir, file_name.split('.')[0].split('/')[1] + '.txt'), df.values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RO UART Trojan Convolution hyperparameters estimation dataset preparation script')
    parser.add_argument('--log_files_dir', type=str, default="ro_uart_logs/profiling_data/convs/csvs",
                        help='sniffer log files dir')
    parser.add_argument('--output_labels_file', type=str, default="dataset/ro_uart/conv_hyp/labels/labels.txt",
                        help='output label files directory')

    args = parser.parse_args()

    log_files_dir = args.log_files_dir

    wkl_names = []

    for i,filename in enumerate(os.listdir(log_files_dir)):
        arr_file_name = filename.split('_')
        if 'conv' not in arr_file_name:
            continue
        wkl_str = '_'.join(arr_file_name[arr_file_name.index('conv')+1:len(arr_file_name)-1])

        wkl_names.append(wkl_str)



    create_labels_file(wkl_names, args.output_labels_file)


