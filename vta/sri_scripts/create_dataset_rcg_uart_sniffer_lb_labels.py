import argparse
import os
import re

from alt_wkl_configs import *
import ast
from get_output_sizes import calc_conv_output_size, calc_maxpool_output_size

broken_networks = []
broken_files = []
perfect_networks = []
imperfect_networks = []


def check_layer_count_match(len_data_records, layer_count_line):
    match = re.search(r"Layer count:: (\d+)", layer_count_line)
    if match:
        layer_count_number = int(match.group(1))
        if layer_count_number == len_data_records:
            return True
        else:
            return False
    else:
        return False


def file_write_line_conv(cur_conv, layer_type):
    conv_out_height, conv_out_width, out_vol = calc_conv_output_size(cur_conv)

    ofm_dim_label = (conv_out_height, conv_out_width, cur_conv.out_filter)

    ifm_dim_label = (cur_conv.height, cur_conv.width, cur_conv.in_filter)

    kernel_dim_label = (cur_conv.hkernel, cur_conv.wkernel)

    stride_label = (cur_conv.hstride, cur_conv.wstride)

    pad_label = (cur_conv.hpad, cur_conv.wpad)

    return '\t'.join([layer_type, str(ifm_dim_label), str(ofm_dim_label), str(out_vol), str(kernel_dim_label),
                      str(stride_label), str(pad_label)]) + '\n'


def file_write_line_maxpool(cur_conv_cfg, maxpool_cfg, layer_type):
    conv_out_height, conv_out_width, _ = calc_conv_output_size(cur_conv_cfg)

    ifm_dim_label = (conv_out_height, conv_out_width, cur_conv_cfg.out_filter)

    mp_out_height, mp_out_width, out_vol = calc_maxpool_output_size(cur_conv_cfg, maxpool_cfg)

    ofm_dim_label = (mp_out_height, mp_out_height, cur_conv_cfg.out_filter)

    kernel_dim_label = (maxpool_cfg.hkernel, maxpool_cfg.wkernel)

    stride_label = (maxpool_cfg.hstride, maxpool_cfg.wstride)

    pad_label = (maxpool_cfg.hpad, maxpool_cfg.wpad)

    return '\t'.join([layer_type, str(ifm_dim_label), str(ofm_dim_label), str(out_vol), str(kernel_dim_label),
                      str(stride_label), str(pad_label)]) + '\n'


def generate_labels_file(networks, output_dataset_dir):
    for network in networks:
        network_name, layers = network.split("::")
        with open(os.path.join(output_dataset_dir, network_name + ".txt"), 'w+') as myfile:
            myfile.write(
                '\t'.join(["layer_type", "ifm_dim", "ofm_dim", "output_vol", "kernel_dim", "stride", "pad"]) + '\n')

        layers_nt = []
        layers = layers.split("), ")
        for layer in layers:
            layer = layer.replace("[", "")
            layer = layer.replace("]", "")
            layer = layer.replace("\n", "")
            if ')' not in layer:
                layer = layer + ")"
            layer_wkl = eval(layer)
            layers_nt.append(layer_wkl)

        layer_type = []
        cur_conv = None
        for i, layer in enumerate(layers_nt):

            if isinstance(layer, Conv2DWorkload):
                if i > 0 and len(layer_type) > 0:
                    with open(os.path.join(output_dataset_dir, network_name + ".txt"), 'a') as myfile:
                        myfile.write(file_write_line_conv(cur_conv, "".join(layer_type)))
                layer_type = []
                cur_conv = layer
                layer_type.append('C')
            elif isinstance(layer, BatchNorm2DConfig):
                layer_type.append('B')
            elif isinstance(layer, ReluConfig):
                layer_type.append('R')
            elif isinstance(layer, MaxPool2DConfig):
                if i > 0 and len(layer_type) > 0:
                    with open(os.path.join(output_dataset_dir, network_name + ".txt"), 'a') as myfile:
                        myfile.write(file_write_line_conv(cur_conv, "".join(layer_type)))

                layer_type = ['M']
                with open(os.path.join(output_dataset_dir, network_name + ".txt"), 'a') as myfile:
                    myfile.write(file_write_line_maxpool(cur_conv, layer, "".join(layer_type)))

                layer_type = []

        if len(layer_type) > 0:
            with open(os.path.join(output_dataset_dir, network_name + ".txt"), 'a') as myfile:
                myfile.write(file_write_line_conv(cur_conv, "".join(layer_type)))


def clean_data_records(log_files_dir, output_dataset_dir):
    final_dir = os.path.join(log_files_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)

    for filename in os.listdir(output_dataset_dir):
        if 'network' not in filename:
            continue

        filename_no_ext = filename.split('.')[0]
        with open(os.path.join(output_dataset_dir, filename)) as dataset_file:
            dataset_file_lines = dataset_file.readlines()[1:]

        len_dataset = len(dataset_file_lines)
        network_broken = True

        for sample_idx in range(5):
            sample_file = os.path.join(log_files_dir, f"{filename_no_ext}_sample{sample_idx}.log")
            if not os.path.exists(sample_file):
                continue

            with open(sample_file, 'r') as data_record_file:
                data_record_file_lines = data_record_file.readlines()
                layer_count_line = data_record_file_lines[-1]
                data_record_file_lines = data_record_file_lines[:-1]

            len_data_record = len(data_record_file_lines)
            if len_data_record == 0:
                broken_files.append(f"{filename_no_ext}_sample{sample_idx}.log")
                continue

            if len_dataset == len_data_record:
                data_record_file_lines.append(layer_count_line)
                for i, line in enumerate(data_record_file_lines[:-1]):
                    data_record_file_lines[i] = line.strip() + ":SE\n"
                with open(sample_file, 'w') as data_record_file:
                    data_record_file.writelines(data_record_file_lines)
                perfect_networks.append(filename_no_ext)
                network_broken = False
                break

            data_record_line_idx = 0
            net_layer_count = 0
            for i, line in enumerate(dataset_file_lines):
                if i >= len_data_record:
                    break
                req_total = int(line.split('\t')[3]) * 4
                cur_total = int(data_record_file_lines[data_record_line_idx].split(':')[1])
                if cur_total == req_total:
                    data_record_file_lines[data_record_line_idx] = data_record_file_lines[data_record_line_idx].strip() + ':SE\n'
                    data_record_line_idx += 1
                    net_layer_count += 1
                elif cur_total < req_total:
                    cur_line = data_record_line_idx
                    while cur_total < req_total:
                        cur_line += 1
                        if cur_line >= len_data_record:
                            break
                        try:
                            cur_total += int(data_record_file_lines[cur_line].split(':')[1])
                        except:
                            print(f'Exception occurred at network {filename_no_ext}. putting it as a broken network')
                            break
                        if cur_total > req_total:
                            excess = cur_total - req_total
                            excess_ratio = excess / int(data_record_file_lines[cur_line].split(':')[1])
                            new_data_record_file_line_1 = []
                            new_data_record_file_line_2 = []

                            for k in range(5):
                                original_value = int(data_record_file_lines[cur_line].split(':')[k])
                                if k == 1:
                                    adjusted_value_1 = original_value - excess
                                else:
                                    adjusted_value_1 = original_value - int(excess_ratio * original_value)
                                new_data_record_file_line_1.append(str(adjusted_value_1))
                                new_data_record_file_line_2.append(str(original_value - adjusted_value_1))

                            new_line_1 = ':'.join(new_data_record_file_line_1) + '\n'
                            new_line_2 = ':'.join(new_data_record_file_line_2) + '\n'
                            data_record_file_lines[cur_line] = new_line_1
                            data_record_file_lines.insert(cur_line + 1, new_line_2)
                            cur_total = req_total

                    if cur_total != req_total:
                        break
                    layer_lines = data_record_file_lines[data_record_line_idx: cur_line + 1]
                    layer_lines[0] = layer_lines[0].strip() + ':S\n'
                    layer_lines[-1] = layer_lines[-1].strip() + ':E\n'
                    for j in range(1, len(layer_lines) - 1):
                        layer_lines[j] = layer_lines[j].strip() + ':L\n'
                    data_record_file_lines[data_record_line_idx: cur_line + 1] = layer_lines
                    data_record_line_idx = cur_line + 1
                    net_layer_count += 1
                    # add network to imperfect networks
                    imperfect_networks.append(filename_no_ext)

                if cur_total != req_total:
                    break

            if data_record_line_idx == len(data_record_file_lines) and len_dataset == net_layer_count:
                new_layer_count_line = f'Layer count:: {len(dataset_file_lines)}'
                data_record_file_lines.append(new_layer_count_line)
                # with open(sample_file, 'w') as data_record_file:
                #     data_record_file.writelines(data_record_file_lines)
                # perfect_networks.append(filename_no_ext)
                network_broken = False
                break

        if network_broken:
            broken_networks.append(filename_no_ext)
        else:
            # write the final log file. add suffix "perfect" or "imperfect" to the filename
            if filename_no_ext in imperfect_networks:
                final_file = os.path.join(final_dir, f"{filename_no_ext}_imperfect.log")
            else:
                final_file = os.path.join(final_dir, f"{filename_no_ext}_perfect.log")

            with open(final_file, 'w') as final_log:
                final_log.writelines(data_record_file_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='UART Sniffer random compute graphs dataset preparation script')
    parser.add_argument('--log_files_dir', type=str, default="uart_sniffer_data/asp_dac/rcg/4x8x8_additional_remaining",
                        help='apm log files dir')
    parser.add_argument('--networks_file', type=str,
                        default="profiling_results/uart_sniffer/asp_dac/rcg/4x8x8_additional_remaining/networks_profiled.log",
                        help='profiled networks list')
    parser.add_argument('--output_dataset_dir', type=str, default="dataset/uart_sniffer/asp_dac/rcg/4x8x8_additional_remaining",
                        help='output dataset directory')

    args = parser.parse_args()

    log_files_dir = args.log_files_dir

    networks_file = args.networks_file

    # with open(networks_file, 'r') as myfile:
    #     networks = myfile.readlines()
    #     # create output dataset directory if it does not exist
    #     os.makedirs(args.output_dataset_dir, exist_ok=True)
    #     generate_labels_file(networks, args.output_dataset_dir)

    clean_data_records(args.log_files_dir, args.output_dataset_dir)
    broken_networks = list(set(broken_networks))
    imperfect_networks = list(set(imperfect_networks))
    perfect_networks = list(set(perfect_networks))

    # remove broken networks from perfect and imperfect networks
    perfect_networks = [network for network in perfect_networks if network not in broken_networks]
    imperfect_networks = [network for network in imperfect_networks if network not in broken_networks]

    print("Broken networks:: ", broken_networks)
    print("Broken files:: ", broken_files)
    print("Perfect networks:: ", perfect_networks)
    print("Imperfect networks:: ", imperfect_networks)
