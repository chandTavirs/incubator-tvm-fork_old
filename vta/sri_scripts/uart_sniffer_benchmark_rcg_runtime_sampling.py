import multiprocessing
import threading
from math import floor, ceil
from random import randrange, random
from random import choice, choices

import serial

from wkl_configs import *
from Wkls import MAX_POOL_WKLS
from Wkls import WKLS_1x1x0, WKLS_1x2x0, WKLS_3x1x0, WKLS_3x1x1, WKLS_3x2x0, WKLS_5x1x0, WKLS_3x2x1, WKLS_5x1x2, \
    WKLS_5x2x2, broken_1x8x32
import argparse

import os
import json
import time
from tvm.contrib import graph_runtime
import pytest
import numpy as np
from collections import namedtuple
from vta.top import graph_pack

import tvm
from tvm import te
from tvm import relay
from tvm import autotvm, rpc
from tvm.contrib import utils
from tvm.contrib.pickle_memoize import memoize
from tvm import topi
import tvm.topi.testing
import vta
from vta import program_fpga, reconfig_runtime
import vta.testing
from vta.testing import simulator
import subprocess
import re
import torch
from torch import nn
import glob

empty_networks = []

CONV2D = Workload.__name__
DCONV2D = DepthwiseConv2D.__name__
MAXPOOL = maxPoolConfig.__name__
RELU = reluConfig.__name__
BN = batchNormConfig.__name__
max_stop_count = 20

env = vta.get_env()
print(env.BATCH)
exit(0)

# schedule_log_files = glob.glob(r'logs/tuning_logs/vta_4x8x8/*.log')
# schedule_log_files = glob.glob(r'logs/tuning_logs/*.log')

zero_line = "slot0::0000000000slot1::0000000000slot2::0000000000slot3::0000000000slot4::0000000000"

sample_re = re.compile(
    "^slot ([\d])\s+write bytes = ([\d]+)\s+read bytes = ([\d]+)\s+ write b/w = ([\d\.]+)\s+read b/w = ([\d\.]+)$")
overall_re = re.compile("^slot ([\d])\s+total write bytes = ([\d]+) and total read bytes = ([\d]+)$")
total_samples_re = re.compile("^total samples =\s+([\d]+)$")


def reset_serial_port(port="/dev/ttyUSB3", baud=921600):
    ser = serial.Serial(port, baud)
    ser.write(b'Ppassword\n')
    time.sleep(2)
    # ser.write(b'password\n')
    ser.close()


def send_sampling_rate(port="/dev/ttyUSB3", baud=921600, sampling_rate=50000):
    ser = serial.Serial(port, baud)
    data = bytearray()
    data.append(0x53)  # 8'h50 (just the hexadecimal value 0x50)
    data.extend(sampling_rate.to_bytes(4, 'big'))
    data.append(0x0A)
    ser.write(data)
    ser.close()


def poll_serial_port(port="/dev/ttyUSB3", baud=921600, log_file="uart_sniffer_data/tmp.log"):
    ser = serial.Serial(port, baud, timeout=5)
    flag = False
    layer_count = 0
    # with open(log_file, 'w') as myfile:
    #     while True:
    #         line = ser.readline().decode()
    #         if(line is not None and line != "" and zero_line not in line):
    #             flag = True
    #             layer_count += 1
    #             myfile.write(line)
    #             # if args.print_to_console:
    #             #     print(line)
    #         elif flag:
    #             # print("Layer count:: {}".format(layer_count))
    #             myfile.write("Layer count:: {}".format(layer_count))
    #             ser.close()
    #             break
    with open(log_file, 'w') as myfile:
        while True:
            byte_line = ser.read(20)
            chunks = [byte_line[i:i + 4] for i in range(0, len(byte_line), 4)]
            int_line = [int.from_bytes(bytearray(val), "big") for val in chunks]
            line = ':'.join(str(item) for item in int_line)
            if (line is not None and line != ""):
                flag = True
                layer_count += 1
                myfile.write(line + '\n')
                # if args.print_to_console:
                #     print(line)
            elif flag:
                # print("Layer count:: {}".format(layer_count))
                myfile.write("Layer count:: {}".format(layer_count))
                ser.close()
                break


def calc_conv_output_shape(wkl):
    out_height = floor((wkl.height + 2 * wkl.hpad - wkl.hkernel) / wkl.hstride + 1)
    out_width = floor((wkl.width + 2 * wkl.wpad - wkl.wkernel) / wkl.wstride + 1)

    return out_height, out_width


def calc_maxpool_output_shape(wkl, pool_cfg):
    in_height, in_width = calc_conv_output_shape(wkl)

    if bool(pool_cfg.ceil_mode):
        floor_or_ceil = ceil
    else:
        floor_or_ceil = floor

    out_height = floor_or_ceil((in_height + 2 * pool_cfg.hpad - pool_cfg.hkernel) / pool_cfg.hstride + 1)
    out_width = floor_or_ceil((in_width + 2 * pool_cfg.wpad - pool_cfg.wkernel) / pool_cfg.wstride + 1)

    return out_height, out_width


def construct_random_graph(args, broken_wkls):
    wkl_prob_tuples = [(args.conv_3x1x1_prob, WKLS_3x1x1),
                       (args.conv_3x2x0_prob, WKLS_3x2x0),
                       (args.conv_3x2x1_prob, WKLS_3x2x1),
                       (args.conv_3x1x0_prob, WKLS_3x1x0),
                       (args.conv_1x2x0_prob, WKLS_1x2x0),
                       (args.conv_1x1x0_prob, WKLS_1x1x0),
                       (args.conv_5x2x2_prob, WKLS_5x2x2),
                       (args.conv_5x1x2_prob, WKLS_5x1x2),
                       (args.conv_5x1x0_prob, WKLS_5x1x0),
                       ]

    probs, wkl_list = zip(*wkl_prob_tuples)

    while True:

        conv_wkl = choice(choices(wkl_list, probs)[0])[1]
        # conv_wkl = wkl_list[randrange(len(wkl_list))][1]
        if conv_wkl.height < 26 or conv_wkl.out_filter < conv_wkl.in_filter or conv_wkl in broken_wkls:
            continue

        layers = []

        while True:
            pot_next_layer = []
            out_height, out_width = calc_conv_output_shape(conv_wkl)
            bn = None
            relu = None
            maxpool = None
            if random() <= args.bn_prob:
                bn = batchNormConfig(conv_wkl.out_filter)
            if random() <= args.relu_prob:
                relu = reluConfig(conv_wkl.out_filter)
            if random() <= args.maxpool_prob:
                maxpool = MAX_POOL_WKLS[randrange(len(MAX_POOL_WKLS))][1]
            if out_height == conv_wkl.height and out_width == conv_wkl.width and conv_wkl.in_filter == conv_wkl.out_filter:
                if conv_wkl.hkernel == 3:
                    # convert conv to depthwise conv based on probability
                    if random() <= args.Depthwise_conv2d_prob:
                        conv_wkl = DepthwiseConv2D(conv_wkl.batch, conv_wkl.height, conv_wkl.width, conv_wkl.in_filter,
                                                   conv_wkl.out_filter, conv_wkl.hkernel, conv_wkl.wkernel,
                                                   conv_wkl.hpad, conv_wkl.wpad, conv_wkl.hstride, conv_wkl.wstride, conv_wkl.out_filter)
                for i in range(randrange(1, args.max_block_depth + 1)):
                    layers.append(conv_wkl)
                    if bn is not None:
                        layers.append(bn)
                    if relu is not None:
                        layers.append(relu)

                if maxpool is not None:
                    layers.append(maxpool)
                    out_height, out_width = calc_maxpool_output_shape(conv_wkl, maxpool)

            else:
                if conv_wkl.hkernel == 3 and conv_wkl.in_filter == conv_wkl.out_filter:
                    # convert conv to depthwise conv based on probability
                    if random() <= args.Depthwise_conv2d_prob:
                        conv_wkl = DepthwiseConv2D(conv_wkl.batch, conv_wkl.height, conv_wkl.width, conv_wkl.in_filter,
                                                   conv_wkl.out_filter, conv_wkl.hkernel, conv_wkl.wkernel,
                                                   conv_wkl.hpad, conv_wkl.wpad, conv_wkl.hstride, conv_wkl.wstride,
                                                   conv_wkl.out_filter)

                layers.append(conv_wkl)
                if bn is not None:
                    layers.append(bn)
                if relu is not None:
                    layers.append(relu)
                if maxpool is not None:
                    layers.append(maxpool)
                    out_height, out_width = calc_maxpool_output_shape(conv_wkl, maxpool)

            # for _, wkl in wkl_list:
            #     if wkl.height == out_height and wkl.width == out_width and conv_wkl.out_filter == wkl.in_filter and \
            #             wkl not in layers:
            #         pot_next_layer.append(wkl)
            pot_3x1x1 = []
            for _, wkl in WKLS_3x1x1:
                if wkl.height == out_height and wkl.width == out_width and conv_wkl.out_filter == wkl.in_filter and \
                        wkl not in layers and wkl not in broken_wkls:
                    pot_3x1x1.append(wkl)
            pot_3x2x1 = []
            for _, wkl in WKLS_3x2x1:
                if wkl.height == out_height and wkl.width == out_width and conv_wkl.out_filter == wkl.in_filter and \
                        wkl not in layers and wkl not in broken_wkls:
                    pot_3x2x1.append(wkl)
            pot_3x2x0 = []
            for _, wkl in WKLS_3x2x0:
                if wkl.height == out_height and wkl.width == out_width and conv_wkl.out_filter == wkl.in_filter and \
                        wkl not in layers and wkl not in broken_wkls:
                    pot_3x2x0.append(wkl)
            pot_1x2x0 = []
            for _, wkl in WKLS_1x2x0:
                if wkl.height == out_height and wkl.width == out_width and conv_wkl.out_filter == wkl.in_filter and \
                        wkl not in layers and wkl not in broken_wkls:
                    pot_1x2x0.append(wkl)
            pot_1x1x0 = []
            for _, wkl in WKLS_1x1x0:
                if wkl.height == out_height and wkl.width == out_width and conv_wkl.out_filter == wkl.in_filter and \
                        wkl not in layers and wkl not in broken_wkls:
                    pot_1x1x0.append(wkl)
            pot_5x2x2 = []
            for _, wkl in WKLS_5x2x2:
                if wkl.height == out_height and wkl.width == out_width and conv_wkl.out_filter == wkl.in_filter and \
                        wkl not in layers and wkl not in broken_wkls:
                    pot_5x2x2.append(wkl)
            pot_5x1x2 = []
            for _, wkl in WKLS_5x1x2:
                if wkl.height == out_height and wkl.width == out_width and conv_wkl.out_filter == wkl.in_filter and \
                        wkl not in layers and wkl not in broken_wkls:
                    pot_5x1x2.append(wkl)
            pot_5x1x0 = []
            for _, wkl in WKLS_5x1x0:
                if wkl.height == out_height and wkl.width == out_width and conv_wkl.out_filter == wkl.in_filter and \
                        wkl not in layers and wkl not in broken_wkls:
                    pot_5x1x0.append(wkl)
            pot_3x1x0 = []
            for _, wkl in WKLS_3x1x0:
                if wkl.height == out_height and wkl.width == out_width and conv_wkl.out_filter == wkl.in_filter and \
                        wkl not in layers and wkl not in broken_wkls:
                    pot_3x1x0.append(wkl)

            if len(pot_3x1x1) == 0 and len(pot_3x2x1) == 0 and len(pot_3x2x0) == 0 and len(pot_3x1x0) == 0 and \
                    len(pot_1x2x0) == 0 and len(pot_1x1x0) == 0 and len(pot_5x2x2) == 0 and len(pot_5x1x2) == 0 and \
                    len(pot_5x1x0) == 0:
                break

            pot_next_layer = [
                (args.conv_3x1x1_prob, pot_3x1x1),
                (args.conv_3x2x0_prob, pot_3x2x0),
                (args.conv_3x2x1_prob, pot_3x2x1),
                (args.conv_3x1x0_prob, pot_3x1x0),
                (args.conv_1x2x0_prob, pot_1x2x0),
                (args.conv_1x1x0_prob, pot_1x1x0),
                (args.conv_5x2x2_prob, pot_5x2x2),
                (args.conv_5x1x2_prob, pot_5x1x2),
                (args.conv_5x1x0_prob, pot_5x1x0),
            ]

            probs_next, pot_wkl_list = zip(*pot_next_layer)
            while True:
                next_wkl_list = choices(pot_wkl_list, probs_next)[0]
                if len(next_wkl_list) == 0:
                    continue
                else:
                    conv_wkl = choice(next_wkl_list)
                    break

        conv_count = 0
        for layer in layers:
            if type(layer).__name__ == CONV2D:
                conv_count += 1

        if conv_count >= args.min_depth:
            return layers


def run_and_collect(env, remote, network_id, network, target, log_file_path, samples=10, host_ip='192.168.2.99',
                    file_prefix="axi_uart_sniffer_fifo", port="/dev/ttyUSB3", baud=921600,
                    sca_log_dir="uart_sniffer_data/asp_dac/rcg/4x8x8_additional/",
                    ht_sampling_rate=50000):
    CPU_exec = nn.Sequential(nn.Conv2d(3, network[0].in_filter, kernel_size=(3, 3), stride=(1, 1),
                                       padding=(1, 1)), nn.ReLU(inplace=True))

    layer_seq = []
    vta_exec_nodes = nn.Sequential()
    for node in network:
        if type(node).__name__ == CONV2D:
            layer_seq.append("conv")
            vta_exec_nodes.append(nn.Conv2d(node.in_filter, node.out_filter, kernel_size=(node.hkernel, node.wkernel),
                                            stride=(node.hstride, node.wstride), padding=(node.hpad, node.wpad)))
        elif type(node).__name__ == DCONV2D:
            layer_seq.append("dconv")
            vta_exec_nodes.append(nn.Conv2d(node.in_filter, node.out_filter, kernel_size=(node.hkernel, node.wkernel),
                                            stride=(node.hstride, node.wstride), padding=(node.hpad, node.wpad),
                                            groups=node.out_filter))
        elif type(node).__name__ == BN:
            layer_seq.append("bn")
            vta_exec_nodes.append(nn.BatchNorm2d(node.num_filters))
        elif type(node).__name__ == RELU:
            layer_seq.append("relu")
            vta_exec_nodes.append(nn.ReLU(inplace=True))
        elif type(node).__name__ == MAXPOOL:
            layer_seq.append("maxpool")
            vta_exec_nodes.append(nn.MaxPool2d(kernel_size=(node.hkernel, node.wkernel),
                                               padding=(node.hpad, node.wpad),
                                               stride=(node.hstride, node.wstride), ceil_mode=bool(node.ceil_mode)))

    last_layer = nn.AdaptiveAvgPool2d((1, 1))

    model = nn.Sequential(CPU_exec, vta_exec_nodes, last_layer)

    # init inputs to the model
    input_name = "input0"

    input_shape = [env.BATCH, 3, network[0].height, network[0].width]
    input_data = torch.randn(input_shape)

    scripted_model = torch.jit.trace(model, input_data).eval()
    shape_list = [(input_name, input_shape)]

    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    if target.device_name == "vta":
        # Perform quantization in Relay
        # Note: We set opt_level to 3 in order to fold batch norm
        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
                mod = relay.quantize.quantize(mod, params=params)

                # print(mod.astext(show_meta_data=False))
                # exit(0)

            # Perform graph packing and constant folding for VTA target
            # assert env.BLOCK_IN == env.BLOCK_OUT
            # do device annotation if target is intelfocl or sim
            relay_prog = graph_pack(
                mod["main"],
                env.BATCH,
                env.BLOCK_IN,
                env.BLOCK_OUT,
                env.WGT_WIDTH,
                start_name='cast',
                start_name_idx=7,
                stop_name='nn.adaptive_avg_pool2d',
                device_annot=(env.TARGET == "intelfocl"),
            )
    else:
        relay_prog = mod["main"]

    with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
        graph, lib, params = relay.build(
            relay_prog, target=target, params=params, target_host=env.target_host
        )

    temp = utils.tempdir()
    lib.export_library(temp.relpath("graphlib_network_{}.tar".format(network_id)))
    remote.upload(temp.relpath("graphlib_network_{}.tar".format(network_id)))
    lib = remote.load_module("graphlib_network_{}.tar".format(network_id))

    ctx = remote.ext_dev(0)
    m = graph_runtime.create(graph, lib, ctx)

    result_dict = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, "total_samples": 0}

    # with open("/home/srchand/Desktop/research/TVM/tvm/vta/sri_trial/logs/conv_profiling_results.txt", 'a') as myfile:
    # myfile.write("\n")
    # myfile.write(str(wl))
    m.set_input(**params)
    m.set_input(input_name, input_data)

    results_list = []
    for i in range(samples):
        print("Sample measurement #{}".format(i))
        total_samples = 0
        result_dict = {"multi_exec": [], "single_exec": []}

        reset_serial_port(port=port, baud=baud)

        if i == 0 and network_id == 0:
            print('SETTING SAMPLING RATE!!')
            send_sampling_rate(port=port, baud=baud, sampling_rate=ht_sampling_rate)

        serial_read_process = multiprocessing.Process(target=poll_serial_port, args=(port, baud,
                                                                                     os.path.join(sca_log_dir,
                                                                                                  'network_{}_sample{}.log'
                                                                                                  .format(network_id,
                                                                                                          i))))

        serial_read_process.start()
        time.sleep(5)

        print("starting polling subprocess for single exec")

        m.run()

        time.sleep(5)
        print("Exiting polling process...")

        serial_read_process.join(10)
        if serial_read_process.is_alive():
            serial_read_process.terminate()
            empty_networks.append("network_{}".format(network_id))

        reset_serial_port(port=port, baud=baud)

        # results_list.append(result_dict)

        results_list.append(result_dict)
    print("Ran network_{} successfully!!".format(network_id))
    with open(log_file_path, 'r+') as myfile:
        file_data = json.load(myfile)
        file_data["results"] = results_list
        file_data["layer_sequence"] = layer_seq
        myfile.seek(0)
        json.dump(file_data, myfile, indent=4)


def connect_and_run(device, networks, log_file_dir="profiling_results/random_graphs/", host_ip='192.168.2.99',
                    num_samples=10,
                    file_prefix="axi_uart_sniffer_fifo", port="/dev/ttyUSB3", baud=921600,
                    sca_log_dir="uart_sniffer_data/asp_dac/rcg/4x8x8_additional/",
                    ht_sampling_rate=50000, schedule_log_files_path='logs/tuning_logs/vta_4x8x8/*.log'):
    # device_host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")

    schedule_log_files = glob.glob(schedule_log_files_path)

    device_host = host_ip

    device_port = os.environ.get("VTA_RPC_PORT", "9091")

    remote = rpc.connect(device_host, int(device_port))
    if device == "vta":
        target = env.target
        if env.TARGET not in ["sim", "tsim", "intelfocl"]:
            assert tvm.runtime.enabled("rpc")
            # program_fpga(remote, bitstream="/mnt/hgfs/vmware_ubuntu_sf/bitstreams/vta_axi_sniffer_mm_thres_75.bit")
            reconfig_runtime(remote)
    elif device == "arm_cpu":
        target = env.target_vta_cpu

    # create sca log dir if it doesn't exist
    if not os.path.exists(sca_log_dir):
        os.makedirs(sca_log_dir)

    with autotvm.tophub.context(target, extra_files=schedule_log_files):
        for i, network in enumerate(networks):
            print(network)
            log_file_path = os.path.join(log_file_dir, "network_{}.json".format(i))
            if not os.path.exists(log_file_path):
                with open(log_file_path, 'w+') as myfile:
                    json.dump({"network_id": i, "network_nodes": str(network), "layer_sequence": [], "results": []},
                              myfile, indent=4)
                run_and_collect(env, remote, i, network, target, log_file_path, samples=num_samples, host_ip=host_ip,
                                file_prefix=file_prefix, port=port, baud=baud, sca_log_dir=sca_log_dir,
                                ht_sampling_rate=ht_sampling_rate,
                                )
            else:
                print("Log file path already exists...", log_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random compute graph builder')
    # parser.add_argument('--log_file', type=str, default="logs/apm_logs/log.json",
    #                     help='output log file path')
    parser.add_argument('--host_ip', type=str, default='10.42.0.188',
                        help='pynq board IP')
    parser.add_argument('--num_graphs', type=int, default=200,
                        help='number of random graphs to generate')
    parser.add_argument('--min_depth', type=int, default=2,
                        help='minimum depth of network')
    parser.add_argument('--max_block_depth', type=int, default=4,
                        help='maximum depth of network')
    parser.add_argument('--bn_prob', type=float, default=1,
                        help='probability that bn follows convolution')
    parser.add_argument('--relu_prob', type=float, default=1,
                        help='probability that relu follows convolution')
    parser.add_argument('--maxpool_prob', type=float, default=0.2,
                        help='probability that maxpool follows convolution')
    parser.add_argument('--Depthwise_conv2d_prob', type=float, default=0.3,
                        help='probability that the convolution is depthwise')
    parser.add_argument('--conv_3x1x1_prob', type=float, default=0.45,
                        help='probability that conv hyp is (K, S, P) (3, 1, 1)')
    parser.add_argument('--conv_3x2x1_prob', type=float, default=0.25,
                        help='probability that conv hyp is (K, S, P) (3, 2, 1)')
    parser.add_argument('--conv_3x2x0_prob', type=float, default=0.05,
                        help='probability that conv hyp is (K, S, P) (3, 2, 0)')
    parser.add_argument('--conv_1x1x0_prob', type=float, default=0.05,
                        help='probability that conv hyp is (K, S, P) (1, 1, 0)')
    parser.add_argument('--conv_1x2x0_prob', type=float, default=0.15,
                        help='probability that conv hyp is (K, S, P) (1, 2, 0)')
    parser.add_argument('--conv_5x2x2_prob', type=float, default=0.05,
                        help='probability that conv hyp is (K, S, P) (5, 2, 2)')
    parser.add_argument('--conv_5x1x2_prob', type=float, default=0.05,
                        help='probability that conv hyp is (K, S, P) (5, 1, 2)')
    parser.add_argument('--conv_3x1x0_prob', type=float, default=0.05,
                        help='probability that conv hyp is (K, S, P) (3, 1, 0)')
    parser.add_argument('--conv_5x1x0_prob', type=float, default=0.05,
                        help='probability that conv hyp is (K, S, P) (5, 1, 0)')
    parser.add_argument('--wkl_list', type=str, default='all',
                        help='wkl list to pick convolutions from. Options - pre, man, incep, all, working')
    parser.add_argument("--log_file_dir", type=str,
                        default='profiling_results/uart_sniffer/asp_dac/rcg/1x16x16_35k_dconv2d/',
                        help="path to json file to store profiling data")
    parser.add_argument("--sca_log_dir", type=str,
                        default='uart_sniffer_data/asp_dac/rcg/1x16x16_35k_dconv2d/',
                        help="path to json file to store profiling data")
    parser.add_argument('--samples', type=int, default=5,
                        help='number of times to run each network')
    parser.add_argument('--data_file_prefix', type=str, default="axi_uart_sniffer_asp_dac_1x16x16_35k_dconv2d",
                        help='prefix of conv uart data files')
    parser.add_argument('--serial_port', type=str, default="/dev/ttyUSB3",
                        help='serial port name')
    parser.add_argument('--baud', type=int, default=921600,
                        help='serial port baud rate')
    parser.add_argument('--ht_sampling_rate', type=int, default=100000,
                        help='hardware trojan sampling rate')
    parser.add_argument('--schedule_log_files_path', type=str, default='logs/tuning_logs/vta_1x16x16/*.log',
                        help='path to schedule log files')

    # parser.add_argument('--load_networks_from_log', type=str, default='',
    #                     help='load random graphs from file and profile them')
    args = parser.parse_args()

    if args.wkl_list == "pre":
        from Wkls import PRE_TUNED_50

        wkl_list = PRE_TUNED_50
    elif args.wkl_list == "man":
        from Wkls import MANUAL_TUNED

        wkl_list = MANUAL_TUNED
    elif args.wkl_list == "incep":
        from Wkls import INCEPTION_TUNED

        wkl_list = INCEPTION_TUNED
    elif args.wkl_list == "working":
        from Wkls import WORKING_CONVS

        wkl_list = WORKING_CONVS
    else:
        from Wkls import ALL_TUNED_WKLS

        wkl_list = ALL_TUNED_WKLS

    networks = []
    # if args.load_networks_from_log != '':
    #     assert os.path.exists(args.load_networks_from_log)
    #     with open(args.load_networks_from_log, 'r') as myfile:
    # broken_1x8x32_wkls = [wkl[1] for wkl in broken_1x8x32]
    broken_wkls = []
    while len(networks) < args.num_graphs:
        networks.append(construct_random_graph(args, broken_wkls))

    # create networks log file dir if it doesn't exist. args.networks_log_file is the file path.
    if not os.path.exists(args.log_file_dir):
        os.makedirs(args.log_file_dir)

    with open(os.path.join(args.log_file_dir, 'networks_profiled.log'), 'w+') as myfile:

        for i, network in enumerate(networks):
            myfile.write("network_{}::".format(i) + str(network) + "\n")
    # else:
    #     print('Networks log file already exists...',args.networks_log_file,"Exiting....")
    #     exit(99)

    connect_and_run(device='vta', networks=networks, log_file_dir=args.log_file_dir, host_ip=args.host_ip,
                    num_samples=args.samples,
                    file_prefix=args.data_file_prefix, port=args.serial_port, baud=args.baud,
                    sca_log_dir=args.sca_log_dir,
                    ht_sampling_rate=args.ht_sampling_rate, schedule_log_files_path=args.schedule_log_files_path)

    # all_wkls = []
    # for wkl in PRE_TUNED_50:
    #     all_wkls.append(wkl[1])
    # for wkl in MANUAL_TUNED:
    #     all_wkls.append(wkl[1])
    # for wkl in INCEPTION_TUNED:
    #     all_wkls.append(wkl[1])
    #
    # all_wkls = list(set(all_wkls))
    #
    # for i, wkl in enumerate(all_wkls):
    #     print("(\"workload_{}\", Workload({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})),".format(
    #         i, wkl.batch, wkl.height, wkl.width, wkl.in_filter, wkl.out_filter, wkl.hkernel, wkl.wkernel,
    #         wkl.hpad, wkl.wpad, wkl.hstride, wkl.wstride
    #     ))

    if len(empty_networks) > 0:
        print("NEED TO RERUN THE FOLLOWING....")
        print(empty_networks)
