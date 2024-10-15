import multiprocessing
import threading
from math import floor, ceil
from random import randrange, random

import serial

from wkl_configs import *
from Wkls import MAX_POOL_WKLS
from Wkls import VGG_11
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
MAXPOOL = maxPoolConfig.__name__
RELU = reluConfig.__name__
BN = batchNormConfig.__name__
max_stop_count = 20

env = vta.get_env()

schedule_log_files = glob.glob(r'logs/tuning_logs/vgg/*.log')

def reset_serial_port(port="/dev/ttyUSB3", baud=921600):
    ser = serial.Serial(port, baud)
    ser.write(b'password\n')
    time.sleep(2)
    ser.write(b'password\n')
    ser.close()

def poll_serial_port(port="/dev/ttyUSB3", baud=921600, log_file="uart_sniffer_data/tmp.log"):
    ser = serial.Serial(port, baud, timeout=5)
    flag = False
    layer_count = 0

    with open(log_file, 'w') as myfile:
        while True:
            byte_line = ser.read(20)
            chunks = [byte_line[i:i + 4] for i in range(0, len(byte_line), 4)]
            int_line = [int.from_bytes(bytearray(val), "big") for val in chunks]
            line = ':'.join(str(item) for item in int_line)
            if(line is not None and line != ""):
                flag = True
                layer_count += 1
                myfile.write(line+'\n')
                # if args.print_to_console:
                #     print(line)
            elif flag:
                # print("Layer count:: {}".format(layer_count))
                myfile.write("Layer count:: {}".format(layer_count))
                ser.close()
                break

def run_and_collect(env, remote, network_id, network, target, log_file_path, samples=10, host_ip ='192.168.2.99',
                    file_prefix="axi_uart_sniffer_fifo", port="/dev/ttyUSB3", baud=921600):

    CPU_exec = nn.Sequential(nn.Conv2d(3, network[0].in_filter, kernel_size=(3, 3), stride=(1, 1),
                                       padding=(1, 1)), nn.ReLU(inplace=True))

    layer_seq = []
    vta_exec_nodes = nn.Sequential()
    for node in network:
        if type(node).__name__ == CONV2D:
            layer_seq.append("conv")
            vta_exec_nodes.append(nn.Conv2d(node.in_filter, node.out_filter, kernel_size=(node.hkernel, node.wkernel),
                  stride=(node.hstride, node.wstride), padding=(node.hpad, node.wpad)))
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
            assert env.BLOCK_IN == env.BLOCK_OUT
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


        serial_read_process = multiprocessing.Process(target=poll_serial_port, args=(port, baud,
                                                                                     "uart_sniffer_data/vgg/network_{}_sample{}.log"
                                                                                     .format(network_id, i)))

        serial_read_process.start()
        time.sleep(3)

        print("starting polling subprocess for single exec")


        m.run()

        time.sleep(3)
        print("Exiting polling process...")

        serial_read_process.join(10)
        if serial_read_process.is_alive():
            serial_read_process.terminate()
            empty_networks.append("network_{}".format(network_id))


        results_list.append(result_dict)



        results_list.append(result_dict)
    print("Ran network_{} successfully!!".format(network_id))
    with open(log_file_path, 'r+') as myfile:
        file_data = json.load(myfile)
        file_data["results"] = results_list
        file_data["layer_sequence"] = layer_seq
        myfile.seek(0)
        json.dump(file_data, myfile, indent=4)


def connect_and_run(device, networks, log_file_dir="profiling_results/random_graphs/", host_ip = '192.168.2.99',num_samples=10,
                    file_prefix="axi_uart_sniffer_fifo", port="/dev/ttyUSB3", baud=921600):
    #device_host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")

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

    with autotvm.tophub.context(target, extra_files=schedule_log_files):
        for i, network in enumerate(networks):
            print(network)
            log_file_path = os.path.join(log_file_dir,"network_{}.json".format(i))
            if not os.path.exists(log_file_path):
                with open(log_file_path, 'w+') as myfile:
                    json.dump({"network_id": i, "network_nodes": str(network), "layer_sequence": [], "results":[]}, myfile, indent=4)
                run_and_collect(env, remote, i, network, target, log_file_path, samples=num_samples, host_ip=host_ip,
                                file_prefix=file_prefix, port=port, baud=baud)
            else:
                print("Log file path already exists...",log_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random compute graph builder')
    # parser.add_argument('--log_file', type=str, default="logs/apm_logs/log.json",
    #                     help='output log file path')
    parser.add_argument('--host_ip', type=str, default='192.168.2.99',
                        help='pynq board IP')

    parser.add_argument("--networks_log_file", type=str, default='profiling_results/uart_sniffer/vgg/networks_profiled.log',
                        help="path to file to store all the random graphs")
    parser.add_argument("--log_file_dir", type=str,
                        default='profiling_results/uart_sniffer/vgg/',
                        help="path to json file to store profiling data")
    parser.add_argument('--samples', type=int, default=10,
                        help='number of times to run each network')
    parser.add_argument('--data_file_prefix', type=str, default="vgg",
                        help='prefix of conv uart data files')
    parser.add_argument('--serial_port', type=str, default="/dev/ttyUSB3",
                        help='serial port name')
    parser.add_argument('--baud', type=int, default=921600,
                        help='serial port baud rate')
    # parser.add_argument('--load_networks_from_log', type=str, default='',
    #                     help='load random graphs from file and profile them')
    args = parser.parse_args()

    networks = []
    # if args.load_networks_from_log != '':
    #     assert os.path.exists(args.load_networks_from_log)
    #     with open(args.load_networks_from_log, 'r') as myfile:

    networks.append([wkl for _, wkl in VGG_11])


    # if not os.path.exists(args.networks_log_file):
    with open(args.networks_log_file,'w+') as myfile:

        for i, network in enumerate(networks):
            myfile.write("network_{}::".format(i)+str(network)+"\n")
    # else:
    #     print('Networks log file already exists...',args.networks_log_file,"Exiting....")
    #     exit(99)


    connect_and_run(device='vta', networks=networks, log_file_dir=args.log_file_dir, host_ip=args.host_ip, num_samples=args.samples,
                    file_prefix=args.data_file_prefix, port=args.serial_port, baud=args.baud)


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