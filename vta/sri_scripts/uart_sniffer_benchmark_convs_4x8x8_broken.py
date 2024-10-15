# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Testing topi conv2d operator for VTA"""
import argparse
import json
import threading

import serial
import os
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
from vta import program_fpga, reconfig_runtime, reset_ro_monitor, start_ro_monitor, stop_ro_monitor
import vta.testing
from vta.testing import simulator
import subprocess
import re
import os
import torch
from torch import nn
from Wkls import ALL_TUNED_WKLS as pynq_wkls
from Wkls import broken_wkls_2x16x16_75000 as broken_wkls
import glob
import pandas as pd
import multiprocessing
zero_line = "slot0::0000000000slot1::0000000000slot2::0000000000slot3::0000000000slot4::0000000000"


# proc.wait()

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


max_stop_count = 20
schedule_log_files = glob.glob(r'logs/tuning_logs/vta_2x16x16/*.log')
ro_reading_re = re.compile("ro_data :: ([d]+).*")

Workload = namedtuple(
    "Conv2DWorkload",
    [
        "batch",
        "height",
        "width",
        "in_filter",
        "out_filter",
        "hkernel",
        "wkernel",
        "hpad",
        "wpad",
        "hstride",
        "wstride",
    ],
)

# Get batch info from env
env = vta.get_env()

empty_wkls = []


# FIXME: we need a custom clip operator to circumvent a pattern detection limitation
@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x


def run_conv2d(env, remote, wl, wl_id, target, check_correctness=True, print_ir=False, samples=10, log_file='logs/log.json',
               host_ip='192.168.2.99', data_file_dir='uart_sniffer_data/asp_dac/convs/broken_wkls',
               file_prefix="axi_uart_sniffer_fifo", port="/dev/ttyUSB3", baud=921600, sampling_rate=50000):
    # Workload assertions
    assert wl.hpad == wl.wpad

    if not os.path.exists(log_file):
        with open(log_file, 'w+') as myfile:
            # json.dump({"workloads": []}, myfile, indent=4)
            json.dump({}, myfile, indent=4)

    CPU_exec = nn.Sequential(nn.Conv2d(3, wl.in_filter, kernel_size=(3, 3), stride=(1, 1),
                                       padding=(1, 1)), nn.BatchNorm2d(wl.in_filter), nn.ReLU(inplace=True))

    vta_exec_conv = nn.Conv2d(wl.in_filter, wl.out_filter, kernel_size=(wl.hkernel, wl.wkernel),
                              stride=(wl.hstride, wl.wstride), padding=(wl.hpad, wl.wpad))

    last_layer = nn.AdaptiveAvgPool2d((1, 1))

    model = nn.Sequential(CPU_exec, vta_exec_conv, last_layer)

    # Load pre-configured AutoTVM schedules
    with autotvm.tophub.context(target, extra_files=schedule_log_files):

        input_name = "input0"

        # Populate the shape and data type dictionary for ImageNet classifier input
        dtype_dict = {input_name: "float32"}
        shape_dict = {input_name: (env.BATCH, 3, wl.height, wl.width)}

        input_shape = [env.BATCH, 3, wl.height, wl.width]
        input_data = torch.randn(input_shape)

        scripted_model = torch.jit.trace(model, input_data).eval()
        shape_list = [(input_name, input_shape)]

        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
        shape_dict.update({k: v.shape for k, v in params.items()})
        dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

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
                    start_name_idx=8,
                    stop_name='nn.adaptive_avg_pool2d',
                    # start_name_idx=6,
                    # start_name='nn.max_pool2d',
                    device_annot=(env.TARGET == "intelfocl"),
                )
        else:
            relay_prog = mod["main"]

        with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
            graph, lib, params = relay.build(
                relay_prog, target=target, params=params, target_host=env.target_host
            )

        temp = utils.tempdir()
        lib.export_library(temp.relpath("graphlib.tar"))
        remote.upload(temp.relpath("graphlib.tar"))
        lib = remote.load_module("graphlib.tar")

        ctx = remote.ext_dev(0)
        m = graph_runtime.create(graph, lib, ctx)

    result_dict = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, "total_samples": 0}
    workload_dict = {"height": wl.height, "width": wl.width, "in_filter": wl.in_filter, "out_filter": wl.out_filter,
                     "hkernel": wl.hkernel, "wkernel": wl.wkernel, "hpad": wl.hpad, "wpad": wl.wpad,
                     "hstride": wl.hstride, "wstride": wl.wstride, "results": [],
                     "workload_str": 'conv_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(wl.height, wl.width, wl.in_filter,
                                                                                 wl.out_filter, wl.hkernel, wl.wkernel,
                                                                                 wl.hpad, wl.wpad, wl.hstride,
                                                                                 wl.wstride)}
    # with open("/home/srchand/Desktop/research/TVM/tvm/vta/sri_trial/logs/conv_profiling_results.txt", 'a') as myfile:
    # myfile.write("\n")
    # myfile.write(str(wl))
    m.set_input(**params)
    m.set_input(input_name, input_data)
    num = 9  # number of times we run module for a single measurement
    rep = 1  # number of measurements (we derive std dev from this)
    timer = m.module.time_evaluator("run", ctx, number=num, repeat=rep)

    for i in range(samples):
        print("Sample measurement #{}".format(i))
        result_dict = {"multi_exec": [], "single_exec": []}

        # serial_read_thread = threading.Thread(target=poll_serial_port, args=(port,baud,
        #                                                                      "uart_sniffer_data/{}_{}_sample{}.log"
        #                                                                      .format(file_prefix,workload_dict['workload_str'],i)))
        # create data file directory if it doesn't exist
        if not os.path.exists(data_file_dir):
            os.makedirs(data_file_dir)

        reset_serial_port(port=port, baud=baud)

        if i == 0 and wl_id == 0:
            print("SETTING SAMPLING RATE")
            send_sampling_rate(port=port, baud=baud, sampling_rate=sampling_rate)

        serial_read_process = multiprocessing.Process(target=poll_serial_port, args=(port, baud,
                                                                                     os.path.join(data_file_dir,
                                                                                                  "{}_{}_sample{}.log"
                                                                                                  .format(file_prefix,
                                                                                                          workload_dict[
                                                                                                              'workload_str'],
                                                                                                          i))))

        # serial_read_thread.start()

        serial_read_process.start()

        time.sleep(3)


        print("starting polling subprocess for single exec")

        m.run()

        time.sleep(3)
        print("Exiting polling process...")

        serial_read_process.join(10)
        if serial_read_process.is_alive():
            serial_read_process.terminate()
            empty_wkls.append(workload_dict['workload_str'])

        reset_serial_port(port=port, baud=baud)

        workload_dict["results"].append(result_dict)
        print("Ran convolution successfully!!")

    with open(log_file, 'r+') as myfile:
        file_data = json.load(myfile)
        # file_data["workloads"].append(workload_dict)
        file_data[workload_dict['workload_str']] = workload_dict
        myfile.seek(0)
        json.dump(file_data, myfile, indent=4)


def test_conv2d(device, log_file="profiling_results/log.json", host_ip='192.168.2.99', num_samples=10, data_file_dir="uart_sniffer_data/asp_dac/convs/broken_wkls",
                file_prefix="axi_uart_sniffer_fifo", port="/dev/ttyUSB3", baud=921600, sampling_rate=50000):
    # device_host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")

    device_host = host_ip

    device_port = os.environ.get("VTA_RPC_PORT", "9091")

    # set log file name suffix as "sampling_rate".json
    log_file = log_file.replace(".json", "try_{}.json".format(sampling_rate))

    # create log_file directory if it doesn't exist
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    # set data file directory name suffix as "try_sampling_rate"
    data_file_dir = data_file_dir + "_try_{}".format(sampling_rate)

    # append "try_sampling_rate" to file_prefix
    file_prefix = file_prefix + "_try_{}".format(sampling_rate)

    # create data_file directory if it doesn't exist
    if not os.path.exists(data_file_dir):
        os.makedirs(data_file_dir)

    remote = rpc.connect(device_host, int(device_port))
    if device == "vta":
        target = env.target
        if env.TARGET not in ["sim", "tsim", "intelfocl"]:
            assert tvm.runtime.enabled("rpc")
            # program_fpga(remote, bitstream="/mnt/hgfs/vmware_ubuntu_sf/bitstreams/vta_axi_sniffer_with_layer_term_reset.bit")
            reconfig_runtime(remote)
    elif device == "arm_cpu":
        target = env.target_vta_cpu
    with autotvm.tophub.context(target):  # load pre-tuned schedule parameters
        for wl_id, (_, wl) in enumerate(broken_wkls):
            print(wl)
            run_conv2d(env, remote, wl, wl_id, target, samples=num_samples, log_file=log_file, host_ip=host_ip,
                       data_file_dir=data_file_dir, file_prefix=file_prefix, port=port, baud=baud, sampling_rate=sampling_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AXI Performance Monitor Convolution Benchmark')
    parser.add_argument('--log_file', type=str, default="profiling_results/log.json",
                        help='output log file path')
    parser.add_argument('--host_ip', type=str, default='192.168.2.99',
                        help='pynq board IP')
    parser.add_argument('--samples', type=int, default=5,
                        help='number of times to run convolutions')
    parser.add_argument('--data_file_dir', type=str, default="uart_sniffer_data/asp_dac/convs/4x8x8/broken_wkls",
                        help='output data file path')
    parser.add_argument('--data_file_prefix', type=str, default="axi_uart_sniffer_fifo",
                        help='prefix of conv uart data files')
    parser.add_argument('--serial_port', type=str, default="/dev/ttyUSB3",
                        help='serial port name')
    parser.add_argument('--baud', type=int, default=921600,
                        help='serial port baud rate')
    parser.add_argument('--ht_sampling_rate', type=int, default=50000,
                        help='hardware trojan sampling rate')

    args = parser.parse_args()
    # test_conv2d(device="arm_cpu")
    test_conv2d(device="vta", log_file=args.log_file, host_ip=args.host_ip, num_samples=args.samples, data_file_dir=args.data_file_dir,
                file_prefix=args.data_file_prefix, port=args.serial_port, baud=args.baud, sampling_rate=args.ht_sampling_rate)

    if len(empty_wkls) > 0:
        print("NEED TO RERUN THE FOLLOWING....")
        print(empty_wkls)
