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
import os
import time
from tvm.contrib import graph_runtime
import pytest
import numpy as np
from collections import namedtuple

from vta.top import graph_pack
import serial
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
import os
import torch
from torch import nn
from Wkls import ALL_TUNED_WKLS as pynq_wkls
from Wkls import MAX_POOL_WKLS as mp_wkls
import glob
import pandas as pd
import multiprocessing
zero_line = "slot0::0000000000slot1::0000000000slot2::0000000000slot3::0000000000slot4::0000000000"
# os.environ["TVM_LOG_DEBUG"] = "1"
schedule_log_files = glob.glob(r'logs/tuning_logs/*.log')
max_stop_count=20


# Get batch info from env
env = vta.get_env()

empty_wkls = []

def poll_serial_port(port="/dev/ttyUSB3", baud=921600, log_file="uart_sniffer_data/tmp.log"):
    ser = serial.Serial(port, baud, timeout=2)
    flag = False
    layer_count = 0
    with open(log_file, 'w') as myfile:
        while True:
            line = ser.readline().decode()
            if (line is not None and line != "" and zero_line not in line):
                flag = True
                layer_count += 1
                myfile.write(line)
                # if args.print_to_console:
                #     print(line)
            elif flag:
                # print("Layer count:: {}".format(layer_count))
                myfile.write("Layer count:: {}".format(layer_count))
                ser.close()
                break

# FIXME: we need a custom clip operator to circumvent a pattern detection limitation
@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x


def run_maxpool(env, remote, conv_wl, pool_cfg, cfg_id, target, log_file='logs/log.json', samples=10, host_ip ='192.168.2.99',
                file_prefix="axi_uart_sniffer_fifo", port="/dev/ttyUSB3", baud=921600):

    # Workload assertions
    assert conv_wl.hpad == conv_wl.wpad

    if not os.path.exists(log_file):
        with open(log_file, 'w+') as myfile:
            json.dump({"workloads": []}, myfile, indent=4)

    workload_dict = {"height": conv_wl.height, "width": conv_wl.width, "in_filter": conv_wl.in_filter, "out_filter": conv_wl.out_filter,
                     "hkernel": conv_wl.hkernel, "wkernel": conv_wl.wkernel, "hpad": conv_wl.hpad, "wpad": conv_wl.wpad,
                     "hstride": conv_wl.hstride, "wstride": conv_wl.wstride, "results": {},
                     "workload_str": '({},{},{},{},{},{},{},{},{},{})'.format(conv_wl.height, conv_wl.width, conv_wl.in_filter,
                                                                              conv_wl.out_filter, conv_wl.hkernel, conv_wl.wkernel,
                                                                              conv_wl.hpad, conv_wl.wpad, conv_wl.hstride, conv_wl.wstride)}


    CPU_exec = nn.Sequential(nn.Conv2d(3, conv_wl.in_filter, kernel_size=(3, 3), stride=(1, 1),
                                       padding=(1,1)), nn.ReLU(inplace=True))

    vta_exec_conv = nn.Sequential(nn.Conv2d(conv_wl.in_filter, conv_wl.out_filter, kernel_size=(conv_wl.hkernel, conv_wl.wkernel),
                              stride=(conv_wl.hstride, conv_wl.wstride), padding=(conv_wl.hpad, conv_wl.wpad)), nn.ReLU(inplace=True))

    vta_exec_maxpool = nn.MaxPool2d(kernel_size=(pool_cfg.hkernel, pool_cfg.wkernel), padding=(pool_cfg.hpad, pool_cfg.wpad),
                                    stride=(pool_cfg.hstride, pool_cfg.wstride), ceil_mode=bool(pool_cfg.ceil_mode))

    last_layer = nn.AdaptiveAvgPool2d((1,1))

    model = nn.Sequential(CPU_exec, vta_exec_conv, vta_exec_maxpool, last_layer)


    input_name = "input0"

    # Populate the shape and data type dictionary for ImageNet classifier input
    dtype_dict = {input_name: "float32"}
    shape_dict = {input_name: (env.BATCH, 3, conv_wl.height, conv_wl.width)}

    input_shape = [env.BATCH, 3, conv_wl.height, conv_wl.width]
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
                start_name_idx=7,
                stop_name='nn.adaptive_avg_pool2d',
                #stop_name_idx=18,
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
    workload_dict = {"height": conv_wl.height, "width": conv_wl.width, "in_filter": conv_wl.in_filter, "out_filter": conv_wl.out_filter,
                     "hkernel": conv_wl.hkernel, "wkernel": conv_wl.wkernel, "hpad": conv_wl.hpad, "wpad": conv_wl.wpad,
                     "hstride": conv_wl.hstride, "wstride": conv_wl.wstride, "results": [],
                     "workload_str": '({},{},{},{},{},{},{},{},{},{})'.format(conv_wl.height, conv_wl.width, conv_wl.in_filter,
                                                                              conv_wl.out_filter, conv_wl.hkernel, conv_wl.wkernel,
                                                                              conv_wl.hpad, conv_wl.wpad, conv_wl.hstride, conv_wl.wstride)}
    # with open("/home/srchand/Desktop/research/TVM/tvm/vta/sri_trial/logs/conv_profiling_results.txt", 'a') as myfile:
    # myfile.write("\n")
    # myfile.write(str(wl))
    m.set_input(**params)
    m.set_input(input_name, input_data)
    for i in range(samples):
        print("Sample measurement #{}".format(i))

        total_samples = 0
        result_dict_temp = {}

        for slot in range(6):
            result_dict_temp[slot] = {"samples": [], "overall": {"write_bytes": 0, "read_bytes": 0}}
        # myfile.write("\nSlot {} ".format(slot))

        print("starting polling subprocess")

        serial_read_process = multiprocessing.Process(target=poll_serial_port, args=(port, baud,
                                                                                     "uart_sniffer_data/maxpools/{}_{}_{}_sample{}.log"
                                                                                     .format(file_prefix,
                                                                                             workload_dict[
                                                                                                 'workload_str'], cfg_id, i)))
        serial_read_process.start()
        time.sleep(3)
        m.run()

        time.sleep(3)
        print("Exiting polling process...")

        serial_read_process.join(10)
        if serial_read_process.is_alive():
            serial_read_process.terminate()
            empty_wkls.append(workload_dict['workload_str']+"_"+cfg_id)
        workload_dict["results"].append(result_dict)
        print("Ran convolution+relu+maxpool successfully!!")

    with open(log_file, 'r+') as myfile:
        file_data = json.load(myfile)
        file_data["workloads"].append(workload_dict)
        myfile.seek(0)
        json.dump(file_data, myfile, indent=4)

def test_maxpool(device, log_file = "logs/log.json", host_ip = '192.168.2.99', num_samples=10,
                 file_prefix="axi_uart_sniffer_fifo", port="/dev/ttyUSB3", baud=921600):
    #device_host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")

    device_host = host_ip

    device_port = os.environ.get("VTA_RPC_PORT", "9091")

    remote = rpc.connect(device_host, int(device_port))
    if device == "vta":
        target = env.target
        if env.TARGET not in ["sim", "tsim", "intelfocl"]:
            assert tvm.runtime.enabled("rpc")
            # program_fpga(remote, bitstream="/home/srchand/Desktop/research/TVM_IL/tvm/vta/sri_scripts/bitstreams/vta_il_apm.bit")
            reconfig_runtime(remote)
    elif device == "arm_cpu":
        target = env.target_vta_cpu
    with autotvm.tophub.context(target, extra_files=schedule_log_files):  # load pre-tuned schedule parameters
        for _, wl in pynq_wkls:
            print(wl)
            for cfg_id, pool_cfg in mp_wkls:
                run_maxpool(env, remote, wl, pool_cfg, cfg_id, target, log_file=log_file, host_ip=host_ip, samples=num_samples,
                            file_prefix=file_prefix, port=port, baud=baud)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AXI Performance Monitor Convolution + Maxpool Benchmark')
    parser.add_argument('--log_file', type=str, default="profiling_results/log.json",
                        help='output log file path')
    parser.add_argument('--host_ip', type=str, default='192.168.2.99',
                        help='pynq board IP')
    parser.add_argument('--samples', type=int, default=10,
                        help='number of times to run convolutions')
    parser.add_argument('--data_file_prefix', type=str, default="axi_uart_sniffer_fifo",
                        help='prefix of conv uart data files')
    parser.add_argument('--serial_port', type=str, default="/dev/ttyUSB3",
                        help='serial port name')
    parser.add_argument('--baud', type=int, default=921600,
                        help='serial port baud rate')

    args = parser.parse_args()
    #test_conv2d(device="arm_cpu")
    test_maxpool(device="vta", log_file = args.log_file, host_ip = args.host_ip, num_samples=args.samples,
                 file_prefix=args.data_file_prefix, port=args.serial_port, baud=args.baud)

    if len(empty_wkls) > 0:
        print("NEED TO RERUN THE FOLLOWING....")
        print(empty_wkls)


