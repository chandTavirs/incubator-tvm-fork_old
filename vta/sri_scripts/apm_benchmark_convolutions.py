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

max_stop_count = 20
schedule_log_files = ['/home/srchand/Desktop/research/VTA_scripts/logs/tuning_logs/vta.regnet_x_400mf.log',
                      '/home/srchand/Desktop/research/VTA_scripts/logs/tuning_logs/vta.googlenet.log',
                      '/home/srchand/Desktop/research/VTA_scripts/logs/tuning_logs/vta.squeezenet1.1.log',
                      '/home/srchand/Desktop/research/VTA_scripts/logs/tuning_logs/vta.mobilenet_v2.log',
                      '/home/srchand/Desktop/research/VTA_scripts/logs/tuning_logs/vta.regnet_x_800mf.log',
                      '/home/srchand/Desktop/research/VTA_scripts/logs/tuning_logs/vta.inceptionv3.log',
                      '/home/srchand/Desktop/research/VTA_scripts/logs/tuning_logs/vta.vgg11.log',
                      '/home/srchand/Desktop/research/VTA_scripts/logs/tuning_logs/vta.wide_resnet50_2.log',
                      '/home/srchand/Desktop/research/VTA_scripts/logs/tuning_logs/vta.regnet_x_1_6gf.log',
                      '/home/srchand/Desktop/research/VTA_scripts/logs/tuning_logs/vta.regnet_x_3_2gf.log',
                      '/home/srchand/Desktop/research/VTA_scripts/logs/tuning_logs/vta.vgg16.log',
                      '/home/srchand/Desktop/research/VTA_scripts/logs/tuning_logs/vta.mnasnet_0_5.log',
                      '/home/srchand/Desktop/research/VTA_scripts/logs/tuning_logs/vta.resnext50_32x4d.log']

sample_re = re.compile("^slot ([\d])\s+write bytes = ([\d]+)\s+read bytes = ([\d]+)\s+ write b/w = ([\d\.]+)\s+read b/w = ([\d\.]+)$")
overall_re = re.compile("^slot ([\d])\s+total write bytes = ([\d]+) and total read bytes = ([\d]+)$")
total_samples_re = re.compile("^total samples =\s+([\d]+)$")
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

# ResNet18 workloads
# resnet_wkls = [
#     # Workloads of resnet18 on imagenet
#     #('resnet-18.C1',  Workload(env.BATCH, 224, 224, 3,   64,  7, 7, 3, 3, 2, 2)),
#     ("resnet-18.C2", Workload(env.BATCH, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1)),
#     ("resnet-18.C3", Workload(env.BATCH, 56, 56, 64, 128, 3, 3, 1, 1, 2, 2)),
#     ("resnet-18.C4", Workload(env.BATCH, 56, 56, 64, 128, 1, 1, 0, 0, 2, 2)),
#     ("resnet-18.C5", Workload(env.BATCH, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1)),
#     ("resnet-18.C6", Workload(env.BATCH, 28, 28, 128, 256, 3, 3, 1, 1, 2, 2)),
#     ("resnet-18.C7", Workload(env.BATCH, 28, 28, 128, 256, 1, 1, 0, 0, 2, 2)),
#     ("resnet-18.C8", Workload(env.BATCH, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1)),
#     ("resnet-18.C9", Workload(env.BATCH, 14, 14, 256, 512, 3, 3, 1, 1, 2, 2)),
#     ("resnet-18.C10", Workload(env.BATCH, 14, 14, 256, 512, 1, 1, 0, 0, 2, 2)),
#     ("resnet-18.C11", Workload(env.BATCH, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1)),
# ]

#resnet_wkls = [
    # Workloads of resnet18 on imagenet
    #('resnet-18.C1',  Workload(env.BATCH, 224, 224, 3,   64,  7, 7, 3, 3, 2, 2)),
    #("resnet-18.C2", Workload(env.BATCH, 112, 112, 64, 128, 3, 3, 1, 1, 1, 1)),
    # ("resnet-18.C2", Workload(env.BATCH, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1)),
    # ("resnet-18.C2", Workload(env.BATCH, 28, 28, 64, 64, 3, 3, 1, 1, 1, 1)),
    # ("resnet-18.C2", Workload(env.BATCH, 14, 14, 64, 64, 3, 3, 1, 1, 1, 1)),
    # ("resnet-18.C11", Workload(env.BATCH, 7, 7, 64, 64, 3, 3, 1, 1, 1, 1)),
#]

resnet_wkls = [
    ("workload_0", Workload(env.BATCH, 208, 208, 16, 32, 3, 3, 1, 1, 1, 1)),
    ("workload_1", Workload(env.BATCH, 104, 104, 32, 64, 3, 3, 1, 1, 1, 1)),
    ("workload_2", Workload(env.BATCH, 26, 26, 128, 256, 3, 3, 1, 1, 1, 1)),
    ("workload_3", Workload(env.BATCH, 13, 13, 256, 512, 3, 3, 1, 1, 1, 1)),
    ("workload_4", Workload(env.BATCH, 13, 13, 512, 1024, 3, 3, 1, 1, 1, 1)),
    ("workload_5", Workload(env.BATCH, 52, 52, 64, 128, 3, 3, 1, 1, 1, 1)),
    ("workload_6", Workload(env.BATCH, 13, 13, 256, 128, 1, 1, 0, 0, 1, 1)),
    ("workload_7", Workload(env.BATCH, 13, 13, 1024, 256, 1, 1, 0, 0, 1, 1)),
    ("workload_8", Workload(env.BATCH, 26, 26, 384, 256, 3, 3, 1, 1, 1, 1)),
    ("workload_9", Workload(env.BATCH, 13, 13, 512, 256, 1, 1, 0, 0, 1, 1)),
    ("workload_10", Workload(env.BATCH, 26, 26, 256, 256, 1, 1, 0, 0, 1, 1)),
    ("workload_11", Workload(env.BATCH, 14, 14, 256, 512, 3, 3, 1, 1, 2, 2)),
    ("workload_12", Workload(env.BATCH, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1)),
    ("workload_13", Workload(env.BATCH, 28, 28, 128, 256, 3, 3, 1, 1, 2, 2)),
    ("workload_14", Workload(env.BATCH, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1)),
    ("workload_15", Workload(env.BATCH, 56, 56, 64, 128, 3, 3, 1, 1, 2, 2)),
    ("workload_16", Workload(env.BATCH, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1)),
    ("workload_17", Workload(env.BATCH, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1)),
    ("workload_18", Workload(env.BATCH, 56, 56, 64, 128, 1, 1, 0, 0, 2, 2)),
    ("workload_19", Workload(env.BATCH, 28, 28, 128, 256, 1, 1, 0, 0, 2, 2)),
    ("workload_20", Workload(env.BATCH, 14, 14, 256, 512, 1, 1, 0, 0, 2, 2)),
    ("workload_21", Workload(env.BATCH, 14, 14, 1024, 2048, 1, 1, 0, 0, 2, 2)),
    ("workload_22", Workload(env.BATCH, 28, 28, 512, 1024, 1, 1, 0, 0, 2, 2)),
    ("workload_23", Workload(env.BATCH, 56, 56, 256, 512, 1, 1, 0, 0, 2, 2)),
    ("workload_24", Workload(env.BATCH, 56, 56, 64, 64, 1, 1, 0, 0, 1, 1)),
    ("workload_25", Workload(env.BATCH, 56, 56, 256, 64, 1, 1, 0, 0, 1, 1)),
    ("workload_26", Workload(env.BATCH, 56, 56, 64, 256, 1, 1, 0, 0, 1, 1)),
    ("workload_27", Workload(env.BATCH, 56, 56, 256, 128, 1, 1, 0, 0, 1, 1)),
    ("workload_28", Workload(env.BATCH, 56, 56, 128, 128, 3, 3, 1, 1, 2, 2)),
    ("workload_29", Workload(env.BATCH, 28, 28, 512, 128, 1, 1, 0, 0, 1, 1)),
    ("workload_30", Workload(env.BATCH, 28, 28, 128, 512, 1, 1, 0, 0, 1, 1)),
    ("workload_31", Workload(env.BATCH, 28, 28, 512, 256, 1, 1, 0, 0, 1, 1)),
    ("workload_32", Workload(env.BATCH, 28, 28, 256, 256, 3, 3, 1, 1, 2, 2)),
    ("workload_33", Workload(env.BATCH, 14, 14, 1024, 256, 1, 1, 0, 0, 1, 1)),
    ("workload_34", Workload(env.BATCH, 14, 14, 256, 1024, 1, 1, 0, 0, 1, 1)),
    ("workload_35", Workload(env.BATCH, 14, 14, 1024, 512, 1, 1, 0, 0, 1, 1)),
    ("workload_36", Workload(env.BATCH, 14, 14, 512, 512, 3, 3, 1, 1, 2, 2)),
    ("workload_37", Workload(env.BATCH, 7, 7, 2048, 512, 1, 1, 0, 0, 1, 1)),
    ("workload_38", Workload(env.BATCH, 7, 7, 512, 2048, 1, 1, 0, 0, 1, 1)),
    ("workload_39", Workload(env.BATCH, 27, 27, 64, 192, 5, 5, 2, 2, 1, 1)),
    ("workload_40", Workload(env.BATCH, 13, 13, 192, 384, 3, 3, 1, 1, 1, 1)),
    ("workload_41", Workload(env.BATCH, 13, 13, 384, 256, 3, 3, 1, 1, 1, 1)),
    ("workload_42", Workload(env.BATCH, 13, 13, 256, 256, 3, 3, 1, 1, 1, 1)),
    ("workload_43", Workload(env.BATCH, 112, 112, 64, 128, 3, 3, 1, 1, 1, 1)),
    ("workload_44", Workload(env.BATCH, 112, 112, 128, 128, 3, 3, 1, 1, 1, 1)),
    ("workload_45", Workload(env.BATCH, 56, 56, 128, 256, 3, 3, 1, 1, 1, 1)),
    ("workload_46", Workload(env.BATCH, 56, 56, 256, 256, 3, 3, 1, 1, 1, 1)),
    ("workload_47", Workload(env.BATCH, 28, 28, 256, 512, 3, 3, 1, 1, 1, 1)),
    ("workload_48", Workload(env.BATCH, 28, 28, 512, 512, 3, 3, 1, 1, 1, 1)),
    ("workload_49", Workload(env.BATCH, 14, 14, 512, 512, 3, 3, 1, 1, 1, 1)),
]

# FIXME: we need a custom clip operator to circumvent a pattern detection limitation
@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x


def run_conv2d(env, remote, wl, target, check_correctness=True, print_ir=False, samples=10, log_file='logs/log.json', host_ip ='192.168.2.99'):

    # Workload assertions
    assert wl.hpad == wl.wpad

    if not os.path.exists(log_file):
        with open(log_file, 'w+') as myfile:
            json.dump({"workloads": []}, myfile, indent=4)

    CPU_exec = nn.Sequential(nn.Conv2d(3, wl.in_filter, kernel_size=(3,3), stride=(1,1),
                                         padding=(1,1)), nn.BatchNorm2d(wl.in_filter), nn.ReLU(inplace=True))

    vta_exec_conv = nn.Conv2d(wl.in_filter, wl.out_filter, kernel_size=(wl.hkernel, wl.wkernel),
                              stride=(wl.hstride, wl.wstride), padding=(wl.hpad, wl.wpad))

    vta_exec_batchnorm = nn.BatchNorm2d(wl.out_filter)
    vta_exec_relu = nn.ReLU()

    last_layer = nn.AdaptiveAvgPool2d((1,1))

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
                     "workload_str":'({},{},{},{},{},{},{},{},{},{})'.format(wl.height, wl.width, wl.in_filter,
                                                                             wl.out_filter, wl.hkernel, wl.wkernel,
                                                                             wl.hpad, wl.wpad, wl.hstride, wl.wstride)}
    #with open("/home/srchand/Desktop/research/TVM/tvm/vta/sri_trial/logs/conv_profiling_results.txt", 'a') as myfile:
        #myfile.write("\n")
        #myfile.write(str(wl))
    m.set_input(**params)
    m.set_input(input_name, input_data)
    for i in range(samples):
        total_samples = 0
        result_dict_temp = {}

        for slot in range(6):
            result_dict_temp[slot] = {"samples":[], "overall": {"write_bytes": 0, "read_bytes": 0}}
        #myfile.write("\nSlot {} ".format(slot))

        print("starting polling subprocess")
        proc = subprocess.Popen(["sshpass", "-p", "Srivat95", "ssh", "-t", "xilinx@{}".format(host_ip), "sudo", "python3",
                             "/home/xilinx/tvm_il/vta/python/vta/poll_apm_simul.py", "--slots", "0,1,2,3,4,5", "--stop_count", str(max_stop_count)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
        count = 0
        for i in range(200000000):
            count += 1
        m.run()

        while True:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.decode("utf-8")
            match_sample = re.search(sample_re, line)
            match_overall = re.search(overall_re, line)
            match_total_samples = re.search(total_samples_re, line)
            if match_sample:

                result_dict_temp[int(match_sample.group(1))]["samples"].append({"write_bytes": int(match_sample.group(2)), "read_bytes": int(match_sample.group(3)),
                                       "write_bw": float(match_sample.group(4)), "read_bw": float(match_sample.group(5))})
            elif match_overall:

                result_dict_temp[int(match_overall.group(1))]["overall"]["write_bytes"] = int(match_overall.group(2))
                result_dict_temp[int(match_overall.group(1))]["overall"]["read_bytes"] = int(match_overall.group(3))

            elif match_total_samples:
                total_samples = int(match_total_samples.group(1))

        for slot in result_dict_temp.keys():
            result_dict_temp[slot]["samples"] = result_dict_temp[slot]["samples"][:total_samples]
            result_dict[slot] = result_dict_temp[slot]
            result_dict["total_samples"] = total_samples
        #myfile.write(str(result_dict[slot]))
        workload_dict["results"].append(result_dict)
        print("Ran convolution successfully!!")

    with open(log_file, 'r+') as myfile:
        file_data = json.load(myfile)
        file_data["workloads"].append(workload_dict)
        myfile.seek(0)
        json.dump(file_data, myfile, indent=4)

def test_conv2d(device, log_file ="profiling_results/log.json", host_ip ='192.168.2.99', num_samples=10):
    #device_host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")

    device_host = host_ip

    device_port = os.environ.get("VTA_RPC_PORT", "9091")

    remote = rpc.connect(device_host, int(device_port))
    if device == "vta":
        target = env.target
        if env.TARGET not in ["sim", "tsim", "intelfocl"]:
            assert tvm.runtime.enabled("rpc")
            program_fpga(remote, bitstream="/home/srchand/Desktop/research/TVM_IL/tvm/vta/sri_scripts/bitstreams/vta_il_apm.bit")
            reconfig_runtime(remote)
    elif device == "arm_cpu":
        target = env.target_vta_cpu
    with autotvm.tophub.context(target):  # load pre-tuned schedule parameters
        for _, wl in resnet_wkls:
            print(wl)
            run_conv2d(env, remote, wl, target, log_file=log_file, host_ip=host_ip, samples=num_samples)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AXI Performance Monitor Convolution Benchmark')
    parser.add_argument('--log_file', type=str, default="profiling_results/log.json",
                        help='output log file path')
    parser.add_argument('--host_ip', type=str, default='192.168.2.99',
                        help='pynq board IP')
    parser.add_argument('--samples', type=int, default=10,
                        help='number of times to run convolutions')

    args = parser.parse_args()
    #test_conv2d(device="arm_cpu")
    test_conv2d(device="vta", log_file = args.log_file, host_ip = args.host_ip,num_samples=args.samples)