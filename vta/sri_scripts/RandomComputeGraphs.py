from math import floor, ceil
from random import randrange, random
from wkl_configs import *
from Wkls import MAX_POOL_WKLS
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

CONV2D = Workload.__name__
MAXPOOL = maxPoolConfig.__name__
RELU = reluConfig.__name__
BN = batchNormConfig.__name__
max_stop_count = 20

env = vta.get_env()

schedule_log_files = ['logs/tuning_logs/vta.regnet_x_400mf.log',
                      'logs/tuning_logs/vta.googlenet.log',
                      'logs/tuning_logs/vta.squeezenet1.1.log',
                      'logs/tuning_logs/vta.mobilenet_v2.log',
                      'logs/tuning_logs/vta.regnet_x_800mf.log',
                      'logs/tuning_logs/vta.inceptionv3.log',
                      'logs/tuning_logs/vta.vgg11.log',
                      'logs/tuning_logs/vta.wide_resnet50_2.log',
                      'logs/tuning_logs/vta.regnet_x_1_6gf.log',
                      'logs/tuning_logs/vta.regnet_x_3_2gf.log',
                      'logs/tuning_logs/vta.vgg16.log',
                      'logs/tuning_logs/vta.mnasnet_0_5.log',
                      'logs/tuning_logs/vta.resnext50_32x4d.log']

sample_re = re.compile("^slot ([\d])\s+write bytes = ([\d]+)\s+read bytes = ([\d]+)\s+ write b/w = ([\d\.]+)\s+read b/w = ([\d\.]+)$")
overall_re = re.compile("^slot ([\d])\s+total write bytes = ([\d]+) and total read bytes = ([\d]+)$")
total_samples_re = re.compile("^total samples =\s+([\d]+)$")


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


def construct_random_graph(args, wkl_list):
    while True:
        conv_wkl = wkl_list[randrange(len(wkl_list))][1]
        if conv_wkl.height < 26 or conv_wkl.out_filter < conv_wkl.in_filter:
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
                layers.append(conv_wkl)
                if bn is not None:
                    layers.append(bn)
                if relu is not None:
                    layers.append(relu)
                if maxpool is not None:
                    layers.append(maxpool)
                    out_height, out_width = calc_maxpool_output_shape(conv_wkl, maxpool)

            for _, wkl in wkl_list:
                if wkl.height == out_height and wkl.width == out_width and conv_wkl.out_filter == wkl.in_filter and \
                        wkl not in layers:
                    pot_next_layer.append(wkl)

            if len(pot_next_layer) != 0:
                conv_wkl = pot_next_layer[randrange(len(pot_next_layer))]

            else:
                break
        conv_count = 0
        for layer in layers:
            if type(layer).__name__ == CONV2D:
                conv_count += 1

        if conv_count >= args.min_depth:
            return layers

def run_and_collect(env, remote, network_id, network, target, log_file_path, samples=10, host_ip ='192.168.2.99'):

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
        print("sample {}:::".format(i))
        total_samples = 0
        result_dict_temp = {}

        for slot in range(6):
            result_dict_temp[slot] = {"samples": [], "overall": {"write_bytes": 0, "read_bytes": 0}}
        # myfile.write("\nSlot {} ".format(slot))

        print("starting polling subprocess")
        proc = subprocess.Popen(
            ["sshpass", "-p", "Srivat95", "ssh", "-t", "xilinx@{}".format(host_ip), "sudo", "python3",
             "/home/xilinx/tvm_il/vta/python/vta/poll_apm_simul.py", "--slots", "0,1,2,3,4,5", "--stop_count",
             str(max_stop_count)], stdout=subprocess.PIPE,
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

                result_dict_temp[int(match_sample.group(1))]["samples"].append(
                    {"write_bytes": int(match_sample.group(2)), "read_bytes": int(match_sample.group(3)),
                     "write_bw": float(match_sample.group(4)), "read_bw": float(match_sample.group(5))})
            elif match_overall:

                result_dict_temp[int(match_overall.group(1))]["overall"]["write_bytes"] = int(
                    match_overall.group(2))
                result_dict_temp[int(match_overall.group(1))]["overall"]["read_bytes"] = int(
                    match_overall.group(3))

            elif match_total_samples:
                total_samples = int(match_total_samples.group(1))

        for slot in result_dict_temp.keys():
            result_dict_temp[slot]["samples"] = result_dict_temp[slot]["samples"][:total_samples]
            result_dict[slot] = result_dict_temp[slot]
            result_dict["total_samples"] = total_samples

        results_list.append(result_dict)

    with open(log_file_path, 'r+') as myfile:
        file_data = json.load(myfile)
        file_data["results"] = results_list
        file_data["layer_sequence"] = layer_seq
        myfile.seek(0)
        json.dump(file_data, myfile, indent=4)


def connect_and_run(device, networks, log_file_dir="profiling_results/random_graphs/", host_ip = '192.168.2.99',num_samples=10):
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

    with autotvm.tophub.context(target, extra_files=schedule_log_files):
        for i, network in enumerate(networks):
            print(network)
            log_file_path = os.path.join(log_file_dir,"network_{}.json".format(i))
            if not os.path.exists(log_file_path):
                with open(log_file_path, 'w+') as myfile:
                    json.dump({"network_id": i, "network_nodes": str(network), "layer_sequence": [], "results":[]}, myfile, indent=4)
                run_and_collect(env, remote, i, network, target, log_file_path, samples=num_samples, host_ip=host_ip)
            else:
                print("Log file path already exists...",log_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random compute graph builder')
    # parser.add_argument('--log_file', type=str, default="logs/apm_logs/log.json",
    #                     help='output log file path')
    parser.add_argument('--host_ip', type=str, default='192.168.2.99',
                        help='pynq board IP')
    parser.add_argument('--num_graphs', type=int, default=10,
                        help='number of random graphs to generate')
    parser.add_argument('--min_depth', type=int, default=2,
                        help='minimum depth of network')
    parser.add_argument('--max_block_depth', type=int, default=4,
                        help='maximum depth of network')
    parser.add_argument('--bn_prob', type=float, default=0.5,
                        help='probability that bn follows convolution')
    parser.add_argument('--relu_prob', type=float, default=0.8,
                        help='probability that relu follows convolution')
    parser.add_argument('--maxpool_prob', type=float, default=0.2,
                        help='probability that maxpool follows convolution')
    parser.add_argument('--wkl_list', type=str, default='all',
                        help='wkl list to pick convolutions from. Options - pre, man, incep, all')
    parser.add_argument("--networks_log_file", type=str, default='profiling_results/random_graphs/networks_profiled.log',
                        help="path to file to store all the random graphs")
    parser.add_argument("--log_file_dir", type=str,
                        default='profiling_results/random_graphs/',
                        help="path to json file to store profiling data")
    parser.add_argument('--samples', type=int, default=10,
                        help='number of times to run each network')
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
    else:
        from Wkls import ALL_TUNED_WKLS

        wkl_list = ALL_TUNED_WKLS

    networks = []
    while len(networks) < args.num_graphs:
        networks.append(construct_random_graph(args, wkl_list))

    # if not os.path.exists(args.networks_log_file):
    with open(args.networks_log_file,'w+') as myfile:

        for i, network in enumerate(networks):
            myfile.write("network_{}::".format(i)+str(network)+"\n")
    # else:
    #     print('Networks log file already exists...',args.networks_log_file,"Exiting....")
    #     exit(99)


    connect_and_run(device='vta', networks=networks, log_file_dir=args.log_file_dir, host_ip=args.host_ip, num_samples=args.samples)


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
