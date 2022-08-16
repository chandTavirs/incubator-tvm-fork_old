from collections import namedtuple
from math import floor, ceil
from wkl_configs import *
import re
import argparse


CONV2D = Workload.__name__
MAXPOOL = maxPoolConfig.__name__

conv = re.compile("(?:Conv2DWorkload|MaxPool2DConfig)\(.*?\)")

wkl_regex = re.compile("([a-z\_]+)=([\d]+)")

def calc_conv_output_size(wkl):
    out_height = floor((wkl.height + 2 * wkl.hpad - wkl.hkernel) / wkl.hstride + 1)
    out_width = floor((wkl.width + 2 * wkl.wpad - wkl.wkernel) / wkl.wstride + 1)

    return out_height, out_width, out_height*out_width*wkl.out_filter


def calc_maxpool_output_size(wkl, pool_cfg):
    in_height, in_width, _ = calc_conv_output_size(wkl)

    if bool(pool_cfg.ceil_mode):
        floor_or_ceil = ceil
    else:
        floor_or_ceil = floor

    out_height = floor_or_ceil((in_height + 2 * pool_cfg.hpad - pool_cfg.hkernel) / pool_cfg.hstride + 1)
    out_width = floor_or_ceil((in_width + 2 * pool_cfg.wpad - pool_cfg.wkernel) / pool_cfg.wstride + 1)

    return out_height, out_width, out_height*out_width*wkl.out_filter



def calc_sizes(network):
    wkls = re.findall(conv, network)

    nodes = []
    for wkl in wkls:
        params = re.findall(wkl_regex, wkl)
        if "Conv2DWorkload" in wkl:
            nodes.append(Workload(int(params[0][1]), int(params[1][1]),
                                  int(params[2][1]), int(params[3][1]),
                                  int(params[4][1]), int(params[5][1]),
                                  int(params[6][1]), int(params[7][1]),
                                  int(params[8][1]), int(params[9][1]),
                                  int(params[10][1])))

        elif "MaxPool2DConfig" in wkl:
            nodes.append(maxPoolConfig(int(params[0][1]), int(params[1][1]),
                                       int(params[2][1]), int(params[3][1]),
                                       int(params[4][1]), int(params[5][1]),
                                       int(params[6][1])))

    output_sizes = []
    for i, node in enumerate(nodes):
        if type(node).__name__ == CONV2D:
            _, _, conv_size = calc_conv_output_size(node)
            output_sizes.append(conv_size)

        elif type(node).__name__ == MAXPOOL:
            _, _, maxp_size = calc_maxpool_output_size(nodes[i - 1], node)
            output_sizes.append(maxp_size)

    return output_sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Network output sizes calculator')
    parser.add_argument("--input_log_file", type=str, default='profiling_results/random_graphs/graphs_500/networks_profiled.log',
                        help="path to file containing profiled network structures")
    parser.add_argument("--output_log_file", type=str, default='profiling_results/random_graphs/graphs_500/networks_profiled_output_sizes.log',
                        help="path to file to store output sizes of networks")

    args = parser.parse_args()

    results = []
    with open(args.input_log_file,"r") as myfile:
        for line in myfile.readlines():
            network_name, network = line.split("::")
            results.append((network_name, calc_sizes(network)))

    with open(args.output_log_file, "w+") as myfile:
        for result in results:
            myfile.write(result[0]+"::"+str(result[1])+"\n")
