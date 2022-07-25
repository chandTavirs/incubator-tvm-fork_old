from tuned_wkls import *
from math import floor, ceil
from random import randrange, random
from max_pool_cfgs import *
import argparse

def calc_conv_output_shape(wkl):
    out_height = floor((wkl.height + 2*wkl.hpad - wkl.hkernel)/wkl.hstride + 1)
    out_width = floor((wkl.width + 2 * wkl.wpad - wkl.wkernel) / wkl.wstride + 1)

    return out_height, out_width

def calc_maxpool_output_shape(wkl, pool_cfg):

    in_height, in_width = calc_conv_output_shape(wkl)

    if bool(pool_cfg.ceil_mode):
        floor_or_ceil = ceil
    else:
        floor_or_ceil = floor

    out_height = floor_or_ceil((in_height + 2*pool_cfg.hpad - pool_cfg.hkernel)/pool_cfg.hstride + 1)
    out_width = floor_or_ceil((in_width + 2 * pool_cfg.wpad - pool_cfg.wkernel)/pool_cfg.wstride + 1)

    return out_height, out_width


def construct_random_graph(args):
    layer_count = 0
    while layer_count <= args.max_depth:
        conv_wkl = PRE_TUNED_50[randrange(len(PRE_TUNED_50))]
        if conv_wkl.height < 26:
            continue





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Random compute graph builder')
    # parser.add_argument('--log_file', type=str, default="logs/apm_logs/log.json",
    #                     help='output log file path')
    # parser.add_argument('--host_ip', type=str, default='192.168.2.99',
    #                     help='pynq board IP')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='maximum depth of network')
    parser.add_argument('--max_block_depth', type=int, default=4,
                        help='maximum depth of network')
    parser.add_argument('--bn_prob', type=float, default=0.5,
                        help='probability that bn follows convolution')
    parser.add_argument('--relu_prob', type=float, default=1.0,
                        help='probability that relu follows convolution')
    parser.add_argument('--maxpool_prob', type=float, default=0.5,
                        help='probability that maxpool follows convolution')

    args = parser.parse_args()

    model = construct_random_graph(args)


