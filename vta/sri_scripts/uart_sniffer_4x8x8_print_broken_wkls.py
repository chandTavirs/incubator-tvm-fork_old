import os
from wkl_configs import *


def print_broken_workloads(log_dir):
    # Print broken workloads
    broken_workloads = []
    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):
            with open(os.path.join(log_dir, filename), "r") as f:
                lines = f.readlines()
                num_data_lines = len(lines) - 1
                if num_data_lines != 1:
                    wkl_name = filename.split("conv_")[1].split(".log")[0].split("_sample")[0]
                    wkl_names = wkl_name.split("_")
                    wkl_names = [int(x) for x in wkl_names]
                    if len(wkl_names) == 10:
                        wkl = Workload(1, *wkl_names)
                        broken_workloads.append(wkl)

    broken_workloads = list(set(broken_workloads))
    for i, wkl in enumerate(broken_workloads):
        print(
            f'(\'workloads_{i}\', Workload({wkl.batch}, {wkl.height}, {wkl.width}, {wkl.in_filter}, {wkl.out_filter}, {wkl.hkernel}, {wkl.wkernel}, {wkl.hpad}, {wkl.wpad}, {wkl.hstride}, {wkl.wstride})),')


log_dir = "uart_sniffer_data/asp_dac/convs/4x8x8/broken_wkls"
print_broken_workloads(log_dir)
