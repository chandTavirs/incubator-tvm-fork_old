from wkl_configs import *
import re
# path to the tuning log
tuning_log_path = 'logs/tuning_logs/vta_1x16x16/autotvm_model_10_obf_pruned.log'

"""sample line from tuning log::
{"input": ["ext_dev -keys=vta,cpu -device=vta -model=zcu104_1x16x16_i8w8a32_15_15_18_17", "conv2d_packed.vta", [["TENSOR", [1, 32, 7, 7, 1, 16], "int8"], ["TENSOR", [32, 32, 3, 3, 16, 16], "int8"], [1, 1], [1, 1, 1, 1], [1, 1], "NCHW1n16c", "int32"], {}], "config": {"index": 241, "code_hash": null, "entity": [["tile_b", "sp", [-1, 1]], ["tile_h", "sp", [-1, 7]], ["tile_w", "sp", [-1, 1]], ["tile_ci", "sp", [-1, 1]], ["tile_co", "sp", [-1, 16]], ["oc_nthread", "ot", 2], ["h_nthread", "ot", 1]]}, "result": [[0.0096332996], 0, 1.5688602924346924, 1740456275.4953175], "version": 0.2, "tvm_version": "0.8.dev0"}
input feature map shape: [1, 32, 7, 7, 1, 16] in VTA layout NCHW1n16c. [1, 512, 7, 7] in NHWC layout. Number of channels = 512, height = 7, width = 7
convolution kernel shape: [32, 32, 3, 3, 16, 16] in VTA layout NCHW1n16c. [3, 3, 512, 512] in NHWC layout. Number of input channels = 512, number of output channels = 512, kernel height = 3, kernel width = 3
"""

tensor_re = re.compile(r"\"TENSOR\", \[(\d+), (\d+), (\d+), (\d+), (\d+), (\d+)\]")
stride_padding_re = re.compile(r"\[(\d+), (\d+)\], \[(\d+), (\d+), (\d+), (\d+)\], \[(\d+), (\d+)\], \"NCHW1n16c\", \"int32\"\]")



# read the tuning log
with open(tuning_log_path, 'r') as f:
    tuning_log = f.readlines()

# extract the workloads from the tuning log
workloads = []
for line in tuning_log:
    tensor_matches = re.findall(tensor_re, line)
    if tensor_matches:
        input_tensor_match = tensor_matches[0]
        convolution_tensor_match = tensor_matches[1]
        input_shape = [int(x) for x in input_tensor_match]
        assert len(input_shape) == 6
        nchw_input_shape = [input_shape[0], input_shape[1]*input_shape[5], input_shape[2], input_shape[3]]

        convolution_shape = [int(x) for x in convolution_tensor_match]
        assert len(convolution_shape) == 6
        nchw_convolution_shape = [convolution_shape[2], convolution_shape[3], convolution_shape[1]*convolution_shape[5], convolution_shape[0]*convolution_shape[4]]

        stride_padding_matches = re.findall(stride_padding_re, line)
        stride_padding = [int(x) for x in stride_padding_matches[0]]
        assert len(stride_padding) == 8
        batch = input_shape[0]
        height = nchw_input_shape[2]
        width = nchw_input_shape[3]
        in_filter = nchw_input_shape[1]
        out_filter = nchw_convolution_shape[3]
        hkernel = nchw_convolution_shape[0]
        wkernel = nchw_convolution_shape[1]
        hpad = stride_padding[2]
        wpad = stride_padding[3]
        hstride = stride_padding[0]
        wstride = stride_padding[1]
        workloads.append(Workload(batch, height, width, in_filter, out_filter, hkernel, wkernel, hpad, wpad, hstride, wstride))

for i, wkl in enumerate(workloads):
    print(f"(\'workload_{i}\', Workload({wkl.batch}, {wkl.height}, {wkl.width}, {wkl.in_filter}, {wkl.out_filter}, {wkl.hkernel}, {wkl.wkernel}, {wkl.hpad}, {wkl.wpad}, {wkl.hstride}, {wkl.wstride})),")