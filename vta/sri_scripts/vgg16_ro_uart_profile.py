from __future__ import absolute_import, print_function

import argparse, json, os, requests, sys, time
import csv
from io import BytesIO
from multiprocessing import Process, Value
from os.path import join, isfile

import serial
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt

import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_runtime, utils, download
from tvm.contrib.debugger import debug_runtime
from tvm.relay import transform

import vta
from vta.testing import simulator
from vta.top import graph_pack

import torch
import torchvision
from tvm.contrib.download import download_testdata

import pickle





from torch import nn
from torch.nn import functional as F

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for i in range(num_convs):
        if i == 0:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

class VGG():
    def __init__(self, arch):
        super().__init__()
        conv_blks = []
        for (num_convs, in_channels, out_channels) in arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        self.net = nn.Sequential(
            *conv_blks, nn.AdaptiveAvgPool2d((1,1)))
def write_password(ser):
    ser.write(b'password\n')

def close_ser(ser):
    ser.close()

# ro_from_ser = bytearray()

# def read_serial_port(ser):
#     flag = False
#     with open('/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/ro_uart_logs/working/resnet18_no_el_333hz_12m_334_non_block.log', 'wb') as myfile:
#         while True:
#             # byte_line = ser.read(2048)
#             # if len(byte_line) > 0:
#             #     flag = True
#             #     #print(byte_line)
#             #     myfile.write(byte_line)
#             # elif flag:
#             #     break
#             if ser.in_waiting > 0:
#                 # flag = True
#                 read_bytes = ser.read(ser.in_waiting)
#                 myfile.write(read_bytes)

shared_flag = Value('i', 0)

def read_serial_port(ser):
    flag = False
    bytes_read = bytearray()
    # with open('/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/ro_uart_logs/working/raw_no_el_12m_334_non_block.log', 'wb') as myfile:
    while True:
        # byte_line = ser.read(9192)
        # if len(byte_line) > 0:
        #     flag = True
        #     #print(byte_line)
        #     myfile.write(byte_line)
        # elif flag:
        #     break
        # read = ser.read(9192)
        if ser.in_waiting > 0 and shared_flag.value == 0:
            # flag = True
            read = ser.read(ser.in_waiting)
            bytes_read.extend(read)
            # elif flag:
            #     break
        elif shared_flag.value == 1:
            with open('/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/ro_uart_logs/working/vgg16_ro_with_axi_6m_667_try.log', 'wb') as myfile:
                myfile.write(bytes_read)
            break


port = '/dev/ttyUSB3'
baud = 6000000

# Make sure that TVM was compiled with RPC=1
assert tvm.runtime.enabled("rpc")

# Load VTA parameters from the 3rdparty/vta-hw/config/vta_config.json file
env = vta.get_env()

# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

# Dictionary lookup for when to start/end bit packing
pack_dict = {
#     "resnet18_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet18": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
    "resnet34": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
    "resnet50": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
    "resnet101": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
    "vgg11": ["nn.max_pool2d", "nn.dense"],
    "vgg16":    ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
    "resnet34_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet18_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet50_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet101_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "mobilenetv2_1.0": ["nn.max_pool2d", "nn.global_avg_pool2d"]
}

# Name of Gluon model to compile
# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA.
#model = "resnet18_v1"
#assert model in pack_dict
model = "vgg16"

remote = None
if env.TARGET not in ["sim", "tsim", "intelfocl"]:

    # Get remote from tracker node if environment variable is set.
    # To set up the tracker, you'll need to follow the "Auto-tuning
    # a convolutional network for VTA" tutorial.
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = os.environ.get("TVM_TRACKER_PORT", None)
    # Otherwise if you have a device you want to program directly from
    # the host, make sure you've set the variables below to the IP of
    # your board.
#     device_host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")
    device_host="192.168.2.99"
    # device_host="10.100.86.111"
    device_port = os.environ.get("VTA_RPC_PORT", "9091")
    if not tracker_host or not tracker_port:
        remote = rpc.connect(device_host, int(device_port))
    else:
        remote = autotvm.measure.request_remote(
            env.TARGET, tracker_host, int(tracker_port), timeout=10000
        )

    # Reconfigure the JIT runtime and FPGA.
    # You can program the FPGA with your own custom bitstream
    # by passing the path to the bitstream file instead of None.
    reconfig_start = time.time()
    vta.reconfig_runtime(remote)
    # vta.program_fpga(remote, bitstream="/mnt/hgfs/vmware_ubuntu_sf/bitstreams/vta_no_el_100hz_12m.bit")
    reconfig_time = time.time() - reconfig_start
    print("Reconfigured FPGA and RPC runtime in {0:.2f}s!".format(reconfig_time))
#
# In simulation mode, host the RPC server locally.
else:
    remote = rpc.LocalSession()

    if env.TARGET in ["intelfocl"]:
        # program intelfocl aocx
        vta.program_fpga(remote, bitstream="vta.bitstream")

# Get execution context from remote
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)

net1 = VGG(arch=((1, 3, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))).net

import glob
schedule_log_files = glob.glob(r'logs/tuning_logs/*.log')

with autotvm.tophub.context(target, extra_files=schedule_log_files):
    input_name = "input0"

    # Populate the shape and data type dictionary for ImageNet classifier input
    dtype_dict = {input_name: "float32"}
    shape_dict = {input_name: (env.BATCH, 3, 224, 224)}


    # pytorch_model = getattr(torchvision.models, model)(pretrained=True)

    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(net1, input_data).eval()

    shape_list = [(input_name, input_shape)]

    # Measure build start time
    build_start = time.time()

    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    if target.device_name == "vta":
        # Perform quantization in Relay
        # Note: We set opt_level to 3 in order to fold batch norm
        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
                mod = relay.quantize.quantize(mod, params=params)
            assert env.BLOCK_IN == env.BLOCK_OUT
            # do device annotation if target is intelfocl or sim
            relay_prog = graph_pack(
                mod["main"],
                env.BATCH,
                env.BLOCK_IN,
                env.BLOCK_OUT,
                env.WGT_WIDTH,
                start_name=pack_dict[model][0],
                stop_name=pack_dict[model][1],
                device_annot=(env.TARGET == "intelfocl"),
            )
    else:
        relay_prog = mod["main"]

    # Compile Relay program with AlterOpLayout disabled
    if target.device_name != "vta":
        with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
            graph, lib, params = relay.build(
                relay_prog, target=target, params=params, target_host=env.target_host
            )
    else:
        if env.TARGET == "intelfocl":
            # multiple targets to run both on cpu and vta
            target = {"cpu": env.target_vta_cpu, "ext_dev": target}
        with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
            graph, lib, params = relay.build(
                relay_prog, target=target, params=params, target_host=env.target_host
            )

        # pickle_file_path = '/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/resnet_lib'

        # with open(os.path.join(pickle_file_path,'graph'), 'wb') as file:
        #     pickle.dump(graph, file)
        #     file.close()
        #
        # with open(os.path.join(pickle_file_path,'lib'), 'wb') as file:
        #     pickle.dump(lib, file)
        #     file.close()



    # Measure Relay build time
    build_time = time.time() - build_start
    print(model + " inference graph built in {0:.2f}s!".format(build_time))

    # Send the inference library over to the remote RPC server
    temp = utils.tempdir()
    lib.export_library(temp.relpath("graphlib.tar"))
    remote.upload(temp.relpath("graphlib.tar"))
    lib = remote.load_module("graphlib.tar")

    if env.TARGET == "intelfocl":
        ctxes = [remote.ext_dev(0), remote.cpu(0)]
        m = graph_runtime.create(graph, lib, ctxes)
    else:
        # Graph runtime
        m = graph_runtime.create(graph, lib, ctx)


categ_url = "https://github.com/uwsampl/web-data/raw/main/vta/models/"
categ_fn = "synset.txt"
download.download(join(categ_url, categ_fn), categ_fn)
synset = eval(open(categ_fn).read())

# Download test image
# image_url = "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg"
image_fn = "jupyter_nbs/pug.jpg"
#download.download(image_url, image_fn)

# Prepare test image for inference
image = Image.open(image_fn).resize((224, 224))
# plt.imshow(image)
# plt.show()
image = np.array(image) - np.array([123.0, 117.0, 104.0])
image /= np.array([58.395, 57.12, 57.375])
image = image.transpose((2, 0, 1))
image = image[np.newaxis, :]
image = np.repeat(image, env.BATCH, axis=0)

# Set the network parameters and inputs
m.set_input(**params)
m.set_input(input_name, image)

num = 9  # number of times we run module for a single measurement
rep = 1  # number of measurements (we derive std dev from this)


timer = m.module.time_evaluator("run", ctx, number=num, repeat=rep)

ser = serial.Serial(port, baud)
ser.timeout = 0


read_process = Process(target=read_serial_port, args=(ser, ))
write_password(ser)
read_process.start()

# time.sleep(0.5)

# tcost = timer()

m.run()
time.sleep(0.01)
shared_flag.value = 1

write_password(ser)

read_process.join(5)
if read_process.is_alive():
    print("terminating")
    read_process.terminate()

# with open('/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/ro_uart_logs/working/resnet18_no_el_333hz_12m_334_non_block.log', 'wb') as myfile:
#     myfile.write(ro_from_ser)

close_ser(ser)

# tvm_output = m.get_output(0, tvm.nd.empty((env.BATCH, 1000), "float32", remote.cpu(0)))
# for b in range(env.BATCH):
#     top_categories = np.argsort(tvm_output.asnumpy()[b])
#     # Report top-5 classification results
#     print("\n{} prediction for sample {}".format(model, b))
#     print("\t#1:", synset[top_categories[-1]])
#     print("\t#2:", synset[top_categories[-2]])
#     print("\t#3:", synset[top_categories[-3]])
#     print("\t#4:", synset[top_categories[-4]])
#     print("\t#5:", synset[top_categories[-5]])
    # This just checks that one of the 5 top categories
    # is one variety of cat; this is by no means an accurate
    # assessment of how quantization affects classification
    # accuracy but is meant to catch changes to the
    # quantization pass that would accuracy in the CI.




