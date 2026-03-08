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
"""
.. _tune_relay_arm:

Auto-tuning a Convolutional Network for ARM CPU
===============================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, `Zhao Wu <https://github.com/FrozenGene>`_, `Eddie Yan <https://github.com/eqy>`_

Auto-tuning for a specific ARM device is critical for getting the best
performance. This is a tutorial about how to tune a whole convolutional
network.

The operator implementation for ARM CPU in TVM is written in template form.
The template has many tunable knobs (tile factor, vectorization, unrolling, etc).
We will tune all convolution and depthwise convolution operators
in the neural network. After tuning, we produce a log file which stores
the best knob values for all required operators. When the TVM compiler compiles
these operators, it will query this log file to get the best knob values.

We also released pre-tuned parameters for some arm devices. You can go to
`ARM CPU Benchmark <https://github.com/apache/tvm/wiki/Benchmark#arm-cpu>`_
to see the results.

Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""

######################################################################
# Install dependencies
# --------------------
# To use the autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado cloudpickle
#
# To make TVM run faster during tuning, it is recommended to use cython
# as FFI of TVM. In the root directory of TVM, execute
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.


import os
import sys

import numpy as np
import torchvision

import tvm
import vta
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.utils import tempdir
from tvm.contrib import graph_runtime, utils, download
from resnet_custom_model import resnet18_base, resnet18_base_orig_dense
import torch
from tvm import topi

#################################################################
# Explicitly Define Tasks to Tune
#################################################################

def create_explicit_tasks(target):
    """
    Create a list of tasks to tune explicitly.

    Returns:
        List of Task objects to tune
    """
    tasks = []

    # Dense tasks for (10, 512)
    dense_512_args = (('TENSOR', (1, 512), 'float32'), ('TENSOR', (10, 512), 'float32'), None, 'float32')

    tasks.append(autotvm.task.create(
        'dense_nopack.x86',
        args=dense_512_args,
        target=target
    ))

    tasks.append(autotvm.task.create(
        'dense_pack.x86',
        args=dense_512_args,
        target=target
    ))

    # Dense tasks for (10, 256)
    dense_256_args = (('TENSOR', (1, 256), 'float32'), ('TENSOR', (10, 256), 'float32'), None, 'float32')

    tasks.append(autotvm.task.create(
        'dense_nopack.x86',
        args=dense_256_args,
        target=target
    ))

    tasks.append(autotvm.task.create(
        'dense_pack.x86',
        args=dense_256_args,
        target=target
    ))

    # Conv2d task with (1, 3, 224, 224) input and (128, 3, 7, 7) weights
    conv2d_args = (
        ('TENSOR', (1, 3, 224, 224), 'float32'),
        ('TENSOR', (128, 3, 7, 7), 'float32'),
        (2, 2),  # stride
        (3, 3, 3, 3),  # padding
        (1, 1),  # dilation
        'float32'
    )

    tasks.append(autotvm.task.create(
        'conv2d_nchw_spatial_pack.arm_cpu',
        args=conv2d_args,
        target=target
    ))

    return tasks

import json
from typing import Dict, Any

candidate_set_json_path="/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/candidates_25_85.json"
arch_config_json_path="/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json"

global_tasks = []
def load_arch_mapping(path: str) -> Dict[str, Any]:
    """Load architectures JSON and normalize to a mapping {id: architecture_dict}.

    Supported input formats:
    - A dict mapping id -> architecture dict (legacy)
    - A dict with key 'architectures' containing a list of items with fields 'id' and 'architecture'
    - A top-level list of items with 'id' and 'architecture'
    """
    with open(path, 'r') as f:
        data = json.load(f)

    # Case 1: already a mapping from id -> arch
    if isinstance(data, dict) and all(isinstance(v, dict) and 'residual_depth_list' in v for v in data.values()):
        return data

    # Case 2: top-level dict with 'architectures' list
    if isinstance(data, dict) and 'architectures' in data and isinstance(data['architectures'], list):
        mapping = {}
        for item in data['architectures']:
            # item may contain fields 'id' and 'architecture' (nested)
            if 'id' in item and 'architecture' in item:
                mapping[item['id']] = item['architecture']
            elif 'id' in item and 'arch' in item:
                mapping[item['id']] = item['arch']
            else:
                # If the item itself is an architecture dict without id, generate an id
                if 'id' in item:
                    mapping[item['id']] = item
        return mapping

    # Case 3: top-level list of architecture items
    if isinstance(data, list):
        mapping = {}
        for idx, item in enumerate(data):
            if isinstance(item, dict) and 'id' in item and 'architecture' in item:
                mapping[item['id']] = item['architecture']
            elif isinstance(item, dict) and 'id' in item:
                mapping[item['id']] = item
            else:
                mapping[f'arch_{idx}'] = item
        return mapping

    # Fallback: unknown format, raise error
    raise ValueError(f"Unsupported architecture file format: {path}")

arch_mapping = load_arch_mapping(arch_config_json_path)
# read candidate set json
with open(candidate_set_json_path, 'r') as f:
    candidate_set = json.load(f)
model_ids = candidate_set['selected_model_ids']

# Robust import of external ofa_base_models without being shadowed by local folder
external_repo_root = "/home/srchand/Desktop/research/OFA_Obfs"
if external_repo_root not in sys.path:
    sys.path.insert(0, external_repo_root)

# If a local shim package is already cached, evict it to allow importing the external one
_mod = sys.modules.get("ofa_base_models")
_mod2 = sys.modules.get("architecture_defense")

if _mod is not None:
    try:
        _mod_file = getattr(_mod, "__file__", "") or ""
        if "TVM_Intel_Fork/tvm/vta/sri_scripts/jupyter_nbs/ofa_base_models" in _mod_file:
            del sys.modules["ofa_base_models"]
    except Exception:
        # If anything goes wrong, clear the cache entry
        sys.modules.pop("ofa_base_models", None)

if _mod2 is not None:
    try:
        _mod2_file = getattr(_mod2, "__file__", "") or ""
        if "TVM_Intel_Fork/tvm/vta/sri_scripts/jupyter_nbs/architecture_defense" in _mod2_file:
            del sys.modules["architecture_defense"]
    except Exception:
        # If anything goes wrong, clear the cache entry
        sys.modules.pop("architecture_defense", None)

try:
    from ofa_base_models import OFADynamicResnetAllMod  # type: ignore
    from architecture_defense import StaticResNetFromArch  # type: ignore
except (ModuleNotFoundError, ImportError):
    # Final fallback: ensure external root is first in path and retry once
    if sys.path[0] != external_repo_root:
        sys.path.insert(0, external_repo_root)
    # Clear any cached partial imports
    sys.modules.pop("ofa_base_models", None)
    sys.modules.pop("architecture_defense", None)
    from ofa_base_models import OFADynamicResnetAllMod  # type: ignore
    from architecture_defense import StaticResNetFromArch  # type: ignore
else:
    # Print where the modules are loaded from for debugging/IDE clarity
    import ofa_base_models as _obm  # type: ignore
    print("ofa_base_models loaded from:", getattr(_obm, "__file__", None))
    try:
        import architecture_defense as _ad  # type: ignore
        print("architecture_defense loaded from:", getattr(_ad, "__file__", None))
    except ImportError:
        pass

ofa_net = OFADynamicResnetAllMod()
model_path = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth"
device_temp = torch.device('cpu')
checkpoint = torch.load(model_path, map_location=device_temp)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state = checkpoint['model_state_dict']
else:
    state = checkpoint
# Allow mismatched keys due to local shim modules
ofa_net.load_state_dict(state, strict=False)
# with torch.no_grad():
#     summary(net, input_size=(1, 3, 224, 224))

# network = ofa_net
# chkpt_path = f'/mnt/hgfs/vmware_ubuntu_sf/adv_attack_checkpoints/model_checkpoints/resnet18_scratch_imagenette_dual_gpu.pth'
# state_dict = torch.load(chkpt_path, map_location=torch.device('cpu'))
# state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
# network.load_state_dict(state_dict)

#################################################################
# Define network
# --------------
# First we need to define the network in relay frontend API.
# We can load some pre-defined network from :code:`relay.testing`.
# We can also load models from MXNet, ONNX and TensorFlow.


def get_network(name, standalone_net, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 10)

    if 'candidate_set' in name:
        input_name = "input0"
        input_data = torch.randn(input_shape)
        standalone_net.eval()
        # pytorch_model = getattr(torchvision.models, "resnet18")(pretrained=True).eval()
        scripted_model = torch.jit.trace(standalone_net, input_data)
        # scripted_model = torch.jit.script(pytorch_model)
        shape_list = [(input_name, input_shape)]

        # Start front end compilation
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


#################################################################
# Start RPC Tracker
# -----------------
# TVM uses RPC session to communicate with ARM boards.
# During tuning, the tuner will send the generated code to the board and
# measure the speed of code on the board.
#
# To scale up the tuning, TVM uses RPC Tracker to manage distributed devices.
# The RPC Tracker is a centralized controller node. We can register all devices to
# the tracker. For example, if we have 10 phones, we can register all of them
# to the tracker, and run 10 measurements in parallel, accelerating the tuning process.
#
# To start an RPC tracker, run this command on the host machine. The tracker is
# required during the whole tuning process, so we need to open a new terminal for
# this command:
#
# .. code-block:: bash
#
#   python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
#
# The expected output is
#
# .. code-block:: bash
#
#   INFO:RPCTracker:bind to 0.0.0.0:9190

#################################################################
# Register Devices to RPC Tracker
# -----------------------------------
# Now we can register our devices to the tracker. The first step is to
# build the TVM runtime for the ARM devices.
#
# * For Linux:
#   Follow this section :ref:`build-tvm-runtime-on-device` to build
#   the TVM runtime on the device. Then register the device to tracker by
#
#   .. code-block:: bash
#
#     python -m tvm.exec.rpc_server --tracker=[HOST_IP]:9190 --key=rk3399
#
#   (replace :code:`[HOST_IP]` with the IP address of your host machine)
#
# * For Android:
#   Follow this `readme page <https://github.com/apache/tvm/tree/main/apps/android_rpc>`_ to
#   install the TVM RPC APK on the android device. Make sure you can pass the android rpc test.
#   Then you have already registered your device. During tuning, you have to go to developer option
#   and enable "Keep screen awake during changing" and charge your phone to make it stable.
#
# After registering devices, we can confirm it by querying rpc_tracker
#
# .. code-block:: bash
#
#   python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
#
# For example, if we have 2 Huawei mate10 pro, 11 Raspberry Pi 3B and 2 rk3399,
# the output can be
#
# .. code-block:: bash
#
#    Queue Status
#    ----------------------------------
#    key          total  free  pending
#    ----------------------------------
#    mate10pro    2      2     0
#    rk3399       2      2     0
#    rpi3b        11     11    0
#    ----------------------------------
#
# You can register multiple devices to the tracker to accelerate the measurement in tuning.

###########################################
# Set Tuning Options
# ------------------
# Before tuning, we should apply some configurations. Here I use an RK3399 board
# as example. In your setting, you should modify the target and device_key accordingly.
# set :code:`use_android` to True if you use android phone.

#### DEVICE CONFIG ####

# Replace "aarch64-linux-gnu" with the correct target of your board.
# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.



# You can skip the implementation of this function for this tutorial.
def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        print("================= Tuning Task %d/%d: %s =================" % (i + 1, len(tasks), str(tsk)))
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb":
            tuner_obj = XGBTuner(tsk, loss_type="reg")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="curve")
        elif tuner == "xgb_rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_rank_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_rank_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_rank_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
        elif tuner == "xgb_rank_binary":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary")
        elif tuner == "xgb_rank_binary_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="knob")
        elif tuner == "xgb_rank_binary_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="itervar")
        elif tuner == "xgb_rank_binary_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # process tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.


def tune_and_evaluate(tuning_opt, network_name, standalone_net):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, _ = get_network(network_name, standalone_net,  batch_size=1)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"), relay.op.get("nn.dense"))
    )
    final_tasks = []
    for task in tasks:
        if ('dense' in task.name or 'conv2d_nchw_spatial_pack' in task.name) and (task.args[1][1] == (10, 1024) or task.args[1][1] == (10, 512) or task.args[1][1] == (10, 256) or 224 in task.args[0][1]) and (task not in global_tasks):
            # add the dense layer to final tasks
            final_tasks.append(task)
            global_tasks.append(task)
        else:
            continue

    tasks = create_explicit_tasks(target)


    # take tasks 0, 1 and 20
    # tasks = [tasks[1]]
    # run tuning tasks
    if len(tasks) == 0:
        print("No tasks to tune for this network.")
        return
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # # compile kernels with history best records
    # with autotvm.apply_history_best(log_file):
    #     print("Compile...")
    #     with tvm.transform.PassContext(opt_level=3):
    #         lib = relay.build_module.build(mod, target=target, params=params)
    #
    #     # export library
    #     tmp = tempdir()
    #     if use_android:
    #         from tvm.contrib import ndk
    #
    #         filename = "net.so"
    #         lib.export_library(tmp.relpath(filename), fcompile=ndk.create_shared)
    #     else:
    #         filename = "net.tar"
    #         lib.export_library(tmp.relpath(filename))
    #
    #     # upload module to device
    #     print("Upload...")
    #     remote = autotvm.measure.request_remote(device_key, "127.0.0.1", 9190, timeout=10000)
    #     remote.upload(tmp.relpath(filename))
    #     rlib = remote.load_module(filename)
    #
    #     # upload parameters to device
    #     dev = remote.device(str(target), 0)
    #     module = graph_runtime.GraphModule(rlib["default"](dev))
    #     data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    #     module.set_input("data", data_tvm)
    #
    #     # evaluate
    #     print("Evaluate inference time cost...")
    #     print(module.benchmark(dev, number=1, repeat=10))


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.
for i_net, id in enumerate(model_ids[1:]):
    target = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu")
    # target = tvm.target.Target("llvm -device=arm_cpu -mtriple=armv7a-linux-eabi")
    # Also replace this with the device key in your tracker
    device_key = "zcu104"

    # Set this to True if you use android phone
    use_android = False

    #### TUNING OPTION ####
    network_name = f"candidate_set_{id}"
    log_file = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/logs/tuning_logs/vta_1x16x16/candidate_set/arm_cpu_candidate_set_remaining_%s.%s.log" % (
        device_key, network_name)
    dtype = "float32"

    tuning_option = {
        "log_filename": log_file,
        "tuner": "xgb",
        "n_trial": 1000,
        "early_stopping": 800,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(build_func="ndk" if use_android else "default"),
            runner=autotvm.RPCRunner(
                device_key,
                host="127.0.0.1",
                port=9190,
                number=5,
                timeout=10,
            ),
        ),
    }
    ofa_net.set_active_subnet(arch_mapping[id])
    standalone_net = StaticResNetFromArch(
        target_arch=arch_mapping[id],
        num_classes=10,
        width_mult_list=(0.5, 1.0, 2.0)
    )
    standalone_net.load_weights_from_ofa_checkpoint(checkpoint_path=model_path, ofa_model=ofa_net)
    print('Tuning network %d/%d: %s' % (i_net+1, len(model_ids), network_name))
    tune_and_evaluate(tuning_option, network_name, standalone_net)

######################################################################
# Sample Output
# -------------
# The tuning needs to compile many programs and extract feature from them.
# So a high performance CPU is recommended.
# One sample output is listed below.
# It takes about 2 hours on a 32T AMD Ryzen Threadripper.
#
# .. code-block:: bash
#
#    Extract tasks...
#    Tuning...
#    [Task  1/12]  Current/Best:   22.37/  52.19 GFLOPS | Progress: (544/1000) | 406.59 s Done.
#    [Task  2/12]  Current/Best:    6.51/  18.77 GFLOPS | Progress: (608/1000) | 325.05 s Done.
#    [Task  3/12]  Current/Best:    4.67/  24.87 GFLOPS | Progress: (480/1000) | 372.31 s Done.
#    [Task  4/12]  Current/Best:   11.35/  46.83 GFLOPS | Progress: (736/1000) | 602.39 s Done.
#    [Task  5/12]  Current/Best:    1.01/  19.80 GFLOPS | Progress: (448/1000) | 262.16 s Done.
#    [Task  6/12]  Current/Best:    2.47/  23.76 GFLOPS | Progress: (672/1000) | 563.85 s Done.
#    [Task  7/12]  Current/Best:   14.57/  33.97 GFLOPS | Progress: (544/1000) | 465.15 s Done.
#    [Task  8/12]  Current/Best:    1.13/  17.65 GFLOPS | Progress: (576/1000) | 365.08 s Done.
#    [Task  9/12]  Current/Best:   14.45/  22.66 GFLOPS | Progress: (928/1000) | 724.25 s Done.
#    [Task 10/12]  Current/Best:    3.22/  15.36 GFLOPS | Progress: (864/1000) | 564.27 s Done.
#    [Task 11/12]  Current/Best:   11.03/  32.23 GFLOPS | Progress: (736/1000) | 635.15 s Done.
#    [Task 12/12]  Current/Best:    8.00/  21.65 GFLOPS | Progress: (1000/1000) | 1111.81 s Done.
#    Compile...
#    Upload...
#    Evaluate inference time cost...
#    Mean inference time (std dev): 162.59 ms (0.06 ms)

######################################################################
#
# .. note:: **Experiencing Difficulties?**
#
#   The auto tuning module is error-prone. If you always see " 0.00/ 0.00 GFLOPS",
#   then there must be something wrong.
#
#   First, make sure you set the correct configuration of your device.
#   Then, you can print debug information by adding these lines in the beginning
#   of the script. It will print every measurement result, where you can find useful
#   error messages.
#
#   .. code-block:: python
#
#      import logging
#      logging.getLogger('autotvm').setLevel(logging.DEBUG)
#
#   Finally, always feel free to ask our community for help on https://discuss.tvm.apache.org
