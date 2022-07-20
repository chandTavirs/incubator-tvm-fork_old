import os
#from mxnet.gluon.model_zoo import vision
import numpy as np
from PIL import Image

from tvm import topi
import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import download
from tvm.autotvm.measure.measure_methods import request_remote
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm import record

import vta
from vta.testing import simulator
from vta.top import graph_pack
import copy
import torch

from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):  # @save
    """The Residual block of ResNet."""

    def __init__(self, input_channels, num_channels, strides=1):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        # Y = self.bn1(self.relu(self.conv1(X)))
        Y = self.conv1(X)
        Y = self.max_pool(Y)
        #         Y = self.bn2(self.relu(self.conv2(Y)))
        #         Y = self.max_pool(Y)

        #         return self.relu(Y)
        return Y


b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(input_channels, num_channels, num_residuals):
    blk = []
    for i in range(num_residuals):
        blk.append(Residual(num_channels, num_channels))
    return blk


b2 = nn.Sequential(*resnet_block(64, 64, 1))

net = nn.Sequential(b1, b2, nn.AdaptiveAvgPool2d((1, 1)))


def compile_network_torch(env, target, model, start_pack, stop_pack):
    input_name = "input0"

    # Populate the shape and data type dictionary for ImageNet classifier input
    dtype_dict = {input_name: "float32"}
    shape_dict = {input_name: (env.BATCH, 3, 224, 224)}

    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()

    shape_list = [(input_name, input_shape)]

    # Start front end compilation
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    # Update shape and type dictionary
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # Perform quantization in Relay
    # Note: We set opt_level to 3 in order to fold batch norm
    with relay.build_config(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)

    # Perform graph packing and constant folding for VTA target
    if target.device_name == "vta":
        assert env.BLOCK_IN == env.BLOCK_OUT
        relay_prog = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_IN,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name='cast',
            start_name_idx=11,
            stop_name='cast',
            stop_name_idx=19,
        )

    return relay_prog, params


# Tracker host and port can be set by your environment
tracker_host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1")
tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))

# Load VTA parameters from the vta/config/vta_config.json file
env = vta.get_env()

# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

# Name of Gluon model to compile
# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA.
network = net
start_pack = "nn.max_pool2d"
stop_pack = "nn.adaptive_avg_pool2d"

# Tuning option
log_file = "logs/tuning_logs/%s.maxpool.%s.log" % (device, "resnet")
tuning_option = {
    "log_filename": log_file,
    "tuner": "random",
    "n_trial": 1000,
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.RPCRunner(
            env.TARGET,
            host=tracker_host,
            port=tracker_port,
            number=5,
            timeout=60,
            # check_correctness=True, # TODO: re-enable when check_correctness works again.
        ),
    ),
}


def log_to_file(file_out, protocol="json"):
    """Log the tuning_logs records into file.
    The rows of the log are stored in the format of autotvm.record.encode.
    for lhs == rhs, we add an extra rhs = [] record

    Parameters
    ----------
    file_out : str
        The file to log to.
    protocol: str, optional
        The log protocol. Can be 'json' or 'pickle'

    Returns
    -------
    callback : callable
        Callback function to do the logging.
    """

    def _callback(_, inputs, results):
        with open(file_out, "a") as f:
            for inp, result in zip(inputs, results):
                f.write(record.encode(inp, result, protocol) + "\n")

                # we only consider task with same lhs and rhs
                if inp.task.args[0] == inp.task.args[1]:
                    args = list(inp.task.args)
                    args[1] = (args[0][0], (), args[0][2])
                    inp_copy = copy.deepcopy(inp)
                    inp_copy.task.args = tuple(args)
                    f.write(record.encode(inp_copy, result, protocol) + "\n")

    return _callback


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=10,
    early_stopping=None,
    log_filename="tuning_logs.log",
    use_transfer_learning=True,
):

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
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

        # do tuning_logs
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def register_vta_tuning_tasks():
    from tvm.autotvm.task import TaskExtractEnv

    @tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
    def my_clip(x, a_min, a_max):
        """Unlike topi's current clip, put min and max into two stages."""
        const_min = tvm.tir.const(a_min, x.dtype)
        const_max = tvm.tir.const(a_max, x.dtype)
        x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
        x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
        return x

    # init autotvm env to register VTA operator
    TaskExtractEnv()

    @autotvm.template("add.vta")
    def _topi_add(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        A, B = args[:2]

        with tvm.target.vta():
            res = vta.top.op.add_packed(*args, **kwargs)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            s = vta.top.op.schedule_add_packed([res])
        else:
            s = te.create_schedule([res.op])
        return s, [A, B, res]

    @autotvm.template("multiply.vta")
    def _topi_multiply(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        A, B = args[:2]

        with tvm.target.vta():
            res = vta.top.op.multiply_packed(*args, **kwargs)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            s = vta.top.op.schedule_multiply_packed([res])
        else:
            s = te.create_schedule([res.op])
        return s, [A, B, res]

    @autotvm.template("pooling_packed.vta")
    def _topi_max_pool2d(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        A = args[0]
        with tvm.target.vta():
            res = vta.top.pooling_packed(*args, **kwargs)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            s = vta.top.schedule_pooling_packed([res], layout=args[6])
        else:
            s = te.create_schedule([res.op])
        return s, [A, res]

def tune_and_evaluate(tuning_opt):

#     if env.TARGET != "intelfocl":
#         print("ALU only op only available for intelfocl target")
#         return

    # Register VTA tuning_logs tasks
    register_vta_tuning_tasks()

    # Perform task extraction on Relay program
    print("Extract tasks...")
    relay_prog, params = compile_network_torch(env, target, network, start_pack, stop_pack)
    mod = tvm.IRModule.from_expr(relay_prog)
    tasks = autotvm.task.extract_from_program(
        mod,
        params=params,
        ops=(
            relay.op.get("nn.max_pool2d"),
        ),
        target=tvm.target.Target(target),
    )
    #print(tasks)
    #print(platak)

    # filter out non-packed alu task
    tasks = list(filter(lambda t: len(t.args[0][1]) > 4, tasks))
    # filter out float alu task
    tasks = list(filter(lambda t: t.args[0][2] != "float32", tasks))

    # We should have extracted 10 convolution tasks
    tasks_set = {}
    print("Extracted {} alu tasks:".format(len(tasks)))
    for tsk in tasks:
        print("tsk = ", tsk)

#         if len(tsk.args[1][1]) == 0:
#             args = list(tsk.args)
#             args[1] = args[0]
#             tsk.args = tuple(args)

        if (tsk.name, tsk.args) in tasks_set:
            print("task {} already exists".format(tsk))
        tasks_set[(tsk.name, tsk.args)] = tsk

    tasks = list(tasks_set.values())
    print("After merged, final #tasks={}, tasks = {}".format(len(tasks), tasks))

    # run tuning_logs tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)


# Run the tuning_logs and evaluate the results
tune_and_evaluate(tuning_option)