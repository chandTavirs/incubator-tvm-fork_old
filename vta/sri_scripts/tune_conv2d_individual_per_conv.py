import argparse

from collections import namedtuple

import tvm
from tvm import te
from tvm import relay
from tvm import autotvm

from tvm import topi
import tvm.topi.testing
import vta
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

import vta.testing

import os
from torch import nn
import torch
from vta.top import graph_pack
from Wkls import ALL_TUNED_WKLS as pynq_wkls
from Wkls import MOBILENET_V2 as mnet_wkls
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

broken_wkls = []

# Get batch info from env
env = vta.get_env()

# tune_wkls = [
#     ('workloads_0', Workload(1, 149, 149, 32, 32, 3, 3, 0, 0, 1, 1)),
# ]

def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
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
                autotvm.callback.log_to_file(tmp_log_file),
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

    @autotvm.template("conv2d_packed.vta")
    def _topi_nn_conv2d(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        A, W = args[:2]

        with tvm.target.vta():
            res = vta.top.conv2d_packed(*args, **kwargs)
            res = topi.right_shift(res, 8)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            s = vta.top.schedule_conv2d_packed([res])
        else:
            s = te.create_schedule([res.op])
        return s, [A, W, res]

def construct_tasks(env, wl, task_name='conv2d_packed.vta'):
    assert wl.hpad == wl.wpad

    net = nn.Sequential(nn.Conv2d(3, wl.in_filter, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                        nn.Conv2d(wl.in_filter, wl.out_filter, kernel_size=(wl.hkernel, wl.wkernel),
                                  stride=(wl.hstride, wl.wstride), padding=(wl.hpad, wl.wpad)),
                        nn.AdaptiveAvgPool2d((1,1)))

    input_name = "input0"
    input_shape = [env.BATCH, 3, wl.height, wl.width]
    input_data = torch.randn(input_shape)

    scripted_model = torch.jit.trace(net, input_data).eval()
    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)

    # assert env.BLOCK_IN == env.BLOCK_OUT
    try:
        relay_prog = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_IN,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name="cast",
            start_name_idx=6,
            stop_name="nn.adaptive_avg_pool2d",
        )
    except:
        # if there is an exception in the graph pack, print name of workload and exception message and return None
        return None

    mod = tvm.IRModule.from_expr(relay_prog)
    extracted_tasks = autotvm.task.extract_from_program(
        mod,
        params=params,
        ops=(relay.op.get("nn.conv2d"),),
        target=env.target,
        target_host=env.target_host,
    )

    extracted_tasks = list(filter(lambda t: len(t.args[0][1]) > 4 and "conv" in t.name, extracted_tasks))

    return extracted_tasks[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AXI Performance Monitor Convolution Benchmark')
    parser.add_argument('--model', type=str, default="mobilenet_v2",
                        help='output log file path')

    args = parser.parse_args()

    # Tracker host and port can be set by your environment
    tracker_host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1")
    tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
    target = env.target
    register_vta_tuning_tasks()
    if args.model == "mobilenet_v2":
        wkls_to_tune = mnet_wkls
    elif args.model == "all_wkls":
        wkls_to_tune = pynq_wkls
    else:
        wkls_to_tune = pynq_wkls

    for wkl_name, wl in wkls_to_tune:
        #tasks.append(construct_tasks(env, wl))

        device = "vta"
        log_file = "logs/tuning_logs/vta_1x8x32/%s.%s.%s.log" % (device, args.model, wkl_name)
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
                    # module_loader=vta.module_loader(),
                    # check_correctness=True, # TODO: re-enable when check_correctness works again.
                ),
            ),
        }

        tasks = [construct_tasks(env, wl)]
        # for _, wl in pynq_wkls:
        #     tasks.append(construct_tasks(env, wl))

        if tasks[0] is None:
            print(f"Error in packing graph for workload {wkl_name}, {wl}")

            # add wl to list of broken workloads
            broken_wkls.append(wkl_name)

            continue

        assert len(tasks) == 1
        print("Extracted {} conv2d tasks:".format(len(tasks)))
        for tsk in tasks:
            inp = tsk.args[0][1]
            wgt = tsk.args[1][1]
            batch = inp[0] * inp[4]
            in_filter = inp[1] * inp[5]
            out_filter = wgt[0] * wgt[4]
            height, width = inp[2], inp[3]
            hkernel, wkernel = wgt[2], wgt[3]
            hstride, wstride = tsk.args[2][0], tsk.args[2][1]
            hpad, wpad = tsk.args[3][0], tsk.args[3][1]
            print(
                "({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
                    batch,
                    height,
                    width,
                    in_filter,
                    out_filter,
                    hkernel,
                    wkernel,
                    hpad,
                    wpad,
                    hstride,
                    wstride,
                )
            )

        # We do not run the tuning_logs in our webpage server since it takes too long.
        # Comment the following line to run it by yourself.
        # return

        # run tuning_logs tasks
        print("Tuning...")
        tune_tasks(tasks, **tuning_option)
        print("Tuning Done!!!!!!")

    # print broken workloads
    print(f"Broken workloads: {broken_wkls}")
