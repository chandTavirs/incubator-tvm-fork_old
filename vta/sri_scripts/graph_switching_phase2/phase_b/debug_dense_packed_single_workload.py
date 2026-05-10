from __future__ import absolute_import, print_function

import argparse
import os
import sys
import traceback

import numpy as np

# ---- path setup ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_DIR = os.path.dirname(SCRIPT_DIR)
SRI_SCRIPTS_DIR = os.path.dirname(PHASE2_DIR)
TVM_ROOT = os.path.abspath(os.path.join(SRI_SCRIPTS_DIR, "..", ".."))
TVM_PYTHON = os.path.join(TVM_ROOT, "python")
VTA_PYTHON = os.path.join(TVM_ROOT, "vta", "python")

for p in [TVM_PYTHON, VTA_PYTHON, SRI_SCRIPTS_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import tvm
from tvm import autotvm, rpc, te, topi
from tvm.contrib import utils as tvm_utils
import vta


env = vta.get_env()

# Exact workload requested for isolation.
DATA_SHAPE = (256, 1, 1, 16)
WEIGHT_SHAPE = (1, 1, 16, 16)
OUT_DTYPE = "int32"
TASK_NAME = "dense_packed.vta"


def register_debug_template():
    """Register a dense_packed template that includes the VTA int8 epilogue."""
    from tvm.autotvm.task import TaskExtractEnv

    TaskExtractEnv()

    @autotvm.template(TASK_NAME)
    def _topi_nn_dense_packed(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        data, weight = args[:2]

        with tvm.target.vta():
            res = vta.top.dense_packed(*args, **kwargs)
            # Match the common VTA dense path where output is shifted and cast.
            res = topi.right_shift(res, 8)
            res = topi.cast(res, env.out_dtype)

        current_target = tvm.target.Target.current()
        if current_target is not None and current_target.device_name == "vta":
            sched = vta.top.schedule_dense_packed([res])
        else:
            sched = te.create_schedule([res.op])
        return sched, [data, weight, res]


def create_task():
    data = te.placeholder(DATA_SHAPE, name="data", dtype=env.inp_dtype)
    weight = te.placeholder(WEIGHT_SHAPE, name="weight", dtype=env.wgt_dtype)
    task = autotvm.task.create(
        TASK_NAME,
        args=(data, weight, None, OUT_DTYPE),
        target=env.target,
        target_host=env.target_host,
    )
    return task


def build_schedule_from_config(task, cfg_index):
    cfg = task.config_space.get(cfg_index)
    with tvm.target.vta(), autotvm.task.ApplyConfig(cfg):
        sch, tensors = task.func(*task.args)
        lowered = tvm.lower(sch, tensors, simple_mode=True)
    return cfg, sch, tensors, lowered


def local_phase(task, cfg_index):
    cfg, sch, tensors, lowered = build_schedule_from_config(task, cfg_index)
    print("[LOCAL] Selected config index:", cfg_index)
    print("[LOCAL] Config:", cfg)
    print("[LOCAL] Lower succeeded")
    print(str(lowered)[:800])

    mod = vta.build(sch, tensors, target=env.target, target_host=env.target_host, name="dense_dbg")
    print("[LOCAL] Build succeeded")
    return mod, tensors


def connect_remote(args):
    if args.use_tracker:
        tracker = rpc.connect_tracker(args.tracker_host, args.tracker_port)
        return tracker.request(args.rpc_key)
    return rpc.connect(args.remote_host, args.remote_port)


class DensePackedRuntimeModule:
    """Tiny runtime-module style wrapper around the exported packed func."""

    def __init__(self, module, func_name):
        self.module = module
        self.func = module[func_name]
        self.inputs = {}
        self.output = None

    def set_input(self, name, value):
        self.inputs[name] = value

    def run(self):
        self.output = self.inputs.get("out")
        self.func(self.inputs["data"], self.inputs["weight"], self.output)

    def get_output(self, index):
        if index != 0:
            raise IndexError("DensePackedRuntimeModule only exposes one output")
        return self.output


def remote_phase(args, mod, tensors):
    remote = connect_remote(args)
    print("[REMOTE] Connected")

    if args.reconfig_runtime:
        vta.reconfig_runtime(remote)
        print("[REMOTE] reconfig_runtime() done")

    tmp_dir = tvm_utils.tempdir()
    obj_path = tmp_dir.relpath("dense_dbg.o")
    mod.save(obj_path)
    remote.upload(obj_path)
    rmod = remote.load_module("dense_dbg.o")
    print("[REMOTE] Module uploaded/loaded")

    ctx = remote.ext_dev(0)
    data_np = np.random.randint(-128, 127, size=DATA_SHAPE).astype("int8")
    weight_np = np.random.randint(-128, 127, size=WEIGHT_SHAPE).astype("int8")
    out_shape = (DATA_SHAPE[0], WEIGHT_SHAPE[0], DATA_SHAPE[2], WEIGHT_SHAPE[2])
    # The VTA packed dense epilogue casts to `env.out_dtype`, so the remote
    # output buffer must use the same dtype to satisfy the generated runtime
    # assertions.
    out_np = np.zeros(out_shape, dtype=env.out_dtype)

    data_nd = tvm.nd.array(data_np, ctx)
    weight_nd = tvm.nd.array(weight_np, ctx)
    out_nd = tvm.nd.array(out_np, ctx)

    # Keep the step-3 style of `set_input(...)` + `run()` while still invoking
    # the exported packed function underneath the wrapper.
    m = DensePackedRuntimeModule(rmod, "dense_dbg")
    m.set_input("data", data_nd)
    m.set_input("weight", weight_nd)
    m.set_input("out", out_nd)
    m.run()
    _ = m.get_output(0)
    print("[REMOTE] Kernel run succeeded")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Isolate dense_packed.vta single-workload failures (including fpga_buff_ errors)"
    )
    parser.add_argument("--mode", choices=["local", "remote", "both"], default="both")
    parser.add_argument("--cfg-index", type=int, default=0)

    parser.add_argument("--use-tracker", action="store_true", default=False)
    parser.add_argument("--tracker-host", type=str, default=os.environ.get("TVM_TRACKER_HOST", "127.0.0.1"))
    parser.add_argument("--tracker-port", type=int, default=int(os.environ.get("TVM_TRACKER_PORT", "9190")))
    parser.add_argument("--rpc-key", type=str, default=env.TARGET)

    parser.add_argument("--remote-host", type=str, default="127.0.0.1")
    parser.add_argument("--remote-port", type=int, default=9091)

    parser.add_argument("--reconfig-runtime", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    register_debug_template()
    task = create_task()

    print("Target:", env.target)
    print("Workload:", task.workload)
    print("Config space:", len(task.config_space))

    mod = None
    tensors = None
    try:
        if args.mode in ("local", "both"):
            mod, tensors = local_phase(task, args.cfg_index)

        if args.mode == "remote":
            # Build module even in remote-only mode, to keep one script path.
            mod, tensors = local_phase(task, args.cfg_index)
            assert mod is not None and tensors is not None
            remote_phase(args, mod, tensors)
        elif args.mode == "both":
            assert mod is not None and tensors is not None
            remote_phase(args, mod, tensors)

    except Exception as err:  # pylint: disable=broad-except
        print("[ERROR]", err)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()




