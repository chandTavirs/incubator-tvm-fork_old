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

"""Tune VTA `dense_packed.vta` workloads.

This script is intended for the packed dense shapes produced by graph-pack
transform-matrix rewrites, e.g.::

    dense_packed.vta
    ('TENSOR', (256, 1, 1, 16), 'int8'), ('TENSOR', (1, 1, 16, 16), 'int8')

It tunes each workload independently and stores one AutoTVM log per workload.
"""

from __future__ import absolute_import, print_function

import argparse
import glob
import os
from collections import namedtuple

import tvm
from tvm import autotvm
from tvm import relay
from tvm import te
from tvm import transform
import vta
from ofa_base_models import OFADynamicResnetAllMod
from ofa_derivation_extractor import OFADerivationExtractor
from ofa_relay_graph_builder import build_relay_with_ofa_pool_vars
from ofa_weight_pool_extractor import load_ofa_pool
from quantize_dynamic_weights import quantize_with_dynamic_weights

env = vta.get_env()

PackedDenseWorkload = namedtuple("PackedDenseWorkload", ["name", "data_shape", "weight_shape"])


def _validate_workload(wkl):
    data_shape = wkl.data_shape
    weight_shape = wkl.weight_shape
    assert len(data_shape) == 4 and len(weight_shape) == 4, "dense_packed workloads must be 4D"
    assert data_shape[1] == weight_shape[1], "input-channel block mismatch"
    assert data_shape[3] == weight_shape[3], "inner lane block mismatch"
    assert data_shape[0] > 0 and data_shape[1] > 0 and data_shape[2] > 0 and data_shape[3] > 0
    assert weight_shape[0] > 0 and weight_shape[1] > 0 and weight_shape[2] > 0 and weight_shape[3] > 0
    assert data_shape[3] == env.BLOCK_IN, "expected data inner block to match env.BLOCK_IN"
    assert weight_shape[2] == env.BLOCK_OUT, "expected weight outer block to match env.BLOCK_OUT"
    assert weight_shape[3] == env.BLOCK_IN, "expected weight inner block to match env.BLOCK_IN"


def register_vta_tuning_tasks():
    """Register dense_packed tuning template in AutoTVM task table."""
    from tvm.autotvm.task import TaskExtractEnv

    # Init autotvm env to register VTA operator templates.
    TaskExtractEnv()

    @autotvm.template("dense_packed.vta")
    def _topi_nn_dense_packed(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        data, weight = args[:2]

        with tvm.target.vta():
            res = vta.top.dense_packed(*args, **kwargs)

        if tvm.target.Target.current().device_name == "vta":
            sched = vta.top.schedule_dense_packed([res])
        else:
            sched = te.create_schedule([res.op])
        return sched, [data, weight, res]


def create_tuning_task(wkl):
    _validate_workload(wkl)
    data = te.placeholder(wkl.data_shape, name="data", dtype=env.inp_dtype)
    weight = te.placeholder(wkl.weight_shape, name="weight", dtype=env.wgt_dtype)
    return autotvm.task.create(
        "dense_packed.vta",
        args=(data, weight, None, env.acc_dtype),
        target=env.target,
        target_host=env.target_host,
    )


def _default_relay_ir_path():
    pattern = (
        "/home/srchand/Desktop/research/TVM_Intel_Fork/"
        "tvm/vta/sri_scripts/graph_switching_phase2/phase_b/step3_results/merged_relay_ir_*.txt"
    )
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _parse_relay_ir_text(src):
    """Parse Relay text, adding version header when omitted by caller dumps."""
    src = src.lstrip()
    try:
        return tvm.parser.fromtext(src)
    except tvm.error.TVMError:
        if not src.startswith("#[version"):
            return tvm.parser.fromtext("#[version = \"0.0.5\"]\n" + src)
        raise


def construct_tasks(env, task_name="dense_packed.vta"):
    """Extract dense-packed AutoTVM tasks from a Relay IR text file."""
    ofa_net = OFADynamicResnetAllMod()
    ckpt = torch.load(OFA_CHECKPOINT, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    ofa_net.load_state_dict(state, strict=False)
    ofa_net.eval()
    print("  OFA loaded in %.1fs" % (time.time() - t0))

    mod = _parse_relay_ir_text(src)
    with transform.PassContext(opt_level=3):
        mod = relay.transform.InferType()(mod)

    extracted = autotvm.task.extract_from_program(
        mod["main"],
        params=None,
        ops=(relay.op.get("nn.dense"),),
        target=env.target,
        target_host=env.target_host,
    )

    uniq = {}
    for task in extracted:
        if task.name != task_name:
            continue
        data_shape = tuple(int(x) for x in task.workload[1][1])
        weight_shape = tuple(int(x) for x in task.workload[2][1])
        wkl_name = "dense_packed.%sx%sx%sx%s__%sx%sx%sx%s" % (
            data_shape[0],
            data_shape[1],
            data_shape[2],
            data_shape[3],
            weight_shape[0],
            weight_shape[1],
            weight_shape[2],
            weight_shape[3],
        )
        uniq[task.workload] = (PackedDenseWorkload(wkl_name, data_shape, weight_shape), task)

    return list(uniq.values())


def tune_tasks(tasks, measure_option, tuner="random", n_trial=None, early_stopping=None, log_file="dense_packed.vta.log"):
    """Tune a list of AutoTVM tasks and merge the results into one log."""
    tmp_log_file = log_file + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for idx, (wkl, task) in enumerate(tasks):
        prefix = "[Task %2d/%2d] %s " % (idx + 1, len(tasks), wkl.name)
        print("\n=== %s ===" % wkl.name)
        print("  data_shape   = %s" % (wkl.data_shape,))
        print("  weight_shape = %s" % (wkl.weight_shape,))
        print("  config_space = %d" % len(task.config_space))

        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = autotvm.tuner.XGBTuner(task, loss_type="rank")
        elif tuner == "xgb_knob":
            tuner_obj = autotvm.tuner.XGBTuner(task, loss_type="rank", feature_type="knob")
        elif tuner == "ga":
            tuner_obj = autotvm.tuner.GATuner(task, pop_size=50)
        elif tuner == "gridsearch":
            tuner_obj = autotvm.tuner.GridSearchTuner(task)
        elif tuner == "random":
            tuner_obj = autotvm.tuner.RandomTuner(task)
        else:
            raise ValueError("Invalid tuner: %s" % tuner)

        if os.path.isfile(tmp_log_file):
            tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        task_trials = len(task.config_space) if n_trial is None else min(n_trial, len(task.config_space))
        tuner_obj.tune(
            n_trial=task_trials,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(task_trials, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    autotvm.record.pick_best(tmp_log_file, log_file)
    os.remove(tmp_log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune VTA dense_packed workloads")
    parser.add_argument(
        "--workloads",
        type=str,
        default="all",
        help="Comma-separated extracted workload names to tune, or 'all' (default)",
    )
    parser.add_argument(
        "--tuner",
        type=str,
        default="xgb",
        choices=["random", "xgb", "xgb-rank", "xgb_knob", "ga", "gridsearch"],
        help="AutoTVM tuner to use",
    )
    parser.add_argument(
        "--n-trial",
        type=int,
        default=None,
        help="Trials per workload (default: full config space)",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=None,
        help="Early stopping for the tuner (default: disabled)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/logs/tuning_logs/vta_1x16x16/candidate_set/dense_vta",
        help="Directory to store per-workload tuning logs",
    )
    parser.add_argument(
        "--builder-parallel",
        type=int,
        default=1,
        help="Number of parallel local build workers (default: 1 for stability)",
    )
    parser.add_argument(
        "--builder-timeout",
        type=int,
        default=120,
        help="LocalBuilder timeout in seconds per build (default: 120)",
    )
    parser.add_argument(
        "--runner-number",
        type=int,
        default=5,
        help="Number of runs per config measurement (default: 5)",
    )
    parser.add_argument(
        "--runner-timeout",
        type=int,
        default=60,
        help="RPC runner timeout in seconds per measurement (default: 60)",
    )
    parser.add_argument(
        "--rpc-key",
        type=str,
        default=env.TARGET,
        help="RPC tracker key for RPCRunner (default: env.TARGET)",
    )
    args = parser.parse_args()

    tracker_host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1")
    tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))

    os.makedirs(args.log_dir, exist_ok=True)

    register_vta_tuning_tasks()



    extracted_tasks = construct_tasks(env, relay_ir_path)
    if not extracted_tasks:
        raise RuntimeError("No dense_packed.vta tasks extracted from Relay IR")

    if args.workloads.strip().lower() == "all":
        tasks = extracted_tasks
    else:
        requested = {name.strip() for name in args.workloads.split(",") if name.strip()}
        tasks = [(wkl, task) for (wkl, task) in extracted_tasks if wkl.name in requested]
        missing = sorted(name for name in requested if name not in {w.name for (w, _) in extracted_tasks})
        if missing:
            raise ValueError("Unknown workload name(s): %s" % ", ".join(missing))
    if not tasks:
        raise RuntimeError("No workloads selected")

    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(
            timeout=max(1, args.builder_timeout),
            n_parallel=max(1, args.builder_parallel),
        ),
        runner=autotvm.RPCRunner(
            args.rpc_key,
            host=tracker_host,
            port=tracker_port,
            number=max(1, args.runner_number),
            repeat=1,
            min_repeat_ms=0,
            timeout=max(1, args.runner_timeout),
        ),
    )

    print("Created %d dense_packed tasks" % len(tasks))
    for wkl, task in tasks:
        print("  %s -> %d configs" % (wkl.name, len(task.config_space)))

    tune_tasks(
        tasks,
        measure_option=measure_option,
        tuner=args.tuner,
        n_trial=args.n_trial,
        early_stopping=args.early_stopping,
        log_file=os.path.join(args.log_dir, "dense_packed.vta.log"),
    )

    print("Done. Logs written to %s" % args.log_dir)


