from __future__ import absolute_import, print_function

import argparse
import faulthandler
import json
import os
import signal
import sys
import time

import numpy as np
import torch

# ---- path setup (must come before local imports) ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_DIR = os.path.dirname(SCRIPT_DIR)
SRI_SCRIPTS_DIR = os.path.dirname(PHASE2_DIR)
TVM_ROOT = os.path.abspath(os.path.join(SRI_SCRIPTS_DIR, "..", ".."))
TVM_PYTHON = os.path.join(TVM_ROOT, "python")
VTA_ROOT = os.path.join(TVM_ROOT, "vta", "python")
EXTERNAL_REPO_ROOT = "/home/srchand/Desktop/research/OFA_Obfs"

for p in [EXTERNAL_REPO_ROOT, TVM_PYTHON, VTA_ROOT, SRI_SCRIPTS_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import tvm
from tvm import autotvm, relay, rpc, te, topi
from tvm.relay.expr_functor import ExprVisitor
import vta
from vta.top import graph_pack
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

from ofa_base_models import OFADynamicResnetAllMod
from ofa_derivation_extractor import OFADerivationExtractor
from ofa_relay_graph_builder import build_relay_with_ofa_pool_vars
from ofa_weight_pool_extractor import load_ofa_pool
from quantize_dynamic_weights import quantize_with_dynamic_weights
from step3_merged_mod_deriv_poc import (
    pick_subnets_from_sa, pick_subnets_from_sa_with_exec,
    build_merged_artifacts, sep,
)
from step3_gemm_mat_trf_integration import step5_quantize_module, step6_materialize_int8_pool


# ============================================================
# Config
# ============================================================
OFA_CHECKPOINT = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth"
ARCH_FILE = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json"
POOL_DIR = os.path.join(SCRIPT_DIR, "ofa_weight_pool")
SA_RESULTS_FILE = "/home/srchand/Desktop/research/OFA_Obfs/optimization_experiments/simulated_annealing/results/sa_results_20260216-175221.json"


GLOBAL_SCALE = 8.0
POOL_VAR_QUANT_SCALE = 16.0
SKIP_CONV_LAYERS = [0]
OPT_LEVEL = 3
MODEL_NAME = "resnet18"
INPUT_NAME = "input0"
INPUT_SHAPE = [1, 3, 224, 224]
FIRST_LAYER_FLOAT_POOL_VARS = {
    "pool_first_layer_0_base_conv_weight",
}

PACK_DICT = {
    "resnet18": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
}

RESULTS_DIR = os.path.join(SCRIPT_DIR, "step3_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SCHEDULE_LOG_DIR = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/logs/tuning_logs/vta_1x16x16/candidate_set"

env = vta.get_env()


def _prepare_vta_remote(tracker_host, tracker_port, rpc_key):
    """Ensure VTA runtime on the RPC device is configured before tuning."""
    if env.TARGET in ("sim", "tsim"):
        return
    try:
        tracker = rpc.connect_tracker(tracker_host, tracker_port)
        remote = tracker.request(rpc_key)
        # vta.program_fpga(remote)
        vta.reconfig_runtime(remote)
        return
    except Exception as err:  # pylint: disable=broad-except
        raise RuntimeError(
            "VTA remote pre-config failed. Ensure matching bitstream is available "
            "on target/cache before tuning. Original error: %s" % err
        ) from err


def _to_int_tuple(shape):
    return tuple(int(x) for x in shape)


class _GraphSplitter(relay.ExprMutator):
    """Collect conv2d weight sub-graphs and replace them with free vars (derive/infer split)."""

    def __init__(self):
        super().__init__()
        self._weight_exprs = []
        self._derived_vars = []
        self._wid_to_var = {}
        self._counter = 0

    def visit_call(self, call):
        if isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.conv2d":
            new_data = self.visit(call.args[0])
            orig_w = call.args[1]
            wid = id(orig_w)
            if wid not in self._wid_to_var:
                shape = [int(d) for d in orig_w.checked_type.shape]
                dtype = str(orig_w.checked_type.dtype)
                v = relay.var("derived_weight_%d" % self._counter, shape=shape, dtype=dtype)
                self._wid_to_var[wid] = v
                self._weight_exprs.append(orig_w)
                self._derived_vars.append(v)
                self._counter += 1
            return relay.Call(call.op, [new_data, self._wid_to_var[wid]], call.attrs, call.type_args)
        return super().visit_call(call)

    @property
    def weight_exprs(self):
        return list(self._weight_exprs)

    @property
    def derived_vars(self):
        return list(self._derived_vars)


def _split_mod(mod_compile):
    """Split mod_compile into (derive_fn, infer_fn, n_weights, derived_vars)."""
    main = mod_compile["main"]
    splitter = _GraphSplitter()
    infer_body = splitter.visit(main.body)
    weights = splitter.weight_exprs
    derived_vars = splitter.derived_vars
    n_conv = len(weights)
    if n_conv == 0:
        raise RuntimeError("_split_mod: no nn.conv2d calls found in mod_compile")
    derive_body = relay.Tuple(weights)
    derive_free_names = {v.name_hint for v in relay.analysis.free_vars(derive_body)}
    derive_params = [p for p in main.params if p.name_hint in derive_free_names]
    derive_fn = relay.Function(derive_params, derive_body)
    infer_free_names = {v.name_hint for v in relay.analysis.free_vars(infer_body)}
    orig_infer_params = [p for p in main.params if p.name_hint in infer_free_names]
    infer_fn = relay.Function(orig_infer_params + derived_vars, infer_body)
    return derive_fn, infer_fn, n_conv, derived_vars


class _GmtfDenseShapeCollector(ExprVisitor):
    """Collect vta.gmtf_dense_small/large input/weight shapes from typed Relay."""

    def __init__(self):
        super().__init__()
        self.workloads = []  # list of (data_shape, weight_shape, kind) where kind in ("small","large")

    def visit_call(self, call):
        if isinstance(call.op, tvm.ir.Op) and call.op.name in (
            "vta.gmtf_dense_small",
            "vta.gmtf_dense_large",
        ):
            kind = "small" if call.op.name == "vta.gmtf_dense_small" else "large"
            data_ty = getattr(call.args[0], "checked_type", None)
            weight_ty = getattr(call.args[1], "checked_type", None)
            if (
                isinstance(data_ty, tvm.ir.TensorType)
                and isinstance(weight_ty, tvm.ir.TensorType)
                and len(data_ty.shape) == 4
                and len(weight_ty.shape) == 4
            ):
                data_shape = _to_int_tuple(data_ty.shape)
                weight_shape = _to_int_tuple(weight_ty.shape)
                self.workloads.append((data_shape, weight_shape, kind))
        super().visit_call(call)


def register_vta_tuning_tasks():
    """Register GMTF dense tuning templates in AutoTVM task table."""
    from tvm.autotvm.task import TaskExtractEnv
    TaskExtractEnv()
    # Side-effect: registers vta.gmtf_dense_small/large strategies.
    from vta.top import vta_gmtf_op as _  # noqa: F401
    from vta.top.vta_gmtf_dense import (
        dense_pack_gmtf_small, schedule_dense_pack_gmtf_small,
        dense_pack_gmtf_large, schedule_dense_pack_gmtf_large,
    )

    # @autotvm.register_topi_compute only covers extract_from_program; task.create()
    # needs an @autotvm.template.  Add right_shift(4)+clip+cast(int8) epilogue so
    # VTA's CopyIntrinInjector sees int8 output instead of int32.

    @autotvm.template("dense_pack_gmtf_small.vta")
    def _tune_gmtf_small(*args, **kwargs):
        assert not kwargs
        data, weight = args[0], args[1]
        with tvm.target.vta():
            res = dense_pack_gmtf_small(data, weight, None, env.acc_dtype)
            res = topi.right_shift(res, 4)
            res = topi.clip(res, -127, 127)
            res = topi.cast(res, env.out_dtype)
        current_target = tvm.target.Target.current()
        if current_target is not None and current_target.device_name == "vta":
            sched = schedule_dense_pack_gmtf_small([res])
        else:
            sched = te.create_schedule([res.op])
        return sched, [data, weight, res]

    @autotvm.template("dense_pack_gmtf_large.vta")
    def _tune_gmtf_large(*args, **kwargs):
        assert not kwargs
        data, weight = args[0], args[1]
        with tvm.target.vta():
            res = dense_pack_gmtf_large(data, weight, None, env.acc_dtype)
            res = topi.right_shift(res, 4)
            res = topi.clip(res, -127, 127)
            res = topi.cast(res, env.out_dtype)
        current_target = tvm.target.Target.current()
        if current_target is not None and current_target.device_name == "vta":
            sched = schedule_dense_pack_gmtf_large([res])
        else:
            sched = te.create_schedule([res.op])
        return sched, [data, weight, res]


def construct_tasks(args):
    print("\n[1] Loading OFA model...")
    t0 = time.time()
    ofa_net = OFADynamicResnetAllMod()
    ckpt = torch.load(OFA_CHECKPOINT, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    ofa_net.load_state_dict(state, strict=False)
    ofa_net.eval()
    print("  OFA loaded in %.1fs" % (time.time() - t0))

    print("\n[2] Loading OFA pool...")
    t0 = time.time()
    pool = load_ofa_pool(POOL_DIR)
    base_weights = pool["base_weights"]
    transform_matrices = pool["transform_matrices"]
    bn_params = pool["bn_params"]
    other_params = pool["other_params"]
    print(
        "  Pool loaded in %.1fs (%d base, %d tm, %d bn, %d other)"
        % (time.time() - t0, len(base_weights), len(transform_matrices), len(bn_params), len(other_params))
    )

    print("\n[3] Selecting subnets...")
    if args.gamma_value is None:
        poc_archs = pick_subnets_from_sa(
            args.sa_results,
            args.arch_file,
            target_n=args.n,
            target_lambda=args.lambda_value,
            target_seed=args.seed,
            k=args.num_subnets,
        )
    else:
        poc_archs = pick_subnets_from_sa_with_exec(
            args.sa_results,
            args.arch_file,
            target_n=args.n,
            target_lambda=args.lambda_value,
            target_gamma=args.gamma_value,
            target_seed=args.seed,
            k=args.num_subnets,
        )

    all_tasks = []
    for subnet_id, arch in poc_archs.items():
        sep("Processing %s" % subnet_id)
        try:
            artifacts = build_merged_artifacts(
                subnet_id, arch, ofa_net, base_weights, transform_matrices, bn_params, other_params,
            )
            mod_full = artifacts["mod_full"]
            tvm_params_full = artifacts["tvm_params_full"]

            all_dynamic_var_names = list(tvm_params_full.keys())
            pool_only_var_names = [k for k in tvm_params_full.keys() if k.startswith("pool_")]

            mod_q, _ = step5_quantize_module(
                mod_full, tvm_params_full, all_dynamic_var_names, enable_dynamic_dense_quant=True,
            )
            mod_compile, _ = step6_materialize_int8_pool(
                mod_q, tvm_params_full, pool_only_var_names,
            )

            mod_compile_typed = relay.transform.InferType()(mod_compile)
            pack_entry = PACK_DICT.get(MODEL_NAME, ["nn.max_pool2d", "nn.adaptive_avg_pool2d"])
            with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
                full_packed_fn = graph_pack(
                    mod_compile_typed["main"], env.BATCH, env.BLOCK_IN, env.BLOCK_OUT, env.WGT_WIDTH,
                    start_name=pack_entry[0], stop_name=pack_entry[1],
                    device_annot=(env.TARGET == "intelfocl"),
                )
            mod_packed = relay.transform.InferType()(tvm.IRModule.from_expr(full_packed_fn))

            derive_fn, _infer_fn, n_w, _derived_vars = _split_mod(mod_packed)
            derive_mod = relay.transform.InferType()(tvm.IRModule.from_expr(derive_fn))

            collector = _GmtfDenseShapeCollector()
            collector.visit(derive_mod["main"])

            print("  %s: %d GMTF dense workloads found" % (subnet_id, len(collector.workloads)))
            for data_shape, weight_shape, kind in collector.workloads:
                data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
                weight = te.placeholder(weight_shape, name="weight", dtype=env.wgt_dtype)
                task_name = "dense_pack_gmtf_small.vta" if kind == "small" else "dense_pack_gmtf_large.vta"
                task = autotvm.task.create(
                    task_name,
                    args=(data, weight, None, env.acc_dtype),
                    target=env.target,
                    target_host=env.target_host,
                )
                all_tasks.append(task)
        except Exception:
            import traceback
            traceback.print_exc()

    uniq = {}
    for task in all_tasks:
        uniq[task.workload] = task
    return list(uniq.values())

def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning_logs.log",
    use_transfer_learning=True,
):
    task_tmp_logs = []


    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
        task_tmp_log = "%s.task_%02d.tmp" % (log_filename, i + 1)
        task_tmp_logs.append(task_tmp_log)
        if os.path.exists(task_tmp_log):
            os.remove(task_tmp_log)

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
            if os.path.isfile(task_tmp_log):
                try:
                    history = list(autotvm.record.load_from_file(task_tmp_log))
                    if history:
                        tuner_obj.load_history(history)
                except Exception as err:
                    print("Skip history load due to invalid records: %s" % err)

        # do tuning_logs
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(task_tmp_log),
            ],
        )

    merged_tmp_log = log_filename + ".tmp"
    if os.path.exists(merged_tmp_log):
        os.remove(merged_tmp_log)

    # Merge per-task logs before selecting global best records.
    with open(merged_tmp_log, "w") as fout:
        for task_log in task_tmp_logs:
            if not os.path.exists(task_log):
                continue
            with open(task_log, "r") as fin:
                fout.write(fin.read())

    # pick best records to a cache file
    autotvm.record.pick_best(merged_tmp_log, log_filename)

    if os.path.exists(merged_tmp_log):
        os.remove(merged_tmp_log)
    for task_log in task_tmp_logs:
        if os.path.exists(task_log):
            os.remove(task_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune VTA dense_packed workloads")
    parser.add_argument("--num-subnets", type=int, default=1)
    parser.add_argument("--sa-results", default=SA_RESULTS_FILE)
    parser.add_argument("--arch-file", default=ARCH_FILE)
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--lambda", dest="lambda_value", type=float, default=4.0)
    parser.add_argument("--gamma", dest="gamma_value", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--tuner",
        type=str,
        default="xgb_knob",
        choices=["random", "xgb", "xgb-rank", "xgb_knob", "ga", "gridsearch"],
        help="AutoTVM tuner to use",
    )
    parser.add_argument(
        "--no-transfer-learning",
        action="store_true",
        default=False,
        help="Disable loading tuning history into the tuner",
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
        default="/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/logs/tuning_logs/vta_1x16x16/candidate_set/dense_vta_n20_exec",
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

    tasks = construct_tasks(args)
    if not tasks:
        print("No GMTF dense tasks extracted.")
        sys.exit(0)

    print("Extracted %d GMTF dense tasks" % len(tasks))
    for idx, task in enumerate(tasks):
        print("  [%d] %s" % (idx, task.workload))

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

    _prepare_vta_remote(tracker_host, tracker_port, args.rpc_key)

    tune_tasks(
        tasks,
        measure_option,
        tuner=args.tuner,
        n_trial=(args.n_trial if args.n_trial is not None else 1000),
        early_stopping=args.early_stopping,
        log_filename=os.path.join(args.log_dir, "gmtf_dense.vta.log"),
        use_transfer_learning=(not args.no_transfer_learning),
    )






