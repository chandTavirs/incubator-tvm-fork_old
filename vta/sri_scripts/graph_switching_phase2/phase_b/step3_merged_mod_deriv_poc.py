"""
Phase B Step 3 (Merged): Single-Module Derivation + Inference POC
=================================================================

This script compiles and runs a *single* Relay module where:
  - weight derivation (slice/reshape/dense transform from pool vars), and
  - convolutional inference
are in one graph.

Unlike the split-mod path, this avoids uploading large derived_w_* tensors.
Instead, pool tensors are uploaded once as runtime inputs and derivation
happens inside the compiled graph.
"""

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
from tvm import autotvm, relay, rpc
from tvm.contrib import graph_runtime, utils as tvm_utils
from tvm.relay.expr_functor import ExprMutator
import vta
from vta.top import graph_pack

from ofa_base_models import OFADynamicResnetAllMod
from ofa_derivation_extractor import OFADerivationExtractor
from ofa_relay_graph_builder import build_relay_with_ofa_pool_vars
from ofa_weight_pool_extractor import load_ofa_pool
from quantize_dynamic_weights import quantize_with_dynamic_weights


# ============================================================
# Config
# ============================================================
OFA_CHECKPOINT = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth"
ARCH_FILE = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json"
POOL_DIR = os.path.join(SCRIPT_DIR, "ofa_weight_pool")
SA_RESULTS_FILE = "/home/srchand/Desktop/research/OFA_Obfs/optimization_experiments/simulated_annealing/results/sa_results_20260216-175221.json"

DEVICE_HOST = "10.42.0.188"
DEVICE_PORT = 9091

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


# ============================================================
# Helpers
# ============================================================
def sep(title="", w=72):
    if title:
        pad = (w - len(title) - 2) // 2
        print("=" * pad + " " + title + " " + "=" * (w - pad - len(title) - 2))
    else:
        print("=" * w)


def load_arch_mapping(arch_file):
    with open(arch_file) as f:
        data = json.load(f)
    if isinstance(data, dict) and "architectures" in data:
        return {item["id"]: item["architecture"] for item in data["architectures"]}
    return data


def _as_float(v):
    try:
        return float(v)
    except Exception:
        return None


def pick_subnets_from_sa(sa_file, arch_file, target_n, target_lambda, target_seed, k):
    with open(sa_file) as f:
        results = json.load(f)
    arch_mapping = load_arch_mapping(arch_file)
    runs = results.get("runs", [])
    nl = [
        r
        for r in runs
        if isinstance(r, dict)
        and r.get("N") == target_n
        and _as_float(r.get("lambda")) is not None
        and abs(_as_float(r.get("lambda")) - float(target_lambda)) < 1e-9
    ]
    if not nl:
        raise ValueError("No run for N=%d, lambda=%s" % (target_n, target_lambda))
    exact = [r for r in nl if r.get("seed") == target_seed]
    run = exact[0] if exact else sorted(nl, key=lambda r: r.get("seed", 0))[0]
    if not exact:
        print("  [warn] seed=%s not found; using seed=%s" % (target_seed, run.get("seed")))
    ids = [x for x in run.get("ids", []) if x in arch_mapping]
    chosen = ids[:k]
    print("  SA run: N=%s, lambda=%s, seed=%s" % (run["N"], run["lambda"], run.get("seed")))
    print("  Subnets: %s" % chosen)
    return {mid: arch_mapping[mid] for mid in chosen}


def load_schedule_logs():
    logs = []
    if os.path.isdir(SCHEDULE_LOG_DIR):
        import glob

        logs = glob.glob(os.path.join(SCHEDULE_LOG_DIR, "*.log"))
    return logs


def summarize_graph_storage(graph_json):
    try:
        g = json.loads(graph_json)
        attrs = g.get("attrs", {})
        shape_attr = attrs.get("shape", [None, []])[1]
        dtype_attr = attrs.get("dltype", [None, []])[1]
        storage_attr = attrs.get("storage_id", [None, []])[1]
        device_attr = attrs.get("device_index", [None, []])[1]
        if not (shape_attr and dtype_attr and storage_attr):
            return None

        dtype_bits = {
            "int8": 8,
            "uint8": 8,
            "int16": 16,
            "uint16": 16,
            "int32": 32,
            "uint32": 32,
            "int64": 64,
            "uint64": 64,
            "float16": 16,
            "float32": 32,
            "float64": 64,
        }

        pool = {}
        for i, sid in enumerate(storage_attr):
            sid = int(sid)
            shape = [int(x) for x in shape_attr[i]]
            dtype = str(dtype_attr[i])
            bits = dtype_bits.get(dtype, 32)
            n_elem = 1
            for d in shape:
                n_elem *= max(1, d)
            nbytes = n_elem * (bits // 8)
            dev_idx = int(device_attr[i]) if i < len(device_attr) else 0
            cur = pool.get(sid)
            if cur is None or nbytes > cur["bytes"]:
                pool[sid] = {"bytes": nbytes, "device_index": dev_idx}

        by_dev = {}
        for _, meta in pool.items():
            by_dev[meta["device_index"]] = by_dev.get(meta["device_index"], 0) + meta["bytes"]
        return {"storage_pool_count": len(pool), "bytes_per_device_index": by_dev}
    except Exception:
        return None


def get_ofa_reference_output(ofa_net, arch, input_np):
    ofa_net.set_active_subnet(arch)
    ofa_net.eval()
    with torch.no_grad():
        out_t = ofa_net(torch.from_numpy(input_np))
    return out_t.cpu().numpy()


def _quantize_np_to_int8(arr, scale=POOL_VAR_QUANT_SCALE):
    x = np.asarray(arr, dtype="float32")
    x = np.round(x * float(scale))
    x = np.clip(x, -127.0, 127.0)
    return x.astype("int8")


def _is_scalar_const(expr):
    if not isinstance(expr, relay.Constant):
        return False
    data = expr.data
    arr = data.asnumpy() if hasattr(data, "asnumpy") else np.asarray(data)
    return arr.size == 1


class _PoolLadderStripper(ExprMutator):
    """Replace pool-only float32->int8 quant ladders with direct int8 pool vars."""

    def __init__(self, int8_pool_var_names):
        super().__init__()
        self.int8_pool_var_names = set(int8_pool_var_names)
        self.wgt_width = int(vta.get_env().WGT_WIDTH)

    def _is_pool_int8_var(self, expr):
        return isinstance(expr, relay.Var) and expr.name_hint in self.int8_pool_var_names

    def _is_int8_expr(self, expr):
        return self._expr_dtype_no_checked_type(expr) == "int8"

    def _is_float32_expr(self, expr):
        return self._expr_dtype_no_checked_type(expr) == "float32"

    def _make_dense_requant_epilogue(self, dense_i32):
        """Match the packed VTA dense epilogue shape used by conv2d paths."""
        shifted = relay.right_shift(dense_i32, relay.const(self.wgt_width, "int32"))
        clipped = relay.clip(shifted, a_min=-127.0, a_max=127.0)
        cast_i8 = relay.cast(clipped, "int8")
        copied = relay.copy(cast_i8)
        stopped = relay.annotation.stop_fusion(copied)
        return relay.cast(stopped, "int32")

    def _strip_quant_ladder(self, call):
        if not (isinstance(call.op, tvm.ir.Op) and call.op.name == "cast"):
            return None
        if call.attrs.dtype != "int8":
            return None
        clip = call.args[0]
        if not (isinstance(clip, relay.Call) and isinstance(clip.op, tvm.ir.Op) and clip.op.name == "clip"):
            return None
        rnd = clip.args[0]
        if not (isinstance(rnd, relay.Call) and isinstance(rnd.op, tvm.ir.Op) and rnd.op.name == "round"):
            return None
        mul = rnd.args[0]
        if not (isinstance(mul, relay.Call) and isinstance(mul.op, tvm.ir.Op) and mul.op.name == "multiply"):
            return None
        lhs, rhs = mul.args
        # After pool vars are rewritten to int8, quant ladders can sit on top of
        # int8-preserving ops (slice/reshape/stop_fusion), not only raw vars.
        if self._is_int8_expr(lhs) and _is_scalar_const(rhs):
            return lhs
        if self._is_int8_expr(rhs) and _is_scalar_const(lhs):
            return rhs
        return None

    def _expr_dtype_no_checked_type(self, expr):
        """Best-effort dtype probe that avoids expr.checked_type during mutation."""
        if isinstance(expr, relay.Constant):
            return str(expr.data.dtype)
        if isinstance(expr, relay.Var) and isinstance(expr.type_annotation, tvm.ir.TensorType):
            return str(expr.type_annotation.dtype)
        if isinstance(expr, relay.Call) and isinstance(expr.op, tvm.ir.Op):
            if expr.op.name == "cast":
                return str(expr.attrs.dtype)
            if expr.op.name == "clip":
                return self._expr_dtype_no_checked_type(expr.args[0])
            passthrough_ops = {
                "strided_slice",
                "reshape",
                "annotation.stop_fusion",
                "copy",
                "transpose",
                "expand_dims",
                "squeeze",
                "right_shift",
            }
            if expr.op.name in passthrough_ops and len(expr.args) >= 1:
                return self._expr_dtype_no_checked_type(expr.args[0])
            # Arithmetic ops propagate dtype: float32 dominates, otherwise all-same wins.
            arith_ops = {"multiply", "add", "subtract", "divide", "sqrt", "negative",
                         "round", "nn.relu", "abs"}
            if expr.op.name in arith_ops:
                arg_dtypes = [self._expr_dtype_no_checked_type(a) for a in expr.args]
                if "float32" in arg_dtypes:
                    return "float32"
                non_none = [d for d in arg_dtypes if d is not None]
                if non_none and len(set(non_none)) == 1:
                    return non_none[0]
        return None

    def visit_call(self, call):
        call = super().visit_call(call)

        # Make mixed int8/float32 multiplies type-safe in rewritten paths.
        if isinstance(call.op, tvm.ir.Op) and call.op.name == "multiply" and len(call.args) == 2:
            lhs, rhs = call.args
            if self._is_int8_expr(lhs) and self._is_float32_expr(rhs):
                return relay.multiply(relay.cast(lhs, "float32"), rhs)
            if self._is_int8_expr(rhs) and self._is_float32_expr(lhs):
                return relay.multiply(lhs, relay.cast(rhs, "float32"))

        stripped = self._strip_quant_ladder(call)
        if stripped is not None:
            return stripped

        # Enforce int32 accumulation for int8 dense paths.
        if isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.dense":
            if len(call.args) != 2:
                return call
            data, weight = call.args
            data_dtype = self._expr_dtype_no_checked_type(data)
            weight_dtype = self._expr_dtype_no_checked_type(weight)
            out_dtype = str(call.attrs.out_dtype) if hasattr(call.attrs, "out_dtype") else ""
            if data_dtype == "int8" and weight_dtype == "int8":
                # Force int32 accumulation, then recreate the VTA-style dense epilogue
                # so downstream packing/fusion sees the same right_shift -> clip -> cast ->
                # copy -> stop_fusion shape used by conv2d paths.
                units = call.attrs.units if hasattr(call.attrs, "units") else None
                dense_i32 = relay.nn.dense(data, weight, units=units, out_dtype="int32")
                return self._make_dense_requant_epilogue(dense_i32)

        return call


def _materialize_int8_pool_constants(mod_q, tvm_params_full, pool_var_names):
    """Rewrite module for int8 pool constants (except selected float32 runtime vars)."""
    main = mod_q["main"]

    int8_pool_vars = sorted([n for n in pool_var_names if n not in FIRST_LAYER_FLOAT_POOL_VARS])
    runtime_float_pool_vars = sorted([n for n in pool_var_names if n in FIRST_LAYER_FLOAT_POOL_VARS])

    param_map = {}
    new_params = []
    for p in main.params:
        name = p.name_hint
        if name in int8_pool_vars:
            p_new = relay.var(name, shape=p.type_annotation.shape, dtype="int8")
            new_params.append(p_new)
            param_map[p] = p_new
        else:
            new_params.append(p)
            param_map[p] = p

    body = relay.expr.bind(main.body, param_map)
    body = _PoolLadderStripper(set(int8_pool_vars)).visit(body)

    new_main = relay.Function(new_params, body)
    mod_i8 = tvm.IRModule.from_expr(new_main)
    mod_i8 = relay.transform.InferType()(mod_i8)

    int8_runtime_pool_params = {}
    runtime_pool_params = {}
    for name in int8_pool_vars:
        arr = tvm_params_full[name]
        if hasattr(arr, "asnumpy"):
            arr = arr.asnumpy()
        int8_runtime_pool_params[name] = _quantize_np_to_int8(arr)

    for name in runtime_float_pool_vars:
        arr = tvm_params_full[name]
        runtime_pool_params[name] = arr.asnumpy() if hasattr(arr, "asnumpy") else np.asarray(arr)

    return mod_i8, int8_runtime_pool_params, runtime_pool_params


# ============================================================
# Build / Compile / Run
# ============================================================
def build_merged_artifacts(subnet_id, arch, ofa_net, base_weights, transform_matrices, bn_params, other_params):
    sep("Build Merged Relay: %s" % subnet_id)

    print("  Extracting derivations...")
    extractor = OFADerivationExtractor(ofa_net, verbose=False)
    derivations = extractor.extract_subnet_derivations(arch, INPUT_SHAPE)
    print("  Derived layers: %d" % len(derivations))

    print("  Building single-module Relay graph...")
    t0 = time.time()
    mod_full, tvm_params_full = build_relay_with_ofa_pool_vars(
        ofa_net=ofa_net,
        arch=arch,
        derivations=derivations,
        base_weights=base_weights,
        transform_matrices=transform_matrices,
        bn_params=bn_params,
        other_params=other_params,
        input_shape=INPUT_SHAPE,
        input_name=INPUT_NAME,
    )
    print("  Build done in %.1fs" % (time.time() - t0))

    pool_var_names = sorted([k for k in tvm_params_full.keys() if k.startswith("pool_")])
    print("  Total params: %d" % len(tvm_params_full))
    print("  Dynamic pool vars: %d" % len(pool_var_names))

    ir_path = os.path.join(RESULTS_DIR, "merged_relay_ir_%s.txt" % subnet_id)
    with open(ir_path, "w") as f:
        f.write(str(mod_full))
    print("  Relay IR -> %s" % ir_path)

    return {
        "mod_full": mod_full,
        "tvm_params_full": tvm_params_full,
        "pool_var_names": pool_var_names,
    }


def step3b_compile_merged(
    subnet_id,
    merged_artifacts,
    env,
    enable_dynamic_dense_quant=False,
    static_debug_mode=False,
    enable_graph_pack=True,
    return_packed_relay_only=False
):
    sep("3B Compile Merged: %s" % subnet_id)

    mod_full = merged_artifacts["mod_full"]
    tvm_params_full = merged_artifacts["tvm_params_full"]
    pool_var_names = merged_artifacts["pool_var_names"]

    print("  Quantizing merged module (single pass)...")
    t0 = time.time()
    with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
        prev_dense_fix_env = os.environ.get("TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX")
        try:
            if enable_dynamic_dense_quant:
                os.environ["TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX"] = "1"
                print("  Dynamic dense quantization: ENABLED")
            else:
                os.environ.pop("TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX", None)

            with relay.quantize.qconfig(
                global_scale=GLOBAL_SCALE,
                skip_conv_layers=SKIP_CONV_LAYERS,
                skip_dense_layer=(not enable_dynamic_dense_quant),
            ):
                mod_q = quantize_with_dynamic_weights(
                    mod_full,
                    tvm_params_full,
                    dynamic_weight_var_names=pool_var_names,
                )
        finally:
            if prev_dense_fix_env is None:
                os.environ.pop("TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX", None)
            else:
                os.environ["TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX"] = prev_dense_fix_env
    print("  Quantization done in %.1fs" % (time.time() - t0))

    q_main_vars = {v.name_hint for v in relay.analysis.free_vars(mod_q["main"].body)}
    kept = sorted(q_main_vars.intersection(set(pool_var_names)))
    print("  Dynamic pool vars kept after quantize: %d" % len(kept))

    # Merged-flow fix: rewrite pool vars so non-first-layer pools become int8
    # runtime inputs, while first-layer base weight stays float32 runtime input.
    if static_debug_mode:
        mod_compile = mod_q
        int8_runtime_pool_params = {}
        runtime_pool_params_np = {
            name: (tvm_params_full[name].asnumpy() if hasattr(tvm_params_full[name], "asnumpy") else np.asarray(tvm_params_full[name]))
            for name in pool_var_names
        }
    else:
        print("  Rewriting merged graph for int8 runtime-pool materialization...")
        mod_compile, int8_runtime_pool_params, runtime_pool_params_np = _materialize_int8_pool_constants(
            mod_q,
            tvm_params_full,
            pool_var_names,
        )
        runtime_pool_params_np.update(int8_runtime_pool_params)
        print("  Int8 runtime pool vars prepared: %d" % len(int8_runtime_pool_params))
        print("  Float32 runtime pool vars kept: %d" % len(runtime_pool_params_np))

    compile_main_vars = {v.name_hint for v in relay.analysis.free_vars(mod_compile["main"].body)}
    if not static_debug_mode:
        missing_pool_inputs = sorted(set(pool_var_names) - compile_main_vars)
        if missing_pool_inputs:
            raise RuntimeError(
                "Expected all pool vars to remain runtime inputs after rewrite; missing: %s"
                % missing_pool_inputs[:8]
            )

    # Build params: non-pool params only. Pool vars are uploaded at runtime.
    if static_debug_mode:
        params_for_build = tvm_params_full
    else:
        params_for_build = {
            k: v for k, v in tvm_params_full.items() if k not in set(pool_var_names)
        }
    print("  Build mode: %s" % ("STATIC DEBUG" if static_debug_mode else "DYNAMIC (all pool vars runtime inputs)"))

    schedule_logs = load_schedule_logs()
    print("  Using %d schedule logs" % len(schedule_logs))

    if enable_graph_pack:
        print("  Applying graph_pack...")
        with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
            relay_prog = graph_pack(
                mod_compile["main"],
                env.BATCH,
                env.BLOCK_IN,
                env.BLOCK_OUT,
                env.WGT_WIDTH,
                start_name=PACK_DICT[MODEL_NAME][0],
                stop_name=PACK_DICT[MODEL_NAME][1],
                device_annot=(env.TARGET == "intelfocl"),
            )
        print("  graph_pack: OK")
    else:
        relay_prog = mod_compile["main"]
        print("  graph_pack: SKIPPED")

    if return_packed_relay_only:
        return relay_prog, params_for_build

    build_target = env.target
    print("  Build target: %s" % build_target)

    build_timeout_sec = int(os.environ.get("STEP3_RELAY_BUILD_TIMEOUT_SEC", "600"))

    def _on_build_timeout(_signum, _frame):
        raise TimeoutError("relay.build exceeded timeout (%ds)" % build_timeout_sec)

    t0 = time.time()
    with autotvm.tophub.context(env.target, extra_files=schedule_logs):
        with vta.build_config(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
            old_handler = None
            if build_timeout_sec > 0:
                print("  relay.build timeout: %ds" % build_timeout_sec)
                old_handler = signal.signal(signal.SIGALRM, _on_build_timeout)
                # Dump python traceback shortly before timeout for diagnostics.
                faulthandler.dump_traceback_later(max(build_timeout_sec - 5, 1), repeat=False)
                signal.alarm(build_timeout_sec)
            try:
                graph, lib, built_params = relay.build(
                    relay_prog,
                    target=build_target,
                    params=params_for_build,
                    target_host=env.target_host,
                )
            finally:
                if build_timeout_sec > 0:
                    signal.alarm(0)
                    faulthandler.cancel_dump_traceback_later()
                    signal.signal(signal.SIGALRM, old_handler)
    print("  relay.build done in %.1fs" % (time.time() - t0))
    print("  Built params: %d" % len(built_params))

    graph_path = os.path.join(RESULTS_DIR, "merged_graph_%s.json" % subnet_id)
    with open(graph_path, "w") as f:
        f.write(graph)
    print("  Graph JSON -> %s" % graph_path)

    storage_summary = summarize_graph_storage(graph)
    if storage_summary is not None:
        by_dev = storage_summary["bytes_per_device_index"]
        desc = ", ".join(["dev%d=%.1fMB" % (d, b / (1024 * 1024.0)) for d, b in sorted(by_dev.items())])
        print("  Storage pools: %d (%s)" % (storage_summary["storage_pool_count"], desc if desc else "n/a"))
        ext_dev_bytes = by_dev.get(0, 0)
        if ext_dev_bytes > (450 * 1024 * 1024):
            print("  [warn] ext_dev storage is high (%.1fMB); runtime init may still fail" % (ext_dev_bytes / (1024 * 1024.0)))

    runtime_pool_var_names = sorted(runtime_pool_params_np.keys())

    return {
        "graph": graph,
        "lib": lib,
        "built_params": built_params,
        "runtime_pool_params_np": runtime_pool_params_np,
        "runtime_pool_var_names": runtime_pool_var_names,
        "static_debug_mode": static_debug_mode,
        "compile_strategy": "single-target(vta)",
    }


def step3c_run_merged(subnet_id, arch, compile_artifacts, ofa_net, input_np, env, remote, ctx):
    sep("3C Run Merged: %s" % subnet_id)

    print("  Uploading library...")
    temp_dir = tvm_utils.tempdir()
    lib_path = temp_dir.relpath("graphlib_step3_merged_%s.tar" % subnet_id)
    compile_artifacts["lib"].export_library(lib_path)
    remote.upload(lib_path)
    remote_lib = remote.load_module("graphlib_step3_merged_%s.tar" % subnet_id)

    print("  Compile strategy: %s" % compile_artifacts.get("compile_strategy", "unknown"))
    m = graph_runtime.create(compile_artifacts["graph"], remote_lib, ctx)
    print("  Runtime context: ext_dev(0)")
    m.set_input(**compile_artifacts["built_params"])

    if not compile_artifacts["static_debug_mode"]:
        copied = 0
        copied_bytes = 0
        for name, arr in sorted(compile_artifacts["runtime_pool_params_np"].items(), key=lambda kv: kv[1].nbytes, reverse=True):
            slot = m.get_input(name)
            if slot is None:
                raise RuntimeError("Missing runtime input for pool var: %s" % name)
            slot.copyfrom(arr)
            copied += 1
            copied_bytes += int(arr.nbytes)
        print("  Uploaded pool vars: %d tensors (%.1f MB)" % (copied, copied_bytes / (1024.0 * 1024.0)))
    else:
        print("  Static debug mode: pool vars are build-bound")

    # input_slot = m.get_input(INPUT_NAME)
    # if input_slot is None:
    #     raise RuntimeError("Missing graph input: %s" % INPUT_NAME)
    # Let graph runtime own the destination context.
    # input_slot.copyfrom(input_np.astype("float32"))
    inp_tvm = tvm.nd.array(
        input_np.astype("float32"),
        remote.ext_dev(0) if env.TARGET != "sim" else tvm.cpu(0)
    )
    m.set_input(INPUT_NAME, inp_tvm)
    print("  Running inference...")
    t0 = time.time()
    m.run()
    inf_time = (time.time() - t0) * 1000.0

    vta_out = m.get_output(0).asnumpy()
    ref_out = get_ofa_reference_output(ofa_net, arch, input_np)

    top1_vta = int(np.argmax(vta_out[0]))
    top1_ref = int(np.argmax(ref_out[0]))

    print("  Top-1 (VTA): %d" % top1_vta)
    print("  Top-1 (REF): %d" % top1_ref)
    print("  Top-1 match: %s" % ("YES" if top1_vta == top1_ref else "NO"))
    print("  Inference: %.1f ms" % inf_time)

    return {
        "top1_vta": top1_vta,
        "top1_ref": top1_ref,
        "top1_match": bool(top1_vta == top1_ref),
        "inf_time_ms": float(inf_time),
    }


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Step3 merged-module derivation+inference POC")
    p.add_argument("--sa-results", default=SA_RESULTS_FILE)
    p.add_argument("--arch-file", default=ARCH_FILE)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--lambda", dest="lambda_value", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-subnets", type=int, default=1)
    p.add_argument("--skip-vta", action="store_true")
    p.add_argument("--skip-cpu", action="store_true", help="Skip optional CPU reference print (still compares against PyTorch in VTA run)")
    p.add_argument("--enable-dynamic-dense-quant", action="store_true")
    p.add_argument(
        "--no-graph-pack",
        action="store_true",
        help="Disable graph_pack in merged flow (memory-safe fallback)",
    )
    p.add_argument("--static-debug-mode", action="store_true")
    p.add_argument("--build-bind-all-params", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main():
    args = parse_args()
    sep("Step3 Merged Mod Deriv POC")

    static_debug_mode = args.static_debug_mode or args.build_bind_all_params
    if args.build_bind_all_params:
        print("  [warn] --build-bind-all-params is deprecated; use --static-debug-mode")
    print("  Runtime mode: %s" % ("STATIC DEBUG" if static_debug_mode else "DYNAMIC (pool vars runtime)"))
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
    poc_archs = pick_subnets_from_sa(
        args.sa_results,
        args.arch_file,
        target_n=args.n,
        target_lambda=args.lambda_value,
        target_seed=args.seed,
        k=args.num_subnets,
    )

    if not args.skip_vta:
        print("\n[4] Setting up VTA RPC...")
        env = vta.get_env()
        remote = rpc.connect(DEVICE_HOST, DEVICE_PORT)
        vta.reconfig_runtime(remote)
        ctx = remote.ext_dev(0)
        print("  Connected to %s:%d" % (DEVICE_HOST, DEVICE_PORT))
        print("  Target: %s" % env.target)
    else:
        print("\n[4] Skipping VTA setup (--skip-vta)")
        env = remote = ctx = None

    rng = np.random.default_rng(99)
    input_np = rng.standard_normal(INPUT_SHAPE).astype("float32")

    all_results = {}
    overall_passed = True

    for subnet_id, arch in poc_archs.items():
        sep("Processing %s" % subnet_id)
        try:
            merged_artifacts = build_merged_artifacts(
                subnet_id,
                arch,
                ofa_net,
                base_weights,
                transform_matrices,
                bn_params,
                other_params,
            )

            if args.skip_vta:
                all_results[subnet_id] = {"build": "OK", "vta": "SKIPPED"}
                continue

            compile_artifacts = step3b_compile_merged(
                subnet_id,
                merged_artifacts,
                env,
                enable_dynamic_dense_quant=args.enable_dynamic_dense_quant,
                static_debug_mode=static_debug_mode,
                enable_graph_pack=(not args.no_graph_pack),
            )

            run_res = step3c_run_merged(
                subnet_id,
                arch,
                compile_artifacts,
                ofa_net,
                input_np,
                env,
                remote,
                ctx,
            )
            all_results[subnet_id] = {"build": "OK", "run": run_res}
            if not run_res["top1_match"]:
                overall_passed = False

        except Exception as e:
            import traceback

            traceback.print_exc()
            all_results[subnet_id] = {"error": str(e)}
            overall_passed = False

    sep("Overall Summary")
    for sid, res in all_results.items():
        print("  %s: %s" % (sid, "OK" if "error" not in res else "FAILED"))

    print("\nOverall: %s" % ("PASS" if overall_passed else "FAIL"))

    summary_path = os.path.join(RESULTS_DIR, "step3_merged_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Summary -> %s" % summary_path)


if __name__ == "__main__":
    main()

