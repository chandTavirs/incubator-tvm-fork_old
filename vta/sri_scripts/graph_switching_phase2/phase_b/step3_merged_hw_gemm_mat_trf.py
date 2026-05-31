"""
Step 3: OFA Subnet Inference with Hardware GEMM_Mat_Trf
========================================================

Mirror of step3_merged_mod_deriv_poc.py, but OFA dense matrix transform
operations (kernel derivation: out = inp_blocks @ T.T) are executed on the
VTA hardware using the new GEMM_Mat_Trf instruction (empty_0 = 0x1/0x3)
instead of CPU numpy.

Flow
----
  1.  Load OFA model
  2.  Load OFA weight pool
  3.  Pick subnets
  4.  Build merged Relay graph (derivation + inference in one module)
  [HW] Step 4: Materialize transforms via hardware GEMM_Mat_Trf
  [HW] Step 4b: Substitute hardware outputs as relay.const
  5.  Quantize merged module
  6.  Rewrite pool vars to int8 runtime inputs
  7.  graph_pack + relay.build
  8.  RPC upload + inference on device

Hardware materialization
------------------------
  For each transform dense op (base_weight_slice @ T.T):
    - Quantize inputs (int8, scale = HW_QUANT_SCALE)
    - Pack into VTA layout (small: (n,1,1,16), large: (n,2,1,16))
    - Execute on ZCU104 via VTAPushGEMMMatTrfOpSmall/Large
    - Dequantize int8 output back to float32 for compatibility with Relay quantize

  The >>8 requantization applied by the hardware encodes a scale factor of
  256/HW_QUANT_SCALE^2.  With HW_QUANT_SCALE=128, dequant = 256/16384 = 0.015625.
  The ALU uses symmetric clip [-127,127] (no ReLU) since transform outputs can be negative.

  Note: this script demonstrates hardware-path functional correctness.
  Precision differs from the CPU float32 path due to int8 quantisation; top-1
  match against the PyTorch reference may not hold for all subnets.

Usage
-----
  # Run with hardware transforms (default):
  VTA_RPC_HOST=10.42.0.188 VTA_RPC_PORT=9091 python step3_merged_hw_gemm_mat_trf.py

  # Compare CPU vs HW materialization:
  python step3_merged_hw_gemm_mat_trf.py --cpu-mode   # fall back to CPU numpy
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

# ---- path setup (must precede local imports) ----
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
from tvm import autotvm, relay, rpc, te
from tvm.contrib import graph_runtime, utils as tvm_utils
from tvm.relay.expr_functor import ExprVisitor, ExprMutator
import vta
from vta.top import graph_pack
from vta import intrin as vta_intrin

from ofa_base_models import OFADynamicResnetAllMod
from ofa_derivation_extractor import OFADerivationExtractor
from ofa_relay_graph_builder import build_relay_with_ofa_pool_vars
from ofa_weight_pool_extractor import load_ofa_pool
from quantize_dynamic_weights import quantize_with_dynamic_weights

# Reuse helpers from the integration script
from step3_gemm_mat_trf_integration import (
    step4b_lower_gemm_mat_trf,
    _normalize_trf_names,
    _eval_expr,
    _materialize_transforms_from_graph,
)


# ============================================================
# Config
# ============================================================
OFA_CHECKPOINT = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth"
ARCH_FILE = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json"
POOL_DIR = os.path.join(SCRIPT_DIR, "ofa_weight_pool")
SA_RESULTS_FILE = "/home/srchand/Desktop/research/OFA_Obfs/optimization_experiments/simulated_annealing/results/sa_results_20260216-175221.json"

DEVICE_HOST = "10.42.0.188"
DEVICE_PORT  = 9091

GLOBAL_SCALE = 8.0
POOL_VAR_QUANT_SCALE = 16.0

# Scale used to quantize pool vars for hardware GEMM_Mat_Trf execution.
# Dequant factor = 256/HW_QUANT_SCALE^2 = 256/16384 ≈ 0.015625.
# Scale=128 covers typical OFA weight range [-1, 1] with ~1% resolution;
# avoids the all-zero bug that occurs with scale=16 (products too small to survive >>8).
HW_QUANT_SCALE = 128.0

# VTA inp_mem holds 2048 entries; large mode uses 2 per batch element.
MAX_BATCH_HW_SMALL = 2048
MAX_BATCH_HW_LARGE = 1024

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

RESULTS_DIR = os.path.join(SCRIPT_DIR, "step3_gmm_mat_trf_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SCHEDULE_LOG_DIR = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/logs/tuning_logs/vta_1x16x16/candidate_set"

# Transform variable names use both 'xtoY' and 'XtoY' conventions; dim 9 = 3x3, dim 25 = 5x5.
_DIM_SMALL = 9
_DIM_LARGE = 25


# ============================================================
# Shared helpers (mirror of step3_merged_mod_deriv_poc.py)
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
        r for r in runs
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
    arr = expr.data.asnumpy() if hasattr(expr.data, "asnumpy") else np.asarray(expr.data)
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
        return self._expr_dtype(expr) == "int8"

    def _is_float32_expr(self, expr):
        return self._expr_dtype(expr) == "float32"

    def _make_dense_requant_epilogue(self, dense_i32):
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
        if self._is_int8_expr(lhs) and _is_scalar_const(rhs):
            return lhs
        if self._is_int8_expr(rhs) and _is_scalar_const(lhs):
            return rhs
        return None

    def _expr_dtype(self, expr):
        if isinstance(expr, relay.Constant):
            return str(expr.data.dtype)
        if isinstance(expr, relay.Var) and isinstance(expr.type_annotation, tvm.ir.TensorType):
            return str(expr.type_annotation.dtype)
        if isinstance(expr, relay.Call) and isinstance(expr.op, tvm.ir.Op):
            if expr.op.name == "cast":
                return str(expr.attrs.dtype)
            if expr.op.name == "clip":
                return self._expr_dtype(expr.args[0])
            passthrough = {"strided_slice", "reshape", "annotation.stop_fusion", "copy",
                           "transpose", "expand_dims", "squeeze", "right_shift"}
            if expr.op.name in passthrough and expr.args:
                return self._expr_dtype(expr.args[0])
        return None

    def visit_call(self, call):
        call = super().visit_call(call)
        if isinstance(call.op, tvm.ir.Op) and call.op.name == "multiply" and len(call.args) == 2:
            lhs, rhs = call.args
            if self._is_int8_expr(lhs) and self._is_float32_expr(rhs):
                return relay.multiply(relay.cast(lhs, "float32"), rhs)
            if self._is_int8_expr(rhs) and self._is_float32_expr(lhs):
                return relay.multiply(lhs, relay.cast(rhs, "float32"))
        stripped = self._strip_quant_ladder(call)
        if stripped is not None:
            return stripped
        if isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.dense" and len(call.args) == 2:
            data, weight = call.args
            if self._expr_dtype(data) == "int8" and self._expr_dtype(weight) == "int8":
                units = call.attrs.units if hasattr(call.attrs, "units") else None
                dense_i32 = relay.nn.dense(data, weight, units=units, out_dtype="int32")
                return self._make_dense_requant_epilogue(dense_i32)
        return call


def _materialize_int8_pool_constants(mod_q, tvm_params_full, pool_var_names):
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
# Hardware GEMM_Mat_Trf execution
# ============================================================

def _build_hw_schedule(env, n_batch, large_mode):
    """Build TVM schedule for GEMM_Mat_Trf (small or large mode)."""
    BATCH    = env.BATCH
    BLOCK_IN = env.BLOCK_IN
    BLOCK_OUT = env.BLOCK_OUT

    if not large_mode:
        # Small mode (DIM=9): same as build_gemm_mat_trf_schedule in benchmark
        data_shape = (n_batch, 1, BATCH, BLOCK_IN)
        trf_shape  = (1, 1, BLOCK_OUT, BLOCK_IN)
        out_shape  = (n_batch, 1, BATCH, BLOCK_OUT)

        data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
        trf  = te.placeholder(trf_shape,  name="trf",  dtype=env.wgt_dtype)
        data_buf = te.compute(data_shape, lambda *i: data(*i), name="data_buf")
        trf_buf  = te.compute(trf_shape,  lambda *i: trf(*i),  name="trf_buf")

        ki = te.reduce_axis((0, BLOCK_IN), name="ki")
        res_gemm = te.compute(
            out_shape,
            lambda bo, co, bi, ci: te.sum(
                data_buf[bo, 0, bi, ki].astype(env.acc_dtype)
                * trf_buf[0, 0, ci, ki].astype(env.acc_dtype),
                axis=[ki],
            ),
            name="res_gemm",
        )
        res_shf = te.compute(out_shape, lambda *i: res_gemm(*i) >> 8, name="res_shf")
        # Symmetric clip [-127, 127]: no ReLU — transform outputs can be negative.
        res_min = te.compute(
            out_shape,
            lambda *i: tvm.te.min(res_shf(*i), (1 << (env.INP_WIDTH - 1)) - 1),
            name="res_min",
        )
        res_max = te.compute(
            out_shape,
            lambda *i: tvm.te.max(res_min(*i), -((1 << (env.INP_WIDTH - 1)) - 1)),
            name="res_max",
        )
        res = te.compute(out_shape, lambda *i: res_max(*i).astype(env.inp_dtype), name="res")

        s = te.create_schedule(res.op)
        s[data_buf].set_scope(env.inp_scope)
        s[trf_buf].set_scope(env.wgt_scope)
        s[res_gemm].set_scope(env.acc_scope)
        s[res_shf].set_scope(env.acc_scope)
        s[res_min].set_scope(env.acc_scope)
        s[res_max].set_scope(env.acc_scope)

        s[trf_buf].pragma(s[trf_buf].op.axis[0], env.dma_copy)
        s[data_buf].pragma(s[data_buf].op.axis[0], env.dma_copy)

        xbo, xco, xbi, xci = s[res_gemm].op.axis
        s[res_gemm].tensorize(xbi, vta_intrin.gemm_mat_trf(env, mock=False, large_mode=False))

        s[res_shf].pragma(s[res_shf].op.axis[0], env.alu)
        s[res_min].pragma(s[res_min].op.axis[0], env.alu)
        s[res_max].pragma(s[res_max].op.axis[0], env.alu)
        s[res].pragma(s[res].op.axis[0], env.dma_copy)

        return s, [data, trf, res], (data_shape, trf_shape, out_shape)
    else:
        # Large mode (DIM=25): same as build_gemm_mat_trf_large_schedule in benchmark
        data_shape = (n_batch, 2, BATCH, BLOCK_IN)
        trf_shape  = (2 * BLOCK_OUT, 1, BLOCK_OUT, BLOCK_IN)
        out_shape  = (n_batch, 2, BATCH, BLOCK_OUT)

        data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
        trf  = te.placeholder(trf_shape,  name="trf",  dtype=env.wgt_dtype)
        data_buf = te.compute(data_shape, lambda *i: data(*i), name="data_buf")
        trf_buf  = te.compute(trf_shape,  lambda *i: trf(*i),  name="trf_buf")

        ko = te.reduce_axis((0, BLOCK_OUT), name="ko")
        ki = te.reduce_axis((0, BLOCK_IN),  name="ki")
        res_gemm = te.compute(
            out_shape,
            lambda bo, co, bi, ci: te.sum(
                data_buf[bo, tvm.tir.min(ko, 1), bi, ki].astype(env.acc_dtype)
                * trf_buf[co * BLOCK_OUT + ci, 0, ko, ki].astype(env.acc_dtype),
                axis=[ko, ki],
            ),
            name="res_gemm",
        )
        res_shf = te.compute(out_shape, lambda *i: res_gemm(*i) >> 8, name="res_shf")
        # Symmetric clip [-127, 127]: no ReLU — transform outputs can be negative.
        res_min = te.compute(
            out_shape,
            lambda *i: tvm.te.min(res_shf(*i), (1 << (env.INP_WIDTH - 1)) - 1),
            name="res_min",
        )
        res_max = te.compute(
            out_shape,
            lambda *i: tvm.te.max(res_min(*i), -((1 << (env.INP_WIDTH - 1)) - 1)),
            name="res_max",
        )
        res = te.compute(out_shape, lambda *i: res_max(*i).astype(env.inp_dtype), name="res")

        s = te.create_schedule(res.op)
        s[data_buf].set_scope(env.inp_scope)
        s[trf_buf].set_scope(env.wgt_scope)
        s[res_gemm].set_scope(env.acc_scope)
        s[res_shf].set_scope(env.acc_scope)
        s[res_min].set_scope(env.acc_scope)
        s[res_max].set_scope(env.acc_scope)

        s[trf_buf].pragma(s[trf_buf].op.axis[0], env.dma_copy)
        s[data_buf].pragma(s[data_buf].op.axis[0], env.dma_copy)

        xbo, xco, xbi, xci = s[res_gemm].op.axis
        s[res_gemm].tensorize(xco, vta_intrin.gemm_mat_trf(env, mock=False, large_mode=True))

        s[res_shf].pragma(s[res_shf].op.axis[0], env.alu)
        s[res_min].pragma(s[res_min].op.axis[0], env.alu)
        s[res_max].pragma(s[res_max].op.axis[0], env.alu)
        s[res].pragma(s[res].op.axis[0], env.dma_copy)

        return s, [data, trf, res], (data_shape, trf_shape, out_shape)


def _hw_gemm_mat_trf_chunk(inp_int8, trf_int8_packed, env, remote, large_mode, _mod_cache):
    """
    Execute one GEMM_Mat_Trf chunk on hardware.

    inp_int8           : np.int8  (n_batch, DIM)
    trf_int8_packed    : np.int8  packed for hardware
    Returns int8 output (n_batch, DIM)
    """
    DIM      = _DIM_LARGE if large_mode else _DIM_SMALL
    BATCH    = env.BATCH
    BLOCK_IN = env.BLOCK_IN
    BLOCK_OUT = env.BLOCK_OUT
    n_batch  = inp_int8.shape[0]

    cache_key = ("large" if large_mode else "small", n_batch)
    if cache_key not in _mod_cache:
        s, tensors, shapes = _build_hw_schedule(env, n_batch, large_mode)
        with vta.build_config():
            mod = vta.build(s, tensors, "ext_dev", env.target_host,
                            name="hw_gemm_mat_trf_%s_%d" % ("L" if large_mode else "S", n_batch))
        tmp = tvm_utils.tempdir()
        fname = "hw_gmtf_%s_%d.o" % ("L" if large_mode else "S", n_batch)
        lib_path = tmp.relpath(fname)
        mod.save(lib_path)
        remote.upload(lib_path)
        _mod_cache[cache_key] = (remote.load_module(fname), shapes)

    f, (data_shape, trf_shape, out_shape) = _mod_cache[cache_key]
    ctx = remote.ext_dev(0)

    # Pack input
    data_np = np.zeros(data_shape, dtype=env.inp_dtype)
    if not large_mode:
        data_np[:, 0, 0, :DIM] = inp_int8
    else:
        data_np[:, 0, 0, :]  = inp_int8[:, :BLOCK_IN]
        data_np[:, 1, 0, :DIM - BLOCK_IN] = inp_int8[:, BLOCK_IN:DIM]

    res_np = np.zeros(out_shape, dtype=env.inp_dtype)

    data_arr = tvm.nd.array(data_np, ctx)
    trf_arr  = tvm.nd.array(trf_int8_packed, ctx)
    res_arr  = tvm.nd.array(res_np, ctx)
    f(data_arr, trf_arr, res_arr)

    # Unpack output: (n_batch, 2 or 1, BATCH, BLOCK_OUT) → (n_batch, DIM)
    result_np = res_arr.asnumpy()
    if not large_mode:
        return result_np[:, 0, 0, :DIM]
    else:
        out = np.zeros((n_batch, DIM), dtype=np.int8)
        out[:, :BLOCK_OUT]       = result_np[:, 0, 0, :BLOCK_OUT].astype(np.int8)
        out[:, BLOCK_OUT:DIM]    = result_np[:, 1, 0, :DIM - BLOCK_OUT].astype(np.int8)
        return out


def _pack_trf_matrix(trf_int8, env, large_mode):
    """Pack a DIM×DIM int8 T matrix into the VTA wgt buffer format."""
    BLOCK_IN  = env.BLOCK_IN
    BLOCK_OUT = env.BLOCK_OUT
    DIM = _DIM_LARGE if large_mode else _DIM_SMALL

    if not large_mode:
        # Shape (1, 1, BLOCK_OUT, BLOCK_IN): T[i][j] = packed[0, 0, i, j]
        packed = np.zeros((1, 1, BLOCK_OUT, BLOCK_IN), dtype=np.int8)
        packed[0, 0, :DIM, :DIM] = trf_int8[:DIM, :DIM]
        return packed
    else:
        # Shape (2*BLOCK_OUT, 1, BLOCK_OUT, BLOCK_IN):
        # T[i][j] = packed[i, 0, j//BLOCK_IN, j%BLOCK_IN]  for j < DIM
        packed = np.zeros((2 * BLOCK_OUT, 1, BLOCK_OUT, BLOCK_IN), dtype=np.int8)
        for i in range(DIM):
            packed[i, 0, 0, :]  = trf_int8[i, :BLOCK_IN]
            packed[i, 0, 1, :DIM - BLOCK_IN] = trf_int8[i, BLOCK_IN:DIM]
        return packed


def hw_gemm_mat_trf(inp_blocks_f32, trf_matrix_f32, env, remote, large_mode,
                     quant_scale=HW_QUANT_SCALE, _mod_cache=None):
    """
    Execute out = inp_blocks @ T.T on VTA hardware using GEMM_Mat_Trf.

    Quantization
    ------------
      inp_int8 = clip(round(inp_float32 * quant_scale), -127, 127)
      T_int8   = clip(round(T_float32   * quant_scale), -127, 127)
      Hardware: acc = inp_int8 @ T_int8.T  (int32)
                out_int8 = clip(acc >> 8, 0, 127)
      Dequantize: out_float = out_int8 * (256 / quant_scale^2)

    With quant_scale=16 (default), dequant factor = 256/256 = 1.0.

    Parameters
    ----------
    inp_blocks_f32 : np.ndarray  (n_batch, DIM)
    trf_matrix_f32 : np.ndarray  (DIM, DIM)
    env            : VTA Environment
    remote         : tvm.rpc.RPCSession
    large_mode     : bool  (True = 25x25, False = 9x9)
    quant_scale    : float
    _mod_cache     : dict  (caller-provided; avoids re-building TVM modules)

    Returns
    -------
    np.ndarray  float32  (n_batch, DIM)
    """
    if _mod_cache is None:
        _mod_cache = {}

    DIM = _DIM_LARGE if large_mode else _DIM_SMALL
    n_batch = inp_blocks_f32.shape[0]
    assert trf_matrix_f32.shape == (DIM, DIM), \
        "T matrix shape mismatch: expected (%d,%d), got %s" % (DIM, DIM, trf_matrix_f32.shape)

    # Quantize T matrix once (shared across all batch chunks)
    trf_int8   = np.clip(np.round(trf_matrix_f32.astype(np.float32) * quant_scale),
                         -127, 127).astype(np.int8)
    trf_packed = _pack_trf_matrix(trf_int8, env, large_mode)

    # Quantize input blocks
    inp_int8 = np.clip(np.round(inp_blocks_f32.astype(np.float32) * quant_scale),
                       -127, 127).astype(np.int8)

    max_batch = MAX_BATCH_HW_LARGE if large_mode else MAX_BATCH_HW_SMALL

    # Process in chunks to respect inp_mem capacity
    out_int8 = np.zeros((n_batch, DIM), dtype=np.int8)
    for start in range(0, n_batch, max_batch):
        end   = min(start + max_batch, n_batch)
        chunk = _hw_gemm_mat_trf_chunk(
            inp_int8[start:end], trf_packed, env, remote, large_mode, _mod_cache
        )
        out_int8[start:end] = chunk

    # Dequantize: out_float = out_int8 * (256 / scale^2)
    dequant_factor = 256.0 / (quant_scale * quant_scale)
    return out_int8.astype(np.float32) * dequant_factor


# ============================================================
# Step 4: Hardware materialization
# ============================================================
def step4_hw_materialize_transforms(mod, pool, transform_var_names, env, remote, verbose=True):
    """
    Walk the Relay graph to find transform dense ops, execute each one
    on hardware using GEMM_Mat_Trf, and return a dict of float32 results.

    This mirrors step4_materialize_transforms in step3_gemm_mat_trf_integration.py
    but replaces the CPU numpy computation with hardware execution.
    """
    sep("Step 4: Hardware GEMM_Mat_Trf Materialization")

    if not transform_var_names:
        print("  No transform matrices; skipping")
        return {}, set()

    print("  Transform vars: %d" % len(transform_var_names))
    print("  Hardware: GEMM_Mat_Trf  quant_scale=%.1f  device=%s:%d"
          % (HW_QUANT_SCALE, DEVICE_HOST, DEVICE_PORT))

    # Build pool var dict for the graph walker
    pool_var_dict = {}
    for key in ["base_weights", "transform_matrices", "bn_params", "other_params"]:
        for name, arr in pool.get(key, {}).items():
            pool_var_dict[name] = arr
            pool_var_dict["pool_" + name] = arr

    t0 = time.time()

    # Walk the graph to get subnet-specific input shapes and T matrix values
    # (reuses _materialize_transforms_from_graph but intercepts the final multiply)
    mod_typed = relay.transform.InferType()(mod)
    normalized = _normalize_trf_names(transform_var_names)

    materialized = {}
    _mod_cache = {}  # shared TVM module cache across transforms

    class _HWMaterializingVisitor(ExprVisitor):
        def __init__(self):
            super().__init__()
            self._dense_cache = {}

        def visit_call(self, call):
            self.visit(call.op)
            for a in call.args:
                self.visit(a)

            if not (isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.dense"):
                return

            # Check second argument is a transform variable
            if len(call.args) < 2 or not isinstance(call.args[1], relay.Var):
                return
            trf_var = call.args[1]
            trf_name = trf_var.name_hint
            if trf_name not in normalized and trf_name.replace("pool_", "") not in normalized:
                return

            # Get T matrix from pool
            trf_data = pool_var_dict.get(trf_name)
            if trf_data is None:
                trf_data = pool_var_dict.get(trf_name.replace("pool_", ""))
            if trf_data is None:
                if verbose:
                    print("  [warn] T matrix not found in pool: %s" % trf_name)
                return

            # Evaluate the input (sliced base weight) using the same numpy walk
            inp_data = _eval_expr(call.args[0], pool_var_dict, self._dense_cache)
            if inp_data is None:
                if verbose:
                    print("  [warn] Could not evaluate input for: %s" % trf_name)
                return

            # Ensure inputs are float32 numpy arrays
            inp_f32 = np.asarray(inp_data, dtype=np.float32)
            trf_f32 = np.asarray(trf_data, dtype=np.float32)

            n_batch, DIM = inp_f32.shape
            assert trf_f32.shape[0] == DIM, \
                "T rows (%d) != inp cols (%d) for %s" % (trf_f32.shape[0], DIM, trf_name)

            large_mode = (DIM == _DIM_LARGE)
            if DIM not in (_DIM_SMALL, _DIM_LARGE):
                if verbose:
                    print("  [warn] Unsupported DIM=%d for %s; falling back to CPU" % (DIM, trf_name))
                output = inp_f32 @ trf_f32.T
                materialized[trf_name] = output.astype(np.float32)
                self._dense_cache[call.handle.value] = output
                return

            if verbose:
                print("  %s: n_batch=%d DIM=%d mode=%s" % (
                    trf_name, n_batch, DIM, "large" if large_mode else "small"))

            output = hw_gemm_mat_trf(inp_f32, trf_f32, env, remote,
                                     large_mode=large_mode, _mod_cache=_mod_cache)
            materialized[trf_name] = output.astype(np.float32)
            # Cache for chained transforms (5→3 after 7→5)
            self._dense_cache[call.handle.value] = output

    _HWMaterializingVisitor().visit(mod_typed["main"])

    elapsed = time.time() - t0
    if materialized:
        total_bytes = sum(v.nbytes for v in materialized.values())
        print("  Materialized: %d transforms (%.1f MB) in %.1fs"
              % (len(materialized), total_bytes / (1024.0 ** 2), elapsed))
    else:
        print("  [warn] No transforms were materialized")

    return materialized, set(materialized.keys())


# ============================================================
# Build / Compile / Run  (mirrors step3_merged_mod_deriv_poc.py)
# ============================================================
def build_merged_artifacts(subnet_id, arch, ofa_net, base_weights,
                           transform_matrices, bn_params, other_params):
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
    print("  Pool variables: %d" % len(pool_var_names))

    ir_path = os.path.join(RESULTS_DIR, "merged_relay_ir_%s.txt" % subnet_id)
    with open(ir_path, "w") as f:
        f.write(str(mod_full))
    print("  Relay IR -> %s" % ir_path)

    return {
        "mod_full": mod_full,
        "tvm_params_full": tvm_params_full,
        "pool_var_names": pool_var_names,
    }


def compile_and_run(subnet_id, arch, merged_artifacts, pool, env, remote, ofa_net,
                    input_np, cpu_mode=False, enable_dynamic_dense_quant=False,
                    enable_graph_pack=True):
    sep("Compile + Run: %s" % subnet_id)

    mod_full       = merged_artifacts["mod_full"]
    tvm_params_full = merged_artifacts["tvm_params_full"]
    pool_var_names  = merged_artifacts["pool_var_names"]

    transform_var_names = {"pool_" + k for k in pool["transform_matrices"].keys()}
    has_transforms = bool(transform_var_names)

    # ------------------------------------------------------------------
    # Step 4: Materialize transforms (hardware or CPU)
    # ------------------------------------------------------------------
    if has_transforms:
        if cpu_mode:
            # CPU path: pure numpy (reference, for comparison)
            from step3_gemm_mat_trf_integration import step4_materialize_transforms
            materialized, _ = step4_materialize_transforms(
                mod_full, pool, transform_var_names, verbose=True
            )
            mode_str = "CPU numpy"
        else:
            # Hardware path: execute transforms on VTA GEMM_Mat_Trf
            materialized, _ = step4_hw_materialize_transforms(
                mod_full, pool, transform_var_names, env, remote, verbose=True
            )
            mode_str = "VTA GEMM_Mat_Trf"
        print("  Materialization mode: %s  (%d ops)" % (mode_str, len(materialized)))

        # Step 4b: substitute hardware results as relay.const
        mod_full = step4b_lower_gemm_mat_trf(
            mod_full,
            transform_var_names,
            precomputed=materialized,
            vta_intrinsic=False,
        )

    # ------------------------------------------------------------------
    # Step 5: Quantize
    # ------------------------------------------------------------------
    sep("Step 5: Quantize")
    t0 = time.time()
    with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
        prev = os.environ.get("TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX")
        try:
            if enable_dynamic_dense_quant:
                os.environ["TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX"] = "1"
            else:
                os.environ.pop("TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX", None)
            with relay.quantize.qconfig(
                global_scale=GLOBAL_SCALE,
                skip_conv_layers=SKIP_CONV_LAYERS,
                skip_dense_layer=(not enable_dynamic_dense_quant),
            ):
                mod_q = quantize_with_dynamic_weights(
                    mod_full, tvm_params_full,
                    dynamic_weight_var_names=pool_var_names,
                )
        finally:
            if prev is None:
                os.environ.pop("TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX", None)
            else:
                os.environ["TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX"] = prev
    print("  Quantize done in %.1fs" % (time.time() - t0))
    kept = sorted({v.name_hint for v in relay.analysis.free_vars(mod_q["main"].body)}
                  .intersection(set(pool_var_names)))
    print("  Pool vars kept after quantize: %d" % len(kept))

    # ------------------------------------------------------------------
    # Step 6: Rewrite pool vars to int8 runtime inputs
    # ------------------------------------------------------------------
    sep("Step 6: Int8 Pool Constants")
    print("  Rewriting graph for int8 pool...")
    mod_compile, int8_pool_params, runtime_pool_params = _materialize_int8_pool_constants(
        mod_q, tvm_params_full, pool_var_names,
    )
    runtime_pool_params.update(int8_pool_params)
    print("  int8: %d  float32: %d"
          % (len(int8_pool_params), len(runtime_pool_params) - len(int8_pool_params)))

    params_for_build = {k: v for k, v in tvm_params_full.items()
                        if k not in set(pool_var_names)}

    # ------------------------------------------------------------------
    # Step 7: graph_pack + relay.build
    # ------------------------------------------------------------------
    sep("Step 7: Build (graph_pack + VTA)")
    schedule_logs = load_schedule_logs()
    print("  Schedule logs: %d" % len(schedule_logs))

    pack_entry = PACK_DICT.get(MODEL_NAME, ["nn.max_pool2d", "nn.adaptive_avg_pool2d"])
    if enable_graph_pack:
        print("  Applying graph_pack...")
        with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
            relay_prog = graph_pack(
                mod_compile["main"],
                env.BATCH, env.BLOCK_IN, env.BLOCK_OUT, env.WGT_WIDTH,
                start_name=pack_entry[0],
                stop_name=pack_entry[1],
                device_annot=(env.TARGET == "intelfocl"),
            )
    else:
        relay_prog = mod_compile["main"]
        print("  graph_pack: SKIPPED")

    t0 = time.time()
    build_timeout = int(os.environ.get("STEP3_RELAY_BUILD_TIMEOUT_SEC", "600"))

    def _on_timeout(_s, _f):
        raise TimeoutError("relay.build timed out (%ds)" % build_timeout)

    with autotvm.tophub.context(env.target, extra_files=schedule_logs):
        with vta.build_config(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
            if build_timeout > 0:
                old = signal.signal(signal.SIGALRM, _on_timeout)
                faulthandler.dump_traceback_later(max(build_timeout - 5, 1), repeat=False)
                signal.alarm(build_timeout)
            try:
                graph, lib, params = relay.build(
                    relay_prog,
                    target=env.target,
                    target_host=env.target_host,
                    params=params_for_build,
                )
            finally:
                if build_timeout > 0:
                    signal.alarm(0)
                    faulthandler.cancel_dump_traceback_later()
                    signal.signal(signal.SIGALRM, old)
    print("  relay.build done in %.1fs" % (time.time() - t0))

    # ------------------------------------------------------------------
    # Step 8: Upload + run inference
    # ------------------------------------------------------------------
    sep("Step 8: Run Inference")

    temp = tvm_utils.tempdir()
    lib_path = temp.relpath("graphlib_hw_gmtf_%s.tar" % subnet_id)
    lib.export_library(lib_path)
    remote.upload(lib_path)
    remote_lib = remote.load_module("graphlib_hw_gmtf_%s.tar" % subnet_id)
    ctx = remote.ext_dev(0)

    m = graph_runtime.create(graph, remote_lib, ctx)
    m.set_input(**params)

    copied = 0
    copied_bytes = 0
    for name, arr in sorted(runtime_pool_params.items(),
                             key=lambda kv: kv[1].nbytes, reverse=True):
        slot = m.get_input(name)
        if slot is None:
            continue
        slot.copyfrom(arr)
        copied += 1
        copied_bytes += int(arr.nbytes)
    print("  Uploaded pool vars: %d  (%.1f MB)"
          % (copied, copied_bytes / (1024.0 * 1024.0)))

    inp_tvm = tvm.nd.array(input_np.astype("float32"), ctx)
    m.set_input(INPUT_NAME, inp_tvm)
    print("  Running inference...")
    t0 = time.time()
    m.run()
    inf_ms = (time.time() - t0) * 1000.0

    vta_out = m.get_output(0).asnumpy()
    ref_out = get_ofa_reference_output(ofa_net, arch, input_np)

    top1_vta = int(np.argmax(vta_out[0]))
    top1_ref = int(np.argmax(ref_out[0]))

    print("  Top-1 VTA  : %d" % top1_vta)
    print("  Top-1 REF  : %d" % top1_ref)
    print("  Top-1 match: %s" % ("YES" if top1_vta == top1_ref else "NO"))
    print("  Inference  : %.1f ms" % inf_ms)

    return {
        "top1_vta": top1_vta,
        "top1_ref": top1_ref,
        "top1_match": bool(top1_vta == top1_ref),
        "inf_time_ms": float(inf_ms),
        "materialization": "cpu" if cpu_mode else "hw_gemm_mat_trf",
    }


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="OFA subnet inference with hardware GEMM_Mat_Trf transforms"
    )
    p.add_argument("--sa-results",  default=SA_RESULTS_FILE)
    p.add_argument("--arch-file",   default=ARCH_FILE)
    p.add_argument("--n",           type=int,   default=25)
    p.add_argument("--lambda",      dest="lambda_value", type=float, default=4.0)
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--num-subnets", type=int,   default=1)
    p.add_argument("--skip-vta",    action="store_true",
                   help="Stop after build, do not run inference on device")
    p.add_argument("--cpu-mode",    action="store_true",
                   help="Use CPU numpy for transform materialization (reference path)")
    p.add_argument("--enable-dynamic-dense-quant", action="store_true")
    p.add_argument("--no-graph-pack", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    sep("OFA Subnet Inference — Hardware GEMM_Mat_Trf")
    print("  Materialization : %s" % ("CPU numpy" if args.cpu_mode else "VTA GEMM_Mat_Trf HW"))
    print("  graph_pack      : %s" % ("DISABLED" if args.no_graph_pack else "ENABLED"))

    # ---- Step 1: Load OFA model ----
    sep("Step 1: Load OFA Model")
    t0 = time.time()
    ofa_net = OFADynamicResnetAllMod()
    ckpt = torch.load(OFA_CHECKPOINT, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    ofa_net.load_state_dict(state, strict=False)
    ofa_net.eval()
    print("  OFA loaded in %.1fs" % (time.time() - t0))

    # ---- Step 2: Load OFA pool ----
    sep("Step 2: Load OFA Pool")
    t0 = time.time()
    pool = load_ofa_pool(POOL_DIR)
    base_weights     = pool["base_weights"]
    transform_matrices = pool["transform_matrices"]
    bn_params        = pool["bn_params"]
    other_params     = pool["other_params"]
    print("  Pool loaded in %.1fs  (%d base, %d tm, %d bn, %d other)"
          % (time.time() - t0, len(base_weights), len(transform_matrices),
             len(bn_params), len(other_params)))

    # ---- Step 3: Select subnets ----
    sep("Step 3: Select Subnets")
    poc_archs = pick_subnets_from_sa(
        args.sa_results, args.arch_file,
        target_n=args.n, target_lambda=args.lambda_value,
        target_seed=args.seed, k=args.num_subnets,
    )

    # ---- Set up VTA RPC (needed for hardware materialization in step 4) ----
    sep("VTA RPC Setup")
    env = vta.get_env()
    remote = rpc.connect(DEVICE_HOST, DEVICE_PORT)
    vta.reconfig_runtime(remote)
    print("  Connected: %s:%d" % (DEVICE_HOST, DEVICE_PORT))
    print("  Config: BATCH=%d BLOCK_IN=%d BLOCK_OUT=%d" % (env.BATCH, env.BLOCK_IN, env.BLOCK_OUT))

    rng = np.random.default_rng(99)
    input_np = rng.standard_normal(INPUT_SHAPE).astype("float32")

    all_results = {}
    overall_passed = True

    for subnet_id, arch in poc_archs.items():
        sep("Processing: %s" % subnet_id)
        try:
            # Build merged Relay graph
            merged_artifacts = build_merged_artifacts(
                subnet_id, arch, ofa_net,
                base_weights, transform_matrices, bn_params, other_params,
            )

            if args.skip_vta:
                print("  --skip-vta: stopping after Relay build")
                all_results[subnet_id] = {"build": "OK", "vta": "SKIPPED"}
                continue

            result = compile_and_run(
                subnet_id, arch, merged_artifacts, pool, env, remote, ofa_net,
                input_np=input_np,
                cpu_mode=args.cpu_mode,
                enable_dynamic_dense_quant=args.enable_dynamic_dense_quant,
                enable_graph_pack=(not args.no_graph_pack),
            )
            all_results[subnet_id] = {"build": "OK", "run": result}
            if not result["top1_match"]:
                overall_passed = False

        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[subnet_id] = {"error": str(e)}
            overall_passed = False

    del remote

    sep("Summary")
    for sid, res in all_results.items():
        status = "OK" if "error" not in res else ("FAILED: " + res["error"][:60])
        print("  %s: %s" % (sid, status))
    print("\nOverall: %s" % ("PASS" if overall_passed else "FAIL (top-1 mismatch or error)"))

    summary_path = os.path.join(RESULTS_DIR, "hw_gemm_mat_trf_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print("Summary -> %s" % summary_path)


if __name__ == "__main__":
    main()
