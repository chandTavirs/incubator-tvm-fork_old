"""
Integration Example: Using GEMM_Mat_Trf with step3_merged_mod_deriv_poc.py
===========================================================================

This script demonstrates how to integrate GEMM_Mat_Trf matrix transform
materialization into the existing step3 workflow WITHOUT modifying step3 itself.

It wraps step3's build/compile/run pipeline with optional pre-materialization
of transforms on the CPU side.
"""

from __future__ import absolute_import, print_function

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Path setup
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
from tvm.contrib import graph_runtime
import vta
from vta.top import graph_pack

from ofa_base_models import OFADynamicResnetAllMod
from ofa_derivation_extractor import OFADerivationExtractor
from ofa_relay_graph_builder import build_relay_with_ofa_pool_vars
from ofa_weight_pool_extractor import load_ofa_pool
from quantize_dynamic_weights import quantize_with_dynamic_weights

# NEW: Import GEMM_Mat_Trf utilities
from vta.runtime_gemm_mat_trf import vta_gemm_mat_trf_cpu
from vta.phase_gemm_mat_trf import (
    identify_transform_matrices,
    materialize_transforms_cpu,
    apply_gemm_mat_trf_pass,
)

# Import step3 functions (reuse as-is)
from step3_merged_mod_deriv_poc import (
    load_arch_mapping,
    pick_subnets_from_sa,
    get_ofa_reference_output,
    load_schedule_logs,
    _materialize_int8_pool_constants as _materialize_int8_pool_constants_base,
    _PoolLadderStripper,
    _quantize_np_to_int8,
    FIRST_LAYER_FLOAT_POOL_VARS,
)
from tvm.relay.expr_functor import ExprMutator


class _SafeParamBinder(ExprMutator):
    """Substitute Relay Vars like relay.expr.bind but without beta-reducing composites.

    relay.expr.bind (C++) beta-reduces any relay.Call whose op is a relay.Function,
    which inlines composite functions like 'vta.gemm_mat_trf' before we can handle
    them.  This Python-level mutator does the same variable substitution but
    explicitly skips beta-reduction for composite calls.
    """

    def __init__(self, var_map):
        super().__init__()
        self._var_map = var_map

    def visit_var(self, var):
        return self._var_map.get(var, var)

    def visit_call(self, call):
        if isinstance(call.op, relay.Function):
            # Only visit args; do NOT call self.visit(call.op) which would trigger
            # the parent's visit_function and eventually beta-reduce the call.
            new_args = [self.visit(arg) for arg in call.args]
            return relay.Call(call.op, new_args, call.attrs, call.type_args)
        return super().visit_call(call)


class _CompositeAwarePoolLadderStripper(_PoolLadderStripper):
    """Extends _PoolLadderStripper to handle composite function calls.

    After _SafeParamBinder rewrites pool vars to int8, composite functions
    (e.g. 'vta.gemm_mat_trf') may receive int8 args but still declare float32
    params.  This subclass detects that mismatch and inserts casts so
    InferType does not fail.  The composite body is never mutated.
    """

    def visit_call(self, call):
        if isinstance(call.op, relay.Function):
            try:
                # tvm.ir.DictAttrs has no .get() — use [] with try/except
                composite_tag = call.op.attrs and call.op.attrs["Composite"]
            except Exception:
                composite_tag = None
            if composite_tag:
                new_args = []
                for arg, param in zip(call.args, call.op.params):
                    new_arg = self.visit(arg)
                    expected = (
                        str(param.type_annotation.dtype)
                        if isinstance(param.type_annotation, tvm.ir.TensorType)
                        else None
                    )
                    actual = self._expr_dtype_no_checked_type(new_arg)
                    if expected and actual and expected != actual:
                        new_arg = relay.cast(new_arg, expected)
                    new_args.append(new_arg)
                return relay.Call(call.op, new_args, call.attrs, call.type_args)
        return super().visit_call(call)


class _Fix3x3BNFoldShift(ExprMutator):
    """Fix 3×3 (two-stage Dense) BN fold: relay.quantize emits shift=12/bias=2048, but
    stage2_int8 × BN_scale_16 = 127 × 16 = 2032 < 2048 → ALL outputs round to zero.

    Root cause: relay.quantize adds 4 to the BN fold shift for each Dense stage's
    right_shift(4) epilogue (from _PoolLadderStripper), but this extra shift makes
    the rounding bias (2048) exceed the maximum product, zeroing every value.

    Fix: use shift=7/bias=64, which gives (127 × 1 + 64) >> 7 = 1 → non-zero
    even when BN_scale rounds to 1 (block3 large-channel case). Safe to apply
    globally: shift=12 and bias=2048 appear ONLY in 3×3 BN folds in this graph.
    """
    def visit_call(self, call):
        call = super().visit_call(call)
        if not isinstance(call.op, tvm.ir.Op):
            return call
        if call.op.name == "right_shift" and len(call.args) == 2:
            lhs, rhs = call.args
            if isinstance(rhs, relay.Constant):
                val = int(rhs.data.asnumpy().flat[0])
                if val == 12:
                    return relay.right_shift(lhs, relay.const(7, "int32"))
        if call.op.name == "add" and len(call.args) == 2:
            lhs, rhs = call.args
            if isinstance(rhs, relay.Constant):
                arr = rhs.data.asnumpy()
                if arr.size == 1 and int(arr.flat[0]) == 2048:
                    return relay.add(lhs, relay.const(64, "int32"))
        return call


def _fix_3x3_bn_fold_shift(mod):
    main = mod["main"]
    new_body = _Fix3x3BNFoldShift().visit(main.body)
    new_main = relay.Function(
        main.params, new_body, main.ret_type, main.type_params, main.attrs
    )
    new_mod = tvm.IRModule.from_expr(new_main)
    return relay.transform.InferType()(new_mod)


class _Fix5x5BNFoldShift(ExprMutator):
    """Fix 5×5 (single-stage Dense) BN fold: relay.quantize emits shift=8/bias=128.

    Root cause: graphpack ×16 on both Dense Large inputs yields 256×/16 = 16×
    over-amplified stage1 vs. what relay.quantize calibrated for. Rounding bias
    (128) exceeds stage1×BN_scale for BN_scale=1 channels: 127×1+128=255 < 256
    → zero for every BN_scale=1 channel (block3 BN_gamma ≈ 0.06).

    Fix: shift=5/bias=16. Threshold drops from stage1≥128 (impossible, int8 max=127)
    to stage1≥16, recovering BN_scale=1 channels: (127×1+16)>>5=4.

    Shape guard: only patches add(X,128)→right_shift(8) where X has 5×5 spatial
    dims. Leaves 3×3 BN folds that also use shift=8 (arch variants) untouched.
    """

    def visit_call(self, call):
        call = super().visit_call(call)
        if not isinstance(call.op, tvm.ir.Op):
            return call
        if call.op.name != "right_shift" or len(call.args) != 2:
            return call
        lhs, rhs = call.args
        if not isinstance(rhs, relay.Constant):
            return call
        if int(rhs.data.asnumpy().flat[0]) != 8:
            return call
        if not (isinstance(lhs, relay.Call) and isinstance(lhs.op, tvm.ir.Op)
                and lhs.op.name == "add" and len(lhs.args) == 2):
            return call
        add_rhs = lhs.args[1]
        if not isinstance(add_rhs, relay.Constant):
            return call
        bias_arr = add_rhs.data.asnumpy()
        if bias_arr.size != 1 or int(bias_arr.flat[0]) != 128:
            return call
        try:
            shape = [int(d) for d in lhs.checked_type.shape]
        except Exception:
            return call
        if len(shape) < 2 or shape[-2] != 5 or shape[-1] != 5:
            return call
        new_add = relay.add(lhs.args[0], relay.const(16, "int32"))
        return relay.right_shift(new_add, relay.const(5, "int32"))


def _fix_5x5_bn_fold_shift(mod):
    main = mod["main"]
    new_body = _Fix5x5BNFoldShift().visit(main.body)
    new_main = relay.Function(
        main.params, new_body, main.ret_type, main.type_params, main.attrs
    )
    new_mod = tvm.IRModule.from_expr(new_main)
    return relay.transform.InferType()(new_mod)


def _materialize_int8_pool_constants(mod_q, tvm_params_full, pool_var_names):
    """Like the base version but uses _CompositeAwarePoolLadderStripper.

    This ensures that composite function calls (e.g. 'vta.gemm_mat_trf')
    receive properly cast args after pool vars are rewritten to int8, so
    relay.transform.InferType() does not raise a BroadcastRel type error.
    """
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

    body = _SafeParamBinder(param_map).visit(main.body)
    body = _CompositeAwarePoolLadderStripper(set(int8_pool_vars)).visit(body)

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
# Config (same as step3)
# ============================================================
OFA_CHECKPOINT = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth"
ARCH_FILE = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json"
POOL_DIR = os.path.join(SCRIPT_DIR, "ofa_weight_pool")
SA_RESULTS_FILE = "/home/srchand/Desktop/research/OFA_Obfs/optimization_experiments/simulated_annealing/results/sa_results_20260216-175221.json"

DEVICE_HOST = "10.42.0.188"
DEVICE_PORT = 9091

GLOBAL_SCALE = 8.0
SKIP_CONV_LAYERS = [0]
OPT_LEVEL = 3
MODEL_NAME = "resnet18"
INPUT_NAME = "input0"
INPUT_SHAPE = [1, 3, 224, 224]

PACK_DICT = {
    "resnet18": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
}

RESULTS_DIR = os.path.join(SCRIPT_DIR, "step3_gmm_mat_trf_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def sep(title="", w=72):
    if title:
        pad = (w - len(title) - 2) // 2
        print("=" * pad + " " + title + " " + "=" * (w - pad - len(title) - 2))
    else:
        print("=" * w)


# ============================================================
# Step 1: Load OFA Model
# ============================================================
def step1_load_ofa_model():
    sep("Step 1: Load OFA Model")
    print("  Loading OFA network...")
    t0 = time.time()
    ofa_net = OFADynamicResnetAllMod()
    ckpt = torch.load(OFA_CHECKPOINT, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    ofa_net.load_state_dict(state, strict=False)
    ofa_net.eval()
    print("  OFA model loaded in %.1fs" % (time.time() - t0))
    return ofa_net


# ============================================================
# Step 2: Load OFA Weight Pool
# ============================================================
def step2_load_ofa_pool():
    sep("Step 2: Load OFA Weight Pool")
    print("  Loading pool from: %s" % POOL_DIR)
    t0 = time.time()
    pool = load_ofa_pool(POOL_DIR)
    base_weights = pool["base_weights"]
    transform_matrices = pool["transform_matrices"]
    bn_params = pool["bn_params"]
    other_params = pool["other_params"]
    elapsed = time.time() - t0
    print(
        "  Pool loaded in %.1fs (%d base, %d tm, %d bn, %d other)"
        % (elapsed, len(base_weights), len(transform_matrices), len(bn_params), len(other_params))
    )
    return {
        "base_weights": base_weights,
        "transform_matrices": transform_matrices,
        "bn_params": bn_params,
        "other_params": other_params,
    }


# ============================================================
# Step 3: Generate Relay Graph
# ============================================================
def step3_generate_relay(ofa_net, arch, pool):
    sep("Step 3: Generate Relay Graph")
    print("  Architecture: %s" % str(arch))
    print("  Building Relay graph with pool variables...")
    t0 = time.time()
    
    extractor = OFADerivationExtractor(ofa_net, verbose=False)
    derivations = extractor.extract_subnet_derivations(arch, INPUT_SHAPE)
    
    mod, tvm_params = build_relay_with_ofa_pool_vars(
        ofa_net=ofa_net,
        arch=arch,
        derivations=derivations,
        base_weights=pool["base_weights"],
        transform_matrices=pool["transform_matrices"],
        bn_params=pool["bn_params"],
        other_params=pool["other_params"],
        input_shape=INPUT_SHAPE,
    )
    print("  Relay graph built in %.1fs" % (time.time() - t0))
    
    # Relay vars are named "pool_<key>" — use that prefix throughout so that
    # _SkipBindParamsPass, quantize_with_dynamic_weights, and
    # _materialize_int8_pool_constants all see consistent names.
    pool_var_names = [k for k in tvm_params.keys() if k.startswith("pool_")]
    print("  Pool variables: %d" % len(pool_var_names))
    
    return mod, tvm_params, pool_var_names


# ============================================================
# Step 4: Pre-Materialize Matrix Transforms (NEW)
# ============================================================
def _normalize_trf_names(names):
    out = set()
    for n in names:
        out.add(n)
        if n.startswith("pool_"):
            out.add(n[5:])
        else:
            out.add("pool_" + n)
    return out


def _eval_expr(expr, pool_var_dict, dense_cache):
    """Recursively evaluate a Relay expression to a numpy array.

    Handles: Var (pool lookup), strided_slice, reshape, and nn.dense (from cache).
    dense_cache maps id(dense_call_node) -> np.ndarray (populated post-order).
    """
    if isinstance(expr, relay.Var):
        v = pool_var_dict.get(expr.name_hint)
        return np.asarray(v, dtype=np.float32) if v is not None else None

    if isinstance(expr, relay.Constant):
        return expr.data.numpy().astype(np.float32)

    if not isinstance(expr, relay.Call) or not isinstance(expr.op, tvm.ir.Op):
        return None

    op_name = expr.op.name

    if op_name == "strided_slice":
        inp = _eval_expr(expr.args[0], pool_var_dict, dense_cache)
        if inp is None:
            return None
        try:
            begin   = [int(b) for b in expr.attrs.begin]
            end     = [int(e) for e in expr.attrs.end]
            strides = ([int(s) for s in expr.attrs.strides]
                       if expr.attrs.strides else [1] * len(begin))
        except Exception:
            return None
        slices = tuple(slice(b, e, s) for b, e, s in zip(begin, end, strides))
        try:
            return inp[slices]
        except Exception:
            return None

    if op_name == "reshape":
        inp = _eval_expr(expr.args[0], pool_var_dict, dense_cache)
        if inp is None:
            return None
        try:
            newshape = [int(d) for d in expr.attrs.newshape]
            # Replace any 0-meaning-keep-dim with actual dim (TVM uses 0 to mean "copy")
            actual = []
            for i, s in enumerate(newshape):
                if s == 0:
                    actual.append(inp.shape[i] if i < len(inp.shape) else 1)
                else:
                    actual.append(s)
            return inp.reshape(actual)
        except Exception:
            try:
                shape = [int(d) for d in expr.checked_type.shape]
                return inp.reshape(shape)
            except Exception:
                return None

    if op_name == "nn.dense":
        # Use C++ handle pointer (stable across Python wrapper re-instantiations).
        return dense_cache.get(expr.handle.value)

    return None


def _materialize_transforms_from_graph(mod, pool_var_dict, transform_var_names, verbose=True):
    """Walk the type-checked Relay graph and compute subnet-specific transform outputs.

    Handles both direct patterns (reshape(strided_slice(strided_slice(Var)))) and
    chained patterns where the dense input is the output of a previous transform dense.
    Uses post-order traversal so chained dense outputs are cached before they are needed.

    Returns dict: trf_var_name -> np.ndarray [subnet_out_ch*subnet_in_ch, tgt_ks^2]
    """
    mod = relay.transform.InferType()(mod)
    normalized = _normalize_trf_names(transform_var_names)
    materialized = {}
    dense_cache = {}  # id(call_node) -> np.ndarray output

    class _Walker(relay.ExprVisitor):
        def visit_call(self, call):
            # Post-order: visit children first so chained dense outputs are cached.
            super().visit_call(call)

            if not (isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.dense"):
                return
            if not (len(call.args) >= 2 and isinstance(call.args[1], relay.Var)):
                return
            trf_var = call.args[1]
            if trf_var.name_hint not in normalized:
                return

            trf_data = pool_var_dict.get(trf_var.name_hint)
            if trf_data is None:
                if verbose:
                    print("  [warn] %s: not in pool; skipping" % trf_var.name_hint)
                return

            flat = _eval_expr(call.args[0], pool_var_dict, dense_cache)
            if flat is None:
                if verbose:
                    print("  [warn] %s: could not eval data expr; skipping" % trf_var.name_hint)
                return

            trf_np = np.asarray(trf_data, dtype=np.float32)
            flat_f32 = flat.reshape(-1, flat.shape[-1]) if flat.ndim > 2 else flat.astype(np.float32)
            output = np.dot(flat_f32, trf_np.T)  # [N, tgt_ks^2]

            if verbose:
                print("  %s: %s @ trf%s -> %s"
                      % (trf_var.name_hint, flat_f32.shape, trf_np.shape, output.shape))

            dense_cache[call.handle.value] = output
            materialized[trf_var.name_hint] = output.astype(np.float32)

    _Walker().visit(mod["main"])
    return materialized


def step4_materialize_transforms(mod, pool, transform_var_names, verbose=True):
    sep("Step 4: Pre-Materialize Matrix Transforms (GEMM_Mat_Trf)")

    if not transform_var_names:
        print("  No transform matrices found; skipping materialization")
        return {}, set()

    print("  Materializing %d transform matrices on CPU (graph-aware)..." % len(transform_var_names))
    t0 = time.time()

    pool_var_dict = {}
    for key in ["base_weights", "transform_matrices", "bn_params", "other_params"]:
        for name, tensor in pool[key].items():
            arr = tensor.asnumpy() if hasattr(tensor, "asnumpy") else np.asarray(tensor)
            pool_var_dict[name] = arr
            pool_var_dict["pool_" + name] = arr

    # Walk the Relay graph to extract subnet-specific strided_slice params, so that
    # the materialized output matches the actual shape used by each nn.dense op.
    materialized = _materialize_transforms_from_graph(
        mod, pool_var_dict, transform_var_names, verbose=verbose
    )
    
    elapsed = time.time() - t0
    if materialized:
        total_bytes = sum(v.nbytes for v in materialized.values())
        print(
            "  Materialized: %d transforms (%.1f MB) in %.1fs"
            % (len(materialized), total_bytes / (1024.0 ** 2), elapsed)
        )
    
    return materialized, set(materialized.keys())


# ============================================================
# Step 4b: Lower GEMM_Mat_Trf Patterns
# ============================================================
def step4b_lower_gemm_mat_trf(mod, transform_var_names, precomputed=None, vta_intrinsic=False):
    """Apply lower_gemm_mat_trf pass.

    Always run BEFORE quantization.

    CPU mode (vta_intrinsic=False):
        Substitutes relay.const(precomputed[name]) for each matched dense transform op
        so the quantizer sees plain constants — eliminates 'Cannot find config' warnings.

    VTA intrinsic mode (vta_intrinsic=True):
        Wraps each dense transform op in a composite function tagged 'vta.gemm_mat_trf'.
        Composite functions are opaque to the quantizer so they survive quantization
        unchanged; the VTA backend schedules them with the gemm_mat_trf intrinsic
        (mode=1 GEMM, bit 7) at build time.
    """
    mode_str = "vta_composite" if vta_intrinsic else "cpu_substitution"
    sep("Step 4b: Lower GEMM_Mat_Trf Patterns (%s)" % mode_str)

    print("  Transform vars: %d" % len(transform_var_names))
    t0 = time.time()

    mod_lowered = apply_gemm_mat_trf_pass(
        mod,
        transform_var_names=transform_var_names,
        cpu_materialization=not vta_intrinsic,
        precomputed=precomputed,
        verbose=True,
    )

    print("  Pass done in %.1fs" % (time.time() - t0))
    return mod_lowered


# ============================================================
# Step 5: Quantize Module
# ============================================================
def step5_quantize_module(mod, tvm_params, pool_var_names, enable_dynamic_dense_quant=True):
    sep("Step 5: Quantize Module")
    
    print("  Quantizing merged module...")
    t0 = time.time()
    
    with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
        prev_dense_fix_env = os.environ.get("TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX")
        try:
            if enable_dynamic_dense_quant:
                os.environ["TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX"] = "1"
                print("    Dynamic dense quantization: ENABLED")
            else:
                os.environ.pop("TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX", None)
            
            with relay.quantize.qconfig(
                global_scale=GLOBAL_SCALE,
                skip_conv_layers=SKIP_CONV_LAYERS,
                skip_dense_layer=(not enable_dynamic_dense_quant),
            ):
                mod_q = quantize_with_dynamic_weights(
                    mod,
                    tvm_params,
                    dynamic_weight_var_names=pool_var_names,
                )
        finally:
            if prev_dense_fix_env is None:
                os.environ.pop("TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX", None)
            else:
                os.environ["TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX"] = prev_dense_fix_env
    
    print("    Quantization done in %.1fs" % (time.time() - t0))
    
    q_main_vars = {v.name_hint for v in relay.analysis.free_vars(mod_q["main"].body)}
    kept = sorted(q_main_vars.intersection(set(pool_var_names)))
    print("    Dynamic pool vars kept: %d" % len(kept))
    
    return mod_q, kept


# ============================================================
# Step 6: Materialize Int8 Pool Constants
# ============================================================
def step6_materialize_int8_pool(mod_q, tvm_params, pool_var_names):
    sep("Step 6: Materialize Int8 Pool Constants")
    
    print("  Rewriting graph for int8 runtime-pool materialization...")
    mod_compile, int8_runtime_pool_params, runtime_pool_params_np = _materialize_int8_pool_constants(
        mod_q,
        tvm_params,
        pool_var_names,
    )
    
    runtime_pool_params_np.update(int8_runtime_pool_params)
    print(
        "  Int8 runtime pool: %d | Float32 runtime pool: %d"
        % (len(int8_runtime_pool_params), len(runtime_pool_params_np))
    )

    print("  Fixing 3x3 BN fold shift (12->7, bias 2048->64) ...")
    mod_compile = _fix_3x3_bn_fold_shift(mod_compile)

    return mod_compile, runtime_pool_params_np


# ============================================================
# Step 7: Apply Graph Pack & Build
# ============================================================
def step7_build(mod_compile, tvm_params, pool_var_names, env):
    sep("Step 7: Build (Graph Pack + VTA Compilation)")
    
    params_for_build = {
        k: v for k, v in tvm_params.items() if k not in set(pool_var_names)
    }
    print("  Build params: %d (non-pool)" % len(params_for_build))
    
    schedule_logs = load_schedule_logs()
    print("  Using %d schedule logs" % len(schedule_logs))
    
    print("  Applying graph_pack...")
    pack_entry = PACK_DICT.get(MODEL_NAME, ["nn.max_pool2d", "nn.global_avg_pool2d"])
    with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
        mod_packed = graph_pack(
            mod_compile["main"],
            env.BATCH,
            env.BLOCK_IN,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=pack_entry[0],
            stop_name=pack_entry[1],
            device_annot=(env.TARGET == "intelfocl"),
        )
    
    print("  Building VTA target...")
    t0 = time.time()
    with autotvm.tophub.context(env.target, extra_files=schedule_logs):
        with vta.build_config(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
            graph, lib, params = relay.build(
                mod_packed,
                target=env.target,
                target_host=env.target_host,
                params=params_for_build,
            )
    
    print("  Build done in %.1fs" % (time.time() - t0))
    
    return graph, lib, params


# ============================================================
# Step 8: Run Inference
# ============================================================
def step8_run_inference(
    graph, lib, params, runtime_pool_params_np, ofa_net, arch, env, verbose=True
):
    sep("Step 8: Run Inference")
    
    if env.TARGET != "sim":
        remote = rpc.connect(DEVICE_HOST, DEVICE_PORT)
        ctx = remote.ext_dev(0)
        print("  Remote device: %s:%d" % (DEVICE_HOST, DEVICE_PORT))
        # Upload compiled library to remote device
        from tvm.contrib import utils as tvm_utils
        temp = tvm_utils.tempdir()
        lib_path = temp.relpath("graphlib_gemm_mat_trf.tar")
        lib.export_library(lib_path)
        remote.upload(lib_path)
        remote_lib = remote.load_module("graphlib_gemm_mat_trf.tar")
        m = graph_runtime.create(graph, remote_lib, ctx)
    else:
        remote = None
        ctx = tvm.cpu(0)
        print("  Target: Simulator")
        m = graph_runtime.create(graph, lib, ctx)
    
    if params:
        m.set_input(**params)
    print("  Loaded %d build-bound params" % len(params))
    
    if runtime_pool_params_np:
        copied = 0
        copied_bytes = 0
        for name, arr in runtime_pool_params_np.items():
            try:
                slot = m.get_input(name)
            except:
                if verbose:
                    print("    [warn] Missing runtime input: %s" % name)
                continue
            slot.copyfrom(arr)
            copied += 1
            copied_bytes += int(arr.nbytes)
        print("  Uploaded pool params: %d tensors (%.1f MB)" % (copied, copied_bytes / (1024.0 ** 2)))
    
    input_np = np.random.randn(*INPUT_SHAPE).astype("float32")
    inp_tvm = tvm.nd.array(input_np, ctx)
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
    
    if remote:
        del remote
    
    return {
        "top1_vta": top1_vta,
        "top1_ref": top1_ref,
        "top1_match": bool(top1_vta == top1_ref),
        "inf_time_ms": float(inf_time),
    }


def integrate_gemm_mat_trf_with_step3(
    subnet_id,
    merged_artifacts,
    ofa_pool,
    enable_gemm_mat_trf=True,
    verbose=True,
):
    """
    Optionally pre-compute GEMM_Mat_Trf transforms before compilation.

    This function takes the merged_artifacts from step3's build phase
    and optionally materializes transforms on the CPU side, reducing
    device memory pressure.

    Parameters
    ----------
    subnet_id : str
        Subnet identifier
    merged_artifacts : dict
        Output from build_merged_artifacts() in step3
    ofa_pool : dict
        OFA pool containing transform matrices
    enable_gemm_mat_trf : bool
        Whether to enable GEMM_Mat_Trf materialization
    verbose : bool
        Print progress information

    Returns
    -------
    dict
        Enhanced artifacts with pre-materialized transforms
    """
    if verbose:
        print("\n[GEMM_Mat_Trf] Integration with step3...")

    if not enable_gemm_mat_trf:
        if verbose:
            print("  GEMM_Mat_Trf: DISABLED")
        return merged_artifacts

    # Identify which pool vars are transformation matrices
    transform_matrices = ofa_pool.get("transform_matrices", {})
    if not transform_matrices:
        if verbose:
            print("  No transform matrices in pool; skipping materialization")
        return merged_artifacts

    trf_var_names = identify_transform_matrices(transform_matrices)
    if verbose:
        print("  Found %d transformation matrices" % len(trf_var_names))

    # Pre-materialize transforms on CPU
    pool_var_dict = merged_artifacts.get("tvm_params_full", {})
    try:
        materialized = materialize_transforms_cpu(
            pool_var_dict=pool_var_dict,
            transform_matrix_names=trf_var_names,
            verbose=verbose,
        )
        if verbose:
            total_bytes = sum(v.nbytes for v in materialized.values())
            print("  Pre-materialized: %d transforms (%.1f MB)" % 
                  (len(materialized), total_bytes / (1024.0 * 1024.0)))
    except Exception as e:
        if verbose:
            print("  [warn] Materialization failed: %s" % str(e))
        return merged_artifacts

    # Store materialized results in artifacts for later use
    merged_artifacts["gemm_mat_trf_materialized"] = materialized
    merged_artifacts["gemm_mat_trf_var_names"] = trf_var_names

    return merged_artifacts


def prepare_runtime_inputs_with_transforms(
    compile_artifacts,
    gemm_mat_trf_materialized=None,
    verbose=True,
):
    """
    Prepare runtime pool inputs, optionally using pre-materialized transforms.

    When GEMM_Mat_Trf is enabled, uses pre-computed transforms instead of
    raw pool variables, reducing device memory transfer.

    Parameters
    ----------
    compile_artifacts : dict
        Output from step3b_compile_merged()
    gemm_mat_trf_materialized : dict, optional
        Pre-materialized transforms (from integration step)
    verbose : bool
        Print progress information

    Returns
    -------
    dict
        Runtime pool params with transforms applied
    """
    runtime_pool_params_np = compile_artifacts.get("runtime_pool_params_np", {})

    if gemm_mat_trf_materialized is None or not gemm_mat_trf_materialized:
        return runtime_pool_params_np

    # Replace transform matrix inputs with pre-materialized results
    enhanced_params = dict(runtime_pool_params_np)
    for trf_name, trf_data in gemm_mat_trf_materialized.items():
        if trf_name in enhanced_params:
            before_size = enhanced_params[trf_name].nbytes
            after_size = trf_data.nbytes
            if verbose:
                print("  Replaced %s: %.1f MB -> %.1f MB" %
                      (trf_name, before_size / (1024.0 ** 2), after_size / (1024.0 ** 2)))
            enhanced_params[trf_name] = trf_data
        else:
            enhanced_params[trf_name] = trf_data
            if verbose:
                print("  Added %s: %.1f MB (new)" %
                      (trf_name, trf_data.nbytes / (1024.0 ** 2)))

    return enhanced_params


# ============================================================
# Main Orchestration
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="step3 + GEMM_Mat_Trf integration")
    p.add_argument("--sa-results", default=SA_RESULTS_FILE)
    p.add_argument("--arch-file", default=ARCH_FILE)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--lambda", dest="lambda_value", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-subnets", type=int, default=1)
    p.add_argument("--skip-vta", action="store_true", help="Stop after build (don't run VTA)")
    p.add_argument("--enable-dynamic-dense-quant", action="store_true")
    p.add_argument("--no-graph-pack", action="store_true")
    p.add_argument("--no-gemm-mat-trf", action="store_true", help="Disable GEMM_Mat_Trf entirely")
    p.add_argument("--vta-intrinsic", action="store_true",
                   help="VTA intrinsic path: skip CPU pre-materialization, wrap transform "
                        "dense ops in vta.gemm_mat_trf composite after quantization")
    return p.parse_args()


def main():
    args = parse_args()
    sep("step3 + GEMM_Mat_Trf Integration")
    print("  GEMM_Mat_Trf: %s" % ("ENABLED" if not args.no_gemm_mat_trf else "DISABLED"))
    print("  Graph Pack: %s" % ("ENABLED" if not args.no_graph_pack else "DISABLED"))
    
    env = vta.get_env()
    print("  VTA Environment: BLOCK_IN=%d, BLOCK_OUT=%d" % (env.BLOCK_IN, env.BLOCK_OUT))
    
    # Step 1: Load OFA model
    ofa_net = step1_load_ofa_model()
    
    # Step 2: Load OFA pool
    pool = step2_load_ofa_pool()
    
    # Select subnet
    print("\n[Select Subnet]")
    poc_archs = pick_subnets_from_sa(
        args.sa_results,
        args.arch_file,
        target_n=args.n,
        target_lambda=args.lambda_value,
        target_seed=args.seed,
        k=args.num_subnets
    )
    
    for i, (subnet_id, arch_dict) in enumerate(poc_archs.items()):
        print("\n" + "=" * 72)
        print("Processing subnet %d/%d: %s" % (i + 1, args.num_subnets, subnet_id))
        print("=" * 72)
        
        try:
            # Step 3: Generate Relay graph
            mod, tvm_params, pool_var_names = step3_generate_relay(ofa_net, arch_dict, pool)

            transform_var_names = {"pool_" + k for k in pool["transform_matrices"].keys()}
            materialized_transforms = {}
            use_gemm_mat_trf = not args.no_gemm_mat_trf and bool(transform_var_names)

            # Step 4 (CPU path only): pre-materialize transforms on CPU, then
            # substitute relay.const(precomputed) so the quantizer sees constants.
            if use_gemm_mat_trf and not args.vta_intrinsic:
                materialized_transforms, _ = step4_materialize_transforms(
                    mod, pool, transform_var_names, verbose=True
                )
                mod = step4b_lower_gemm_mat_trf(
                    mod,
                    transform_var_names,
                    precomputed=materialized_transforms,
                    vta_intrinsic=False,
                )

            # Step 5: Quantize
            # For VTA intrinsic mode the graph still has raw nn.dense transform ops
            # here — the quantizer skips dense layers (skip_dense_layer=True default),
            # so they pass through unchanged.  We wrap them in composites AFTER this
            # step so the quantizer cannot inline them (TVM's quantization passes
            # beta-reduce relay.Function call ops they encounter).
            mod_q, kept_vars = step5_quantize_module(
                mod, tvm_params, pool_var_names,
                enable_dynamic_dense_quant=args.enable_dynamic_dense_quant
            )

            # Step 4b (VTA intrinsic path only): wrap transform dense ops in
            # "vta.gemm_mat_trf" composite functions AFTER quantization so the
            # composites survive into step 6.
            if use_gemm_mat_trf and args.vta_intrinsic:
                mod_q = step4b_lower_gemm_mat_trf(
                    mod_q,
                    transform_var_names,
                    precomputed=None,
                    vta_intrinsic=True,
                )

            # Step 6: Materialize int8 pool
            mod_compile, runtime_pool_params_np = step6_materialize_int8_pool(
                mod_q, tvm_params, pool_var_names
            )
            
            # Step 7: Build
            graph, lib, params = step7_build(
                mod_compile, tvm_params, pool_var_names, env
            )
            
            if args.skip_vta:
                print("\n[Skip VTA] --skip-vta flag set; stopping here")
                continue
            
            # Step 8: Run inference
            result = step8_run_inference(
                graph, lib, params, runtime_pool_params_np,
                ofa_net, arch_dict, env, verbose=True
            )
            
            # Save result
            result_file = os.path.join(RESULTS_DIR, "%s_result.json" % subnet_id)
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            print("\n  Result saved: %s" % result_file)
            
        except Exception as e:
            print("\n  [ERROR] %s: %s" % (subnet_id, str(e)))
            import traceback
            traceback.print_exc()
            continue
    
    sep("Complete")


if __name__ == "__main__":
    main()
