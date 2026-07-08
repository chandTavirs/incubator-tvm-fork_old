"""
GMTF Derive-then-Infer Switching Runtime  (step7)
==================================================

Architecture
------------
Separates the GMTF kernel derivation from conv2d inference into two relay
modules:

  Derive module  (VTA target — vta.gmtf_dense_small/large execute on FPGA):
    Inputs : shared base weights (pre-uploaded to ext_dev CMA, device-local) +
             per-subnet transform matrices (~50 KB, small RPC upload) +
             per-subnet BN scale params (gamma, var) for weight fusion
    Outputs: derived int8 conv kernels as a Tuple  (in ext_dev CMA)
    Cost   : paid ONCE per switch (~1.4-2.6s VTA GMTF + ~50ms bind)

  Infer module  (VTA target, identical CMA footprint to step4 subnet):
    Inputs : image + BN bias params (mean, beta) + FC params +
             derived_weight_i NDArrays (ext_dev→ext_dev CMA memcpy, ~GB/s)
    Outputs: predictions
    Cost   : ~117ms per inference (same as K-resident step4)

Switch overhead = derive_run (~1.4-2.6s VTA GMTF) + bind_derived (CMA memcpy ~ms)
Per-inference   = ~117ms (VTA conv2d only, no re-derivation)

Comparison table (projected)
-----------------------------
  K-resident (step4): 0 ms switch,  ~117 ms run,  ~53·K MB CMA
  Lazy K=1   (step5): ~18 s switch, ~117 ms run,  ~53 MB CMA
  GMTF-VTA   (step6): ~368 ms switch, ~4079 ms run, ~211 MB CMA
  GMTF deriv (step7): ~4-7 s switch, ~117 ms run,  ~66 MB CMA (K-independent)

Usage
-----
  VTA_RPC_HOST=10.42.0.188 VTA_RPC_PORT=9092 \\
      python step7_gmtf_deriv_infer.py --num-subnets 2 --switch-iters 20
"""
from __future__ import absolute_import, print_function

import argparse
import collections
import gc
import json
import os
import random
import sys
import time

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_DIR = os.path.dirname(SCRIPT_DIR)
SRI_SCRIPTS_DIR = os.path.dirname(PHASE2_DIR)
TVM_ROOT = os.path.abspath(os.path.join(SRI_SCRIPTS_DIR, "..", ".."))
EXTERNAL_REPO_ROOT = "/home/srchand/Desktop/research/OFA_Obfs"
for p in [EXTERNAL_REPO_ROOT, os.path.join(TVM_ROOT, "python"),
          os.path.join(TVM_ROOT, "vta", "python"), SRI_SCRIPTS_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import tvm
from tvm import autotvm, relay, rpc
from tvm.contrib import graph_runtime, utils as tvm_utils
import vta
from vta.top import graph_pack

from ofa_base_models import OFADynamicResnetAllMod
from ofa_weight_pool_extractor import load_ofa_pool

from step3_merged_mod_deriv_poc import (
    OFA_CHECKPOINT, ARCH_FILE, POOL_DIR, SA_RESULTS_FILE,
    DEVICE_HOST, DEVICE_PORT, GLOBAL_SCALE, SKIP_CONV_LAYERS, OPT_LEVEL,
    MODEL_NAME, INPUT_NAME, INPUT_SHAPE, PACK_DICT,
    pick_subnets_from_sa, pick_subnets_from_sa_with_exec, get_ofa_reference_output, load_schedule_logs,
    build_merged_artifacts,
)
from step3_gemm_mat_trf_integration import (
    step5_quantize_module,
    step6_materialize_int8_pool,
)


# ============================================================
# Graph splitter: derive_fn + infer_fn
# ============================================================

class _GraphSplitter(relay.ExprMutator):
    """Single-pass: collect conv2d weight expressions AND replace them with free vars.

    id() is stable within ONE traversal (the same Python wrapper object is seen
    in every visit_call invocation for the same C++ node), so wid_to_var lookup
    is reliable.  The two-pass approach (_ConvWeightFinder + _InferBodyBuilder)
    was broken because id() values differed between separate traversal contexts.

    For each nn.conv2d we recurse into args[0] (activation) but NOT args[1]
    (weight), so the entire GMTF derivation sub-graph is excised from the infer
    body and replaced by a new derived_weight_i free var.
    """

    def __init__(self):
        super().__init__()
        self._weight_exprs = []    # original weight expressions (for derive body)
        self._derived_vars = []    # new free vars (for infer body params)
        self._wid_to_var = {}      # id(orig_w) -> derived_weight_i var
        self._counter = 0

    def visit_call(self, call):
        if isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.conv2d":
            new_data = self.visit(call.args[0])   # recurse into activation only
            orig_w = call.args[1]
            wid = id(orig_w)
            if wid not in self._wid_to_var:
                shape = [int(d) for d in orig_w.checked_type.shape]
                dtype = str(orig_w.checked_type.dtype)
                v = relay.var("derived_weight_%d" % self._counter,
                              shape=shape, dtype=dtype)
                self._wid_to_var[wid] = v
                self._weight_exprs.append(orig_w)
                self._derived_vars.append(v)
                self._counter += 1
            return relay.Call(call.op, [new_data, self._wid_to_var[wid]],
                              call.attrs, call.type_args)
        return super().visit_call(call)

    @property
    def weight_exprs(self):
        return list(self._weight_exprs)

    @property
    def derived_vars(self):
        return list(self._derived_vars)


def _split_mod(mod_compile):
    """Split mod_compile into (derive_fn, infer_fn, n_weights, derived_vars).

    derive_fn: (pool_* vars, BN scale vars) -> Tuple(weight_0, ..., weight_N)
    infer_fn:  (image, BN bias vars, FC vars, derived_weight_0..N) -> predictions
    """
    main = mod_compile["main"]

    # Single pass: build infer_body (with derived_weight_i free vars) and
    # collect the original weight expressions for the derive body.
    splitter = _GraphSplitter()
    infer_body = splitter.visit(main.body)
    weights = splitter.weight_exprs
    derived_vars = splitter.derived_vars
    n_conv = len(weights)
    if n_conv == 0:
        raise RuntimeError("_split_mod: no nn.conv2d calls found in mod_compile")

    print("  [split] Found %d conv2d weight expressions" % n_conv, flush=True)

    # Derive function: inputs = pool+BN-scale vars, outputs = Tuple(all conv weights)
    derive_body = relay.Tuple(weights)
    derive_free_names = {v.name_hint for v in relay.analysis.free_vars(derive_body)}
    derive_params = [p for p in main.params if p.name_hint in derive_free_names]
    derive_fn = relay.Function(derive_params, derive_body)

    # Infer function: inputs = image + BN-bias + FC + derived_weight_i
    infer_free_names = {v.name_hint for v in relay.analysis.free_vars(infer_body)}
    orig_infer_params = [p for p in main.params if p.name_hint in infer_free_names]
    infer_fn = relay.Function(orig_infer_params + derived_vars, infer_body)

    derive_mb = sum(
        np.prod([int(d) for d in w.checked_type.shape]) * np.dtype(str(w.checked_type.dtype)).itemsize
        for w in weights
    ) / (1024.0 ** 2)
    print("  [split] Derive outputs: %d tensors  total=%.1f MB" % (n_conv, derive_mb), flush=True)
    print("  [split] Derive params:  %d  Infer orig params: %d"
          % (len(derive_params), len(orig_infer_params)), flush=True)

    return derive_fn, infer_fn, n_conv, derived_vars


# ============================================================
# Fix scale inconsistency introduced by _Fix3x3BNFoldShift
# ============================================================

class _Fix3x3InferConv2dShift(relay.ExprMutator):
    """Fix infer-module conv2d output shift for 3×3 kernels derived via two-stage Dense.

    _Fix3x3BNFoldShift changed the derive module's 3×3 BN fold from shift=12→7
    (and bias 2048→64), making the 3×3 derived weights 2^(12-7)=32× larger than
    relay.quantize expected. The infer conv2d for those layers still uses the
    relay.quantize-generated right_shift (e.g. 4), which now under-shifts by 5 bits.

    Pattern in the post-graphpack infer_fn:
        nn.conv2d(..., kernel_size=[3,3], ...) → add(bias) → right_shift(N) → clip → cast

    Fix: for 3×3 conv2d outputs, right_shift(N) → right_shift(N+5) and update the
    rounding bias to 2^(N+5-1) = 2^(N+4).

    All other kernel sizes (1×1, 5×5, 7×7) are unchanged — only 3×3 paths went
    through two Dense stages whose BN fold we patched.
    """
    _EXTRA_SHIFT = 5  # = orig_bn_fold_shift(12) - new_bn_fold_shift(7)

    def visit_call(self, call):
        call = super().visit_call(call)
        if not isinstance(call.op, tvm.ir.Op):
            return call
        if call.op.name != "right_shift" or len(call.args) != 2:
            return call
        rhs = call.args[1]
        if not isinstance(rhs, relay.Constant):
            return call
        # lhs should be add(conv2d_output, rounding_bias)
        lhs = call.args[0]
        if not (isinstance(lhs, relay.Call) and isinstance(lhs.op, tvm.ir.Op)
                and lhs.op.name == "add" and len(lhs.args) == 2):
            return call
        add_rhs = lhs.args[1]
        if not isinstance(add_rhs, relay.Constant):
            return call
        conv = lhs.args[0]
        if not (isinstance(conv, relay.Call) and isinstance(conv.op, tvm.ir.Op)
                and conv.op.name == "nn.conv2d"):
            return call
        # Only fix 3×3 kernels — those are the two-stage Dense paths
        ks = list(conv.attrs.kernel_size)
        if ks != [3, 3]:
            return call
        old_shift = int(rhs.data.asnumpy().flat[0])
        new_shift = old_shift + self._EXTRA_SHIFT
        new_bias  = 1 << (new_shift - 1)
        new_add   = relay.add(conv, relay.const(new_bias, "int32"))
        return relay.right_shift(new_add, relay.const(new_shift, "int32"))


def _fix_3x3_infer_conv2d_shift(infer_fn):
    new_body = _Fix3x3InferConv2dShift().visit(infer_fn.body)
    new_fn = relay.Function(
        infer_fn.params, new_body, infer_fn.ret_type,
        infer_fn.type_params, infer_fn.attrs,
    )
    new_mod = tvm.IRModule.from_expr(new_fn)
    return relay.transform.InferType()(new_mod)["main"]


class _Fix5x5InferConv2dShift(relay.ExprMutator):
    """Fix infer-module conv2d output shift for 5×5 kernels derived via single-stage Dense.

    _FixDeriveBNFoldShift8 changes the 5×5 BN fold from
      multiply(X, S) → add(128) → right_shift(8)
    to
      multiply(X, S) → add(8)   → right_shift(4)
    making the 5×5 derived weights ~16× larger (ratio is exact for large |S×GMTF|).
    The infer conv2d for those layers still uses the relay.quantize-generated right_shift,
    which now under-shifts by 4 bits.

    Pattern: nn.conv2d(..., kernel_size=[5,5], ...) → add(bias) → right_shift(N)
    Fix: right_shift(N) → right_shift(N+4), bias → 2^(N+4-1).
    Only 5×5 conv2d outputs are patched.
    """
    _EXTRA_SHIFT = 3  # derive weights 8× larger after add(16)→right_shift(5) formula (2^3=8)

    def visit_call(self, call):
        call = super().visit_call(call)
        if not isinstance(call.op, tvm.ir.Op):
            return call
        if call.op.name != "right_shift" or len(call.args) != 2:
            return call
        rhs = call.args[1]
        if not isinstance(rhs, relay.Constant):
            return call
        lhs = call.args[0]
        if not (isinstance(lhs, relay.Call) and isinstance(lhs.op, tvm.ir.Op)
                and lhs.op.name == "add" and len(lhs.args) == 2):
            return call
        add_rhs = lhs.args[1]
        if not isinstance(add_rhs, relay.Constant):
            return call
        conv = lhs.args[0]
        if not (isinstance(conv, relay.Call) and isinstance(conv.op, tvm.ir.Op)
                and conv.op.name == "nn.conv2d"):
            return call
        ks = list(conv.attrs.kernel_size)
        if ks != [5, 5]:
            return call
        old_shift = int(rhs.data.asnumpy().flat[0])
        new_shift = old_shift + self._EXTRA_SHIFT
        new_bias  = 1 << (new_shift - 1)
        new_add   = relay.add(conv, relay.const(new_bias, "int32"))
        return relay.right_shift(new_add, relay.const(new_shift, "int32"))


def _fix_5x5_infer_conv2d_shift(infer_fn):
    new_body = _Fix5x5InferConv2dShift().visit(infer_fn.body)
    new_fn = relay.Function(
        infer_fn.params, new_body, infer_fn.ret_type,
        infer_fn.type_params, infer_fn.attrs,
    )
    new_mod = tvm.IRModule.from_expr(new_fn)
    return relay.transform.InferType()(new_mod)["main"]


class _FixDeriveBNFoldShift8(relay.ExprMutator):
    """Fix derive-fn BN fold add(128)→right_shift(8) patterns post-graphpack.

    Two distinct patterns are handled:

    1. NO-MULTIPLY 3×3 (single-stage Dense, no BN_scale multiply visible):
           add(GMTF_int32, 128) → right_shift(8)
       (GMTF+128)>>8 = 0 for all |GMTF|≤127 → ALL weights are zero.
       Fix: _FIXES[spatial] = (new_shift, new_bias).  For spatial=3:
           add(GMTF, 4) → right_shift(3)  → (127+4)/8=16 (max), non-zero for |GMTF|≥4.

    2. WITH-MULTIPLY 5×5: NOT FIXED here (spatial==5 returns None).
       Post-graphpack correction is fundamentally incompatible with relay.quantize
       calibration: calibration sees float32 non-zero weights, so infer right_shifts
       are calibrated for that scale. Any post-graphpack fix (shift=5, runtime
       maximum(BN_scale,2)) puts derived weights at a different scale than calibration
       expected → arch_0585 regression (vta=2→8, tested both approaches).
       Proper fix requires pre-quantize formula correction in step3.

    TOP-DOWN detection reads checked_type on ORIGINAL nodes (before super().visit_call()
    creates new untyped nodes).  Prerequisite: InferType() before this pass.
    """

    _FIXES = {3: (3, 4)}  # NO-MULTIPLY: spatial → (new_shift, new_bias)

    def visit_call(self, call):
        if isinstance(call.op, tvm.ir.Op) and call.op.name == "right_shift":
            fixed = self._try_fix(call)
            if fixed is not None:
                return self.visit(fixed)
        return super().visit_call(call)

    def _try_fix(self, call):
        lhs, rhs = call.args[0], call.args[1]
        if not isinstance(rhs, relay.Constant):
            return None
        if int(rhs.data.asnumpy().flat[0]) != 8:
            return None
        if not (isinstance(lhs, relay.Call) and isinstance(lhs.op, tvm.ir.Op)
                and lhs.op.name == "add" and len(lhs.args) == 2):
            return None
        add_rhs = lhs.args[1]
        if not isinstance(add_rhs, relay.Constant):
            return None
        bias_arr = add_rhs.data.asnumpy()
        if bias_arr.size != 1 or int(bias_arr.flat[0]) != 128:
            return None
        try:
            shape = [int(d) for d in lhs.checked_type.shape]
        except Exception:
            return None
        if len(shape) < 2 or shape[-2] != shape[-1]:
            return None
        spatial = shape[-1]
        add_lhs = lhs.args[0]

        # WITH-MULTIPLY 5×5: no post-graphpack fix possible.
        # Problem: relay.quantize calibrated infer right_shifts for float32 non-zero weights.
        # Any post-graphpack weight correction changes BN_scale=1 channels from 0 to non-zero
        # at a scale that doesn't match calibration → infer accumulation off → arch_0585 2→8.
        # Proper fix requires pre-quantize formula correction in step3 so calibration sees
        # correct values; post-graphpack patching is fundamentally incompatible with calibration.
        if spatial == 5:
            return None

        # NO-MULTIPLY 3×3: spatial in _FIXES dict → all channels were all-zero, fix them.
        if spatial not in self._FIXES:
            return None
        new_shift, new_bias = self._FIXES[spatial]
        new_add = relay.add(add_lhs, relay.const(new_bias, "int32"))
        return relay.right_shift(new_add, relay.const(new_shift, "int32"))


def _fix_derive_bn_fold_shift_8(derive_fn):
    """Fix 3×3 and 5×5 BN fold shift=8 patterns on derive_fn (post-graphpack)."""
    derive_mod = relay.transform.InferType()(tvm.IRModule.from_expr(derive_fn))
    fn = derive_mod["main"]
    new_body = _FixDeriveBNFoldShift8().visit(fn.body)
    new_fn = relay.Function(fn.params, new_body, fn.ret_type, fn.type_params, fn.attrs)
    new_mod = relay.transform.InferType()(tvm.IRModule.from_expr(new_fn))
    return new_mod["main"]


# ============================================================
# Build one derive + infer module pair
# ============================================================

def build_deriv_infer_subnet(subnet_id, arch, ofa_net, pool, env, schedule_logs, verbose=True):
    """Build (derive_module, infer_module) pair for one subnet.

    Returns
    -------
    dict with keys:
        subnet_id, arch,
        derive_graph, derive_lib,      # VTA-target derive module (GMTF on FPGA)
        infer_graph,  infer_lib,       # VTA-target infer module
        infer_params,                  # non-pool constants from relay.build (usually empty)
        derive_param_names,            # ordered list of derive module input param names
        infer_orig_param_names,        # ordered list of infer module original input names
        n_derived_weights,             # N (number of conv2d weight outputs from derive)
        runtime_pool_params_np,        # int8 pool vars (base weights + transform mats)
        non_pool_params_np,            # float32 BN + FC params (for infer module)
        base_weight_names,             # set of names that are base weights (for pre-upload)
    """
    t0 = time.time()

    artifacts = build_merged_artifacts(
        subnet_id, arch, ofa_net,
        pool["base_weights"], pool["transform_matrices"],
        pool["bn_params"], pool["other_params"],
    )
    mod_full = artifacts["mod_full"]
    tvm_params_full = artifacts["tvm_params_full"]

    all_dynamic_var_names = list(tvm_params_full.keys())
    pool_only_var_names = [k for k in tvm_params_full.keys() if k.startswith("pool_")]
    # Quantize keeping all vars dynamic.
    # enable_dynamic_dense_quant=True wraps the transform var in a cast chain
    # (cast→multiply→round→clip→cast).  We do NOT call step4b (vta.gemm_mat_trf
    # composites) here: DeriveGMTFPacker handles packing directly on the raw
    # nn.dense nodes, which graphpack needs to see as nn.dense (not composites).
    mod_q, _ = step5_quantize_module(
        mod_full, tvm_params_full, all_dynamic_var_names, enable_dynamic_dense_quant=True,
    )

    # Materialize int8 pool vars; BN/FC stay float32.
    mod_compile, runtime_pool_params_np = step6_materialize_int8_pool(
        mod_q, tvm_params_full, pool_only_var_names,
    )

    # Float32 non-pool params (BN gamma/beta/mean/var, FC weight/bias).
    non_pool_var_names = [k for k in all_dynamic_var_names if not k.startswith("pool_")]
    non_pool_params_np = {}
    for k in non_pool_var_names:
        v = tvm_params_full[k]
        non_pool_params_np[k] = v.asnumpy() if hasattr(v, "asnumpy") else np.asarray(v, dtype=np.float32)

    # Apply graphpack to the FULL module BEFORE splitting so that the layout_transform
    # for each conv2d weight (OIHW → VTA packed) becomes part of the derive module output.
    # This moves ~65 MB of data reordering from inference time (every m_infer.run())
    # to switch time (once per subnet switch in m_derive.run()), saving ~50-70 ms/inference.
    mod_compile_typed = relay.transform.InferType()(mod_compile)
    pack_entry = PACK_DICT.get(MODEL_NAME, ["nn.max_pool2d", "nn.adaptive_avg_pool2d"])
    with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
        full_packed_fn = graph_pack(
            mod_compile_typed["main"], env.BATCH, env.BLOCK_IN, env.BLOCK_OUT, env.WGT_WIDTH,
            start_name=pack_entry[0], stop_name=pack_entry[1],
            device_annot=(env.TARGET == "intelfocl"),
        )
    mod_packed = relay.transform.InferType()(tvm.IRModule.from_expr(full_packed_fn))

    # Split into derive_fn + infer_fn from the packed module.
    # derive_fn outputs VTA-packed weights (includes layout_transform).
    # infer_fn receives pre-packed derived_weight_i — no layout_transform at inference time.
    derive_fn, infer_fn, n_w, derived_vars = _split_mod(mod_packed)

    # Post-graphpack BN fold fixes: recover all-zero weights caused by quantized BN fold formulas.
    # (1) NO-MULTIPLY 3×3: add(GMTF, 128)→right_shift(8) → add(GMTF, 4)→right_shift(3)
    # (2) WITH-MULTIPLY 5×5: clamp BN_scale=1→2 in multiply(GMTF, BN_scale) node.
    #     Keeps add(128)→right_shift(8) formula → no saturation, no infer compensation.
    print("  [fix-bn-fold] Fixing post-graphpack derive BN fold shift/8 patterns ...", flush=True)
    derive_fn = _fix_derive_bn_fold_shift_8(derive_fn)

    # Save post-graphpack derive + infer IR to files for inspection
    _ir_dir = os.path.join(SCRIPT_DIR, "step3_results")
    _derive_ir_path = os.path.join(_ir_dir, "derive_fn_ir_%s.txt" % subnet_id)
    _infer_ir_path  = os.path.join(_ir_dir, "infer_fn_ir_%s.txt"  % subnet_id)
    with open(_derive_ir_path, "w") as _f:
        _f.write(derive_fn.astext(show_meta_data=False))
    with open(_infer_ir_path, "w") as _f:
        _f.write(infer_fn.astext(show_meta_data=False))

    # BN scale params (gamma, var) that appear in derive_fn are per-subnet constants —
    # fold them into the derive module at build time so they need not be bound at runtime.
    derive_fn_param_names = {p.name_hint for p in derive_fn.params}
    bn_derive_params = {k: v for k, v in non_pool_params_np.items()
                        if k in derive_fn_param_names}

    # ---- Build derive module (VTA target — GMTF runs on FPGA) ----
    # graphpack (applied to the full module above) already converted nn.dense → vta.gmtf_dense_large/small
    # in the derive portion and inserted the VTA-packed data/weight layout + right_shift+clip+cast
    # epilogue. No additional packing pass is needed here.
    derive_mod = tvm.IRModule.from_expr(derive_fn)
    with autotvm.tophub.context(env.target, extra_files=schedule_logs):
        with vta.build_config(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
            derive_graph, derive_lib, derive_params_build = relay.build(
                derive_mod,
                target=env.target,
                target_host=env.target_host,
                params=bn_derive_params,
            )
    # derive_params_build contains pre-folded BN scale constants (gamma/sqrt(var+eps) evaluated
    # at compile time).  They are small (~KB) and set once per switch via set_input.

    # ---- Build infer module (VTA target) ----
    # infer_fn already uses VTA-packed weight layout from the split above; no graph_pack needed.
    infer_mod = relay.transform.InferType()(tvm.IRModule.from_expr(infer_fn))
    with autotvm.tophub.context(env.target, extra_files=schedule_logs):
        with vta.build_config(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
            infer_graph, infer_lib, infer_params_build = relay.build(
                infer_mod,
                target=env.target,
                target_host=env.target_host,
                params={},
            )

    # Derive module runtime inputs: all derive_fn params minus the folded BN scale params.
    derive_fn_typed = relay.transform.InferType()(tvm.IRModule.from_expr(derive_fn))["main"]
    derive_param_names = [p.name_hint for p in derive_fn_typed.params
                          if p.name_hint not in bn_derive_params]

    # Infer module original input parameter order (before derived_weight_i appended).
    infer_fn_typed = relay.transform.InferType()(tvm.IRModule.from_expr(infer_fn))["main"]
    all_infer_param_names = [p.name_hint for p in infer_fn_typed.params]
    derived_weight_names = ["derived_weight_%d" % i for i in range(n_w)]
    infer_orig_param_names = [n for n in all_infer_param_names if n not in set(derived_weight_names)]

    build_s = time.time() - t0
    base_weight_names = {"pool_" + k for k in pool["base_weights"].keys()}
    if verbose:
        derive_mb = sum(a.nbytes for a in runtime_pool_params_np.values()) / (1024.0 ** 2)
        print("  [build-deriv-infer] %s  n_weights=%d  pool=%.1f MB  %.0fs"
              % (subnet_id, n_w, derive_mb, build_s), flush=True)

    return {
        "subnet_id": subnet_id,
        "arch": arch,
        "derive_graph": derive_graph,
        "derive_lib": derive_lib,
        "derive_params_build": derive_params_build,
        "infer_graph": infer_graph,
        "infer_lib": infer_lib,
        "infer_params_build": infer_params_build,
        "derive_param_names": derive_param_names,
        "infer_orig_param_names": infer_orig_param_names,
        "n_derived_weights": n_w,
        "runtime_pool_params_np": runtime_pool_params_np,
        "non_pool_params_np": non_pool_params_np,
        "base_weight_names": base_weight_names,
    }


# ============================================================
# Runtime
# ============================================================

class GmtfDerivInferRuntime:
    """GMTF derive-then-infer switching runtime.

    Base weights uploaded to ext_dev(0) CMA once at startup.
    Per switch:
      1. Create derive module on ext_dev(0) → bind base weights (ext_dev→ext_dev
         CMA memcpy, ~GB/s) + transform mats (small RPC) + BN scale (small RPC)
         → run (VTA FPGA GMTF, ~1.4-2.6s).
      2. Create infer module on ext_dev(0) → bind derived_weight_i (ext_dev→ext_dev
         CMA memcpy) + BN bias + FC → ready.
    Per inference:
      set_input(image) + run() → ~117 ms  (no re-derivation).
    """

    def __init__(self, remote, ctx_cpu, ctx_vta):
        self.remote = remote
        self.ctx_cpu = ctx_cpu
        self.ctx_vta = ctx_vta
        self.subnets = []

        self.live_idx = None
        self.live_derive_m = None
        self.live_infer_m = None

        self._tmpdir = tvm_utils.tempdir()

        # Base weights pre-uploaded to ext_dev(0) CMA: name -> tvm.nd.NDArray on ext_dev
        self._shared_base_vta = {}
        self._base_weight_names = set()
        self._base_mb = 0.0

    # ------------------------------------------------------------------
    def init_shared_base_weights(self, runtime_pool_params_np, base_weight_names):
        """Upload shared int8 base weights to ext_dev(0) CMA ONCE at startup.

        Uploading to ext_dev (VTA CMA) means the derive module — also created on
        ext_dev — can bind them via same-device CMA-to-CMA assignment at switch
        time instead of a USB round-trip.  ARM CPU ops in the derive module can
        read/write CMA memory freely (it is ordinary DDR, just reserved).
        """
        self._base_weight_names = set(base_weight_names)
        t0 = time.time()
        total_bytes = 0
        for name in sorted(base_weight_names):
            arr = runtime_pool_params_np.get(name)
            if arr is None:
                continue
            vta_arr = tvm.nd.array(arr, self.ctx_vta)   # host -> ZCU104 VTA CMA (one-time)
            self._shared_base_vta[name] = vta_arr
            total_bytes += int(arr.nbytes)
        self._base_mb = total_bytes / (1024.0 ** 2)
        elapsed_ms = (time.time() - t0) * 1000.0
        print("  Shared base weights (ext_dev CMA): %d tensors  %.1f MB  upload=%.0f ms"
              % (len(self._shared_base_vta), self._base_mb, elapsed_ms), flush=True)

    # ------------------------------------------------------------------
    def register(self, built):
        """Export + upload both derive and infer libs. Keep params on host."""
        sid = built["subnet_id"]

        derive_fname = "derive_%s.tar" % sid
        infer_fname  = "infer_%s.tar" % sid
        derive_path = self._tmpdir.relpath(derive_fname)
        infer_path  = self._tmpdir.relpath(infer_fname)

        built["derive_lib"].export_library(derive_path)
        built["infer_lib"].export_library(infer_path)

        t0 = time.time()
        self.remote.upload(derive_path)
        self.remote.upload(infer_path)
        up_ms = (time.time() - t0) * 1000.0

        # Per-subnet transform matrices (int8, small).
        non_base_pool_np = {
            k: v for k, v in built["runtime_pool_params_np"].items()
            if k not in self._base_weight_names
        }
        # Float32 BN + FC params (infer module only; BN scale is folded into derive .so).
        non_pool_np = built["non_pool_params_np"]

        self.subnets.append({
            "subnet_id": sid,
            "arch": built["arch"],
            "derive_graph": built["derive_graph"],
            "derive_fname": derive_fname,
            "derive_params_build": built["derive_params_build"],  # folded BN scale constants
            "infer_graph": built["infer_graph"],
            "infer_fname": infer_fname,
            "infer_params_build": built["infer_params_build"],
            "derive_param_names": built["derive_param_names"],
            "infer_orig_param_names": built["infer_orig_param_names"],
            "n_derived_weights": built["n_derived_weights"],
            "non_base_pool_np": non_base_pool_np,    # int8 transform matrices
            "non_pool_np": non_pool_np,               # float32 BN + FC (infer only)
        })
        nb_mb = (
            sum(a.nbytes for a in non_base_pool_np.values())
            + sum(a.nbytes for a in non_pool_np.values())
        ) / (1024.0 ** 2)
        print("  Registered %-30s  non-base=%.1f MB  upload=%.0f ms"
              % (sid, nb_mb, up_ms), flush=True)

    # ------------------------------------------------------------------
    def _evict(self):
        if self.live_derive_m is not None:
            del self.live_derive_m
            self.live_derive_m = None
        if self.live_infer_m is not None:
            del self.live_infer_m
            self.live_infer_m = None
        self.live_idx = None
        gc.collect()

    # ------------------------------------------------------------------
    def _switch(self, idx):
        """Run derive module + bind infer module. Called when idx != live_idx."""
        sub = self.subnets[idx]
        self._evict()

        # ---- Step A: run derive module ----
        # CRITICAL: create on ctx_vta (ext_dev CMA) even though the module was compiled
        # for ARM CPU target.  ARM ops can read/write CMA freely (it is ordinary DDR).
        # This makes get_output(i) return ext_dev NDArrays, so set_input on the infer
        # module (also ext_dev) is a same-device CMA memcpy (~GB/s) not a USB round-trip.
        t0 = time.time()
        rlib_derive = self.remote.load_module(sub["derive_fname"])
        m_derive = graph_runtime.create(sub["derive_graph"], rlib_derive, self.ctx_vta)
        create_derive_ms = (time.time() - t0) * 1000.0

        # Bind base weights: ext_dev → ext_dev same-device CMA copy (~GB/s, not USB).
        t0 = time.time()
        for name, vta_arr in self._shared_base_vta.items():
            try:
                m_derive.set_input(name, vta_arr)
            except Exception:
                pass
        bind_base_ms = (time.time() - t0) * 1000.0

        # Bind per-subnet transform matrices (~50 KB, host numpy → ext_dev via RPC).
        # Also set folded BN scale constants (~KB total, pre-computed at build time).
        t0 = time.time()
        for name, arr in sub["non_base_pool_np"].items():
            try:
                m_derive.set_input(name, arr)
            except Exception:
                pass
        if sub["derive_params_build"]:
            m_derive.set_input(**sub["derive_params_build"])
        bind_derive_other_ms = (time.time() - t0) * 1000.0

        # Run derive (ARM CPU GMTF matmul; outputs land in ext_dev CMA).
        t0 = time.time()
        m_derive.run()
        self.ctx_vta.sync()
        derive_run_ms = (time.time() - t0) * 1000.0

        # ---- Step B: create infer module on VTA ext_dev ----
        t0 = time.time()
        rlib_infer = self.remote.load_module(sub["infer_fname"])
        m_infer = graph_runtime.create(sub["infer_graph"], rlib_infer, self.ctx_vta)
        create_infer_ms = (time.time() - t0) * 1000.0

        # Bind derived weights: ext_dev → ext_dev same-device assignment.
        # Both m_derive (ctx_vta) and m_infer (ctx_vta) are on the same device.
        # device_type check: kDLExtDev|session == kDLExtDev|session → passes on host side.
        # Server-side: memcpy within CMA at DDR bandwidth (~4 GB/s), not USB.
        t0 = time.time()
        n_w = sub["n_derived_weights"]
        for i in range(n_w):
            w_name = "derived_weight_%d" % i
            try:
                w_arr = m_derive.get_output(i)   # ext_dev NDArray in CMA
                m_infer.set_input(w_name, w_arr)  # same-device → CMA memcpy
            except Exception as e:
                print("  [warn] derived_weight_%d bind failed: %s" % (i, e), flush=True)
        bind_derived_ms = (time.time() - t0) * 1000.0

        # Bind infer module's original params (BN bias/mean + FC, float32).
        t0 = time.time()
        for name, arr in sub["non_pool_np"].items():
            if name in set(sub["infer_orig_param_names"]):
                try:
                    m_infer.set_input(name, arr)
                except Exception:
                    pass
        if sub["infer_params_build"]:
            m_infer.set_input(**sub["infer_params_build"])
        bind_infer_other_ms = (time.time() - t0) * 1000.0

        self.live_derive_m = m_derive   # keep alive (holds output buffers)
        self.live_infer_m  = m_infer
        self.live_idx      = idx

        return {
            "create_derive":     create_derive_ms,
            "bind_base":         bind_base_ms,
            "bind_derive_other": bind_derive_other_ms,
            "derive_run":        derive_run_ms,
            "create_infer":      create_infer_ms,
            "bind_derived":      bind_derived_ms,
            "bind_infer_other":  bind_infer_other_ms,
            "switch": (create_derive_ms + bind_base_ms + bind_derive_other_ms
                       + derive_run_ms + create_infer_ms + bind_derived_ms
                       + bind_infer_other_ms),
        }

    # ------------------------------------------------------------------
    def run(self, idx, input_np):
        """Run inference on subnet idx. Returns (output_np, timing_dict, subnet_id)."""
        assert 0 <= idx < len(self.subnets)
        sub = self.subnets[idx]

        switch_tm = None
        if idx != self.live_idx:
            switch_tm = self._switch(idx)

        inp = tvm.nd.array(input_np.astype("float32"), self.ctx_vta)
        self.live_infer_m.set_input(INPUT_NAME, inp)

        t0 = time.time()
        self.live_infer_m.run()
        self.ctx_vta.sync()
        run_ms = (time.time() - t0) * 1000.0

        out = self.live_infer_m.get_output(0).asnumpy()

        tm = {}
        if switch_tm is not None:
            tm.update(switch_tm)
        else:
            for k in ("create_derive", "bind_base", "bind_derive_other",
                      "derive_run", "create_infer", "bind_derived",
                      "bind_infer_other", "switch"):
                tm[k] = 0.0
        tm["run"] = run_ms
        return out, tm, sub["subnet_id"]

    # ------------------------------------------------------------------
    def teardown(self):
        self._evict()
        self._shared_base_vta.clear()
        self.subnets = []
        gc.collect()

    def __len__(self):
        return len(self.subnets)


# ============================================================
# CLI / main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="GMTF derive-then-infer switching runtime")
    p.add_argument("--sa-results", default=SA_RESULTS_FILE)
    p.add_argument("--arch-file", default=ARCH_FILE)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--lambda", dest="lambda_value", type=float, default=4.0)
    p.add_argument("--gamma", dest="gamma_value", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-subnets", type=int, default=2,
                   help="K: how many subnets (POC default = 2)")
    p.add_argument("--switch-iters", type=int, default=20)
    p.add_argument("--warm-iters", type=int, default=3)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--build-only", action="store_true",
                   help="Stop after relay.build (no VTA device needed)")
    p.add_argument("--results-out", type=str, default=None,
                   help="Path to write per-iteration JSONL results (default: auto in SCRIPT_DIR)")
    return p.parse_args()


def sep(t=""):
    print(("=" * 8) + " " + t + " " + ("=" * max(0, 60 - len(t))), flush=True)


def main():
    args = parse_args()
    sep("GMTF Derive+Infer Switching  (K=%d)" % args.num_subnets)

    # ------------------------------------------------------------------
    print("[1] Load OFA model + pool ...", flush=True)
    ofa_net = OFADynamicResnetAllMod()
    ck = torch.load(OFA_CHECKPOINT, map_location="cpu")
    ofa_net.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    ofa_net.eval()
    pool = load_ofa_pool(POOL_DIR)

    base_weight_names = {"pool_" + k for k in pool["base_weights"].keys()}

    # ------------------------------------------------------------------
    print("[2] Select K=%d subnets ..." % args.num_subnets, flush=True)
    if args.gamma_value is None:
        archs = pick_subnets_from_sa(
            args.sa_results, args.arch_file,
            target_n=args.n, target_lambda=args.lambda_value,
            target_seed=args.seed, k=args.num_subnets,
        )
    else:
        archs = pick_subnets_from_sa_with_exec(
            args.sa_results, args.arch_file,
            target_n=args.n, target_lambda=args.lambda_value, target_gamma=args.gamma_value,
            target_seed=args.seed, k=args.num_subnets,
        )

    # ------------------------------------------------------------------
    env = vta.get_env()
    schedule_logs = load_schedule_logs()

    print("[3] Build %d derive+infer subnet module pairs ..." % len(archs), flush=True)
    print("    (Build: ~60-90 s/subnet)", flush=True)
    built_list = []
    for sid, arch in archs.items():
        built = build_deriv_infer_subnet(
            sid, arch, ofa_net, pool, env, schedule_logs)
        built_list.append(built)

    if args.build_only:
        sep("Build-only mode: done")
        return

    # ------------------------------------------------------------------
    print("[4] Connect VTA RPC ...", flush=True)
    remote = rpc.connect(DEVICE_HOST, DEVICE_PORT)
    vta.reconfig_runtime(remote)
    ctx_vta = remote.ext_dev(0)
    ctx_cpu = remote.cpu(0)

    # ------------------------------------------------------------------
    print("[5] Init runtime + upload shared base weights to ext_dev(0) CMA ...", flush=True)
    rt = GmtfDerivInferRuntime(remote, ctx_cpu, ctx_vta)
    rt.init_shared_base_weights(built_list[0]["runtime_pool_params_np"], base_weight_names)

    print("[6] Register %d module pairs ..." % len(built_list), flush=True)
    for built in built_list:
        rt.register(built)

    rng = np.random.default_rng(99)
    input_np = rng.standard_normal(INPUT_SHAPE).astype("float32")

    # ------------------------------------------------------------------
    print("", flush=True)
    print("[7] Correctness: one inference per subnet ...", flush=True)
    for i, built in enumerate(built_list):
        out, tm, sid = rt.run(i, input_np)
        ref = get_ofa_reference_output(ofa_net, built["arch"], input_np)
        t1v = int(np.argmax(out[0]))
        t1r = int(np.argmax(ref[0]))
        match = "OK" if t1v == t1r else "MISMATCH"
        print("  %-30s vta=%4d ref=%4d %-9s"
              "  switch=%6.0f ms (derive_run=%5.0f  bind_derived=%5.0f)"
              "  run=%5.0f ms"
              % (sid, t1v, t1r, match,
                 tm["switch"], tm["derive_run"], tm["bind_derived"],
                 tm["run"]), flush=True)

    # ------------------------------------------------------------------
    print("", flush=True)
    print("[7b] Derive IR + output diagnostics ...", flush=True)
    for i, built in enumerate(built_list):   # all subnets
        sid = built["subnet_id"]
        ir_dir = os.path.join(SCRIPT_DIR, "step3_results")
        derive_ir_path = os.path.join(ir_dir, "derive_fn_ir_%s.txt" % sid)
        infer_ir_path  = os.path.join(ir_dir, "infer_fn_ir_%s.txt"  % sid)

        with open(derive_ir_path) as f:
            derive_ir = f.read()
        with open(infer_ir_path) as f:
            infer_ir = f.read()

        has_gmtf_large = "vta.gmtf_dense_large" in derive_ir
        has_gmtf_small = "vta.gmtf_dense_small" in derive_ir
        gmtf_large_count = derive_ir.count("vta.gmtf_dense_large")
        gmtf_small_count = derive_ir.count("vta.gmtf_dense_small")
        has_nn_dense     = "nn.dense" in derive_ir
        print("  [derive IR] %s" % sid, flush=True)
        print("    vta.gmtf_dense_large: %d occurrences" % gmtf_large_count, flush=True)
        print("    vta.gmtf_dense_small: %d occurrences" % gmtf_small_count, flush=True)
        print("    nn.dense (uncompiled): %s" % ("YES — fallback ops remain!" if has_nn_dense else "none"), flush=True)

        # Print first 60 lines of derive IR
        print("  --- derive_fn IR (first 60 lines) ---", flush=True)
        for ln in derive_ir.splitlines()[:60]:
            print("  " + ln, flush=True)

        # Infer IR snippet
        has_conv2d_infer = "nn.conv2d" in infer_ir
        infer_line_count = len(infer_ir.splitlines())
        print("  [infer IR]  lines=%d  nn.conv2d present=%s" % (infer_line_count, has_conv2d_infer), flush=True)
        print("  --- infer_fn IR (first 30 lines) ---", flush=True)
        for ln in infer_ir.splitlines()[:30]:
            print("  " + ln, flush=True)

        # Switch to this subnet and inspect derive outputs
        rt._evict()
        rt._switch(i)
        m_derive = rt.live_derive_m
        n_w = built["n_derived_weights"]

        # Compute float32 reference derived weights from pool numpy arrays.
        # Reproduce the GMTF steps: strided_slice(pool_base) → reshape → TM_mul → reshape.
        # We use the merged relay IR (pre-graphpack) op ordering to know which TM goes with which base.
        # For simplicity, compute float32 reference RMS from the numpy pool directly:
        # float32_rms = RMS of raw float32 derived weight (no quantization).
        # VTA int8 output represents float32 × HW_QUANT_SCALE; so:
        #   expected VTA RMS ≈ float32_rms × HW_QUANT_SCALE
        # We do the GMTF manually for each derive output using pool numpy arrays.
        HW_QUANT_SCALE = 128.0
        INT8_SCALE = 16.0   # input pool weight quantization: int8 = round(float32 × 16)

        # Build (name → numpy float32) for pool vars
        base_w_np = {k: v.numpy() if hasattr(v, "numpy") else np.array(v)
                     for k, v in pool["base_weights"].items()}
        tm_np = {k: v.numpy() if hasattr(v, "numpy") else np.array(v)
                 for k, v in pool["transform_matrices"].items()}

        print("  [derive outputs] RMS check (n_weights=%d):" % n_w, flush=True)
        all_zero_count = 0
        saturated_count = 0
        for wi in range(n_w):
            try:
                arr = m_derive.get_output(wi).asnumpy().astype(np.float32)
            except Exception as e:
                print("    weight_%d: ERROR %s" % (wi, e), flush=True)
                continue
            rms = float(np.sqrt(np.mean(arr ** 2)))
            amax = float(np.max(np.abs(arr)))
            nz_pct = float(100.0 * np.count_nonzero(arr) / arr.size)
            flags = []
            if rms < 0.1:
                flags.append("ALL-ZERO?")
                all_zero_count += 1
            if amax >= 126.5:
                flags.append("SATURATED?")
                saturated_count += 1
            flag_str = "  *** " + " ".join(flags) if flags else ""
            print("    weight_%2d  shape=%-30s  rms=%6.2f  max_abs=%5.1f  nz=%.0f%%%s"
                  % (wi, str(arr.shape), rms, amax, nz_pct, flag_str), flush=True)

        print("  Summary: %d all-zero, %d saturated (out of %d)"
              % (all_zero_count, saturated_count, n_w), flush=True)

    # ------------------------------------------------------------------
    print("", flush=True)
    print("[8] Warm run on subnet 0 (%d iters) ..." % args.warm_iters, flush=True)
    print("    iter 0 = switch; iters 1+ = same-subnet (fast, infer only)", flush=True)
    rt._evict()
    warm_run_times = []
    for i in range(args.warm_iters):
        out, tm, sid = rt.run(0, input_np)
        tag = "[switch]" if tm["switch"] > 0 else "[warm]  "
        print("  iter %d %s  switch=%6.0f ms  run=%5.0f ms"
              "  (derive_run=%5.0f  bind_derived=%5.0f)"
              % (i, tag, tm["switch"], tm["run"],
                 tm["derive_run"], tm["bind_derived"]), flush=True)
        if i > 0:
            warm_run_times.append(tm["run"])
    if warm_run_times:
        a = np.array(warm_run_times)
        print("  Same-subnet run (no switch): mean=%.0f ms  min=%.0f  max=%.0f"
              % (a.mean(), a.min(), a.max()), flush=True)

    # ------------------------------------------------------------------
    print("", flush=True)
    print("[9] Switch sweep: all %d subnets in order ..." % len(rt), flush=True)
    rt._evict()
    sweep_switch, sweep_derive, sweep_bind_derived, sweep_run = [], [], [], []
    sweep_records = []
    for i in range(len(rt)):
        _, tm, sid = rt.run(i, input_np)
        sweep_switch.append(tm["switch"])
        sweep_derive.append(tm["derive_run"])
        sweep_bind_derived.append(tm["bind_derived"])
        sweep_run.append(tm["run"])
        sweep_records.append({
            "phase": "sweep", "order_idx": i, "subnet_idx": i, "subnet_id": sid,
            "n_derived_weights": rt.subnets[i]["n_derived_weights"],
            "is_actual_switch": True, **tm,
        })
        print("  -> %-30s  switch=%6.0f ms (derive=%5.0f bind_w=%5.0f)  run=%5.0f ms"
              % (sid, tm["switch"], tm["derive_run"], tm["bind_derived"],
                 tm["run"]), flush=True)
    sw = np.array(sweep_switch)
    print("  Switch overhead: mean=%.0f ms  min=%.0f  max=%.0f"
          % (sw.mean(), sw.min(), sw.max()), flush=True)

    # ------------------------------------------------------------------
    print("", flush=True)
    print("[10] Random switching: %d iters ..." % args.switch_iters, flush=True)
    random.seed(args.rng_seed)
    agg_keys = ("create_derive", "bind_base", "bind_derive_other",
                "derive_run", "create_infer", "bind_derived",
                "bind_infer_other", "switch", "run")
    agg = {k: [] for k in agg_keys}
    same_run = []
    seq = []
    random_records = []
    prev_idx = rt.live_idx
    for iter_i in range(args.switch_iters):
        k = random.randrange(len(rt))
        is_actual_switch = (k != prev_idx)
        out, tm, sid = rt.run(k, input_np)
        for key in agg_keys:
            agg[key].append(tm[key])
        seq.append(k)
        random_records.append({
            "phase": "random", "iter": iter_i, "subnet_idx": k, "subnet_id": sid,
            "n_derived_weights": rt.subnets[k]["n_derived_weights"],
            "is_actual_switch": is_actual_switch, **tm,
        })
        if k == prev_idx:
            same_run.append(tm["run"])
        prev_idx = k

    print("  Switch sequence (first 20): %s" % seq[:20], flush=True)
    print("  Selection histogram: %s"
          % dict(sorted(collections.Counter(seq).items())), flush=True)
    print("", flush=True)
    for key in agg_keys:
        a = np.array(agg[key])
        print("  %-18s  mean=%6.0f ms  min=%6.0f  max=%6.0f"
              % (key, a.mean(), a.min(), a.max()), flush=True)
    if same_run:
        ss = np.array(same_run)
        print("  Same-subnet (no switch):  mean=%.0f ms  n=%d" % (ss.mean(), len(ss)),
              flush=True)

    # ------------------------------------------------------------------
    # Save per-iteration JSONL results
    if args.results_out is None:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(SCRIPT_DIR, "step7_results_%s.jsonl" % ts)
    else:
        results_path = args.results_out
    all_records = sweep_records + random_records
    with open(results_path, "w") as _f:
        for _rec in all_records:
            _f.write(json.dumps(_rec) + "\n")
    print("  Results saved -> %s  (%d records)" % (results_path, len(all_records)), flush=True)

    # ------------------------------------------------------------------
    sep("Summary")
    cma_est = rt._base_mb + 66.0   # base weights (ext_dev CMA) + one live infer module
    print("  K=%d  |  derive on VTA FPGA (GMTF)  |  infer on VTA" % len(rt), flush=True)
    print("", flush=True)
    print("  CMA breakdown (K-independent):", flush=True)
    print("    Shared base weights (ext_dev CMA, persistent): %.1f MB" % rt._base_mb,
          flush=True)
    print("    Live infer module   (ext_dev CMA, persistent): ~66 MB", flush=True)
    print("    Derive module outputs (ext_dev CMA, per-switch then freed): ~66 MB", flush=True)
    print("    VTA command queues  (ext_dev CMA, lazy):        ~64 MB", flush=True)
    print("    Steady-state CMA:   ~%.0f MB  |  Peak (during switch): ~%.0f MB"
          % (cma_est + 64.0, cma_est + 64.0 + 66.0), flush=True)
    print("", flush=True)
    print("  Timings (random-switch sweep):", flush=True)
    for key in ("switch", "derive_run", "bind_derived", "run"):
        a = np.array(agg[key])
        print("    %-18s  mean=%6.0f ms" % (key, a.mean()), flush=True)
    print("", flush=True)
    print("  Professor comparison:", flush=True)
    print("    K-resident (step4):  0 ms switch,    ~117 ms run,  ~%d MB CMA"
          % int(53 * len(rt)), flush=True)
    print("    Lazy K=1   (step5):  ~18000 ms switch (net bw), ~117 ms run, ~53 MB CMA",
          flush=True)
    print("    GMTF-VTA   (step6):  ~368 ms switch,  ~4079 ms run, ~211 MB CMA",
          flush=True)
    print("    GMTF deriv (step7):  ~%.0f ms switch,  ~%.0f ms run,  ~%.0f MB CMA"
          % (np.mean(agg["switch"]), np.mean(agg["run"]), cma_est), flush=True)
    sep("Done")

    rt.teardown()
    del remote
    gc.collect()
    print("Released device + RPC, exiting.", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
