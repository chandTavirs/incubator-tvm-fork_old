"""Debug script: find int8/float32 BroadcastRel mismatches before InferType."""
import os, sys
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
from tvm import relay
import vta
import torch

from ofa_base_models import OFADynamicResnetAllMod
from ofa_weight_pool_extractor import load_ofa_pool
from step3_merged_mod_deriv_poc import (
    FIRST_LAYER_FLOAT_POOL_VARS, _PoolLadderStripper, _quantize_np_to_int8,
    pick_subnets_from_sa, load_schedule_logs,
    _materialize_int8_pool_constants as _materialize_int8_pool_constants_base,
    build_merged_artifacts,
    OFA_CHECKPOINT, ARCH_FILE, POOL_DIR, SA_RESULTS_FILE,
)
from step3_gemm_mat_trf_integration import (
    step5_quantize_module, step4b_lower_gemm_mat_trf,
    _SafeParamBinder, _CompositeAwarePoolLadderStripper,
)
from tvm.relay.expr_functor import ExprVisitor


class MixedDtypeChecker(ExprVisitor):
    """Walk body and report binary ops where arg dtypes don't match."""

    BINARY_OPS = {"multiply", "add", "subtract", "divide", "nn.dense"}

    def __init__(self, stripper):
        super().__init__()
        self._s = stripper
        self.issues = []

    def _dt(self, expr):
        return self._s._expr_dtype_no_checked_type(expr)

    def visit_call(self, call):
        super().visit_call(call)
        if not isinstance(call.op, tvm.ir.Op):
            return
        op_name = call.op.name
        if op_name in self.BINARY_OPS and len(call.args) >= 2:
            lhs_dt = self._dt(call.args[0])
            rhs_dt = self._dt(call.args[1])
            if lhs_dt and rhs_dt and lhs_dt != rhs_dt:
                self.issues.append((op_name, lhs_dt, rhs_dt, call))
                print(f"  MISMATCH [{op_name}]: lhs={lhs_dt}  rhs={rhs_dt}")
                # Print a snippet showing what the args look like
                try:
                    lhs_str = str(call.args[0])[:120]
                    rhs_str = str(call.args[1])[:120]
                    print(f"    lhs: {lhs_str}")
                    print(f"    rhs: {rhs_str}")
                except Exception:
                    pass


def run():
    ofa_net = OFADynamicResnetAllMod()
    ck = torch.load(OFA_CHECKPOINT, map_location="cpu")
    ofa_net.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    ofa_net.eval()
    pool = load_ofa_pool(POOL_DIR)

    archs = pick_subnets_from_sa(SA_RESULTS_FILE, ARCH_FILE, target_n=25,
                                  target_lambda=4.0, target_seed=0, k=1)
    (subnet_id, arch), = archs.items()

    artifacts = build_merged_artifacts(subnet_id, arch, ofa_net,
                                       pool["base_weights"], pool["transform_matrices"],
                                       pool["bn_params"], pool["other_params"])
    mod_full       = artifacts["mod_full"]
    tvm_params_full = artifacts["tvm_params_full"]

    all_dynamic_var_names  = list(tvm_params_full.keys())
    pool_only_var_names    = [k for k in tvm_params_full.keys() if k.startswith("pool_")]
    transform_var_names    = {"pool_" + k for k in pool["transform_matrices"].keys()}

    mod_q, _ = step5_quantize_module(
        mod_full, tvm_params_full, all_dynamic_var_names, enable_dynamic_dense_quant=True)
    mod_q = step4b_lower_gemm_mat_trf(
        mod_q, transform_var_names, precomputed=None, vta_intrinsic=True)

    # Reproduce _materialize_int8_pool_constants up to just before InferType
    main = mod_q["main"]
    int8_pool_vars = sorted([n for n in pool_only_var_names
                              if n not in FIRST_LAYER_FLOAT_POOL_VARS])

    param_map  = {}
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

    print("\n=== Checking for int8/float32 mismatches after PoolLadderStripper ===")
    stripper_instance = _CompositeAwarePoolLadderStripper(set(int8_pool_vars))
    checker = MixedDtypeChecker(stripper_instance)
    checker.visit(body)
    if not checker.issues:
        print("  No mismatches found — the crash must be from a different cause.")
    else:
        print(f"\nTotal mismatches: {len(checker.issues)}")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    run()
