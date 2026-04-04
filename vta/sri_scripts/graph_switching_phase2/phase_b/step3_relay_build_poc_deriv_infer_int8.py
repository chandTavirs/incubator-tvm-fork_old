"""
Phase B Step 3: Split Deriv/Infer POC with Int8 Boundary
=========================================================

This variant keeps separate derivation and inference modules, but allows the
module boundary to use int8 derived weights:

- mod_deriv computes float32 derived weights and quantizes them to int8.
- mod_infer accepts int8 derived weights and dequantizes internally.

Fallback mode is also supported for A/B debugging:
- --derived-weight-dtype=float32

The rest of the flow mirrors step3_relay_build_poc.py.
"""

from __future__ import absolute_import, print_function

import argparse
import os
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
EXTERNAL_REPO_ROOT = "/home/srchand/Desktop/research/OFA_Obfs"
VTA_ROOT = os.path.join(TVM_ROOT, "vta", "python")

for p in [EXTERNAL_REPO_ROOT, TVM_PYTHON, VTA_ROOT, SRI_SCRIPTS_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ofa_derivation_extractor import OFADerivationExtractor
from ofa_relay_graph_builder import (
    build_relay_with_ofa_pool_vars,
    split_derivation_and_inference_modules_quant_boundary,
)
from ofa_weight_pool_extractor import load_ofa_pool
from ofa_base_models import OFADynamicResnetAllMod

import step3_relay_build_poc as base


def build_first_step_relay_with_boundary(
    subnet_id,
    arch,
    ofa_net,
    base_weights,
    transform_matrices,
    bn_params,
    other_params,
    derived_weight_dtype,
    derived_weight_scale,
):
    print("  Extracting derivations...")
    extractor = OFADerivationExtractor(ofa_net, verbose=False)
    derivations = extractor.extract_subnet_derivations(arch, base.INPUT_SHAPE)
    print("  Derived layers: %d" % len(derivations))

    print("  Building OFA-pool Relay graph...")
    t0 = time.time()
    mod_full, tvm_params_full = build_relay_with_ofa_pool_vars(
        ofa_net=ofa_net,
        arch=arch,
        derivations=derivations,
        base_weights=base_weights,
        transform_matrices=transform_matrices,
        bn_params=bn_params,
        other_params=other_params,
        input_shape=base.INPUT_SHAPE,
        input_name=base.INPUT_NAME,
    )

    mod_deriv, deriv_params, mod_infer, infer_params, derived_weight_names = (
        split_derivation_and_inference_modules_quant_boundary(
            mod_full,
            tvm_params_full,
            boundary_dtype=derived_weight_dtype,
            boundary_scale=derived_weight_scale,
        )
    )

    print("  Build done in %.1fs" % (time.time() - t0))
    print("  Pool variables: %d" % len(deriv_params))
    print("  Other params:   %d" % len(infer_params))
    print("  Derived weights: %d" % len(derived_weight_names))
    print("  Derived boundary dtype: %s (scale=%.4f)" % (derived_weight_dtype, derived_weight_scale))

    return {
        "derivations": derivations,
        "mod_full": mod_full,
        "mod_deriv": mod_deriv,
        "mod_infer": mod_infer,
        "tvm_params_full": tvm_params_full,
        "deriv_params": deriv_params,
        "infer_params": infer_params,
        "derived_weight_names": derived_weight_names,
        "dynamic_weight_var_names": sorted(deriv_params.keys()),
        "derived_weight_dtype": derived_weight_dtype,
        "derived_weight_scale": float(derived_weight_scale),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Phase B Step 3 split deriv/infer with int8 boundary")
    p.add_argument("--sa-results", default=base.SA_RESULTS_FILE)
    p.add_argument("--arch-file", default=base.ARCH_FILE)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--lambda", dest="lambda_value", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-subnets", type=int, default=1)
    p.add_argument("--skip-vta", action="store_true", help="Only run CPU validation")
    p.add_argument("--skip-cpu", action="store_true", help="Skip CPU validation")
    p.add_argument(
        "--enable-dynamic-dense-quant",
        action="store_true",
        help="Enable dense quantization for dynamic-weight graph",
    )
    p.add_argument(
        "--static-debug-mode",
        action="store_true",
        help="Bind all params at infer relay.build and skip runtime derivation",
    )
    p.add_argument(
        "--derived-weight-dtype",
        choices=["int8", "float32"],
        default="int8",
        help="Boundary dtype between derivation and inference modules",
    )
    p.add_argument(
        "--derived-weight-scale",
        type=float,
        default=base.GLOBAL_SCALE,
        help="Fixed quantization scale used when --derived-weight-dtype=int8",
    )
    return p.parse_args()


def main():
    args = parse_args()
    base.sep("Phase B Step 3: Split Deriv/Infer Int8 Boundary")
    print("  Runtime mode: %s" % ("STATIC DEBUG" if args.static_debug_mode else "DYNAMIC"))
    print(
        "  Boundary mode: %s%s"
        % (
            args.derived_weight_dtype,
            (" (scale=%.4f)" % args.derived_weight_scale) if args.derived_weight_dtype == "int8" else "",
        )
    )

    print("\n[1] Loading OFA model...")
    t0 = time.time()
    ofa_net = OFADynamicResnetAllMod()
    ckpt = torch.load(base.OFA_CHECKPOINT, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    ofa_net.load_state_dict(state, strict=False)
    ofa_net.eval()
    print("  OFA loaded in %.1fs" % (time.time() - t0))

    print("\n[2] Loading OFA pool...")
    t0 = time.time()
    pool = load_ofa_pool(base.POOL_DIR)
    base_weights = pool["base_weights"]
    transform_matrices = pool["transform_matrices"]
    bn_params = pool["bn_params"]
    other_params = pool["other_params"]
    print(
        "  Pool loaded in %.1fs (%d base, %d tm, %d bn, %d other)"
        % (time.time() - t0, len(base_weights), len(transform_matrices), len(bn_params), len(other_params))
    )

    print("\n[3] Selecting subnets...")
    poc_archs = base.pick_subnets_from_sa(
        args.sa_results,
        args.arch_file,
        target_n=args.n,
        target_lambda=args.lambda_value,
        target_seed=args.seed,
        k=args.num_subnets,
    )

    if not args.skip_vta:
        print("\n[4] Setting up VTA environment...")
        env = base.vta.get_env()
        remote = base.rpc.connect(base.DEVICE_HOST, base.DEVICE_PORT)
        base.vta.reconfig_runtime(remote)
        ctx = remote.ext_dev(0)
        print("  Connected to %s:%d" % (base.DEVICE_HOST, base.DEVICE_PORT))
        print("  Target: %s" % env.target)
    else:
        print("\n[4] Skipping VTA setup (--skip-vta)")
        env = remote = ctx = None

    rng = np.random.default_rng(42)
    input_np = rng.standard_normal(base.INPUT_SHAPE).astype("float32")

    all_results = {}
    overall_passed = True
    deriv_weight_cache = {}

    for subnet_id, arch in poc_archs.items():
        base.sep("Processing %s" % subnet_id)
        try:
            split_artifacts = build_first_step_relay_with_boundary(
                subnet_id,
                arch,
                ofa_net,
                base_weights,
                transform_matrices,
                bn_params,
                other_params,
                derived_weight_dtype=args.derived_weight_dtype,
                derived_weight_scale=args.derived_weight_scale,
            )

            r3a = None
            if not args.skip_cpu:
                r3a = base.step3a_cpu_validation(subnet_id, arch, ofa_net, input_np, split_artifacts)
                if not r3a["passed"]:
                    overall_passed = False

            if args.skip_vta:
                all_results[subnet_id] = {
                    "step3a": {} if r3a is None else {k: v for k, v in r3a.items() if k not in ("derived_weight_map",)},
                    "vta": "SKIPPED",
                    "boundary": {
                        "dtype": args.derived_weight_dtype,
                        "scale": float(args.derived_weight_scale),
                    },
                }
                continue

            compile_artifacts = base.step3b_vta_compile(
                subnet_id,
                arch,
                split_artifacts,
                env,
                experimental_routing=False,
                enable_dynamic_dense_quant=args.enable_dynamic_dense_quant,
                static_debug_mode=args.static_debug_mode,
            )

            r3c = base.step3c_vta_inference(
                subnet_id,
                arch,
                compile_artifacts,
                split_artifacts,
                deriv_weight_cache,
                ofa_net,
                input_np,
                env,
                remote,
                ctx,
            )

            all_results[subnet_id] = {
                "step3a": {} if r3a is None else {k: v for k, v in r3a.items() if k not in ("derived_weight_map",)},
                "step3b": "OK",
                "step3c": r3c,
                "boundary": {
                    "dtype": args.derived_weight_dtype,
                    "scale": float(args.derived_weight_scale),
                },
            }
            if not r3c["top1_match"]:
                overall_passed = False

        except Exception as e:
            import traceback

            traceback.print_exc()
            all_results[subnet_id] = {"error": str(e)}
            overall_passed = False

    base.sep("Overall Summary")
    for sid, res in all_results.items():
        print("  %s: %s" % (sid, "OK" if "error" not in res else "FAILED"))

    print("\nOverall: %s" % ("PASS" if overall_passed else "FAIL"))

    summary_path = os.path.join(base.RESULTS_DIR, "step3_deriv_infer_int8_summary.json")
    with open(summary_path, "w") as f:
        base.json.dump(all_results, f, indent=2)
    print("Summary -> %s" % summary_path)


if __name__ == "__main__":
    main()

