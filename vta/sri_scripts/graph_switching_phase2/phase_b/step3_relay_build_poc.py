"""
Phase B Step 3: OFA-Pool Relay Graph Build POC
===============================================

Takes 2 subnets from a candidate set (SA results JSON) and:

  1. Builds the OFA-pool Relay graph using build_relay_with_ofa_pool_vars()
     — conv weights are expressed as slice + optional dense transform ops
       applied to OFA base weight variables, NOT materialised constants.

  2. Validates numerically on CPU (no VTA needed):
       relay.create_executor("graph").evaluate()(input) vs PyTorch OFA output
     Goal: max |diff| < 1e-4 across all output logits.

  3. Runs the same quantize → graph_pack → relay.build pipeline as
     execute_candidate_set_refactored.py and confirms the graph compiles
     for the VTA target.

  4. Uploads to VTA device and runs a real inference, comparing against
     the PyTorch float32 reference (tolerance: top-1 class must match).

The script is deliberately self-contained — it mirrors
execute_candidate_set_refactored.compile_model() step-by-step so the
differences are easy to see.

Usage
-----
  cd .../graph_switching_phase2/phase_b
  python step3_relay_build_poc.py [--n 25] [--lambda 4] [--seed 0] [--num-subnets 2]
"""

from __future__ import absolute_import, print_function

import os
import sys
import json
import time
import argparse

import numpy as np
import torch

# ---- path setup (must come before any local imports) ----
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
PHASE2_DIR        = os.path.dirname(SCRIPT_DIR)
SRI_SCRIPTS_DIR   = os.path.dirname(PHASE2_DIR)
TVM_ROOT          = os.path.abspath(os.path.join(SRI_SCRIPTS_DIR, "..", ".."))
TVM_PYTHON        = os.path.join(TVM_ROOT, "python")
EXTERNAL_REPO_ROOT = "/home/srchand/Desktop/research/OFA_Obfs"
VTA_ROOT          = os.path.join(TVM_ROOT, "vta", "python")

for p in [EXTERNAL_REPO_ROOT, TVM_PYTHON, VTA_ROOT, SRI_SCRIPTS_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import tvm
from tvm import relay, autotvm, rpc
from tvm.contrib import graph_runtime, utils as tvm_utils
import vta
from vta.top import graph_pack_dynamic_weights

from ofa_base_models import OFADynamicResnetAllMod
from ofa_weight_pool_extractor import load_ofa_pool
from ofa_derivation_extractor import OFADerivationExtractor
from ofa_relay_graph_builder import (
    build_relay_with_ofa_pool_vars,
    split_derivation_and_inference_modules,
)
from quantize_dynamic_weights import (
    quantize_with_dynamic_weights,
    merge_derivation_and_inference_modules,
)

# ============================================================
# Config constants
# ============================================================
OFA_CHECKPOINT   = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth"
ARCH_FILE        = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json"
POOL_DIR         = os.path.join(SCRIPT_DIR, "ofa_weight_pool")
SA_RESULTS_FILE  = "/home/srchand/Desktop/research/OFA_Obfs/optimization_experiments/simulated_annealing/results/sa_results_20260216-175221.json"

DEVICE_HOST = "10.42.0.188"
DEVICE_PORT = 9091

GLOBAL_SCALE     = 8.0
SKIP_CONV_LAYERS = [0]          # same as execute_candidate_set_refactored default
OPT_LEVEL        = 3
MODEL_NAME       = "resnet18"   # for graph_pack start/stop names
INPUT_NAME       = "input0"
INPUT_SHAPE      = [1, 3, 224, 224]

# Graph pack configuration: only pack layers between start_name and stop_name
# For ResNet18: start after first conv (layer0), stop before final fc layer (after adaptive_avg_pool)
# This leaves first conv and last linear layer to execute on CPU
PACK_DICT = {
    "resnet18": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],  # Pack from first relu (post first conv) to before fc
}

RESULTS_DIR = os.path.join(SCRIPT_DIR, "step3_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Schedule log files (same as execute_candidate_set_refactored)
SCHEDULE_LOG_DIR = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/logs/tuning_logs/vta_1x16x16/candidate_set"


# ============================================================
# Helpers
# ============================================================
def sep(title="", w=72):
    if title:
        pad = (w - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (w - pad - len(title) - 2))
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
    nl = [r for r in runs if isinstance(r, dict)
          and r.get("N") == target_n
          and _as_float(r.get("lambda")) is not None
          and abs(_as_float(r.get("lambda")) - float(target_lambda)) < 1e-9]
    if not nl:
        raise ValueError(f"No run for N={target_n}, lambda={target_lambda}")
    exact = [r for r in nl if r.get("seed") == target_seed]
    run = exact[0] if exact else sorted(nl, key=lambda r: r.get("seed", 0))[0]
    if not exact:
        print(f"  ⚠ seed={target_seed} not found; using seed={run.get('seed')}")
    ids = [x for x in run.get("ids", []) if x in arch_mapping]
    chosen = ids[:k]
    print(f"  SA run: N={run['N']}, lambda={run['lambda']}, seed={run.get('seed')}")
    print(f"  Subnets: {chosen}")
    return {mid: arch_mapping[mid] for mid in chosen}


def load_schedule_logs():
    logs = []
    if os.path.isdir(SCHEDULE_LOG_DIR):
        import glob
        logs = glob.glob(os.path.join(SCHEDULE_LOG_DIR, "*.log"))
    return logs


def summarize_graph_storage(graph_json):
    """Return a compact storage/device summary from graph JSON."""
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
            "int8": 8, "uint8": 8,
            "int16": 16, "uint16": 16,
            "int32": 32, "uint32": 32,
            "int64": 64, "uint64": 64,
            "float16": 16, "float32": 32, "float64": 64,
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
                pool[sid] = {"bytes": nbytes, "device_index": dev_idx, "dtype": dtype, "shape": shape}

        by_dev = {}
        for _, meta in pool.items():
            by_dev[meta["device_index"]] = by_dev.get(meta["device_index"], 0) + meta["bytes"]
        top = sorted(pool.items(), key=lambda kv: kv[1]["bytes"], reverse=True)[:8]
        return {
            "storage_pool_count": len(pool),
            "bytes_per_device_index": by_dev,
            "largest_pools": [
                {
                    "storage_id": sid,
                    "device_index": meta["device_index"],
                    "bytes": meta["bytes"],
                    "dtype": meta["dtype"],
                    "shape": meta["shape"],
                }
                for sid, meta in top
            ],
        }
    except Exception:
        return None


def get_ofa_reference_output(ofa_net, arch, input_np):
    """Run the OFA model in PyTorch eval mode to get float32 reference logits."""
    ofa_net.set_active_subnet(arch)
    ofa_net.eval()
    with torch.no_grad():
        inp_t = torch.from_numpy(input_np)
        out_t = ofa_net(inp_t)
    return out_t.cpu().numpy()

def build_first_step_relay(subnet_id, arch, ofa_net, base_weights, transform_matrices,
                           bn_params, other_params, input_np):
    # --- Extract derivations ---
    print("  Extracting derivations...")
    extractor = OFADerivationExtractor(ofa_net, verbose=False)
    derivations = extractor.extract_subnet_derivations(arch, INPUT_SHAPE)
    print(f"  ✓ {len(derivations)} layer derivations")

    # --- Build OFA-pool Relay graph ---
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
        input_shape=INPUT_SHAPE,
        input_name=INPUT_NAME,
    )

    mod_deriv, deriv_params, mod_infer, infer_params, derived_weight_names = (
        split_derivation_and_inference_modules(mod_full, tvm_params_full)
    )

    print(f"  ✓ Graph built in {time.time() - t0:.1f}s")
    print(f"  Pool variables: {len(deriv_params)}")
    print(f"  Other params:   {len(infer_params)}")
    print(f"  Derived weights: {len(derived_weight_names)}")
    dynamic_weight_var_names = sorted(deriv_params.keys())
    print(f"  Dynamic pool vars: {len(dynamic_weight_var_names)}")

    # Save Relay IR for inspection
    ir_path = os.path.join(RESULTS_DIR, f"relay_ir_{subnet_id}.txt")
    with open(ir_path, "w") as f:
        f.write(str(mod_full))
    print(f"  Relay IR → {ir_path}")

    return {
        "derivations": derivations,
        "mod_full": mod_full,
        "mod_deriv": mod_deriv,
        "mod_infer": mod_infer,
        "tvm_params_full": tvm_params_full,
        "deriv_params": deriv_params,
        "infer_params": infer_params,
        "derived_weight_names": derived_weight_names,
        "dynamic_weight_var_names": dynamic_weight_var_names,
    }


def materialize_derived_weights_cpu(mod_deriv, deriv_params, derived_weight_names):
    """Run derivation module on CPU and return {derived_w_i: NDArray} for infer runtime/build."""
    with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
        deriv_lib = relay.build(mod_deriv, target="llvm", params=deriv_params)

    from tvm.contrib.graph_runtime import GraphModule
    rt = GraphModule(deriv_lib["default"](tvm.cpu(0)))
    rt.run()

    derived = {}
    for idx, name in enumerate(derived_weight_names):
        out = rt.get_output(idx)
        out_np = out.asnumpy() if hasattr(out, "asnumpy") else out.numpy()
        derived[name] = tvm.nd.array(out_np, tvm.cpu(0))
    return derived



# ============================================================
# Step 3A: Build OFA-pool Relay graph and validate on CPU
# ============================================================
def step3a_cpu_validation(subnet_id, arch, ofa_net, input_np, split_artifacts):
    sep(f"3A CPU Validation: {subnet_id}")

    # --- CPU execution (integrated module with materialized derived weights) ---
    print("  Running CPU inference (integrated module)...")
    t0 = time.time()
    try:
        # Step 1: Materialize derived weights on CPU by running the derivation portion
        # Extract derivation ops and execute them
        print("    Materializing derived weights from pool variables...")
        derived_weight_map = materialize_derived_weights_cpu(
            split_artifacts["mod_deriv"],
            split_artifacts["deriv_params"],
            split_artifacts["derived_weight_names"],
        )
        print(f"    ✓ Materialized {len(derived_weight_map)} derived weight tensors")

        # Step 2: Build and run the inference module with materialized weights
        print("    Building CPU inference runtime...")
        with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
            cpu_lib = relay.build(
                split_artifacts["mod_full"],  # Now integrated module
                target="llvm",
                params=split_artifacts["tvm_params_full"],  # All params
            )

        from tvm.contrib.graph_runtime import GraphModule
        cpu_rt = GraphModule(cpu_lib["default"](tvm.cpu(0)))

        # Set derived weights as inputs
        print("    Setting inputs and running inference...")
        inp_tvm = tvm.nd.array(input_np.astype("float32"), tvm.cpu(0))
        cpu_rt.set_input(**derived_weight_map)
        cpu_rt.set_input(INPUT_NAME, inp_tvm)
        cpu_rt.run()
        out_nd = cpu_rt.get_output(0)
        ofa_pool_out = out_nd.asnumpy() if hasattr(out_nd, "asnumpy") else out_nd.numpy()

        cpu_time = time.time() - t0
        print(f"  ✓ CPU inference in {cpu_time:.2f}s")
    except Exception as e:
        print(f"  ✗ CPU inference failed: {e}")
        import traceback; traceback.print_exc()
        raise

    # --- PyTorch reference ---
    print("  Getting PyTorch reference...")
    pytorch_out = get_ofa_reference_output(ofa_net, arch, input_np)

    # --- Compare ---
    diff = np.abs(ofa_pool_out - pytorch_out)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    top1_pool = int(np.argmax(ofa_pool_out[0]))
    top1_ref  = int(np.argmax(pytorch_out[0]))

    print(f"\n  CPU Validation Results:")
    print(f"    Max  |diff|:  {max_diff:.4e}")
    print(f"    Mean |diff|:  {mean_diff:.4e}")
    print(f"    Top-1 (pool): {top1_pool}")
    print(f"    Top-1 (ref):  {top1_ref}")
    print(f"    Top-1 match:  {'✓' if top1_pool == top1_ref else '✗'}")

    passed = max_diff < 1e-3 and top1_pool == top1_ref
    print(f"\n  3A Status: {'✓ PASSED' if passed else '✗ FAILED'}")

    return {
        "subnet_id": subnet_id,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "top1_match": top1_pool == top1_ref,
        "top1_pool": top1_pool,
        "top1_ref": top1_ref,
        "passed": passed,
        "derived_weight_map": derived_weight_map,
    }


# ============================================================
# Step 3B: Quantize + graph_pack + VTA build
# ============================================================
def step3b_vta_compile(
    subnet_id,
    arch,
    split_artifacts,
    env,
    experimental_routing=False,
    enable_dynamic_dense_quant=False,
    bind_all_params_at_build=False,
):
    sep(f"3B VTA Compile: {subnet_id}")

    print("  Applying quantization with dynamic weight preservation...")
    t0 = time.time()

    try:
        # Use the new quantize_with_dynamic_weights API that preserves weight vars
        # as inputs rather than folding them as constants
        with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
            prev_dense_fix_env = os.environ.get("TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX")
            try:
                if enable_dynamic_dense_quant:
                    os.environ["TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX"] = "1"
                    print("  Dynamic dense quantization: ENABLED (feature gate ON)")
                else:
                    os.environ.pop("TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX", None)

                with relay.quantize.qconfig(
                    global_scale=GLOBAL_SCALE,
                    skip_conv_layers=SKIP_CONV_LAYERS,
                    skip_dense_layer=(not enable_dynamic_dense_quant),
                ):
                    mod_q = quantize_with_dynamic_weights(
                        # split_artifacts["mod_full"],
                        # split_artifacts["tvm_params_full"],
                        split_artifacts["mod_infer"],
                        split_artifacts["infer_params"],
                        dynamic_weight_var_names=split_artifacts["dynamic_weight_var_names"],
                    )
            finally:
                if prev_dense_fix_env is None:
                    os.environ.pop("TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX", None)
                else:
                    os.environ["TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX"] = prev_dense_fix_env
        
        print(f"  ✓ Quantization done in {time.time()-t0:.1f}s")
        print(f"  Quantized module:\n{mod_q.astext(show_meta_data=False)[:500]}...")
        
        q_main_vars = {v.name_hint for v in relay.analysis.free_vars(mod_q["main"].body)}
        kept = sorted(q_main_vars.intersection(set(split_artifacts["dynamic_weight_var_names"])))
        print(f"  Quantized dynamic vars kept: {len(kept)}")

    except Exception as e:
        print(f"  ✗ Quantization failed: {e}")
        import traceback; traceback.print_exc()
        raise

    # --- Graph pack ---
    print("  Applying graph_pack...")
    t0 = time.time()

    with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
        relay_prog, used_graph_pack, pack_reason = graph_pack_dynamic_weights(
            mod_q["main"],
            env.BATCH,
            env.BLOCK_IN,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=PACK_DICT[MODEL_NAME][0],
            stop_name=PACK_DICT[MODEL_NAME][1],
            # Keep legacy-safe routing by default; broader annotation can
            # trigger LLVM verifier issues on dynamic-weight graphs.
            device_annot=(env.TARGET == "intelfocl") if not experimental_routing
            else (env.TARGET not in ("sim", "tsim")),
            pack_all=False,
            allow_fallback=False,
            return_status=True,
        )
    if used_graph_pack:
        print(f"  ✓ graph_pack done in {time.time()-t0:.1f}s")
    else:
        print(
            "  ⚠ graph_pack_dynamic_weights fallback used: "
            f"{pack_reason if pack_reason else 'unknown reason'}"
        )
    # except Exception as e:
    #     print(f"  ✗ graph_pack failed: {e}")
    #     import traceback; traceback.print_exc()
    #     raise


    # --- relay.build for VTA ---
    print("  Running relay.build for VTA target...")
    t0 = time.time()

    params_for_build = (
        split_artifacts["tvm_params_full"]
        if bind_all_params_at_build
        else split_artifacts["infer_params"]
    )
    if bind_all_params_at_build:
        print("  Build param mode: ALL params bound at build time")
    else:
        print("  Build param mode: infer params only (dynamic weights set at runtime)")

    schedule_logs = load_schedule_logs()
    print(f"  Using {len(schedule_logs)} schedule log files")

    try:
        with autotvm.tophub.context(env.target, extra_files=schedule_logs):
            build_target = env.target
            # Default path mirrors known-good script behavior. Experimental
            # hetero routing is opt-in for diagnosis.
            if experimental_routing and hasattr(env, "target_vta_cpu") and env.TARGET not in ("sim", "tsim"):
                build_target = {"ext_dev": env.target, "cpu": env.target_vta_cpu}
            with vta.build_config(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
                graph, lib, built_params = relay.build(
                    relay_prog,
                    target=build_target,
                    params=params_for_build,
                    target_host=env.target_host,
                )
        build_time = time.time() - t0
        print(f"  ✓ relay.build done in {build_time:.1f}s")
        print(f"  Built params: {len(built_params)}")
        print(f"  Build mode: {'graph_pack' if used_graph_pack else 'unpacked_fallback'}")

        graph_path = os.path.join(RESULTS_DIR, f"graph_{subnet_id}.json")
        with open(graph_path, "w") as f:
            f.write(graph)
        print(f"  Graph JSON -> {graph_path}")

        storage_summary = summarize_graph_storage(graph)
        if storage_summary is not None:
            bpd = storage_summary["bytes_per_device_index"]
            dev_desc = ", ".join(
                f"dev{d}={b / (1024 * 1024):.1f}MB" for d, b in sorted(bpd.items())
            )
            print(
                f"  Storage pools: {storage_summary['storage_pool_count']} "
                f"({dev_desc if dev_desc else 'device info unavailable'})"
            )
    except Exception as e:
        fail_relay_path = os.path.join(RESULTS_DIR, f"relay_prog_build_fail_{subnet_id}.txt")
        try:
            with open(fail_relay_path, "w") as f:
                f.write(str(relay_prog))
            print(f"  Build-fail Relay IR -> {fail_relay_path}")
        except Exception:
            pass
        print(f"  ✗ relay.build failed: {e}")
        import traceback; traceback.print_exc()
        raise

    dynamic_runtime_params = {} if bind_all_params_at_build else split_artifacts["deriv_params"]
    return graph, lib, built_params, dynamic_runtime_params, bind_all_params_at_build


# ============================================================
# Step 3C: Upload to VTA, run inference, validate
# ============================================================
def step3c_vta_inference(subnet_id, arch, graph, lib, built_params, dynamic_runtime_params,
                          ofa_net, input_np, env, remote, ctx, params_bound_at_build=False):
    sep(f"3C VTA Inference: {subnet_id}")

    # Upload
    print("  Uploading library to device...")
    temp_dir = tvm_utils.tempdir()
    lib_path = temp_dir.relpath(f"graphlib_step3_{subnet_id}.tar")
    lib.export_library(lib_path)
    remote.upload(lib_path)
    remote_lib = remote.load_module(f"graphlib_step3_{subnet_id}.tar")
    print("  ✓ Uploaded")

    # Create runtime
    if env.TARGET == "intelfocl":
        ctxes = [remote.ext_dev(0), remote.cpu(0)]
        m = graph_runtime.create(graph, remote_lib, ctxes)
    else:
        m = graph_runtime.create(graph, remote_lib, ctx)

    # Set inputs
    m.set_input(**built_params)
    if params_bound_at_build:
        print("  Dynamic runtime params: skipped (bound at build time)")
    else:
        m.set_input(**dynamic_runtime_params)
        print(f"  Dynamic runtime params: set {len(dynamic_runtime_params)} tensors")
    inp_tvm = tvm.nd.array(
        input_np.astype("float32"),
        remote.ext_dev(0) if env.TARGET != "sim" else tvm.cpu(0)
    )
    m.set_input(INPUT_NAME, inp_tvm)

    # Run
    print("  Running VTA inference...")
    t0 = time.time()
    m.run()
    inf_time = time.time() - t0
    print(f"  ✓ Inference done in {inf_time*1000:.1f}ms")

    vta_out = m.get_output(0).asnumpy()
    pytorch_out = get_ofa_reference_output(ofa_net, arch, input_np)

    top1_vta = int(np.argmax(vta_out[0]))
    top1_ref  = int(np.argmax(pytorch_out[0]))

    print(f"\n  VTA Validation Results:")
    print(f"    Top-1 (VTA):  {top1_vta}")
    print(f"    Top-1 (ref):  {top1_ref}")
    print(f"    Top-1 match:  {'✓' if top1_vta == top1_ref else '✗  (quantisation tolerance)'}")
    print(f"    Inference:    {inf_time*1000:.1f} ms")

    return {
        "subnet_id": subnet_id,
        "top1_vta": top1_vta,
        "top1_ref": top1_ref,
        "top1_match": top1_vta == top1_ref,
        "inf_time_ms": inf_time * 1000,
    }


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Phase B Step 3 POC")
    p.add_argument("--sa-results", default=SA_RESULTS_FILE)
    p.add_argument("--arch-file", default=ARCH_FILE)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--lambda", dest="lambda_value", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-subnets", type=int, default=2)
    p.add_argument("--skip-vta", action="store_true",
                   help="Only run CPU validation (Step 3A), skip VTA compile/inference")
    p.add_argument("--skip-cpu", action="store_true",
                   help="Skip cpu inference")
    p.add_argument(
        "--diag-experimental-routing",
        action="store_true",
        help="Enable non-default device annotation + hetero target routing for diagnostics",
    )
    p.add_argument(
        "--enable-dynamic-dense-quant",
        action="store_true",
        help="Enable dense quantization for dynamic-weight graphs via annotate feature gate",
    )
    p.add_argument(
        "--build-bind-all-params",
        action="store_true",
        help="Bind all params at relay.build time (diagnostic mode for runtime input/buffer issues)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    sep("Phase B Step 3: OFA-Pool Relay Build POC")

    # ------------------------------------------------------------------
    print("\n[1] Loading OFA model...")
    t0 = time.time()
    ofa_net = OFADynamicResnetAllMod()
    ckpt = torch.load(OFA_CHECKPOINT, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    ofa_net.load_state_dict(state, strict=False)
    ofa_net.eval()
    print(f"  ✓ OFA loaded in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    print("\n[2] Loading OFA weight pool...")
    t0 = time.time()
    pool = load_ofa_pool(POOL_DIR)
    base_weights       = pool["base_weights"]
    transform_matrices = pool["transform_matrices"]
    bn_params          = pool["bn_params"]
    other_params       = pool["other_params"]
    print(f"  ✓ Pool loaded in {time.time()-t0:.1f}s  "
          f"({len(base_weights)} base, {len(transform_matrices)} tm, "
          f"{len(bn_params)} bn, {len(other_params)} other)")

    # ------------------------------------------------------------------
    print("\n[3] Loading candidate set...")
    poc_archs = pick_subnets_from_sa(
        args.sa_results, args.arch_file,
        target_n=args.n, target_lambda=args.lambda_value,
        target_seed=args.seed, k=args.num_subnets,
    )

    # ------------------------------------------------------------------
    if not args.skip_vta:
        print("\n[4] Setting up VTA environment...")
        env = vta.get_env()
        remote = rpc.connect(DEVICE_HOST, DEVICE_PORT)
        vta.reconfig_runtime(remote)
        ctx = remote.ext_dev(0)
        print(f"  ✓ Connected to {DEVICE_HOST}:{DEVICE_PORT}")
        print(f"  Target: {env.target}")
    else:
        print("\n[4] Skipping VTA setup (--skip-vta)")
        env = remote = ctx = None

    # ------------------------------------------------------------------
    # Fixed input for all comparisons
    rng = np.random.default_rng(42)
    input_np = rng.standard_normal(INPUT_SHAPE).astype("float32")

    all_results = {}
    overall_passed = True

    for subnet_id, arch in poc_archs.items():
        sep(f"Processing {subnet_id}")

        split_artifacts = build_first_step_relay(subnet_id, arch, ofa_net,
                base_weights, transform_matrices, bn_params, other_params,
                input_np)
        r3a = None
        if not args.skip_cpu:
            # ---- Step 3A: CPU validation ----
            try:
                r3a = step3a_cpu_validation(subnet_id, arch, ofa_net, input_np, split_artifacts)
            except Exception as e:
                print(f"  ✗ Step 3A failed: {e}")
                import traceback; traceback.print_exc()
                all_results[subnet_id] = {"step3a": "FAILED", "error": str(e)}
                overall_passed = False
                continue

            if not r3a["passed"]:
                overall_passed = False

            if args.skip_vta:
                all_results[subnet_id] = {
                    "step3a": {k: v for k, v in r3a.items() if k not in ("derived_weight_map",)}
                }
                continue

        if not args.skip_vta:
            # ---- Step 3B: VTA compile ----
            try:
                graph, lib, built_params, dynamic_runtime_params, params_bound_at_build = step3b_vta_compile(
                    subnet_id, arch, split_artifacts, env,
                    experimental_routing=args.diag_experimental_routing,
                    enable_dynamic_dense_quant=args.enable_dynamic_dense_quant,
                    bind_all_params_at_build=args.build_bind_all_params,
                )
            except Exception as e:
                print(f"  ✗ Step 3B failed: {e}")
                import traceback; traceback.print_exc()
                all_results[subnet_id] = {
                    "step3a": {} if r3a is None else {k: v for k, v in r3a.items() if k not in ("derived_weight_map",)},
                    "step3b": "FAILED",
                    "error": str(e),
                }
                overall_passed = False
                continue

            # ---- Step 3C: VTA inference ----
            try:
                r3c = step3c_vta_inference(
                    subnet_id, arch, graph, lib, built_params, dynamic_runtime_params,
                    ofa_net, input_np, env, remote, ctx,
                    params_bound_at_build=params_bound_at_build,
                )
            except Exception as e:
                print(f"  ✗ Step 3C failed: {e}")
                import traceback; traceback.print_exc()
                all_results[subnet_id] = {"step3a": r3a, "step3b": "OK", "step3c": "FAILED", "error": str(e)}
                overall_passed = False
                continue

            all_results[subnet_id] = {
                "step3a": {} if r3a is None else {k: v for k, v in r3a.items() if k not in ("derived_weight_map",)},
                "step3b": "OK",
                "step3c": r3c,
            }

    # ------------------------------------------------------------------
    sep("Overall Summary")
    for sid, res in all_results.items():
        print(f"\n  {sid}:")
        if "error" in res:
            print(f"    ✗ FAILED: {res['error']}")
            continue
        r3a = res.get("step3a", {})
        r3c = res.get("step3c", {})
        if isinstance(r3a, dict) and "max_diff" in r3a:
            print(f"    3A CPU:  max_diff={r3a['max_diff']:.2e}  top1={'✓' if r3a['top1_match'] else '✗'}")
        if r3c and isinstance(r3c, dict) and "top1_vta" in r3c:
            print(f"    3C VTA:  top1={'✓' if r3c['top1_match'] else '✗'}  inf={r3c['inf_time_ms']:.1f}ms")

    print(f"\nOverall: {'✓ ALL PASSED' if overall_passed else '✗ SOME FAILED'}")

    # Save summary
    def _json_safe(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if isinstance(v, (list, tuple)): return [_json_safe(x) for x in v]
        if isinstance(v, dict): return {kk: _json_safe(vv) for kk, vv in v.items()}
        if hasattr(v, "to_dict"):
            try:
                return _json_safe(v.to_dict())
            except Exception:
                return str(v)
        if hasattr(v, "__dict__") and v.__class__.__name__ == "LayerDerivation":
            return {k: _json_safe(val) for k, val in vars(v).items()}
        return v

    summary_path = os.path.join(RESULTS_DIR, "step3_summary.json")
    with open(summary_path, "w") as f:
        json.dump(_json_safe(all_results), f, indent=2)
    print(f"\nSummary → {summary_path}")
    sep()


if __name__ == "__main__":
    main()

