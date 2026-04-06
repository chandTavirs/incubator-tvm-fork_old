"""
Phase B Step 3: Pool Variable Folding POC
==========================================

Demonstrates the integrated module + post-quantization constant folding workflow:

  1. Build a single integrated Relay graph with OFA pool variables using
     build_relay_with_ofa_pool_vars() — weights are expressed as slices +
     optional dense transforms applied to shared pool variables.

  2. Quantize the entire graph with relay.quantize(), preserving pool vars
     as graph inputs (not folding them as constants during quantization).

  3. Apply relay.transform.FoldConstant() only to pool vars that are safe to
     materialize, while keeping the selected runtime pool path dynamic so
     kernel-transform dense ops remain in the graph.

  4. Graph pack + relay.build for VTA target.

  5. Upload compiled library to VTA device (foldable weights are constants;
     runtime pool vars are loaded once per session).

  6. Run inference with only the selected runtime pool variables as inputs.

Key improvements over split-module approach:
  ✓ Single integrated compiled artifact (no separate derivation module)
  ✓ Foldable pool variables are materialized into constants
  ✓ Selected runtime pool vars stay dynamic so transform ops are preserved
  ✓ First layer remains float32 via skip_conv_layers (minimal overhead)
  ✓ Deterministic, reproducible weights across runs

Usage
-----
  cd .../graph_switching_phase2/phase_b
  python step3_quantize_pool_vars_poc.py [--sa-results PATH] [--arch-file PATH] \\
    [--n 25] [--lambda 4] [--seed 0] [--skip-vta] [--debug-print-ir]
"""

from __future__ import absolute_import, print_function

import os
import sys
import json
import time
import argparse

import numpy as np
import torch

# ---- path setup ----
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
from ofa_relay_graph_builder import build_relay_with_ofa_pool_vars
from quantize_dynamic_weights import quantize_with_dynamic_weights

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
SKIP_CONV_LAYERS = [0]          # First layer stays float32
OPT_LEVEL        = 3
MODEL_NAME       = "resnet18"
INPUT_NAME       = "input0"
INPUT_SHAPE      = [1, 3, 224, 224]

PACK_DICT = {
    "resnet18": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
}

RESULTS_DIR = os.path.join(SCRIPT_DIR, "step3_quantize_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SCHEDULE_LOG_DIR = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/logs/tuning_logs/vta_1x16x16/candidate_set"


# ============================================================
# Helpers
# ============================================================
def sep(title="", w=72):
    """Print a separator line."""
    if title:
        pad = (w - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (w - pad - len(title) - 2))
    else:
        print("=" * w)


def load_arch_mapping(arch_file):
    """Load architecture mapping from JSON file."""
    with open(arch_file) as f:
        data = json.load(f)
    if isinstance(data, dict) and "architectures" in data:
        return {item["id"]: item["architecture"] for item in data["architectures"]}
    return data


def _as_float(v):
    """Safely convert to float."""
    try:
        return float(v)
    except Exception:
        return None


def pick_single_subnet_from_sa(sa_file, arch_file, target_n, target_lambda, target_seed):
    """Pick a single subnet from SA results."""
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
    if not ids:
        raise ValueError("No valid architectures found in SA run")
    
    subnet_id = ids[0]  # Pick first one
    print(f"  SA run: N={run['N']}, lambda={run['lambda']}, seed={run.get('seed')}")
    print(f"  Selected subnet: {subnet_id}")
    
    return subnet_id, arch_mapping[subnet_id]


def load_schedule_logs():
    """Load tuning schedule logs."""
    logs = []
    if os.path.isdir(SCHEDULE_LOG_DIR):
        import glob
        logs = glob.glob(os.path.join(SCHEDULE_LOG_DIR, "*.log"))
    return logs


def count_pool_vars_in_graph(mod):
    """Count pool variables in the graph (that will be folded)."""
    free_vars = relay.analysis.free_vars(mod["main"].body)
    pool_vars = [v for v in free_vars if v.name_hint.startswith("pool_")]
    return len(pool_vars), {v.name_hint for v in pool_vars}


def get_first_layer_runtime_pool_vars(derivations, all_pool_var_names):
    """Return pool var names that should remain runtime inputs (first conv only)."""
    runtime = set()
    for d in derivations:
        layer_path = str(getattr(d, "layer_path", ""))
        if not layer_path.startswith("first_layer.0"):
            continue

        base_key = getattr(d, "base_weight_key", None)
        if base_key:
            runtime.add(f"pool_{base_key}")

    runtime &= set(all_pool_var_names)

    # Fallback for older derivation metadata where only name pattern is available.
    if not runtime:
        runtime = {n for n in all_pool_var_names if n.startswith("pool_first_layer_0_")}

    return runtime


def _is_base_pool_var_name(var_name):
    return var_name.startswith("pool_") and var_name.endswith("_base_conv_weight")


def get_runtime_pool_vars_from_first_conv(mod_q, fallback_runtime_vars):
    """Infer runtime pool vars from the first pool-dependent conv2d weight path.

    Relay let-binding order is more reliable than generic visitor traversal here.
    We walk the main expression in execution order, find the first conv2d whose
    weight expression actually depends on pool vars, and prefer the smallest
    runtime set by keeping only base-weight pool vars when available.
    """

    def _iter_let_values(expr):
        cur = expr
        while isinstance(cur, relay.expr.Let):
            yield cur.value
            cur = cur.body
        yield cur

    selected_pool_vars = set()
    for value in _iter_let_values(mod_q["main"].body):
        if not isinstance(value, relay.Call):
            continue
        if not (isinstance(value.op, tvm.ir.Op) and value.op.name == "nn.conv2d"):
            continue

        weight_expr = value.args[1]
        pool_vars = {
            v.name_hint
            for v in relay.analysis.free_vars(weight_expr)
            if v.name_hint.startswith("pool_")
        }
        if pool_vars:
            selected_pool_vars = pool_vars
            break

    if not selected_pool_vars:
        return set(fallback_runtime_vars)

    base_vars = {v for v in selected_pool_vars if _is_base_pool_var_name(v)}
    return base_vars if base_vars else selected_pool_vars


def summarize_graph_storage(graph_json):
    """Extract storage pool information from compiled graph JSON."""
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


def summarize_graph_devices(graph_json):
    """Return node count per device_index from graph JSON."""
    try:
        g = json.loads(graph_json)
        devs = g.get("attrs", {}).get("device_index", [None, []])[1]
        out = {}
        for d in devs:
            d = int(d)
            out[d] = out.get(d, 0) + 1
        return out
    except Exception:
        return {}


def get_ofa_reference_output(ofa_net, arch, input_np):
    """Run OFA model in PyTorch eval mode to get reference logits."""
    ofa_net.set_active_subnet(arch)
    ofa_net.eval()
    with torch.no_grad():
        inp_t = torch.from_numpy(input_np)
        out_t = ofa_net(inp_t)
    return out_t.cpu().numpy()


def build_integrated_relay_graph(subnet_id, arch, ofa_net, base_weights, transform_matrices,
                                 bn_params, other_params, input_np, debug_print_ir=False):
    """Build integrated Relay graph with pool variables (no split)."""
    sep(f"Building integrated graph for {subnet_id}")

    # Extract derivations
    print("  Extracting derivations...")
    extractor = OFADerivationExtractor(ofa_net, verbose=False)
    derivations = extractor.extract_subnet_derivations(arch, INPUT_SHAPE)
    print(f"  ✓ {len(derivations)} layer derivations")

    # Build integrated Relay graph
    print("  Building integrated Relay graph...")
    t0 = time.time()
    mod, params = build_relay_with_ofa_pool_vars(
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
    build_time = time.time() - t0
    print(f"  ✓ Graph built in {build_time:.1f}s")

    # Count pool variables
    pool_var_count, pool_var_names = count_pool_vars_in_graph(mod)
    print(f"  Pool variables to fold: {pool_var_count}")
    print(f"  Other params: {len(params)}")

    if debug_print_ir:
        ir_path = os.path.join(RESULTS_DIR, f"relay_ir_pre_quant_{subnet_id}.txt")
        with open(ir_path, "w") as f:
            f.write(str(mod))
        print(f"  Pre-quant Relay IR → {ir_path}")

    return {
        "mod": mod,
        "params": params,
        "derivations": derivations,
        "pool_var_names": pool_var_names,
        "pool_var_count": pool_var_count,
    }


def quantize_and_fold_pool_vars(subnet_id, relay_artifacts, debug_print_ir=False):
    """
    Quantize graph and then fold pool variables into int8 constants.
    
    First layer (layer0) is skipped via skip_conv_layers=[0], so its
    pool variables remain float32.
    """
    sep(f"Quantizing and folding pool vars for {subnet_id}")

    mod = relay_artifacts["mod"]
    params = relay_artifacts["params"]
    derivations = relay_artifacts["derivations"]
    pool_var_names = relay_artifacts["pool_var_names"]
    expected_runtime_pool_var_names = get_first_layer_runtime_pool_vars(derivations, pool_var_names)
    if not expected_runtime_pool_var_names:
        raise RuntimeError("Could not identify first-layer runtime pool vars.")

    # Step 1: Quantize with qconfig
    print("  Quantizing graph...")
    t0 = time.time()
    
    with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
        with relay.quantize.qconfig(
            global_scale=GLOBAL_SCALE,
            skip_conv_layers=SKIP_CONV_LAYERS,
            skip_dense_layer=True,
        ):
            mod_q = quantize_with_dynamic_weights(
                mod,
                params,
                dynamic_weight_var_names=sorted(pool_var_names),
            )
    
    quant_time = time.time() - t0
    print(f"  ✓ Quantization done in {quant_time:.1f}s")

    # Check which pool vars remain after quantization
    free_vars_after_q = relay.analysis.free_vars(mod_q["main"].body)
    remaining_vars = {v.name_hint for v in free_vars_after_q if v.name_hint.startswith("pool_")}
    print(f"  Pool variables remaining after quantize: {len(remaining_vars)}")
    if not remaining_vars:
        raise RuntimeError(
            "No pool_* vars remain after quantize; dynamic derivation path was materialized too early."
        )

    runtime_pool_var_names = get_runtime_pool_vars_from_first_conv(
        mod_q,
        fallback_runtime_vars=expected_runtime_pool_var_names,
    )
    runtime_pool_var_names &= remaining_vars
    if not runtime_pool_var_names:
        raise RuntimeError(
            "First-conv runtime pool vars are missing after quantize; "
            "FoldConstant would erase the intended dynamic path."
        )

    foldable_pool_var_names = remaining_vars - runtime_pool_var_names
    print(f"  Runtime pool vars to keep dynamic (pool-dependent conv deps): {len(runtime_pool_var_names)}")
    print(f"  Pool vars to fold after quantize: {len(foldable_pool_var_names)}")

    if debug_print_ir:
        ir_path = os.path.join(RESULTS_DIR, f"relay_ir_post_quant_{subnet_id}.txt")
        with open(ir_path, "w") as f:
            f.write(str(mod_q))
        print(f"  Post-quant Relay IR → {ir_path}")

    # Step 2: Bind only foldable pool vars, then run FoldConstant.
    # Keeping runtime vars unbound preserves the dynamic transform path.
    print("  Binding foldable pool vars before FoldConstant...")
    t0 = time.time()
    from tvm.relay.quantize.quantize import _bind_params

    params_to_bind = {
        k: params[k]
        for k in foldable_pool_var_names
        if k in params
    }
    mod_q_main_bound = _bind_params(mod_q["main"], params_to_bind)
    mod_q_bound = tvm.IRModule.from_expr(mod_q_main_bound)
    bind_time = time.time() - t0
    print(f"  ✓ Bound {len(params_to_bind)} pool vars in {bind_time:.1f}s")

    print("  Applying FoldConstant to fold bound pool variables...")
    t0 = time.time()
    
    with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
        mod_folded = relay.transform.FoldConstant()(mod_q_bound)
    
    fold_time = time.time() - t0
    print(f"  ✓ FoldConstant done in {fold_time:.1f}s")

    # Check how many pool vars were folded
    free_vars_after_fold = relay.analysis.free_vars(mod_folded["main"].body)
    remaining_vars_after_fold = {v.name_hint for v in free_vars_after_fold if v.name_hint.startswith("pool_")}
    folded_count = len(foldable_pool_var_names - remaining_vars_after_fold)
    missing_runtime_after_fold = runtime_pool_var_names - remaining_vars_after_fold
    if missing_runtime_after_fold:
        raise RuntimeError(
            "FoldConstant removed required runtime pool vars: "
            f"{sorted(missing_runtime_after_fold)}"
        )
    unexpected_remaining = remaining_vars_after_fold - runtime_pool_var_names
    if unexpected_remaining:
        raise RuntimeError(
            "FoldConstant left unexpected pool vars dynamic: "
            f"{sorted(unexpected_remaining)}"
        )

    params_for_build = {k: v for k, v in params.items() if k not in pool_var_names}
    runtime_pool_params = {k: params[k] for k in remaining_vars_after_fold if k in params}
    
    print(f"  Pool variables folded: {folded_count}/{len(foldable_pool_var_names)} (non-first-layer)")
    if remaining_vars_after_fold:
        print(f"  Pool variables still dynamic (runtime inputs): {sorted(remaining_vars_after_fold)[:5]}...")
    print(f"  Params bound at build: {len(params_for_build)}")
    print(f"  Pool params left for runtime: {len(runtime_pool_params)}")

    if debug_print_ir:
        ir_path = os.path.join(RESULTS_DIR, f"relay_ir_post_fold_{subnet_id}.txt")
        with open(ir_path, "w") as f:
            f.write(str(mod_folded))
        print(f"  Post-fold Relay IR → {ir_path}")

    return {
        "mod_folded": mod_folded,
        "params_for_build": params_for_build,
        "runtime_pool_params": runtime_pool_params,
        "remaining_pool_vars": remaining_vars_after_fold,
        "folded_count": folded_count,
    }


def graph_pack_and_compile(subnet_id, fold_artifacts, env, experimental_routing=False):
    """Apply graph_pack and compile for VTA target."""
    sep(f"Graph pack and compile for {subnet_id}")

    mod_folded = fold_artifacts["mod_folded"]
    params_for_build = fold_artifacts["params_for_build"]

    # Graph pack
    print("  Applying graph_pack...")
    t0 = time.time()
    
    with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
        relay_prog, used_graph_pack, pack_reason = graph_pack_dynamic_weights(
            mod_folded["main"],
            env.BATCH,
            env.BLOCK_IN,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=PACK_DICT[MODEL_NAME][0],
            stop_name=PACK_DICT[MODEL_NAME][1],
            device_annot=experimental_routing,
            pack_all=False,
            allow_fallback=False,
            return_status=True,
        )
    
    pack_time = time.time() - t0
    if used_graph_pack:
        print(f"  ✓ graph_pack done in {pack_time:.1f}s")
    else:
        print(f"  ⚠ graph_pack fallback: {pack_reason if pack_reason else 'unknown'}")

    # Compile for VTA
    print("  Compiling for VTA target...")
    t0 = time.time()
    
    schedule_logs = load_schedule_logs()
    print(f"  Using {len(schedule_logs)} schedule log files")

    try:
        with autotvm.tophub.context(env.target, extra_files=schedule_logs):
            with vta.build_config(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
                graph, lib, built_params = relay.build(
                    relay_prog,
                    target=env.target,
                    params=params_for_build,
                    target_host=env.target_host,
                )
        
        compile_time = time.time() - t0
        print(f"  ✓ Compilation done in {compile_time:.1f}s")
        print(f"  Built params: {len(built_params)}")
        
        # Save graph for inspection
        graph_path = os.path.join(RESULTS_DIR, f"compiled_graph_{subnet_id}.json")
        with open(graph_path, "w") as f:
            f.write(graph)
        print(f"  Compiled graph JSON → {graph_path}")
        
        # Analyze storage
        dev_counts = summarize_graph_devices(graph)
        if dev_counts:
            dev_desc = ", ".join(f"dev{k}={v}" for k, v in sorted(dev_counts.items()))
            print(f"  Device node distribution: {dev_desc}")
        
        storage_summary = summarize_graph_storage(graph)
        if storage_summary:
            bpd = storage_summary["bytes_per_device_index"]
            dev_desc = ", ".join(
                f"dev{d}={b / (1024 * 1024):.1f}MB" for d, b in sorted(bpd.items())
            )
            print(f"  Storage pools: {storage_summary['storage_pool_count']} ({dev_desc})")
        
        return {
            "graph": graph,
            "lib": lib,
            "built_params": built_params,
            "used_graph_pack": used_graph_pack,
        }
    
    except Exception as e:
        fail_relay_path = os.path.join(RESULTS_DIR, f"relay_prog_build_fail_{subnet_id}.txt")
        try:
            with open(fail_relay_path, "w") as f:
                f.write(str(relay_prog))
            print(f"  Build-fail Relay IR → {fail_relay_path}")
        except Exception:
            pass
        print(f"  ✗ Compilation failed: {e}")
        raise


def run_vta_inference(subnet_id, compile_artifacts, fold_artifacts, ofa_net, arch, input_np,
                      env, remote, ctx):
    """Upload library and run inference on VTA device."""
    sep(f"VTA Inference: {subnet_id}")

    # Upload library
    print("  Uploading compiled library to device...")
    temp_dir = tvm_utils.tempdir()
    lib_path = temp_dir.relpath(f"graphlib_poolvar_poc_{subnet_id}.tar")
    compile_artifacts["lib"].export_library(lib_path)
    remote.upload(lib_path)
    remote_lib = remote.load_module(f"graphlib_poolvar_poc_{subnet_id}.tar")
    print("  ✓ Library uploaded")

    # Create runtime
    m = graph_runtime.create(compile_artifacts["graph"], remote_lib, ctx)
    m.set_input(**compile_artifacts["built_params"])

    # Provide runtime pool vars that intentionally remain dynamic after FoldConstant.
    runtime_pool_params = fold_artifacts.get("runtime_pool_params", {})
    for name in sorted(runtime_pool_params):
        arr = runtime_pool_params[name]
        if isinstance(arr, tvm.nd.NDArray):
            arr = arr.asnumpy()
        m.set_input(name, tvm.nd.array(np.asarray(arr), ctx))

    # Set input
    inp_tvm = tvm.nd.array(
        input_np.astype("float32"),
        remote.ext_dev(0) if env.TARGET != "sim" else tvm.cpu(0)
    )
    m.set_input(INPUT_NAME, inp_tvm)

    # Run inference
    print("  Running VTA inference...")
    t0 = time.time()
    m.run()
    inf_time = time.time() - t0
    print(f"  ✓ Inference done in {inf_time*1000:.1f}ms")

    # Get output
    vta_out = m.get_output(0).asnumpy()
    pytorch_out = get_ofa_reference_output(ofa_net, arch, input_np)

    top1_vta = int(np.argmax(vta_out[0]))
    top1_ref = int(np.argmax(pytorch_out[0]))

    print(f"\n  VTA Inference Results:")
    print(f"    Top-1 (VTA):   {top1_vta}")
    print(f"    Top-1 (PyTorch): {top1_ref}")
    print(f"    Top-1 match:   {'✓ YES' if top1_vta == top1_ref else '✗ NO (quantization tolerance)'}")
    print(f"    Inference time: {inf_time*1000:.1f} ms")

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
    p = argparse.ArgumentParser(
        description="Phase B Step 3: Pool Variable Folding POC"
    )
    p.add_argument("--sa-results", default=SA_RESULTS_FILE,
                   help="Path to SA results JSON")
    p.add_argument("--arch-file", default=ARCH_FILE,
                   help="Path to architectures JSON")
    p.add_argument("--n", type=int, default=25,
                   help="OFA N parameter")
    p.add_argument("--lambda", dest="lambda_value", type=float, default=4.0,
                   help="OFA lambda parameter")
    p.add_argument("--seed", type=int, default=0,
                   help="SA random seed")
    p.add_argument("--skip-vta", action="store_true",
                   help="Skip VTA compilation and inference")
    p.add_argument("--debug-print-ir", action="store_true",
                   help="Save Relay IR at each stage for inspection")
    return p.parse_args()


def main():
    args = parse_args()
    sep("Phase B Step 3: Pool Variable Folding POC")
    print("Integrated module + post-quantization FoldConstant approach")

    # ====== 1. Load OFA model ======
    print("\n[1] Loading OFA model...")
    t0 = time.time()
    ofa_net = OFADynamicResnetAllMod()
    ckpt = torch.load(OFA_CHECKPOINT, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    ofa_net.load_state_dict(state, strict=False)
    ofa_net.eval()
    print(f"  ✓ OFA loaded in {time.time()-t0:.1f}s")

    # ====== 2. Load OFA weight pool ======
    print("\n[2] Loading OFA weight pool...")
    t0 = time.time()
    pool = load_ofa_pool(POOL_DIR)
    base_weights       = pool["base_weights"]
    transform_matrices = pool["transform_matrices"]
    bn_params          = pool["bn_params"]
    other_params       = pool["other_params"]
    print(f"  ✓ Pool loaded in {time.time()-t0:.1f}s")
    print(f"    {len(base_weights)} base weights")
    print(f"    {len(transform_matrices)} transform matrices")
    print(f"    {len(bn_params)} BN params")
    print(f"    {len(other_params)} other params")

    # ====== 3. Pick subnet from SA results ======
    print("\n[3] Selecting subnet from SA results...")
    subnet_id, arch = pick_single_subnet_from_sa(
        args.sa_results, args.arch_file,
        target_n=args.n,
        target_lambda=args.lambda_value,
        target_seed=args.seed,
    )

    # ====== 4. Fixed input for all stages ======
    rng = np.random.default_rng(42)
    input_np = rng.standard_normal(INPUT_SHAPE).astype("float32")
    print(f"\n[4] Using fixed random input: shape={INPUT_SHAPE}")

    # ====== 5. Build integrated Relay graph ======
    relay_artifacts = build_integrated_relay_graph(
        subnet_id, arch, ofa_net,
        base_weights, transform_matrices, bn_params, other_params,
        input_np,
        debug_print_ir=args.debug_print_ir,
    )

    # ====== 6. Quantize and fold pool vars ======
    fold_artifacts = quantize_and_fold_pool_vars(
        subnet_id, relay_artifacts,
        debug_print_ir=args.debug_print_ir,
    )

    # ====== 7. Graph pack and compile ======
    if not args.skip_vta:
        print("\n[5] Setting up VTA environment...")
        env = vta.get_env()
        remote = rpc.connect(DEVICE_HOST, DEVICE_PORT)
        vta.reconfig_runtime(remote)
        ctx = remote.ext_dev(0)
        print(f"  ✓ Connected to {DEVICE_HOST}:{DEVICE_PORT}")
        print(f"  Target: {env.target}")

        compile_artifacts = graph_pack_and_compile(subnet_id, fold_artifacts, env)

        # ====== 8. Run inference ======
        result = run_vta_inference(
            subnet_id, compile_artifacts, fold_artifacts,
            ofa_net, arch, input_np,
            env, remote, ctx,
        )

        # ====== 9. Save summary ======
        sep("Summary")
        summary = {
            "subnet_id": subnet_id,
            "pool_vars_folded": fold_artifacts["folded_count"],
            "pool_vars_remaining": len(fold_artifacts["remaining_pool_vars"]),
            "runtime_pool_inputs": sorted(fold_artifacts["runtime_pool_params"].keys()),
            "graph_pack_used": compile_artifacts["used_graph_pack"],
            "vta_inference": result,
        }

        summary_path = os.path.join(RESULTS_DIR, f"summary_{subnet_id}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary → {summary_path}")

        print(f"\n✓ POC COMPLETE: Pool vars folded successfully, VTA inference {'✓ PASSED' if result['top1_match'] else '✗ FAILED (quantization tolerance)'}")

    else:
        print("\n[5] Skipping VTA (--skip-vta)")
        sep("Summary")
        summary = {
            "subnet_id": subnet_id,
            "pool_vars_folded": fold_artifacts["folded_count"],
            "pool_vars_remaining": len(fold_artifacts["remaining_pool_vars"]),
            "runtime_pool_inputs": sorted(fold_artifacts["runtime_pool_params"].keys()),
            "pool_vars_remaining_names": sorted(fold_artifacts["remaining_pool_vars"]),
        }
        summary_path = os.path.join(RESULTS_DIR, f"summary_{subnet_id}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary → {summary_path}")
        print(f"\n✓ Quantization and folding complete (VTA skipped)")

    sep()


if __name__ == "__main__":
    main()

