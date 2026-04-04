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
import json
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
VTA_ROOT = os.path.join(TVM_ROOT, "vta", "python")
EXTERNAL_REPO_ROOT = "/home/srchand/Desktop/research/OFA_Obfs"

for p in [EXTERNAL_REPO_ROOT, TVM_PYTHON, VTA_ROOT, SRI_SCRIPTS_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import tvm
from tvm import autotvm, relay, rpc
from tvm.contrib import graph_runtime, utils as tvm_utils
import vta
from vta.top import graph_pack_dynamic_weights

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
SKIP_CONV_LAYERS = [0]
OPT_LEVEL = 3
MODEL_NAME = "resnet18"
INPUT_NAME = "input0"
INPUT_SHAPE = [1, 3, 224, 224]

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
    enable_hetero_routing=True,
    enable_graph_pack=True,
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

    # Dynamic mode keeps pool vars unbound at build.
    params_for_build = tvm_params_full if static_debug_mode else {
        k: v for k, v in tvm_params_full.items() if k not in set(pool_var_names)
    }
    print("  Build mode: %s" % ("STATIC DEBUG" if static_debug_mode else "DYNAMIC (pool vars runtime)"))

    schedule_logs = load_schedule_logs()
    print("  Using %d schedule logs" % len(schedule_logs))

    if not enable_hetero_routing:
        raise RuntimeError("Hetero routing is required for this POC run")

    if enable_graph_pack:
        print("  Applying graph_pack...")
        with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
            relay_prog, used_graph_pack, pack_reason = graph_pack_dynamic_weights(
                mod_q["main"],
                env.BATCH,
                env.BLOCK_IN,
                env.BLOCK_OUT,
                env.WGT_WIDTH,
                start_name=PACK_DICT[MODEL_NAME][0],
                stop_name=PACK_DICT[MODEL_NAME][1],
                pack_all=False,
                allow_fallback=True,
                return_status=True,
                device_annot=True,
                annot_start_name="nn.conv2d",
                annot_end_name="annotation.stop_fusion",
            )
        if used_graph_pack:
            print("  graph_pack: OK")
        else:
            print("  graph_pack fallback: %s" % (pack_reason if pack_reason else "unknown"))
    else:
        relay_prog = mod_q["main"]
        print("  graph_pack: SKIPPED")

    build_target = {
        "cpu": getattr(env, "target_vta_cpu", "llvm"),
        "ext_dev": env.target,
    }
    print("  Build target: hetero(cpu+ext_dev)")

    # Hetero codegen on arm target can hit LLVM verifier errors when vectorized stack
    # buffers are passed into VTABufferCPUPtr. Disable TIR vectorization in this path.
    hetero_pass_ctx = tvm.transform.PassContext(
        opt_level=OPT_LEVEL,
        disabled_pass={"AlterOpLayout"},
        config={"tir.disable_vectorize": True},
    )

    t0 = time.time()
    with autotvm.tophub.context(env.target, extra_files=schedule_logs):
        with hetero_pass_ctx:
            with vta.build_config(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
                graph, lib, built_params = relay.build(
                    relay_prog,
                    target=build_target,
                    params=params_for_build,
                    target_host=env.target_host,
                )
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

    pool_params_np = {
        name: (tvm_params_full[name].asnumpy() if hasattr(tvm_params_full[name], "asnumpy") else np.array(tvm_params_full[name]))
        for name in pool_var_names
    }

    return {
        "graph": graph,
        "lib": lib,
        "built_params": built_params,
        "pool_params_np": pool_params_np,
        "pool_var_names": pool_var_names,
        "static_debug_mode": static_debug_mode,
        "use_hetero_routing": True,
        "compile_strategy": "hetero(cpu+ext_dev)",
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
    if compile_artifacts.get("use_hetero_routing", False):
        m = graph_runtime.create(
            compile_artifacts["graph"],
            remote_lib,
            [remote.ext_dev(0), remote.cpu(0)],
        )
        print("  Runtime contexts: [ext_dev(0), cpu(0)]")
    else:
        m = graph_runtime.create(compile_artifacts["graph"], remote_lib, ctx)
        print("  Runtime contexts: [ext_dev(0)]")
    m.set_input(**compile_artifacts["built_params"])

    if not compile_artifacts["static_debug_mode"]:
        copied = 0
        copied_bytes = 0
        for name, arr in sorted(compile_artifacts["pool_params_np"].items(), key=lambda kv: kv[1].nbytes, reverse=True):
            slot = m.get_input(name)
            if slot is None:
                raise RuntimeError("Missing runtime input for pool var: %s" % name)
            slot.copyfrom(arr)
            copied += 1
            copied_bytes += int(arr.nbytes)
        print("  Uploaded pool vars: %d tensors (%.1f MB)" % (copied, copied_bytes / (1024.0 * 1024.0)))
    else:
        print("  Static debug mode: pool vars are build-bound")

    input_slot = m.get_input(INPUT_NAME)
    if input_slot is None:
        raise RuntimeError("Missing graph input: %s" % INPUT_NAME)
    # Let graph runtime own the destination context (cpu/ext_dev) for hetero safety.
    input_slot.copyfrom(input_np.astype("float32"))

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
        "--no-hetero-routing",
        action="store_true",
        help="Disable hetero routing (not supported in this script; kept for compatibility)",
    )
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
    if args.no_hetero_routing:
        raise RuntimeError("This script now requires hetero routing; remove --no-hetero-routing")

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

    rng = np.random.default_rng(42)
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
                enable_hetero_routing=True,
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

