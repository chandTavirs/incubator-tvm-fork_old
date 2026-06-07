"""
Multi-Graph Subnet Switching Runtime (side-channel defense)
===========================================================

Builds K OFA subnet graphs with the derived transform weights FOLDED as
constants at compile time (no GMTF, no runtime derivation), loads all K
compiled modules onto the VTA device, and switches between them at runtime by
random selection. The switching itself is the research goal: an attacker
observing side-channel leakage cannot uniquely fingerprint a single architecture
because any inference could have been produced by any of the K subnets.

Build path per subnet (the "weight folding" the user asked for):
  1. build_relay_with_ofa_pool_vars  -> merged graph with pool vars
  2. quantize_with_dynamic_weights(..., dynamic_weight_var_names=[])
        empty list => nothing protected => all pool vars bound as constants,
        FoldConstant collapses each strided_slice→reshape→nn.dense(T)→reshape
        transform into a single constant derived conv kernel, then quantizes.
        (This is the inverse of the dynamic path, which protects pool vars.)
  3. graph_pack + relay.build  -> self-contained VTA module (only input = image)

Memory: each module's folded int8 weights stay resident in device DRAM
(~53 MB/subnet; K=25 ≈ 1.3 GB, fits the 2 GB ZCU104).

Usage:
  VTA_RPC_HOST=10.42.0.188 VTA_RPC_PORT=9091 \
      python step4_multigraph_switching.py --num-subnets 25 --switch-iters 50
"""
from __future__ import absolute_import, print_function

import argparse
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
from quantize_dynamic_weights import quantize_with_dynamic_weights

# Reuse config + helpers from the merged POC
from step3_merged_mod_deriv_poc import (
    OFA_CHECKPOINT, ARCH_FILE, POOL_DIR, SA_RESULTS_FILE,
    DEVICE_HOST, DEVICE_PORT, GLOBAL_SCALE, SKIP_CONV_LAYERS, OPT_LEVEL,
    MODEL_NAME, INPUT_NAME, INPUT_SHAPE, PACK_DICT,
    pick_subnets_from_sa, get_ofa_reference_output, load_schedule_logs,
    build_merged_artifacts,
)
from step3_gemm_mat_trf_integration import (
    step4_materialize_transforms, step4b_lower_gemm_mat_trf,
)


# ============================================================
# Build one folded, self-contained subnet module
# ============================================================
def build_folded_subnet(subnet_id, arch, ofa_net, pool, env, schedule_logs, verbose=True):
    """Return dict(graph, lib, params, subnet_id) for a fully-folded VTA module."""
    t0 = time.time()
    artifacts = build_merged_artifacts(
        subnet_id, arch, ofa_net,
        pool["base_weights"], pool["transform_matrices"], pool["bn_params"], pool["other_params"],
    )
    mod_full = artifacts["mod_full"]
    tvm_params_full = artifacts["tvm_params_full"]

    # 1) Materialize the OFA transforms via numpy (fast, graph-aware) and substitute
    #    relay.const for each — so the quantizer doesn't re-evaluate huge matmuls.
    transform_var_names = {"pool_" + k for k in pool["transform_matrices"].keys()}
    materialized, _ = step4_materialize_transforms(mod_full, pool, transform_var_names, verbose=False)
    mod_full = step4b_lower_gemm_mat_trf(
        mod_full, transform_var_names, precomputed=materialized, vta_intrinsic=False,
    )

    # 2) Standard quantize with all pool params bound => everything folds to constants
    #    (derived conv kernels + base weights). Faster than the dynamic-preserving wrapper.
    with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
        with relay.quantize.qconfig(global_scale=GLOBAL_SCALE,
                                    skip_conv_layers=SKIP_CONV_LAYERS,
                                    skip_dense_layer=True):
            mod_q = relay.quantize.quantize(mod_full, params=tvm_params_full)

    pack_entry = PACK_DICT.get(MODEL_NAME, ["nn.max_pool2d", "nn.adaptive_avg_pool2d"])
    with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
        relay_prog = graph_pack(
            mod_q["main"], env.BATCH, env.BLOCK_IN, env.BLOCK_OUT, env.WGT_WIDTH,
            start_name=pack_entry[0], stop_name=pack_entry[1],
            device_annot=(env.TARGET == "intelfocl"),
        )

    with autotvm.tophub.context(env.target, extra_files=schedule_logs):
        with vta.build_config(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
            graph, lib, params = relay.build(
                relay_prog, target=env.target, target_host=env.target_host, params={},
            )
    if verbose:
        wbytes = sum(int(v.asnumpy().nbytes) for v in params.values())
        print("  [build] %s  folded params=%d (%.1f MB)  %.1fs"
              % (subnet_id, len(params), wbytes / (1024.0 ** 2), time.time() - t0))
    return {"subnet_id": subnet_id, "arch": arch, "graph": graph, "lib": lib, "params": params}


# ============================================================
# Multi-graph runtime
# ============================================================
class MultiGraphRuntime:
    """Switch between K subnet modules on one VTA device.

    mode="resident":  pre-create all K GraphModules (fast switch, K x full pool CMA).
    mode="transient": keep K weight-sets resident as device NDArrays; per switch
                      create the target module, zero-copy-bind the resident weights
                      (no re-transfer), run, destroy (one module's pool live at a time).
    """

    def __init__(self, remote, ctx, mode="resident"):
        self.remote = remote
        self.ctx = ctx
        self.mode = mode
        self.entries = []       # resident: (sid, GraphModule); transient: (sid, graph, rlib, resident_w)
        self.weights_mb = 0.0   # total resident weight bytes
        self._cur = None        # transient: (idx, live GraphModule)

    def load(self, built):
        temp = tvm_utils.tempdir()
        fname = "switchlib_%s.tar" % built["subnet_id"]
        lib_path = temp.relpath(fname)
        built["lib"].export_library(lib_path)
        self.remote.upload(lib_path)
        rlib = self.remote.load_module(fname)
        wbytes = sum(int(v.asnumpy().nbytes) for v in built["params"].values())
        self.weights_mb += wbytes / (1024.0 ** 2)
        if self.mode == "resident":
            m = graph_runtime.create(built["graph"], rlib, self.ctx)
            m.set_input(**built["params"])      # weights resident in this module's pool
            self.entries.append((built["subnet_id"], m))
        else:
            # Upload each folded weight ONCE as a resident ext_dev NDArray on the device.
            # Use self.ctx (ext_dev, same as the module's internal storage) so that when
            # m.set_input(name, arr) is called, RPC sends just the handle (no re-upload);
            # the server does a local ext_dev→ext_dev CopyFrom (no network data transfer).
            # remote.cpu(0) does NOT work: the RPC session mask is baked into its device_type,
            # causing "Can not copy across different ctx types" when CopyFrom runs on the server.
            resident_w = {}
            for name, v in built["params"].items():
                np_arr = v.asnumpy()
                dev_arr = tvm.nd.array(np_arr, self.ctx)
                resident_w[name] = dev_arr
            self.entries.append((built["subnet_id"], built["graph"], rlib, resident_w))

    def run(self, idx, input_np):
        inp = tvm.nd.array(input_np.astype("float32"), self.ctx)
        if self.mode == "resident":
            sid, m = self.entries[idx]
            m.set_input(INPUT_NAME, inp)
            t0 = time.time(); m.run(); self.ctx.sync()
            run_ms = (time.time() - t0) * 1000.0
            return m.get_output(0).asnumpy(), {"run": run_ms, "create": 0, "bind": 0, "free": 0}, sid

        # transient
        sid, graph, rlib, resident_w = self.entries[idx]
        t0 = time.time()
        if self._cur is not None:           # tear down previous live module -> free its pool
            del self._cur
            self._cur = None
            import gc as _gc; _gc.collect()  # force RPC free before next alloc; del alone may defer it
        m = graph_runtime.create(graph, rlib, self.ctx)
        create_ms = (time.time() - t0) * 1000.0
        t0 = time.time()
        for name, arr in resident_w.items():
            m.set_input(name, arr)          # handle-only RPC call → fast ARM-local memcpy, no re-upload
        m.set_input(INPUT_NAME, inp)
        bind_ms = (time.time() - t0) * 1000.0
        t0 = time.time(); m.run(); self.ctx.sync()
        run_ms = (time.time() - t0) * 1000.0
        out = m.get_output(0).asnumpy()
        self._cur = m                       # keep live until next switch (then freed)
        return out, {"run": run_ms, "create": create_ms, "bind": bind_ms, "free": 0}, sid

    def __len__(self):
        return len(self.entries)

    def teardown(self):
        """Release all device-resident objects so the process can exit without
        hanging on graph_runtime teardown over RPC (which holds the single-client VTA)."""
        self._cur = None
        self.entries = []
        import gc
        gc.collect()


# ============================================================
# Main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="Multi-graph OFA subnet switching runtime")
    p.add_argument("--sa-results", default=SA_RESULTS_FILE)
    p.add_argument("--arch-file", default=ARCH_FILE)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--lambda", dest="lambda_value", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-subnets", type=int, default=25, help="K: how many subnets to load")
    p.add_argument("--switch-iters", type=int, default=50, help="random switching inferences to run")
    p.add_argument("--rng-seed", type=int, default=1234)
    p.add_argument("--mode", choices=["resident", "transient"], default="resident",
                   help="resident=pre-create all K modules; transient=resident weights + one live module")
    return p.parse_args()


def sep(t=""):
    print("=" * 8 + " " + t + " " + "=" * max(0, 60 - len(t)))


def main():
    args = parse_args()
    sep("Multi-Graph Subnet Switching (folded weights, no GMTF)")

    print("[1] Load OFA model + pool ...")
    ofa_net = OFADynamicResnetAllMod()
    ck = torch.load(OFA_CHECKPOINT, map_location="cpu")
    ofa_net.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    ofa_net.eval()
    pool = load_ofa_pool(POOL_DIR)

    print("[2] Select K=%d subnets ..." % args.num_subnets)
    archs = pick_subnets_from_sa(args.sa_results, args.arch_file,
                                 target_n=args.n, target_lambda=args.lambda_value,
                                 target_seed=args.seed, k=args.num_subnets)

    print("[3] VTA RPC + build %d folded subnet modules ..." % len(archs))
    env = vta.get_env()
    remote = rpc.connect(DEVICE_HOST, DEVICE_PORT)
    vta.reconfig_runtime(remote)
    ctx = remote.ext_dev(0)
    schedule_logs = load_schedule_logs()

    print("  mode = %s" % args.mode)
    rt = MultiGraphRuntime(remote, ctx, mode=args.mode)
    built_list = []
    for sid, arch in archs.items():
        built = build_folded_subnet(sid, arch, ofa_net, pool, env, schedule_logs)
        rt.load(built)
        built_list.append(built)
    print("  Loaded %d subnets, resident weights = %.1f MB" % (len(rt), rt.weights_mb))

    print("[4] Correctness: one inference per subnet vs PyTorch top-1 ...")
    rng = np.random.default_rng(99)
    input_np = rng.standard_normal(INPUT_SHAPE).astype("float32")
    for i, b in enumerate(built_list):
        out, tm, sid = rt.run(i, input_np)
        ref = get_ofa_reference_output(ofa_net, b["arch"], input_np)
        t1v, t1r = int(np.argmax(out[0])), int(np.argmax(ref[0]))
        print("  %-30s vta=%d ref=%d %s  total=%.0f ms (create=%.0f bind=%.0f run=%.0f)" %
              (sid, t1v, t1r, "OK" if t1v == t1r else "MISMATCH",
               tm["create"] + tm["bind"] + tm["run"], tm["create"], tm["bind"], tm["run"]))

    print("[5] Random switching: %d inferences ..." % args.switch_iters)
    random.seed(args.rng_seed)
    agg = {"create": [], "bind": [], "run": [], "total": []}
    seq = []
    for _ in range(args.switch_iters):
        k = random.randrange(len(rt))
        out, tm, sid = rt.run(k, input_np)
        for key in ("create", "bind", "run"):
            agg[key].append(tm[key])
        agg["total"].append(tm["create"] + tm["bind"] + tm["run"])
        seq.append(k)
    import collections
    print("  switch sequence (first 30): %s" % seq[:30])
    print("  selection histogram: %s" % dict(sorted(collections.Counter(seq).items())))
    for key in ("create", "bind", "run", "total"):
        a = np.array(agg[key])
        print("  %-7s mean=%.0f ms  min=%.0f  max=%.0f" % (key, a.mean(), a.min(), a.max()))
    print("  ==> switch overhead (create+bind) mean=%.0f ms; inference run mean=%.0f ms" %
          (np.mean(agg["create"]) + np.mean(agg["bind"]), np.mean(agg["run"])))
    sep("Done")

    # Explicit teardown so the process releases the device and exits cleanly
    # (graph_runtime teardown over RPC is sticky; without this the process can
    #  hang holding the single-client VTA RPC and deadlock the next run).
    rt.teardown()
    del remote
    import gc; gc.collect()
    print("Released device + RPC, exiting.", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
