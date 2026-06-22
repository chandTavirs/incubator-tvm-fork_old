"""
Lazy K=1 Cached Subnet Switching Runtime
=========================================

All K compiled module libs are pre-built and pre-uploaded to the VTA device at
startup. At any point in time, only ONE subnet's weights (and module pool) are
resident in device CMA. A switch = evict current module, load_module + create
new GraphModule (CMA alloc), set_input host params (host->device transfer). A
repeat inference on the same subnet = 0 ms overhead.

This contrasts with the K-resident mode (step4_multigraph_switching.py):
  K-resident : 0 ms switch, ~53*K MB CMA   (e.g. K=25 -> 1325 MB)
  lazy K=1   : ~300 ms switch, ~53 MB CMA  (regardless of K)

Build path: same folded-constants pipeline as step4 (numpy-materialize OFA
transforms, standard quantize+FoldConstant, graph_pack, relay.build). Each
module is entirely self-contained; only the input image is needed at run time.

Usage:
    VTA_RPC_HOST=10.42.0.188 VTA_RPC_PORT=9091 \\
        python step5_lazy_k_cached.py --num-subnets 2 --switch-iters 20
"""
from __future__ import absolute_import, print_function

import argparse
import collections
import gc
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
from tvm import rpc
from tvm.contrib import graph_runtime, utils as tvm_utils
import vta

from ofa_base_models import OFADynamicResnetAllMod
from ofa_weight_pool_extractor import load_ofa_pool

from step3_merged_mod_deriv_poc import (
    OFA_CHECKPOINT, ARCH_FILE, POOL_DIR, SA_RESULTS_FILE,
    DEVICE_HOST, DEVICE_PORT,
    INPUT_NAME, INPUT_SHAPE,
    pick_subnets_from_sa, get_ofa_reference_output, load_schedule_logs,
)
from step4_multigraph_switching import build_folded_subnet


# ============================================================
# Lazy K=1 Runtime
# ============================================================

class LazyK1Runtime:
    """
    Lazy K=1 cached switching runtime.

    All K module libs are pre-uploaded to the device at register() time.
    Only ONE GraphModule (one CMA pool) is live at any time.

    Switch (idx != live_idx):
        evict current module  -> free CMA pool
        remote.load_module    -> load .so from device disk
        graph_runtime.create  -> allocate new CMA pool
        m.set_input(**params) -> host-to-device weight transfer (RPC)
        m.set_input(image)    -> host-to-device image transfer
        m.run()               -> inference on VTA

    Same subnet (idx == live_idx):
        m.set_input(image)    -> host-to-device image transfer
        m.run()               -> inference on VTA  (0 ms switch overhead)
    """

    def __init__(self, remote, ctx):
        self.remote = remote
        self.ctx = ctx
        self.subnets = []       # list of dicts: subnet_id, graph, remote_fname, host_params, param_mb
        self.live_m = None      # current live GraphModule (or None)
        self.live_idx = None    # index into self.subnets (or None)
        self._tmpdir = tvm_utils.tempdir()  # keep alive; holds exported .tar files on host
        self.switch_count = 0

    def register(self, built):
        """Export lib to local temp, upload to device. Params stay on host as numpy."""
        sid = built["subnet_id"]
        fname = "lazy_%s.tar" % sid
        lib_path = self._tmpdir.relpath(fname)
        built["lib"].export_library(lib_path)

        t0 = time.time()
        self.remote.upload(lib_path)
        upload_ms = (time.time() - t0) * 1000.0

        # Keep params as HOST numpy arrays -- never allocate device CMA here.
        host_params = {k: v.asnumpy() for k, v in built["params"].items()}
        param_mb = sum(a.nbytes for a in host_params.values()) / (1024.0 ** 2)

        self.subnets.append({
            "subnet_id": sid,
            "graph": built["graph"],
            "remote_fname": fname,
            "host_params": host_params,
            "param_mb": param_mb,
        })
        print("  Registered %-30s  %.1f MB params  upload %.0f ms"
              % (sid, param_mb, upload_ms), flush=True)

    def _evict(self):
        """Delete the live module, freeing its CMA pool."""
        if self.live_m is not None:
            del self.live_m
            self.live_m = None
            self.live_idx = None
            gc.collect()    # force GC so RPC sends free message before next alloc

    def run(self, idx, input_np):
        """Run inference on subnet idx. Returns (output_np, timing_dict, subnet_id)."""
        assert 0 <= idx < len(self.subnets)
        sub = self.subnets[idx]
        inp = tvm.nd.array(input_np.astype("float32"), self.ctx)
        create_ms = 0.0
        bind_ms = 0.0

        if idx != self.live_idx:
            self.switch_count += 1
            self._evict()

            # load_module: device loads .so from its local temp (already uploaded)
            # graph_runtime.create: allocates storage pool on device CMA
            t0 = time.time()
            rlib = self.remote.load_module(sub["remote_fname"])
            m = graph_runtime.create(sub["graph"], rlib, self.ctx)
            create_ms = (time.time() - t0) * 1000.0

            # set_input: host-to-device weight transfer over RPC (one RPC call per param)
            t0 = time.time()
            m.set_input(**sub["host_params"])
            bind_ms = (time.time() - t0) * 1000.0

            self.live_m = m
            self.live_idx = idx

        self.live_m.set_input(INPUT_NAME, inp)
        t0 = time.time()
        self.live_m.run()
        self.ctx.sync()
        run_ms = (time.time() - t0) * 1000.0

        out = self.live_m.get_output(0).asnumpy()
        return out, {
            "create": create_ms,
            "bind": bind_ms,
            "switch": create_ms + bind_ms,
            "run": run_ms,
        }, sub["subnet_id"]

    def teardown(self):
        """Release all device-resident objects for clean RPC exit."""
        self._evict()
        self.subnets = []
        gc.collect()

    def __len__(self):
        return len(self.subnets)


# ============================================================
# CLI + main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Lazy K=1 cached subnet switching runtime")
    p.add_argument("--sa-results", default=SA_RESULTS_FILE)
    p.add_argument("--arch-file", default=ARCH_FILE)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--lambda", dest="lambda_value", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-subnets", type=int, default=2,
                   help="K: how many subnets (POC default = 2)")
    p.add_argument("--switch-iters", type=int, default=20,
                   help="random switching inferences")
    p.add_argument("--warm-iters", type=int, default=5,
                   help="same-subnet inferences to measure steady-state latency")
    p.add_argument("--rng-seed", type=int, default=42)
    return p.parse_args()


def sep(t=""):
    print(("=" * 8) + " " + t + " " + ("=" * max(0, 60 - len(t))), flush=True)


def main():
    args = parse_args()
    sep("Lazy K=1 Cached Switching  (K=%d)" % args.num_subnets)

    # ------------------------------------------------------------------
    print("[1] Load OFA model + pool ...", flush=True)
    ofa_net = OFADynamicResnetAllMod()
    ck = torch.load(OFA_CHECKPOINT, map_location="cpu")
    ofa_net.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    ofa_net.eval()
    pool = load_ofa_pool(POOL_DIR)

    # ------------------------------------------------------------------
    print("[2] Select K=%d subnets ..." % args.num_subnets, flush=True)
    archs = pick_subnets_from_sa(
        args.sa_results, args.arch_file,
        target_n=args.n, target_lambda=args.lambda_value,
        target_seed=args.seed, k=args.num_subnets,
    )

    # ------------------------------------------------------------------
    print("[3] Connect VTA RPC ...", flush=True)
    env = vta.get_env()
    remote = rpc.connect(DEVICE_HOST, DEVICE_PORT)
    vta.reconfig_runtime(remote)
    ctx = remote.ext_dev(0)
    schedule_logs = load_schedule_logs()

    # ------------------------------------------------------------------
    print("[4] Build + register %d subnet modules ..." % len(archs), flush=True)
    print("    (folded-weight build: ~30 s/subnet)", flush=True)
    rt = LazyK1Runtime(remote, ctx)
    built_list = []
    for sid, arch in archs.items():
        built = build_folded_subnet(sid, arch, ofa_net, pool, env, schedule_logs)
        rt.register(built)
        built_list.append(built)

    total_param_mb = sum(s["param_mb"] for s in rt.subnets)
    live_param_mb = rt.subnets[0]["param_mb"]
    print("", flush=True)
    print("  All %d libs uploaded." % len(rt), flush=True)
    print("  Total params (if all resident): %.1f MB" % total_param_mb, flush=True)
    print("  Lazy device footprint (1 live): ~%.1f MB" % live_param_mb, flush=True)

    rng = np.random.default_rng(99)
    input_np = rng.standard_normal(INPUT_SHAPE).astype("float32")

    # ------------------------------------------------------------------
    print("", flush=True)
    print("[5] Correctness: one inference per subnet ...", flush=True)
    for i, b in enumerate(built_list):
        out, tm, sid = rt.run(i, input_np)
        ref = get_ofa_reference_output(ofa_net, b["arch"], input_np)
        t1v = int(np.argmax(out[0]))
        t1r = int(np.argmax(ref[0]))
        match = "OK" if t1v == t1r else "MISMATCH"
        print("  %-30s vta=%4d ref=%4d %-9s  "
              "switch=%5.0f ms (create=%5.0f bind=%5.0f)  run=%5.0f ms"
              % (sid, t1v, t1r, match, tm["switch"], tm["create"], tm["bind"], tm["run"]),
              flush=True)

    # ------------------------------------------------------------------
    # Warm run: first iter pays switch cost (evict K-1, load subnet 0).
    # Subsequent iters have 0 ms switch overhead.
    print("", flush=True)
    print("[6] Warm run on subnet 0 (%d iters) ..." % args.warm_iters, flush=True)
    print("    iter 0 = first-access (switch cost); iters 1+ = pure inference", flush=True)
    rt._evict()
    warm_switch_ms = None
    pure_run_times = []
    for i in range(args.warm_iters):
        out, tm, sid = rt.run(0, input_np)
        tag = "[switch]" if tm["switch"] > 0 else "[warm]  "
        print("  iter %d %s  switch=%5.0f ms  run=%5.0f ms"
              "  (create=%5.0f  bind=%5.0f)"
              % (i, tag, tm["switch"], tm["run"], tm["create"], tm["bind"]), flush=True)
        if i == 0:
            warm_switch_ms = tm["switch"]
        else:
            pure_run_times.append(tm["run"])
    if warm_switch_ms is not None:
        print("  First-access switch cost: %.0f ms" % warm_switch_ms, flush=True)
    if pure_run_times:
        arr = np.array(pure_run_times)
        print("  Steady-state (no switch):  mean=%.0f ms  min=%.0f  max=%.0f"
              % (arr.mean(), arr.min(), arr.max()), flush=True)

    # ------------------------------------------------------------------
    # Switch sweep: cycle through all K subnets, always evicting first.
    print("", flush=True)
    print("[7] Switch sweep: all %d subnets in order ..." % len(rt), flush=True)
    rt._evict()
    sweep_switch = []
    sweep_run = []
    for i in range(len(rt)):
        _, tm, sid = rt.run(i, input_np)
        sweep_switch.append(tm["switch"])
        sweep_run.append(tm["run"])
        print("  -> %-30s  switch=%5.0f ms (create=%5.0f bind=%5.0f)  run=%5.0f ms"
              % (sid, tm["switch"], tm["create"], tm["bind"], tm["run"]), flush=True)
    sw_arr = np.array(sweep_switch)
    print("  Switch overhead: mean=%.0f ms  min=%.0f  max=%.0f"
          % (sw_arr.mean(), sw_arr.min(), sw_arr.max()), flush=True)

    # ------------------------------------------------------------------
    print("", flush=True)
    print("[8] Random switching: %d iters ..." % args.switch_iters, flush=True)
    random.seed(args.rng_seed)
    agg = {"create": [], "bind": [], "switch": [], "run": []}
    same_run = []
    seq = []
    prev_idx = rt.live_idx
    for _ in range(args.switch_iters):
        k = random.randrange(len(rt))
        out, tm, sid = rt.run(k, input_np)
        for key in ("create", "bind", "switch", "run"):
            agg[key].append(tm[key])
        seq.append(k)
        if k == prev_idx:
            same_run.append(tm["run"])
        prev_idx = k

    print("  Switch sequence (first 20): %s" % seq[:20], flush=True)
    print("  Selection histogram: %s"
          % dict(sorted(collections.Counter(seq).items())), flush=True)
    print("", flush=True)
    for key in ("create", "bind", "switch", "run"):
        a = np.array(agg[key])
        print("  %-8s  mean=%6.0f ms  min=%6.0f  max=%6.0f"
              % (key, a.mean(), a.min(), a.max()), flush=True)
    if same_run:
        ss = np.array(same_run)
        print("  Same-subnet (no switch):  mean=%.0f ms  min=%.0f  max=%.0f  n=%d"
              % (ss.mean(), ss.min(), ss.max(), len(ss)), flush=True)

    # ------------------------------------------------------------------
    sep("Summary")
    print("  K=%d subnets  |  1 live module on device at a time" % len(rt), flush=True)
    print("  Device CMA (lazy-1): ~%.1f MB  vs  K-resident: ~%.1f MB"
          % (live_param_mb, total_param_mb), flush=True)
    print("  Switch overhead:  mean=%.0f ms  (create %.0f ms + bind %.0f ms)"
          % (np.mean(agg["switch"]), np.mean(agg["create"]), np.mean(agg["bind"])),
          flush=True)
    print("  Inference (run):  mean=%.0f ms" % np.mean(agg["run"]), flush=True)
    sep("Done")

    rt.teardown()
    del remote
    gc.collect()
    print("Released device + RPC, exiting.", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
