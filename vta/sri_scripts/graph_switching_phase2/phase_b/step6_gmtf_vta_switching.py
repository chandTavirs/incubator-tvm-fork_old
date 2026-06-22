"""
GMTF-on-VTA Shared-Pool Switching Runtime
==========================================

Architecture
------------
All K subnets compile against the SAME OFA weight pool, sharing the base weights
as a single set of pre-resident ext_dev NDArrays on the VTA device.  Each subnet's
*transform matrices* (tiny, ~50 KB total) are the only per-subnet data sent over
the RPC link per switch.

  Shared base weights: ~16 MB, uploaded to VTA ext_dev ONCE at startup.
  Per-subnet transform mats: ~50 KB, sent on every switch (RPC transfer).
  Module CMA: one live GMTF+conv module per subnet (~66 MB).
  Total CMA: ~82 MB, independent of K.

Each inference runs the FULL GMTF+conv pipeline on VTA hardware (~3.3 s).
GMTF executes via the 'vta.gemm_mat_trf' composite intrinsic — the OFA kernel
transform (e.g. 7x7 -> 5x5) runs in hardware and writes derived weights directly
into VTA scratchpad/CMA without any host involvement.

Comparison table (final)
------------------------
  K-resident  (step4)  : switch 0 ms,   run 117 ms, CMA ~ 53*K MB
  Lazy K=1    (step5)  : switch ~18 s,  run 117 ms, CMA ~ 53 MB  (network bottleneck)
  GMTF shared (step6)  : switch ~320 ms, run ~3300 ms, CMA ~ 82 MB  (all derived on VTA)

The ~320 ms switch = create (~70 ms) + bind_base_local (~120 ms) + bind_transforms (~130 ms)
The ~3300 ms run = GMTF hardware derivation + conv2d inference

Usage
-----
  VTA_RPC_HOST=10.42.0.188 VTA_RPC_PORT=9091 \\
      python step6_gmtf_vta_switching.py --num-subnets 2 --switch-iters 20
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
    pick_subnets_from_sa, get_ofa_reference_output, load_schedule_logs,
    build_merged_artifacts,
)
from step3_gemm_mat_trf_integration import (
    step5_quantize_module,
    step4b_lower_gemm_mat_trf,
    step6_materialize_int8_pool,
)


# ============================================================
# Build one VTA-intrinsic subnet module (GMTF + conv on VTA)
# ============================================================

def build_gmtf_vta_subnet(subnet_id, arch, ofa_net, pool, env, schedule_logs, verbose=True):
    """Build the GMTF+conv VTA intrinsic module for one subnet.

    Pipeline:
      build_merged_artifacts
      -> step5_quantize (pool vars dynamic)
      -> step4b_lower_gmtf (VTA composite, NOT CPU substitution)
      -> step6_materialize_int8 (rewrite pool vars to int8 typed free vars)
      -> graph_pack + relay.build (pool vars remain as runtime inputs)

    Returns
    -------
    dict: subnet_id, arch, graph, lib, params (empty / non-pool),
          runtime_pool_params_np (dict[name -> np.ndarray int8])
    """
    t0 = time.time()

    artifacts = build_merged_artifacts(
        subnet_id, arch, ofa_net,
        pool["base_weights"], pool["transform_matrices"],
        pool["bn_params"], pool["other_params"],
    )
    mod_full = artifacts["mod_full"]
    tvm_params_full = artifacts["tvm_params_full"]

    # All 62 vars (pool_*, _bn_*, fc_weight, fc_bias) kept as free vars so each
    # subnet's BN/fc params can differ at runtime.
    all_dynamic_var_names = list(tvm_params_full.keys())

    # Only pool_* vars undergo int8 treatment in step6: BN/fc are float32 throughout.
    pool_only_var_names = [k for k in tvm_params_full.keys() if k.startswith("pool_")]
    transform_var_names = {"pool_" + k for k in pool["transform_matrices"].keys()}

    # Step 5: Quantize with ALL vars kept dynamic (free vars).
    #   enable_dynamic_dense_quant=True (skip_dense_layer=False): the quantizer wraps
    #   GMTF nn.dense args in int8 cast chains.  DenseTransformDetector in step4b will
    #   then find 0 ops (args are no longer plain Vars), so no composites are created.
    #   After step6 _PoolLadderStripper strips quant ladders, the GMTF ops become plain
    #   nn.dense(int8, int8, out_dtype="int32") with a square weight.  graphpack detects
    #   the square kernel and routes to vta.gmtf_dense_small/large — no composites needed.
    mod_q, _ = step5_quantize_module(
        mod_full, tvm_params_full, all_dynamic_var_names, enable_dynamic_dense_quant=True,
    )

    # Step 4b: no-op here (0 GMTF ops detected because quantize wrapped args in cast chains).
    #   Kept for structural parity with step3; composites are NOT the detection mechanism —
    #   graphpack handles GMTF via square kernel_shape on plain nn.dense.
    mod_q = step4b_lower_gemm_mat_trf(
        mod_q, transform_var_names, precomputed=None, vta_intrinsic=True,
    )

    # Step 6: Rewrite pool_* vars to int8 typed free vars; BN/fc stay float32.
    #   Passing all_dynamic_var_names here would crash: _materialize_int8_pool_constants
    #   would rewrite BN vars (_bn_*) to int8, breaking float32 BN arithmetic.
    mod_compile, runtime_pool_params_np = step6_materialize_int8_pool(
        mod_q, tvm_params_full, pool_only_var_names,
    )

    # Float32 per-subnet params (BN, fc): kept as free vars, set at runtime.
    non_pool_var_names = [k for k in all_dynamic_var_names if not k.startswith("pool_")]
    non_pool_params_np = {}
    for k in non_pool_var_names:
        v = tvm_params_full[k]
        non_pool_params_np[k] = v.asnumpy() if hasattr(v, "asnumpy") else np.asarray(v, dtype=np.float32)

    # Step 7: graph_pack + relay.build
    #   params_for_build is EMPTY: all pool vars are runtime inputs.
    pack_entry = PACK_DICT.get(MODEL_NAME, ["nn.max_pool2d", "nn.adaptive_avg_pool2d"])
    with tvm.transform.PassContext(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
        relay_prog = graph_pack(
            mod_compile["main"], env.BATCH, env.BLOCK_IN, env.BLOCK_OUT, env.WGT_WIDTH,
            start_name=pack_entry[0], stop_name=pack_entry[1],
            device_annot=(env.TARGET == "intelfocl"),
        )

    with autotvm.tophub.context(env.target, extra_files=schedule_logs):
        with vta.build_config(opt_level=OPT_LEVEL, disabled_pass={"AlterOpLayout"}):
            graph, lib, params = relay.build(
                relay_prog, target=env.target, target_host=env.target_host,
                params={},  # empty: all pool vars are runtime inputs
            )

    build_s = time.time() - t0
    pool_mb = sum(a.nbytes for a in runtime_pool_params_np.values()) / (1024.0 ** 2)
    nonpool_mb = sum(a.nbytes for a in non_pool_params_np.values()) / (1024.0 ** 2)
    if verbose:
        print("  [build-gmtf] %s  pool=%.1f MB  non-pool=%.1f MB  extra=%d  %.0fs"
              % (subnet_id, pool_mb, nonpool_mb, len(params), build_s), flush=True)

    return {
        "subnet_id": subnet_id,
        "arch": arch,
        "graph": graph,
        "lib": lib,
        "params": params,                        # non-pool constants (usually empty)
        "runtime_pool_params_np": runtime_pool_params_np,   # int8 pool_* vars
        "non_pool_params_np": non_pool_params_np,           # float32 BN + fc vars
    }


# ============================================================
# Runtime
# ============================================================

class GmtfVtaRuntime:
    """GMTF-on-VTA shared-pool switching runtime.

    Base weights are uploaded to ext_dev ONCE at startup and reused for all
    subnets via device-local set_input (~6 ms/tensor, no network re-transfer).
    Transform matrices (~50 KB per subnet) are the only data crossing the RPC
    link per switch.

    Each m.run() executes the FULL GMTF+conv pipeline on VTA hardware:
      - transform ops via the gemm_mat_trf intrinsic (FPGA)
      - conv2d + BN + pooling via the standard VTA conv schedule (FPGA)
    Total: ~3.3 s per inference (GMTF dominates).

    For same-subnet repeats the module is kept alive, so only
    set_input(image) + run() are needed (still ~3.3 s, no switch overhead).
    """

    def __init__(self, remote, ctx):
        self.remote = remote
        self.ctx = ctx
        self.subnets = []
        self.live_m = None
        self.live_idx = None
        self._tmpdir = tvm_utils.tempdir()

        # Shared base-weight ext_dev NDArrays (uploaded once, reused for all subnets)
        self._shared_base_ext = {}    # param_name -> tvm.nd.NDArray on ext_dev
        self._base_weight_names = set()
        self._base_mb = 0.0

    # ------------------------------------------------------------------
    def init_shared_base_weights(self, runtime_pool_params_np, base_weight_names):
        """Upload shared base weights to ext_dev ONCE.

        int8 base weight values are identical across all subnets (same pool,
        same fixed quantization scale), so we only need one copy in CMA.

        Parameters
        ----------
        runtime_pool_params_np : dict[str -> np.ndarray]
            Pool params from any one built subnet (base weights are the same).
        base_weight_names : set[str]
            Names of base weight pool vars (e.g. {'pool_base_weight_layer3', ...}).
        """
        self._base_weight_names = set(base_weight_names)
        t0 = time.time()
        total_bytes = 0
        for name in sorted(base_weight_names):
            arr = runtime_pool_params_np.get(name)
            if arr is None:
                continue
            ext_arr = tvm.nd.array(arr, self.ctx)   # host -> ext_dev (RPC, done once)
            self._shared_base_ext[name] = ext_arr
            total_bytes += int(arr.nbytes)
        self._base_mb = total_bytes / (1024.0 ** 2)
        elapsed_ms = (time.time() - t0) * 1000.0
        print("  Shared base weights: %d tensors  %.1f MB  upload=%.0f ms"
              % (len(self._shared_base_ext), self._base_mb, elapsed_ms), flush=True)

    # ------------------------------------------------------------------
    def register(self, built):
        """Export + upload compiled lib for one subnet. Non-base params on host."""
        sid = built["subnet_id"]
        fname = "gmtf_%s.tar" % sid
        lib_path = self._tmpdir.relpath(fname)
        built["lib"].export_library(lib_path)

        t0 = time.time()
        self.remote.upload(lib_path)
        up_ms = (time.time() - t0) * 1000.0

        # Non-base pool params: int8 transform matrices.  Kept as host numpy.
        non_base_params = {
            k: v for k, v in built["runtime_pool_params_np"].items()
            if k not in self._base_weight_names
        }
        # Float32 per-subnet params (BN gamma/beta/mean/var, fc_weight, fc_bias).
        non_pool_params = built.get("non_pool_params_np", {})
        non_base_mb = (
            sum(a.nbytes for a in non_base_params.values())
            + sum(a.nbytes for a in non_pool_params.values())
        ) / (1024.0 ** 2)

        self.subnets.append({
            "subnet_id": sid,
            "arch": built["arch"],
            "graph": built["graph"],
            "remote_fname": fname,
            "extra_build_params": built["params"],
            "non_base_params_np": non_base_params,  # int8 transform matrices
            "non_pool_params_np": non_pool_params,  # float32 BN + fc
            "non_base_mb": non_base_mb,
        })
        print("  Registered %-30s  non-base=%.1f MB  upload=%.0f ms"
              % (sid, non_base_mb, up_ms), flush=True)

    # ------------------------------------------------------------------
    def _evict(self):
        if self.live_m is not None:
            del self.live_m
            self.live_m = None
            self.live_idx = None
            gc.collect()

    # ------------------------------------------------------------------
    def run(self, idx, input_np):
        """Run inference on subnet idx. Returns (output_np, timing_dict, subnet_id)."""
        assert 0 <= idx < len(self.subnets)
        sub = self.subnets[idx]
        inp = tvm.nd.array(input_np.astype("float32"), self.ctx)

        create_ms = bind_base_ms = bind_other_ms = 0.0

        if idx != self.live_idx:
            self._evict()

            t0 = time.time()
            rlib = self.remote.load_module(sub["remote_fname"])
            m = graph_runtime.create(sub["graph"], rlib, self.ctx)
            create_ms = (time.time() - t0) * 1000.0

            # Shared base weights: device-local ext_dev -> module-pool copy (~6 ms/tensor)
            t0 = time.time()
            for name, ext_arr in self._shared_base_ext.items():
                try:
                    m.set_input(name, ext_arr)
                except Exception:
                    pass  # module may not use every base weight key
            bind_base_ms = (time.time() - t0) * 1000.0

            # Non-base pool params (int8 transform matrices) + float32 BN/fc: RPC transfer
            t0 = time.time()
            for name, arr in sub["non_base_params_np"].items():
                try:
                    m.set_input(name, arr)
                except Exception:
                    pass
            for name, arr in sub["non_pool_params_np"].items():
                try:
                    m.set_input(name, arr)
                except Exception:
                    pass
            if sub["extra_build_params"]:
                m.set_input(**sub["extra_build_params"])
            bind_other_ms = (time.time() - t0) * 1000.0

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
            "bind_base": bind_base_ms,
            "bind_other": bind_other_ms,
            "switch": create_ms + bind_base_ms + bind_other_ms,
            "run": run_ms,
        }, sub["subnet_id"]

    # ------------------------------------------------------------------
    def teardown(self):
        self._evict()
        self._shared_base_ext.clear()
        self.subnets = []
        gc.collect()

    def __len__(self):
        return len(self.subnets)


# ============================================================
# CLI / main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="GMTF-on-VTA shared-pool switching runtime")
    p.add_argument("--sa-results", default=SA_RESULTS_FILE)
    p.add_argument("--arch-file", default=ARCH_FILE)
    p.add_argument("--n", type=int, default=25)
    p.add_argument("--lambda", dest="lambda_value", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-subnets", type=int, default=2,
                   help="K: how many subnets (POC default = 2)")
    p.add_argument("--switch-iters", type=int, default=20)
    p.add_argument("--warm-iters", type=int, default=3)
    p.add_argument("--rng-seed", type=int, default=42)
    p.add_argument("--build-only", action="store_true",
                   help="Stop after relay.build (no VTA device needed)")
    return p.parse_args()


def sep(t=""):
    print(("=" * 8) + " " + t + " " + ("=" * max(0, 60 - len(t))), flush=True)


def main():
    args = parse_args()
    sep("GMTF-on-VTA Shared-Pool Switching  (K=%d)" % args.num_subnets)

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
    archs = pick_subnets_from_sa(
        args.sa_results, args.arch_file,
        target_n=args.n, target_lambda=args.lambda_value,
        target_seed=args.seed, k=args.num_subnets,
    )

    # ------------------------------------------------------------------
    # Build phase is CPU-only (relay.build) — no VTA device needed.
    env = vta.get_env()
    schedule_logs = load_schedule_logs()

    print("[3] Build %d VTA-intrinsic (GMTF+conv) subnet modules ..." % len(archs), flush=True)
    print("    (GMTF+conv build: ~60-90 s/subnet)", flush=True)
    built_list = []
    for sid, arch in archs.items():
        built = build_gmtf_vta_subnet(sid, arch, ofa_net, pool, env, schedule_logs)
        built_list.append(built)

    if args.build_only:
        sep("Build-only mode: done (all relay.build calls succeeded)")
        return

    # ------------------------------------------------------------------
    print("[4] Connect VTA RPC ...", flush=True)
    remote = rpc.connect(DEVICE_HOST, DEVICE_PORT)
    vta.reconfig_runtime(remote)
    ctx = remote.ext_dev(0)

    # ------------------------------------------------------------------
    print("[5] Init runtime + upload shared base weights ...", flush=True)  # noqa
    rt = GmtfVtaRuntime(remote, ctx)
    # Base weight int8 values are the same for all subnets; use first subnet's pool params.
    rt.init_shared_base_weights(built_list[0]["runtime_pool_params_np"], base_weight_names)

    print("[6] Register %d modules ..." % len(built_list), flush=True)
    for built in built_list:
        rt.register(built)

    cma_est = rt._base_mb + 66.0  # base_weights + one live module pool
    non_base_total = sum(s["non_base_mb"] for s in rt.subnets)
    print("", flush=True)
    print("  Shared base weights on ext_dev: %.1f MB" % rt._base_mb, flush=True)
    print("  Non-base per-subnet (host numpy): %.1f MB avg"
          % (non_base_total / max(1, len(rt))), flush=True)
    print("  Est. CMA (base + 1 live module): ~%.0f MB  (vs step4 K-resident: ~%.0f MB)"
          % (cma_est, 53.0 * len(rt)), flush=True)

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
        print("  %-30s vta=%4d ref=%4d %-9s  "
              "switch=%5.0f ms (create=%5.0f base=%5.0f other=%5.0f)  run=%5.0f ms"
              % (sid, t1v, t1r, match,
                 tm["switch"], tm["create"], tm["bind_base"], tm["bind_other"],
                 tm["run"]), flush=True)

    # ------------------------------------------------------------------
    print("", flush=True)
    print("[8] Warm run on subnet 0 (%d iters) ..." % args.warm_iters, flush=True)
    print("    iter 0 = switch; iters 1+ = same-subnet (GMTF still runs each time)", flush=True)
    rt._evict()
    warm_run_times = []
    for i in range(args.warm_iters):
        out, tm, sid = rt.run(0, input_np)
        tag = "[switch]" if tm["switch"] > 0 else "[warm]  "
        print("  iter %d %s  switch=%5.0f ms  run=%5.0f ms"
              "  (create=%5.0f base=%5.0f other=%5.0f)"
              % (i, tag, tm["switch"], tm["run"],
                 tm["create"], tm["bind_base"], tm["bind_other"]), flush=True)
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
    sweep_switch, sweep_run = [], []
    for i in range(len(rt)):
        _, tm, sid = rt.run(i, input_np)
        sweep_switch.append(tm["switch"])
        sweep_run.append(tm["run"])
        print("  -> %-30s  switch=%5.0f ms (create=%5.0f base=%5.0f other=%5.0f)  run=%5.0f ms"
              % (sid, tm["switch"], tm["create"], tm["bind_base"], tm["bind_other"],
                 tm["run"]), flush=True)
    sw = np.array(sweep_switch)
    print("  Switch overhead: mean=%.0f ms  min=%.0f  max=%.0f"
          % (sw.mean(), sw.min(), sw.max()), flush=True)

    # ------------------------------------------------------------------
    print("", flush=True)
    print("[10] Random switching: %d iters ..." % args.switch_iters, flush=True)
    random.seed(args.rng_seed)
    agg = {"create": [], "bind_base": [], "bind_other": [], "switch": [], "run": []}
    same_run = []
    seq = []
    prev_idx = rt.live_idx
    for _ in range(args.switch_iters):
        k = random.randrange(len(rt))
        out, tm, sid = rt.run(k, input_np)
        for key in agg:
            agg[key].append(tm[key])
        seq.append(k)
        if k == prev_idx:
            same_run.append(tm["run"])
        prev_idx = k

    print("  Switch sequence (first 20): %s" % seq[:20], flush=True)
    print("  Selection histogram: %s"
          % dict(sorted(collections.Counter(seq).items())), flush=True)
    print("", flush=True)
    for key in ("create", "bind_base", "bind_other", "switch", "run"):
        a = np.array(agg[key])
        print("  %-10s  mean=%6.0f ms  min=%6.0f  max=%6.0f"
              % (key, a.mean(), a.min(), a.max()), flush=True)
    if same_run:
        ss = np.array(same_run)
        print("  Same-subnet (no switch):  mean=%.0f ms  n=%d" % (ss.mean(), len(ss)),
              flush=True)

    # ------------------------------------------------------------------
    sep("Summary")
    print("  K=%d  |  GMTF+conv on VTA  |  base weights shared (ext_dev)" % len(rt),
          flush=True)
    print("", flush=True)
    print("  CMA breakdown:", flush=True)
    print("    Shared base weights (ext_dev, permanent): %.1f MB" % rt._base_mb, flush=True)
    print("    Live module pool (one at a time, est.):  ~66 MB", flush=True)
    print("    Total (est.):                            ~%.0f MB  (fixed, K-independent)"
          % cma_est, flush=True)
    print("", flush=True)
    print("  Timings (random-switch sweep):", flush=True)
    print("    Switch overhead:  mean=%.0f ms"
          "  (create %.0f + base_bind %.0f + other_bind %.0f)"
          % (np.mean(agg["switch"]), np.mean(agg["create"]),
             np.mean(agg["bind_base"]), np.mean(agg["bind_other"])), flush=True)
    print("    Inference (run):  mean=%.0f ms  [GMTF+conv, all on VTA FPGA]"
          % np.mean(agg["run"]), flush=True)
    print("", flush=True)
    print("  Professor comparison:", flush=True)
    print("    K-resident (step4):  0 ms switch,   ~117 ms run,  ~%d MB CMA"
          % int(53 * len(rt)), flush=True)
    print("    Lazy K=1   (step5):  ~18000 ms switch (net bw), ~117 ms run, ~53 MB CMA",
          flush=True)
    print("    GMTF-VTA   (step6):  ~%.0f ms switch,  ~%.0f ms run, ~%.0f MB CMA"
          % (np.mean(agg["switch"]), np.mean(agg["run"]), cma_est), flush=True)
    sep("Done")

    rt.teardown()
    del remote
    gc.collect()
    print("Released device + RPC, exiting.", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
