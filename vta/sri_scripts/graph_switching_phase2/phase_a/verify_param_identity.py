"""
Phase A: Weight Identity Verification
======================================

Purpose
-------
Answer the four foundational questions for Option B before any implementation:

  Q1. Are quantized weights deterministic?
      - Compile the same subnet TWICE. Are the params dicts bitwise identical?
      - If not, the whole deduplication strategy must account for non-determinism.

  Q2. Are same-config layers in different subnets identical post-compilation?
      - Find two subnets that share a layer with the SAME configuration
        (same block_idx, same out_ch, same in_ch, same kernel_size, same decomp_type).
      - Confirm their post-compiled weight tensors are bitwise identical.

  Q3. How many unique weight tensors exist across all N subnets?
      - Compile all subnets in the candidate set.
      - Hash every param tensor from relay.build output.
      - Count unique hashes → this is the theoretical memory savings.

  Q4. Is param order (p0, p1, ...) stable across independent compilations of the
      SAME subnet?
      - Compile the same subnet twice independently.
      - Verify that p0 in run1 == p0 in run2, p1 in run1 == p1 in run2, etc.

Outputs
-------
  phase_a/results/
    param_identity_report.json       - Full per-subnet, per-param analysis
    weight_uniqueness_report.json    - Deduplication analysis across all subnets
    determinism_report.json          - Q1 + Q4 results
    PHASE_A_SUMMARY.md               - Human-readable summary of all findings

Usage
-----
  cd .../graph_switching_phase2/phase_a
  python verify_param_identity.py --num_subnets 3   # quick test (3 subnets)
  python verify_param_identity.py --num_subnets 25  # full run  (all subnets)
"""

from __future__ import absolute_import, print_function

import os
import sys
import json
import time
import hashlib
import argparse
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np



# ---------------------------------------------------------------------------
# Path setup — must come before TVM/VTA/OFA imports
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_SRI_SCRIPTS = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))      # .../tvm/vta/sri_scripts
_VTA_DIR     = os.path.abspath(os.path.join(_SRI_SCRIPTS, ".."))           # .../tvm/vta
_VTA_PYTHON  = os.path.abspath(os.path.join(_VTA_DIR, "python"))           # .../tvm/vta/python  (vta pkg lives here)
_TVM_DIR     = os.path.abspath(os.path.join(_VTA_DIR, ".."))               # .../tvm
_TVM_PYTHON  = os.path.abspath(os.path.join(_TVM_DIR, "python"))           # .../tvm/python

EXTERNAL_REPO_ROOT = "/home/srchand/Desktop/research/OFA_Obfs"
OFA_CHECKPOINT     = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth"
CANDIDATE_SET_JSON = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/candidate_sets_results_all_expts.json"
ARCH_CONFIG_JSON   = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json"
SCHEDULE_LOG_GLOB  = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/logs/tuning_logs/vta_1x16x16/candidate_set/*.log"
EXPERIMENT_NAME    = "sa_lam_2.0"
DEVICE_HOST        = "10.42.0.188"
DEVICE_PORT        = 9091

RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")

# NOTE: _VTA_DIR is intentionally NOT added to sys.path.
# tvm/vta/ has its own __init__.py which would shadow tvm/vta/python/vta/
# (the real package that contains vta.top). Only _VTA_PYTHON is added.
for p in [_VTA_PYTHON, _TVM_PYTHON, _SRI_SCRIPTS, EXTERNAL_REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Utility: array hashing
# ---------------------------------------------------------------------------

def hash_array(arr: np.ndarray) -> str:
    """SHA-256 of the raw bytes of a numpy array (shape + dtype + data)."""
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def arrays_bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff arrays have same shape, dtype, and every element is equal."""
    if a.shape != b.shape:
        return False
    if a.dtype != b.dtype:
        return False
    return np.array_equal(a, b)


def param_dict_to_numpy(params: Dict) -> Dict[str, np.ndarray]:
    """Convert a TVM params dict (tvm.nd.NDArray values) to numpy."""
    out = {}
    for k, v in params.items():
        try:
            out[k] = v.asnumpy()
        except Exception:
            out[k] = np.asarray(v)
    return out


# ---------------------------------------------------------------------------
# Imports that require the path setup above
# ---------------------------------------------------------------------------

def _do_imports():
    """Lazy imports so path setup happens first."""
    import torch
    import tvm
    from tvm import rpc, autotvm, relay
    from tvm.contrib import graph_runtime, utils

    # _VTA_PYTHON is at position 0 of sys.path (set at module level).
    # Import vta from there — it has vta.top. No eviction needed.
    import vta
    from vta.top import graph_pack

    # OFA / architecture imports
    for mod_name in ["ofa_base_models", "architecture_defense"]:
        existing = sys.modules.get(mod_name)
        if existing is not None:
            mod_file = getattr(existing, "__file__", "") or ""
            if "sri_scripts" in mod_file:
                del sys.modules[mod_name]

    from ofa_base_models import OFADynamicResnetAllMod          # type: ignore
    from architecture_defense import StaticResNetFromArch        # type: ignore

    return (torch, tvm, rpc, autotvm, relay,
            graph_runtime, utils, vta, graph_pack,
            OFADynamicResnetAllMod, StaticResNetFromArch)


# ---------------------------------------------------------------------------
# Compilation helpers (mirrors execute_candidate_set_refactored.py)
# ---------------------------------------------------------------------------

PACK_DICT = {
    "resnet18": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
}


def compile_subnet(
    model_id: str,
    arch: Dict[str, Any],
    ofa_net,
    env,
    target,
    target_host,
    schedule_log_files: List[str],
    torch_mod,
    tvm_mod,
    relay_mod,
    graph_runtime_mod,
    vta_mod,
    graph_pack_fn,
    StaticResNetFromArch_cls,
    autotvm_mod,
    opt_level: int = 3,
    global_scale: float = 8.0,
    skip_conv_layers: Optional[List[int]] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Compile a single subnet all the way to relay.build params.

    Returns a dict  {param_name: numpy_array}  (post-quantization, post-packing)
    or None on failure.
    """
    if skip_conv_layers is None:
        skip_conv_layers = [0]

    try:
        # Build static model with transformed weights
        ofa_net.set_active_subnet(arch)
        static_net = StaticResNetFromArch_cls(
            target_arch=arch,
            num_classes=10,
            width_mult_list=(0.5, 1.0, 2.0)
        )
        static_net.load_weights_from_ofa_checkpoint(
            checkpoint_path=OFA_CHECKPOINT, ofa_model=ofa_net
        )
        static_net.eval()
        with autotvm_mod.tophub.context(target, extra_files=schedule_log_files):

            input_shape = [env.BATCH, 3, 224, 224]
            input_data  = torch_mod.randn(input_shape)

            with torch_mod.no_grad():
                scripted = torch_mod.jit.trace(static_net, input_data).eval()

            shape_list = [("input0", input_shape)]

            # Frontend conversion
            mod, params = relay_mod.frontend.from_pytorch(scripted, shape_list)
            mod = relay_mod.transform.InferType()(mod)

            # Keep the float32 params for matching later
            float32_params = {k: np.array(v.asnumpy()) for k, v in params.items()}

            # Quantize + pack (VTA path)
            with tvm_mod.transform.PassContext(
                opt_level=opt_level,
                disabled_pass={"AlterOpLayout"}
            ):
                with relay_mod.quantize.qconfig(
                    global_scale=global_scale,
                    skip_conv_layers=skip_conv_layers
                ):
                    mod_q = relay_mod.quantize.quantize(mod, params=params)

                assert env.BLOCK_IN == env.BLOCK_OUT
                relay_prog = graph_pack_fn(
                    mod_q["main"],
                    env.BATCH,
                    env.BLOCK_IN,
                    env.BLOCK_OUT,
                    env.WGT_WIDTH,
                    start_name=PACK_DICT["resnet18"][0],
                    stop_name=PACK_DICT["resnet18"][1],
                    device_annot=(env.TARGET == "intelfocl"),
                )

            # Build
            build_target = target
            if env.TARGET == "intelfocl":
                build_target = {"cpu": env.target_vta_cpu, "ext_dev": target}

            with vta_mod.build_config(
                opt_level=opt_level,
                disabled_pass={"AlterOpLayout"}
            ):
                graph, lib, built_params = relay_mod.build(
                    relay_prog,
                    target=build_target,
                    params=params,
                    target_host=target_host
                )

            built_params_np = param_dict_to_numpy(built_params)

        return {
            "model_id":       model_id,
            "graph":          graph,
            "lib":            lib,
            "float32_params": float32_params,     # pre-quantization, for matching
            "built_params":   built_params_np,    # post-build, for deduplication
            "param_shapes":   {k: list(v.shape) for k, v in built_params_np.items()},
            "param_dtypes":   {k: str(v.dtype)  for k, v in built_params_np.items()},
        }

    except Exception as e:
        import traceback
        print(f"  [FAILED] {model_id}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Q1 + Q4: Determinism test  (same subnet, compiled twice)
# ---------------------------------------------------------------------------

def test_determinism(
    model_id: str,
    arch: Dict[str, Any],
    ofa_net,
    compile_kwargs: Dict,
) -> Dict[str, Any]:
    """
    Compile the same subnet twice independently.
    Check:
      Q1 - Are the post-build param tensors bitwise identical?
      Q4 - Is the param ordering (key set + order) stable?
    """
    print(f"\n{'='*70}")
    print(f"Determinism Test:  {model_id}")
    print(f"{'='*70}")

    results = {"model_id": model_id, "q1_deterministic": None, "q4_order_stable": None,
               "run1_param_count": None, "run2_param_count": None,
               "mismatches": [], "details": {}}

    print("  Compiling run 1 ...")
    t0 = time.time()
    run1 = compile_subnet(model_id, arch, ofa_net, **compile_kwargs)
    print(f"  Run 1 done in {time.time()-t0:.1f}s")

    if run1 is None:
        results["q1_deterministic"] = False
        results["q4_order_stable"]  = False
        results["error"] = "Run 1 compilation failed"
        return results

    print("  Compiling run 2 ...")
    t0 = time.time()
    run2 = compile_subnet(model_id, arch, ofa_net, **compile_kwargs)
    print(f"  Run 2 done in {time.time()-t0:.1f}s")

    if run2 is None:
        results["q1_deterministic"] = False
        results["q4_order_stable"]  = False
        results["error"] = "Run 2 compilation failed"
        return results

    params1 = run1["built_params"]
    params2 = run2["built_params"]

    results["run1_param_count"] = len(params1)
    results["run2_param_count"] = len(params2)

    # Q4: key order stable?
    keys1 = list(params1.keys())
    keys2 = list(params2.keys())
    order_stable = (keys1 == keys2)
    results["q4_order_stable"] = order_stable
    if not order_stable:
        results["key_diff"] = {
            "run1_only": [k for k in keys1 if k not in keys2],
            "run2_only": [k for k in keys2 if k not in keys1],
        }
        print(f"  ⚠  Q4 FAILED — param key ordering differs between runs!")
    else:
        print(f"  ✓  Q4 PASSED — param ordering is stable ({len(keys1)} params)")

    # Q1: bitwise identity
    mismatches = []
    all_match  = True
    for k in keys1:
        if k not in params2:
            mismatches.append({"param": k, "issue": "missing in run2"})
            all_match = False
            continue
        a, b = params1[k], params2[k]
        equal = arrays_bitwise_equal(a, b)
        if not equal:
            all_match = False
            mismatches.append({
                "param":      k,
                "shape":      list(a.shape),
                "dtype":      str(a.dtype),
                "max_diff":   float(np.max(np.abs(a.astype(float) - b.astype(float)))),
                "issue":      "values differ"
            })

    results["q1_deterministic"] = all_match
    results["mismatches"]       = mismatches

    if all_match:
        print(f"  ✓  Q1 PASSED — all {len(keys1)} param tensors are bitwise identical across runs")
    else:
        print(f"  ✗  Q1 FAILED — {len(mismatches)} params differ between runs")
        for m in mismatches[:5]:
            print(f"       {m['param']}: {m.get('issue','')}  max_diff={m.get('max_diff','?')}")

    return results


# ---------------------------------------------------------------------------
# Q2: Cross-subnet identity  (same-config layer = identical tensor?)
# ---------------------------------------------------------------------------

def analyze_cross_subnet_identity(
    compiled_results: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    For every pair of subnets, check if any params are numerically identical.

    Strategy:
      - Hash every built_params tensor for every subnet.
      - Group by (shape, dtype, hash) → tensors with the same hash are identical.
      - Report which subnets share which tensors.
    """
    print(f"\n{'='*70}")
    print("Cross-Subnet Weight Identity Analysis (Q2 + Q3)")
    print(f"{'='*70}")

    # Step 1: hash every param in every subnet
    # hash_to_subnets[h] = list of (subnet_id, param_name, shape, dtype)
    hash_to_entries = defaultdict(list)
    total_param_count = 0

    for model_id, result in compiled_results.items():
        if result is None:
            continue
        for pname, arr in result["built_params"].items():
            h = hash_array(arr)
            hash_to_entries[h].append({
                "subnet_id":  model_id,
                "param_name": pname,
                "shape":      list(arr.shape),
                "dtype":      str(arr.dtype),
                "bytes":      arr.nbytes,
            })
            total_param_count += 1

    unique_hashes  = len(hash_to_entries)
    shared_hashes  = sum(1 for entries in hash_to_entries.values() if len(entries) > 1)
    private_hashes = unique_hashes - shared_hashes

    # Memory calculation
    unique_bytes = sum(
        entries[0]["bytes"] for entries in hash_to_entries.values()
    )
    total_bytes = sum(
        e["bytes"]
        for entries in hash_to_entries.values()
        for e in entries
    )
    savings_bytes   = total_bytes - unique_bytes
    savings_percent = 100.0 * savings_bytes / total_bytes if total_bytes > 0 else 0.0

    print(f"\n  Subnets analyzed:          {len(compiled_results)}")
    print(f"  Total param tensors:       {total_param_count}")
    print(f"  Unique param tensors:      {unique_hashes}")
    print(f"  Shared tensors (in >1 subnet): {shared_hashes}")
    print(f"  Private tensors (in 1 subnet): {private_hashes}")
    print(f"\n  Total weight bytes:        {total_bytes/1e6:.2f} MB")
    print(f"  Unique weight bytes:       {unique_bytes/1e6:.2f} MB")
    print(f"  Memory savings:            {savings_bytes/1e6:.2f} MB  ({savings_percent:.1f}%)")

    # Per-subnet breakdown
    subnet_stats = {}
    for model_id, result in compiled_results.items():
        if result is None:
            continue
        params = result["built_params"]
        private_count = 0
        shared_count  = 0
        for pname, arr in params.items():
            h = hash_array(arr)
            if len(hash_to_entries[h]) > 1:
                shared_count  += 1
            else:
                private_count += 1
        subnet_stats[model_id] = {
            "total_params":   len(params),
            "shared_params":  shared_count,
            "private_params": private_count,
            "sharing_pct":    100.0 * shared_count / len(params) if params else 0.0,
        }

    # Most widely shared tensors (top 10)
    sorted_by_sharing = sorted(
        hash_to_entries.items(),
        key=lambda kv: len(kv[1]),
        reverse=True
    )
    top_shared = []
    for h, entries in sorted_by_sharing[:10]:
        top_shared.append({
            "hash":       h[:12],
            "shape":      entries[0]["shape"],
            "dtype":      entries[0]["dtype"],
            "bytes":      entries[0]["bytes"],
            "num_subnets": len(entries),
            "subnets":    [e["subnet_id"] for e in entries],
            "param_names": [e["param_name"] for e in entries],
        })

    print(f"\n  Top shared weight tensors:")
    for t in top_shared[:5]:
        print(f"    shape={t['shape']} dtype={t['dtype']}  "
              f"used by {t['num_subnets']} subnets  "
              f"({t['bytes']/1024:.1f} KB each)")

    # Q2 verdict
    q2_confirmed = shared_hashes > 0
    print(f"\n  Q2 Answer: Same-config layers produce identical post-compiled tensors?  "
          f"{'✓ YES' if q2_confirmed else '✗ NO'}")
    print(f"  Q3 Answer: Unique tensors across all subnets = {unique_hashes}  "
          f"(saves {savings_percent:.1f}% memory)")

    return {
        "total_param_count":  total_param_count,
        "unique_tensor_count": unique_hashes,
        "shared_tensor_count": shared_hashes,
        "private_tensor_count": private_hashes,
        "total_bytes":         total_bytes,
        "unique_bytes":        unique_bytes,
        "savings_bytes":       savings_bytes,
        "savings_percent":     savings_percent,
        "subnet_stats":        subnet_stats,
        "top_shared_tensors":  top_shared,
        "q2_confirmed":        q2_confirmed,
        "hash_to_entries":     {
            h: entries
            for h, entries in sorted_by_sharing
        },
    }


# ---------------------------------------------------------------------------
# Pairwise param-match table  (for Q2 deep-dive)
# ---------------------------------------------------------------------------

def build_pairwise_match_table(
    compiled_results: Dict[str, Dict],
    subnet_ids: List[str],
) -> Dict[str, Any]:
    """
    For a given pair of subnets, produce a table showing which pN in subnet1
    matches which pM in subnet2.
    """
    if len(subnet_ids) < 2:
        return {}

    s1_id, s2_id = subnet_ids[0], subnet_ids[1]
    r1 = compiled_results.get(s1_id)
    r2 = compiled_results.get(s2_id)
    if r1 is None or r2 is None:
        return {}

    p1 = r1["built_params"]
    p2 = r2["built_params"]

    # Hash lookup for subnet2
    hash_to_p2name = {}
    for pname, arr in p2.items():
        hash_to_p2name[hash_array(arr)] = pname

    table = []
    for pname1, arr1 in p1.items():
        h1   = hash_array(arr1)
        p2nm = hash_to_p2name.get(h1, None)
        table.append({
            "subnet1_param": pname1,
            "shape":         list(arr1.shape),
            "dtype":         str(arr1.dtype),
            "bytes":         arr1.nbytes,
            "subnet2_param": p2nm,
            "match":         p2nm is not None,
        })

    matched   = sum(1 for r in table if r["match"])
    unmatched = len(table) - matched

    print(f"\n  Pairwise Match:  {s1_id}  vs  {s2_id}")
    print(f"  {s1_id} has {len(p1)} params,  {s2_id} has {len(p2)} params")
    print(f"  Matched: {matched}   Unmatched (unique to {s1_id}): {unmatched}")

    for row in table:
        status = f"  ↔  {row['subnet2_param']}" if row["match"] else "  (unique)"
        print(f"    {row['subnet1_param']:6s}  shape={str(row['shape']):25s}  {status}")

    return {
        "subnet1": s1_id,
        "subnet2": s2_id,
        "table":   table,
        "matched": matched,
        "unmatched": unmatched,
    }


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

def write_markdown_summary(
    determinism_results: List[Dict],
    cross_subnet: Dict,
    pairwise: Dict,
    output_path: str,
):
    lines = []
    lines.append("# Phase A: Weight Identity Verification — Summary\n")
    lines.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n\n")

    # ---- Q1 + Q4 ----
    lines.append("## Q1 + Q4: Compilation Determinism\n")
    for r in determinism_results:
        mid = r["model_id"]
        q1  = "✅ PASS" if r.get("q1_deterministic") else "❌ FAIL"
        q4  = "✅ PASS" if r.get("q4_order_stable")  else "❌ FAIL"
        lines.append(f"### {mid}\n")
        lines.append(f"- **Q1** (bitwise identical across two compilations): {q1}\n")
        lines.append(f"- **Q4** (param ordering stable): {q4}\n")
        n = r.get("run1_param_count", "?")
        lines.append(f"- Params compiled: {n}\n")
        if r.get("mismatches"):
            lines.append(f"- Mismatches: {len(r['mismatches'])}\n")
            for m in r["mismatches"][:3]:
                lines.append(f"  - `{m['param']}`: {m.get('issue','')} "
                              f"max_diff={m.get('max_diff','?')}\n")
        lines.append("\n")

    # ---- Q2 + Q3 ----
    if cross_subnet:
        cs = cross_subnet
        lines.append("## Q2 + Q3: Cross-Subnet Weight Identity\n")
        lines.append(f"| Metric | Value |\n|---|---|\n")
        lines.append(f"| Subnets analyzed | {len(cs.get('subnet_stats', {}))} |\n")
        lines.append(f"| Total param tensors | {cs['total_param_count']} |\n")
        lines.append(f"| Unique param tensors | {cs['unique_tensor_count']} |\n")
        lines.append(f"| Shared tensors (>1 subnet) | {cs['shared_tensor_count']} |\n")
        lines.append(f"| Total weight bytes | {cs['total_bytes']/1e6:.2f} MB |\n")
        lines.append(f"| Unique weight bytes | {cs['unique_bytes']/1e6:.2f} MB |\n")
        lines.append(f"| **Memory savings** | **{cs['savings_percent']:.1f}%** |\n")
        lines.append(f"| Q2 confirmed | {'✅ YES' if cs['q2_confirmed'] else '❌ NO'} |\n")
        lines.append("\n")

        lines.append("### Top 10 Most Shared Weight Tensors\n")
        lines.append("| Hash (prefix) | Shape | dtype | Size (KB) | Used by N subnets |\n")
        lines.append("|---|---|---|---|---|\n")
        for t in cs.get("top_shared_tensors", []):
            lines.append(f"| `{t['hash']}` | {t['shape']} | {t['dtype']} "
                         f"| {t['bytes']/1024:.1f} | {t['num_subnets']} |\n")
        lines.append("\n")

        lines.append("### Per-Subnet Sharing Breakdown\n")
        lines.append("| Subnet | Total Params | Shared | Private | Sharing % |\n")
        lines.append("|---|---|---|---|---|\n")
        for sid, s in cs.get("subnet_stats", {}).items():
            lines.append(f"| {sid} | {s['total_params']} | {s['shared_params']} "
                         f"| {s['private_params']} | {s['sharing_pct']:.1f}% |\n")
        lines.append("\n")

    # ---- Pairwise ----
    if pairwise:
        s1, s2 = pairwise.get("subnet1","?"), pairwise.get("subnet2","?")
        lines.append(f"## Pairwise Param Match: `{s1}` vs `{s2}`\n")
        lines.append(f"- Matched: {pairwise['matched']}   "
                     f"Unique to {s1}: {pairwise['unmatched']}\n\n")
        lines.append(f"| {s1} param | Shape | Match in {s2} |\n|---|---|---|\n")
        for row in pairwise.get("table", []):
            match_str = f"`{row['subnet2_param']}`" if row["match"] else "_(unique)_"
            lines.append(f"| `{row['subnet1_param']}` | {row['shape']} | {match_str} |\n")
        lines.append("\n")

    # ---- Verdict ----
    lines.append("## Final Verdict for Option B\n")
    q1_ok = all(r.get("q1_deterministic") for r in determinism_results)
    q4_ok = all(r.get("q4_order_stable")  for r in determinism_results)
    q2_ok = cross_subnet.get("q2_confirmed", False) if cross_subnet else False
    savings = cross_subnet.get("savings_percent", 0.0) if cross_subnet else 0.0

    lines.append(f"- Q1 (deterministic compilation):  {'✅' if q1_ok else '❌'}\n")
    lines.append(f"- Q2 (identical tensors across subnets): {'✅' if q2_ok else '❌'}\n")
    lines.append(f"- Q3 (memory savings potential):  {savings:.1f}%\n")
    lines.append(f"- Q4 (stable param ordering):  {'✅' if q4_ok else '❌'}\n\n")

    if q1_ok and q2_ok and q4_ok:
        lines.append("**✅ All foundational checks PASSED. Option B is viable. Proceed to Phase B.**\n")
    else:
        lines.append("**⚠️  Some checks failed. Review results before proceeding.**\n")

    with open(output_path, "w") as f:
        f.writelines(lines)
    print(f"\n  Markdown summary written → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase A: Verify weight identity across subnet compilations"
    )
    parser.add_argument(
        "--num_subnets", type=int, default=3,
        help="Number of subnets to analyze (default=3 for quick test, 25 for full)"
    )
    parser.add_argument(
        "--skip_determinism", action="store_true",
        help="Skip Q1/Q4 determinism test (saves ~2× compile time per subnet)"
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("PHASE A: Weight Identity Verification")
    print(f"Subnets to analyze: {args.num_subnets}")
    print("=" * 70)

    # --- Imports ---
    print("\nLoading imports...")
    (torch, tvm, rpc, autotvm, relay,
     graph_runtime, utils, vta, graph_pack,
     OFADynamicResnetAllMod, StaticResNetFromArch) = _do_imports()

    import glob as glob_mod

    # --- VTA env + target ---
    env    = vta.get_env()
    target = env.target

    # --- RPC ---
    print("Connecting to VTA device...")
    if env.TARGET not in ["sim", "tsim", "intelfocl"]:
        remote = rpc.connect(DEVICE_HOST, DEVICE_PORT)
        vta.reconfig_runtime(remote)
        print(f"  Connected to {DEVICE_HOST}:{DEVICE_PORT}")
    else:
        remote = rpc.LocalSession()
        print("  Using local session (sim)")

    # --- Schedule logs ---
    schedule_log_files = sorted(glob_mod.glob(SCHEDULE_LOG_GLOB))
    print(f"  Loaded {len(schedule_log_files)} schedule log files")

    # --- OFA model ---
    print("\nLoading OFA model (once)...")
    ofa_net = OFADynamicResnetAllMod()
    checkpoint = torch.load(OFA_CHECKPOINT, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    ofa_net.load_state_dict(state, strict=False)
    ofa_net.eval()
    print("  OFA model loaded")

    # --- Candidate set ---
    print("\nLoading candidate set...")
    # Import the battle-tested loader from the refactored script
    sys.path.insert(0, _SRI_SCRIPTS)
    from execute_candidate_set_refactored import load_arch_mapping  # type: ignore

    with open(CANDIDATE_SET_JSON) as f:
        results_data = json.load(f)
    model_ids_all = results_data[EXPERIMENT_NAME]["ids"]
    print(f"Loading experiment '{EXPERIMENT_NAME}': {len(model_ids_all)} models")

    arch_all = load_arch_mapping(ARCH_CONFIG_JSON)

    model_ids = [mid for mid in model_ids_all if mid in arch_all][:args.num_subnets]
    print(f"  Using {len(model_ids)} subnets: {model_ids}")

    # Shared kwargs for compile_subnet
    compile_kwargs = dict(
        env=env,
        target=target,
        target_host=env.target_host,
        schedule_log_files=schedule_log_files,
        torch_mod=torch,
        tvm_mod=tvm,
        relay_mod=relay,
        graph_runtime_mod=graph_runtime,
        vta_mod=vta,
        graph_pack_fn=graph_pack,
        StaticResNetFromArch_cls=StaticResNetFromArch,
        autotvm_mod=autotvm
    )

    # -----------------------------------------------------------------------
    # Q1 + Q4: Determinism (first subnet only, unless skipped)
    # -----------------------------------------------------------------------
    determinism_results = []
    if not args.skip_determinism:
        det_subnet_id = model_ids[0]
        det_arch      = arch_all[det_subnet_id]
        print(f"\n{'='*70}")
        print(f"Determinism Test — compiling {det_subnet_id} TWICE")
        det_result = test_determinism(
            det_subnet_id, det_arch, ofa_net, compile_kwargs
        )
        determinism_results.append(det_result)

        det_path = os.path.join(RESULTS_DIR, "determinism_report.json")
        with open(det_path, "w") as f:
            # mismatches may contain numpy-incompatible types; sanitise
            json.dump(det_result, f, indent=2, default=str)
        print(f"\n  Determinism report → {det_path}")
    else:
        print("\n  [Skipped Q1/Q4 determinism test]")

    # -----------------------------------------------------------------------
    # Compile all N subnets (one pass each)
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Compiling {len(model_ids)} subnets for cross-subnet analysis...")
    print(f"{'='*70}")

    compiled_results: Dict[str, Optional[Dict]] = {}

    with autotvm.tophub.context(target, extra_files=schedule_log_files):
        for i, model_id in enumerate(model_ids):
            arch = arch_all[model_id]
            print(f"\n[{i+1}/{len(model_ids)}] Compiling {model_id} ...")
            t0 = time.time()
            result = compile_subnet(model_id, arch, ofa_net, **compile_kwargs)
            elapsed = time.time() - t0
            if result is not None:
                n = len(result["built_params"])
                print(f"  ✓ Done in {elapsed:.1f}s  ({n} params)")
            else:
                print(f"  ✗ Failed after {elapsed:.1f}s")
            compiled_results[model_id] = result

    successful = {k: v for k, v in compiled_results.items() if v is not None}
    print(f"\n  Compiled successfully: {len(successful)}/{len(model_ids)}")

    # -----------------------------------------------------------------------
    # Q2 + Q3: Cross-subnet identity analysis
    # -----------------------------------------------------------------------
    cross_subnet = {}
    pairwise     = {}
    if len(successful) >= 2:
        cross_subnet = analyze_cross_subnet_identity(successful)

        # Pairwise deep-dive on the first two
        pairwise = build_pairwise_match_table(
            successful, list(successful.keys())[:2]
        )

        # Save cross-subnet results (strip hash_to_entries — too large for JSON)
        cs_save = {k: v for k, v in cross_subnet.items() if k != "hash_to_entries"}
        cs_path = os.path.join(RESULTS_DIR, "weight_uniqueness_report.json")
        with open(cs_path, "w") as f:
            json.dump(cs_save, f, indent=2, default=str)
        print(f"\n  Weight uniqueness report → {cs_path}")

        pw_path = os.path.join(RESULTS_DIR, "pairwise_match_report.json")
        with open(pw_path, "w") as f:
            json.dump(pairwise, f, indent=2, default=str)
        print(f"  Pairwise match report  → {pw_path}")

    # Save per-subnet param shapes/dtypes (no raw arrays — too large)
    identity_report = {}
    for mid, res in compiled_results.items():
        if res is None:
            identity_report[mid] = {"status": "failed"}
        else:
            identity_report[mid] = {
                "status":       "ok",
                "param_count":  len(res["built_params"]),
                "param_shapes": res["param_shapes"],
                "param_dtypes": res["param_dtypes"],
            }
    id_path = os.path.join(RESULTS_DIR, "param_identity_report.json")
    with open(id_path, "w") as f:
        json.dump(identity_report, f, indent=2, default=str)
    print(f"  Param identity report  → {id_path}")

    # -----------------------------------------------------------------------
    # Markdown summary
    # -----------------------------------------------------------------------
    md_path = os.path.join(RESULTS_DIR, "PHASE_A_SUMMARY.md")
    write_markdown_summary(determinism_results, cross_subnet, pairwise, md_path)

    # -----------------------------------------------------------------------
    # Final print
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("PHASE A COMPLETE")
    print(f"{'='*70}")
    if determinism_results:
        q1 = determinism_results[0].get("q1_deterministic")
        q4 = determinism_results[0].get("q4_order_stable")
        print(f"  Q1 (deterministic weights):   {'✓ YES' if q1 else '✗ NO'}")
        print(f"  Q4 (stable param order):      {'✓ YES' if q4 else '✗ NO'}")
    if cross_subnet:
        print(f"  Q2 (cross-subnet identity):   {'✓ YES' if cross_subnet['q2_confirmed'] else '✗ NO'}")
        print(f"  Q3 (memory savings):          {cross_subnet['savings_percent']:.1f}%")
    print(f"\n  Results in:  {RESULTS_DIR}/")
    print(f"  Summary:     {md_path}")


if __name__ == "__main__":
    main()









