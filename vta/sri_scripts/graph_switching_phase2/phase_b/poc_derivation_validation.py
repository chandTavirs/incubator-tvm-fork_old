"""
Phase B POC Test Script
========================
Validates OFA weight derivation extractor against ground-truth OFA outputs.

This is a PURE PYTHON / NUMPY test — no VTA required.
It answers two questions:

  Q1. Does OFADerivationExtractor correctly capture the channel slice + kernel
      transform parameters for each layer of a subnet?

  Q2. Does derive_weight_numpy() reproduce the exact same weights that the
      OFA model materialises during its forward pass?

If both Q1 and Q2 pass we have a solid foundation for Phase B Step 3
(wiring derivation ops into the Relay graph).

Usage
-----
  cd .../graph_switching_phase2/phase_b
  python poc_derivation_validation.py

Expected output: all layers showing "✓" with max_abs_diff < 1e-5
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE2_DIR = os.path.dirname(SCRIPT_DIR)
SRI_SCRIPTS_DIR = os.path.dirname(PHASE2_DIR)
EXTERNAL_REPO_ROOT = "/home/srchand/Desktop/research/OFA_Obfs"
TVM_ROOT = os.path.abspath(os.path.join(SRI_SCRIPTS_DIR, "..", ".."))
TVM_PYTHON = os.path.join(TVM_ROOT, "python")
INPUT_SHAPE = [1, 3, 224, 224]

for p in [EXTERNAL_REPO_ROOT, TVM_PYTHON, SRI_SCRIPTS_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from ofa_base_models import OFADynamicResnetAllMod
from ofa_derivation_extractor import (
    OFADerivationExtractor,
    derive_weight_numpy,
    validate_derivation_against_ofa,
    save_derivations,
    load_derivations,
)
from ofa_weight_pool_extractor import load_ofa_pool


# ===========================================================================
# Config
# ===========================================================================
OFA_CHECKPOINT = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth"
POOL_DIR       = os.path.join(SCRIPT_DIR, "ofa_weight_pool")
SA_RESULTS_FILE = "/home/srchand/Desktop/research/OFA_Obfs/optimization_experiments/simulated_annealing/results/sa_results_20260216-175221.json"
TARGET_N = 25
TARGET_LAMBDA = 4.0
TARGET_SEED = 0
POC_NUM_SUBNETS = 2

# Legacy defaults kept for backward compatibility
CANDIDATE_SET  = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/candidate_sets_results_all_expts.json"
ARCH_FILE      = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json"
EXPERIMENT     = "sa_lam_2.0"

# POC: pick 2 subnets that have different kernel sizes (to exercise transform path)
# arch_20250927_180844_0578 uses kernel 7 & 3 (first block uses 7→? transform in some layers)
# We'll pick two architectures that definitely include a kernel size < 7 so we exercise transforms.
POC_SUBNET_IDS = [
    "arch_20250927_180844_0578",  # has mixed kernel sizes including 3x3 layers
    "arch_20250927_180844_0034",  # deeper network, also mixed kernels
]

RESULTS_DIR = os.path.join(SCRIPT_DIR, "poc_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================================================================
# Helpers
# ===========================================================================
def load_arch_mapping(arch_file: str) -> dict:
    with open(arch_file) as f:
        data = json.load(f)
    mapping = {}
    for item in data["architectures"]:
        mapping[item["id"]] = item["architecture"]
    return mapping


def _as_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _pick_run_from_sa_results(results_obj: dict, target_n: int, target_lambda: float, target_seed: int):
    runs = results_obj.get("runs", [])
    if not isinstance(runs, list) or not runs:
        raise ValueError("Invalid SA results file: missing non-empty 'runs' list")

    # Exact N + lambda matches first.
    nl_matches = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        n = run.get("N")
        lam = _as_float(run.get("lambda"))
        if n == target_n and lam is not None and abs(lam - float(target_lambda)) < 1e-9:
            nl_matches.append(run)

    if not nl_matches:
        raise ValueError(
            f"No run found for N={target_n}, lambda={target_lambda} in SA results file"
        )

    # Prefer exact seed, otherwise deterministically use smallest available seed.
    exact_seed = [r for r in nl_matches if r.get("seed") == target_seed]
    if exact_seed:
        return exact_seed[0], False

    seeds = [r.get("seed") for r in nl_matches if isinstance(r.get("seed"), int)]
    if not seeds:
        return nl_matches[0], True

    min_seed = min(seeds)
    for r in nl_matches:
        if r.get("seed") == min_seed:
            return r, True

    return nl_matches[0], True


def _extract_ids_from_run(run: dict):
    ids = run.get("ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("Selected SA run does not contain a non-empty 'ids' list")

    norm_ids = [x for x in ids if isinstance(x, str) and x]
    # De-duplicate while preserving file order.
    unique_ids = list(dict.fromkeys(norm_ids))
    if not unique_ids:
        raise ValueError("Selected SA run has no valid subnet IDs")
    return unique_ids


def load_candidate_set(
    results_path: str,
    arch_path: str,
    experiment: str = None,
    target_n: int = None,
    target_lambda: float = None,
    target_seed: int = None,
) -> dict:
    with open(results_path) as f:
        results = json.load(f)

    arch_mapping = load_arch_mapping(arch_path)

    # New SA schema: top-level {meta, runs:[...]}
    if isinstance(results, dict) and "runs" in results:
        if target_n is None or target_lambda is None:
            raise ValueError("For SA 'runs' format, target_n and target_lambda are required")

        run, used_seed_fallback = _pick_run_from_sa_results(
            results, target_n=target_n, target_lambda=target_lambda, target_seed=target_seed
        )
        model_ids = _extract_ids_from_run(run)

        if used_seed_fallback:
            print(
                "    ⚠ Seed match not found; using fallback run "
                f"with seed={run.get('seed')} for N={target_n}, lambda={target_lambda}"
            )
        else:
            print(
                f"    ✓ Selected SA run: N={run.get('N')}, "
                f"lambda={run.get('lambda')}, seed={run.get('seed')}"
            )

        return {mid: arch_mapping[mid] for mid in model_ids if mid in arch_mapping}

    # Legacy schema: {experiment_name: {ids:[...]}}
    if experiment is None:
        raise ValueError("Legacy candidate-set format requires an experiment name")
    model_ids = results[experiment]["ids"]
    return {mid: arch_mapping[mid] for mid in model_ids if mid in arch_mapping}


def print_separator(title: str = "", width: int = 70):
    if title:
        pad = (width - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (width - pad - len(title) - 2))
    else:
        print("=" * width)


# ===========================================================================
# Main validation
# ===========================================================================
def main():
    args = parse_args()
    print_separator("Phase B POC: Derivation Extractor Validation")

    # ------------------------------------------------------------------
    print("\n[1] Loading OFA model...")
    t0 = time.time()
    ofa_net = OFADynamicResnetAllMod()
    checkpoint = torch.load(OFA_CHECKPOINT, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    ofa_net.load_state_dict(state_dict, strict=False)
    ofa_net.eval()
    print(f"    ✓ OFA model loaded in {time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    print("\n[2] Loading OFA weight pool from disk...")
    t0 = time.time()
    pool = load_ofa_pool(POOL_DIR)
    base_weights      = {k: pool["base_weights"][k] for k in pool["base_weights"]}
    transform_matrices = {k: pool["transform_matrices"][k] for k in pool["transform_matrices"]}
    print(f"    ✓ Pool loaded in {time.time()-t0:.1f}s  "
          f"({len(base_weights)} base weights, {len(transform_matrices)} transform matrices)")

    # ------------------------------------------------------------------
    print("\n[3] Loading candidate set...")

    # Prefer new SA schema file; fallback to legacy file/key if needed.
    try:
        all_archs = load_candidate_set(
            results_path=args.sa_results,
            arch_path=args.arch_file,
            target_n=args.n,
            target_lambda=args.lambda_value,
            target_seed=args.seed,
        )
    except Exception as sa_err:
        print(f"    ⚠ SA runs loader failed ({sa_err}); trying legacy format...")
        all_archs = load_candidate_set(
            results_path=args.legacy_candidate_set,
            arch_path=args.arch_file,
            experiment=args.legacy_experiment,
        )

    # Deterministic selection: first K IDs in source order.
    selected_ids = list(all_archs.keys())[: max(1, int(args.num_subnets))]
    poc_archs = {k: all_archs[k] for k in selected_ids}

    if len(poc_archs) < max(1, int(args.num_subnets)):
        print(
            f"    ⚠ Requested {args.num_subnets} subnets but only found {len(poc_archs)} "
            "matching entries in architecture mapping"
        )

    print(f"    ✓ Using subnets: {list(poc_archs.keys())}")

    # ------------------------------------------------------------------
    print("\n[4] Creating extractor...")
    extractor = OFADerivationExtractor(ofa_net, verbose=False)

    # ------------------------------------------------------------------
    total_layers = 0
    total_matched = 0
    all_results = {}
    global_max_diff = 0.0

    for subnet_id, arch in poc_archs.items():
        print_separator(f"Subnet: {subnet_id}")

        # --- Extract derivation metadata ---
        print(f"\n  Extracting derivation metadata...")
        t0 = time.time()
        derivations = extractor.extract_subnet_derivations(arch, INPUT_SHAPE)
        print(f"  ✓ Extracted {len(derivations)} layer derivations in {time.time()-t0:.1f}s")

        # Print summary
        have_transform = sum(1 for d in derivations if d.transform_sequence)
        have_split = sum(1 for d in derivations if d.decompose_type > 0)
        print(f"  Layers with kernel transform: {have_transform}/{len(derivations)}")
        print(f"  Layers with decomposed conv:  {have_split}/{len(derivations)}")
        print()

        # Print first 5 derivations
        print("  Sample derivations:")
        for i, d in enumerate(derivations[:5]):
            tm = d.transform_sequence if d.transform_sequence else "none"
            print(f"    [{i:2d}] {d.layer_path:<55s} "
                  f"out[{d.out_start}:{d.out_end}] (layer_total={d.layer_total_out_ch}) "
                  f"in[{d.in_start}:{d.in_end}] (layer_total={d.layer_total_in_ch}) "
                  f"ks={d.active_kernel_size}/{d.max_kernel_size} "
                  f"transforms={tm}")

        # --- Validate each derivation numerically ---
        print(f"\n  Validating derivations against OFA model...")
        results = validate_derivation_against_ofa(
            ofa_net=ofa_net,
            arch=arch,
            base_weights=base_weights,
            transform_matrices=transform_matrices,
            derivations=derivations,
            verbose=True,
        )

        total_layers  += results["total"]
        total_matched += results["matched"]
        global_max_diff = max(global_max_diff, results["max_abs_diff"])
        all_results[subnet_id] = results

        print(f"\n  ── Result for {subnet_id} ──")
        print(f"  Layers validated: {results['total']}")
        print(f"  Matched:          {results['matched']}/{results['total']}")
        print(f"  Mismatched:       {results['mismatched']}/{results['total']}")
        print(f"  Max |diff|:       {results['max_abs_diff']:.2e}")

        if results["mismatched"] == 0:
            print(f"  ✓ ALL LAYERS CORRECT")
        else:
            print(f"  ✗ SOME LAYERS INCORRECT – see details above")

        # --- Save derivations ---
        out_file = os.path.join(RESULTS_DIR, f"derivations_{subnet_id}.json")
        save_derivations(derivations, out_file)
        print(f"\n  Saved derivations → {out_file}")

    # ------------------------------------------------------------------
    print_separator("Overall Summary")
    print(f"Total subnets:         {len(poc_archs)}")
    print(f"Total layers checked:  {total_layers}")
    print(f"Total matched:         {total_matched}/{total_layers}")
    print(f"Global max |diff|:     {global_max_diff:.2e}")

    all_passed = all(r["mismatched"] == 0 for r in all_results.values())
    if all_passed:
        print("\n✓ PHASE B POC VALIDATION PASSED")
        print("  → OFADerivationExtractor correctly captures all derivation metadata")
        print("  → derive_weight_numpy() matches OFA model weights to floating-point precision")
        print("  → Ready to proceed to Step 3: Relay graph construction with pool variables")
    else:
        print("\n✗ PHASE B POC VALIDATION FAILED")
        print("  Some layers did not match – check details above")

    # ------------------------------------------------------------------
    # Save summary
    summary = {
        "total_subnets": len(poc_archs),
        "total_layers": total_layers,
        "total_matched": total_matched,
        "global_max_abs_diff": global_max_diff,
        "all_passed": all_passed,
        "per_subnet": {
            sid: {
                "total": r["total"],
                "matched": r["matched"],
                "mismatched": r["mismatched"],
                "max_abs_diff": r["max_abs_diff"],
            }
            for sid, r in all_results.items()
        },
    }
    import json
    summary_path = os.path.join(RESULTS_DIR, "poc_validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")
    print_separator()


def parse_args():
    parser = argparse.ArgumentParser(description="Phase B POC derivation validation")
    parser.add_argument("--sa-results", default=SA_RESULTS_FILE, help="Path to SA results json")
    parser.add_argument("--arch-file", default=ARCH_FILE, help="Path to architectures json")
    parser.add_argument("--n", type=int, default=TARGET_N, help="Target N in SA runs")
    parser.add_argument("--lambda", dest="lambda_value", type=float, default=TARGET_LAMBDA,
                        help="Target lambda in SA runs")
    parser.add_argument("--seed", type=int, default=TARGET_SEED,
                        help="Target seed in SA runs")
    parser.add_argument("--num-subnets", type=int, default=POC_NUM_SUBNETS,
                        help="How many subnet IDs to validate")
    parser.add_argument("--legacy-candidate-set", default=CANDIDATE_SET,
                        help="Legacy candidate set json path")
    parser.add_argument("--legacy-experiment", default=EXPERIMENT,
                        help="Legacy experiment key")
    return parser.parse_args()


if __name__ == "__main__":
    main()

