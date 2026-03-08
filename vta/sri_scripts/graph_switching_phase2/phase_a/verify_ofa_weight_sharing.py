"""
Phase A Enhanced: OFA-Level Weight Sharing Analysis
====================================================

Purpose
-------
The previous verify_param_identity.py showed only 2.9% savings because it compared
post-quantization, post-packing VTA tensors. These have different shapes due to
channel packing and quantization.

This script analyzes weight sharing at the **OFA/PyTorch level** (pre-quantization)
to determine the TRUE sharing potential:

  Q1. How many unique float32 weight tensors exist across all subnets at the PyTorch level?
  Q2. What is the theoretical memory savings if we deduplicate at the OFA level?
  Q3. Can we map each subnet layer back to its source OFA layer configuration?

Strategy
--------
  1. For each subnet, extract the StaticResNet model BEFORE quantization
  2. Get the state_dict() — these are the transformed float32 weights
  3. For each weight tensor, track:
     - Layer name (e.g., "blocks.0.mobile_inverted_conv.depth_conv.conv.weight")
     - Shape
     - Source OFA configuration (from arch dict)
     - Hash of the tensor
  4. Build a mapping: OFA_layer_config → set of subnets that use it
  5. Calculate true deduplication savings

Outputs
-------
  phase_a/results/
    ofa_weight_sharing_report.json    - OFA-level deduplication analysis
    layer_config_mapping.json         - Maps OFA configs to subnet layers
    OFA_WEIGHT_SHARING_SUMMARY.md     - Human-readable summary

Usage
-----
  cd .../graph_switching_phase2/phase_a
  python verify_ofa_weight_sharing.py --num_subnets 3   # quick test
  python verify_ofa_weight_sharing.py --num_subnets 25  # full analysis
"""

from __future__ import absolute_import, print_function

import os
import sys
import json
import time
import hashlib
import argparse
from typing import Dict, Any, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_SRI_SCRIPTS = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_VTA_DIR     = os.path.abspath(os.path.join(_SRI_SCRIPTS, ".."))
_VTA_PYTHON  = os.path.abspath(os.path.join(_VTA_DIR, "python"))
_TVM_DIR     = os.path.abspath(os.path.join(_VTA_DIR, ".."))
_TVM_PYTHON  = os.path.abspath(os.path.join(_TVM_DIR, "python"))

EXTERNAL_REPO_ROOT = "/home/srchand/Desktop/research/OFA_Obfs"
OFA_CHECKPOINT     = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth"
CANDIDATE_SET_JSON = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/candidate_sets_results_all_expts.json"
ARCH_CONFIG_JSON   = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json"
EXPERIMENT_NAME    = "sa_lam_2.0"

RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")

for p in [_VTA_PYTHON, _TVM_PYTHON, _SRI_SCRIPTS, EXTERNAL_REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Utility: array hashing
# ---------------------------------------------------------------------------

def hash_array(arr: np.ndarray) -> str:
    """SHA-256 of the raw bytes of a numpy array."""
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff arrays have same shape, dtype, and every element is equal."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Layer configuration extractor
# ---------------------------------------------------------------------------

@dataclass
class LayerConfig:
    """Normalized layer configuration for OFA weight sharing."""
    layer_type: str          # "conv", "depth_conv", "point_conv", "fc", etc.
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    groups: int = 1
    decomp_type: Optional[str] = None  # For decomposed layers
    block_idx: Optional[int] = None
    stage_idx: Optional[int] = None

    def to_key(self) -> str:
        """Unique key for this configuration."""
        parts = [
            self.layer_type,
            f"in{self.in_channels}",
            f"out{self.out_channels}",
            f"k{self.kernel_size}",
            f"s{self.stride}",
            f"g{self.groups}",
        ]
        if self.decomp_type:
            parts.append(f"decomp_{self.decomp_type}")
        if self.block_idx is not None:
            parts.append(f"blk{self.block_idx}")
        if self.stage_idx is not None:
            parts.append(f"stg{self.stage_idx}")
        return "_".join(parts)


def extract_layer_config_from_arch(arch: Dict[str, Any]) -> Dict[str, LayerConfig]:
    """
    Parse the arch dict to extract layer configurations.

    Returns:
        dict: {layer_name: LayerConfig}
    """
    configs = {}

    # First conv (stem)
    configs["first_conv"] = LayerConfig(
        layer_type="conv",
        in_channels=3,
        out_channels=arch.get("first_conv", {}).get("out_channels", 32),
        kernel_size=7,
        stride=2,
        groups=1
    )

    # Blocks
    blocks = arch.get("blocks", [])
    for blk_idx, block in enumerate(blocks):
        mobile_inverted_conv = block.get("mobile_inverted_conv", {})

        # Inverted bottleneck
        in_ch = mobile_inverted_conv.get("in_channels", 0)
        out_ch = mobile_inverted_conv.get("out_channels", 0)
        expand_ratio = mobile_inverted_conv.get("expand_ratio", 1)
        kernel = mobile_inverted_conv.get("kernel_size", 3)
        stride = mobile_inverted_conv.get("stride", 1)

        mid_ch = int(in_ch * expand_ratio)

        # Inverted bottleneck: point conv (expand)
        if expand_ratio > 1:
            configs[f"blocks.{blk_idx}.inverted_bottleneck"] = LayerConfig(
                layer_type="point_conv",
                in_channels=in_ch,
                out_channels=mid_ch,
                kernel_size=1,
                stride=1,
                groups=1,
                block_idx=blk_idx
            )

        # Depth conv
        decomp = mobile_inverted_conv.get("decomp_type", None)
        if decomp:
            configs[f"blocks.{blk_idx}.depth_conv"] = LayerConfig(
                layer_type="depth_conv_decomposed",
                in_channels=mid_ch,
                out_channels=mid_ch,
                kernel_size=kernel,
                stride=stride,
                groups=mid_ch,
                decomp_type=decomp,
                block_idx=blk_idx
            )
        else:
            configs[f"blocks.{blk_idx}.depth_conv"] = LayerConfig(
                layer_type="depth_conv",
                in_channels=mid_ch,
                out_channels=mid_ch,
                kernel_size=kernel,
                stride=stride,
                groups=mid_ch,
                block_idx=blk_idx
            )

        # Point conv (project)
        configs[f"blocks.{blk_idx}.point_linear"] = LayerConfig(
            layer_type="point_conv",
            in_channels=mid_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            groups=1,
            block_idx=blk_idx
        )

    # Final expand + classifier
    final_expand = arch.get("final_expand_layer", {})
    configs["final_expand"] = LayerConfig(
        layer_type="conv",
        in_channels=final_expand.get("in_channels", 512),
        out_channels=final_expand.get("out_channels", 1024),
        kernel_size=1,
        stride=1,
        groups=1
    )

    classifier = arch.get("classifier", {})
    configs["classifier"] = LayerConfig(
        layer_type="fc",
        in_channels=classifier.get("in_features", 1024),
        out_channels=classifier.get("out_features", 10),
        kernel_size=1,
        stride=1,
        groups=1
    )

    return configs


# ---------------------------------------------------------------------------
# Build static models and extract weights
# ---------------------------------------------------------------------------

def build_static_model_and_extract_weights(
    model_id: str,
    arch: Dict[str, Any],
    ofa_net,
    StaticResNetFromArch_cls,
    torch_mod,
) -> Optional[Dict[str, Any]]:
    """
    Build a StaticResNet from arch, load transformed weights from OFA,
    and extract the float32 state_dict.

    Returns:
        {
            "model_id": str,
            "state_dict": {layer_name: np.ndarray},
            "layer_configs": {layer_name: LayerConfig},
            "total_params": int,
            "total_bytes": int
        }
    """
    try:
        print(f"  Building {model_id}...")
        t0 = time.time()

        # Set active subnet in OFA
        ofa_net.set_active_subnet(arch)

        # Build static model
        static_net = StaticResNetFromArch_cls(
            target_arch=arch,
            num_classes=10,
            width_mult_list=(0.5, 1.0, 2.0)
        )

        # Load transformed weights from OFA checkpoint
        static_net.load_weights_from_ofa_checkpoint(
            checkpoint_path=OFA_CHECKPOINT,
            ofa_model=ofa_net
        )
        static_net.eval()

        # Extract state_dict (float32 weights)
        state_dict_torch = static_net.state_dict()
        state_dict_np = {
            k: v.detach().cpu().numpy()
            for k, v in state_dict_torch.items()
            if len(v.shape) > 0  # Skip scalar buffers
        }

        # Extract layer configs from arch
        layer_configs = extract_layer_config_from_arch(arch)

        total_params = sum(v.size for v in state_dict_np.values())
        total_bytes  = sum(v.nbytes for v in state_dict_np.values())

        elapsed = time.time() - t0
        print(f"    ✓ {len(state_dict_np)} weight tensors, "
              f"{total_bytes/1e6:.2f} MB, {elapsed:.1f}s")

        return {
            "model_id": model_id,
            "state_dict": state_dict_np,
            "layer_configs": layer_configs,
            "total_params": total_params,
            "total_bytes": total_bytes,
        }

    except Exception as e:
        import traceback
        print(f"    ✗ Failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# OFA-level weight sharing analysis
# ---------------------------------------------------------------------------

def analyze_ofa_weight_sharing(
    subnet_weights: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    Analyze weight sharing at the OFA/PyTorch level (pre-quantization).

    Returns detailed deduplication statistics.
    """
    print(f"\n{'='*70}")
    print("OFA-Level Weight Sharing Analysis")
    print(f"{'='*70}")

    # Step 1: Hash every weight tensor from every subnet
    # hash → list of (subnet_id, layer_name, shape, bytes)
    hash_to_entries = defaultdict(list)

    # config_key → list of (subnet_id, layer_name, hash)
    config_to_entries = defaultdict(list)

    total_tensors = 0
    total_bytes = 0

    for subnet_id, data in subnet_weights.items():
        if data is None:
            continue

        state_dict = data["state_dict"]
        layer_configs = data["layer_configs"]

        for layer_name, weight_arr in state_dict.items():
            h = hash_array(weight_arr)
            hash_to_entries[h].append({
                "subnet_id": subnet_id,
                "layer_name": layer_name,
                "shape": list(weight_arr.shape),
                "dtype": str(weight_arr.dtype),
                "bytes": weight_arr.nbytes,
            })

            # Try to match layer_name to a config
            matched_config = None
            for cfg_name, cfg in layer_configs.items():
                if cfg_name in layer_name:
                    matched_config = cfg.to_key()
                    break

            if matched_config:
                config_to_entries[matched_config].append({
                    "subnet_id": subnet_id,
                    "layer_name": layer_name,
                    "hash": h,
                    "shape": list(weight_arr.shape),
                    "bytes": weight_arr.nbytes,
                })

            total_tensors += 1
            total_bytes += weight_arr.nbytes

    unique_hashes = len(hash_to_entries)
    shared_hashes = sum(1 for entries in hash_to_entries.values() if len(entries) > 1)
    private_hashes = unique_hashes - shared_hashes

    unique_bytes = sum(entries[0]["bytes"] for entries in hash_to_entries.values())
    savings_bytes = total_bytes - unique_bytes
    savings_percent = 100.0 * savings_bytes / total_bytes if total_bytes > 0 else 0.0

    print(f"\n  Subnets analyzed:          {len(subnet_weights)}")
    print(f"  Total weight tensors:      {total_tensors}")
    print(f"  Unique weight tensors:     {unique_hashes}")
    print(f"  Shared tensors (>1 subnet): {shared_hashes}")
    print(f"  Private tensors:           {private_hashes}")
    print(f"\n  Total weight bytes:        {total_bytes/1e6:.2f} MB")
    print(f"  Unique weight bytes:       {unique_bytes/1e6:.2f} MB")
    print(f"  **Memory savings:          {savings_bytes/1e6:.2f} MB  ({savings_percent:.1f}%)**")

    # Per-subnet stats
    subnet_stats = {}
    for subnet_id, data in subnet_weights.items():
        if data is None:
            continue
        state_dict = data["state_dict"]
        shared_count = 0
        private_count = 0
        for layer_name, arr in state_dict.items():
            h = hash_array(arr)
            if len(hash_to_entries[h]) > 1:
                shared_count += 1
            else:
                private_count += 1

        subnet_stats[subnet_id] = {
            "total_tensors": len(state_dict),
            "shared_tensors": shared_count,
            "private_tensors": private_count,
            "sharing_pct": 100.0 * shared_count / len(state_dict) if state_dict else 0.0,
            "total_bytes": data["total_bytes"],
        }

    # Top shared tensors
    sorted_by_sharing = sorted(
        hash_to_entries.items(),
        key=lambda kv: (len(kv[1]), kv[1][0]["bytes"]),
        reverse=True
    )

    top_shared = []
    for h, entries in sorted_by_sharing[:20]:
        # Get a representative layer name
        layer_names = [e["layer_name"] for e in entries]
        top_shared.append({
            "hash": h[:12],
            "shape": entries[0]["shape"],
            "dtype": entries[0]["dtype"],
            "bytes": entries[0]["bytes"],
            "num_subnets": len(entries),
            "subnets": [e["subnet_id"] for e in entries],
            "layer_names": layer_names[:3],  # First 3 for brevity
        })

    print(f"\n  Top shared weight tensors:")
    for i, t in enumerate(top_shared[:10], 1):
        print(f"    {i:2d}. shape={str(t['shape']):25s}  "
              f"used by {t['num_subnets']} subnets  "
              f"({t['bytes']/1024:.1f} KB each)  "
              f"layer: {t['layer_names'][0]}")

    # Config-level sharing
    print(f"\n  Layer config analysis:")
    print(f"    Unique layer configs: {len(config_to_entries)}")
    config_sharing = []
    for cfg_key, entries in config_to_entries.items():
        unique_hashes_in_cfg = len(set(e["hash"] for e in entries))
        config_sharing.append({
            "config": cfg_key,
            "num_instances": len(entries),
            "unique_weights": unique_hashes_in_cfg,
            "sharing_pct": 100.0 * (len(entries) - unique_hashes_in_cfg) / len(entries) if entries else 0.0,
        })

    config_sharing.sort(key=lambda x: x["num_instances"], reverse=True)
    for i, cs in enumerate(config_sharing[:5], 1):
        print(f"      {i}. {cs['config'][:60]:60s}  "
              f"instances={cs['num_instances']:3d}  "
              f"unique_weights={cs['unique_weights']:3d}  "
              f"sharing={cs['sharing_pct']:.1f}%")

    return {
        "total_tensors": total_tensors,
        "unique_tensors": unique_hashes,
        "shared_tensors": shared_hashes,
        "private_tensors": private_hashes,
        "total_bytes": total_bytes,
        "unique_bytes": unique_bytes,
        "savings_bytes": savings_bytes,
        "savings_percent": savings_percent,
        "subnet_stats": subnet_stats,
        "top_shared_tensors": top_shared,
        "config_sharing": config_sharing,
        "hash_to_entries": {
            h: entries
            for h, entries in sorted_by_sharing
        },
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_ofa_markdown_summary(
    analysis: Dict[str, Any],
    output_path: str,
):
    lines = []
    lines.append("# Phase A Enhanced: OFA-Level Weight Sharing Analysis\n")
    lines.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n\n")

    lines.append("## Summary\n")
    lines.append("This analysis examines weight sharing at the **PyTorch/OFA level** ")
    lines.append("(before quantization and VTA packing) to determine the TRUE memory savings potential.\n\n")

    lines.append("## Results\n")
    lines.append("| Metric | Value |\n|---|---|\n")
    lines.append(f"| Subnets analyzed | {len(analysis.get('subnet_stats', {}))} |\n")
    lines.append(f"| Total weight tensors | {analysis['total_tensors']} |\n")
    lines.append(f"| Unique weight tensors | {analysis['unique_tensors']} |\n")
    lines.append(f"| Shared tensors (>1 subnet) | {analysis['shared_tensors']} |\n")
    lines.append(f"| Total weight bytes | {analysis['total_bytes']/1e6:.2f} MB |\n")
    lines.append(f"| Unique weight bytes | {analysis['unique_bytes']/1e6:.2f} MB |\n")
    lines.append(f"| **Memory savings** | **{analysis['savings_bytes']/1e6:.2f} MB ({analysis['savings_percent']:.1f}%)** |\n\n")

    lines.append("## Top 20 Most Shared Weight Tensors\n")
    lines.append("| Rank | Hash | Shape | dtype | Size (KB) | Used by N subnets | Example Layer |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for i, t in enumerate(analysis.get("top_shared_tensors", []), 1):
        layer = t['layer_names'][0] if t['layer_names'] else "?"
        lines.append(f"| {i} | `{t['hash']}` | {t['shape']} | {t['dtype']} | "
                     f"{t['bytes']/1024:.1f} | {t['num_subnets']} | `{layer}` |\n")
    lines.append("\n")

    lines.append("## Per-Subnet Sharing\n")
    lines.append("| Subnet | Total Tensors | Shared | Private | Sharing % | Total Size (MB) |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for sid, s in analysis.get("subnet_stats", {}).items():
        lines.append(f"| {sid} | {s['total_tensors']} | {s['shared_tensors']} | "
                     f"{s['private_tensors']} | {s['sharing_pct']:.1f}% | "
                     f"{s['total_bytes']/1e6:.2f} |\n")
    lines.append("\n")

    lines.append("## Layer Configuration Sharing\n")
    lines.append("Top layer configurations by instance count:\n\n")
    lines.append("| Rank | Configuration | Instances | Unique Weights | Sharing % |\n")
    lines.append("|---|---|---|---|---|\n")
    for i, cs in enumerate(analysis.get("config_sharing", [])[:10], 1):
        lines.append(f"| {i} | `{cs['config']}` | {cs['num_instances']} | "
                     f"{cs['unique_weights']} | {cs['sharing_pct']:.1f}% |\n")
    lines.append("\n")

    lines.append("## Interpretation\n")
    savings_pct = analysis['savings_percent']
    if savings_pct > 50:
        verdict = "✅ **EXCELLENT** — Over 50% memory savings possible with OFA-level deduplication."
    elif savings_pct > 20:
        verdict = "✅ **GOOD** — Significant memory savings possible."
    elif savings_pct > 5:
        verdict = "⚠️  **MODERATE** — Some savings possible, but may not justify complexity."
    else:
        verdict = "❌ **LOW** — Limited benefit from weight deduplication."

    lines.append(f"{verdict}\n\n")

    lines.append("## Next Steps for Option B\n")
    if savings_pct > 20:
        lines.append("1. **Proceed to Phase B**: Design the OFA weight pool + subnet runtime mapping\n")
        lines.append("2. Implement weight pool manager that stores unique tensors only\n")
        lines.append("3. Build subnet-to-pool mapping for runtime weight lookup\n")
        lines.append("4. Test end-to-end with quantization applied to pooled weights\n")
    else:
        lines.append("⚠️  Consider focusing on Option A (multi-runtime with separate parameters) ")
        lines.append("or optimizing other aspects like graph structure sharing.\n")

    with open(output_path, "w") as f:
        f.writelines(lines)
    print(f"\n  Markdown summary written → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase A Enhanced: Analyze OFA-level weight sharing (pre-quantization)"
    )
    parser.add_argument(
        "--num_subnets", type=int, default=3,
        help="Number of subnets to analyze (default=3 for quick test)"
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("PHASE A ENHANCED: OFA-Level Weight Sharing Analysis")
    print(f"Subnets to analyze: {args.num_subnets}")
    print("=" * 70)

    # --- Imports ---
    print("\nLoading imports...")
    for mod_name in ["ofa_base_models", "architecture_defense"]:
        existing = sys.modules.get(mod_name)
        if existing is not None:
            mod_file = getattr(existing, "__file__", "") or ""
            if "sri_scripts" in mod_file:
                del sys.modules[mod_name]

    import torch
    from ofa_base_models import OFADynamicResnetAllMod
    from architecture_defense import StaticResNetFromArch

    # --- Load OFA model ---
    print("\nLoading OFA model...")
    ofa_net = OFADynamicResnetAllMod()
    checkpoint = torch.load(OFA_CHECKPOINT, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint)
    ofa_net.load_state_dict(state, strict=False)
    ofa_net.eval()
    print("  ✓ OFA model loaded")

    # --- Load candidate set ---
    print("\nLoading candidate set...")
    sys.path.insert(0, _SRI_SCRIPTS)
    from execute_candidate_set_refactored import load_arch_mapping

    with open(CANDIDATE_SET_JSON) as f:
        results_data = json.load(f)
    model_ids_all = results_data[EXPERIMENT_NAME]["ids"]
    print(f"Loading experiment '{EXPERIMENT_NAME}': {len(model_ids_all)} models")

    arch_all = load_arch_mapping(ARCH_CONFIG_JSON)
    model_ids = [mid for mid in model_ids_all if mid in arch_all][:args.num_subnets]
    print(f"  Using {len(model_ids)} subnets: {model_ids}")

    # --- Build static models and extract weights ---
    print(f"\n{'='*70}")
    print(f"Building {len(model_ids)} static models and extracting weights...")
    print(f"{'='*70}")

    subnet_weights = {}
    for i, model_id in enumerate(model_ids):
        arch = arch_all[model_id]
        print(f"\n[{i+1}/{len(model_ids)}] {model_id}")
        result = build_static_model_and_extract_weights(
            model_id, arch, ofa_net, StaticResNetFromArch, torch
        )
        subnet_weights[model_id] = result

    successful = {k: v for k, v in subnet_weights.items() if v is not None}
    print(f"\n  Successfully processed: {len(successful)}/{len(model_ids)}")

    # --- OFA-level weight sharing analysis ---
    if len(successful) >= 2:
        analysis = analyze_ofa_weight_sharing(successful)

        # Save results
        analysis_save = {k: v for k, v in analysis.items() if k != "hash_to_entries"}
        report_path = os.path.join(RESULTS_DIR, "ofa_weight_sharing_report.json")
        with open(report_path, "w") as f:
            json.dump(analysis_save, f, indent=2, default=str)
        print(f"\n  Report saved → {report_path}")

        # Markdown summary
        md_path = os.path.join(RESULTS_DIR, "OFA_WEIGHT_SHARING_SUMMARY.md")
        write_ofa_markdown_summary(analysis, md_path)

        # Final summary
        print(f"\n{'='*70}")
        print("PHASE A ENHANCED COMPLETE")
        print(f"{'='*70}")
        print(f"  OFA-level memory savings:  {analysis['savings_percent']:.1f}%")
        print(f"  Unique tensors:            {analysis['unique_tensors']}")
        print(f"  Shared tensors:            {analysis['shared_tensors']}")
        print(f"\n  Results in:  {RESULTS_DIR}/")
        print(f"  Summary:     {md_path}")
    else:
        print("\n  ✗ Not enough successful builds for analysis")


if __name__ == "__main__":
    main()

