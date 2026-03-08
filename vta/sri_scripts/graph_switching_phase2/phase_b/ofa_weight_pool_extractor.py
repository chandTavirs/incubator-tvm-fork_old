"""
Step 1: OFA Weight Pool Extraction
===================================

Extract base weights and transform matrices from OFA checkpoint.
This will be uploaded ONCE to VTA device and shared across all subnets.
"""

from __future__ import absolute_import, print_function

import os
import sys
import json
import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, Any, List, Tuple

# Add paths
EXTERNAL_REPO_ROOT = "/home/srchand/Desktop/research/OFA_Obfs"
sys.path.insert(0, EXTERNAL_REPO_ROOT)

from ofa_base_models import OFADynamicResnetAllMod


def extract_ofa_weight_pool(checkpoint_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Extract OFA base weights and transform matrices from checkpoint.

    Args:
        checkpoint_path: Path to OFA checkpoint (.pth file)
        verbose: Print extraction progress

    Returns:
        {
            "base_weights": {
                "first_conv.weight": np.ndarray,
                "blocks.0.conv.weight": np.ndarray,
                ...
            },
            "transform_matrices": {
                "7to5_matrix": np.ndarray,
                "5to3_matrix": np.ndarray,
                ...
            },
            "bn_params": {
                "first_bn.weight": np.ndarray,
                "first_bn.bias": np.ndarray,
                ...
            },
            "other_params": {
                "fc.weight": np.ndarray,
                "fc.bias": np.ndarray,
                ...
            },
            "metadata": {
                "total_base_weights": int,
                "total_transform_matrices": int,
                "total_params": int,
                "total_bytes": int,
                "base_weights_bytes": int,
                "transform_matrices_bytes": int,
            }
        }
    """
    if verbose:
        print("="*70)
        print("OFA Weight Pool Extraction")
        print("="*70)
        print(f"Checkpoint: {checkpoint_path}\n")

    # Load OFA model
    if verbose:
        print("Loading OFA model...")

    ofa_net = OFADynamicResnetAllMod()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    ofa_net.load_state_dict(state_dict, strict=False)
    ofa_net.eval()

    if verbose:
        print(f"  ✓ Loaded {len(state_dict)} parameters\n")

    # Initialize containers
    base_weights = OrderedDict()
    transform_matrices = OrderedDict()
    bn_params = OrderedDict()
    other_params = OrderedDict()

    total_params = 0
    total_bytes = 0
    base_weights_bytes = 0
    transform_matrices_bytes = 0

    if verbose:
        print("Extracting weights by category...\n")

    # Categorize and extract parameters
    for name, param in ofa_net.named_parameters():
        if param.dim() == 0:  # Skip scalars
            continue

        param_np = param.detach().cpu().numpy()
        param_bytes = param_np.nbytes
        total_params += 1
        total_bytes += param_bytes

        # Category 1: Transform matrices (for kernel size adaptation)
        if 'to' in name and 'matrix' in name:
            # e.g., "blocks.0.mobile_inverted_conv.depth_conv.conv.7to5_matrix"
            # Extract transform name: "7to5_matrix"
            parts = name.split('.')
            matrix_name = None
            for part in parts:
                if 'to' in part and 'matrix' in part:
                    matrix_name = part
                    break

            if matrix_name:
                # Use unique key combining layer path + matrix name
                key = name.replace('.', '_')
                transform_matrices[key] = param_np
                transform_matrices_bytes += param_bytes

                if verbose:
                    print(f"  [TRANSFORM] {name[:60]:60s} {str(param_np.shape):20s} {param_bytes/1024:8.2f} KB")

        # Category 2: Base conv weights (from DynamicConv2DAll.base_conv)
        elif 'base_conv.weight' in name or ('conv.weight' in name and 'depth_conv' not in name):
            # Main convolution base weights
            key = name.replace('.', '_')
            base_weights[key] = param_np
            base_weights_bytes += param_bytes

            if verbose:
                print(f"  [BASE_CONV] {name[:60]:60s} {str(param_np.shape):20s} {param_bytes/1024:8.2f} KB")

        # Category 3: BatchNorm parameters
        elif any(bn_key in name for bn_key in ['bn', 'batch_norm', 'running_mean', 'running_var']):
            key = name.replace('.', '_')
            bn_params[key] = param_np

            if verbose:
                print(f"  [BN       ] {name[:60]:60s} {str(param_np.shape):20s} {param_bytes/1024:8.2f} KB")

        # Category 4: Other parameters (FC, biases, etc.)
        else:
            key = name.replace('.', '_')
            other_params[key] = param_np

            if verbose:
                print(f"  [OTHER    ] {name[:60]:60s} {str(param_np.shape):20s} {param_bytes/1024:8.2f} KB")

    # Also extract buffers (running_mean, running_var if not in parameters)
    for name, buffer in ofa_net.named_buffers():
        if buffer.dim() == 0:
            continue

        if name not in [p_name for p_name, _ in ofa_net.named_parameters()]:
            buffer_np = buffer.detach().cpu().numpy()
            buffer_bytes = buffer_np.nbytes
            total_bytes += buffer_bytes

            key = name.replace('.', '_')

            if any(bn_key in name for bn_key in ['running_mean', 'running_var', 'num_batches_tracked']):
                bn_params[key] = buffer_np
                if verbose:
                    print(f"  [BN_BUF   ] {name[:60]:60s} {str(buffer_np.shape):20s} {buffer_bytes/1024:8.2f} KB")
            else:
                other_params[key] = buffer_np
                if verbose:
                    print(f"  [OTHER_BUF] {name[:60]:60s} {str(buffer_np.shape):20s} {buffer_bytes/1024:8.2f} KB")

    # Build metadata
    metadata = {
        "total_base_weights": len(base_weights),
        "total_transform_matrices": len(transform_matrices),
        "total_bn_params": len(bn_params),
        "total_other_params": len(other_params),
        "total_params": total_params,
        "total_bytes": total_bytes,
        "base_weights_bytes": base_weights_bytes,
        "transform_matrices_bytes": transform_matrices_bytes,
    }

    if verbose:
        print("\n" + "="*70)
        print("Extraction Summary")
        print("="*70)
        print(f"  Base conv weights:     {len(base_weights):4d}  ({base_weights_bytes/1e6:8.2f} MB)")
        print(f"  Transform matrices:    {len(transform_matrices):4d}  ({transform_matrices_bytes/1e6:8.2f} MB)")
        print(f"  BatchNorm params:      {len(bn_params):4d}")
        print(f"  Other params:          {len(other_params):4d}")
        print(f"  Total parameters:      {total_params:4d}")
        print(f"  Total size:            {total_bytes/1e6:8.2f} MB")
        print("="*70)

    return {
        "base_weights": base_weights,
        "transform_matrices": transform_matrices,
        "bn_params": bn_params,
        "other_params": other_params,
        "metadata": metadata,
    }


def save_ofa_pool(pool: Dict[str, Any], output_dir: str):
    """
    Save OFA weight pool to disk for reuse.

    Args:
        pool: Output from extract_ofa_weight_pool()
        output_dir: Directory to save pool files
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving OFA pool to {output_dir}/...")

    # Save each category as separate .npz file
    for category in ["base_weights", "transform_matrices", "bn_params", "other_params"]:
        if category in pool and pool[category]:
            npz_path = os.path.join(output_dir, f"{category}.npz")
            np.savez_compressed(npz_path, **pool[category])
            print(f"  ✓ Saved {category} → {npz_path}")

    # Save metadata as JSON
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(pool["metadata"], f, indent=2)
    print(f"  ✓ Saved metadata → {metadata_path}")

    print("Done!\n")


def load_ofa_pool(pool_dir: str) -> Dict[str, Any]:
    """
    Load saved OFA weight pool from disk.

    Args:
        pool_dir: Directory containing saved pool files

    Returns:
        Same format as extract_ofa_weight_pool()
    """
    print(f"Loading OFA pool from {pool_dir}/...")

    pool = {}

    for category in ["base_weights", "transform_matrices", "bn_params", "other_params"]:
        npz_path = os.path.join(pool_dir, f"{category}.npz")
        if os.path.exists(npz_path):
            data = np.load(npz_path)
            pool[category] = OrderedDict((k, data[k]) for k in data.files)
            print(f"  ✓ Loaded {category}: {len(pool[category])} tensors")
        else:
            pool[category] = OrderedDict()

    # Load metadata
    metadata_path = os.path.join(pool_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            pool["metadata"] = json.load(f)
        print(f"  ✓ Loaded metadata")
    else:
        pool["metadata"] = {}

    print("Done!\n")
    return pool


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract OFA weight pool from checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth",
        help="Path to OFA checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ofa_weight_pool",
        help="Directory to save extracted pool"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save to disk (just extract and print stats)"
    )

    args = parser.parse_args()

    # Extract pool
    pool = extract_ofa_weight_pool(args.checkpoint, verbose=True)

    # Save if requested
    if not args.no_save:
        save_ofa_pool(pool, args.output_dir)

    # Print sample entries
    print("\nSample Base Weights:")
    for i, (name, weight) in enumerate(pool["base_weights"].items()):
        if i >= 5:
            break
        print(f"  {name}: {weight.shape}")

    print("\nSample Transform Matrices:")
    for i, (name, matrix) in enumerate(pool["transform_matrices"].items()):
        if i >= 5:
            break
        print(f"  {name}: {matrix.shape}")

