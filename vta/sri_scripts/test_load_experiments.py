#!/usr/bin/env python3
"""
Test script to demonstrate loading candidate sets from experiment results.

This script shows how to use the new load_candidate_set_from_experiments() function
to load architecture configurations from the experiment results file.
"""

import json
from typing import Dict, Any

# Path configuration
experiment_results_path = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/candidate_sets_results_all_expts.json"
arch_config_json_path = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json"


def load_arch_mapping(path: str) -> Dict[str, Any]:
    """Load architectures JSON and normalize to a mapping {id: architecture_dict}."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Case 1: already a mapping from id -> arch
    if isinstance(data, dict) and all(isinstance(v, dict) and 'residual_depth_list' in v for v in data.values()):
        return data

    # Case 2: top-level dict with 'architectures' list
    if isinstance(data, dict) and 'architectures' in data and isinstance(data['architectures'], list):
        mapping = {}
        for item in data['architectures']:
            if 'id' in item and 'architecture' in item:
                mapping[item['id']] = item['architecture']
            elif 'id' in item and 'arch' in item:
                mapping[item['id']] = item['arch']
            else:
                if 'id' in item:
                    mapping[item['id']] = item
        return mapping

    # Case 3: top-level list of architecture items
    if isinstance(data, list):
        mapping = {}
        for idx, item in enumerate(data):
            if isinstance(item, dict) and 'id' in item and 'architecture' in item:
                mapping[item['id']] = item['architecture']
            elif isinstance(item, dict) and 'id' in item:
                mapping[item['id']] = item
            else:
                mapping[f'arch_{idx}'] = item
        return mapping

    raise ValueError(f"Unsupported architecture file format: {path}")


def load_candidate_set_from_experiments(results_path: str, arch_path: str, experiment_name: str) -> tuple:
    """Load candidate set and architectures from experiment results file.

    Args:
        results_path: Path to the experiment results JSON (e.g., candidate_sets_results_all_expts.json)
        arch_path: Path to the architectures JSON file
        experiment_name: Name of the experiment to load (e.g., 'greedy_swap', 'ilp_tau_0.18757952189126595',
                        'sa_lam_2.0', 'sa_lam_1.0', 'ga')

    Returns:
        Tuple of (arch_mapping, model_ids)
    """
    # Load the experiment results
    with open(results_path, 'r') as f:
        results = json.load(f)

    if experiment_name not in results:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. "
            f"Available experiments: {list(results.keys())}"
        )

    # Get the IDs from the selected experiment
    model_ids = results[experiment_name]['ids']

    print(f"Loading experiment '{experiment_name}':")
    print(f"  - {len(model_ids)} models selected")
    print(f"  - Mean accuracy: {results[experiment_name].get('mean_acc', 'N/A')}")
    print(f"  - ASPT: {results[experiment_name].get('ASPT', 'N/A')}")
    print(f"  - Max pairwise: {results[experiment_name].get('max_pairwise', 'N/A')}")

    # Load all architectures
    arch_mapping = load_arch_mapping(arch_path)

    # Filter to only include the selected model IDs
    filtered_arch_mapping = {
        model_id: arch_mapping[model_id]
        for model_id in model_ids
        if model_id in arch_mapping
    }

    if len(filtered_arch_mapping) != len(model_ids):
        missing = set(model_ids) - set(filtered_arch_mapping.keys())
        print(f"Warning: {len(missing)} model IDs not found in architecture file")

    return filtered_arch_mapping, model_ids


def main():
    """Test the loading functions with different experiment types."""

    # List available experiments
    print("=" * 80)
    print("Available Experiments")
    print("=" * 80)
    with open(experiment_results_path, 'r') as f:
        results = json.load(f)

    for exp_name in results.keys():
        exp_data = results[exp_name]
        print(f"\n{exp_name}:")
        print(f"  Models: {exp_data.get('size', len(exp_data.get('ids', [])))}")
        print(f"  Mean Accuracy: {exp_data.get('mean_acc', 'N/A'):.4f}" if isinstance(exp_data.get('mean_acc'), float) else f"  Mean Accuracy: N/A")
        print(f"  ASPT: {exp_data.get('ASPT', 'N/A'):.4f}" if isinstance(exp_data.get('ASPT'), float) else f"  ASPT: N/A")

    # Test loading a specific experiment
    print("\n" + "=" * 80)
    print("Loading 'greedy_swap' Experiment")
    print("=" * 80)

    arch_mapping, model_ids = load_candidate_set_from_experiments(
        experiment_results_path,
        arch_config_json_path,
        'greedy_swap'
    )

    print(f"\nSuccessfully loaded {len(arch_mapping)} architectures")
    print(f"Model IDs: {model_ids[:5]}..." if len(model_ids) > 5 else f"Model IDs: {model_ids}")

    # Show a sample architecture
    if model_ids:
        sample_id = model_ids[0]
        print(f"\nSample architecture (ID: {sample_id}):")
        arch = arch_mapping[sample_id]
        for key in ['residual_depth_list', 'out_channel_setting_list', 'decomp_type_list']:
            if key in arch:
                print(f"  {key}: {arch[key]}")


if __name__ == "__main__":
    main()

