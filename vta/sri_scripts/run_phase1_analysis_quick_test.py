"""
Quick Test for Phase 1 Analysis - Compiles only 3 models for fast testing

This is a trimmed-down version of run_phase1_analysis.py for quick validation.
"""

from __future__ import absolute_import, print_function

import os
import sys
import time
import json
import glob
from typing import Dict, Any, List, Optional

import numpy as np
import tvm
from tvm import rpc, autotvm

import vta

# Import from existing scripts (read-only)
from execute_candidate_set_refactored import (
    Config,
    CompiledModel,
    setup_external_imports,
    load_ofa_model,
    setup_rpc_connection,
    load_candidate_set_from_experiments,
    compile_model,
)

from multi_graph_analyzer import GraphJSONAnalyzer, print_analysis_report, save_analysis_json
from extended_multi_graph_merger import ExtendedMultiGraphMerger


# ==============================================================================
# Quick Test Configuration
# ==============================================================================

class QuickTestConfig(Config):
    """Quick test configuration - only 3 models."""

    def __init__(self):
        super().__init__()

        # Override defaults for quick test
        self.experiment_name = "sa_lam_2.0"
        self.num_models_to_test = None

        # Analysis-specific settings
        self.max_models_to_compile = 3  # QUICK TEST: Only 3 models
        self.analysis_output_dir = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/phase1_results"
        self.save_compiled_models = False

        # Create output directory if it doesn't exist
        os.makedirs(self.analysis_output_dir, exist_ok=True)


# ==============================================================================
# Phase 1 Analysis Functions
# ==============================================================================

def compile_models_for_analysis(
    model_ids: List[str],
    arch_mapping: Dict[str, Any],
    ofa_net,
    env,
    target,
    config: QuickTestConfig,
    schedule_log_files: List[str],
    max_models: Optional[int] = None
) -> Dict[str, CompiledModel]:
    """Compile models for Phase 1 analysis."""
    print(f"\n{'='*80}")
    print(f"PHASE 1: COMPILING MODELS FOR ANALYSIS (QUICK TEST)")
    print(f"{'='*80}")

    if max_models is not None:
        model_ids = model_ids[:max_models]
        print(f"Limiting to first {max_models} models for quick test")

    compiled_models = {}
    failed_models = []

    for idx, model_id in enumerate(model_ids, 1):
        print(f"\n[{idx}/{len(model_ids)}] Compiling {model_id}...")

        arch = arch_mapping[model_id]
        compiled = compile_model(
            model_id, arch, ofa_net, env, target, config, schedule_log_files
        )

        if compiled is not None:
            compiled_models[model_id] = compiled
            print(f"  ✓ Success ({len(compiled_models)}/{idx} compiled)")
        else:
            failed_models.append(model_id)
            print(f"  ✗ Failed ({len(failed_models)} failures so far)")

    print(f"\n{'='*80}")
    print(f"Compilation Summary:")
    print(f"  Total attempted: {len(model_ids)}")
    print(f"  Successfully compiled: {len(compiled_models)}")
    print(f"  Failed: {len(failed_models)}")
    if failed_models:
        print(f"  Failed IDs: {failed_models}")
    print(f"{'='*80}")

    return compiled_models


def analyze_compiled_models(
    compiled_models: Dict[str, CompiledModel],
    config: QuickTestConfig
) -> Dict[str, Any]:
    """Run Phase 1 analysis on compiled models."""
    print(f"\n{'='*80}")
    print(f"PHASE 1: MULTI-GRAPH ANALYSIS")
    print(f"{'='*80}")

    # Initialize analyzer
    analyzer = GraphJSONAnalyzer()

    # Analyze each compiled model
    print(f"\nAnalyzing {len(compiled_models)} graphs...")
    for idx, (model_id, compiled) in enumerate(compiled_models.items(), 1):
        print(f"  [{idx}/{len(compiled_models)}] Analyzing {model_id}...")
        analyzer.analyze_graph(
            graph_json=compiled.graph,
            model_id=model_id,
            params=compiled.params
        )

    # Perform multi-graph analysis
    print(f"\nPerforming cross-graph analysis...")
    multi_analysis = analyzer.analyze_multi_graph()

    # Print comprehensive report
    print(f"\n{'='*80}")
    print(f"ANALYSIS REPORT")
    print(f"{'='*80}")
    print_analysis_report(multi_analysis)

    # Save analysis to JSON
    output_file = os.path.join(config.analysis_output_dir, "quick_test_analysis_results.json")
    save_analysis_json(multi_analysis, output_file)
    print(f"\n✓ Analysis saved to: {output_file}")

    return multi_analysis


def merge_graphs_for_analysis(
    compiled_models: Dict[str, CompiledModel],
    config: QuickTestConfig
) -> None:
    """Run Phase 1 graph merging analysis."""
    print(f"\n{'='*80}")
    print(f"PHASE 1: MULTI-GRAPH MERGER ANALYSIS")
    print(f"{'='*80}")

    # Initialize merger
    merger = ExtendedMultiGraphMerger()

    # Add each graph
    print(f"\nMerging {len(compiled_models)} graphs...")
    for idx, (model_id, compiled) in enumerate(compiled_models.items(), 1):
        print(f"  [{idx}/{len(compiled_models)}] Adding {model_id}...")
        merger.add_graph(
            graph_json=compiled.graph,  # Note: compiled.graph contains graph JSON string
            graph_id=model_id,
            params=compiled.params
        )

    # Print summary
    print(f"\n{'='*80}")
    print(f"MERGER SUMMARY")
    print(f"{'='*80}")
    merger.print_summary()

    # Save merged structure
    output_file = os.path.join(config.analysis_output_dir, "quick_test_merged_structure.json")
    merger.save_merged_structure(output_file)
    print(f"\n✓ Merged structure saved to: {output_file}")


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Run quick test Phase 1 analysis on 3 OFA models."""

    print("="*80)
    print("PHASE 1 QUICK TEST: MULTI-GRAPH RUNTIME ANALYSIS")
    print("Testing with 3 models from sa_lam_2.0")
    print("="*80)

    # Initialize configuration
    config = QuickTestConfig()

    # Setup TVM/VTA environment
    assert tvm.runtime.enabled("rpc")
    env = vta.get_env()
    target = env.target if config.device == "vta" else env.target_vta_cpu

    # Setup external imports
    print("\nSetting up external imports...")
    OFADynamicResnetAllMod, StaticResNetFromArch = setup_external_imports(config.external_repo_root)

    # Load OFA model
    print("\nLoading OFA model...")
    ofa_net = load_ofa_model(config.model_path)

    # Setup RPC connection
    print("\nSetting up RPC connection...")
    remote = setup_rpc_connection(env, config)

    # Load schedule logs
    schedule_log_files = glob.glob(config.schedule_log_dir)
    print(f"Loaded {len(schedule_log_files)} schedule logs")

    # Load candidate set
    print("\nLoading candidate set...")
    arch_mapping, model_ids = load_candidate_set_from_experiments(
        config.candidate_set_json,
        config.arch_config_json,
        config.experiment_name
    )

    print(f"Found {len(model_ids)} models in experiment '{config.experiment_name}'")
    print(f"Quick test will compile only {config.max_models_to_compile} models")

    # Compile models for analysis
    compiled_models = compile_models_for_analysis(
        model_ids,
        arch_mapping,
        ofa_net,
        env,
        target,
        config,
        schedule_log_files,
        max_models=config.max_models_to_compile
    )

    if not compiled_models:
        print("\n❌ ERROR: No models compiled successfully. Cannot proceed with analysis.")
        return

    # Run Phase 1 Analysis
    multi_analysis = analyze_compiled_models(compiled_models, config)

    # Run Phase 1 Merging
    merge_graphs_for_analysis(compiled_models, config)

    # Final summary
    print(f"\n{'='*80}")
    print(f"PHASE 1 QUICK TEST COMPLETE")
    print(f"{'='*80}")
    print(f"\n✓ Analyzed {len(compiled_models)} models from experiment '{config.experiment_name}'")
    print(f"\n✓ Results saved to: {config.analysis_output_dir}/")
    print(f"   - quick_test_analysis_results.json")
    print(f"   - quick_test_merged_structure.json")
    print(f"\n📊 This was a quick test with {config.max_models_to_compile} models.")
    print(f"   To run full analysis with all 25 models, use run_phase1_analysis.py")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()


