"""
Phase 1 Analysis: Multi-Graph Runtime Analysis with Real OFA Candidate Set

This script runs Phase 1 analysis on the actual OFA candidate set to:
1. Compile models from the candidate set (sa_lam_2.0 experiment)
2. Analyze graph structures and parameter sharing patterns
3. Generate feasibility report for multi-graph runtime implementation
4. Save analysis results for Phase 2 planning

This is a standalone script that doesn't modify any existing files.
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
# Phase 1 Configuration
# ==============================================================================

class Phase1Config(Config):
    """Extended configuration for Phase 1 analysis."""

    def __init__(self):
        super().__init__()

        # Override defaults for Phase 1 analysis
        self.experiment_name = "sa_lam_2.0"
        self.num_models_to_test = None  # Analyze all models in the experiment

        # Analysis-specific settings
        self.max_models_to_compile = 25  # Limit for compilation (can be None for all)
        self.analysis_output_dir = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/phase1_results"
        self.save_compiled_models = False  # Don't save compiled artifacts

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
    config: Phase1Config,
    schedule_log_files: List[str],
    max_models: Optional[int] = None
) -> Dict[str, CompiledModel]:
    """Compile models for Phase 1 analysis.

    Args:
        model_ids: List of model IDs to compile
        arch_mapping: Architecture configurations
        ofa_net: OFA network
        env: VTA environment
        target: Compilation target
        config: Phase 1 configuration
        schedule_log_files: AutoTVM schedule logs
        max_models: Maximum number of models to compile (None for all)

    Returns:
        Dictionary of compiled models
    """
    print(f"\n{'='*80}")
    print(f"PHASE 1: COMPILING MODELS FOR ANALYSIS")
    print(f"{'='*80}")

    if max_models is not None:
        model_ids = model_ids[:max_models]
        print(f"Limiting to first {max_models} models")

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
        print(f"  Failed IDs: {failed_models[:10]}{'...' if len(failed_models) > 10 else ''}")
    print(f"{'='*80}")

    return compiled_models


def analyze_compiled_models(
    compiled_models: Dict[str, CompiledModel],
    config: Phase1Config
) -> Dict[str, Any]:
    """Run Phase 1 analysis on compiled models.

    Args:
        compiled_models: Dictionary of compiled models
        config: Phase 1 configuration

    Returns:
        Analysis results dictionary
    """
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
    output_file = os.path.join(config.analysis_output_dir, "multi_graph_analysis_results.json")
    save_analysis_json(multi_analysis, output_file)
    print(f"\n✓ Analysis saved to: {output_file}")

    return multi_analysis


def merge_graphs_for_analysis(
    compiled_models: Dict[str, CompiledModel],
    config: Phase1Config
) -> None:
    """Run Phase 1 graph merging analysis.

    Args:
        compiled_models: Dictionary of compiled models
        config: Phase 1 configuration
    """
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

    # Optionally save merged structure (conceptual)
    output_file = os.path.join(config.analysis_output_dir, "multi_graph_merged_structure.json")
    merger.save_merged_structure(output_file)
    print(f"\n✓ Merged structure saved to: {output_file}")


def generate_phase1_report(
    multi_analysis: Dict[str, Any],
    config: Phase1Config
) -> None:
    """Generate comprehensive Phase 1 report.

    Args:
        multi_analysis: Analysis results
        config: Phase 1 configuration
    """
    report_file = os.path.join(config.analysis_output_dir, "PHASE1_ANALYSIS_REPORT.md")

    with open(report_file, 'w') as f:
        f.write("# Phase 1 Analysis Report: Multi-Graph Runtime Feasibility\n\n")
        f.write(f"**Experiment**: {config.experiment_name}\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Models Analyzed**: {len(multi_analysis.graphs)}\n\n")

        f.write("## Executive Summary\n\n")

        # Parameter sharing
        total_params = sum(len(g.param_names) for g in multi_analysis.graphs)
        unique_params = len(multi_analysis.all_param_names)
        shared_params = len(multi_analysis.shared_params)
        sharing_ratio = (shared_params / unique_params * 100) if unique_params > 0 else 0

        f.write(f"- **Total Parameters (across all models)**: {total_params}\n")
        f.write(f"- **Unique Parameters**: {unique_params}\n")
        f.write(f"- **Shared Parameters**: {shared_params} ({sharing_ratio:.1f}%)\n\n")

        # Storage analysis
        separate_mb = multi_analysis.total_storage_if_separate / (1024 * 1024)
        merged_mb = multi_analysis.total_storage_if_merged / (1024 * 1024)
        reduction = multi_analysis.storage_reduction_pct

        f.write(f"- **Storage (separate graphs)**: {separate_mb:.2f} MB\n")
        f.write(f"- **Storage (merged)**: {merged_mb:.2f} MB\n")
        f.write(f"- **Storage Reduction**: {reduction:.1f}%\n\n")

        # Feasibility assessment
        f.write("## Feasibility Assessment\n\n")

        if sharing_ratio >= 70:
            f.write("### ✅ **HIGHLY RECOMMENDED** to proceed to Phase 2\n\n")
            f.write("The analysis shows excellent parameter sharing characteristics:\n")
            f.write(f"- Very high sharing ratio: {sharing_ratio:.1f}%\n")
            f.write("- Significant memory savings achievable\n")
            f.write("- Strong ROI for implementing MultiGraphRuntime\n\n")
        elif sharing_ratio >= 40:
            f.write("### ⚠️ **BENEFICIAL** to proceed to Phase 2\n\n")
            f.write("The analysis shows moderate parameter sharing:\n")
            f.write(f"- Moderate sharing ratio: {sharing_ratio:.1f}%\n")
            f.write("- Notable memory savings achievable\n")
            f.write("- Consider Quick Win approach (SetInputZeroCopy) first\n\n")
        else:
            f.write("### ❌ **NOT RECOMMENDED** to proceed to Phase 2\n\n")
            f.write("The analysis shows low parameter sharing:\n")
            f.write(f"- Low sharing ratio: {sharing_ratio:.1f}%\n")
            f.write("- Limited memory savings\n")
            f.write("- Investigate why sharing is low or stay with current approach\n\n")

        # Detailed statistics
        f.write("## Detailed Statistics\n\n")
        f.write("### Parameter Usage Distribution\n\n")

        usage_dist = {}
        for param, models in multi_analysis.param_usage.items():
            count = len(models)
            usage_dist[count] = usage_dist.get(count, 0) + 1

        f.write("| Used by N graphs | Number of parameters |\n")
        f.write("|------------------|----------------------|\n")
        for count in sorted(usage_dist.keys(), reverse=True):
            f.write(f"| {count} | {usage_dist[count]} |\n")
        f.write("\n")

        # Per-model details
        f.write("### Per-Model Statistics\n\n")
        f.write("| Model ID | Nodes | Params | Storage (KB) |\n")
        f.write("|----------|-------|--------|-------------|\n")
        for g in multi_analysis.graphs:
            storage_kb = g.total_storage_bytes / 1024
            f.write(f"| {g.model_id} | {g.num_nodes} | {len(g.param_names)} | {storage_kb:.1f} |\n")
        f.write("\n")

        # Next steps
        f.write("## Next Steps\n\n")
        f.write("Based on the analysis results:\n\n")
        f.write("1. **Review this report** and the detailed JSON analysis\n")
        f.write("2. **Decide on implementation path**:\n")
        f.write("   - Option A: Proceed to Phase 2 (MultiGraphRuntime implementation)\n")
        f.write("   - Option B: Quick Win approach (SetInputZeroCopy parameter caching)\n")
        f.write("   - Option C: Stay with current approach\n")
        f.write("3. **Consult** `MULTI_GRAPH_RUNTIME_DESIGN.md` for implementation details\n\n")

        f.write("## Files Generated\n\n")
        f.write("- `multi_graph_analysis_results.json`: Detailed analysis data\n")
        f.write("- `multi_graph_merged_structure.json`: Conceptual merged structure\n")
        f.write("- `PHASE1_ANALYSIS_REPORT.md`: This report\n\n")

    print(f"\n✓ Comprehensive report saved to: {report_file}")


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    """Run Phase 1 analysis on real OFA candidate set."""

    print("="*80)
    print("PHASE 1: MULTI-GRAPH RUNTIME ANALYSIS")
    print("Real OFA Candidate Set (sa_lam_2.0)")
    print("="*80)

    # Initialize configuration
    config = Phase1Config()

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

    # Setup RPC connection (but don't upload or run anything)
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

    # Generate comprehensive report
    generate_phase1_report(multi_analysis, config)

    # Final summary
    print(f"\n{'='*80}")
    print(f"PHASE 1 ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\n✓ Analyzed {len(compiled_models)} models from experiment '{config.experiment_name}'")
    print(f"\n✓ Results saved to: {config.analysis_output_dir}/")
    print(f"\n📊 Next Steps:")
    print(f"   1. Review {config.analysis_output_dir}/PHASE1_ANALYSIS_REPORT.md")
    print(f"   2. Check {config.analysis_output_dir}/multi_graph_analysis_results.json")
    print(f"   3. Decide on implementation path based on findings")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()


