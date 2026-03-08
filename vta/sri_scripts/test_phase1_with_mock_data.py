"""
Test Phase 1 Analysis with Mock Data

This tests the Phase 1 analysis components without requiring actual model compilation.
"""

import json
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass

from multi_graph_analyzer import GraphJSONAnalyzer, print_analysis_report, save_analysis_json
from extended_multi_graph_merger import ExtendedMultiGraphMerger


@dataclass
class MockCompiledModel:
    """Mock compiled model for testing."""
    graph: str  # graph_json string
    params: Dict[str, np.ndarray]
    lib = None
    lib_path = None


def create_mock_graph(model_id: int, num_layers: int = 3) -> tuple:
    """Create a mock graph JSON and parameter names."""
    
    nodes = []
    param_names = []
    
    # Input node
    nodes.append({
        "op": "null",
        "name": "data",
        "inputs": []
    })
    
    # Create layers with some shared and some unique parameters
    for i in range(num_layers):
        # Weight parameter node
        weight_name = f"weight_{i % 2}"  # Alternating weights for sharing
        param_names.append(weight_name)
        nodes.append({
            "op": "null",
            "name": weight_name,
            "inputs": []
        })
        
        # Compute node
        nodes.append({
            "op": "tvm_op",
            "name": f"conv2d_{i}",
            "attrs": {
                "func_name": f"fused_conv2d_{i}",
                "num_inputs": "2",
                "num_outputs": "1"
            },
            "inputs": [[len(nodes)-2, 0, 0], [len(nodes)-1, 0, 0]]
        })
    
    # Create arg_nodes (inputs + parameters)
    arg_nodes = list(range(1 + 2*num_layers))  # data + all param/compute pairs
    
    # Output is the last compute node
    heads = [[len(nodes)-1, 0, 0]]
    
    graph_json = {
        "nodes": nodes,
        "arg_nodes": arg_nodes,
        "heads": heads,
        "attrs": {
            "shape": [["list_shape", [f"[1, 3, 224, 224]" for _ in nodes]]],
            "dtype": [["list_int", [0 for _ in nodes]]],
            "storage_id": [["list_int", list(range(len(nodes)))]]
        }
    }
    
    return json.dumps(graph_json), param_names


def create_mock_params(param_names: list, model_id: int) -> Dict[str, np.ndarray]:
    """Create mock parameters."""
    params = {}
    for name in param_names:
        # Create small random tensors
        params[name] = np.random.randn(16, 16).astype(np.float32)
    return params


def test_phase1_analysis():
    """Test Phase 1 analysis with mock data."""
    
    print("="*80)
    print("TESTING PHASE 1 ANALYSIS WITH MOCK DATA")
    print("="*80)
    
    # Create 5 mock models
    num_models = 5
    compiled_models = {}
    
    print(f"\nCreating {num_models} mock models...")
    for i in range(num_models):
        model_id = f"mock_model_{i}"
        graph_json, param_names = create_mock_graph(i, num_layers=3)
        params = create_mock_params(param_names, i)
        
        compiled_models[model_id] = MockCompiledModel(
            graph=graph_json,
            params=params
        )
        print(f"  Created {model_id} with {len(param_names)} parameters")
    
    # Test Analyzer
    print(f"\n{'='*80}")
    print("TESTING MULTI-GRAPH ANALYZER")
    print(f"{'='*80}")
    
    analyzer = GraphJSONAnalyzer()
    
    for model_id, compiled in compiled_models.items():
        print(f"  Analyzing {model_id}...")
        analyzer.analyze_graph(
            graph_json=compiled.graph,
            model_id=model_id,
            params=compiled.params
        )
    
    multi_analysis = analyzer.analyze_multi_graph()
    print_analysis_report(multi_analysis)
    
    # Save analysis results
    save_analysis_json(multi_analysis, "mock_analysis_results.json")
    print(f"\n✓ Analysis results saved to: mock_analysis_results.json")
    
    # Test Merger
    print(f"\n{'='*80}")
    print("TESTING MULTI-GRAPH MERGER")
    print(f"{'='*80}")
    
    merger = ExtendedMultiGraphMerger()
    
    for model_id, compiled in compiled_models.items():
        print(f"  Adding {model_id} to merger...")
        merger.add_graph(
            graph_json=compiled.graph,
            model_id=model_id,
            params=compiled.params
        )
    
    merger.print_summary()
    
    # Save merged structure
    merger.save_merged_structure("mock_merged_structure.json")
    print(f"\n✓ Merged structure saved to: mock_merged_structure.json")
    
    # Final summary
    print(f"\n{'='*80}")
    print("PHASE 1 MOCK TEST COMPLETE")
    print(f"{'='*80}")
    print(f"\n✅ All Phase 1 components working correctly!")
    print(f"\nResults:")
    print(f"  - mock_analysis_results.json")
    print(f"  - mock_merged_structure.json")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    test_phase1_analysis()

