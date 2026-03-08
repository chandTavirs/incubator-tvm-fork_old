"""
Phase 1 Unit Test - Validates analyzer and merger with mock data
"""

import json
import numpy as np
from multi_graph_analyzer import GraphJSONAnalyzer, print_analysis_report
from multi_graph_merger import merge_compiled_models
from dataclasses import dataclass
from typing import Dict

@dataclass
class MockCompiledModel:
    """Mock compiled model for testing."""
    graph_json: str
    params: Dict[str, np.ndarray]


def create_mock_graph(graph_id: int, num_layers: int = 3):
    """Create a mock graph JSON for testing."""
    nodes = []

    # Input node
    nodes.append({
        "op": "null",
        "name": "data",
        "inputs": []
    })

    # Parameter nodes
    param_names = []
    for i in range(num_layers):
        # Some params shared across graphs, some unique
        if i == 0:
            param_name = f"shared_weight_{i}"  # Shared
        else:
            param_name = f"weight_graph{graph_id}_layer{i}"  # Unique

        param_names.append(param_name)
        nodes.append({
            "op": "null",
            "name": param_name,
            "inputs": []
        })

    # Operator nodes
    for i in range(num_layers):
        nodes.append({
            "op": "tvm_op",
            "name": f"conv2d_{i}",
            "attrs": {
                "func_name": f"fused_conv2d_{i}",
                "num_inputs": "2",
                "num_outputs": "1"
            },
            "inputs": [[i*2+1, 0, 0], [i*2+2, 0, 0]]
        })

    graph = {
        "nodes": nodes,
        "arg_nodes": list(range(1, num_layers*2+1)),
        "heads": [[len(nodes)-1, 0, 0]],
        "attrs": {
            "shape": [[1, 3, 224, 224]] + [[64, 64, 3, 3]] * num_layers + [[1, 64, 112, 112]] * num_layers,
            "dltype": ["float32"] * len(nodes),
            "storage_id": list(range(len(nodes)))
        }
    }

    return json.dumps(graph), param_names


def create_mock_params(param_names, graph_id):
    """Create mock parameters."""
    params = {}
    for i, name in enumerate(param_names):
        if "shared" in name:
            # Shared params have same size
            params[name] = np.random.randn(64, 64, 3, 3).astype(np.float32)
        else:
            # Unique params vary slightly
            size = 64 + graph_id * 8
            params[name] = np.random.randn(size, 64, 3, 3).astype(np.float32)
    return params


def test_analyzer():
    """Test the multi-graph analyzer."""
    print("\n" + "="*80)
    print("TEST 1: Multi-Graph Analyzer")
    print("="*80)

    analyzer = GraphJSONAnalyzer()

    # Create 3 mock graphs
    for i in range(3):
        graph_json, param_names = create_mock_graph(i, num_layers=3)
        params = create_mock_params(param_names, i)

        model_id = f"mock_model_{i}"
        print(f"\nAnalyzing {model_id}...")
        analysis = analyzer.analyze_graph(graph_json, model_id, params)

        print(f"  Nodes: {analysis.num_nodes}")
        print(f"  Parameters: {len(analysis.param_names)}")
        print(f"  Param names: {list(analysis.param_names)}")

    # Multi-graph analysis
    print("\nPerforming multi-graph analysis...")
    multi_analysis = analyzer.analyze_multi_graph()

    # Print report
    print_analysis_report(multi_analysis)

    # Validate results
    assert len(multi_analysis.graphs) == 3, "Should have 3 graphs"
    assert len(multi_analysis.shared_params) > 0, "Should have shared parameters"

    print("\n✅ Analyzer test PASSED")
    return multi_analysis


def test_merger():
    """Test the multi-graph merger."""
    print("\n" + "="*80)
    print("TEST 2: Multi-Graph Merger")
    print("="*80)

    # Create mock compiled models
    compiled_models = {}

    for i in range(3):
        graph_json, param_names = create_mock_graph(i, num_layers=3)
        params = create_mock_params(param_names, i)

        model_id = f"mock_model_{i}"
        compiled_models[model_id] = MockCompiledModel(
            graph_json=graph_json,
            params=params
        )

    # Merge models
    merger = merge_compiled_models(compiled_models)
    merger.print_summary()

    # Validate results
    assert len(merger.graph_mappings) == 3, "Should have 3 graphs"
    assert len(merger.unified_params) > 0, "Should have unified parameters"

    # Check for shared parameters
    shared_count = sum(1 for p in merger.unified_params.values() if len(p.used_by_graphs) > 1)
    assert shared_count > 0, "Should have at least one shared parameter"

    print("\n✅ Merger test PASSED")
    return merger


def test_integration():
    """Test analyzer + merger integration."""
    print("\n" + "="*80)
    print("TEST 3: Integration Test")
    print("="*80)

    # Run both tests
    multi_analysis = test_analyzer()
    merger = test_merger()

    # Cross-validate
    assert len(multi_analysis.graphs) == len(merger.graph_mappings), \
        "Analyzer and merger should have same number of graphs"

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED")
    print("="*80)
    print("\nPhase 1 components are working correctly!")
    print("Ready to run with real compiled models.")


if __name__ == "__main__":
    print("Phase 1 Component Testing")
    print("="*80)

    try:
        test_integration()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    print("\n✓ Phase 1 validation complete!")

