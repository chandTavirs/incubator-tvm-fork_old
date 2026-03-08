#!/usr/bin/env python3
"""
Quick test of storage calculation fix
"""

import numpy as np
import json
from multi_graph_analyzer import GraphJSONAnalyzer

# Create simple test data
graph_json = json.dumps({
    "nodes": [
        {"op": "null", "name": "data", "inputs": []},
        {"op": "null", "name": "weight1", "inputs": []},
        {"op": "tvm_op", "name": "conv1", "inputs": [[0, 0, 0], [1, 0, 0]]}
    ],
    "arg_nodes": [0, 1],
    "heads": [[2, 0, 0]],
    "attrs": {
        "storage_id": [[0], [1], [2]],
        "shape": [[1, 3, 224, 224], [64, 3, 3, 3], [1, 64, 112, 112]],
        "dltype": ["float32", "float32", "float32"]
    }
})

# Create test params - weight1 is 64x3x3x3 float32 = 64*3*3*3*4 = 6912 bytes
params = {
    "weight1": np.random.randn(64, 3, 3, 3).astype(np.float32)
}

# Test analyzer
analyzer = GraphJSONAnalyzer()
result = analyzer.analyze_graph(graph_json, "test_model", params)

print(f"Test Results:")
print(f"  Model ID: {result.model_id}")
print(f"  Parameters: {result.param_names}")
print(f"  Total storage bytes: {result.total_storage_bytes}")
print(f"  Expected bytes: {params['weight1'].nbytes} (64*3*3*3*4 = {64*3*3*3*4})")

if result.total_storage_bytes == params['weight1'].nbytes:
    print("\n✅ Storage calculation is CORRECT!")
else:
    print(f"\n❌ Storage calculation is WRONG!")
    print(f"   Expected: {params['weight1'].nbytes}")
    print(f"   Got: {result.total_storage_bytes}")

