"""
Phase 1: Multi-Graph Runtime Analysis Tool

This script analyzes compiled TVM graphs to understand:
1. Graph JSON structure and parameter usage
2. Parameter sharing patterns across models
3. Storage allocation and memory requirements
4. Feasibility of merging graphs into unified runtime

Usage:
    python multi_graph_analyzer.py
"""

import json
import numpy as np
from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import os


@dataclass
class GraphAnalysis:
    """Analysis results for a single graph."""
    model_id: str
    num_nodes: int
    num_inputs: int
    num_outputs: int
    input_nodes: List[int]
    output_nodes: List[Tuple[int, int, int]]
    param_names: Set[str]
    param_shapes: Dict[str, Tuple]
    param_dtypes: Dict[str, str]
    storage_ids: List[int]
    unique_storage_ids: Set[int]
    total_storage_bytes: int
    operators: List[str]
    func_names: Set[str]


@dataclass
class MultiGraphAnalysis:
    """Analysis results for multiple graphs."""
    graphs: List[GraphAnalysis]

    # Parameter sharing analysis
    all_param_names: Set[str]
    shared_params: Set[str]  # Params used by multiple graphs
    unique_params: Dict[str, str]  # Params used by only one graph
    param_usage: Dict[str, List[str]]  # param_name -> list of model_ids

    # Storage analysis
    total_storage_per_graph: Dict[str, int]
    total_storage_if_merged: int
    total_storage_if_separate: int
    storage_reduction_pct: float

    # Operator analysis
    all_operators: Set[str]
    all_func_names: Set[str]
    operator_frequency: Dict[str, int]
    func_frequency: Dict[str, int]


class GraphJSONAnalyzer:
    """Analyzes TVM graph JSON structures."""

    def __init__(self):
        self.analyses: List[GraphAnalysis] = []

    def analyze_graph(self, graph_json: str, model_id: str, params: Dict[str, np.ndarray]) -> GraphAnalysis:
        """Analyze a single graph JSON."""
        graph = json.loads(graph_json)

        # Extract basic structure
        nodes = graph['nodes']
        arg_nodes = graph.get('arg_nodes', [])
        heads = graph['heads']
        attrs = graph.get('attrs', {})

        # Identify input nodes (excluding parameters)
        input_nodes = []
        param_names = set()

        for node_id in arg_nodes:
            node = nodes[node_id]
            node_name = node['name']
            if node_name in params:
                param_names.add(node_name)
            else:
                input_nodes.append(node_id)

        # Extract parameter metadata
        param_shapes = {}
        param_dtypes = {}
        for name, array in params.items():
            param_shapes[name] = array.shape
            param_dtypes[name] = str(array.dtype)

        # Storage analysis from graph JSON (for metadata)
        storage_ids_raw = attrs.get('storage_id', [])
        # Handle case where storage_id might be nested list [[x], [y], ...] or list of lists
        storage_ids = []
        for sid in storage_ids_raw:
            if isinstance(sid, list):
                # If it's a list, take first element or flatten
                if len(sid) > 0:
                    storage_ids.append(sid[0] if not isinstance(sid[0], list) else sid[0][0])
            else:
                storage_ids.append(sid)
        unique_storage_ids = set(storage_ids)

        # Calculate ACTUAL storage requirements from parameter arrays
        # This is the real memory footprint, not from graph JSON which may be incomplete
        total_storage_bytes = 0
        for name, array in params.items():
            # Handle both TVM NDArray and numpy array
            if hasattr(array, 'nbytes'):
                # numpy array
                array_bytes = array.nbytes
            elif hasattr(array, 'shape') and hasattr(array, 'dtype'):
                # TVM NDArray - calculate from shape and dtype
                num_elements = int(np.prod(array.shape))
                dtype_str = str(array.dtype)
                dtype_bytes = self._get_dtype_bytes(dtype_str)
                array_bytes = num_elements * dtype_bytes
            else:
                # Fallback - try to convert to numpy
                try:
                    array_np = np.array(array)
                    array_bytes = array_np.nbytes
                except:
                    array_bytes = 0
            total_storage_bytes += array_bytes

        # Extract operators and functions
        operators = []
        func_names = set()

        for node in nodes:
            if node['op'] == 'tvm_op':
                operators.append(node['name'])
                if 'attrs' in node and 'func_name' in node['attrs']:
                    func_names.add(node['attrs']['func_name'])

        analysis = GraphAnalysis(
            model_id=model_id,
            num_nodes=len(nodes),
            num_inputs=len(input_nodes),
            num_outputs=len(heads),
            input_nodes=input_nodes,
            output_nodes=heads,
            param_names=param_names,
            param_shapes=param_shapes,
            param_dtypes=param_dtypes,
            storage_ids=storage_ids,
            unique_storage_ids=unique_storage_ids,
            total_storage_bytes=total_storage_bytes,
            operators=operators,
            func_names=func_names
        )

        self.analyses.append(analysis)
        return analysis

    def analyze_multi_graph(self) -> MultiGraphAnalysis:
        """Analyze patterns across multiple graphs."""
        if not self.analyses:
            raise ValueError("No graphs analyzed yet")

        # Collect all parameter names
        all_param_names = set()
        param_usage = defaultdict(list)

        for analysis in self.analyses:
            all_param_names.update(analysis.param_names)
            for param_name in analysis.param_names:
                param_usage[param_name].append(analysis.model_id)

        # Identify shared vs unique parameters
        shared_params = {name for name, models in param_usage.items() if len(models) > 1}
        unique_params = {name: param_usage[name][0] for name, models in param_usage.items() if len(models) == 1}

        # Storage analysis
        total_storage_per_graph = {
            analysis.model_id: analysis.total_storage_bytes
            for analysis in self.analyses
        }

        total_storage_if_separate = sum(total_storage_per_graph.values())

        # Calculate merged storage based on actual parameter sharing
        # When merged: each unique param counted once (not duplicated across models)
        param_size_map = {}

        # Build parameter size map from actual arrays
        for analysis in self.analyses:
            for param_name, shape in analysis.param_shapes.items():
                if param_name not in param_size_map:
                    dtype = analysis.param_dtypes[param_name]
                    # Calculate size from shape and dtype
                    num_elements = int(np.prod(shape))
                    dtype_bytes = self._get_dtype_bytes(dtype)
                    size = num_elements * dtype_bytes
                    param_size_map[param_name] = size

        # Total parameter storage if merged (each unique param counted once)
        total_storage_if_merged = sum(param_size_map.values())

        # Calculate reduction percentage
        storage_reduction_pct = 100 * (1 - total_storage_if_merged / total_storage_if_separate) if total_storage_if_separate > 0 else 0

        # Operator analysis
        all_operators = set()
        all_func_names = set()
        operator_frequency = defaultdict(int)
        func_frequency = defaultdict(int)

        for analysis in self.analyses:
            all_operators.update(analysis.operators)
            all_func_names.update(analysis.func_names)

            for op in analysis.operators:
                operator_frequency[op] += 1
            for func in analysis.func_names:
                func_frequency[func] += 1

        return MultiGraphAnalysis(
            graphs=self.analyses,
            all_param_names=all_param_names,
            shared_params=shared_params,
            unique_params=unique_params,
            param_usage=dict(param_usage),
            total_storage_per_graph=total_storage_per_graph,
            total_storage_if_merged=total_storage_if_merged,
            total_storage_if_separate=total_storage_if_separate,
            storage_reduction_pct=storage_reduction_pct,
            all_operators=all_operators,
            all_func_names=all_func_names,
            operator_frequency=dict(operator_frequency),
            func_frequency=dict(func_frequency)
        )

    @staticmethod
    def _get_dtype_bytes(dtype_str: str) -> int:
        """Get bytes per element for a dtype string."""
        dtype_lower = str(dtype_str).lower()

        # Handle numpy dtype objects
        if 'float64' in dtype_lower or 'double' in dtype_lower:
            return 8
        elif 'float32' in dtype_lower or 'single' in dtype_lower:
            return 4
        elif 'float16' in dtype_lower or 'half' in dtype_lower:
            return 2
        elif 'int64' in dtype_lower or 'long' in dtype_lower:
            return 8
        elif 'int32' in dtype_lower:
            return 4
        elif 'int16' in dtype_lower or 'short' in dtype_lower:
            return 2
        elif 'int8' in dtype_lower:
            return 1
        elif 'uint64' in dtype_lower:
            return 8
        elif 'uint32' in dtype_lower:
            return 4
        elif 'uint16' in dtype_lower:
            return 2
        elif 'uint8' in dtype_lower or 'byte' in dtype_lower:
            return 1
        else:
            # Default to 4 bytes (most common case)
            return 4


def print_analysis_report(multi_analysis: MultiGraphAnalysis):
    """Print comprehensive analysis report."""
    print("\n" + "="*80)
    print("MULTI-GRAPH ANALYSIS REPORT")
    print("="*80)

    print(f"\n📊 OVERVIEW")
    print(f"{'─'*80}")
    print(f"Total graphs analyzed: {len(multi_analysis.graphs)}")
    print(f"Total unique parameters: {len(multi_analysis.all_param_names)}")
    print(f"Shared parameters: {len(multi_analysis.shared_params)}")
    print(f"Graph-specific parameters: {len(multi_analysis.unique_params)}")
    print(f"Total unique operators: {len(multi_analysis.all_operators)}")
    print(f"Total unique functions: {len(multi_analysis.all_func_names)}")

    print(f"\n💾 STORAGE ANALYSIS")
    print(f"{'─'*80}")
    print(f"Total storage (separate graphs): {multi_analysis.total_storage_if_separate / 1024 / 1024:.2f} MB")
    print(f"Estimated storage (merged): {multi_analysis.total_storage_if_merged / 1024 / 1024:.2f} MB")
    print(f"Storage reduction: {multi_analysis.storage_reduction_pct:.2f}%")

    print(f"\nPer-graph storage:")
    for model_id, storage in sorted(multi_analysis.total_storage_per_graph.items()):
        print(f"  {model_id}: {storage / 1024 / 1024:.2f} MB")

    print(f"\n🔗 PARAMETER SHARING (Based on Shape/Dtype)")
    print(f"{'─'*80}")
    print(f"⚠️  Note: Analysis based on parameter signatures (shape + dtype).")
    print(f"    For OFA networks, parameters with same shape may have different values.")
    print(f"    Use find_truly_shared_params() for value-based sharing analysis.")

    # Count parameters by usage
    usage_counts = defaultdict(int)
    for param_name, models in multi_analysis.param_usage.items():
        usage_counts[len(models)] += 1

    print(f"\nParameter signature distribution:")
    for num_models in sorted(usage_counts.keys(), reverse=True):
        count = usage_counts[num_models]
        if num_models == 1:
            print(f"  Used by 1 graph: {count} parameter signatures")
        else:
            print(f"  Used by {num_models} graphs: {count} parameter signatures")

    # Show top shared parameters
    print(f"\nMost widely shared parameter signatures (top 10):")
    sorted_params = sorted(multi_analysis.param_usage.items(),
                          key=lambda x: len(x[1]), reverse=True)
    for param_name, models in sorted_params[:10]:
        print(f"  {param_name}: used by {len(models)} graphs")

    print(f"\n⚙️  OPERATOR ANALYSIS")
    print(f"{'─'*80}")
    print(f"Total unique compiled functions: {len(multi_analysis.all_func_names)}")

    print(f"\nMost frequent operators (top 10):")
    sorted_ops = sorted(multi_analysis.operator_frequency.items(),
                       key=lambda x: x[1], reverse=True)
    for op_name, count in sorted_ops[:10]:
        print(f"  {op_name}: {count} occurrences")

    print(f"\n📋 PER-GRAPH DETAILS")
    print(f"{'─'*80}")
    for analysis in multi_analysis.graphs:
        print(f"\n{analysis.model_id}:")
        print(f"  Nodes: {analysis.num_nodes}")
        print(f"  Inputs: {analysis.num_inputs}")
        print(f"  Outputs: {analysis.num_outputs}")
        print(f"  Parameters: {len(analysis.param_names)}")
        print(f"  Storage slots: {len(analysis.unique_storage_ids)}")
        print(f"  Operators: {len(analysis.operators)}")
        print(f"  Functions: {len(analysis.func_names)}")

    print(f"\n✅ FEASIBILITY ASSESSMENT")
    print(f"{'─'*80}")

    # Check for potential issues
    issues = []

    # Check if all graphs have same input/output structure
    input_counts = [g.num_inputs for g in multi_analysis.graphs]
    output_counts = [g.num_outputs for g in multi_analysis.graphs]

    if len(set(input_counts)) > 1:
        issues.append(f"⚠️  Graphs have different input counts: {set(input_counts)}")
    else:
        print(f"✓ All graphs have consistent input count: {input_counts[0]}")

    if len(set(output_counts)) > 1:
        issues.append(f"⚠️  Graphs have different output counts: {set(output_counts)}")
    else:
        print(f"✓ All graphs have consistent output count: {output_counts[0]}")

    # Check parameter sharing ratio
    sharing_ratio = len(multi_analysis.shared_params) / len(multi_analysis.all_param_names) * 100
    if sharing_ratio > 50:
        print(f"✓ High parameter sharing ratio: {sharing_ratio:.1f}%")
    else:
        issues.append(f"⚠️  Low parameter sharing ratio: {sharing_ratio:.1f}%")

    # Check storage reduction
    if multi_analysis.storage_reduction_pct > 20:
        print(f"✓ Significant storage reduction achievable: {multi_analysis.storage_reduction_pct:.1f}%")
    else:
        issues.append(f"⚠️  Limited storage reduction: {multi_analysis.storage_reduction_pct:.1f}%")

    if issues:
        print(f"\n⚠️  Potential Issues:")
        for issue in issues:
            print(f"  {issue}")

    print(f"\n{'='*80}")


def save_analysis_json(multi_analysis: MultiGraphAnalysis, output_file: str):
    """Save analysis results to JSON for further processing."""

    def convert_to_native_types(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        import numpy as np
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return [convert_to_native_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native_types(item) for item in obj]
        return obj

    data = {
        'summary': {
            'num_graphs': len(multi_analysis.graphs),
            'total_params': len(multi_analysis.all_param_names),
            'shared_params': len(multi_analysis.shared_params),
            'unique_params': len(multi_analysis.unique_params),
            'storage_reduction_pct': float(multi_analysis.storage_reduction_pct),
        },
        'graphs': [
            {
                'model_id': str(g.model_id),
                'num_nodes': int(g.num_nodes),
                'num_inputs': int(g.num_inputs),
                'num_outputs': int(g.num_outputs),
                'num_params': int(len(g.param_names)),
                'storage_bytes': int(g.total_storage_bytes),
                'param_names': list(g.param_names),
            }
            for g in multi_analysis.graphs
        ],
        'param_usage': {
            str(name): list(models) for name, models in multi_analysis.param_usage.items()
        },
        'shared_params': list(multi_analysis.shared_params),
        'storage_analysis': {
            'total_if_separate': int(multi_analysis.total_storage_if_separate),
            'total_if_merged': int(multi_analysis.total_storage_if_merged),
            'reduction_pct': float(multi_analysis.storage_reduction_pct),
        }
    }

    # Convert all numpy types to native Python types
    data = convert_to_native_types(data)

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\n💾 Analysis saved to: {output_file}")


def example_usage():
    """Example usage with compiled models."""
    # This will be called from shared_parameter_execution.py
    # after models are compiled
    pass


if __name__ == "__main__":
    print("Multi-Graph Analyzer - Phase 1")
    print("This module should be imported and used with compiled models.")
    print("\nSee shared_parameter_execution.py for integration example.")




