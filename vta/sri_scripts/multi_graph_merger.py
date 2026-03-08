"""
Phase 1: Multi-Graph Merger - Creates unified multi-graph JSON structure
"""
import json
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import copy
@dataclass
class UnifiedParameter:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    storage_id: int
    used_by_graphs: List[str]
    data: np.ndarray = None
@dataclass
class GraphMapping:
    graph_id: str
    original_graph: Dict[str, Any]
    param_mappings: Dict[str, int]
    input_node_ids: List[int]
    output_node_ids: List[Tuple[int, int, int]]
class MultiGraphMerger:
    def __init__(self):
        self.unified_params: Dict[str, UnifiedParameter] = {}
        self.graph_mappings: List[GraphMapping] = []
        self.next_storage_id = 0
    def add_graph(self, graph_json: str, graph_id: str, params: Dict[str, np.ndarray]):
        graph = json.loads(graph_json)
        param_mappings = {}
        for param_name, param_data in params.items():
            if param_name not in self.unified_params:
                unified_param = UnifiedParameter(
                    name=param_name, shape=param_data.shape, dtype=str(param_data.dtype),
                    storage_id=self.next_storage_id, used_by_graphs=[graph_id], data=param_data
                )
                self.unified_params[param_name] = unified_param
                self.next_storage_id += 1
            else:
                self.unified_params[param_name].used_by_graphs.append(graph_id)
            param_mappings[param_name] = self.unified_params[param_name].storage_id
        arg_nodes = graph.get('arg_nodes', [])
        input_node_ids = [n for n in arg_nodes if graph['nodes'][n]['name'] not in params]
        mapping = GraphMapping(
            graph_id=graph_id, original_graph=graph, param_mappings=param_mappings,
            input_node_ids=input_node_ids, output_node_ids=graph['heads']
        )
        self.graph_mappings.append(mapping)
        return mapping
    def print_summary(self):
        print("\n" + "="*80)
        print("MULTI-GRAPH MERGER SUMMARY")
        print("="*80)
        shared = sum(1 for p in self.unified_params.values() if len(p.used_by_graphs) > 1)
        print(f"\nTotal parameters: {len(self.unified_params)}")
        print(f"Shared parameters: {shared}")
        print(f"Total graphs: {len(self.graph_mappings)}")
        print("="*80)
def merge_compiled_models(compiled_models: Dict[str, Any]):
    merger = MultiGraphMerger()
    for model_id, compiled in compiled_models.items():
        merger.add_graph(compiled.graph_json, model_id, compiled.params)
    return merger
