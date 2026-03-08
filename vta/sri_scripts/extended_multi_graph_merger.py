"""
Extended Multi-Graph Merger for Phase 1 Analysis

This extends the base MultiGraphMerger with additional functionality
for saving merged structures. This is a separate file to avoid modifying
the original implementation.
"""

import json
import numpy as np
from typing import Dict, Any

from multi_graph_merger import MultiGraphMerger, UnifiedParameter


class ExtendedMultiGraphMerger(MultiGraphMerger):
    """Extended merger with save functionality."""

    def save_merged_structure(self, output_file: str):
        """Save the conceptual merged structure to JSON.

        Args:
            output_file: Path to output JSON file
        """
        merged_structure = {
            "num_graphs": len(self.graph_mappings),
            "num_unified_params": len(self.unified_params),

            "unified_parameters": [
                {
                    "name": param.name,
                    "shape": list(param.shape),
                    "dtype": param.dtype,
                    "storage_id": param.storage_id,
                    "used_by_graphs": param.used_by_graphs,
                    "num_users": len(param.used_by_graphs),
                    "is_shared": len(param.used_by_graphs) > 1
                }
                for param in self.unified_params.values()
            ],

            "graphs": [
                {
                    "graph_id": mapping.graph_id,
                    "num_nodes": len(mapping.original_graph.get('nodes', [])),
                    "num_params": len(mapping.param_mappings),
                    "param_names": list(mapping.param_mappings.keys()),
                    "param_storage_ids": list(mapping.param_mappings.values()),
                    "num_inputs": len(mapping.input_node_ids),
                    "num_outputs": len(mapping.output_node_ids),
                }
                for mapping in self.graph_mappings
            ],

            "statistics": {
                "total_params": len(self.unified_params),
                "shared_params": sum(1 for p in self.unified_params.values() if len(p.used_by_graphs) > 1),
                "unique_params": sum(1 for p in self.unified_params.values() if len(p.used_by_graphs) == 1),
                "max_sharing": max((len(p.used_by_graphs) for p in self.unified_params.values()), default=0),
            }
        }

        with open(output_file, 'w') as f:
            json.dump(merged_structure, f, indent=2)

