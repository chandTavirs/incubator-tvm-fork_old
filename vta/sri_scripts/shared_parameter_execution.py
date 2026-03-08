"""
Shared parameter execution utilities for VTA candidate set.

This module extends the refactored execution script to support efficient
multi-model execution with shared parameter management.

Key Features:
- Parameter Deduplication: Identifies and tracks shared parameters across models
  to reduce memory footprint and understand parameter reuse patterns.
- Runtime Caching: Caches graph runtime instances to avoid recompilation overhead
  when switching between models.
- Per-Model Parameter Upload: Each model gets its required parameters uploaded
  to match its specific graph structure (parameters are deduplicated in host
  memory but uploaded per-model to device memory).
- Efficient Model Switching: Pre-loads all models and their parameters during
  initialization for fast switching during inference.

Note: While parameters are deduplicated in host memory (showing memory reduction
statistics), each model still needs its own parameter upload to match its graph
structure. The main benefit is faster model switching through runtime caching
and organized parameter management.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import time
import numpy as np

import tvm
from tvm.contrib import graph_runtime

from execute_candidate_set_refactored import CompiledModel


# ==============================================================================
# Shared Parameter Management
# ==============================================================================

@dataclass
class SharedParameterManager:
    """Manages shared parameters across multiple models."""

    # All unique parameters across models
    unified_params: Dict[str, tvm.nd.NDArray] = field(default_factory=dict)

    # Mapping of model_id -> list of parameter names it uses
    model_param_mapping: Dict[str, List[str]] = field(default_factory=dict)

    # Track parameter statistics
    total_params: int = 0
    unique_params: int = 0
    shared_params: int = 0

    def merge_parameters(self, compiled_models: Dict[str, CompiledModel]):
        """Merge parameters from all compiled models.

        Args:
            compiled_models: Dictionary of compiled model artifacts
        """
        print("\n" + "="*80)
        print("Merging Parameters Across Models")
        print("="*80)

        param_hash_map = {}  # hash -> (param_name, param_value)

        for model_id, model_data in compiled_models.items():
            self.model_param_mapping[model_id] = list(model_data.params.keys())

            for param_name, param_value in model_data.params.items():
                self.total_params += 1

                # Create hash of parameter data for deduplication
                param_hash = hash(param_value.asnumpy().tobytes())

                if param_hash not in param_hash_map:
                    # New unique parameter
                    param_hash_map[param_hash] = (param_name, param_value)
                    self.unified_params[param_name] = param_value
                else:
                    # Shared parameter
                    self.shared_params += 1

        self.unique_params = len(self.unified_params)

        print(f"Total parameters across all models: {self.total_params}")
        print(f"Unique parameters: {self.unique_params}")
        print(f"Shared parameters: {self.shared_params}")
        print(f"Memory reduction: {(1 - self.unique_params / self.total_params) * 100:.2f}%")

    def get_model_params(self, model_id: str) -> Dict[str, tvm.nd.NDArray]:
        """Get the subset of parameters needed for a specific model.

        Args:
            model_id: ID of the model

        Returns:
            Dictionary of parameters for this model
        """
        param_names = self.model_param_mapping.get(model_id, [])
        return {
            name: self.unified_params[name]
            for name in param_names
            if name in self.unified_params
        }


# ==============================================================================
# Runtime Cache Management
# ==============================================================================

@dataclass
class RuntimeCache:
    """Caches graph runtime instances for efficient model switching."""

    _cache: Dict[str, tvm.runtime.Module] = field(default_factory=dict)
    _hit_count: int = 0
    _miss_count: int = 0

    def get_or_create(
        self,
        model_id: str,
        model_data: CompiledModel,
        env,
        remote,
        ctx
    ) -> tvm.runtime.Module:
        """Get cached runtime or create new one.

        Args:
            model_id: ID of the model
            model_data: Compiled model artifacts
            env: VTA environment
            remote: RPC connection
            ctx: Execution context

        Returns:
            Graph runtime module
        """
        if model_id in self._cache:
            self._hit_count += 1
            return self._cache[model_id]

        self._miss_count += 1

        # Create new runtime
        if env.TARGET == "intelfocl":
            ctxes = [remote.ext_dev(0), remote.cpu(0)]
            runtime = graph_runtime.create(model_data.graph, model_data.remote_lib, ctxes)
        else:
            runtime = graph_runtime.create(model_data.graph, model_data.remote_lib, ctx)

        self._cache[model_id] = runtime
        return runtime

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cached_models': len(self._cache),
            'cache_hits': self._hit_count,
            'cache_misses': self._miss_count,
            'hit_rate': self._hit_count / max(1, self._hit_count + self._miss_count)
        }

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0


# ==============================================================================
# Shared Parameter Execution Engine
# ==============================================================================

class SharedParameterExecutor:
    """Executes multiple models with shared parameter loading."""

    def __init__(
        self,
        compiled_models: Dict[str, CompiledModel],
        env,
        remote,
        ctx,
        input_name: str = "input0"
    ):
        """Initialize the shared parameter executor.

        Args:
            compiled_models: Dictionary of compiled model artifacts
            env: VTA environment
            remote: RPC connection
            ctx: Execution context
            input_name: Name of the input tensor
        """
        self.compiled_models = compiled_models
        self.env = env
        self.remote = remote
        self.ctx = ctx
        self.input_name = input_name

        # Initialize managers
        self.param_manager = SharedParameterManager()
        self.runtime_cache = RuntimeCache()

        # Upload parameters once
        self._initialize_shared_parameters()

    def _initialize_shared_parameters(self):
        """Initialize and upload shared parameters."""
        print("\n" + "="*80)
        print("Initializing Shared Parameter Execution")
        print("="*80)

        # Merge parameters from all models
        self.param_manager.merge_parameters(self.compiled_models)

        # Pre-create runtimes for all models and upload their parameters
        print("\nPre-loading parameters for all models...")
        total_upload_time = 0

        for model_id, model_data in self.compiled_models.items():
            # Create runtime for this model
            runtime = self.runtime_cache.get_or_create(
                model_id, model_data, self.env, self.remote, self.ctx
            )

            # Upload only the parameters this model needs
            model_params = self.param_manager.get_model_params(model_id)

            upload_start = time.time()
            runtime.set_input(**model_params)
            upload_time = time.time() - upload_start
            total_upload_time += upload_time

            print(f"  {model_id}: {len(model_params)} params in {upload_time:.4f}s")

        print(f"\nTotal parameter upload time: {total_upload_time:.4f}s")
        print(f"Average per model: {total_upload_time/len(self.compiled_models):.4f}s")
        print("All model runtimes cached and ready")

    def run_inference(
        self,
        model_id: str,
        image_data: np.ndarray,
        skip_param_upload: bool = True
    ) -> np.ndarray:
        """Run inference with a specific model.

        Args:
            model_id: ID of the model to use
            image_data: Input image data
            skip_param_upload: If True, reuse already uploaded parameters (default: True)
                              Parameters are uploaded during initialization, so this should
                              generally remain True unless debugging.

        Returns:
            Model output as numpy array
        """
        if model_id not in self.compiled_models:
            raise ValueError(f"Model {model_id} not found in compiled models")

        model_data = self.compiled_models[model_id]

        # Check if runtime exists in cache
        runtime_was_cached = model_id in self.runtime_cache._cache

        # Get or create runtime for this model
        runtime = self.runtime_cache.get_or_create(
            model_id, model_data, self.env, self.remote, self.ctx
        )

        # Upload parameters if:
        # 1. Explicitly requested (skip_param_upload=False), OR
        # 2. Runtime was just created (not in cache)
        if not skip_param_upload or not runtime_was_cached:
            model_params = self.param_manager.get_model_params(model_id)
            runtime.set_input(**model_params)

        # Set input data (always needed)
        runtime.set_input(self.input_name, image_data)

        # Run inference
        runtime.run()

        # Get output
        output = runtime.get_output(
            0, tvm.nd.empty((self.env.BATCH, 10), "float32", self.remote.cpu(0))
        )

        return output.asnumpy()

    def benchmark_model_switching(
        self,
        image_data: np.ndarray,
        num_iterations: int = 10
    ) -> Dict[str, float]:
        """Benchmark the overhead of switching between models.

        Args:
            image_data: Sample input data
            num_iterations: Number of iterations per model

        Returns:
            Dictionary of timing statistics
        """
        print("\n" + "="*80)
        print("Benchmarking Model Switching Overhead")
        print("="*80)

        model_ids = list(self.compiled_models.keys())
        results = {}

        for model_id in model_ids:
            times = []

            for i in range(num_iterations):
                start = time.time()
                self.run_inference(model_id, image_data, skip_param_upload=True)
                elapsed = time.time() - start
                times.append(elapsed)

            avg_time = np.mean(times)
            std_time = np.std(times)

            results[model_id] = {
                'avg_time': avg_time,
                'std_time': std_time,
                'min_time': np.min(times),
                'max_time': np.max(times)
            }

            print(f"{model_id}: {avg_time:.4f}s ± {std_time:.4f}s")

        return results

    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        return {
            'parameter_stats': {
                'total_params': self.param_manager.total_params,
                'unique_params': self.param_manager.unique_params,
                'shared_params': self.param_manager.shared_params,
                'memory_reduction_pct': (
                    (1 - self.param_manager.unique_params / max(1, self.param_manager.total_params)) * 100
                )
            },
            'cache_stats': self.runtime_cache.get_stats()
        }


# ==============================================================================
# Helper Functions
# ==============================================================================

def create_shared_executor(
    compiled_models: Dict[str, CompiledModel],
    env,
    remote,
    ctx,
    input_name: str = "input0"
) -> SharedParameterExecutor:
    """Factory function to create a shared parameter executor.

    Args:
        compiled_models: Dictionary of compiled model artifacts
        env: VTA environment
        remote: RPC connection
        ctx: Execution context
        input_name: Name of the input tensor

    Returns:
        SharedParameterExecutor instance
    """
    return SharedParameterExecutor(compiled_models, env, remote, ctx, input_name)


def run_multi_model_inference(
    executor: SharedParameterExecutor,
    model_sequence: List[str],
    image_data_list: List[np.ndarray]
) -> List[np.ndarray]:
    """Run inference on multiple models with different images.

    Args:
        executor: SharedParameterExecutor instance
        model_sequence: List of model IDs to execute in sequence
        image_data_list: List of image data arrays

    Returns:
        List of model outputs
    """
    if len(model_sequence) != len(image_data_list):
        raise ValueError("model_sequence and image_data_list must have same length")

    results = []

    print("\n" + "="*80)
    print("Running Multi-Model Inference")
    print("="*80)

    for i, (model_id, image_data) in enumerate(zip(model_sequence, image_data_list)):
        print(f"\nInference {i+1}/{len(model_sequence)}: {model_id}")

        start = time.time()
        output = executor.run_inference(model_id, image_data, skip_param_upload=True)
        elapsed = time.time() - start

        print(f"  Inference time: {elapsed:.4f}s")
        results.append(output)

    return results


# ==============================================================================
# Multi-Graph Analysis Integration
# ==============================================================================

def analyze_compiled_models(compiled_models: Dict[str, CompiledModel]):
    """Analyze compiled models for multi-graph runtime feasibility.

    This performs Phase 1 analysis to understand:
    - Parameter sharing patterns
    - Storage requirements
    - Graph structure compatibility
    - Feasibility of unified runtime
    """
    from multi_graph_analyzer import GraphJSONAnalyzer, print_analysis_report, save_analysis_json

    print("\n" + "="*80)
    print("PHASE 1: MULTI-GRAPH ANALYSIS")
    print("="*80)

    analyzer = GraphJSONAnalyzer()

    # Analyze each compiled model
    print("\nAnalyzing individual graphs...")
    for model_id, compiled in compiled_models.items():
        print(f"  Analyzing {model_id}...")
        analyzer.analyze_graph(
            graph_json=compiled.graph,  # Note: CompiledModel has 'graph' not 'graph_json'
            model_id=model_id,
            params=compiled.params
        )

    # Perform multi-graph analysis
    print("\nPerforming multi-graph analysis...")
    multi_analysis = analyzer.analyze_multi_graph()

    # Print comprehensive report
    print_analysis_report(multi_analysis)

    # Save analysis to JSON
    output_file = "multi_graph_analysis_results.json"
    save_analysis_json(multi_analysis, output_file)

    return multi_analysis


# ==============================================================================
# Example Usage
# ==============================================================================

def example_usage():
    """Example of how to use the shared parameter execution."""
    from execute_candidate_set_refactored import (
        Config, setup_external_imports, load_ofa_model,
        setup_rpc_connection, load_candidate_set_from_experiments,
        compile_model, upload_compiled_models, ImageNetteDataLoader
    )
    import vta
    from tvm.contrib import utils
    import glob

    # Setup
    config = Config()
    env = vta.get_env()
    target = env.target if config.device == "vta" else env.target_vta_cpu

    OFADynamicResnetAllMod, _ = setup_external_imports(config.external_repo_root)
    ofa_net = load_ofa_model(config.model_path)
    remote = setup_rpc_connection(env, config)
    ctx = remote.ext_dev(0) if config.device == "vta" else remote.cpu(0)

    # Load models
    schedule_log_files = glob.glob(config.schedule_log_dir)
    arch_mapping, model_ids = load_candidate_set_from_experiments(
        config.candidate_set_json, config.arch_config_json, config.experiment_name
    )

    # Compile models
    compiled_models = {}
    temp = utils.tempdir()

    for model_id in model_ids[:5]:  # Compile first 5 models
        arch = arch_mapping[model_id]
        compiled = compile_model(model_id, arch, ofa_net, env, target, config, schedule_log_files)
        if compiled:
            compiled_models[model_id] = compiled

    # Upload to device
    upload_compiled_models(compiled_models, remote, temp)

    # ============================================================================
    # PHASE 1 ANALYSIS: Analyze graph structure and parameter sharing
    # ============================================================================
    print("\n" + "="*80)
    print("Running Phase 1 Multi-Graph Analysis...")
    print("="*80)

    multi_analysis = analyze_compiled_models(compiled_models)

    # ============================================================================
    # PHASE 1 MERGING: Demonstrate graph merging
    # ============================================================================
    from multi_graph_merger import merge_compiled_models

    print("\n" + "="*80)
    print("Running Phase 1 Graph Merging...")
    print("="*80)

    merger = merge_compiled_models(compiled_models)
    merger.print_summary()

    # ============================================================================
    # Continue with execution (original behavior)
    # ============================================================================

    # Create shared executor (parameters uploaded once here)
    executor = create_shared_executor(compiled_models, env, remote, ctx, config.input_name)

    # Load test images
    image_loader = ImageNetteDataLoader(config.imagenette_base_dir, env.BATCH)

    # Run inference switching between models
    for i, model_id in enumerate(compiled_models.keys()):
        image_data = image_loader.load_and_preprocess(i % 10)
        output = executor.run_inference(model_id, image_data)
        print(f"Model {model_id} - Top prediction: {np.argmax(output[0])}")

    # Print statistics
    stats = executor.get_statistics()
    print("\nExecution Statistics:")
    print(f"  Memory reduction: {stats['parameter_stats']['memory_reduction_pct']:.2f}%")
    print(f"  Cache hit rate: {stats['cache_stats']['hit_rate']:.2%}")


if __name__ == "__main__":
    example_usage()

