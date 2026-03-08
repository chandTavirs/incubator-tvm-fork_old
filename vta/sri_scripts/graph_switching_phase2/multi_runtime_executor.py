"""
Multi-Runtime Executor for Phase 2 Quick Win

Manages multiple graph runtimes with shared parameters for fast model switching.

Key Features:
- Compile and manage multiple models
- Shared parameter pool (zero-copy)
- Fast model switching (<1ms)
- Unified inference interface
"""

from typing import Dict, List, Any, Optional
import time
import numpy as np

import tvm
from tvm import rpc
from tvm.contrib import graph_runtime, utils

from shared_param_manager import SharedParamManager


class CompiledModelInfo:
    """Information about a compiled model."""

    def __init__(self, model_id: str, graph_json: str, lib: tvm.runtime.Module,
                 param_names: List[str], params: Dict[str, np.ndarray]):
        self.model_id = model_id
        self.graph_json = graph_json
        self.lib = lib
        self.param_names = param_names
        self.params = params


class MultiRuntimeExecutor:
    """Executor that manages multiple runtimes with shared parameters."""

    def __init__(self, ctx: tvm.runtime.TVMContext, remote=None):
        """
        Initialize multi-runtime executor.

        Args:
            ctx: TVM context (device) for execution
            remote: RPC remote session (if using remote device)
        """
        self.ctx = ctx
        self.remote = remote

        # Shared parameter manager
        self.param_manager = SharedParamManager(ctx)

        # Runtime storage
        self.runtimes: Dict[str, tvm.contrib.graph_runtime.GraphModule] = {}
        self.model_ids: List[str] = []
        self.compiled_models: Dict[str, CompiledModelInfo] = {}

        # Active model
        self.active_model_id: Optional[str] = None
        self.active_runtime: Optional[tvm.contrib.graph_runtime.GraphModule] = None

        # Statistics
        self.inference_times: Dict[str, List[float]] = {}
        self.switch_times: List[float] = []

        # Temp directory for remote upload
        self._temp_dir = utils.tempdir() if self.remote is not None else None

    def add_model(self, compiled_info: CompiledModelInfo) -> None:
        """
        Add a compiled model to the executor.

        Args:
            compiled_info: CompiledModelInfo object with model details
        """
        model_id = compiled_info.model_id

        print(f"  Adding model: {model_id}")

        # Store compiled info
        self.compiled_models[model_id] = compiled_info
        self.model_ids.append(model_id)

        # Handle remote upload if using RPC
        lib = compiled_info.lib
        print(f"    Initial lib type_key: {lib.type_key if hasattr(lib, 'type_key') else 'N/A'}")

        if self.remote is not None:
            # Export library to temp file and upload
            lib_filename = f"{model_id}_lib.tar"
            lib_path = self._temp_dir.relpath(lib_filename)
            print(f"    Exporting to: {lib_path}")
            compiled_info.lib.export_library(lib_path)

            # Upload to remote
            print(f"    Uploading...")
            self.remote.upload(lib_path)

            # Load on remote - this creates an RPC module
            print(f"    Loading remote module...")
            lib = self.remote.load_module(lib_filename)
            print(f"    Remote lib type_key: {lib.type_key}")

        # Create runtime - handle intelfocl multi-context
        import vta
        env = vta.get_env()

        print(f"    TARGET: {env.TARGET}")
        print(f"    ctx: {self.ctx}")
        print(f"    ctx.device_type: {self.ctx.device_type}")

        if env.TARGET == "intelfocl":
            # intelfocl requires both ext_dev and cpu contexts
            ctxes = [self.remote.ext_dev(0), self.remote.cpu(0)]
            print(f"    Creating runtime with intelfocl contexts...")
            runtime = graph_runtime.create(
                compiled_info.graph_json,
                lib,
                ctxes
            )
        else:
            # Other targets use single context
            print(f"    Creating runtime with single context...")
            runtime = graph_runtime.create(
                compiled_info.graph_json,
                lib,
                self.ctx
            )

        self.runtimes[model_id] = runtime
        self.inference_times[model_id] = []

        print(f"    ✓ Runtime created")

    def upload_shared_params(self) -> None:
        """
        Upload parameters from all compiled models to shared device memory.
        Automatically detects and shares parameters with matching shapes.
        """
        # Collect parameters from all compiled models
        all_model_params = {}
        for model_id, compiled_info in self.compiled_models.items():
            all_model_params[model_id] = compiled_info.params

        # Upload with smart sharing
        self.param_manager.upload_params(all_model_params)

    def link_all_params(self) -> None:
        """Link all runtimes to shared parameters using zero-copy."""
        print(f"\n{'='*80}")
        print("Linking Runtime Parameters to Shared Memory")
        print(f"{'='*80}\n")

        for model_id in self.model_ids:
            runtime = self.runtimes[model_id]
            compiled_info = self.compiled_models[model_id]

            self.param_manager.link_runtime_params(
                runtime,
                model_id,
                compiled_info.param_names
            )

        print(f"\n✓ Linked all {len(self.model_ids)} models to shared parameters")
        print(f"{'='*80}\n")

    def set_active_model(self, model_id: str) -> None:
        """
        Switch to a different model for execution.

        Args:
            model_id: ID of model to activate
        """
        if model_id not in self.runtimes:
            raise ValueError(f"Model {model_id} not found. Available: {self.model_ids}")

        start = time.time()

        self.active_model_id = model_id
        self.active_runtime = self.runtimes[model_id]

        switch_time = time.time() - start
        self.switch_times.append(switch_time)

    def set_input(self, input_name: str, data: np.ndarray) -> None:
        """
        Set input tensor for active model.

        Args:
            input_name: Name of input (e.g., "data", "input0")
            data: Input data as numpy array
        """
        if self.active_runtime is None:
            raise RuntimeError("No active model. Call set_active_model() first.")

        self.active_runtime.set_input(input_name, data)

    def run(self) -> None:
        """Execute inference on active model."""
        if self.active_runtime is None:
            raise RuntimeError("No active model. Call set_active_model() first.")

        start = time.time()
        self.active_runtime.run()
        inference_time = time.time() - start

        self.inference_times[self.active_model_id].append(inference_time)

    def get_output(self, index: int) -> tvm.nd.NDArray:
        """
        Get output tensor from active model.

        Args:
            index: Output index (usually 0)

        Returns:
            Output tensor
        """
        if self.active_runtime is None:
            raise RuntimeError("No active model. Call set_active_model() first.")

        return self.active_runtime.get_output(index)

    def get_num_outputs(self) -> int:
        """Get number of outputs for active model."""
        if self.active_runtime is None:
            raise RuntimeError("No active model. Call set_active_model() first.")

        return self.active_runtime.get_num_outputs()

    def benchmark_model_switching(self, num_switches: int = 100) -> Dict[str, float]:
        """
        Benchmark model switching speed.

        Args:
            num_switches: Number of switches to perform

        Returns:
            Dictionary with timing statistics
        """
        if len(self.model_ids) < 2:
            print("⚠️  Need at least 2 models for switching benchmark")
            return {}

        print(f"\n{'='*80}")
        print(f"Benchmarking Model Switching ({num_switches} switches)")
        print(f"{'='*80}\n")

        times = []
        for i in range(num_switches):
            # Switch between first two models
            target_id = self.model_ids[i % 2]

            start = time.time()
            self.set_active_model(target_id)
            switch_time = (time.time() - start) * 1000  # ms
            times.append(switch_time)

        results = {
            'num_switches': num_switches,
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times)
        }

        print(f"Results:")
        print(f"  Mean: {results['mean_ms']:.3f} ms")
        print(f"  Std:  {results['std_ms']:.3f} ms")
        print(f"  Min:  {results['min_ms']:.3f} ms")
        print(f"  Max:  {results['max_ms']:.3f} ms")
        print(f"\n{'='*80}\n")

        return results

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        param_stats = self.param_manager.get_memory_stats()
        savings = self.param_manager.estimate_memory_savings(len(self.model_ids))

        return {
            'num_models': len(self.model_ids),
            'shared_params': param_stats,
            'memory_savings': savings
        }

    def print_summary(self) -> None:
        """Print comprehensive executor summary."""
        print(f"\n{'='*80}")
        print("MULTI-RUNTIME EXECUTOR SUMMARY")
        print(f"{'='*80}\n")

        print(f"📊 Configuration:")
        print(f"  Device: {self.ctx}")
        print(f"  Total models: {len(self.model_ids)}")
        print(f"  Active model: {self.active_model_id}")

        # Memory stats
        mem_stats = self.get_memory_stats()
        print(f"\n💾 Memory Statistics:")
        shared = mem_stats['shared_params']
        print(f"  Shared params: {shared['total_params']}")
        print(f"  Shared memory: {shared['total_memory_mb']:.2f} MB")

        savings = mem_stats['memory_savings']
        print(f"\n💰 Memory Savings:")
        print(f"  Without sharing: {savings['separate_memory_mb']:.2f} MB")
        print(f"  With sharing: {savings['shared_memory_mb']:.2f} MB")
        print(f"  Savings: {savings['savings_mb']:.2f} MB ({savings['savings_percent']:.1f}%)")

        # Performance stats
        if self.switch_times:
            print(f"\n⚡ Model Switching:")
            print(f"  Switches performed: {len(self.switch_times)}")
            print(f"  Avg switch time: {np.mean(self.switch_times)*1000:.3f} ms")

        if any(self.inference_times.values()):
            print(f"\n🚀 Inference Performance:")
            total_inferences = sum(len(times) for times in self.inference_times.values())
            print(f"  Total inferences: {total_inferences}")

            # Show per-model stats if available
            for model_id in self.model_ids[:3]:  # Show first 3
                if self.inference_times[model_id]:
                    times = self.inference_times[model_id]
                    print(f"  {model_id}: {np.mean(times)*1000:.2f} ms avg "
                          f"({len(times)} runs)")

        print(f"\n{'='*80}\n")

    def cleanup(self) -> None:
        """Clean up resources."""
        print("Cleaning up executor...")
        self.runtimes.clear()
        self.param_manager.cleanup()
        self.active_runtime = None
        self.active_model_id = None
        # utils.tempdir() handles cleanup automatically
        print("✓ Cleanup complete")


class MultiRuntimeExecutorBuilder:
    """Builder for creating MultiRuntimeExecutor from compiled models."""

    @staticmethod
    def from_compiled_models(
        compiled_models: List[CompiledModelInfo],
        ctx: tvm.runtime.TVMContext,
        remote=None
    ) -> MultiRuntimeExecutor:
        """
        Build executor from pre-compiled models.

        Args:
            compiled_models: List of CompiledModelInfo objects
            ctx: TVM context for execution
            remote: RPC remote session (if using remote device)

        Returns:
            Configured MultiRuntimeExecutor
        """
        print(f"\n{'='*80}")
        print(f"Building Multi-Runtime Executor")
        print(f"{'='*80}\n")

        print(f"Number of models: {len(compiled_models)}")
        print(f"Context: {ctx}")

        # Create executor
        executor = MultiRuntimeExecutor(ctx, remote)

        # Add all models
        print(f"\nAdding models...")
        for compiled_info in compiled_models:
            executor.add_model(compiled_info)

        print(f"\n✓ Added {len(compiled_models)} models")

        # Upload shared parameters (extracted from compiled models)
        executor.upload_shared_params()

        # Link all runtimes to shared params
        executor.link_all_params()

        # Set first model as active
        if executor.model_ids:
            executor.set_active_model(executor.model_ids[0])
            print(f"✓ Set active model: {executor.model_ids[0]}")

        print(f"\n{'='*80}")
        print("✓ Executor ready!")
        print(f"{'='*80}\n")

        return executor


if __name__ == "__main__":
    print("Testing MultiRuntimeExecutor...")



    print("(This requires actual compiled models - see test_phase2_quick.py)")

