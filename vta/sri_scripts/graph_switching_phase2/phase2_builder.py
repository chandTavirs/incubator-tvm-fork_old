"""
Phase 2 Builder - Compiles models and sets up shared parameter infrastructure

Integrates with existing execute_candidate_set_refactored.py to compile models
and extract parameter requirements.
"""
import glob
import sys
import os
from typing import Dict, List, Tuple, Any
import json
import vta
from vta.testing import simulator
from vta.top import graph_pack

# Add parent directory to path to import existing scripts
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from execute_candidate_set_refactored import (
    Config,
    setup_external_imports,
    load_ofa_model,
    setup_rpc_connection,
    load_candidate_set_from_experiments,
    compile_model,
    CompiledModel
)

from multi_runtime_executor import CompiledModelInfo, MultiRuntimeExecutor, MultiRuntimeExecutorBuilder


class Phase2Builder:
    """Builds multi-runtime executor from candidate set models."""

    def __init__(self, config: Config):
        """
        Initialize builder with configuration.

        Args:
            config: Configuration object from execute_candidate_set_refactored
        """
        self.config = config
        self.ofa_model = None
        self.remote = None
        self.ctx = None
        self.env = None
        self.OFADynamicResnetAllMod = None
        self.StaticResNetFromArch = None
        self.schedule_log_files = []

    def setup(self):
        """Set up external dependencies and connections."""
        print(f"\n{'='*80}")
        print("Phase 2 Setup")
        print(f"{'='*80}\n")

        # Get VTA environment
        self.env = vta.get_env()

        # Setup imports - returns model classes, not env
        print("Setting up external imports...")
        self.OFADynamicResnetAllMod, self.StaticResNetFromArch = setup_external_imports(self.config.external_repo_root)

        # Load OFA model
        print("Loading OFA model...")
        self.ofa_model = load_ofa_model(self.config.model_path)

        # Setup RPC connection
        print("Setting up RPC connection...")
        self.remote = setup_rpc_connection(self.env, self.config)
        self.ctx = self.remote.ext_dev(0) if self.config.device == "vta" else self.remote.cpu(0)

        # Load schedule logs
        self.schedule_log_files = glob.glob(self.config.schedule_log_dir)
        print(f"Loaded {len(self.schedule_log_files)} schedule logs")

        print(f"\n✓ Setup complete")
        print(f"{'='*80}\n")

    def load_models_from_candidate_set(self, num_models: int = None) -> List[Dict[str, Any]]:
        """
        Load model architectures from candidate set.

        Args:
            num_models: Number of models to load (None = all)

        Returns:
            List of model architecture dictionaries
        """
        print(f"\n{'='*80}")
        print("Loading Candidate Set")
        print(f"{'='*80}\n")

        print(f"Experiment: {self.config.experiment_name}")

        # Load from candidate set using correct signature
        arch_mapping, model_ids = load_candidate_set_from_experiments(
            self.config.candidate_set_json,
            self.config.arch_config_json,
            self.config.experiment_name
        )

        # Convert to list format with 'architecture' key
        arch_maps = []
        for model_id in model_ids:
            if model_id in arch_mapping:
                arch_maps.append({
                    'architecture': model_id,
                    **arch_mapping[model_id]
                })

        if num_models is not None:
            arch_maps = arch_maps[:num_models]

        print(f"✓ Loaded {len(arch_maps)} model architectures")
        print(f"{'='*80}\n")

        return arch_maps

    def compile_models(
        self,
        arch_maps: List[Dict[str, Any]]
    ) -> Tuple[List[CompiledModelInfo], Dict[str, Any]]:
        """
        Compile all models and extract parameter information.

        Args:
            arch_maps: List of model architecture dictionaries

        Returns:
            Tuple of (compiled_model_infos, ofa_params_dict)
        """
        print(f"\n{'='*80}")
        print(f"Compiling {len(arch_maps)} Models")
        print(f"{'='*80}\n")

        # Get target
        target = self.env.target if self.config.device == "vta" else self.env.target_vta_cpu

        compiled_infos = []
        all_param_names = set()
        ofa_params = {}

        for idx, arch_map in enumerate(arch_maps):
            model_id = arch_map['architecture']

            print(f"[{idx+1}/{len(arch_maps)}] Compiling {model_id}...")

            try:
                # Compile model using correct signature
                compiled = compile_model(
                    model_id,
                    arch_map,
                    self.ofa_model,
                    self.env,
                    target,
                    self.config,
                    self.schedule_log_files
                )

                if compiled is None:
                    print(f"  ✗ Compilation failed")
                    continue

                # Extract parameter names from params dict
                param_names = list(compiled.params.keys())
                all_param_names.update(param_names)

                # Store OFA params (first time we see them)
                for name, value in compiled.params.items():
                    if name not in ofa_params:
                        # Convert to numpy if it's an NDArray
                        if hasattr(value, 'asnumpy'):
                            ofa_params[name] = value.asnumpy()
                        else:
                            ofa_params[name] = value

                # Create CompiledModelInfo
                compiled_info = CompiledModelInfo(
                    model_id=model_id,
                    graph_json=compiled.graph,
                    lib=compiled.lib,
                    param_names=param_names,
                    params=ofa_params  # Use unified params dict
                )

                compiled_infos.append(compiled_info)

                print(f"  ✓ Success ({len(param_names)} parameters)")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n{'='*80}")
        print(f"Compilation Summary:")
        print(f"  Total attempted: {len(arch_maps)}")
        print(f"  Successfully compiled: {len(compiled_infos)}")
        print(f"  Failed: {len(arch_maps) - len(compiled_infos)}")
        print(f"  Unique parameters: {len(all_param_names)}")
        print(f"{'='*80}\n")

        return compiled_infos, ofa_params

    def build_executor(
        self,
        compiled_infos: List[CompiledModelInfo]
    ) -> MultiRuntimeExecutor:
        """
        Build multi-runtime executor from compiled models.

        Args:
            compiled_infos: List of compiled model information

        Returns:
            Configured MultiRuntimeExecutor
        """
        return MultiRuntimeExecutorBuilder.from_compiled_models(
            compiled_infos,
            self.ctx,
            self.remote
        )

    def build_from_experiment(
        self,
        experiment_name: str,
        num_models: int = None
    ) -> MultiRuntimeExecutor:
        """
        One-stop method to build executor from experiment.

        Args:
            experiment_name: Name of experiment in candidate set
            num_models: Number of models to compile (None = all)

        Returns:
            Ready-to-use MultiRuntimeExecutor
        """
        # Update config
        self.config.experiment_name = experiment_name

        # Setup
        self.setup()

        # Load models
        arch_maps = self.load_models_from_candidate_set(num_models)

        # Compile models
        compiled_infos, ofa_params = self.compile_models(arch_maps)

        if not compiled_infos:
            raise RuntimeError("No models compiled successfully")

        # Build executor (ofa_params no longer needed - uses compiled model params)
        executor = self.build_executor(compiled_infos)

        return executor


def quick_build(
    experiment_name: str = "sa_lam_2.0",
    num_models: int = 3,
    device_host: str = "10.42.0.188"
) -> MultiRuntimeExecutor:
    """
    Quick builder function for testing.

    Args:
        experiment_name: Experiment name
        num_models: Number of models to compile
        device_host: RPC device IP

    Returns:
        MultiRuntimeExecutor ready for use
    """
    # Create config
    config = Config()
    config.experiment_name = experiment_name
    config.device_host = device_host

    # Build
    builder = Phase2Builder(config)
    executor = builder.build_from_experiment(experiment_name, num_models)

    return executor


if __name__ == "__main__":
    print("Phase 2 Builder Test")
    print("="*80)
    print("\nThis script requires:")
    print("  1. OFA model checkpoint")
    print("  2. VTA RPC connection")
    print("  3. Candidate set file")
    print("\nFor full testing, run test_phase2_quick.py")



