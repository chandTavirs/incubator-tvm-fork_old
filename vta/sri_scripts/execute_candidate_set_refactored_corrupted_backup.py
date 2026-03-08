"""
Refactored VTA candidate set execution script.
This module provides a modular approach to building and executing neural network models
from a candidate set on VTA hardware.
"""

from __future__ import absolute_import, print_function

import os
import sys
import time
import json
import glob
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import nn
from torchvision import transforms

import tvm
from tvm import te, rpc, autotvm, relay
from tvm.contrib import graph_runtime, utils
from tvm.relay import transform

import vta
from vta.top import graph_pack


# ==============================================================================
# Configuration and Constants
# ==============================================================================

@dataclass
class Config:
    """Configuration for VTA execution."""
    # Paths
    external_repo_root: str = "/home/srchand/Desktop/research/OFA_Obfs"
    model_path: str = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth"
    candidate_set_json: str = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/candidate_sets_results_all_expts.json"
    arch_config_json: str = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json"
    schedule_log_dir: str = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/logs/tuning_logs/vta_1x16x16/candidate_set/*.log"
    imagenette_base_dir: str = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/jupyter_nbs/ImageNette_Images/"

    # Device settings
    device: str = "vta"  # "vta" or "arm_cpu"
    device_host: str = "10.42.0.188"
    device_port: str = "9091"
    bitstring_path: str = "/mnt/hgfs/vmware_ubuntu_sf/vta_4x8x8/vta_new_1x16x16_memory_trojan_runtime_sampling.bit"
    program_device: bool = False

    # Model settings
    model_name: str = "resnet18"
    experiment_name: str = "sa_lam_2.0"
    num_models_to_test: int = 2  # Set to None to test all models

    # Input settings
    input_name: str = "input0"
    input_shape: Tuple[int, int, int, int] = None  # Will be set from env.BATCH

    # Build settings
    opt_level: int = 3
    global_scale: float = 8.0
    skip_conv_layers: List[int] = None

    # Run settings
    run_single_time = False # if True, we only run each model once (instead of multiple times to get std dev) to save time
    run_num = 4  # number of times we run module for a single measurement
    run_rep = 3  # number of measurements (we derive std dev from this)

    def __post_init__(self):
        if self.skip_conv_layers is None:
            self.skip_conv_layers = [0]


# Graph packing configuration for different model types
PACK_DICT = {
    "resnet18_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet18": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
    "resnet34": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
    "resnet50": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
    "resnet101": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
    "vgg11": ["nn.max_pool2d", "nn.dense"],
    "vgg16": ["nn.max_pool2d", "nn.dense"],
    "resnet34_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet18_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet50_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet101_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "mobilenetv2_1.0": ["nn.max_pool2d", "nn.global_avg_pool2d"]
}


# ==============================================================================
# Module Import Utilities
# ==============================================================================

def setup_external_imports(external_repo_root: str):
    """Setup imports from external OFA repository."""
    if external_repo_root not in sys.path:
        sys.path.insert(0, external_repo_root)

    # Clear any cached local shim modules
    for module_name in ["ofa_base_models", "architecture_defense"]:
        _mod = sys.modules.get(module_name)
        if _mod is not None:
            try:
                _mod_file = getattr(_mod, "__file__", "") or ""
                if "TVM_Intel_Fork/tvm/vta/sri_scripts/jupyter_nbs" in _mod_file:
                    del sys.modules[module_name]
            except Exception:
                sys.modules.pop(module_name, None)

    # Import required modules
    try:
        from ofa_base_models import OFADynamicResnetAllMod
        from architecture_defense import StaticResNetFromArch
    except (ModuleNotFoundError, ImportError):
        if sys.path[0] != external_repo_root:
            sys.path.insert(0, external_repo_root)
        sys.modules.pop("ofa_base_models", None)
        sys.modules.pop("architecture_defense", None)
        from ofa_base_models import OFADynamicResnetAllMod
        from architecture_defense import StaticResNetFromArch
    else:
        # Print module locations for debugging
        import ofa_base_models as _obm
        print("ofa_base_models loaded from:", getattr(_obm, "__file__", None))
        try:
            import architecture_defense as _ad
            print("architecture_defense loaded from:", getattr(_ad, "__file__", None))
        except ImportError:
            pass

    return OFADynamicResnetAllMod, StaticResNetFromArch


# ==============================================================================
# Architecture Loading Utilities
# ==============================================================================

def load_arch_mapping(path: str) -> Dict[str, Any]:
    """Load architectures JSON and normalize to a mapping {id: architecture_dict}.

    Supported input formats:
    - A dict mapping id -> architecture dict (legacy)
    - A dict with key 'architectures' containing a list of items with fields 'id' and 'architecture'
    - A top-level list of items with 'id' and 'architecture'

    Note: For experiment results files, use load_candidate_set_from_experiments() instead.
    """
    with open(path, 'r') as f:
        data = json.load(f)

    # Case 1: already a mapping from id -> arch
    if isinstance(data, dict) and all(isinstance(v, dict) and 'residual_depth_list' in v for v in data.values()):
        return data

    # Case 2: top-level dict with 'architectures' list
    if isinstance(data, dict) and 'architectures' in data and isinstance(data['architectures'], list):
        mapping = {}
        for item in data['architectures']:
            if 'id' in item and 'architecture' in item:
                mapping[item['id']] = item['architecture']
            elif 'id' in item and 'arch' in item:
                mapping[item['id']] = item['arch']
            elif 'id' in item:
                mapping[item['id']] = item
        return mapping

    # Case 3: top-level list of architecture items
    if isinstance(data, list):
        mapping = {}
        for idx, item in enumerate(data):
            if isinstance(item, dict) and 'id' in item and 'architecture' in item:
                mapping[item['id']] = item['architecture']
            elif isinstance(item, dict) and 'id' in item:
                mapping[item['id']] = item
            else:
                mapping[f'arch_{idx}'] = item
        return mapping

    raise ValueError(f"Unsupported architecture file format: {path}")


def load_candidate_set_from_experiments(
    results_path: str,
    arch_path: str,
    experiment_name: str
) -> Tuple[Dict[str, Any], List[str]]:
    """Load candidate set and architectures from experiment results file.

    Args:
        results_path: Path to the experiment results JSON
        arch_path: Path to the architectures JSON file
        experiment_name: Name of the experiment to load (e.g., 'greedy_swap', 'sa_lam_2.0')

    Returns:
        Tuple of (arch_mapping, model_ids)
    """
    with open(results_path, 'r') as f:
        results = json.load(f)

    if experiment_name not in results:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. "
            f"Available experiments: {list(results.keys())}"
        )

    model_ids = results[experiment_name]['ids']

    print(f"Loading experiment '{experiment_name}':")
    print(f"  - {len(model_ids)} models selected")
    print(f"  - Mean accuracy: {results[experiment_name].get('mean_acc', 'N/A')}")
    print(f"  - ASPT: {results[experiment_name].get('ASPT', 'N/A')}")
    print(f"  - Max pairwise: {results[experiment_name].get('max_pairwise', 'N/A')}")

    arch_mapping = load_arch_mapping(arch_path)

    # Filter to only include the selected model IDs
    filtered_arch_mapping = {
        model_id: arch_mapping[model_id]
        for model_id in model_ids
        if model_id in arch_mapping
    }

    if len(filtered_arch_mapping) != len(model_ids):
        missing = set(model_ids) - set(filtered_arch_mapping.keys())
        print(f"Warning: {len(missing)} model IDs not found in architecture file: {list(missing)[:5]}")

    return filtered_arch_mapping, model_ids


# ==============================================================================
# Model Loading and Setup
# ==============================================================================

def load_ofa_model(model_path: str, device: torch.device = torch.device('cpu')):
    """Load OFA model from checkpoint."""
    try:
        from ofa_base_models import OFADynamicResnetAllMod
    except ImportError:
        raise ImportError("Could not import OFADynamicResnetAllMod. Ensure setup_external_imports() was called.")

    ofa_net = OFADynamicResnetAllMod()
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    else:
        state = checkpoint

    ofa_net.load_state_dict(state, strict=False)
    return ofa_net


        standalone_net = create_standalone_net_from_arch(config.model_path, arch, ofa_net)
        pytorch_model = standalone_net
    """Create a standalone PyTorch model from the given architecture and OFA net."""
    try:
        from architecture_defense import StaticResNetFromArch
    except ImportError:
) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:

    standalone_net = StaticResNetFromArch(
        num_classes=10,
        width_mult_list=(0.5, 1.0, 2.0)
    )
    standalone_net.load_weights_from_ofa_checkpoint(checkpoint_path=model_path, ofa_model=ofa_net)
    return standalone_net

# ==============================================================================
# RPC and Device Setup
) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]:

def setup_rpc_connection(env, config: Config):
    """Setup RPC connection to VTA device or simulator."""
    remote = None

    scripted_model = torch.jit.trace(pytorch_model, input_data).eval()
    runtime_module: Optional[graph_runtime.GraphModule] = None  # Runtime with params loaded
        else:
            remote = autotvm.measure.request_remote(
                env.TARGET, tracker_host, int(tracker_port), timeout=10000
            )

        # Reconfigure the JIT runtime and FPGA
        reconfig_start = time.time()
        vta.reconfig_runtime(remote)
        if config.program_device:
            vta.program_fpga(remote, bitstream=config.bitstring_path)
        reconfig_time = time.time() - reconfig_start
        print(f"Reconfigured FPGA and RPC runtime in {reconfig_time:.2f}s!")

    else:
        # Simulation mode
        remote = rpc.LocalSession()


    with torch.no_grad():
        scripted_model = torch.jit.trace(pytorch_model, input_data).eval()

    shape_list = [(input_name, input_shape)]

    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    mod = relay.transform.InferType()(mod)

    return mod, params
            vta.program_fpga(remote, bitstream="vta.bitstream")

    return remote


# ==============================================================================
# Relay Graph Building
# ==============================================================================

@dataclass
class CompiledModel:
    """Container for compiled model artifacts."""
    model_id: str
    graph: str
    lib: tvm.runtime.Module
    params: Dict[str, tvm.nd.NDArray]
    remote_lib: Optional[tvm.runtime.Module] = None


def pytorch_to_relay(
    pytorch_model: nn.Module,
    input_shape: List[int],
    input_name: str = "input0"
) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
    """Convert PyTorch model to Relay IR."""
    pytorch_model.eval()
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(pytorch_model, input_data).eval()
    shape_list = [(input_name, input_shape)]

    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    mod = relay.transform.InferType()(mod)

    return mod, params


def apply_quantization_and_packing(
    mod: tvm.ir.IRModule,
    params: Dict[str, tvm.nd.NDArray],
    env,
    model_name: str,
    config: Config
) -> tvm.relay.Function:
    """Apply quantization and graph packing for VTA."""
    with tvm.transform.PassContext(opt_level=config.opt_level, disabled_pass={"AlterOpLayout"}):
        with relay.quantize.qconfig(global_scale=config.global_scale, skip_conv_layers=config.skip_conv_layers):
            mod = relay.quantize.quantize(mod, params=params)

        # Perform graph packing
        assert env.BLOCK_IN == env.BLOCK_OUT
        relay_prog = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_IN,
        print(f"Initializing runtime for {model_id}...")
            device_annot=(env.TARGET == "intelfocl"),
        )

    return relay_prog


def build_relay_graph(
    relay_prog: tvm.relay.Function,
    target,
    target_host,
    params: Dict[str, tvm.nd.NDArray],
    env,
    config: Config
) -> Tuple[str, tvm.runtime.Module, Dict[str, tvm.nd.NDArray]]:
    """Build Relay graph to executable format."""
    if target.device_name != "vta":
        # Create runtime
        runtime_module = create_runtime(
            model_data.graph,
            model_data.remote_lib,
            env,
            remote,
        # Load parameters once
        runtime_module.set_input(**model_data.params)
            build_target = {"cpu": env.target_vta_cpu, "ext_dev": target}

        with vta.build_config(opt_level=config.opt_level, disabled_pass={"AlterOpLayout"}):
            graph, lib, params = relay.build(
                relay_prog, target=build_target, params=params, target_host=target_host
            )

    return graph, lib, params


        # Store the initialized runtime
        model_data.runtime_module = runtime_module
    print(f"\nAll {len(compiled_models)} runtimes initialized!")
    return compiled_models
    """Compile a single model from the candidate set.

    Returns:
        CompiledModel object if successful, None if compilation fails
    """
    print(f"\n{'='*80}")
    print(f"Compiling Model: {model_id}")
    print(f"{'='*80}")
    print(f"Architecture: {arch}")

    try:
        # Set active subnet
        ofa_net.set_active_subnet(arch)
        pytorch_model = ofa_net

        with autotvm.tophub.context(target, extra_files=schedule_log_files):
            # Build input shape
            input_shape = [env.BATCH, 3, 224, 224]

            build_start = time.time()

            # Convert PyTorch to Relay

            # Build the graph
            graph, lib, params = build_relay_graph(
                relay_prog, target, env.target_host, params, env, config
            )

            build_time = time.time() - build_start
            print(f"{config.model_name} inference graph built in {build_time:.2f}s!")

            return CompiledModel(
                model_id=model_id,
                graph=graph,
                lib=lib,
                params=params
            )

    except Exception as e:
        print(f"FAILED to compile model {model_id}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        return None


def upload_compiled_models(
    compiled_models: Dict[str, CompiledModel],
    remote,
    temp_dir
) -> Dict[str, CompiledModel]:
    """Upload compiled models to remote device."""
    print(f"\n{'='*80}")
    print("Uploading Libraries to Remote Device")
    print(f"{'='*80}")

    for model_id, model_data in compiled_models.items():
        lib_path = temp_dir.relpath(f"graphlib_{model_id}.tar")
        model_data.lib.export_library(lib_path)
        remote.upload(lib_path)
        model_data.remote_lib = remote.load_module(f"graphlib_{model_id}.tar")
        print(f"Uploaded: {model_id}")

    return compiled_models


# ==============================================================================
# Image Loading and Preprocessing
# ==============================================================================

class ImageNetteDataLoader:
    """Helper class for loading and preprocessing ImageNette images."""

    def __init__(self, base_dir: str, batch_size: int):
        self.base_dir = base_dir
        self.batch_size = batch_size

        # Load class names
        imagenette_classes_path = os.path.join(base_dir, "imagenette_classes.txt")
        self.classes = eval(open(imagenette_classes_path).read())

        # Define image paths
        self.image_paths = [
            f"{base_dir}/images/original/tench.JPEG",
            f"{base_dir}/images/original/english_springer.JPEG",
            f"{base_dir}/images/original/cassette_player.JPEG",
            f"{base_dir}/images/original/chainsaw.JPEG",
            f"{base_dir}/images/original/church.JPEG",
            f"{base_dir}/images/original/french_horn.JPEG",
            f"{base_dir}/images/original/garbage_truck.JPEG",
            f"{base_dir}/images/original/gas_pump.JPEG",
            f"{base_dir}/images/original/golf_ball.JPEG",
            f"{base_dir}/images/original/parachute.JPEG",
        ]

    def load_and_preprocess(self, idx: int, show_image: bool = False) -> np.ndarray:
        """Load and preprocess image at given index."""
        image = Image.open(self.image_paths[idx]).resize((224, 224))

        if show_image:
            plt.imshow(image)
            plt.show()

        # Convert to numpy and normalize
        image = np.array(image) - np.array([123.0, 117.0, 104.0])
        image /= np.array([58.395, 57.12, 57.375])

        # Transpose to CHW format
        image = image.transpose((2, 0, 1))

        # Add batch dimension and repeat
        image = image[np.newaxis, :]
        image = np.repeat(image, self.batch_size, axis=0)

        return image


# ==============================================================================
# Inference Execution
# ==============================================================================


        print(f"  ✓ Runtime initialized and parameters loaded for {model_id}")

    print(f"\nAll {len(compiled_models)} runtimes initialized!")
    return compiled_models
        return graph_runtime.create(graph, lib, ctxes)
    else:
        return graph_runtime.create(graph, lib, ctx)


def run_inference(
    model_data: CompiledModel,
    image_data: np.ndarray,
    input_name: str,
    env,
    """Run inference on a single image with the given model.

    This function assumes that initialize_all_runtimes() has already been called,
    so the runtime_module is created and parameters are already loaded.

    This function ONLY:
    1. Sets the input image data
    2. Runs inference (m.run() or timer)
    3. Returns the output

    Args:
        model_data: CompiledModel with runtime_module already initialized
        image_data: Input image data
        input_name: Name of the input tensor
        env: VTA environment
    """Run inference on a single image with the given model."""
    # Create runtime
    m = create_runtime(model_data.graph, model_data.remote_lib, env, remote, ctx)
        mean = tcost.mean * 1000
        print("\nPerformed inference in %.2fms (std = %.2f) for %d samples" % (mean, std, env.BATCH))
        print("Average per sample inference time: %.2fms" % (mean / env.BATCH))

    # Get output
    output = m.get_output(0, tvm.nd.empty((env.BATCH, 10), "float32", remote.cpu(0)))
    return output.asnumpy()


def print_top5_predictions(output: np.ndarray, classes: List[str], model_name: str, batch_idx: int = 0):
    """Print top-5 predictions for a given output."""
    top_categories = np.argsort(output[batch_idx])

    print(f"\n{model_name} prediction for sample {batch_idx}")
    print(f"\t#1: {classes[top_categories[-1]]}")
    # Set inputs
    m.set_input(**model_data.params)
# Main Execution
# ==============================================================================

def main():
    """Main execution function."""
    # Initialize configuration
    config = Config()

    # Setup TVM and VTA
    assert tvm.runtime.enabled("rpc"), "TVM must be compiled with RPC=1"
    env = vta.get_env()

    # Update config with env-specific values
    config.input_shape = (env.BATCH, 3, 224, 224)

    # Setup target
    target = env.target if config.device == "vta" else env.target_vta_cpu

    # Setup external imports
    OFADynamicResnetAllMod, StaticResNetFromArch = setup_external_imports(config.external_repo_root)

    # Load OFA model
    print("Loading OFA model...")
    ofa_net = load_ofa_model(config.model_path)

    # Setup RPC connection
    print("\nSetting up RPC connection...")
    remote = setup_rpc_connection(env, config)
    ctx = remote.ext_dev(0) if config.device == "vta" else remote.cpu(0)

    # Load schedule logs
    schedule_log_files = glob.glob(config.schedule_log_dir)
    print(f"Loaded {len(schedule_log_files)} schedule log files")

    # Load candidate set and architectures
    print("\nLoading candidate set...")
    arch_mapping, model_ids = load_candidate_set_from_experiments(
        config.candidate_set_json,
        config.arch_config_json,
        config.experiment_name
    )

    # Determine which models to compile
    if config.num_models_to_test is not None:
            print(f"  Error: {type(e).__name__}: {str(e)}")
    return results
    print(f"{'='*80}")

    compiled_models = {}
    failed_models = []

    temp = utils.tempdir()

    for i_net, model_id in enumerate(model_ids):
        arch = arch_mapping[model_id]
        compiled = compile_model(
            model_id, arch, ofa_net, env, target, config, schedule_log_files
        )

        if compiled is not None:
            compiled_models[model_id] = compiled
        else:
            failed_models.append((i_net, model_id))

    print(f"\nSuccessfully compiled: {len(compiled_models)}/{len(model_ids)} models")
    if failed_models:
        print(f"Failed models: {failed_models}")

    # Upload compiled models
    if compiled_models:
        upload_compiled_models(compiled_models, remote, temp)
        
            print(f"✗ Failed inference for {model_id}")
            print(f"  Error: {type(e).__name__}: {str(e)}")


        # Setup image data loader
        print("\nSetting up image data loader...")
    return results

        # Run inference on test images
        print(f"\n{'='*80}")
        print("Running Inference")
        print(f"{'='*80}")

        # Load test image
        image_data = image_loader.load_and_preprocess(0, show_image=True)
        
        # Run inference on all models with the same image
        # Test first model with first image
        first_model_id = list(compiled_models.keys())[0]
        first_model = compiled_models[first_model_id]

        results = run_inference_all_models(
            compiled_models,
            image_data,
            config.input_name,
            env,
            remote,
            config,
            image_loader.classes,
            config.model_name
        )
        
        print(f"\nSuccessfully ran inference on {len(results)}/{len(compiled_models)} models")

    # Print summary
    print(f"\n{'='*80}")
    print("Execution Summary")
    print(f"{'='*80}")
    print(f"Total models in candidate set: {len(model_ids)}")
    print(f"Successfully compiled: {len(compiled_models)}")
    print(f"Failed: {len(failed_models)}")
    if failed_models:
        print(f"Failed model IDs: {[model_id for _, model_id in failed_models]}")


if __name__ == "__main__":
    main()

