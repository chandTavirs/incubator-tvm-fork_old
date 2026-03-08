"""
Refactored VTA candidate set execution script.
This module provides a modular approach to building and executing neural network models
from a candidate set on VTA hardware.

Key improvements:
1. Separate runtime initialization from inference execution
2. Load parameters once, reuse across multiple inferences
3. Efficient model switching without overhead
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

import tvm
from tvm import te, rpc, autotvm, relay
from tvm.contrib import graph_runtime, utils
from tvm.relay import transform

import vta
from vta.top import graph_pack


# ==============================================================================
# Configuration
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
    bitstream_path: str = "/mnt/hgfs/vmware_ubuntu_sf/vta_4x8x8/vta_new_1x16x16_memory_trojan_runtime_sampling.bit"

    # Device settings
    device: str = "vta"
    device_host: str = "10.42.0.188"
    device_port: str = "9091"
    program_device: bool = False

    # Model settings
    model_name: str = "resnet18"
    experiment_name: str = "sa_lam_2.0"
    num_models_to_test: int = 1

    # Input settings
    input_name: str = "input0"

    # Build settings
    opt_level: int = 3
    global_scale: float = 8.0
    skip_conv_layers: List[int] = None

    # Run settings
    run_single_time: bool = True
    run_num: int = 4
    run_rep: int = 3

    def __post_init__(self):
        if self.skip_conv_layers is None:
            self.skip_conv_layers = [0]


PACK_DICT = {
    "resnet18": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
    "resnet18_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
    "resnet50": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
}


# ==============================================================================
# External Imports
# ==============================================================================

def setup_external_imports(external_repo_root: str):
    """Setup imports from external OFA repository."""
    if external_repo_root not in sys.path:
        sys.path.insert(0, external_repo_root)

    for module_name in ["ofa_base_models", "architecture_defense"]:
        _mod = sys.modules.get(module_name)
        if _mod is not None:
            try:
                _mod_file = getattr(_mod, "__file__", "") or ""
                if "TVM_Intel_Fork/tvm/vta/sri_scripts/jupyter_nbs" in _mod_file:
                    del sys.modules[module_name]
            except Exception:
                sys.modules.pop(module_name, None)

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

    return OFADynamicResnetAllMod, StaticResNetFromArch


# ==============================================================================
# Architecture Loading
# ==============================================================================

def load_arch_mapping(path: str) -> Dict[str, Any]:
    """Load architectures JSON."""
    with open(path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and all(isinstance(v, dict) and 'residual_depth_list' in v for v in data.values()):
        return data

    if isinstance(data, dict) and 'architectures' in data:
        mapping = {}
        for item in data['architectures']:
            if 'id' in item and 'architecture' in item:
                mapping[item['id']] = item['architecture']
            elif 'id' in item:
                mapping[item['id']] = item
        return mapping

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

    raise ValueError(f"Unsupported architecture file format")


def load_candidate_set_from_experiments(
    results_path: str,
    arch_path: str,
    experiment_name: str
) -> Tuple[Dict[str, Any], List[str]]:
    """Load candidate set from experiment results."""
    with open(results_path, 'r') as f:
        results = json.load(f)

    if experiment_name not in results:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    model_ids = results[experiment_name]['ids']
    print(f"Loading experiment '{experiment_name}': {len(model_ids)} models")

    arch_mapping = load_arch_mapping(arch_path)
    filtered_arch_mapping = {
        model_id: arch_mapping[model_id]
        for model_id in model_ids
        if model_id in arch_mapping
    }

    return filtered_arch_mapping, model_ids


# ==============================================================================
# Model Loading
# ==============================================================================

def load_ofa_model(model_path: str):
    """Load OFA model from checkpoint."""
    from ofa_base_models import OFADynamicResnetAllMod

    ofa_net = OFADynamicResnetAllMod()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    else:
        state = checkpoint

    ofa_net.load_state_dict(state, strict=False)
    return ofa_net

def load_static_resnet_from_arch(model_path:str, arch: Dict[str, Any], ofa_net):
    """Create StaticResNetFromArch from architecture dict."""
    from architecture_defense import StaticResNetFromArch

    standalone_net = StaticResNetFromArch(
        target_arch=arch,
        num_classes=10,
        width_mult_list=(0.5, 1.0, 2.0)
    )
    standalone_net.load_weights_from_ofa_checkpoint(checkpoint_path=model_path, ofa_model=ofa_net)

    return standalone_net


# ==============================================================================
# RPC Setup
# ==============================================================================

def setup_rpc_connection(env, config: Config):
    """Setup RPC connection."""
    if env.TARGET not in ["sim", "tsim", "intelfocl"]:
        tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
        tracker_port = os.environ.get("TVM_TRACKER_PORT", None)

        if not tracker_host or not tracker_port:
            remote = rpc.connect(config.device_host, int(config.device_port))
        else:
            remote = autotvm.measure.request_remote(
                env.TARGET, tracker_host, int(tracker_port), timeout=10000
            )

        reconfig_start = time.time()
        vta.reconfig_runtime(remote)
        if config.program_device:
            vta.program_fpga(remote,config.bitstream_path)

        reconfig_time = time.time() - reconfig_start
        print(f"Reconfigured runtime in {reconfig_time:.2f}s")
    else:
        remote = rpc.LocalSession()
        if env.TARGET in ["intelfocl"]:
            vta.program_fpga(remote, bitstream="vta.bitstream")

    return remote


# ==============================================================================
# Compiled Model Container
# ==============================================================================

@dataclass
class CompiledModel:
    """Container for compiled model artifacts."""
    model_id: str
    graph: str
    lib: tvm.runtime.Module
    params: Dict[str, tvm.nd.NDArray]
    remote_lib: Optional[tvm.runtime.Module] = None
    runtime_module: Optional[graph_runtime.GraphModule] = None


# ==============================================================================
# Relay Building
# ==============================================================================

def pytorch_to_relay(
    pytorch_model: nn.Module,
    input_shape: List[int],
    input_name: str = "input0"
) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
    """Convert PyTorch model to Relay IR."""
    pytorch_model.eval()
    input_data = torch.randn(input_shape)

    with torch.no_grad():
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
    """Apply quantization and graph packing."""
    with tvm.transform.PassContext(opt_level=config.opt_level, disabled_pass={"AlterOpLayout"}):
        with relay.quantize.qconfig(global_scale=config.global_scale, skip_conv_layers=config.skip_conv_layers):
            mod = relay.quantize.quantize(mod, params=params)

        assert env.BLOCK_IN == env.BLOCK_OUT
        relay_prog = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_IN,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=PACK_DICT[model_name][0],
            stop_name=PACK_DICT[model_name][1],
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
    """Build Relay graph."""
    if target.device_name != "vta":
        with tvm.transform.PassContext(opt_level=config.opt_level, disabled_pass={"AlterOpLayout"}):
            graph, lib, params = relay.build(
                relay_prog, target=target, params=params, target_host=target_host
            )
    else:
        build_target = target
        if env.TARGET == "intelfocl":
            build_target = {"cpu": env.target_vta_cpu, "ext_dev": target}

        with vta.build_config(opt_level=config.opt_level, disabled_pass={"AlterOpLayout"}):
            graph, lib, params = relay.build(
                relay_prog, target=build_target, params=params, target_host=target_host
            )

    return graph, lib, params


def compile_model(
    model_id: str,
    arch: Dict[str, Any],
    ofa_net,
    env,
    target,
    config: Config,
    schedule_log_files: List[str]
) -> Optional[CompiledModel]:
    """Compile a single model."""
    print(f"\nCompiling {model_id}...")

    try:
        ofa_net.set_active_subnet(arch)
        standalone_net = load_static_resnet_from_arch(config.model_path, arch, ofa_net)
        pytorch_model = standalone_net

        with autotvm.tophub.context(target, extra_files=schedule_log_files):
            input_shape = [env.BATCH, 3, 224, 224]
            build_start = time.time()

            mod, params = pytorch_to_relay(pytorch_model, input_shape, config.input_name)

            if target.device_name == "vta":
                relay_prog = apply_quantization_and_packing(mod, params, env, config.model_name, config)
            else:
                relay_prog = mod["main"]

            graph, lib, params = build_relay_graph(relay_prog, target, env.target_host, params, env, config)

            build_time = time.time() - build_start
            print(f"  Built in {build_time:.2f}s")

            return CompiledModel(
                model_id=model_id,
                graph=graph,
                lib=lib,
                params=params
            )

    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {str(e)}")
        return None


def upload_compiled_models(
    compiled_models: Dict[str, CompiledModel],
    remote,
    temp_dir
) -> Dict[str, CompiledModel]:
    """Upload compiled models to remote."""
    print(f"\nUploading {len(compiled_models)} models...")

    for model_id, model_data in compiled_models.items():
        lib_path = temp_dir.relpath(f"graphlib_{model_id}.tar")
        model_data.lib.export_library(lib_path)
        remote.upload(lib_path)
        model_data.remote_lib = remote.load_module(f"graphlib_{model_id}.tar")

    return compiled_models


# ==============================================================================
# Runtime Initialization
# ==============================================================================

def create_runtime(graph: str, lib: tvm.runtime.Module, env, remote, ctx):
    """Create graph runtime."""
    if env.TARGET == "intelfocl":
        ctxes = [remote.ext_dev(0), remote.cpu(0)]
        return graph_runtime.create(graph, lib, ctxes)
    else:
        return graph_runtime.create(graph, lib, ctx)


def initialize_all_runtimes(
    compiled_models: Dict[str, CompiledModel],
    env,
    remote,
    ctx
) -> Dict[str, CompiledModel]:
    """Initialize runtimes and load parameters ONCE for all models.

    This function:
    1. Creates runtime module for each model
    2. Loads parameters into each runtime

    After this, run_inference() only needs to set input data and run.
    """
    print(f"\n{'='*80}")
    print("Initializing Runtimes and Loading Parameters")
    print(f"{'='*80}")

    for model_id, model_data in compiled_models.items():
        # Create runtime
        runtime_module = create_runtime(
            model_data.graph,
            model_data.remote_lib,
            env,
            remote,
            ctx
        )

        # Load parameters ONCE
        runtime_module.set_input(**model_data.params)

        # Store initialized runtime
        model_data.runtime_module = runtime_module
        print(f"  ✓ {model_id}")

    print(f"\nAll {len(compiled_models)} runtimes initialized!")
    return compiled_models


# ==============================================================================
# Inference
# ==============================================================================

def run_inference(
    model_data: CompiledModel,
    image_data: np.ndarray,
    input_name: str,
    env,
    remote,
    config: Config
) -> np.ndarray:
    """Run inference - assumes runtime already initialized.

    This function ONLY:
    1. Sets input image data
    2. Runs m.run() or timer()
    3. Returns output
    """
    if model_data.runtime_module is None:
        raise RuntimeError(f"Runtime not initialized for {model_data.model_id}")

    m = model_data.runtime_module
    ctx = remote.ext_dev(0) if config.device == "vta" else remote.cpu(0)

    # Set ONLY input image (params already loaded)
    m.set_input(input_name, image_data)

    if config.run_single_time:
        m.run()
    else:
        timer = m.module.time_evaluator("run", ctx, number=config.run_num, repeat=config.run_rep)
        tcost = timer()
        std = np.std(tcost.results) * 1000
        mean = tcost.mean * 1000
        print(f"    Inference: {mean:.2f}ms (std={std:.2f})")

    output = m.get_output(0, tvm.nd.empty((env.BATCH, 10), "float32", remote.cpu(0)))
    return output.asnumpy()


def print_top5_predictions(output: np.ndarray, classes: List[str], model_name: str, batch_idx: int = 0):
    """Print top-5 predictions."""
    top_categories = np.argsort(output[batch_idx])
    print(f"\n{model_name} - Top 5:")
    for i in range(5, 0, -1):
        print(f"  #{6-i}: {classes[top_categories[-i]]}")


def run_inference_all_models(
    compiled_models: Dict[str, CompiledModel],
    image_data: np.ndarray,
    input_name: str,
    env,
    remote,
    config: Config,
    classes: List[str],
    model_name: str = "Model"
) -> Dict[str, np.ndarray]:
    """Run inference on all models with same input - efficient switching."""
    print(f"\n{'='*80}")
    print(f"Running Inference on All {len(compiled_models)} Models")
    print(f"{'='*80}")

    results = {}

    for model_id, model_data in compiled_models.items():
        print(f"\n{model_id}:")

        try:
            start = time.time()
            output = run_inference(model_data, image_data, input_name, env, remote, config)
            elapsed = time.time() - start

            results[model_id] = output
            print(f"  Total time: {elapsed*1000:.2f}ms")
            print_top5_predictions(output, classes, f"{model_name} ({model_id})", 0)

        except Exception as e:
            print(f"  ✗ FAILED: {type(e).__name__}: {str(e)}")

    return results


# ==============================================================================
# Image Loading
# ==============================================================================

class ImageNetteDataLoader:
    """ImageNette data loader."""

    def __init__(self, base_dir: str, batch_size: int):
        self.base_dir = base_dir
        self.batch_size = batch_size

        imagenette_classes_path = os.path.join(base_dir, "imagenette_classes.txt")
        self.classes = eval(open(imagenette_classes_path).read())

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
        """Load and preprocess image."""
        image = Image.open(self.image_paths[idx]).resize((224, 224))

        if show_image:
            plt.imshow(image)
            plt.show()

        image = np.array(image) - np.array([123.0, 117.0, 104.0])
        image /= np.array([58.395, 57.12, 57.375])
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]
        image = np.repeat(image, self.batch_size, axis=0)

        return image


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Main execution."""
    config = Config()

    assert tvm.runtime.enabled("rpc")
    env = vta.get_env()
    target = env.target if config.device == "vta" else env.target_vta_cpu

    # Setup
    OFADynamicResnetAllMod, StaticResNetFromArch = setup_external_imports(config.external_repo_root)

    print("Loading OFA model...")
    ofa_net = load_ofa_model(config.model_path)

    print("Setting up RPC...")
    remote = setup_rpc_connection(env, config)
    ctx = remote.ext_dev(0) if config.device == "vta" else remote.cpu(0)

    schedule_log_files = glob.glob(config.schedule_log_dir)
    print(f"Loaded {len(schedule_log_files)} schedule logs")

    print("Loading candidate set...")
    arch_mapping, model_ids = load_candidate_set_from_experiments(
        config.candidate_set_json,
        config.arch_config_json,
        config.experiment_name
    )

    if config.num_models_to_test is not None:
        model_ids = model_ids[:config.num_models_to_test]

    # Compile models
    print(f"\n{'='*80}")
    print(f"Compiling {len(model_ids)} Models")
    print(f"{'='*80}")

    compiled_models = {}
    failed_models = []
    temp = utils.tempdir()

    for model_id in model_ids:
        arch = arch_mapping[model_id]
        compiled = compile_model(model_id, arch, ofa_net, env, target, config, schedule_log_files)

        if compiled is not None:
            compiled_models[model_id] = compiled
        else:
            failed_models.append(model_id)

    print(f"\nCompiled: {len(compiled_models)}/{len(model_ids)}")

    if compiled_models:
        # Upload models
        upload_compiled_models(compiled_models, remote, temp)

        # Initialize runtimes and load params ONCE
        initialize_all_runtimes(compiled_models, env, remote, ctx)

        # Load test image
        print("\nLoading test image...")
        image_loader = ImageNetteDataLoader(config.imagenette_base_dir, env.BATCH)
        image_data = image_loader.load_and_preprocess(0, show_image=False)

        # Run inference on all models
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

        print(f"\n{'='*80}")
        print(f"Successfully ran inference on {len(results)}/{len(compiled_models)} models")
        print(f"{'='*80}")

    # Summary
    print(f"\nTotal: {len(model_ids)}, Compiled: {len(compiled_models)}, Failed: {len(failed_models)}")
    if failed_models:
        print(f"Failed IDs: {failed_models}")


if __name__ == "__main__":
    main()

