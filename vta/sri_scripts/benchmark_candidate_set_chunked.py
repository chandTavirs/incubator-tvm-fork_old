"""
Benchmark candidate set models on VTA in chunks.
Measures mean and std execution time for each model.

Usage:
    python benchmark_candidate_set_chunked.py --chunk_id 0 --chunk_size 10
    python benchmark_candidate_set_chunked.py --chunk_id 1 --chunk_size 10
    ...
"""

from __future__ import absolute_import, print_function

import os
import sys
import time
import json
import csv
import argparse
import glob
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import torch

import tvm
from tvm import te, rpc, autotvm, relay
from tvm.contrib import graph_runtime, utils
from tvm.relay import transform

import vta
from vta.top import graph_pack

# Import necessary functions from execute_candidate_set_refactored.py
# (These will be imported at runtime to avoid circular dependencies)


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for VTA benchmarking."""
    # Paths
    external_repo_root: str = "/home/srchand/Desktop/research/OFA_Obfs"
    model_path: str = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth"
    arch_config_json: str = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json"
    arch_meta_jsonl: str = "/home/srchand/Desktop/research/OFA_Obfs/transferability_matrix_try_final_remaining/transfer_meta.jsonl"
    schedule_log_dir: str = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/logs/tuning_logs/vta_1x16x16/candidate_set/*.log"
    bitstream_path: str = "/mnt/hgfs/vmware_ubuntu_sf/vta_4x8x8/vta_new_1x16x16_memory_trojan_runtime_sampling.bit"

    # Output paths
    output_dir: str = "benchmark_results"

    # Device settings
    device: str = "vta"
    device_host: str = "10.42.0.188"
    device_port: str = "9091"
    program_device: bool = False

    # Model settings
    model_name: str = "resnet18"
    input_name: str = "input0"

    # Build settings
    opt_level: int = 3
    global_scale: float = 8.0
    skip_conv_layers: List[int] = None

    # Timing settings (same as execute_candidate_set_refactored.py)
    run_num: int = 4
    run_rep: int = 3

    # Chunk settings
    chunk_size: int = 10
    chunk_id: int = 0

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
# External Imports Setup
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
                if "TVM_Intel_Fork/tvm/vta/sri_scripts" in _mod_file:
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


# ==============================================================================
# Data Loading
# ==============================================================================

def load_meta(jsonl_path: str) -> Tuple[List[str], np.ndarray]:
    """Load model IDs and accuracies from meta file."""
    ids = []
    accs = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                line2 = line.split('//')[0].strip()
                obj = json.loads(line2)
            ids.append(obj['model_id'])
            accs.append(float(obj.get('pred_acc', obj.get('acc', 0.0))))
    return ids, np.array(accs, dtype=float)


def load_arch_mapping(path: str) -> Dict[str, Any]:
    """Load architectures JSON and normalize to a mapping {id: architecture_dict}."""
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
            else:
                if 'id' in item:
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


# ==============================================================================
# Model Filtering (same as extract_workloads_chunked.py)
# ==============================================================================

def filter_candidate_models(ids: List[str], arch_map: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Filter out models with unsupported patterns.
    Same logic as extract_workloads_chunked.py
    """
    candidate_indices = []
    skipped_due_to_arch = []

    for model_id in ids:
        skip = False
        if model_id in arch_map:
            arch = arch_map[model_id]
            # check out_channel_setting_list[0] == 0 then look into decomp_type_list[0]
            try:
                ocs = arch.get('out_channel_setting_list', None)
                dtl = arch.get('decomp_type_list', None)
                if isinstance(ocs, list) and len(ocs) > 0 and int(ocs[0]) == 0 and isinstance(dtl, list) and len(dtl) > 0:
                    # dtl[0] expected to be a list of residual lists
                    for decomp_types_residual in dtl[0]:
                        for decomp_type in decomp_types_residual:
                            if int(decomp_type) == 2:
                                skip = True
                                break
                        if skip:
                            break
            except Exception:
                # be conservative and do not skip if structure unexpected
                skip = False
        if skip:
            skipped_due_to_arch.append(model_id)
            continue
        candidate_indices.append(model_id)

    return candidate_indices, skipped_due_to_arch


# ==============================================================================
# OFA Model Loading
# ==============================================================================

def load_ofa_model(model_path: str):
    """Load OFA model."""
    from ofa_base_models import OFADynamicResnetAllMod

    ofa_net = OFADynamicResnetAllMod()
    device_temp = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device_temp)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    else:
        state = checkpoint
    ofa_net.load_state_dict(state, strict=False)

    return ofa_net


def load_static_resnet_from_arch(model_path: str, arch: Dict[str, Any], ofa_net):
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

def setup_rpc_connection(env, config: BenchmarkConfig):
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
            vta.program_fpga(remote, config.bitstream_path)

        reconfig_time = time.time() - reconfig_start
        print(f"Reconfigured runtime in {reconfig_time:.2f}s")
    else:
        remote = rpc.LocalSession()
        if env.TARGET in ["intelfocl"]:
            vta.program_fpga(remote, bitstream="vta.bitstream")

    return remote


# ==============================================================================
# Relay Building
# ==============================================================================

def pytorch_to_relay(pytorch_model, input_shape: List[int], input_name: str = "input0"):
    """Convert PyTorch model to Relay IR."""
    pytorch_model.eval()
    input_data = torch.randn(input_shape)

    with torch.no_grad():
        scripted_model = torch.jit.trace(pytorch_model, input_data).eval()

    shape_list = [(input_name, input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    mod = relay.transform.InferType()(mod)

    return mod, params


def apply_quantization_and_packing(mod, params, env, model_name: str, config: BenchmarkConfig):
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


def build_relay_graph(relay_prog, target, target_host, params, env, config: BenchmarkConfig):
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


# ==============================================================================
# Model Compilation and Benchmarking
# ==============================================================================

@dataclass
class BenchmarkResult:
    """Result of benchmarking a single model."""
    model_id: str
    mean_ms: float
    std_ms: float
    compile_time_s: float
    success: bool
    error_msg: Optional[str] = None


def compile_and_benchmark_model(
    model_id: str,
    arch: Dict[str, Any],
    ofa_net,
    env,
    target,
    remote,
    temp_dir,
    config: BenchmarkConfig,
    schedule_log_files: List[str]
) -> BenchmarkResult:
    """Compile and benchmark a single model."""
    print(f"\nProcessing {model_id}...")

    try:
        # 1. Load model
        ofa_net.set_active_subnet(arch)
        standalone_net = load_static_resnet_from_arch(config.model_path, arch, ofa_net)
        pytorch_model = standalone_net

        # 2. Compile model
        with autotvm.tophub.context(target, extra_files=schedule_log_files):
            input_shape = [env.BATCH, 3, 224, 224]
            compile_start = time.time()

            mod, params = pytorch_to_relay(pytorch_model, input_shape, config.input_name)

            if target.device_name == "vta":
                relay_prog = apply_quantization_and_packing(mod, params, env, config.model_name, config)
            else:
                relay_prog = mod["main"]

            graph, lib, params = build_relay_graph(relay_prog, target, env.target_host, params, env, config)

            compile_time = time.time() - compile_start
            print(f"  Compiled in {compile_time:.2f}s")

        # 3. Upload to device
        lib_path = temp_dir.relpath(f"graphlib_{model_id}.tar")
        lib.export_library(lib_path)
        remote.upload(lib_path)
        remote_lib = remote.load_module(f"graphlib_{model_id}.tar")

        # 4. Create runtime and load parameters
        ctx = remote.ext_dev(0) if config.device == "vta" else remote.cpu(0)
        if env.TARGET == "intelfocl":
            ctxes = [remote.ext_dev(0), remote.cpu(0)]
            m = graph_runtime.create(graph, remote_lib, ctxes)
        else:
            m = graph_runtime.create(graph, remote_lib, ctx)

        m.set_input(**params)

        # 5. Prepare dummy input
        image_data = np.random.uniform(size=(env.BATCH, 3, 224, 224)).astype("float32")
        m.set_input(config.input_name, image_data)

        # 6. Benchmark with timer
        print(f"  Benchmarking...")
        timer = m.module.time_evaluator("run", ctx, number=config.run_num, repeat=config.run_rep)
        tcost = timer()

        mean_ms = tcost.mean * 1000
        std_ms = np.std(tcost.results) * 1000

        print(f"  ✓ Mean: {mean_ms:.2f}ms, Std: {std_ms:.2f}ms")

        # Clean up
        del standalone_net, pytorch_model

        return BenchmarkResult(
            model_id=model_id,
            mean_ms=mean_ms,
            std_ms=std_ms,
            compile_time_s=compile_time,
            success=True
        )

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"  ✗ FAILED: {error_msg}")
        return BenchmarkResult(
            model_id=model_id,
            mean_ms=0.0,
            std_ms=0.0,
            compile_time_s=0.0,
            success=False,
            error_msg=error_msg
        )


# ==============================================================================
# Output Functions
# ==============================================================================

def save_chunk_results_json(results: List[BenchmarkResult], output_path: str):
    """Save results to JSON file (format similar to transfer_meta.jsonl)."""
    with open(output_path, 'w') as f:
        for result in results:
            obj = {
                'model_id': result.model_id,
                'mean_exec_time_ms': result.mean_ms,
                'std_exec_time_ms': result.std_ms,
                'compile_time_s': result.compile_time_s,
                'success': result.success
            }
            if result.error_msg:
                obj['error'] = result.error_msg
            f.write(json.dumps(obj) + '\n')


def save_chunk_results_csv(results: List[BenchmarkResult], output_path: str):
    """Save results to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_id', 'mean_exec_time_ms', 'std_exec_time_ms',
                        'compile_time_s', 'success', 'error'])
        for result in results:
            writer.writerow([
                result.model_id,
                f"{result.mean_ms:.4f}" if result.success else "N/A",
                f"{result.std_ms:.4f}" if result.success else "N/A",
                f"{result.compile_time_s:.2f}",
                result.success,
                result.error_msg or ""
            ])


def save_error_log(failed_results: List[BenchmarkResult], output_path: str, chunk_id: int):
    """Save detailed error log."""
    if not failed_results:
        return

    with open(output_path, 'w') as f:
        f.write(f"Chunk {chunk_id} - Failed Models\n")
        f.write("=" * 80 + "\n\n")
        for result in failed_results:
            f.write(f"Model ID: {result.model_id}\n")
            f.write(f"Error: {result.error_msg}\n")
            f.write("-" * 80 + "\n\n")


# ==============================================================================
# Main Function
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark candidate set models on VTA in chunks')
    parser.add_argument('--chunk_id', type=int, required=True, help='Chunk ID (0-indexed)')
    parser.add_argument('--chunk_size', type=int, default=10, help='Number of models per chunk')
    parser.add_argument('--arch_config', type=str,
                       default="/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json",
                       help='Path to architecture config JSON')
    parser.add_argument('--arch_meta', type=str,
                       default="/home/srchand/Desktop/research/OFA_Obfs/transferability_matrix_try_final_remaining/transfer_meta.jsonl",
                       help='Path to architecture meta JSONL')
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                       help='Output directory for benchmark results')
    parser.add_argument('--device_host', type=str, default='10.42.0.188',
                       help='RPC device host')
    parser.add_argument('--device_port', type=str, default='9091',
                       help='RPC device port')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize config
    config = BenchmarkConfig(
        chunk_id=args.chunk_id,
        chunk_size=args.chunk_size,
        arch_config_json=args.arch_config,
        arch_meta_jsonl=args.arch_meta,
        output_dir=args.output_dir,
        device_host=args.device_host,
        device_port=args.device_port
    )

    print("=" * 80)
    print(f"VTA Candidate Set Benchmark - Chunk {config.chunk_id}")
    print("=" * 80)

    # Setup external imports
    print("\nSetting up external imports...")
    setup_external_imports(config.external_repo_root)

    # Load architecture mapping and meta
    print("Loading architecture configurations...")
    arch_map = load_arch_mapping(config.arch_config_json)
    ids, accs = load_meta(config.arch_meta_jsonl)

    # Filter models (same logic as extract_workloads_chunked.py)
    print("Filtering models...")
    candidate_indices, skipped_due_to_arch = filter_candidate_models(ids, arch_map)

    if skipped_due_to_arch:
        print(f'  Skipped {len(skipped_due_to_arch)} models due to unsupported patterns')

    print(f"  Total valid candidate models: {len(candidate_indices)}")

    # Calculate chunk boundaries
    start_idx = config.chunk_id * config.chunk_size
    end_idx = min(start_idx + config.chunk_size, len(candidate_indices))

    if start_idx >= len(candidate_indices):
        print(f"\n✗ Chunk {config.chunk_id} is out of range. Total models: {len(candidate_indices)}")
        return

    chunk_model_ids = candidate_indices[start_idx:end_idx]
    print(f"\n{'='*80}")
    print(f"Processing Chunk {config.chunk_id}")
    print(f"  Models {start_idx} to {end_idx-1} ({len(chunk_model_ids)} models)")
    print(f"{'='*80}")

    # Load OFA model
    print("\nLoading OFA model...")
    ofa_net = load_ofa_model(config.model_path)

    # Setup VTA environment
    print("\nSetting up VTA environment...")
    env = vta.get_env()
    target = env.target if env.TARGET != "intelfocl" else env.target_vta_cpu

    # Setup RPC connection
    print("\nConnecting to RPC device...")
    remote = setup_rpc_connection(env, config)

    # Get context
    ctx = remote.ext_dev(0) if config.device == "vta" else remote.cpu(0)

    # Create temp directory
    temp_dir = utils.tempdir()

    # Load schedule logs
    schedule_log_files = glob.glob(config.schedule_log_dir)
    print(f"  Found {len(schedule_log_files)} schedule log files")

    # Benchmark each model in the chunk
    print(f"\n{'='*80}")
    print(f"Benchmarking {len(chunk_model_ids)} models...")
    print(f"{'='*80}")

    results = []
    for i, model_id in enumerate(chunk_model_ids, 1):
        print(f"\n[{i}/{len(chunk_model_ids)}]", end=" ")
        arch = arch_map[model_id]

        result = compile_and_benchmark_model(
            model_id=model_id,
            arch=arch,
            ofa_net=ofa_net,
            env=env,
            target=target,
            remote=remote,
            temp_dir=temp_dir,
            config=config,
            schedule_log_files=schedule_log_files
        )
        results.append(result)

    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")
    print(f"{'='*80}")

    # JSON output
    json_output = os.path.join(config.output_dir, f"benchmark_chunk_{config.chunk_id}.jsonl")
    save_chunk_results_json(results, json_output)
    print(f"  ✓ JSON: {json_output}")

    # CSV output
    csv_output = os.path.join(config.output_dir, f"benchmark_chunk_{config.chunk_id}.csv")
    save_chunk_results_csv(results, csv_output)
    print(f"  ✓ CSV: {csv_output}")

    # Error log
    failed_results = [r for r in results if not r.success]
    if failed_results:
        error_log = os.path.join(config.output_dir, f"errors_chunk_{config.chunk_id}.log")
        save_error_log(failed_results, error_log, config.chunk_id)
        print(f"  ⚠ Error log: {error_log}")

    # Summary
    print(f"\n{'='*80}")
    print("Chunk Summary")
    print(f"{'='*80}")
    successful = [r for r in results if r.success]
    print(f"  Total models: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed_results)}")

    if successful:
        mean_times = [r.mean_ms for r in successful]
        print(f"\n  Execution times:")
        print(f"    Min: {min(mean_times):.2f}ms")
        print(f"    Max: {max(mean_times):.2f}ms")
        print(f"    Avg: {np.mean(mean_times):.2f}ms")

    print(f"\n{'='*80}")
    print(f"✓ Chunk {config.chunk_id} complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

