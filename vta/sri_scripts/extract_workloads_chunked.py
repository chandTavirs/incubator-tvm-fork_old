from __future__ import absolute_import, print_function
# add external modules to PYTHONPATH via environment variable
import os, sys, time, argparse
from PIL import Image

import numpy as np
from matplotlib import pyplot as plt

import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_runtime, utils, download
from tvm.contrib.debugger import debug_runtime
from tvm.relay import transform

import vta
from vta.testing import simulator
from vta.top import graph_pack
from torch import nn
import torch
import torchvision
from tvm.contrib.download import download_testdata
# import logging
# logging.basicConfig(level=logging.DEBUG)


# Robust import of external ofa_base_models without being shadowed by local folder
external_repo_root = "/home/srchand/Desktop/research/OFA_Obfs"
if external_repo_root not in sys.path:
    sys.path.insert(0, external_repo_root)

# If a local shim package is already cached, evict it to allow importing the external one
_mod = sys.modules.get("ofa_base_models")
_mod2 = sys.modules.get("architecture_defense")

if _mod is not None:
    try:
        _mod_file = getattr(_mod, "__file__", "") or ""
        if "TVM_Intel_Fork/tvm/vta/sri_scripts/jupyter_nbs/ofa_base_models" in _mod_file:
            del sys.modules["ofa_base_models"]
    except Exception:
        # If anything goes wrong, clear the cache entry
        sys.modules.pop("ofa_base_models", None)

if _mod2 is not None:
    try:
        _mod2_file = getattr(_mod2, "__file__", "") or ""
        if "TVM_Intel_Fork/tvm/vta/sri_scripts/jupyter_nbs/architecture_defense" in _mod2_file:
            del sys.modules["architecture_defense"]
    except Exception:
        # If anything goes wrong, clear the cache entry
        sys.modules.pop("architecture_defense", None)

try:
    from ofa_base_models import OFADynamicResnetAllMod  # type: ignore
    from architecture_defense import StaticResNetFromArch  # type: ignore
except (ModuleNotFoundError, ImportError):
    # Final fallback: ensure external root is first in path and retry once
    if sys.path[0] != external_repo_root:
        sys.path.insert(0, external_repo_root)
    # Clear any cached partial imports
    sys.modules.pop("ofa_base_models", None)
    sys.modules.pop("architecture_defense", None)
    from ofa_base_models import OFADynamicResnetAllMod  # type: ignore
    from architecture_defense import StaticResNetFromArch  # type: ignore
else:
    # Print where the modules are loaded from for debugging/IDE clarity
    import ofa_base_models as _obm  # type: ignore
    print("ofa_base_models loaded from:", getattr(_obm, "__file__", None))
    try:
        import architecture_defense as _ad  # type: ignore
        print("architecture_defense loaded from:", getattr(_ad, "__file__", None))
    except ImportError:
        pass

import torch
from collections import namedtuple
from torchvision import transforms


# Make sure that TVM was compiled with RPC=1
assert tvm.runtime.enabled("rpc")

Workload = namedtuple(
    "Conv2DWorkload",
    [
        "batch",
        "height",
        "width",
        "in_filter",
        "out_filter",
        "hkernel",
        "wkernel",
        "hpad",
        "wpad",
        "hstride",
        "wstride",
    ],
)

import re
channels_re_with_stride = re.compile('.*Tensor\[\(([\d]+), ([\d]+), [\d]+, [\d]+\).*strides=\[([\d]+), ([\d]+)\].*padding=\[([\d]+), ([\d]+).*kernel_size=\[([\d]+), ([\d]+)\].*')
cast_re = re.compile('cast.*Tensor\[\([\d]+, [\d]+, ([\d]+), ([\d]+)\).*')
channels_re_no_stride = re.compile('.*Tensor\[\(([\d]+), ([\d]+), [\d]+, [\d]+\).*padding=\[([\d]+), ([\d]+).*kernel_size=\[([\d]+), ([\d]+)\].*')


from torchinfo import summary

import json
from typing import Dict, Any

def load_meta(jsonl_path):
    ids = []
    accs = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # allow trailing comments
            try:
                obj = json.loads(line)
            except Exception:
                # try to strip // comments
                line2 = line.split('//')[0].strip()
                obj = json.loads(line2)
            ids.append(obj['model_id'])
            accs.append(float(obj.get('pred_acc', obj.get('acc', 0.0))))
    return ids, np.array(accs, dtype=float)


def load_arch_mapping(path: str) -> Dict[str, Any]:
    """Load architectures JSON and normalize to a mapping {id: architecture_dict}.

    Supported input formats:
    - A dict mapping id -> architecture dict (legacy)
    - A dict with key 'architectures' containing a list of items with fields 'id' and 'architecture'
    - A top-level list of items with 'id' and 'architecture'

    Note: For experiment results files (candidate_sets_results_all_expts.json),
          use load_candidate_set_from_experiments() instead.
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
            # item may contain fields 'id' and 'architecture' (nested)
            if 'id' in item and 'architecture' in item:
                mapping[item['id']] = item['architecture']
            elif 'id' in item and 'arch' in item:
                mapping[item['id']] = item['arch']
            else:
                # If the item itself is an architecture dict without id, generate an id
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

    # Fallback: unknown format, raise error
    raise ValueError(f"Unsupported architecture file format: {path}")


def extract_wkls(ofa_net, arch_mapping, id):
    print(f"Processing Model ID: {id}")

    arch = arch_mapping[id]

    ofa_net.set_active_subnet(arch)
    standalone_net = StaticResNetFromArch(
        target_arch=arch,
        num_classes=10,
        width_mult_list=(0.5, 1.0, 2.0)
    )
    standalone_net.load_weights_from_ofa_checkpoint(checkpoint_path=model_path, ofa_model=ofa_net)

    standalone_net.eval()
    workloads = []
    count = 0
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)

    for layer in standalone_net.modules():
        if type(layer) == StaticResNetFromArch:
            module_list = layer.blocks.modules()
            # extract conv layers from module_list
            for mod in module_list:
                if type(mod) == torch.nn.modules.conv.Conv2d:
                    if(mod.in_channels % 8 == 0 and mod.out_channels % 8 ==0 and mod.padding[0] == mod.padding[1]):
                        workloads.append(Workload(1, 0, 0, mod.in_channels, mod.out_channels,
                                          mod.kernel_size[0], mod.kernel_size[1], mod.padding[0], mod.padding[1]
                                         , mod.stride[0], mod.stride[1]))
        elif type(layer) == torch.nn.modules.conv.Conv2d:
            if(layer.in_channels % 8 == 0 and layer.out_channels % 8 ==0 and layer.padding[0] == layer.padding[1]):
                workloads.append(Workload(1, 0, 0, layer.in_channels, layer.out_channels,
                                  layer.kernel_size[0], layer.kernel_size[1], layer.padding[0], layer.padding[1]
                                 , layer.stride[0], layer.stride[1]))

    scripted_model = torch.jit.trace(standalone_net, input_data).eval()
    shape_list = [("input0", input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    # Ensure types are inferred before quantization
    mod = relay.transform.InferType()(mod)

    # Perform quantization - let quantize() handle parameter binding internally
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
             mod = relay.quantize.quantize(mod, params=params)
    mod_as_string = mod.astext(show_meta_data=False)

    cast_line = ""
    cast_line_idx = -1
    final_workloads = []
    for i, line in enumerate(mod_as_string.split('\n')):
        if "cast" in line and "int8" in line:
            cast_line = line
            cast_line_idx = i
        elif "conv2d" in line and "int8" in line:
            match = re.search(channels_re_with_stride, line)
            if match:
                if int(match.group(1)) % 8 == 0 and int(match.group(2)) % 8 == 0:
                    match_cast = re.search(cast_re, cast_line)
                    if match_cast:
                        matched_in_filter = int(match.group(2))
                        matched_out_filter = int(match.group(1))
                        matched_hkernel = int(match.group(7))
                        matched_wkernel = int(match.group(8))
                        matched_hpad = int(match.group(5))
                        matched_wpad = int(match.group(6))
                        matched_hstride = int(match.group(3))
                        matched_wstride = int(match.group(4))
                        for tmp_wkl in workloads:
                            if ((tmp_wkl.in_filter == matched_in_filter) and
                                    (tmp_wkl.out_filter == matched_out_filter) and
                                    (tmp_wkl.hkernel == matched_hkernel) and
                                    (tmp_wkl.wkernel == matched_wkernel) and
                                    (tmp_wkl.hpad == matched_hpad) and
                                    (tmp_wkl.wpad == matched_wpad) and
                                    (tmp_wkl.hstride == matched_hstride) and
                                    (tmp_wkl.wstride == matched_wstride)):
                                wkl = tmp_wkl
                                final_workloads.append(
                                'Workload({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})'.format(
                                1, match_cast.group(1), match_cast.group(2), wkl.in_filter, wkl.out_filter,
                                wkl.hkernel,wkl.wkernel, wkl.hpad, wkl.wpad, wkl.hstride, wkl.wstride))
            else:
                match = re.search(channels_re_no_stride, line)
                if match:
                    if int(match.group(1)) % 8 == 0 and int(match.group(2)) % 8 == 0:
                        match_cast = re.search(cast_re, cast_line)
                        if match_cast:
                            matched_in_filter = int(match.group(2))
                            matched_out_filter = int(match.group(1))
                            matched_hkernel = int(match.group(5))
                            matched_wkernel = int(match.group(6))
                            matched_hpad = int(match.group(3))
                            matched_wpad = int(match.group(4))
                            for tmp_wkl in workloads:
                                if ((tmp_wkl.in_filter == matched_in_filter) and
                                    (tmp_wkl.out_filter == matched_out_filter) and
                                    (tmp_wkl.hkernel == matched_hkernel) and
                                    (tmp_wkl.wkernel == matched_wkernel) and
                                    (tmp_wkl.hpad == matched_hpad) and
                                    (tmp_wkl.wpad == matched_wpad) and
                                    (tmp_wkl.hstride == 1) and
                                    (tmp_wkl.wstride == 1)):
                                    wkl = tmp_wkl
                                    final_workloads.append(
                                    'Workload({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})'.format(
                                    1, match_cast.group(1), match_cast.group(2), wkl.in_filter, wkl.out_filter,
                                    wkl.hkernel,wkl.wkernel, wkl.hpad, wkl.wpad, wkl.hstride, wkl.wstride))

    # remove standalone_net from memory
    del standalone_net

    return final_workloads


def main():
    parser = argparse.ArgumentParser(description='Extract workloads from candidate set in chunks')
    parser.add_argument('--chunk_id', type=int, required=True, help='Chunk ID (0-indexed)')
    parser.add_argument('--chunk_size', type=int, default=20, help='Number of networks per chunk')
    parser.add_argument('--arch_config', type=str,
                        default="/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json",
                        help='Path to architecture config JSON')
    parser.add_argument('--arch_meta', type=str,
                        default="/home/srchand/Desktop/research/OFA_Obfs/transferability_matrix_try_final_remaining/transfer_meta.jsonl",
                        help='Path to architecture meta JSONL')
    parser.add_argument('--output_dir', type=str, default='wkl_extraction',
                        help='Output directory for workload files')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup error log file
    error_log_file = os.path.join(args.output_dir, f"errors_chunk_{args.chunk_id}.log")

    # Load OFA model
    print("Loading OFA model...")
    global model_path
    ofa_net = OFADynamicResnetAllMod()
    model_path = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth"
    device_temp = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device_temp)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    else:
        state = checkpoint
    # Allow mismatched keys due to local shim modules
    ofa_net.load_state_dict(state, strict=False)

    # Load architecture mapping and meta
    print("Loading architecture configurations...")
    arch_map = load_arch_mapping(args.arch_config)

    candidate_indices = []
    skipped_due_to_arch = []
    ids, accs = load_meta(args.arch_meta)

    M = len(accs)
    for i in range(M):
        skip = False
        if args.arch_config and ids[i] in arch_map:
            arch = arch_map[ids[i]]
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
            skipped_due_to_arch.append(ids[i])
            continue
        candidate_indices.append(ids[i])

    if skipped_due_to_arch:
        print(f'Skipped {len(skipped_due_to_arch)} architectures due to relay frontend unsupported pattern')

    print(f"Total valid candidate models: {len(candidate_indices)}")

    # Calculate chunk boundaries
    start_idx = args.chunk_id * args.chunk_size
    end_idx = min(start_idx + args.chunk_size, len(candidate_indices))

    if start_idx >= len(candidate_indices):
        print(f"Chunk {args.chunk_id} is out of range. Total models: {len(candidate_indices)}")
        return

    chunk_model_ids = candidate_indices[start_idx:end_idx]
    print(f"\nProcessing Chunk {args.chunk_id}: models {start_idx} to {end_idx-1} ({len(chunk_model_ids)} models)")

    # Extract workloads for this chunk
    all_wkls = []
    failed_models = []

    for model_id in chunk_model_ids:
        try:
            with torch.no_grad():
                extracted_wkls = extract_wkls(ofa_net, arch_map, model_id)
            all_wkls.extend(set(extracted_wkls))
            print(f"  Extracted {len(extracted_wkls)} workloads from {model_id}")
            print(f"  Extracted {len(set(extracted_wkls))} unique workloads from {model_id}")
        except Exception as e:
            error_msg = f"ERROR processing model {model_id}: {str(e)}"
            print(f"  {error_msg}")

            # Log the failed model
            failed_models.append({
                'model_id': model_id,
                'architecture': arch_map.get(model_id, None),
                'error': str(e)
            })
            continue

    # Write error log if there were failures
    if failed_models:
        with open(error_log_file, 'w') as f:
            f.write(f"Chunk {args.chunk_id} - Failed Models\n")
            f.write("=" * 80 + "\n\n")
            for failed in failed_models:
                f.write(f"Model ID: {failed['model_id']}\n")
                f.write(f"Error: {failed['error']}\n")
                f.write(f"Architecture: {json.dumps(failed['architecture'], indent=2)}\n")
                f.write("-" * 80 + "\n\n")
        print(f"\n  WARNING: {len(failed_models)} model(s) failed. See {error_log_file}")

    # Save workloads to chunk-specific file
    output_file = os.path.join(args.output_dir, f"workloads_chunk_{args.chunk_id}.txt")
    # Remove duplicates by converting to a set before writing
    all_wkls = set(all_wkls)
    with open(output_file, 'w') as f:
        for wkl in all_wkls:
            f.write(wkl + "\n")

    print(f"\nChunk {args.chunk_id} complete!")
    print(f"  Total unique workloads extracted: {len(all_wkls)}")
    print(f"  Successful models: {len(chunk_model_ids) - len(failed_models)}/{len(chunk_model_ids)}")
    print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    main()

