from __future__ import absolute_import, print_function
# add external modules to PYTHONPATH via environment variable
import os, sys, time
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
from Wkls import candidate_set_wkls
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

# summary(net, input_size=(1, 3, 224, 224))
#
# arch = net.sample_arch()
# print(arch)
# net.set_active_subnet(arch)
# # net.precompute_active_weights(arch)
# # net.enable_auto_precompute()
# summary(net, input_size=(1, 3, 224, 224))

import json
from typing import Dict, Any

candidate_set_json_path="/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/candidate_sets_results_all_expts.json"
arch_config_json_path="/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json"

# For using experiment results format:
# experiment_results_path = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/candidate_sets_results_all_expts.json"
# arch_mapping, model_ids = load_candidate_set_from_experiments(
#     experiment_results_path,
#     arch_config_json_path,
#     'greedy_swap'  # or 'ilp_tau_0.18757952189126595', 'sa_lam_2.0', 'sa_lam_1.0', 'ga'
# )

def load_candidate_set_from_experiments(results_path: str, arch_path: str, experiment_name: str) -> tuple:
    """Load candidate set and architectures from experiment results file.

    This function handles the case where you have:
    1. An experiment results file with different experiment methods (greedy_swap, ilp, sa, ga, etc.)
    2. A separate architectures file with the actual architecture definitions

    Args:
        results_path: Path to the experiment results JSON (e.g., candidate_sets_results_all_expts.json)
        arch_path: Path to the architectures JSON file
        experiment_name: Name of the experiment to load (e.g., 'greedy_swap', 'ilp_tau_0.18757952189126595',
                        'sa_lam_2.0', 'sa_lam_1.0', 'ga')

    Returns:
        Tuple of (arch_mapping, model_ids) where:
            arch_mapping: Dict mapping architecture ID to architecture config (filtered to selected models)
            model_ids: List of model IDs selected in the experiment

    Example:
        arch_mapping, model_ids = load_candidate_set_from_experiments(
            '/path/to/candidate_sets_results_all_expts.json',
            '/path/to/architectures_20250927_180844.json',
            'greedy_swap'
        )
    """
    # Load the experiment results
    with open(results_path, 'r') as f:
        results = json.load(f)

    if experiment_name not in results:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. "
            f"Available experiments: {list(results.keys())}"
        )

    # Get the IDs from the selected experiment
    model_ids = results[experiment_name]['ids']

    print(f"Loading experiment '{experiment_name}':")
    print(f"  - {len(model_ids)} models selected")
    print(f"  - Mean accuracy: {results[experiment_name].get('mean_acc', 'N/A')}")
    print(f"  - ASPT: {results[experiment_name].get('ASPT', 'N/A')}")
    print(f"  - Max pairwise: {results[experiment_name].get('max_pairwise', 'N/A')}")

    # Load all architectures
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

# arch_mapping = load_arch_mapping(arch_config_json_path)
# # read candidate set json
# with open(candidate_set_json_path, 'r') as f:
#     candidate_set = json.load(f)
# model_ids = candidate_set['selected_model_ids']

arch_mapping, model_ids = load_candidate_set_from_experiments(candidate_set_json_path, arch_config_json_path, 'sa_lam_2.0')



def extract_wkls(ofa_net, arch_mapping, id):
    print("Model ID in candidate set:", id)
    print("With architecture:", arch_mapping[id])

    # Skip network if out_channel_setting_list[0] is 0 and decomp_type_list[0] contains 2 or 4
    # (unsupported patterns in relay frontend)
    arch = arch_mapping[id]
    # if arch['out_channel_setting_list'][0] == 0:
    #     has_unsupported_pattern = any(
    #         decomp_type in [2]
    #         for decomp_types_residual in arch['decomp_type_list'][0]
    #         for decomp_type in decomp_types_residual
    #         # for decomp_type in arch['decomp_type_list'][0][0]
    #     )
    #     if has_unsupported_pattern:
    #         print(f"Skipping model id {id} due to unsupported pattern in relay frontend")
    #         continue

    ofa_net.set_active_subnet(arch)
    standalone_net = StaticResNetFromArch(
        target_arch=arch,
        num_classes=10,
        width_mult_list=(0.5, 1.0, 2.0)
    )
    standalone_net.load_weights_from_ofa_checkpoint(checkpoint_path=model_path, ofa_model=ofa_net)
    net = standalone_net
    pytorch_model = net
    # arch=arch_mapping[model_id]
    # pytorch_model.set_active_subnet(arch)
    # pytorch_model.precompute_active_weights(arch_mapping[model_id])
    pytorch_model.eval()
    # for mod in pytorch_model.modules():
    #     if hasattr(mod, 'export_detach_cached_filters'):
    #         mod.export_detach_cached_filters = True
    workloads = []
    count = 0
    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)

    for layer in pytorch_model.modules():
        if type(layer) == torch.nn.modules.conv.Conv2d:
            if(layer.in_channels % 8 == 0 and layer.out_channels % 8 ==0 and layer.padding[0] == layer.padding[1]):
                workloads.append(Workload(1, 0, 0, layer.in_channels, layer.out_channels,
                                  layer.kernel_size[0], layer.kernel_size[1], layer.padding[0], layer.padding[1]
                                 , layer.stride[0], layer.stride[1]))
        elif type(layer) == DynamicConv2DAll:
            # print the active channels and kernel sizes and strides and paddings
            active_in_channels = layer.exec_in_channels
            active_out_channels = layer.exec_out_channels
            padding = layer.exec_padding
            kernel_size = layer.exec_kernel_size
            stride = layer.base_conv.stride

            for ic, oc in zip(active_in_channels, active_out_channels):
                # print("Found sub-conv of DynamicConv2DAll with in_channels={}, out_channels={}, kernel_size={}, stride={}, padding={}".format(ic, oc, kernel_size, stride, padding))
                if(ic % 8 == 0 and oc % 8 ==0):
                    workloads.append(Workload(1, 0, 0, ic, oc,
                                      kernel_size, kernel_size, padding, padding
                                     , stride[0], stride[1]))


    scripted_model = torch.jit.trace(pytorch_model, input_data).eval()
    shape_list = [("input0", input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    # print(mod.astext(show_meta_data=False))

    # Ensure types are inferred before quantization
    mod = relay.transform.InferType()(mod)

    # Perform quantization - let quantize() handle parameter binding internally
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
             mod = relay.quantize.quantize(mod, params=params)
    mod_as_string = mod.astext(show_meta_data=False)
    # print(mod_as_string)
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

    return final_workloads

all_wkls = []
for model_id in model_ids:
    with torch.no_grad():
        extracted_wkls = extract_wkls(ofa_net,arch_mapping, model_id)
        # print("Model ID: {}, Extracted Workloads: {}".format(model_id, extracted_wkls))
    all_wkls.extend(set(extracted_wkls))

all_wkls = set(all_wkls)
# remove wkls already in candidate_set_wkls
for wkl in all_wkls:
    if any(wkl == candidate_wkl[1] for candidate_wkl in candidate_set_wkls):
        print("Workload {} already in candidate set, skipping".format(wkl))
        all_wkls.remove(wkl)
print("Total unique workloads extracted from candidate set: {}".format(len(all_wkls)))
for i,wkl in enumerate(all_wkls):
    file_line_str = f"(\"workload_{i}\", {wkl}),"
    print(file_line_str)
    # write file_line_str to candidates_wkls.txt
    with open("candidates_wkls.txt", "a") as f:
        f.write(file_line_str + "\n")