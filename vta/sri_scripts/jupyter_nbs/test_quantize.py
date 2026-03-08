#!/usr/bin/env python3
"""Test script to run quantization and capture detailed error messages."""

from __future__ import absolute_import, print_function
import os, sys, time
import numpy as np
from collections import namedtuple
import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.relay import transform

import torch
import torchvision

# Robust import of external ofa_base_models
external_repo_root = "/home/srchand/Desktop/research/OFA_Obfs"
if external_repo_root not in sys.path:
    sys.path.insert(0, external_repo_root)

_mod = sys.modules.get("ofa_base_models")
if _mod is not None:
    try:
        _mod_file = getattr(_mod, "__file__", "") or ""
        if "TVM_Intel_Fork/tvm/vta/sri_scripts/jupyter_nbs/ofa_base_models" in _mod_file:
            del sys.modules["ofa_base_models"]
    except Exception:
        sys.modules.pop("ofa_base_models", None)

from ofa_base_models import OFADynamicResnetAllMod

import json
import re

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

channels_re = re.compile('.*Tensor\[\(([\d]+), ([\d]+), [\d]+, [\d]+\).*padding.*Tensor\[\([\d]+, [\d]+, ([\d]+), ([\d]+)\).*')
cast_re = re.compile('cast.*Tensor\[\([\d]+, [\d]+, ([\d]+), ([\d]+)\).*')

def extract_wkls(ofa_net, arch_mapping, model_id):
    pytorch_model = ofa_net
    pytorch_model.set_active_subnet(arch_mapping[model_id])
    pytorch_model.precompute_active_weights(arch_mapping[model_id])
    pytorch_model.eval()
    for mod in pytorch_model.modules():
        if hasattr(mod, 'export_detach_cached_filters'):
            mod.export_detach_cached_filters = True

    workloads = []
    count = 0
    for layer in pytorch_model.modules():
        if type(layer) == torch.nn.modules.conv.Conv2d:
            if(layer.in_channels % 16 == 0 and layer.out_channels % 16 ==0 and layer.padding[0] == layer.padding[1]):
                workloads.append(Workload(1, 0, 0, layer.in_channels, layer.out_channels,
                                  layer.kernel_size[0], layer.kernel_size[1], layer.padding[0], layer.padding[1]
                                 , layer.stride[0], layer.stride[1]))

    print(f"Found {len(workloads)} conv2d layers to process")

    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape)

    print("Tracing PyTorch model...")
    scripted_model = torch.jit.trace(pytorch_model, input_data).eval()

    print("Converting to Relay IR...")
    shape_list = [("input0", input_shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    print(f"Number of parameters: {len(params)}")

    # Ensure types are inferred before quantization
    print("Running InferType pass...")
    mod = relay.transform.InferType()(mod)
    print("  InferType completed successfully")

    # Perform quantization - let quantize() handle parameter binding internally
    print("\nStarting quantization...")
    try:
        with tvm.transform.PassContext(opt_level=3):
            with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
                print("  Calling relay.quantize.quantize()...")
                mod = relay.quantize.quantize(mod, params=params)
        print("Quantization successful!")
    except Exception as e:
        print(f"\n{'='*80}")
        print("QUANTIZATION FAILED")
        print(f"{'='*80}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message:\n{str(e)}")
        print(f"{'='*80}")
        raise

    mod_as_string = mod.astext(show_meta_data=False)
    cast_line = ""
    cast_line_idx = -1
    final_workloads = []

    for i, line in enumerate(mod_as_string.split('\n')):
        if "cast" in line and "int8" in line:
            cast_line = line
            cast_line_idx = i
        elif "conv2d" in line and "int8" in line:
            match = re.search(channels_re, line)
            if match:
                if int(match.group(1)) % 16 == 0 and int(match.group(2)) % 16 == 0:
                    match_cast = re.search(cast_re, cast_line)
                    if match_cast:
                        wkl = workloads[count]
                        final_workloads.append(
                        'Workload({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})'.format(
                        1, match_cast.group(1), match_cast.group(2), wkl.in_filter, wkl.out_filter,
                        wkl.hkernel,wkl.wkernel, wkl.hpad, wkl.wpad, wkl.hstride, wkl.wstride))
                        count += 1

    return final_workloads


if __name__ == "__main__":
    print("Loading OFA model...")
    net = OFADynamicResnetAllMod()
    model_path = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth"
    device_temp = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device_temp)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    else:
        state = checkpoint
    net.load_state_dict(state, strict=False)

    print("Loading architecture configurations...")
    candidate_set_json_path = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/candidates_25_85.json"
    arch_config_json_path = "/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json"

    def load_arch_mapping(path):
        with open(path, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict) and all(isinstance(v, dict) and 'residual_depth_list' in v for v in data.values()):
            return data

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

    arch_mapping = load_arch_mapping(arch_config_json_path)

    with open(candidate_set_json_path, 'r') as f:
        candidate_set = json.load(f)
    model_ids = candidate_set['selected_model_ids']

    print(f"Testing with model ID: {model_ids[0]}")
    print(f"\n{'='*80}")
    print("STARTING WORKLOAD EXTRACTION")
    print(f"{'='*80}\n")

    with torch.no_grad():
        workloads = extract_wkls(net, arch_mapping, model_ids[0])

    print(f"\n{'='*80}")
    print(f"SUCCESS! Extracted {len(workloads)} workloads")
    print(f"{'='*80}")
    for i, wkl in enumerate(workloads[:5]):  # Show first 5
        print(f"{i+1}. {wkl}")
    if len(workloads) > 5:
        print(f"... and {len(workloads) - 5} more")

