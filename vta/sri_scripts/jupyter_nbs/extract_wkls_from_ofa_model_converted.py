#!/usr/bin/env python
# coding: utf-8
{
 "cells": [
  {
   "cell_type": "code",
   "id": "initial_id",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2026-01-17T07:08:45.033354274Z",
     "start_time": "2026-01-17T07:08:43.713804209Z"
    }
   },
   "source": [
    "from __future__ import absolute_import, print_function\n",
    "# add external modules to PYTHONPATH via environment variable\n",
    "import os, sys, time\n",
    "from PIL import Image\n",
    "\n",
    "import numpy as np\n",
    "from matplotlib import pyplot as plt\n",
    "from collections import namedtuple\n",
    "import tvm\n",
    "from tvm import te\n",
    "from tvm import rpc, autotvm, relay\n",
    "from tvm.contrib import graph_runtime, utils, download\n",
    "from tvm.contrib.debugger import debug_runtime\n",
    "from tvm.relay import transform\n",
    "\n",
    "import vta\n",
    "from vta.testing import simulator\n",
    "from vta.top import graph_pack\n",
    "\n",
    "import torch\n",
    "import torchvision\n",
    "from tvm.contrib.download import download_testdata\n",
    "\n",
    "# Robust import of external ofa_base_models without being shadowed by local folder\n",
    "external_repo_root = \"/home/srchand/Desktop/research/OFA_Obfs\"\n",
    "if external_repo_root not in sys.path:\n",
    "    sys.path.insert(0, external_repo_root)\n",
    "\n",
    "# If a local shim package is already cached, evict it to allow importing the external one\n",
    "_mod = sys.modules.get(\"ofa_base_models\")\n",
    "if _mod is not None:\n",
    "    try:\n",
    "        _mod_file = getattr(_mod, \"__file__\", \"\") or \"\"\n",
    "        if \"TVM_Intel_Fork/tvm/vta/sri_scripts/jupyter_nbs/ofa_base_models\" in _mod_file:\n",
    "            del sys.modules[\"ofa_base_models\"]\n",
    "    except Exception:\n",
    "        # If anything goes wrong, clear the cache entry\n",
    "        sys.modules.pop(\"ofa_base_models\", None)\n",
    "\n",
    "try:\n",
    "    from ofa_base_models import OFADynamicResnetAllMod  # type: ignore\n",
    "except (ModuleNotFoundError, ImportError):\n",
    "    # Final fallback: ensure external root is first in path and retry once\n",
    "    if sys.path[0] != external_repo_root:\n",
    "        sys.path.insert(0, external_repo_root)\n",
    "    # Clear any cached partial import\n",
    "    sys.modules.pop(\"ofa_base_models\", None)\n",
    "    from ofa_base_models import OFADynamicResnetAllMod  # type: ignore\n",
    "else:\n",
    "    # Print where the module is loaded from for debugging/IDE clarity\n",
    "    import ofa_base_models as _obm  # type: ignore\n",
    "    print(\"ofa_base_models loaded from:\", getattr(_obm, \"__file__\", None))\n",
    "\n",
    "import torch\n",
    "\n",
    "from torchvision import transforms\n",
    "\n",
    "\n",
    "# Make sure that TVM was compiled with RPC=1\n",
    "assert tvm.runtime.enabled(\"rpc\")"
   ],
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ofa_base_models loaded from: /home/srchand/Desktop/research/OFA_Obfs/ofa_base_models/__init__.py\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/srchand/anaconda3/envs/tvm-build-il-2/lib/python3.8/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: /home/srchand/anaconda3/envs/tvm-build-il-2/lib/python3.8/site-packages/torchvision/image.so: undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE\n",
      "  warn(f\"Failed to load image Python extension: {e}\")\n"
     ]
    }
   ],
   "execution_count": 1
  },
  {
   "cell_type": "code",
   "id": "3f3507d466c4ae99",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2026-01-17T07:08:45.140694255Z",
     "start_time": "2026-01-17T07:08:45.042652166Z"
    }
   },
   "source": [
    "Workload = namedtuple(\n",
    "    \"Conv2DWorkload\",\n",
    "    [\n",
    "        \"batch\",\n",
    "        \"height\",\n",
    "        \"width\",\n",
    "        \"in_filter\",\n",
    "        \"out_filter\",\n",
    "        \"hkernel\",\n",
    "        \"wkernel\",\n",
    "        \"hpad\",\n",
    "        \"wpad\",\n",
    "        \"hstride\",\n",
    "        \"wstride\",\n",
    "    ],\n",
    ")"
   ],
   "outputs": [],
   "execution_count": 2
  },
  {
   "cell_type": "code",
   "id": "2905709080e12e3a",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2026-01-17T07:08:45.200070029Z",
     "start_time": "2026-01-17T07:08:45.144435033Z"
    }
   },
   "source": [
    "import re\n",
    "channels_re = re.compile('.*Tensor\\[\\(([\\d]+), ([\\d]+), [\\d]+, [\\d]+\\).*padding.*Tensor\\[\\([\\d]+, [\\d]+, ([\\d]+), ([\\d]+)\\).*')\n",
    "cast_re = re.compile('cast.*Tensor\\[\\([\\d]+, [\\d]+, ([\\d]+), ([\\d]+)\\).*')"
   ],
   "outputs": [],
   "execution_count": 3
  },
  {
   "cell_type": "code",
   "id": "2e4f50e591fa3c03",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2026-01-17T07:08:49.170401046Z",
     "start_time": "2026-01-17T07:08:45.202441496Z"
    }
   },
   "source": [
    "from torchinfo import summary\n",
    "\n",
    "net = OFADynamicResnetAllMod()\n",
    "model_path = \"/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/all_mod_aggressive_reg_final_checkpoint.pth\"\n",
    "device_temp = torch.device('cpu')\n",
    "checkpoint = torch.load(model_path, map_location=device_temp)\n",
    "if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:\n",
    "    state = checkpoint['model_state_dict']\n",
    "else:\n",
    "    state = checkpoint\n",
    "# Allow mismatched keys due to local shim modules\n",
    "net.load_state_dict(state, strict=False)\n",
    "\n",
    "summary(net, input_size=(1, 3, 224, 224))"
   ],
   "outputs": [
    {
     "data": {
      "text/plain": [
       "=========================================================================================================\n",
       "Layer (type:depth-idx)                                  Output Shape              Param #\n",
       "=========================================================================================================\n",
       "OFADynamicResnetAllMod                                  [1, 10]                   --\n",
       "├─Sequential: 1-1                                       --                        --\n",
       "│    └─DynamicConv2DAll: 2-1                            [1, 64, 112, 112]         18,944\n",
       "│    └─DynamicBatchNorm2d: 2-2                          [1, 64, 112, 112]         256\n",
       "│    └─ReLU: 2-3                                        [1, 64, 112, 112]         --\n",
       "│    └─MaxPool2d: 2-4                                   [1, 64, 56, 56]           --\n",
       "├─ModuleList: 1-2                                       --                        --\n",
       "│    └─ResNetBlock: 2-5                                 [1, 64, 56, 56]           --\n",
       "│    │    └─ModuleList: 3-1                             --                        3,215,624\n",
       "│    └─ResNetBlock: 2-6                                 [1, 128, 28, 28]          --\n",
       "│    │    └─ModuleList: 3-2                             --                        11,278,344\n",
       "│    └─ResNetBlock: 2-7                                 [1, 256, 14, 14]          --\n",
       "│    │    └─ModuleList: 3-3                             --                        45,098,248\n",
       "│    └─ResNetBlock: 2-8                                 [1, 512, 7, 7]            --\n",
       "│    │    └─ModuleList: 3-4                             --                        180,371,208\n",
       "├─Sequential: 1-3                                       [1, 10]                   --\n",
       "│    └─AdaptiveAvgPool2d: 2-9                           [1, 512, 1, 1]            --\n",
       "│    └─Flatten: 2-10                                    [1, 512]                  --\n",
       "│    └─DynamicLinearLayer: 2-11                         [1, 10]                   --\n",
       "│    │    └─DynamicLinear: 3-5                          [1, 10]                   10,250\n",
       "=========================================================================================================\n",
       "Total params: 239,992,874\n",
       "Trainable params: 239,992,874\n",
       "Non-trainable params: 0\n",
       "Total mult-adds (M): 0\n",
       "=========================================================================================================\n",
       "Input size (MB): 0.60\n",
       "Forward/backward pass size (MB): 0.00\n",
       "Params size (MB): 0.00\n",
       "Estimated Total Size (MB): 0.60\n",
       "========================================================================================================="
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "execution_count": 4
  },
  {
   "metadata": {
    "ExecuteTime": {
     "end_time": "2026-01-17T07:08:50.321494631Z",
     "start_time": "2026-01-17T07:08:49.867815992Z"
    }
   },
   "cell_type": "code",
   "source": [
    "arch = net.sample_arch()\n",
    "print(arch)\n",
    "net.set_active_subnet(arch)\n",
    "net.precompute_active_weights(arch)\n",
    "# net.enable_auto_precompute()\n",
    "summary(net, input_size=(1, 3, 224, 224))"
   ],
   "id": "cfde36bc7858dcbf",
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'residual_depth_list': [[1, 1], [2, 1], [2, 2], [1, 2]], 'decomp_type_list': [[[2], [2]], [[4, 4], [4]], [[4, 0], [3, 1]], [[2], [4, 1]]], 'downsample_decomp_type_list': [0, 4, 4], 'kernel_size_list': [[[3], [7]], [[3, 5], [3]], [[7, 7], [5, 7]], [[5], [3, 5]]], 'out_channel_setting_list': [1, 1, 2, 2, 2]}\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "=========================================================================================================\n",
       "Layer (type:depth-idx)                                  Output Shape              Param #\n",
       "=========================================================================================================\n",
       "OFADynamicResnetAllMod                                  [1, 10]                   --\n",
       "├─Sequential: 1-1                                       --                        --\n",
       "│    └─DynamicConv2DAll: 2-1                            [1, 64, 112, 112]         18,944\n",
       "│    └─DynamicBatchNorm2d: 2-2                          [1, 64, 112, 112]         256\n",
       "│    └─ReLU: 2-3                                        [1, 64, 112, 112]         --\n",
       "│    └─MaxPool2d: 2-4                                   [1, 64, 56, 56]           --\n",
       "├─ModuleList: 1-2                                       --                        --\n",
       "│    └─ResNetBlock: 2-5                                 [1, 64, 56, 56]           --\n",
       "│    │    └─ModuleList: 3-1                             --                        3,215,624\n",
       "│    └─ResNetBlock: 2-6                                 [1, 256, 28, 28]          --\n",
       "│    │    └─ModuleList: 3-2                             --                        11,278,344\n",
       "│    └─ResNetBlock: 2-7                                 [1, 512, 14, 14]          --\n",
       "│    │    └─ModuleList: 3-3                             --                        45,098,248\n",
       "│    └─ResNetBlock: 2-8                                 [1, 1024, 7, 7]           --\n",
       "│    │    └─ModuleList: 3-4                             --                        180,371,208\n",
       "├─Sequential: 1-3                                       [1, 10]                   --\n",
       "│    └─AdaptiveAvgPool2d: 2-9                           [1, 1024, 1, 1]           --\n",
       "│    └─Flatten: 2-10                                    [1, 1024]                 --\n",
       "│    └─DynamicLinearLayer: 2-11                         [1, 10]                   --\n",
       "│    │    └─DynamicLinear: 3-5                          [1, 10]                   10,250\n",
       "=========================================================================================================\n",
       "Total params: 239,992,874\n",
       "Trainable params: 239,992,874\n",
       "Non-trainable params: 0\n",
       "Total mult-adds (M): 0.01\n",
       "=========================================================================================================\n",
       "Input size (MB): 0.60\n",
       "Forward/backward pass size (MB): 9.23\n",
       "Params size (MB): 0.05\n",
       "Estimated Total Size (MB): 9.88\n",
       "========================================================================================================="
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "execution_count": 5
  },
  {
   "cell_type": "code",
   "id": "3b25149a7d4515bd",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2026-01-17T07:08:50.382617370Z",
     "start_time": "2026-01-17T07:08:50.326212719Z"
    }
   },
   "source": [
    "import json\n",
    "from typing import Dict, Any\n",
    "\n",
    "candidate_set_json_path=\"/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/candidates_25_85.json\"\n",
    "arch_config_json_path=\"/mnt/hgfs/vmware_ubuntu_sf/OFA_networks/candidate_set_final/architectures_20250927_180844.json\"\n",
    "\n",
    "def load_arch_mapping(path: str) -> Dict[str, Any]:\n",
    "    \"\"\"Load architectures JSON and normalize to a mapping {id: architecture_dict}.\n",
    "\n",
    "    Supported input formats:\n",
    "    - A dict mapping id -> architecture dict (legacy)\n",
    "    - A dict with key 'architectures' containing a list of items with fields 'id' and 'architecture'\n",
    "    - A top-level list of items with 'id' and 'architecture'\n",
    "    \"\"\"\n",
    "    with open(path, 'r') as f:\n",
    "        data = json.load(f)\n",
    "\n",
    "    # Case 1: already a mapping from id -> arch\n",
    "    if isinstance(data, dict) and all(isinstance(v, dict) and 'residual_depth_list' in v for v in data.values()):\n",
    "        return data\n",
    "\n",
    "    # Case 2: top-level dict with 'architectures' list\n",
    "    if isinstance(data, dict) and 'architectures' in data and isinstance(data['architectures'], list):\n",
    "        mapping = {}\n",
    "        for item in data['architectures']:\n",
    "            # item may contain fields 'id' and 'architecture' (nested)\n",
    "            if 'id' in item and 'architecture' in item:\n",
    "                mapping[item['id']] = item['architecture']\n",
    "            elif 'id' in item and 'arch' in item:\n",
    "                mapping[item['id']] = item['arch']\n",
    "            else:\n",
    "                # If the item itself is an architecture dict without id, generate an id\n",
    "                if 'id' in item:\n",
    "                    mapping[item['id']] = item\n",
    "        return mapping\n",
    "\n",
    "    # Case 3: top-level list of architecture items\n",
    "    if isinstance(data, list):\n",
    "        mapping = {}\n",
    "        for idx, item in enumerate(data):\n",
    "            if isinstance(item, dict) and 'id' in item and 'architecture' in item:\n",
    "                mapping[item['id']] = item['architecture']\n",
    "            elif isinstance(item, dict) and 'id' in item:\n",
    "                mapping[item['id']] = item\n",
    "            else:\n",
    "                mapping[f'arch_{idx}'] = item\n",
    "        return mapping\n",
    "\n",
    "    # Fallback: unknown format, raise error\n",
    "    raise ValueError(f\"Unsupported architecture file format: {path}\")\n",
    "\n",
    "arch_mapping = load_arch_mapping(arch_config_json_path)\n"
   ],
   "outputs": [],
   "execution_count": 6
  },
  {
   "cell_type": "code",
   "id": "3f7d8f871e8c47c",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2026-01-17T07:08:50.447803366Z",
     "start_time": "2026-01-17T07:08:50.384020164Z"
    }
   },
   "source": [
    "# read candidate set json\n",
    "with open(candidate_set_json_path, 'r') as f:\n",
    "    candidate_set = json.load(f)\n",
    "model_ids = candidate_set['selected_model_ids']"
   ],
   "outputs": [],
   "execution_count": 7
  },
  {
   "cell_type": "code",
   "id": "f5d98d4fec106628",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2026-01-17T07:08:50.515026213Z",
     "start_time": "2026-01-17T07:08:50.449842098Z"
    }
   },
   "source": [
    "def extract_wkls(ofa_net, arch_mapping, model_id):\n",
    "    pytorch_model = ofa_net\n",
    "    pytorch_model.set_active_subnet(arch_mapping[model_id])\n",
    "    pytorch_model.precompute_active_weights(arch_mapping[model_id])\n",
    "    pytorch_model.eval()\n",
    "    for mod in pytorch_model.modules():\n",
    "        if hasattr(mod, 'export_detach_cached_filters'):\n",
    "            mod.export_detach_cached_filters = True\n",
    "    workloads = []\n",
    "    count = 0\n",
    "    for layer in pytorch_model.modules():\n",
    "        if type(layer) == torch.nn.modules.conv.Conv2d:\n",
    "            if(layer.in_channels % 16 == 0 and layer.out_channels % 16 ==0 and layer.padding[0] == layer.padding[1]):\n",
    "                workloads.append(Workload(1, 0, 0, layer.in_channels, layer.out_channels,\n",
    "                                  layer.kernel_size[0], layer.kernel_size[1], layer.padding[0], layer.padding[1]\n",
    "                                 , layer.stride[0], layer.stride[1]))\n",
    "    input_shape = [1, 3, 224, 224]\n",
    "    input_data = torch.randn(input_shape)\n",
    "    scripted_model = torch.jit.trace(pytorch_model, input_data).eval()\n",
    "    shape_list = [(\"input0\", input_shape)]\n",
    "    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)\n",
    "    # print(mod.astext(show_meta_data=False))\n",
    "\n",
    "    # Bind parameters and run type inference before quantization\n",
    "    mod = relay.transform.InferType()(mod)\n",
    "    fn = mod[\"main\"]\n",
    "    fn = relay.build_module.bind_params_by_name(fn, params)\n",
    "    mod = tvm.IRModule.from_expr(fn)\n",
    "    mod = relay.transform.InferType()(mod)\n",
    "\n",
    "    with tvm.transform.PassContext(opt_level=3):\n",
    "        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):\n",
    "             # params already bound, so pass None\n",
    "             mod = relay.quantize.quantize(mod, params=None)\n",
    "\n",
    "    mod_as_string = mod.astext(show_meta_data=False)\n",
    "    cast_line = \"\"\n",
    "    cast_line_idx = -1\n",
    "    final_workloads = []\n",
    "    for i, line in enumerate(mod_as_string.split('\\n')):\n",
    "        if \"cast\" in line and \"int8\" in line:\n",
    "            cast_line = line\n",
    "            cast_line_idx = i\n",
    "        elif \"conv2d\" in line and \"int8\" in line:\n",
    "            match = re.search(channels_re, line)\n",
    "            if match:\n",
    "                if int(match.group(1)) % 16 == 0 and int(match.group(2)) % 16 == 0:\n",
    "                    match_cast = re.search(cast_re, cast_line)\n",
    "                    if match_cast:\n",
    "                        wkl = workloads[count]\n",
    "                        final_workloads.append(\n",
    "                        'Workload({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})'.format(\n",
    "                        1, match_cast.group(1), match_cast.group(2), wkl.in_filter, wkl.out_filter,\n",
    "                        wkl.hkernel,wkl.wkernel, wkl.hpad, wkl.wpad, wkl.hstride, wkl.wstride))\n",
    "                        count += 1\n",
    "\n",
    "    return final_workloads"
   ],
   "outputs": [],
   "execution_count": 8
  },
  {
   "cell_type": "code",
   "id": "16031c550ee229ee",
   "metadata": {
    "ExecuteTime": {
     "end_time": "2026-01-17T07:08:57.192829584Z",
     "start_time": "2026-01-17T07:08:50.516748354Z"
    }
   },
   "source": [
    "with torch.no_grad():\n",
    "    extract_wkls(net,arch_mapping, model_ids[0])"
   ],
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/home/srchand/Desktop/research/OFA_Obfs/ofa_base_models/ofa_ops/dynamic_bn.py:15: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:\n",
      "/home/srchand/Desktop/research/OFA_Obfs/ofa_base_models/ofa_ops/dynamic_residual_all.py:49: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!\n",
      "  elif identity.shape[1] != x.shape[1]:\n"
     ]
    },
    {
     "ename": "TVMError",
     "evalue": "Traceback (most recent call last):\n  [bt] (8) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(TVMFuncCall+0x63) [0x7f005011f1c3]\n  [bt] (7) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(+0x9224f7) [0x7f004f5084f7]\n  [bt] (6) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const+0x33c) [0x7f004f50702c]\n  [bt] (5) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const+0xccf) [0x7f005003c49f]\n  [bt] (4) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const+0x23b) [0x7f004f50798b]\n  [bt] (3) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(+0x132f46a) [0x7f004ff1546a]\n  [bt] (2) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::relay::TypeInferencer::Infer(tvm::GlobalVar, tvm::relay::Function)+0x75) [0x7f004ff146f5]\n  [bt] (1) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(+0x545f1d) [0x7f004f12bf1d]\n  [bt] (0) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(+0x118f5f8) [0x7f004fd755f8]\n  [bt] (8) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const+0x33c) [0x7f004f50702c]\n  [bt] (7) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const+0xccf) [0x7f005003c49f]\n  [bt] (6) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const+0x23b) [0x7f004f50798b]\n  [bt] (5) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(+0x132f46a) [0x7f004ff1546a]\n  [bt] (4) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::relay::TypeInferencer::Infer(tvm::GlobalVar, tvm::relay::Function)+0x75) [0x7f004ff146f5]\n  [bt] (3) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::relay::TypeSolver::Solve()+0x459) [0x7f004fd78089]\n  [bt] (2) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::runtime::TypedPackedFunc<bool (tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>::AssignTypedLambda<bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>(bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&))::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*) const+0x2c2) [0x7f004f562f62]\n  [bt] (1) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::relay::quantize::SimulatedQuantizeRel(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)+0x7c0) [0x7f004fda70c0]\n  [bt] (0) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(+0x11c0578) [0x7f004fda6578]\n  File \"/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/src/relay/analysis/type_solver.cc\", line 624\nTVMError: \n---------------------------------------------------------------\nAn internal invariant was violated during the execution of TVM.\nPlease read TVM's error reporting guidelines.\nMore details can be found here: https://discuss.tvm.ai/t/error-reporting/7793.\n---------------------------------------------------------------\n  Check failed: false == false: [15:08:56] /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/src/relay/quantize/quantize.cc:52: \n---------------------------------------------------------------\nAn internal invariant was violated during the execution of TVM.\nPlease read TVM's error reporting guidelines.\nMore details can be found here: https://discuss.tvm.ai/t/error-reporting/7793.\n---------------------------------------------------------------\n\n  Check failed: data->shape.size() != 0 (0 vs. 0) : Input shape cannot be empty\n",
     "output_type": "error",
     "traceback": [
      "\u001B[0;31m---------------------------------------------------------------------------\u001B[0m",
      "\u001B[0;31mTVMError\u001B[0m                                  Traceback (most recent call last)",
      "Input \u001B[0;32mIn [9]\u001B[0m, in \u001B[0;36m<cell line: 1>\u001B[0;34m()\u001B[0m\n\u001B[1;32m      1\u001B[0m \u001B[38;5;28;01mwith\u001B[39;00m torch\u001B[38;5;241m.\u001B[39mno_grad():\n\u001B[0;32m----> 2\u001B[0m     \u001B[43mextract_wkls\u001B[49m\u001B[43m(\u001B[49m\u001B[43mnet\u001B[49m\u001B[43m,\u001B[49m\u001B[43march_mapping\u001B[49m\u001B[43m,\u001B[49m\u001B[43m \u001B[49m\u001B[43mmodel_ids\u001B[49m\u001B[43m[\u001B[49m\u001B[38;5;241;43m0\u001B[39;49m\u001B[43m]\u001B[49m\u001B[43m)\u001B[49m\n",
      "Input \u001B[0;32mIn [8]\u001B[0m, in \u001B[0;36mextract_wkls\u001B[0;34m(ofa_net, arch_mapping, model_id)\u001B[0m\n\u001B[1;32m     28\u001B[0m \u001B[38;5;28;01mwith\u001B[39;00m tvm\u001B[38;5;241m.\u001B[39mtransform\u001B[38;5;241m.\u001B[39mPassContext(opt_level\u001B[38;5;241m=\u001B[39m\u001B[38;5;241m3\u001B[39m):\n\u001B[1;32m     29\u001B[0m     \u001B[38;5;28;01mwith\u001B[39;00m relay\u001B[38;5;241m.\u001B[39mquantize\u001B[38;5;241m.\u001B[39mqconfig(global_scale\u001B[38;5;241m=\u001B[39m\u001B[38;5;241m8.0\u001B[39m, skip_conv_layers\u001B[38;5;241m=\u001B[39m[\u001B[38;5;241m0\u001B[39m]):\n\u001B[0;32m---> 30\u001B[0m          mod \u001B[38;5;241m=\u001B[39m \u001B[43mrelay\u001B[49m\u001B[38;5;241;43m.\u001B[39;49m\u001B[43mquantize\u001B[49m\u001B[38;5;241;43m.\u001B[39;49m\u001B[43mquantize\u001B[49m\u001B[43m(\u001B[49m\u001B[43mmod\u001B[49m\u001B[43m,\u001B[49m\u001B[43m \u001B[49m\u001B[43mparams\u001B[49m\u001B[38;5;241;43m=\u001B[39;49m\u001B[43mparams\u001B[49m\u001B[43m)\u001B[49m\n\u001B[1;32m     32\u001B[0m mod_as_string \u001B[38;5;241m=\u001B[39m mod\u001B[38;5;241m.\u001B[39mastext(show_meta_data\u001B[38;5;241m=\u001B[39m\u001B[38;5;28;01mFalse\u001B[39;00m)\n\u001B[1;32m     33\u001B[0m cast_line \u001B[38;5;241m=\u001B[39m \u001B[38;5;124m\"\u001B[39m\u001B[38;5;124m\"\u001B[39m\n",
      "File \u001B[0;32m~/Desktop/research/TVM_Intel_Fork/tvm/python/tvm/relay/quantize/quantize.py:371\u001B[0m, in \u001B[0;36mquantize\u001B[0;34m(mod, params, dataset)\u001B[0m\n\u001B[1;32m    367\u001B[0m \u001B[38;5;28;01mwith\u001B[39;00m tvm\u001B[38;5;241m.\u001B[39mtransform\u001B[38;5;241m.\u001B[39mPassContext(\n\u001B[1;32m    368\u001B[0m     opt_level\u001B[38;5;241m=\u001B[39m\u001B[38;5;241m3\u001B[39m, required_pass\u001B[38;5;241m=\u001B[39m[\u001B[38;5;124m\"\u001B[39m\u001B[38;5;124mQuantizeAnnotate\u001B[39m\u001B[38;5;124m\"\u001B[39m, \u001B[38;5;124m\"\u001B[39m\u001B[38;5;124mQuantizeCalibrate\u001B[39m\u001B[38;5;124m\"\u001B[39m, \u001B[38;5;124m\"\u001B[39m\u001B[38;5;124mQuantizeRealize\u001B[39m\u001B[38;5;124m\"\u001B[39m]\n\u001B[1;32m    369\u001B[0m ):\n\u001B[1;32m    370\u001B[0m     \u001B[38;5;28;01mwith\u001B[39;00m quantize_context():\n\u001B[0;32m--> 371\u001B[0m         mod \u001B[38;5;241m=\u001B[39m \u001B[43mquantize_seq\u001B[49m\u001B[43m(\u001B[49m\u001B[43mmod\u001B[49m\u001B[43m)\u001B[49m\n\u001B[1;32m    373\u001B[0m q_cfg \u001B[38;5;241m=\u001B[39m current_qconfig()\n\u001B[1;32m    374\u001B[0m \u001B[38;5;28;01massert\u001B[39;00m q_cfg\u001B[38;5;241m.\u001B[39mpartition_conversions \u001B[38;5;129;01min\u001B[39;00m [\u001B[38;5;124m\"\u001B[39m\u001B[38;5;124mdisabled\u001B[39m\u001B[38;5;124m\"\u001B[39m, \u001B[38;5;124m\"\u001B[39m\u001B[38;5;124menabled\u001B[39m\u001B[38;5;124m\"\u001B[39m, \u001B[38;5;124m\"\u001B[39m\u001B[38;5;124mfully_integral\u001B[39m\u001B[38;5;124m\"\u001B[39m]\n",
      "File \u001B[0;32m~/Desktop/research/TVM_Intel_Fork/tvm/python/tvm/ir/transform.py:127\u001B[0m, in \u001B[0;36mPass.__call__\u001B[0;34m(self, mod)\u001B[0m\n\u001B[1;32m    113\u001B[0m \u001B[38;5;28;01mdef\u001B[39;00m \u001B[38;5;21m__call__\u001B[39m(\u001B[38;5;28mself\u001B[39m, mod):\n\u001B[1;32m    114\u001B[0m     \u001B[38;5;124;03m\"\"\"Execute the pass. Note that for sequential pass, the dependency among\u001B[39;00m\n\u001B[1;32m    115\u001B[0m \u001B[38;5;124;03m    different passes will be resolved in the backend.\u001B[39;00m\n\u001B[1;32m    116\u001B[0m \n\u001B[0;32m   (...)\u001B[0m\n\u001B[1;32m    125\u001B[0m \u001B[38;5;124;03m        The updated module after applying this pass.\u001B[39;00m\n\u001B[1;32m    126\u001B[0m \u001B[38;5;124;03m    \"\"\"\u001B[39;00m\n\u001B[0;32m--> 127\u001B[0m     \u001B[38;5;28;01mreturn\u001B[39;00m \u001B[43m_ffi_transform_api\u001B[49m\u001B[38;5;241;43m.\u001B[39;49m\u001B[43mRunPass\u001B[49m\u001B[43m(\u001B[49m\u001B[38;5;28;43mself\u001B[39;49m\u001B[43m,\u001B[49m\u001B[43m \u001B[49m\u001B[43mmod\u001B[49m\u001B[43m)\u001B[49m\n",
      "File \u001B[0;32m~/Desktop/research/TVM_Intel_Fork/tvm/python/tvm/_ffi/_ctypes/packed_func.py:237\u001B[0m, in \u001B[0;36mPackedFuncBase.__call__\u001B[0;34m(self, *args)\u001B[0m\n\u001B[1;32m    225\u001B[0m ret_tcode \u001B[38;5;241m=\u001B[39m ctypes\u001B[38;5;241m.\u001B[39mc_int()\n\u001B[1;32m    226\u001B[0m \u001B[38;5;28;01mif\u001B[39;00m (\n\u001B[1;32m    227\u001B[0m     _LIB\u001B[38;5;241m.\u001B[39mTVMFuncCall(\n\u001B[1;32m    228\u001B[0m         \u001B[38;5;28mself\u001B[39m\u001B[38;5;241m.\u001B[39mhandle,\n\u001B[0;32m   (...)\u001B[0m\n\u001B[1;32m    235\u001B[0m     \u001B[38;5;241m!=\u001B[39m \u001B[38;5;241m0\u001B[39m\n\u001B[1;32m    236\u001B[0m ):\n\u001B[0;32m--> 237\u001B[0m     \u001B[38;5;28;01mraise\u001B[39;00m get_last_ffi_error()\n\u001B[1;32m    238\u001B[0m _ \u001B[38;5;241m=\u001B[39m temp_args\n\u001B[1;32m    239\u001B[0m _ \u001B[38;5;241m=\u001B[39m args\n",
      "\u001B[0;31mTVMError\u001B[0m: Traceback (most recent call last):\n  [bt] (8) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(TVMFuncCall+0x63) [0x7f005011f1c3]\n  [bt] (7) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(+0x9224f7) [0x7f004f5084f7]\n  [bt] (6) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const+0x33c) [0x7f004f50702c]\n  [bt] (5) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const+0xccf) [0x7f005003c49f]\n  [bt] (4) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const+0x23b) [0x7f004f50798b]\n  [bt] (3) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(+0x132f46a) [0x7f004ff1546a]\n  [bt] (2) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::relay::TypeInferencer::Infer(tvm::GlobalVar, tvm::relay::Function)+0x75) [0x7f004ff146f5]\n  [bt] (1) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(+0x545f1d) [0x7f004f12bf1d]\n  [bt] (0) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(+0x118f5f8) [0x7f004fd755f8]\n  [bt] (8) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::transform::SequentialNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const+0x33c) [0x7f004f50702c]\n  [bt] (7) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::relay::transform::FunctionPassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const+0xccf) [0x7f005003c49f]\n  [bt] (6) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const+0x23b) [0x7f004f50798b]\n  [bt] (5) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(+0x132f46a) [0x7f004ff1546a]\n  [bt] (4) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::relay::TypeInferencer::Infer(tvm::GlobalVar, tvm::relay::Function)+0x75) [0x7f004ff146f5]\n  [bt] (3) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::relay::TypeSolver::Solve()+0x459) [0x7f004fd78089]\n  [bt] (2) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::runtime::TypedPackedFunc<bool (tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>::AssignTypedLambda<bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>(bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&))::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}::operator()(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*) const+0x2c2) [0x7f004f562f62]\n  [bt] (1) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(tvm::relay::quantize::SimulatedQuantizeRel(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)+0x7c0) [0x7f004fda70c0]\n  [bt] (0) /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/build/libtvm.so(+0x11c0578) [0x7f004fda6578]\n  File \"/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/src/relay/analysis/type_solver.cc\", line 624\nTVMError: \n---------------------------------------------------------------\nAn internal invariant was violated during the execution of TVM.\nPlease read TVM's error reporting guidelines.\nMore details can be found here: https://discuss.tvm.ai/t/error-reporting/7793.\n---------------------------------------------------------------\n  Check failed: false == false: [15:08:56] /home/srchand/Desktop/research/TVM_Intel_Fork/tvm/src/relay/quantize/quantize.cc:52: \n---------------------------------------------------------------\nAn internal invariant was violated during the execution of TVM.\nPlease read TVM's error reporting guidelines.\nMore details can be found here: https://discuss.tvm.ai/t/error-reporting/7793.\n---------------------------------------------------------------\n\n  Check failed: data->shape.size() != 0 (0 vs. 0) : Input shape cannot be empty\n"
     ]
    }
   ],
   "execution_count": 9
  },
  {
   "cell_type": "code",
   "id": "a3abc753",
   "metadata": {},
   "source": [],
   "outputs": [],
   "execution_count": null
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:tvm-build-il-2] *",
   "language": "python",
   "name": "conda-env-tvm-build-il-2-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

