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

from torchvision import transforms


# Make sure that TVM was compiled with RPC=1
assert tvm.runtime.enabled("rpc")

# Load VTA parameters from the 3rdparty/vta-hw/config/vta_config.json file
env = vta.get_env()

# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

# Dictionary lookup for when to start/end bit packing
pack_dict = {
    "resnet18_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet18": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
    "resnet34": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
    "resnet50": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
    "resnet101": ["nn.max_pool2d", "nn.adaptive_avg_pool2d"],
    "vgg11": ["nn.max_pool2d", "nn.dense"],
    "vgg16":    ["nn.max_pool2d", "nn.dense"],
    "resnet34_v1": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet18_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet34_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet50_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "resnet101_v2": ["nn.max_pool2d", "nn.global_avg_pool2d"],
    "mobilenetv2_1.0": ["nn.max_pool2d", "nn.global_avg_pool2d"]
}

# Name of Gluon model to compile
# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA.
#model = "resnet18_v1"
#assert model in pack_dict
model = "resnet18"

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
# with torch.no_grad():
#     summary(net, input_size=(1, 3, 224, 224))


remote = None
if env.TARGET not in ["sim", "tsim", "intelfocl"]:

    # Get remote from tracker node if environment variable is set.
    # To set up the tracker, you'll need to follow the "Auto-tuning
    # a convolutional network for VTA" tutorial.
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = os.environ.get("TVM_TRACKER_PORT", None)
    # Otherwise if you have a device you want to program directly from
    # the host, make sure you've set the variables below to the IP of
    # your board.
#     device_host = os.environ.get("VTA_RPC_HOST", "192.168.2.99")
    device_host="10.42.0.188"
#     device_host="10.100.86.111"
    device_port = os.environ.get("VTA_RPC_PORT", "9091")
    if not tracker_host or not tracker_port:
        remote = rpc.connect(device_host, int(device_port))
    else:
        remote = autotvm.measure.request_remote(
            env.TARGET, tracker_host, int(tracker_port), timeout=10000
        )

    # Reconfigure the JIT runtime and FPGA.
    # You can program the FPGA with your own custom bitstream
    # by passing the path to the bitstream file instead of None.
    reconfig_start = time.time()
    vta.reconfig_runtime(remote)

#     vta.program_fpga(remote, bitstream="/mnt/hgfs/vmware_ubuntu_sf/bitstreams/vta_pynq_sniffer_reset_on_read.bit")
#     vta.program_fpga(remote, bitstream='/mnt/hgfs/vmware_ubuntu_sf/vta_4x8x8/vta_new_1x16x16_memory_trojan_runtime_sampling.bit')
#     vta.program_fpga(remote, bitstream='/mnt/hgfs/vmware_ubuntu_sf/vta_4x8x8/vta_1x16x16_acc_18_memory_trojan_runtime_sampling.bit')
#     vta.program_fpga(remote, bitstream='/mnt/hgfs/vmware_ubuntu_sf/vta_4x8x8/vta_new_1x16x16_memory_trojan_runtime_sampling.bit')
    reconfig_time = time.time() - reconfig_start
    print("Reconfigured FPGA and RPC runtime in {0:.2f}s!".format(reconfig_time))

# In simulation mode, host the RPC server locally.
else:
    remote = rpc.LocalSession()

    if env.TARGET in ["intelfocl"]:
        # program intelfocl aocx
        vta.program_fpga(remote, bitstream="vta.bitstream")

# Get execution context from remote
ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)

import glob
# schedule_log_files = glob.glob(r'../logs/tuning_logs/vta_1x16x16_pynq_arm/*.log')
schedule_log_files = glob.glob(r'/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/logs/tuning_logs/vta_1x16x16/candidate_set/*.log')
# schedule_log_files = glob.glob(r'/home/srchand/Desktop/research/neurobfuscator/seq_obfuscator/obf_tmp_file/autotvm_model_10_obf_pruned.log')

# schedule_log_files = glob.glob(r'../logs/tuning_logs/*.log')

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

# print("Using model id:", model_ids[4])
# print("With architecture:", arch_mapping[model_ids[4]])
# net.set_active_subnet(arch_mapping[model_ids[4]])
not_working = []
pytorch_model = ofa_net
# model_ids = ["arch_20250927_180844_0882"]
for i_net, id in enumerate(model_ids[:2]):
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

    pytorch_model.set_active_subnet(arch)
    standalone_net = StaticResNetFromArch(
        target_arch=arch,
        num_classes=10,
        width_mult_list=(0.5, 1.0, 2.0)
    )
    standalone_net.load_weights_from_ofa_checkpoint(checkpoint_path=model_path, ofa_model=ofa_net)
    # net = standalone_net
    # remove the first batch norm layer for the first conv layer to avoid unsupported pattern in relay frontend
    # net.blocks[0][0].main_path[1] = nn.Identity()
    try:
        temp = utils.tempdir()
        with autotvm.tophub.context(target, extra_files=schedule_log_files):
            # with autotvm.tophub.context(target):

            input_name = "input0"

            # Populate the shape and data type dictionary for ImageNet classifier input
            dtype_dict = {input_name: "float32"}
            shape_dict = {input_name: (env.BATCH, 3, 224, 224)}

            #     # Get off the shelf gluon model, and convert to relay
            #     gluon_model = vision.get_model(model, pretrained=True)

            # pytorch_model = getattr(torchvision.models, model)(pretrained=True).eval()
            # pytorch_model = net
            pytorch_model.eval()

            input_shape = [env.BATCH, 3, 224, 224]
            input_data = torch.randn(input_shape)
            # pytorch_model(input_data)
            scripted_model = torch.jit.trace(pytorch_model, input_data).eval()

            shape_list = [(input_name, input_shape)]

            # Measure build start time
            build_start = time.time()

            #     Start front end compilation
            mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

            #     mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

            #     #mod, params = relay.frontend.from_mxnet(net, shape_dict)

            #     # Update shape and type dictionary
            # print(mod.astext(show_meta_data=False))

            mod = relay.transform.InferType()(mod)
            shape_dict.update({k: v.shape for k, v in params.items()})
            dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

            if target.device_name == "vta":
                # Perform quantization in Relay
                # Note: We set opt_level to 3 in order to fold batch norm
                with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
                    with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
                        mod = relay.quantize.quantize(mod, params=params)


                    # Perform graph packing and constant folding for VTA target
                    assert env.BLOCK_IN == env.BLOCK_OUT
                    # do device annotation if target is intelfocl or sim
                    relay_prog = graph_pack(
                        mod["main"],
                        env.BATCH,
                        env.BLOCK_IN,
                        env.BLOCK_OUT,
                        env.WGT_WIDTH,
                        start_name=pack_dict[model][0],
                        #                 stop_name='cast',
                        #                 stop_name_idx=114,
                        stop_name=pack_dict[model][1],
                        #                 start_name='nn.relu',
                        #                 start_name_idx=2,
                        #                 stop_name='nn.adaptive_avg_pool2d',
                        #                 start_name="cast",
                        #                 start_name_idx=8,
                        #                 stop_name="cast",
                        #                 stop_name_idx=71,
                        device_annot=(env.TARGET == "intelfocl"),
                    )

            else:
                relay_prog = mod["main"]

            # Compile Relay program with AlterOpLayout disabled
            if target.device_name != "vta":
                with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
                    graph, lib, params = relay.build(
                        relay_prog, target=target, params=params, target_host=env.target_host
                    )
            else:
                if env.TARGET == "intelfocl":
                    # multiple targets to run both on cpu and vta
                    target = {"cpu": env.target_vta_cpu, "ext_dev": target}
                with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
                    graph, lib, params = relay.build(
                        relay_prog, target=target, params=params, target_host=env.target_host
                    )

            # Measure Relay build time
            build_time = time.time() - build_start
            print(model + " inference graph built in {0:.2f}s!".format(build_time))

            # Send the inference library over to the remote RPC server

            lib.export_library(temp.relpath(f"graphlib_{id}.tar"))
            remote.upload(temp.relpath(f"graphlib_{id}.tar"))
            lib = remote.load_module(f"graphlib_{id}.tar")

            if env.TARGET == "intelfocl":
                ctxes = [remote.ext_dev(0), remote.cpu(0)]
                m = graph_runtime.create(graph, lib, ctxes)
            else:
                # Graph runtime
                m = graph_runtime.create(graph, lib, ctx)

        img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        base_dir = "/home/srchand/Desktop/research/TVM_Intel_Fork/tvm/vta/sri_scripts/jupyter_nbs/ImageNette_Images/"
        imagenette_classes = os.path.join(base_dir, "imagenette_classes.txt")
        imagenette = eval(open(imagenette_classes).read())

        imagenette_image = [
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


        def load_images_and_run(idx):
            image = Image.open(imagenette_image[idx]).resize((224, 224))
            plt.imshow(image)
            plt.show()
            image = np.array(image) - np.array([123.0, 117.0, 104.0])
            image /= np.array([58.395, 57.12, 57.375])

            image = image.transpose((2, 0, 1))
            #     image = img_transforms(image).numpy()

            image = image[np.newaxis, :]
            image = np.repeat(image, env.BATCH, axis=0)

            # Set the network parameters and inputs
            m.set_input(**params)
            m.set_input(input_name, image)

            m.run()

            tvm_output = m.get_output(0, tvm.nd.empty((env.BATCH, 10), "float32", remote.cpu(0)))
            for b in range(env.BATCH):
                top_categories = np.argsort(tvm_output.asnumpy()[b])
                # Report top-5 classification results
                print("\n{} prediction for sample {}".format(model, b))
                print("\t#1:", imagenette[top_categories[-1]])
                print("\t#2:", imagenette[top_categories[-2]])
                print("\t#3:", imagenette[top_categories[-3]])
                print("\t#4:", imagenette[top_categories[-4]])
                print("\t#5:", imagenette[top_categories[-5]])
                # This just checks that one of the 5 top categories
                # is one variety of cat; this is by no means an accurate
                # assessment of how quantization affects classification
                # accuracy but is meant to catch changes to the
                # quantization pass that would accuracy in the CI.
                # detected = False
                # for k in top_categories[-5:]:
                #     if imagenette[idx] in imagenette[k]:
                #         detected = True
                # assert detected

        for i in range(1):
            load_images_and_run(i)
    except Exception as e:
        not_working.append((i_net, id))
        print("FAILED for model id:", model_ids[i_net])
        print("With architecture:", arch_mapping[model_ids[i_net]])
        print("Error type:", type(e).__name__)
        print("Error message:", str(e))

print(f"\n\nThese ids graph pack didn't work - {not_working}")