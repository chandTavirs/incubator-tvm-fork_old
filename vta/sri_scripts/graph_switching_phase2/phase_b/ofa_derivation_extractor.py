"""
Phase B Step 2: OFA Weight Derivation Metadata Extractor
=========================================================

For each subnet architecture, extract EXACTLY how every conv layer's weights
are derived from the OFA base weight pool (slices + optional kernel transforms).

This is the ground truth that the Relay graph builder will use to construct
explicit slice/transform operations in the VTA graph.

Key insight from DynamicConv2DAll.get_active_weights():
  1. Slice channels:  base_weight[out_start:out_end, in_start:in_end, k_s:k_e, k_s:k_e]
  2. Apply transform: if kernel_size < max_kernel_size:
       for each (src_ks → target_ks):
           w = w.view(out*in, src_ks**2)
           w = F.linear(w, transform_matrix)   # key: transform_matrix stored per-layer
           w = w.view(out, in, target_ks, target_ks)
"""

from __future__ import absolute_import, print_function

import sys
import json
import copy
from typing import Dict, Any, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXTERNAL_REPO_ROOT = "/home/srchand/Desktop/research/OFA_Obfs"
if EXTERNAL_REPO_ROOT not in sys.path:
    sys.path.insert(0, EXTERNAL_REPO_ROOT)


# ---------------------------------------------------------------------------
# Helper: mirror DynamicConv2DAll.sub_filter_start_end
# ---------------------------------------------------------------------------
def _sub_filter_start_end(max_ks: int, target_ks: int) -> Tuple[int, int]:
    center = max_ks // 2
    dev = target_ks // 2
    start = center - dev
    end = center + dev + 1
    assert end - start == target_ks
    return start, end


# ---------------------------------------------------------------------------
# Helper: compute output spatial dimensions after convolution
# ---------------------------------------------------------------------------
def _conv2d_out_hw(h: int, w: int, k: int, stride: int, pad: int) -> Tuple[int, int]:
    """
    Compute output height and width after 2D convolution.
    Assumes dilation=1.
    
    Args:
        h: input height
        w: input width
        k: kernel size
        stride: stride
        pad: padding
    
    Returns:
        (output_h, output_w)
    """
    ho = (h + 2 * pad - k) // stride + 1
    wo = (w + 2 * pad - k) // stride + 1
    return ho, wo


# ---------------------------------------------------------------------------
# Single-layer derivation record
# ---------------------------------------------------------------------------
class LayerDerivation:
    """
    Describes how a single conv weight tensor is derived from the OFA pool.

    Fields
    ------
    layer_path          : dotted path in OFA model (e.g. 'first_layer.0')
    base_weight_key     : key in the base_weights.npz pool
    out_ch, in_ch       : active output/input channels for THIS sub-conv (scalar ints)
    out_start, out_end  : output channel slice range [out_start:out_end] within base weight
    in_start,  in_end   : input  channel slice range [in_start:in_end]  within base weight
    layer_total_out_ch  : TOTAL active output channels for the whole layer (sum of all
                          decomposed groups). For dt=0 equals out_ch. Used for the
                          "transform-once, slice N times" optimisation in the Relay builder:
                          the transform is applied to the full [layer_total_out_ch, in_ch]
                          slice once, then each sub-conv slices its portion from the result.
    layer_total_in_ch   : TOTAL active input channels for the whole layer (analogous).
    input_h, input_w    : spatial dimensions of the input activation tensor to this layer
                          (output H/W of the previous layer)
    output_h, output_w  : spatial dimensions of the output activation tensor from this layer
    max_kernel_size     : maximum kernel size of this DynamicConv2DAll
    active_kernel_size  : the kernel size actually used by this subnet
    transform_sequence  : list of (src_ks, target_ks) pairs, in application order
                          e.g. [(7,5)] means apply 7to5 matrix once
                          e.g. [(7,5),(5,3)] means 7→5 first, then 5→3
    transform_keys      : list of keys in transform_matrices.npz, parallel to transform_sequence
                          e.g. ['resnet_blocks_0_residuals_0_convs_0_7to5_matrix']
    decompose_type      : 0-4 (controls how the derived weight is split into sub-convolutions)
    is_shortcut         : True if this is a residual shortcut (always kernel 1×1, no transform)
    """

    def __init__(self):
        self.layer_path: str = ""
        self.base_weight_key: str = ""
        self.out_ch: int = 0
        self.in_ch: int = 0
        self.out_start: int = 0
        self.out_end: int = 0
        self.in_start: int = 0
        self.in_end: int = 0
        self.layer_total_out_ch: int = 0   # full layer output channels
        self.layer_total_in_ch: int = 0    # full layer input channels
        self.input_h: int = 0              # spatial height of input activation
        self.input_w: int = 0              # spatial width  of input activation
        self.output_h: int = 0             # spatial height of output activation
        self.output_w: int = 0             # spatial width  of output activation
        self.max_kernel_size: int = 0
        self.active_kernel_size: int = 0
        self.transform_sequence: List[Tuple[int, int]] = []
        self.transform_keys: List[str] = []
        self.decompose_type: int = 0
        self.is_shortcut: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_path": self.layer_path,
            "base_weight_key": self.base_weight_key,
            "out_ch": self.out_ch,
            "in_ch": self.in_ch,
            "out_start": self.out_start,
            "out_end": self.out_end,
            "in_start": self.in_start,
            "in_end": self.in_end,
            "layer_total_out_ch": self.layer_total_out_ch,
            "layer_total_in_ch": self.layer_total_in_ch,
            "input_h": self.input_h,
            "input_w": self.input_w,
            "output_h": self.output_h,
            "output_w": self.output_w,
            "max_kernel_size": self.max_kernel_size,
            "active_kernel_size": self.active_kernel_size,
            "transform_sequence": self.transform_sequence,
            "transform_keys": self.transform_keys,
            "decompose_type": self.decompose_type,
            "is_shortcut": self.is_shortcut,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LayerDerivation":
        obj = cls()
        for k, v in d.items():
            setattr(obj, k, v)
        return obj

    def __repr__(self):
        return (
            f"LayerDerivation(path={self.layer_path}, "
            f"out={self.out_start}:{self.out_end}, "
            f"in={self.in_start}:{self.in_end}, "
            f"spatial=in[{self.input_h}x{self.input_w}]→out[{self.output_h}x{self.output_w}], "
            f"ks={self.active_kernel_size}/{self.max_kernel_size}, "
            f"transforms={self.transform_sequence}, "
            f"decomp={self.decompose_type})"
        )


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------
class OFADerivationExtractor:
    """
    Extracts the weight derivation metadata for a given subnet architecture
    by introspecting the OFA model's DynamicConv2DAll layers.

    No actual weight values are stored – only the recipe to derive them from
    the base weight pool.
    """

    def __init__(self, ofa_net, verbose: bool = False):
        """
        Parameters
        ----------
        ofa_net : OFADynamicResnetAllMod (already loaded from checkpoint)
        verbose  : print extraction details
        """
        self.ofa_net = ofa_net
        self.verbose = verbose

        # Pre-build key mapping: module_path → pool key
        self._path_to_base_key: Dict[str, str] = {}
        self._path_to_transform_keys: Dict[str, Dict[str, str]] = {}
        self._build_key_maps()

    # ------------------------------------------------------------------
    def _build_key_maps(self):
        """Map OFA module dotted paths to pool keys (replacing '.' with '_')."""
        for name, module in self.ofa_net.named_modules():
            # Import locally to avoid circular import issues at top level
            try:
                from ofa_base_models.ofa_ops.dynamic_conv_all import DynamicConv2DAll
            except ImportError:
                raise ImportError("Cannot import DynamicConv2DAll – check EXTERNAL_REPO_ROOT")

            if isinstance(module, DynamicConv2DAll):
                base_key = name.replace(".", "_") + "_base_conv_weight"
                self._path_to_base_key[name] = base_key

                # Also map transform matrix keys
                ks_set = sorted(set(module.kernel_size_list))
                tm_keys: Dict[str, str] = {}
                for i in range(len(ks_set) - 1):
                    src_ks = ks_set[i + 1]
                    tgt_ks = ks_set[i]
                    matrix_attr = f"{src_ks}to{tgt_ks}_matrix"
                    pool_key = name.replace(".", "_") + f"_{matrix_attr}"
                    tm_keys[matrix_attr] = pool_key
                self._path_to_transform_keys[name] = tm_keys

        if self.verbose:
            print(f"[Extractor] Mapped {len(self._path_to_base_key)} DynamicConv2DAll layers")

    # ------------------------------------------------------------------
    def _derive_transform_sequence(
        self,
        module,          # DynamicConv2DAll instance
        module_path: str,
        active_kernel_size: int,
    ) -> Tuple[List[Tuple[int, int]], List[str]]:
        """
        Mirror the loop in DynamicConv2DAll.get_active_weights() to find
        the sequence of (src_ks → tgt_ks) transforms needed.

        Returns (transform_sequence, transform_key_list).
        """
        max_ks = module.max_kernel_size
        _ks_set = sorted(set(module.kernel_size_list))

        if active_kernel_size == max_ks:
            return [], []

        seq_pairs: List[Tuple[int, int]] = []
        seq_keys: List[str] = []
        tm_keys = self._path_to_transform_keys.get(module_path, {})

        for i in range(len(_ks_set) - 1, 0, -1):
            src_ks = _ks_set[i]
            if src_ks <= active_kernel_size:
                break
            tgt_ks = _ks_set[i - 1]
            pair = (src_ks, tgt_ks)
            seq_pairs.append(pair)
            matrix_attr = f"{src_ks}to{tgt_ks}_matrix"
            pool_key = tm_keys.get(matrix_attr, f"{module_path.replace('.','_')}_{matrix_attr}")
            seq_keys.append(pool_key)

        return seq_pairs, seq_keys

    # ------------------------------------------------------------------
    def _extract_single_layer(
        self,
        module_path: str,
        module,               # DynamicConv2DAll
        active_in_ch: int,
        active_out_ch: int,
        active_kernel_size: int,
        decompose_type: int,
        in_start: int = 0,
        out_start: int = 0,
        is_shortcut: bool = False,
        layer_total_out_ch: int = 0,   # full layer out channels (0 = same as active_out_ch)
        layer_total_in_ch: int = 0,    # full layer in  channels (0 = same as active_in_ch)
        input_h: int = 0,              # spatial height of input activation
        input_w: int = 0,              # spatial width  of input activation
        output_h: int = 0,             # spatial height of output activation
        output_w: int = 0,             # spatial width  of output activation
    ) -> LayerDerivation:
        """Build a LayerDerivation for one (possibly decomposed) sub-conv."""
        d = LayerDerivation()
        d.layer_path = module_path
        d.base_weight_key = self._path_to_base_key[module_path]
        d.out_ch = active_out_ch
        d.in_ch = active_in_ch
        d.out_start = out_start
        d.out_end = out_start + active_out_ch
        d.in_start = in_start
        d.in_end = in_start + active_in_ch
        d.layer_total_out_ch = layer_total_out_ch if layer_total_out_ch > 0 else active_out_ch
        d.layer_total_in_ch  = layer_total_in_ch  if layer_total_in_ch  > 0 else active_in_ch
        d.input_h = input_h
        d.input_w = input_w
        d.output_h = output_h
        d.output_w = output_w
        d.max_kernel_size = module.max_kernel_size
        d.active_kernel_size = active_kernel_size
        d.transform_sequence, d.transform_keys = self._derive_transform_sequence(
            module, module_path, active_kernel_size
        )
        d.decompose_type = decompose_type
        d.is_shortcut = is_shortcut
        return d

    # ------------------------------------------------------------------
    def extract_subnet_derivations(
        self, arch: Dict[str, Any], input_shape=[1, 3, 224, 224]
    ) -> List[LayerDerivation]:
        """
        Main entry point.

        Set the OFA net to this subnet's architecture and walk all layers
        in forward-pass order, recording the derivation metadata for each conv.

        Returns an ordered list of LayerDerivation objects (one per sub-conv
        in the VTA-compiled graph, matching the compilation order used in
        execute_candidate_set_refactored.py).
        """
        from training_ops.utils import make_divisible, MyNetwork

        self.ofa_net.set_active_subnet(arch)
        self.ofa_net.eval()

        derivations: List[LayerDerivation] = []
        width_mult_list = sorted(self.ofa_net.width_mult_list)

        # Track spatial dimensions through the network
        # ImageNet input: 224x224
        current_h, current_w = input_shape[2], input_shape[3]

        # ---- first conv (always 7×7, stride=2, no kernel transform, no decomposition) ----
        out_ch_setting = arch["out_channel_setting_list"][0]
        out_ch = make_divisible(
            64 * width_mult_list[out_ch_setting], MyNetwork.CHANNEL_DIVISIBLE
        )
        in_ch = 3
        first_module_path = "first_layer.0"
        first_module = self.ofa_net.first_layer[0]
        
        # First conv: 7x7 kernel, stride=2, padding=3 (to keep 224->112)
        # For 7x7 conv with stride=2, padding=3:
        # output = (224 + 2*3 - 7) // 2 + 1 = (224 + 6 - 7) // 2 + 1 = 223 // 2 + 1 = 111 + 1 = 112
        first_k = 7
        first_stride = 2
        first_pad = 3  # Standard padding for 7x7 conv to reduce 224->112
        output_h, output_w = _conv2d_out_hw(current_h, current_w, first_k, first_stride, first_pad)
        
        d = self._extract_single_layer(
            first_module_path, first_module,
            active_in_ch=in_ch, active_out_ch=out_ch,
            active_kernel_size=7,   # first conv is always 7×7
            decompose_type=0,
            input_h=current_h, input_w=current_w,
            output_h=output_h, output_w=output_w,
        )
        derivations.append(d)
        if self.verbose:
            print(f"  [first_conv] {d}")

        # Update spatial dimensions for next stage
        current_h, current_w = output_h, output_w

        # After maxpool layer, current_h and current_w get divided by 2
        current_h, current_w = current_h//2, current_w//2

        # ---- residual blocks ----
        current_channels = out_ch
        for block_idx in range(len(arch["residual_depth_list"])):
            out_ch_setting = arch["out_channel_setting_list"][block_idx + 1]
            stage_base = [64, 128, 256, 512][block_idx]
            out_channels = make_divisible(
                stage_base * width_mult_list[out_ch_setting], MyNetwork.CHANNEL_DIVISIBLE
            )

            # First residual in this block applies stride, others are stride=1
            stride = 2 if block_idx > 0 else 1

            residual_depths   = arch["residual_depth_list"][block_idx]
            decomp_types      = arch["decomp_type_list"][block_idx]
            kernel_sizes      = arch["kernel_size_list"][block_idx]
            downsample_decomp = (
                arch["downsample_decomp_type_list"][block_idx - 1] if block_idx > 0 else 0
            )

            ofa_block = self.ofa_net.resnet_blocks[block_idx]

            for residual_idx, depth in enumerate(residual_depths):
                if depth == 0:
                    continue

                ofa_residual = ofa_block.residuals[residual_idx]
                residual_in_channels = current_channels if residual_idx == 0 else out_channels
                
                # Stride applies only to first residual in each block
                residual_stride = stride if residual_idx == 0 else 1
                residual_input_h, residual_input_w = current_h, current_w
                
                # For residual blocks: kernel sizes are 3, 5, 7 with stride and padding
                # In OFA ResNet: padding = kernel_size // 2 (to keep spatial dims same for stride=1)
                # For stride=2, spatial dims are halved
                # We'll compute output dims based on the first conv in this residual block
                
                for conv_idx in range(depth):
                    ks = kernel_sizes[residual_idx][conv_idx]
                    dt = decomp_types[residual_idx][conv_idx]
                    conv_in_ch = residual_in_channels if conv_idx == 0 else out_channels
                    conv_out_ch = out_channels
                    
                    # Standard padding: kernel_size // 2
                    conv_pad = ks // 2
                    # Stride applies only to first conv of first residual
                    conv_stride = residual_stride if conv_idx == 0 else 1
                    
                    # Compute input/output spatial dims for this conv
                    conv_input_h = residual_input_h if conv_idx == 0 else residual_output_h
                    conv_input_w = residual_input_w if conv_idx == 0 else residual_output_w
                    conv_output_h, conv_output_w = _conv2d_out_hw(
                        conv_input_h, conv_input_w, ks, conv_stride, conv_pad
                    )
                    
                    # Store residual block's output dims (computed from first conv)
                    if conv_idx == 0:
                        residual_output_h, residual_output_w = conv_output_h, conv_output_w

                    module_path = (
                        f"resnet_blocks.{block_idx}.residuals.{residual_idx}.convs.{conv_idx}"
                    )
                    module = ofa_residual.convs[conv_idx]

                    # Build derivation(s) for this conv (may be split by decompose_type)
                    sub_derivations = self._expand_decomposed_conv(
                        module_path=module_path,
                        module=module,
                        in_ch=conv_in_ch,
                        out_ch=conv_out_ch,
                        kernel_size=ks,
                        decompose_type=dt,
                        input_h=conv_input_h,
                        input_w=conv_input_w,
                        output_h=conv_output_h,
                        output_w=conv_output_w,
                    )
                    derivations.extend(sub_derivations)

                    if self.verbose:
                        for sd in sub_derivations:
                            print(f"  [blk{block_idx}.res{residual_idx}.conv{conv_idx}] {sd}")

                # ---- shortcut (downsample) ----
                need_downsample = (
                    (stride != 1 and residual_idx == 0) or
                    (residual_in_channels != out_channels)
                )
                if need_downsample and residual_idx == 0:
                    sc_decomp = downsample_decomp if block_idx > 0 else 0
                    sc_module_path = (
                        f"resnet_blocks.{block_idx}.residuals.{residual_idx}.downsample"
                    )
                    sc_module = ofa_residual.downsample
                    
                    # Shortcut is always 1x1 with same padding behavior
                    sc_k = 1
                    sc_pad = 0
                    sc_stride = stride
                    sc_output_h, sc_output_w = _conv2d_out_hw(
                        residual_input_h, residual_input_w, sc_k, sc_stride, sc_pad
                    )

                    sub_derivations = self._expand_decomposed_conv(
                        module_path=sc_module_path,
                        module=sc_module,
                        in_ch=residual_in_channels,
                        out_ch=out_channels,
                        kernel_size=1,   # shortcut is always 1×1
                        decompose_type=sc_decomp,
                        is_shortcut=True,
                        input_h=residual_input_h,
                        input_w=residual_input_w,
                        output_h=sc_output_h,
                        output_w=sc_output_w,
                    )
                    derivations.extend(sub_derivations)

                    if self.verbose:
                        for sd in sub_derivations:
                            print(f"  [blk{block_idx}.sc] {sd}")

                # Update spatial dimensions after this residual block
                current_h, current_w = residual_output_h, residual_output_w

            current_channels = out_channels

        # ---- final linear ----
        # (not a conv – handled separately; we include a sentinel for completeness)

        return derivations

    # ------------------------------------------------------------------
    def _expand_decomposed_conv(
        self,
        module_path: str,
        module,           # DynamicConv2DAll
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        decompose_type: int,
        is_shortcut: bool = False,
        input_h: int = 0,
        input_w: int = 0,
        output_h: int = 0,
        output_w: int = 0,
    ) -> List[LayerDerivation]:
        """
        For decompose_type > 0 the single base conv is split into sub-convolutions.
        We return one LayerDerivation per sub-conv, each with the correct
        channel slice offsets AND the layer_total_out_ch / layer_total_in_ch fields
        populated so the Relay builder can apply the kernel transform ONCE on the
        full channel span and then slice each group (proven mathematically equivalent).
        
        All sub-convolutions share the same spatial dimensions (input_h/w, output_h/w).
        """
        derivations: List[LayerDerivation] = []

        if decompose_type == 0:
            d = self._extract_single_layer(
                module_path, module, in_ch, out_ch, kernel_size, 0,
                in_start=0, out_start=0, is_shortcut=is_shortcut,
                layer_total_out_ch=out_ch, layer_total_in_ch=in_ch,
                input_h=input_h, input_w=input_w,
                output_h=output_h, output_w=output_w,
            )
            derivations.append(d)

        elif decompose_type == 1:
            # Split output channels in 2; same in_ch for all groups
            mid_out = out_ch // 2
            for os, oe in [(0, mid_out), (mid_out, out_ch)]:
                d = self._extract_single_layer(
                    module_path, module, in_ch, oe - os, kernel_size, 1,
                    in_start=0, out_start=os, is_shortcut=is_shortcut,
                    layer_total_out_ch=out_ch, layer_total_in_ch=in_ch,
                    input_h=input_h, input_w=input_w,
                    output_h=output_h, output_w=output_w,
                )
                derivations.append(d)

        elif decompose_type == 2:
            # Split output channels in 4; same in_ch for all groups
            q = out_ch // 4
            for os, oe in [(0, q), (q, 2*q), (2*q, 3*q), (3*q, out_ch)]:
                d = self._extract_single_layer(
                    module_path, module, in_ch, oe - os, kernel_size, 2,
                    in_start=0, out_start=os, is_shortcut=is_shortcut,
                    layer_total_out_ch=out_ch, layer_total_in_ch=in_ch,
                    input_h=input_h, input_w=input_w,
                    output_h=output_h, output_w=output_w,
                )
                derivations.append(d)

        elif decompose_type == 3:
            # Split input channels in 2; same out_ch for all groups
            mid_in = in_ch // 2
            for is_, ie in [(0, mid_in), (mid_in, in_ch)]:
                d = self._extract_single_layer(
                    module_path, module, ie - is_, out_ch, kernel_size, 3,
                    in_start=is_, out_start=0, is_shortcut=is_shortcut,
                    layer_total_out_ch=out_ch, layer_total_in_ch=in_ch,
                    input_h=input_h, input_w=input_w,
                    output_h=output_h, output_w=output_w,
                )
                derivations.append(d)

        elif decompose_type == 4:
            # Split input channels in 4; same out_ch for all groups
            q = in_ch // 4
            for is_, ie in [(0, q), (q, 2*q), (2*q, 3*q), (3*q, in_ch)]:
                d = self._extract_single_layer(
                    module_path, module, ie - is_, out_ch, kernel_size, 4,
                    in_start=is_, out_start=0, is_shortcut=is_shortcut,
                    layer_total_out_ch=out_ch, layer_total_in_ch=in_ch,
                    input_h=input_h, input_w=input_w,
                    output_h=output_h, output_w=output_w,
                )
                derivations.append(d)

        else:
            raise ValueError(f"Unknown decompose_type: {decompose_type}")

        return derivations


# ---------------------------------------------------------------------------
# Numerical validator – derives the weight using NumPy, compares to OFA
# ---------------------------------------------------------------------------
def derive_weight_numpy(
    base_weights: Dict[str, np.ndarray],
    transform_matrices: Dict[str, np.ndarray],
    deriv: LayerDerivation,
) -> np.ndarray:
    """
    Reproduce DynamicConv2DAll.get_active_weights() in pure NumPy.

    OFA loop (exactly):
      start_filter = base[out_s:out_e, in_s:in_e, :, :]   # full spatial, channel-sliced
      for (src_ks → target_ks) in transform_sequence:
          crop_s, crop_e = sub_filter_start_end(src_ks, target_ks)
          _input = start_filter[:, :, crop_s:crop_e, crop_s:crop_e]  # crop of CURRENT filter
          _flat  = _input.reshape(-1, target_ks * target_ks)          # [out*in, tgt_ks^2]
          _out   = _flat @ T.T                                         # T is [tgt_ks^2, tgt_ks^2]
          start_filter = _out.reshape(out_c, in_c, target_ks, target_ks)

    Key points:
      - crop is taken from start_filter (which shrinks each iteration), NOT the original base
      - sub_filter_start_end(src_ks, target_ks) uses src_ks = intended source size = current
        spatial size of start_filter at that step
      - reshape uses target_ks^2 (the output spatial size), not src_ks^2
      - transform matrices are square: [tgt_ks^2, tgt_ks^2]
    """
    base   = base_weights[deriv.base_weight_key]
    max_ks = deriv.max_kernel_size
    tgt_ks = deriv.active_kernel_size
    out_c  = deriv.out_end - deriv.out_start
    in_c   = deriv.in_end  - deriv.in_start

    if not deriv.transform_sequence:
        # No transform: channel slice + optional centre-crop
        w = base[deriv.out_start:deriv.out_end,
                 deriv.in_start:deriv.in_end, :, :]
        if tgt_ks < max_ks:
            ks, ke = _sub_filter_start_end(max_ks, tgt_ks)
            w = w[:, :, ks:ke, ks:ke]
        return w.copy()

    # Has transform(s): start from full-spatial channel slice
    start_filter = base[deriv.out_start:deriv.out_end,
                        deriv.in_start:deriv.in_end, :, :].copy()

    for (src_ks, target_ks), tm_key in zip(deriv.transform_sequence, deriv.transform_keys):
        # Centre-crop start_filter (current size = src_ks × src_ks) → target_ks × target_ks
        crop_s, crop_e = _sub_filter_start_end(src_ks, target_ks)
        _input = start_filter[:, :, crop_s:crop_e, crop_s:crop_e].copy()
        # _input shape: [out_c, in_c, target_ks, target_ks]

        # Reshape to [out_c*in_c, target_ks^2]
        _flat = _input.reshape(-1, target_ks * target_ks)

        # F.linear(a, T) = a @ T.T,  T shape [target_ks^2, target_ks^2]
        tm = transform_matrices[tm_key]
        _out = _flat @ tm.T

        start_filter = _out.reshape(out_c, in_c, target_ks, target_ks)

    return start_filter


# ---------------------------------------------------------------------------
# Quick numerical sanity-check
# ---------------------------------------------------------------------------
def validate_derivation_against_ofa(
    ofa_net,
    arch: Dict[str, Any],
    base_weights: Dict[str, np.ndarray],
    transform_matrices: Dict[str, np.ndarray],
    derivations: List[LayerDerivation],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run OFA forward pass on a dummy input, hook weight outputs from each
    DynamicConv2DAll, then compare against the NumPy derivation.

    Returns a result dict with per-layer match/mismatch info.
    """
    ofa_net.set_active_subnet(arch)
    ofa_net.eval()

    # --- capture active weights via hooks ---
    captured: OrderedDict = OrderedDict()

    def make_hook(path):
        def hook_fn(module, inp, outp):
            # get_active_weights is called internally; we capture the base_conv.weight slice
            # We'll re-derive it ourselves from the pool and compare
            pass
        return hook_fn

    # Actually, we re-derive each layer's weight from the OFA model directly
    # to compare against our numpy derivation
    from training_ops.utils import make_divisible, MyNetwork
    width_mult_list = sorted(ofa_net.width_mult_list)

    results = {
        "total": 0,
        "matched": 0,
        "mismatched": 0,
        "max_abs_diff": 0.0,
        "details": [],
    }

    deriv_iter = iter(derivations)

    def compare_next_derivation(module_path, module, in_ch, out_ch, ks, dt, in_s=0, out_s=0):
        try:
            d = next(deriv_iter)
        except StopIteration:
            return

        # OFA reference weight — mirror how forward() calls get_active_weights()
        # forward() sets active_kernel_size, active_in_channel, active_out_channel then calls:
        #   top_level = base_conv.weight[:active_out, :active_in, :, :]
        #   kernel_start, kernel_end = sub_filter_start_end(max_ks, active_ks)
        #   get_active_weights([(in_s, in_e)], [(out_s, out_e)], kernel_start, kernel_end, top_level)
        total_out = out_s + out_ch
        total_in  = in_s  + in_ch
        module.active_kernel_size = ks
        module.active_in_channel  = total_in
        module.active_out_channel = total_out

        max_ks = module.max_kernel_size
        if ks < max_ks:
            kernel_s, kernel_e = _sub_filter_start_end(max_ks, ks)
        else:
            kernel_s, kernel_e = 0, max_ks

        top_level = module.base_conv.weight[:total_out, :total_in, :, :]

        with torch.no_grad():
            ofa_w = module.get_active_weights(
                active_in_channels=[(in_s, total_in)],
                active_out_channels=[(out_s, total_out)],
                kernel_start=kernel_s,
                kernel_end=kernel_e,
                top_level_active_filters=top_level,
            ).cpu().numpy()

        # Our numpy derivation
        our_w = derive_weight_numpy(base_weights, transform_matrices, d)

        diff = np.abs(ofa_w - our_w)
        max_diff = float(diff.max())
        matched = max_diff < 1e-5

        results["total"] += 1
        results["max_abs_diff"] = max(results["max_abs_diff"], max_diff)
        if matched:
            results["matched"] += 1
        else:
            results["mismatched"] += 1

        detail = {
            "layer": module_path,
            "shape": list(ofa_w.shape),
            "max_abs_diff": max_diff,
            "matched": matched,
        }
        results["details"].append(detail)

        if verbose:
            status = "✓" if matched else "✗"
            print(f"  {status} {module_path:<60s}  shape={list(ofa_w.shape)}  maxdiff={max_diff:.2e}")

    # Walk layers in the same order as extract_subnet_derivations
    # (mirrored from that function)
    out_ch_setting = arch["out_channel_setting_list"][0]
    current_channels = make_divisible(
        64 * width_mult_list[out_ch_setting], MyNetwork.CHANNEL_DIVISIBLE
    )
    compare_next_derivation(
        "first_layer.0", ofa_net.first_layer[0],
        in_ch=3, out_ch=current_channels, ks=7, dt=0,
    )

    for block_idx in range(len(arch["residual_depth_list"])):
        out_ch_setting = arch["out_channel_setting_list"][block_idx + 1]
        stage_base = [64, 128, 256, 512][block_idx]
        out_channels = make_divisible(
            stage_base * width_mult_list[out_ch_setting], MyNetwork.CHANNEL_DIVISIBLE
        )
        stride = 2 if block_idx > 0 else 1
        residual_depths   = arch["residual_depth_list"][block_idx]
        decomp_types      = arch["decomp_type_list"][block_idx]
        kernel_sizes      = arch["kernel_size_list"][block_idx]
        downsample_decomp = arch["downsample_decomp_type_list"][block_idx - 1] if block_idx > 0 else 0
        ofa_block = ofa_net.resnet_blocks[block_idx]

        for residual_idx, depth in enumerate(residual_depths):
            if depth == 0:
                continue
            ofa_residual = ofa_block.residuals[residual_idx]
            residual_in_channels = current_channels if residual_idx == 0 else out_channels

            for conv_idx in range(depth):
                ks  = kernel_sizes[residual_idx][conv_idx]
                dt  = decomp_types[residual_idx][conv_idx]
                cin = residual_in_channels if conv_idx == 0 else out_channels
                cout = out_channels
                mp  = f"resnet_blocks.{block_idx}.residuals.{residual_idx}.convs.{conv_idx}"
                mod = ofa_residual.convs[conv_idx]
                # Handle sub-convolutions for decomposed
                if dt == 0:
                    compare_next_derivation(mp, mod, cin, cout, ks, dt)
                elif dt == 1:
                    mid = cout // 2
                    compare_next_derivation(mp, mod, cin, mid, ks, dt, out_s=0)
                    compare_next_derivation(mp, mod, cin, cout-mid, ks, dt, out_s=mid)
                elif dt == 2:
                    q = cout // 4
                    for os in [0, q, 2*q, 3*q]:
                        oe = min(os + q, cout)
                        compare_next_derivation(mp, mod, cin, oe-os, ks, dt, out_s=os)
                elif dt == 3:
                    mid = cin // 2
                    compare_next_derivation(mp, mod, mid, cout, ks, dt, in_s=0)
                    compare_next_derivation(mp, mod, cin-mid, cout, ks, dt, in_s=mid)
                elif dt == 4:
                    q = cin // 4
                    for is_ in [0, q, 2*q, 3*q]:
                        ie = min(is_ + q, cin)
                        compare_next_derivation(mp, mod, ie-is_, cout, ks, dt, in_s=is_)

            need_downsample = (stride != 1 and residual_idx == 0) or (residual_in_channels != out_channels)
            if need_downsample and residual_idx == 0:
                sc_dt = downsample_decomp if block_idx > 0 else 0
                sc_mp = f"resnet_blocks.{block_idx}.residuals.{residual_idx}.downsample"
                sc_mod = ofa_residual.downsample
                if sc_dt == 0:
                    compare_next_derivation(sc_mp, sc_mod, residual_in_channels, out_channels, 1, sc_dt)
                elif sc_dt == 1:
                    mid = out_channels // 2
                    compare_next_derivation(sc_mp, sc_mod, residual_in_channels, mid, 1, sc_dt, out_s=0)
                    compare_next_derivation(sc_mp, sc_mod, residual_in_channels, out_channels-mid, 1, sc_dt, out_s=mid)
                elif sc_dt == 2:
                    q = out_channels // 4
                    for os in [0, q, 2*q, 3*q]:
                        oe = min(os + q, out_channels)
                        compare_next_derivation(sc_mp, sc_mod, residual_in_channels, oe-os, 1, sc_dt, out_s=os)
                elif sc_dt == 3:
                    mid = residual_in_channels // 2
                    compare_next_derivation(sc_mp, sc_mod, mid, out_channels, 1, sc_dt, in_s=0)
                    compare_next_derivation(sc_mp, sc_mod, residual_in_channels-mid, out_channels, 1, sc_dt, in_s=mid)
                elif sc_dt == 4:
                    q = residual_in_channels // 4
                    for is_ in [0, q, 2*q, 3*q]:
                        ie = min(is_ + q, residual_in_channels)
                        compare_next_derivation(sc_mp, sc_mod, ie-is_, out_channels, 1, sc_dt, in_s=is_)

        current_channels = out_channels

    return results


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def save_derivations(derivations: List[LayerDerivation], path: str):
    import json
    data = [d.to_dict() for d in derivations]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_derivations(path: str) -> List[LayerDerivation]:
    import json
    with open(path, "r") as f:
        data = json.load(f)
    return [LayerDerivation.from_dict(d) for d in data]


