"""
Phase B Step 3: OFA-Aware Relay Graph Builder
==============================================

Builds a TVM Relay computation graph for a subnet where:
  - Every conv weight is expressed as explicit Relay ops (slice + optional
    dense/reshape transform) applied to OFA base weight *variables*.
  - The OFA weight pool variables are shared across all subnets.
  - FoldConstant is intentionally not applied to weight derivation ops,
    so they become runtime operations that read from the shared pool.

Architecture of a single derived-weight conv (simplified):

  ofa_base   [max_out, max_in, max_k, max_k]   ← Relay var (shared)
      |
  strided_slice → [out_ch, in_ch, max_k, max_k]   (channel crop)
      |
  [optional] kernel_transform:
      reshape  → [out_ch*in_ch, max_k**2]
      dense    → [out_ch*in_ch, tgt_k**2]    using transform matrix var
      reshape  → [out_ch, in_ch, tgt_k, tgt_k]
      |
  conv2d(input, derived_weight, ...)

Then quantisation + graph_pack are applied on top, exactly as today.

Usage
-----
builder = OFARelayGraphBuilder(ofa_pool_dir, arch, env, config)
mod, params = builder.build()
# mod["main"] has OFA pool variables instead of materialised constants
# params contains the OFA pool numpy arrays (uploaded once to the device)
"""

from __future__ import absolute_import, print_function

import os
import sys
import copy
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

import tvm
from tvm import relay
from tvm.relay import transform as relay_transform

EXTERNAL_REPO_ROOT = "/home/srchand/Desktop/research/OFA_Obfs"
if EXTERNAL_REPO_ROOT not in sys.path:
    sys.path.insert(0, EXTERNAL_REPO_ROOT)

# Local phase-b modules
_PHASE_B_DIR = os.path.dirname(__file__)
if _PHASE_B_DIR not in sys.path:
    sys.path.insert(0, _PHASE_B_DIR)

from ofa_derivation_extractor import (
    LayerDerivation,
    OFADerivationExtractor,
    derive_weight_numpy,
)


# ===========================================================================
# Helpers
# ===========================================================================
def _sub_filter_start_end(max_ks: int, target_ks: int) -> Tuple[int, int]:
    center = max_ks // 2
    dev = target_ks // 2
    start = center - dev
    end = center + dev + 1
    return start, end


def _apply_transform_sequence(
    w: relay.Expr,                          # [out_ch, in_ch, current_ks, current_ks]
    out_ch: int,
    in_ch: int,
    max_ks: int,
    transform_sequence: List[Tuple[int, int]],
    transform_keys: List[str],
    transform_vars: Dict[str, relay.Var],
) -> relay.Expr:
    """
    Apply the OFA kernel-size transform sequence to a weight expression.

    Mathematically proven equivalent for decomposed convolutions:
      transform(concat([slice_0, slice_1, ...])) == concat([transform(slice_0), ...])
    because the transform acts row-independently on [out*in, ks^2].

    Therefore we ALWAYS transform the FULL out_ch slice once, then each
    decomposed sub-conv just slices from the result — one dense op per layer,
    not one per sub-conv.
    """
    current_ks = max_ks
    w_curr = w

    for (src_ks, tgt_ks), tm_key in zip(transform_sequence, transform_keys):
        # Center-crop current w_curr from current_ks → tgt_ks spatial
        crop_s, crop_e = _sub_filter_start_end(src_ks, tgt_ks)
        w_curr = relay.strided_slice(
            w_curr,
            begin=[0, 0, crop_s, crop_s],
            end=[out_ch, in_ch, crop_e, crop_e],
            strides=[1, 1, 1, 1],
        )
        # reshape [out_ch, in_ch, tgt_ks, tgt_ks] → [out_ch*in_ch, tgt_ks^2]
        tgt_flat = tgt_ks * tgt_ks
        w_flat = relay.reshape(w_curr, newshape=[-1, tgt_flat])
        # dense: [out*in, tgt_ks^2] x [tgt_ks^2, tgt_ks^2]^T → [out*in, tgt_ks^2]
        # (transform matrix is square: [tgt_ks^2, tgt_ks^2])
        tm_var = transform_vars[tm_key]
        w_flat = relay.nn.dense(w_flat, tm_var)
        # reshape back to [out_ch, in_ch, tgt_ks, tgt_ks]
        w_curr = relay.reshape(w_flat, newshape=[out_ch, in_ch, tgt_ks, tgt_ks])
        current_ks = tgt_ks

    return w_curr


def _make_full_layer_weight_expr(
    base_var: relay.Var,
    transform_vars: Dict[str, relay.Var],
    deriv: LayerDerivation,
    total_out_ch: int,   # full active_out_ch for this layer (sum of all decomposed groups)
) -> relay.Expr:
    """
    Compute the FULL transformed weight for one conv layer as a single Relay expr:
      shape → [total_out_ch, in_ch, active_ks, active_ks]

    For decomposed convs (dt > 0), we transform the full weight ONCE and let
    each sub-conv slice its group from the result.  This is valid because:
      T applied to [out_0..out_N, in, ks, ks] equals
      concat of T applied to each [out_i..out_j, in, ks, ks]
    (the transform operates row-independently on the [out*in, ks^2] view).

    For non-decomposed convs (dt == 0), total_out_ch == deriv.out_ch and
    this just returns the single sub-conv weight directly.
    """
    max_ks = deriv.max_kernel_size
    tgt_ks = deriv.active_kernel_size
    in_ch  = deriv.in_ch

    # Step 1: channel slice — always start from [in_start:in_end] on input dim
    # and [out_start .. out_start+total_out_ch] on output dim.
    # (For dt=0, out_start=0 and total_out_ch=out_ch.)
    w = relay.strided_slice(
        base_var,
        begin=[deriv.out_start, deriv.in_start, 0, 0],
        end=[deriv.out_start + total_out_ch, deriv.in_end, max_ks, max_ks],
        strides=[1, 1, 1, 1],
    )

    if not deriv.transform_sequence:
        # No transform: just a spatial center-crop if needed
        if tgt_ks < max_ks:
            ks, ke = _sub_filter_start_end(max_ks, tgt_ks)
            w = relay.strided_slice(
                w,
                begin=[0, 0, ks, ks],
                end=[total_out_ch, in_ch, ke, ke],
                strides=[1, 1, 1, 1],
            )
        return w

    # Step 2: transform sequence (once, on full total_out_ch)
    return _apply_transform_sequence(
        w, total_out_ch, in_ch, max_ks,
        deriv.transform_sequence, deriv.transform_keys, transform_vars,
    )


def _make_derived_weight_expr(
    base_var: relay.Var,
    transform_vars: Dict[str, relay.Var],
    deriv: LayerDerivation,
) -> relay.Expr:
    """
    Single-derivation weight expression (for dt==0 or when called directly).
    Uses _make_full_layer_weight_expr with total_out_ch == deriv.out_ch.
    """
    return _make_full_layer_weight_expr(
        base_var, transform_vars, deriv, total_out_ch=deriv.out_ch
    )


# ===========================================================================
# OFASubnetModel: a PyTorch module that exposes OFA pool vars
# ===========================================================================
class OFASubnetRelayBuilder:
    """
    Builds a Relay IRModule for a single subnet such that:
      - Conv weights are derived from OFA pool Relay variables (not constants)
      - BN + bias weights are separate Relay variables (per-subnet, small)
      - The graph is otherwise identical to current VTA compilation

    Strategy
    --------
    1. Compile the subnet via the existing pipeline (StaticResNetFromArch) to get
       the quantised, graph-packed Relay IR with constant weights.
    2. Walk the resulting graph and identify which Relay constants correspond to
       which OFA pool weight.
    3. Replace those constants with derived expressions built from pool variables.

    This is simpler and safer than building the Relay graph from scratch.
    However, replacing constants inside quantised graphs is tricky.

    Alternative strategy used here (cleaner):
    -----------------------------------------
    Build an OFASubnetPyTorchModel that is a thin PyTorch wrapper that:
      - Stores OFA pool weights as NAMED parameters using canonical pool keys
      - Builds the subnet computation using those exact parameters (no copying)
      - When traced by TorchScript, produces a graph where pool keys are shared

    Then compile that model normally. Relay's from_pytorch will produce the
    correct graph with shared parameter names.

    NOTE: We can't easily share parameters across different traced models since
    torch.jit.trace flattens each model independently. The simpler and most
    robust approach for the POC:

      1. Compile subnet normally to get (graph_json, lib, materialised_params).
      2. Build the OFA-param mapping: for each `pN` in params, find which OFA
         pool entry it came from (by value matching via the derivation metadata).
      3. Replace the params dict with OFA pool references.
      4. At runtime, substitute OFA pool arrays instead of per-subnet params.

    This "param substitution at load time" approach:
      - Keeps the compiler pipeline unchanged
      - Achieves memory sharing by having all subnet runtimes SET the same
        underlying NDArray objects for shared weights
      - Defers the full "derivation in Relay graph" to Phase C

    For the POC this is the RIGHT first step – it proves weight sharing is
    correct before we tackle the full graph rewrite.
    """

    def __init__(
        self,
        ofa_net,
        ofa_pool_base_weights: Dict[str, np.ndarray],
        ofa_pool_transform_matrices: Dict[str, np.ndarray],
        derivations: List[LayerDerivation],
        env,
        config,
        verbose: bool = True,
    ):
        self.ofa_net = ofa_net
        self.base_weights = ofa_pool_base_weights
        self.transform_matrices = ofa_pool_transform_matrices
        self.derivations = derivations
        self.env = env
        self.config = config
        self.verbose = verbose

    def materialise_params(self) -> Dict[str, np.ndarray]:
        """
        For each LayerDerivation, compute the derived weight (NumPy) and return
        a dict {canonical_key → np.ndarray}.

        These are the CORRECT weights the subnet expects.
        They are derived on the HOST from the OFA pool.

        canonical_key format: "ofa_pool/{base_weight_key}/out{os}_{oe}/in{is}_{ie}/ks{ks}"
        """
        params: Dict[str, np.ndarray] = {}
        for d in self.derivations:
            key = self._derivation_canonical_key(d)
            if key not in params:  # may be shared across subnets
                params[key] = derive_weight_numpy(
                    self.base_weights, self.transform_matrices, d
                )
        return params

    @staticmethod
    def _derivation_canonical_key(d: LayerDerivation) -> str:
        """
        A deterministic key for a derived weight, based purely on:
        - which base weight it comes from
        - which channel slice
        - which kernel size (after transforms)

        Two subnets that use the SAME slice/transform from the SAME base weight
        will produce the SAME canonical key → true weight sharing.
        """
        tm_str = "_".join(f"{s}to{t}" for s, t in d.transform_sequence) if d.transform_sequence else "none"
        return (
            f"ofa_pool/{d.base_weight_key}"
            f"/o{d.out_start}_{d.out_end}"
            f"/i{d.in_start}_{d.in_end}"
            f"/ks{d.active_kernel_size}"
            f"/tm{tm_str}"
        )


# ===========================================================================
# Build the full Relay graph using pool vars + explicit derivation ops
# (Used for the POC relay-level test, not for VTA compilation yet)
# ===========================================================================
def build_relay_with_ofa_pool_vars(
    ofa_net,
    arch: Dict[str, Any],
    derivations: List[LayerDerivation],
    base_weights: Dict[str, np.ndarray],
    transform_matrices: Dict[str, np.ndarray],
    bn_params: Dict[str, np.ndarray],
    other_params: Dict[str, np.ndarray],
    input_shape: List[int],
    input_name: str = "input0",
) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
    """
    Build a PURE Relay graph where all conv weights are expressed as
    derivation expressions (slice + optional transform) applied to
    OFA pool variables.

    This is the target architecture for Phase B.

    Returns
    -------
    mod    : Relay IRModule with pool variables
    params : dict of pool variable name → NDArray (to be uploaded ONCE and shared)
    """
    # --- Create Relay variables for all pool entries used by this subnet ---
    needed_base_keys = set(d.base_weight_key for d in derivations)
    needed_tm_keys   = set(key for d in derivations for key in d.transform_keys)

    pool_vars: Dict[str, relay.Var] = {}
    pool_params: Dict[str, np.ndarray] = {}

    for key in needed_base_keys:
        w = base_weights[key]
        var = relay.var(f"pool_{key}", shape=w.shape, dtype="float32")
        pool_vars[key] = var
        pool_params[f"pool_{key}"] = w

    for key in needed_tm_keys:
        m = transform_matrices[key]
        var = relay.var(f"pool_{key}", shape=m.shape, dtype="float32")
        pool_vars[key] = var
        pool_params[f"pool_{key}"] = m

    from training_ops.utils import make_divisible, MyNetwork
    width_mult_list = sorted(ofa_net.width_mult_list)

    x = relay.var(input_name, shape=input_shape, dtype="float32")
    deriv_iter = iter(derivations)

    # ------------------------------------------------------------------
    # Helper: consume N derivations and return one Relay expr per sub-conv
    # using transform-once-then-slice optimisation.
    # ------------------------------------------------------------------
    def get_conv_weight_exprs(
        n_subconvs: int,
        decomp_type: Optional[int] = None,
        return_derivs: bool = False,
    ):
        """
        Consume n_subconvs derivations from deriv_iter.

        Optimization policy:
          - dt in [1, 2] (output-split): transform once, then slice by out_ch.
          - dt in [3, 4] (input-split): keep per-subconv derivation so each
            branch preserves its exact in-channel slice.
        """
        if n_subconvs == 1:
            d = next(deriv_iter)
            base_var = pool_vars[d.base_weight_key]
            tm_vars = {k: pool_vars[k] for k in d.transform_keys}
            exprs = [_make_derived_weight_expr(base_var, tm_vars, d)]
            return (exprs, [d]) if return_derivs else exprs

        sub_derivs = [next(deriv_iter) for _ in range(n_subconvs)]

        if decomp_type in [3, 4]:
            exprs = [
                _make_derived_weight_expr(
                    pool_vars[d.base_weight_key],
                    {k: pool_vars[k] for k in d.transform_keys},
                    d,
                )
                for d in sub_derivs
            ]
            return (exprs, sub_derivs) if return_derivs else exprs

        base_key = sub_derivs[0].base_weight_key
        same_base = all(d.base_weight_key == base_key for d in sub_derivs)
        same_tm = all(
            d.transform_sequence == sub_derivs[0].transform_sequence
            and d.transform_keys == sub_derivs[0].transform_keys
            for d in sub_derivs
        )

        if not (same_base and same_tm):
            exprs = [
                _make_derived_weight_expr(
                    pool_vars[d.base_weight_key],
                    {k: pool_vars[k] for k in d.transform_keys},
                    d,
                )
                for d in sub_derivs
            ]
            return (exprs, sub_derivs) if return_derivs else exprs

        out_start_global = sub_derivs[0].out_start
        total_out = sum(d.out_ch for d in sub_derivs)
        in_ch = sub_derivs[0].in_ch

        template = sub_derivs[0]
        base_var = pool_vars[base_key]
        tm_vars = {k: pool_vars[k] for k in template.transform_keys}

        w_full = _make_full_layer_weight_expr(base_var, tm_vars, template, total_out_ch=total_out)

        exprs = []
        tgt_ks = template.active_kernel_size
        for d in sub_derivs:
            local_start = d.out_start - out_start_global
            local_end = local_start + d.out_ch
            w_i = relay.strided_slice(
                w_full,
                begin=[local_start, 0, 0, 0],
                end=[local_end, in_ch, tgt_ks, tgt_ks],
                strides=[1, 1, 1, 1],
            )
            exprs.append(w_i)
        return (exprs, sub_derivs) if return_derivs else exprs

    # ------------------------------------------------------------------
    # first conv (always dt=0)
    # ------------------------------------------------------------------
    out_ch_setting = arch["out_channel_setting_list"][0]
    first_out_ch = make_divisible(64 * width_mult_list[out_ch_setting], MyNetwork.CHANNEL_DIVISIBLE)

    [w_first] = get_conv_weight_exprs(1)
    x = relay.nn.conv2d(x, w_first,
                        strides=(2, 2), padding=(3, 3), kernel_size=(7, 7),
                        channels=first_out_ch, data_layout="NCHW", kernel_layout="OIHW")
    x = _apply_bn_from_ofa(x, ofa_net.first_layer[1], first_out_ch, pool_params)
    x = relay.nn.relu(x)
    x = relay.nn.max_pool2d(x, pool_size=(2, 2), strides=(2, 2))

    current_channels = first_out_ch

    # ------------------------------------------------------------------
    # residual blocks
    # ------------------------------------------------------------------
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
            res_stride = stride if residual_idx == 0 else 1
            x_res = x

            for conv_idx in range(depth):
                ks  = kernel_sizes[residual_idx][conv_idx]
                dt  = decomp_types[residual_idx][conv_idx]
                cin = residual_in_channels if conv_idx == 0 else out_channels
                cout = out_channels
                conv_stride = res_stride if conv_idx == 0 else 1
                pad = ks // 2

                # Determine number of sub-convolutions for this dt
                if dt == 0:
                    n_sub = 1
                elif dt in [1, 3]:
                    n_sub = 2
                else:  # dt in [2, 4]
                    n_sub = 4

                weight_exprs, weight_derivs = get_conv_weight_exprs(
                    n_sub, decomp_type=dt, return_derivs=True
                )

                if dt == 0:
                    x_res = relay.nn.conv2d(x_res, weight_exprs[0],
                                            strides=(conv_stride, conv_stride),
                                            padding=(pad, pad), kernel_size=(ks, ks),
                                            channels=cout, data_layout="NCHW", kernel_layout="OIHW")
                elif dt in [1, 2]:
                    # output-split → conv each group, concatenate
                    q = cout // n_sub
                    parts = []
                    for i, w_i in enumerate(weight_exprs):
                        ch_i = q if i < n_sub - 1 else (cout - q * (n_sub - 1))
                        parts.append(relay.nn.conv2d(
                            x_res, w_i,
                            strides=(conv_stride, conv_stride),
                            padding=(pad, pad), kernel_size=(ks, ks),
                            channels=ch_i, data_layout="NCHW", kernel_layout="OIHW",
                        ))
                    x_res = relay.concatenate(parts, axis=1)
                elif dt in [3, 4]:
                    # input-split -> conv each channel slice of x_res, then sum
                    parts = []
                    for w_i, d_i in zip(weight_exprs, weight_derivs):
                        x_slice = _slice_nchw_channels(x_res, d_i.in_start, d_i.in_end, d_i.input_h, d_i.input_w)
                        parts.append(
                            relay.nn.conv2d(
                                x_slice,
                                w_i,
                                strides=(conv_stride, conv_stride),
                                padding=(pad, pad),
                                kernel_size=(ks, ks),
                                channels=cout,
                                data_layout="NCHW",
                                kernel_layout="OIHW",
                            )
                        )
                    x_res = parts[0]
                    for p in parts[1:]:
                        x_res = relay.add(x_res, p)

                ofa_bn = ofa_residual.bns[conv_idx]
                x_res = _apply_bn_from_ofa(x_res, ofa_bn, cout, pool_params)
                if conv_idx < depth - 1:
                    x_res = relay.nn.relu(x_res)

            # Shortcut
            need_downsample = (
                (stride != 1 and residual_idx == 0) or
                (residual_in_channels != out_channels)
            )
            if need_downsample and residual_idx == 0:
                sc_dt = downsample_decomp if block_idx > 0 else 0
                if sc_dt == 0:
                    n_sc = 1
                elif sc_dt in [1, 3]:
                    n_sc = 2
                else:
                    n_sc = 4

                sc_weight_exprs, sc_weight_derivs = get_conv_weight_exprs(
                    n_sc, decomp_type=sc_dt, return_derivs=True
                )

                if sc_dt == 0:
                    shortcut = relay.nn.conv2d(
                        x, sc_weight_exprs[0],
                        strides=(res_stride, res_stride),
                        padding=(0, 0), kernel_size=(1, 1),
                        channels=out_channels, data_layout="NCHW", kernel_layout="OIHW",
                    )
                elif sc_dt in [1, 2]:
                    q = out_channels // n_sc
                    parts = []
                    for i, w_sc in enumerate(sc_weight_exprs):
                        ch_i = q if i < n_sc - 1 else (out_channels - q * (n_sc - 1))
                        parts.append(relay.nn.conv2d(
                            x, w_sc,
                            strides=(res_stride, res_stride),
                            padding=(0, 0), kernel_size=(1, 1),
                            channels=ch_i, data_layout="NCHW", kernel_layout="OIHW",
                        ))
                    shortcut = relay.concatenate(parts, axis=1)
                elif sc_dt in [3, 4]:
                    parts = []
                    for w_sc, d_sc in zip(sc_weight_exprs, sc_weight_derivs):
                        x_slice = _slice_nchw_channels(x, d_sc.in_start, d_sc.in_end, d_sc.input_h, d_sc.input_w)
                        parts.append(
                            relay.nn.conv2d(
                                x_slice,
                                w_sc,
                                strides=(res_stride, res_stride),
                                padding=(0, 0),
                                kernel_size=(1, 1),
                                channels=out_channels,
                                data_layout="NCHW",
                                kernel_layout="OIHW",
                            )
                        )
                    shortcut = parts[0]
                    for p in parts[1:]:
                        shortcut = relay.add(shortcut, p)
                else:
                    shortcut = x
            else:
                shortcut = x

            x = relay.add(x_res, shortcut)
            x = relay.nn.relu(x)

        current_channels = out_channels

    # ---- final layers ----
    x = relay.nn.adaptive_avg_pool2d(x, output_size=[1, 1])
    x = relay.reshape(x, newshape=[-1, current_channels])

    ofa_linear = ofa_net.last_layer[2]
    if hasattr(ofa_linear, 'linear'):
        actual_linear = ofa_linear.linear.linear if hasattr(ofa_linear.linear, 'linear') else ofa_linear.linear
    else:
        actual_linear = ofa_linear

    with torch.no_grad():
        fc_w = actual_linear.weight[:10, :current_channels].cpu().numpy()
        fc_b = (actual_linear.bias[:10].cpu().numpy()
                if actual_linear.bias is not None else np.zeros(10, dtype="float32"))

    fc_w_var = relay.var("fc_weight", shape=fc_w.shape, dtype="float32")
    fc_b_var = relay.var("fc_bias",   shape=fc_b.shape, dtype="float32")
    pool_params["fc_weight"] = fc_w
    pool_params["fc_bias"]   = fc_b

    x = relay.nn.dense(x, fc_w_var, units=10)
    x = relay.nn.bias_add(x, fc_b_var, axis=-1)

    # mod = tvm.IRModule()

    func = relay.Function(relay.analysis.free_vars(x), x)
    mod = tvm.IRModule.from_expr(func)
    # print(mod)
    mod = relay_transform.InferType()(mod)
    # mod["main"] = relay.Function(relay.analysis.free_vars(x), x)
    # Convert params to TVM NDArrays
    tvm_params = {k: tvm.nd.array(v) for k, v in pool_params.items()}

    return mod, tvm_params


def _apply_bn_from_ofa(x, ofa_bn_module, channels: int, pool_params: Dict[str, np.ndarray]) -> relay.Expr:
    """Apply BatchNorm to x using weights from OFA BN module."""
    bn_mod = ofa_bn_module.bn if hasattr(ofa_bn_module, 'bn') else ofa_bn_module

    with torch.no_grad():
        w  = bn_mod.weight[:channels].cpu().numpy()
        b  = bn_mod.bias[:channels].cpu().numpy()
        rm = bn_mod.running_mean[:channels].cpu().numpy()
        rv = bn_mod.running_var[:channels].cpu().numpy()

    eps = float(bn_mod.eps) if hasattr(bn_mod, 'eps') else 1e-5

    # Create unique var names
    base = str(id(ofa_bn_module))
    names = {
        f"_bn_{base}_gamma":    w,
        f"_bn_{base}_beta":     b,
        f"_bn_{base}_mean":     rm,
        f"_bn_{base}_var":      rv,
    }
    vars_ = {}
    for n, arr in names.items():
        pool_params[n] = arr
        vars_[n] = relay.var(n, shape=(channels,), dtype="float32")

    x = relay.nn.batch_norm(
        x,
        vars_[f"_bn_{base}_gamma"],
        vars_[f"_bn_{base}_beta"],
        vars_[f"_bn_{base}_mean"],
        vars_[f"_bn_{base}_var"],
        epsilon=eps,
        center=True,
        scale=True,
    )[0]
    return x


def _slice_nchw_channels(x: relay.Expr, c_begin: int, c_end: int, h_end: int, w_end: int) -> relay.Expr:
    """Explicit NCHW channel slice with static-friendly bounds.

    We intentionally avoid -1 in end indices and avoid querying x.shape at
    construction time. Large H/W end values are clipped by strided_slice.
    """
    return relay.strided_slice(
        x,
        begin=[0, c_begin, 0, 0],
        end=[1, c_end, h_end, w_end],
        strides=[1, 1, 1, 1],
    )

