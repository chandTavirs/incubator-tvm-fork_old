"""Pack standalone GMTF dense ops in the derive module for VTA FPGA execution.

The derive module's body contains expressions of the form:

    reshape(cast(clip(right_shift(nn.dense(reshape(slice(base_weight)),
                                           transform_var),
                                  shift_const), -127, 127), "int8"),
            [O, I, ks, ks])

where transform_var may be wrapped in the enable_dynamic_dense_quant quantize chain
(cast→multiply→round→clip→cast).  Without packing, VTA cannot schedule these as
GMTF ops because the data is in flat [N, K] format instead of the
[N//BATCH, k_outer, BATCH, BLOCK_IN] packed layout.

DeriveGMTFPacker replaces each such subgraph (detected by the K×K weight shape,
K=9 or K=25) with:

  1. Pack data [N, K] → [N, k_outer, 1, BLOCK_IN]
  2. Pad+pack weight [K, K] → packed weight for vta.gmtf_dense_small/large
  3. vta.gmtf_dense_small/large(...) → [N, k_outer, 1, BLOCK_OUT]   (int32, in ACC)
  4. right_shift + clip + cast(int8) on the 4D result           (VTA ALU epilogue)
  5. Unpack 4D int8 → [N, K] int8                              (ARM CPU, tiny)

The unpack and outer reshape to OIHW run on ARM CPU.  The VTA DMA store correctly
writes int8 (required by VTA's CopyIntrinInjector).  All GMTF compute + quantize
runs on VTA FPGA.

Two patterns are handled:
  A) Full quant chain: cast(clip(right_shift(nn.dense, shift), lo, hi), "int8")
     → replace the entire chain; quant ops applied to 4D VTA output.
  B) Raw nn.dense with no surrounding quant chain (fallback, int32 output).
     → VTA DMA-store of int32 will FAIL unless built with CPU target; included
       as a fallback for non-quantized paths.

This pass must be applied AFTER relay.transform.InferType() and on raw nn.dense
nodes (NOT vta.gemm_mat_trf composites).  Do NOT call step4b before this pass.
"""

import math

import tvm
import tvm.ir
from tvm import relay
from tvm.relay import op
from tvm.relay.expr_functor import ExprMutator

from vta.top.graphpack import (
    _pack_batch_channel_dense,
    _pack_weight_dense,
    _pack_weight_gmtf_large,
    _weight_shape_match_dense,
    _GMTF_DIM_SMALL,
    _GMTF_DIM_LARGE,
)


class DeriveGMTFPacker(ExprMutator):
    """Replace nn.dense with 9×9 or 25×25 weight with packed vta.gmtf_dense_small/large.

    Handles two patterns:
      A) cast(int8, clip(right_shift(nn.dense, shift), lo, hi))  — full quant chain.
         Detected TOP-DOWN from the cast node (before visiting children) so that
         right_shift+clip+cast can be applied to the 4D GMTF output rather than to
         the 2D unpacked result.  This is required because VTA's DMA store only
         supports int8 output, so the quant ops must fuse INTO the VTA function as
         VTA ALU instructions.
      B) Raw nn.dense with no surrounding quant chain (int32 output).  The VTA DMA
         store of int32 will fail with a CopyIntrinInjector error if built for VTA
         target; only use this path with CPU target.
    """

    def __init__(self, bfactor, blockin, blockout):
        super().__init__()
        self.bfactor = bfactor
        self.blockin = blockin
        self.blockout = blockout
        self.n_packed = 0

    # ------------------------------------------------------------------
    # Top-level visitor
    # ------------------------------------------------------------------

    def visit_call(self, call):
        # Pattern A: top-down detection of cast(int8, clip(right_shift(nn.dense, ...))).
        # MUST be checked BEFORE super().visit_call() so we inspect original (un-visited)
        # args and can apply quant ops to the 4D VTA output rather than the 2D unpack.
        if isinstance(call.op, tvm.ir.Op) and call.op.name == "cast":
            result = self._try_handle_quant_chain(call)
            if result is not None:
                return result

        # Standard bottom-up visit.
        orig_args = list(call.args)
        call = super().visit_call(call)

        # Pattern B: raw nn.dense (fallback, produces int32 output).
        if isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.dense":
            return self._try_handle_raw_dense(call, orig_args)

        return call

    # ------------------------------------------------------------------
    # Pattern A: full quant chain
    # ------------------------------------------------------------------

    def _try_handle_quant_chain(self, cast_call):
        """Detect cast(int8, clip(right_shift(nn.dense(data, weight_gmtf), shift), lo, hi)).

        Returns the packed replacement (4D GMTF + quant ALU ops + unpack) or None.
        """
        if str(cast_call.attrs.dtype) != "int8":
            return None

        clip_call = cast_call.args[0]
        if not (isinstance(clip_call, relay.Call) and isinstance(clip_call.op, tvm.ir.Op)
                and clip_call.op.name == "clip"):
            return None

        shift_call = clip_call.args[0]
        if not (isinstance(shift_call, relay.Call) and isinstance(shift_call.op, tvm.ir.Op)
                and shift_call.op.name == "right_shift"):
            return None

        dense_call = shift_call.args[0]
        if not (isinstance(dense_call, relay.Call) and isinstance(dense_call.op, tvm.ir.Op)
                and dense_call.op.name == "nn.dense"):
            return None

        # Check GMTF weight shape.
        try:
            w_shape = [int(d) for d in dense_call.args[1].checked_type.shape]
            d_shape = [int(d) for d in dense_call.args[0].checked_type.shape]
        except Exception:
            return None

        is_small = (len(w_shape) == 2
                    and w_shape[0] == _GMTF_DIM_SMALL
                    and w_shape[1] == _GMTF_DIM_SMALL)
        is_large = (len(w_shape) == 2
                    and w_shape[0] == _GMTF_DIM_LARGE
                    and w_shape[1] == _GMTF_DIM_LARGE)
        if not (is_small or is_large) or len(d_shape) != 2:
            return None

        N, K = d_shape
        if N % self.bfactor != 0:
            return None

        # Extract quant chain info (visit constants so any sub-vars are substituted).
        shift_val = self.visit(shift_call.args[1])
        lo = clip_call.attrs.a_min
        hi = clip_call.attrs.a_max

        # Visit the dense's data and weight (handles nested substitutions, e.g., the
        # enable_dynamic_dense_quant chain on the weight var).
        attrs = dense_call.attrs
        out_dtype = (attrs.out_dtype if (attrs and attrs.out_dtype) else "") or "int32"
        data   = self.visit(dense_call.args[0])
        weight = self.visit(dense_call.args[1])

        from vta.top import vta_gmtf_op as gmtf  # noqa: F401

        # Build packed 4D GMTF result (int32, in VTA ACC scope).
        if is_small:
            gmtf_4d = self._gmtf_small_4d(data, weight, d_shape, w_shape, N, K, out_dtype, gmtf)
        else:
            gmtf_4d = self._gmtf_large_4d(data, weight, d_shape, w_shape, N, K, out_dtype, gmtf)

        # Apply right_shift + clip + cast to the 4D result.
        # These element-wise ops fuse into the VTA function as VTA ALU instructions.
        # VTA's DMA store requires int8; the cast here satisfies that constraint.
        gmtf_4d = relay.right_shift(gmtf_4d, shift_val)
        gmtf_4d = relay.clip(gmtf_4d, a_min=lo, a_max=hi)
        gmtf_4d = relay.cast(gmtf_4d, "int8")

        # Unpack (CPU): 4D int8 → [N, blockout*n_buses] int8 → [N, K] int8.
        n_buses = math.ceil(K / self.blockin)
        out = op.reshape(gmtf_4d, newshape=[N, n_buses * self.blockout])
        out = op.strided_slice(out, begin=[0, 0], end=[N, K])

        self.n_packed += 1
        return out

    # ------------------------------------------------------------------
    # Pattern B: raw nn.dense (fallback, int32 output)
    # ------------------------------------------------------------------

    def _try_handle_raw_dense(self, call, orig_args):
        """Fallback for nn.dense without a surrounding quant chain (int32 output).

        NOTE: Produces int32 output.  VTA's DMA-store constraint (int8 only) means
        relay.build with VTA target will fail with CopyIntrinInjector.  Only use
        this path if the derive module is compiled for ARM CPU target, or if the
        downstream context handles the int32-output VTA function.
        """
        if not (isinstance(call.op, tvm.ir.Op) and call.op.name == "nn.dense"):
            return call
        if len(orig_args) < 2:
            return call

        try:
            w_shape = [int(d) for d in orig_args[1].checked_type.shape]
            d_shape = [int(d) for d in orig_args[0].checked_type.shape]
        except Exception:
            return call

        is_small = (len(w_shape) == 2
                    and w_shape[0] == _GMTF_DIM_SMALL
                    and w_shape[1] == _GMTF_DIM_SMALL)
        is_large = (len(w_shape) == 2
                    and w_shape[0] == _GMTF_DIM_LARGE
                    and w_shape[1] == _GMTF_DIM_LARGE)
        if not (is_small or is_large) or len(d_shape) != 2:
            return call

        N, K = d_shape
        if N % self.bfactor != 0:
            return call

        attrs = call.attrs
        out_dtype = (attrs.out_dtype if (attrs and attrs.out_dtype) else "") or "int32"
        data   = call.args[0]
        weight = call.args[1]

        from vta.top import vta_gmtf_op as gmtf  # noqa: F401

        if is_small:
            out = self._pack_small(data, weight, d_shape, w_shape, N, K, out_dtype, gmtf)
        else:
            out = self._pack_large(data, weight, d_shape, w_shape, N, K, out_dtype, gmtf)

        self.n_packed += 1
        return out

    # ------------------------------------------------------------------
    # 4D GMTF builders (no unpack — for use with pattern A)
    # ------------------------------------------------------------------

    def _gmtf_small_4d(self, data, weight, d_shape, w_shape, N, K, out_dtype, gmtf):
        """Pack + emit vta.gmtf_dense_small; return raw 4D [N, 1, 1, BLOCK_OUT] int32."""
        packed_data = _pack_batch_channel_dense(data, d_shape, self.bfactor, self.blockin)
        weight, w_shape_p, _ = _weight_shape_match_dense(
            weight, w_shape, None, self.blockout, self.blockin
        )
        packed_weight = _pack_weight_dense(weight, w_shape_p, self.blockout, self.blockin)
        return gmtf.gmtf_dense_small(packed_data, packed_weight, out_dtype=out_dtype)

    def _gmtf_large_4d(self, data, weight, d_shape, w_shape, N, K, out_dtype, gmtf):
        """Pack + emit vta.gmtf_dense_large; return raw 4D [N, 2, 1, BLOCK_OUT] int32."""
        packed_data = _pack_batch_channel_dense(data, d_shape, self.bfactor, self.blockin)
        packed_weight = _pack_weight_gmtf_large(weight, w_shape, self.blockout, self.blockin)
        return gmtf.gmtf_dense_large(packed_data, packed_weight, out_dtype=out_dtype)

    # ------------------------------------------------------------------
    # Pattern-B helpers (with unpack, int32 output)
    # ------------------------------------------------------------------

    def _pack_small(self, data, weight, d_shape, w_shape, N, K, out_dtype, gmtf):
        out = self._gmtf_small_4d(data, weight, d_shape, w_shape, N, K, out_dtype, gmtf)
        out = op.reshape(out, newshape=[N, self.blockout])
        out = op.strided_slice(out, begin=[0, 0], end=[N, K])
        return out

    def _pack_large(self, data, weight, d_shape, w_shape, N, K, out_dtype, gmtf):
        out = self._gmtf_large_4d(data, weight, d_shape, w_shape, N, K, out_dtype, gmtf)
        n_buses = math.ceil(K / self.blockin)
        out = op.reshape(out, newshape=[N, n_buses * self.blockout])
        out = op.strided_slice(out, begin=[0, 0], end=[N, K])
        return out


def pack_derive_gmtf(derive_mod, bfactor, blockin, blockout, verbose=False):
    """Apply DeriveGMTFPacker to the derive module.

    Detects nn.dense nodes with 9×9 or 25×25 weight shapes (surrounded by a
    right_shift+clip+cast(int8) quant chain) and replaces them with:
      vta.gmtf_dense_small/large in packed layout
      + right_shift+clip+cast(int8) on the 4D output (VTA ALU epilogue)
      + unpack back to [N, K] int8 (ARM CPU)

    The VTA function outputs int8, satisfying VTA's DMA-store constraint.

    Parameters
    ----------
    derive_mod : tvm.IRModule
        Must have InferType already applied.
    bfactor, blockin, blockout : int
        From vta.get_env(): env.BATCH, env.BLOCK_IN, env.BLOCK_OUT.
    verbose : bool

    Returns
    -------
    tvm.IRModule with GMTF dense ops packed for VTA scheduling.
    """
    derive_mod = relay.transform.InferType()(derive_mod)
    packer = DeriveGMTFPacker(bfactor, blockin, blockout)
    new_body = packer.visit(derive_mod["main"].body)

    if verbose:
        print("[pack_derive_gmtf] packed %d GMTF dense op(s) for VTA FPGA" % packer.n_packed)

    if packer.n_packed == 0:
        return derive_mod

    fn = derive_mod["main"]
    new_fn = relay.Function(fn.params, new_body, fn.ret_type, fn.type_params, fn.attrs)
    new_mod = tvm.IRModule.from_expr(new_fn)
    return relay.transform.InferType()(new_mod)
