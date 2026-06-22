"""Relay pass for lowering dense-transform subgraphs using GEMM_Mat_Trf.

Pattern produced by _apply_transform_sequence in ofa_relay_graph_builder:

  strided_slice(base_var, ...)          # channel + spatial crop
    → reshape([N, tgt_ks^2])
    → nn.dense(_, transform_var)        # THE TARGET: weight is a known transform matrix
    → reshape([out_ch, in_ch, tgt_ks, tgt_ks])
    → nn.conv2d(data, derived_weight)

Two lowering modes:

  cpu_materialization=True  (validation / step-4 path)
    If a precomputed dict is provided, substitutes relay.const for each matched dense.
    Otherwise leaves the dense op unchanged (CPU will run it via x86 dense).

  cpu_materialization=False  (VTA intrinsic path)
    Wraps each matched nn.dense in a composite function tagged "vta.gemm_mat_trf".
    The VTA backend will schedule these with the gemm_mat_trf intrinsic (mode=1 GEMM).
"""

from __future__ import absolute_import, print_function

import tvm
import tvm.ir
from tvm import relay
from tvm.relay.expr_functor import ExprMutator, ExprVisitor


def _normalize_names(transform_var_names):
    """Return a set containing both 'pool_X' and 'X' forms for every name."""
    normalized = set()
    for n in (transform_var_names or []):
        normalized.add(n)
        if n.startswith("pool_"):
            normalized.add(n[5:])
        else:
            normalized.add("pool_" + n)
    return normalized


def _unwrap_quant_chain_to_var(expr):
    """Walk cast/clip/round/multiply chains to find an underlying relay.Var.

    Handles the pattern produced by enable_dynamic_dense_quant=True:
      cast(clip(round(multiply(cast(transform_var, float32), scale))), int8)
    Returns the relay.Var if found, else None.
    """
    if isinstance(expr, relay.Var):
        return expr
    if not isinstance(expr, relay.Call) or not isinstance(expr.op, tvm.ir.Op):
        return None
    op_name = expr.op.name
    if op_name in ("cast", "round", "clip", "annotation.stop_fusion"):
        return _unwrap_quant_chain_to_var(expr.args[0])
    if op_name == "multiply":
        v = _unwrap_quant_chain_to_var(expr.args[0])
        if v is not None:
            return v
        return _unwrap_quant_chain_to_var(expr.args[1])
    return None


def _is_transform_weight(expr, normalized_names):
    """True if expr is a known transform var, or a quantize chain wrapping one.

    The enable_dynamic_dense_quant path wraps transform vars in:
      cast(clip(round(multiply(cast(var, float32), scale))), int8)
    We detect this by unwrapping the chain to find the underlying Var.
    """
    if isinstance(expr, relay.Var):
        return expr.name_hint in normalized_names
    underlying = _unwrap_quant_chain_to_var(expr)
    return (underlying is not None
            and isinstance(underlying, relay.Var)
            and underlying.name_hint in normalized_names)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class DenseTransformDetector(ExprVisitor):
    """Walk the graph and record every nn.dense whose weight is a transform var."""

    def __init__(self, transform_var_names):
        super().__init__()
        self._names = _normalize_names(transform_var_names)
        self.detected = []   # relay.Call nodes

    def visit_call(self, call):
        super().visit_call(call)
        if (
            isinstance(call.op, tvm.ir.Op)
            and call.op.name == "nn.dense"
            and len(call.args) >= 2
            and _is_transform_weight(call.args[1], self._names)
        ):
            self.detected.append(call)


# ---------------------------------------------------------------------------
# Lowerer
# ---------------------------------------------------------------------------

class DenseTransformLowerer(ExprMutator):
    """Replace detected transform dense ops according to the chosen mode."""

    def __init__(self, transform_var_names, cpu_materialization=True, precomputed=None):
        super().__init__()
        self._names = _normalize_names(transform_var_names)
        self.cpu_materialization = cpu_materialization
        # precomputed: raw_transform_name -> np.ndarray [N, tgt_ks^2]
        # (output of materialize_transforms_cpu; keys may or may not have 'pool_' prefix)
        self._precomputed = _normalize_names_dict(precomputed or {})
        self.n_lowered = 0

    def visit_call(self, call):
        # Capture the original (type-checked) args before super() rebuilds the call.
        # After ExprMutator rebuilds nodes, the new objects lose their checked_type
        # annotation, which _lower_vta needs to construct the composite's param vars.
        orig_args = list(call.args)

        call = super().visit_call(call)

        if not (
            isinstance(call.op, tvm.ir.Op)
            and call.op.name == "nn.dense"
            and len(call.args) >= 2
            and _is_transform_weight(call.args[1], self._names)
        ):
            return call

        data_arg   = call.args[0]
        weight_arg = call.args[1]
        self.n_lowered += 1

        if self.cpu_materialization:
            return self._lower_cpu(data_arg, weight_arg, call)
        else:
            return self._lower_vta(data_arg, weight_arg, call,
                                   orig_data=orig_args[0], orig_weight=orig_args[1])

    # ------------------------------------------------------------------
    def _lower_cpu(self, data_arg, weight_arg, original_call):
        """Substitute relay.const if pre-computed, else leave for CPU dense."""
        import numpy as np
        # weight_arg may be a bare Var or a quantize chain wrapping one.
        underlying = (weight_arg if isinstance(weight_arg, relay.Var)
                      else _unwrap_quant_chain_to_var(weight_arg))
        name = getattr(underlying, "name_hint", None)
        if name is not None:
            arr = self._precomputed.get(name)
            if arr is not None:
                return relay.const(arr.astype("float32"))
        # No pre-computed value: CPU dense will execute it at runtime.
        return original_call

    # ------------------------------------------------------------------
    def _lower_vta(self, data_arg, weight_arg, original_call,
                   orig_data=None, orig_weight=None):
        """Wrap in a composite function tagged 'vta.gemm_mat_trf'.

        The composite function preserves the nn.dense semantics.
        The 'Composite' attribute is the hook for the VTA backend to schedule
        using the gemm_mat_trf intrinsic (mode=1 GEMM instruction, bit 7).

        Requires the module to be type-checked (InferType run beforehand) so
        that shape/dtype information is available.  orig_data / orig_weight are
        the pre-visit args that still carry their checked_type annotation; the
        post-visit (rebuilt) data_arg / weight_arg are used in the composite call.
        """
        type_src_data   = orig_data   if orig_data   is not None else data_arg
        type_src_weight = orig_weight if orig_weight is not None else weight_arg
        try:
            x_type = type_src_data.checked_type
            w_type = type_src_weight.checked_type
        except Exception:
            # Type info unavailable — cannot build composite; leave as-is.
            return original_call

        x_shape = [int(d) for d in x_type.shape]
        w_shape = [int(d) for d in w_type.shape]

        x = relay.var("x", shape=x_shape, dtype=x_type.dtype)
        w = relay.var("w", shape=w_shape, dtype=w_type.dtype)

        attrs = original_call.attrs
        out_dtype = attrs.out_dtype if (attrs and attrs.out_dtype) else ""
        inner = relay.nn.dense(x, w, out_dtype=out_dtype if out_dtype else None)

        fn = relay.Function([x, w], inner)
        fn = fn.with_attr("Composite", "vta.gemm_mat_trf")
        fn = fn.with_attr("PartitionedFromPattern", "nn.dense_")

        return relay.Call(fn, [data_arg, weight_arg])


# ---------------------------------------------------------------------------
# Helper: normalize a dict so both 'pool_X' and 'X' keys resolve to the value
# ---------------------------------------------------------------------------

def _normalize_names_dict(d):
    out = {}
    for k, v in d.items():
        out[k] = v
        if k.startswith("pool_"):
            out[k[5:]] = v
        else:
            out["pool_" + k] = v
    return out


# ---------------------------------------------------------------------------
# Public pass entry point
# ---------------------------------------------------------------------------

def lower_gemm_mat_trf(
    mod,
    transform_var_names=None,
    cpu_materialization=True,
    precomputed=None,
    verbose=False,
):
    """Detect and lower dense-transform subgraph patterns.

    Parameters
    ----------
    mod : tvm.IRModule
    transform_var_names : set or list, optional
        Names of pool variables that are OFA transform matrices.
        Accepts both 'pool_X' and raw 'X' forms.
    cpu_materialization : bool
        True  → substitute relay.const(precomputed[name]) when available,
                 else leave as nn.dense for CPU execution.
        False → wrap in composite function "vta.gemm_mat_trf" for VTA intrinsic scheduling.
    precomputed : dict, optional
        {transform_name -> np.ndarray} from materialize_transforms_cpu.
        Only used when cpu_materialization=True.
    verbose : bool
        Print detection / lowering summary.

    Returns
    -------
    tvm.IRModule
    """
    if transform_var_names is None:
        transform_var_names = set()

    # Type-check first — needed for shape/dtype in VTA composite construction
    # and for correct InferType after the pass.
    mod = relay.transform.InferType()(mod)

    detector = DenseTransformDetector(transform_var_names)
    detector.visit(mod["main"])

    if verbose:
        print(
            "[lower_gemm_mat_trf] detected %d transform dense op(s)" % len(detector.detected)
        )

    if not detector.detected:
        return mod

    lowerer = DenseTransformLowerer(
        transform_var_names=transform_var_names,
        cpu_materialization=cpu_materialization,
        precomputed=precomputed,
    )
    new_body = lowerer.visit(mod["main"].body)

    if verbose:
        mode = "cpu" if cpu_materialization else "vta_composite"
        print("[lower_gemm_mat_trf] lowered %d op(s) → %s" % (lowerer.n_lowered, mode))

    func = mod["main"]
    new_func = relay.Function(
        func.params,
        new_body,
        func.ret_type,
        func.type_params,
        func.attrs,
    )
    mod = tvm.IRModule.from_expr(new_func)
    return relay.transform.InferType()(mod)
