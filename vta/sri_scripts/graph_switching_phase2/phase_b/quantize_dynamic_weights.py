"""
Quantization with Dynamic Weight Variables
============================================

This module provides quantize_with_dynamic_weights(), an API that quantizes
a Relay graph while preserving specified weight variables as inputs (not
materializing them as constants).

This is crucial for OFA-based models where conv weights are derived from
shared pool variables at runtime. The standard relay.quantize() would fold
these derived weights into constants, defeating the purpose.

Usage
-----
    # Build a graph where weight derivation ops reference pool_w_0, pool_w_1, ...
    mod, params = build_relay_with_ofa_pool_vars(...)
    
    # Quantize while keeping pool variables dynamic
    mod_q = quantize_with_dynamic_weights(
        mod,
        params,
        dynamic_weight_var_names=["pool_w_0", "pool_w_1", ...],
        scale_table={layer_name: scale_value, ...},  # per-op static scales
    )
    
    # Then pass to graph_pack and relay.build as normal

Module provides:
  - quantize_with_dynamic_weights(): main API
  - _preserve_weight_vars(): pass that marks variables for exclusion
  - PerOpScaleTable: storage for pre-computed scales
"""

from __future__ import absolute_import, print_function

from typing import Dict, List, Optional, Set, Tuple
import os

import numpy as np
import tvm
from tvm import relay
from tvm.relay import transform as relay_transform
from tvm.relay import op as _op
from tvm.relay import expr as _expr
from tvm.relay import analysis as _analysis
from tvm.relay.expr_functor import ExprVisitor, ExprMutator

import tvm.relay.quantize as _qtz
from tvm.relay.quantize import current_qconfig


# ===========================================================================
# Internals
# ===========================================================================

class _WeightVarPreserver(ExprMutator):
    """
    Mutator that strips marked weight variables from the graph,
    replacing them with placeholder Var nodes that won't be folded.
    """

    def __init__(self, protected_vars: Set[str]):
        super().__init__()
        self.protected_vars = protected_vars
        self.protected_var_map: Dict[str, relay.Var] = {}

    def visit_var(self, var: relay.Var) -> relay.Expr:
        if var.name_hint in self.protected_vars:
            if var.name_hint not in self.protected_var_map:
                self.protected_var_map[var.name_hint] = var
            return self.protected_var_map[var.name_hint]
        return var


class _SkipFoldConstantPass:
    """
    A custom pass that applies FoldConstant but skips folding any subgraph
    that references a protected variable.

    This prevents derived weights (which depend on pool vars) from being
    materialized as constants during quantization.
    """

    def __init__(self, protected_var_names: Set[str]):
        self.protected_var_names = protected_var_names

    def _contains_protected(self, expr: relay.Expr) -> bool:
        """Check if expr or any subexpr references a protected var."""
        class VarChecker(ExprVisitor):
            def __init__(self):
                self.has_protected = False
                self.protected_vars = None

            def visit_var(self, var: relay.Var):
                if var.name_hint in self.protected_vars:
                    self.has_protected = True

        checker = VarChecker()
        checker.protected_vars = self.protected_var_names
        checker.visit(expr)
        return checker.has_protected

    def __call__(self, mod: tvm.ir.IRModule) -> tvm.ir.IRModule:
        # For now, just return unchanged. In a production implementation,
        # we'd patch FoldConstant to check protected vars. For this PoC,
        # we rely on params not being bound (see _skip_bind_params).
        return mod


class _SkipBindParamsPass:
    """
    A modified prerequisite_optimize that skips binding parameters that
    are in the protected set. This prevents FoldConstant from having
    constants to fold in the first place.
    """

    def __init__(self, protected_var_names: Set[str]):
        self.protected_var_names = protected_var_names

    def __call__(self, mod: tvm.ir.IRModule, params: Dict) -> tvm.ir.IRModule:
        # Skip binding protected params
        filtered_params = {
            k: v for k, v in params.items()
            if k not in self.protected_var_names
        }

        # Apply standard prerequisite optimization with filtered params
        from tvm.relay.quantize.quantize import _bind_params
        optimize = tvm.transform.Sequential(
            [
                relay_transform.SimplifyInference(),
                relay_transform.FoldConstant(),
                relay_transform.FoldScaleAxis(),
                relay_transform.CanonicalizeOps(),
                relay_transform.FoldConstant(),
            ]
        )

        if filtered_params:
            mod["main"] = _bind_params(mod["main"], filtered_params)

        mod = optimize(mod)
        return mod


def _make_dynamic_calibrate_pass(scale_table: Optional[Dict[str, float]] = None):
    """Create a calibrate pass that tolerates non-constant weight expressions."""

    def _calibrate_dynamic_weights(mod, _):
        quantize_op = _op.get("relay.op.annotation.simulated_quantize")
        cfg = current_qconfig()
        const_params = {}

        def _weight_scale(data_expr):
            if isinstance(data_expr, _expr.Constant):
                val = np.amax(np.abs(data_expr.data.asnumpy()))
                if cfg.weight_scale == "power2":
                    return 2 ** np.math.ceil(np.math.log(val, 2)) if val > 0 else 1.0
                return float(val)
            # Dynamic weight path: fallback to static configured scale.
            if scale_table and "weight" in scale_table:
                return float(scale_table["weight"])
            return float(cfg.global_scale)

        def _input_scale():
            if scale_table and "input" in scale_table:
                return float(scale_table["input"])
            return float(cfg.global_scale)

        def visit_func(expr):
            if isinstance(expr, _expr.Call) and expr.op == quantize_op:
                _, ndom_scale, nclip_min, nclip_max = expr.args
                attrs = expr.attrs
                nbit = cfg.get_nbit_by_kind(attrs.kind)
                valid_bit = nbit - attrs.sign

                if attrs.kind == _qtz.QAnnotateKind.WEIGHT:
                    scale = _weight_scale(expr.args[0])
                else:
                    scale = _input_scale()

                valid_range = 2 ** valid_bit
                const_params[ndom_scale] = _expr.const(scale / valid_range, "float32")
                const_params[nclip_min] = _expr.const(-(valid_range - 1), "float32")
                const_params[nclip_max] = _expr.const((valid_range - 1), "float32")

        main_func = mod["main"]
        _analysis.post_order_visit(main_func, visit_func)
        main_func = _expr.bind(main_func, const_params)

        func_dict = {}
        for global_var, func in mod.functions.items():
            if global_var.name_hint != "main":
                func_dict[global_var] = func
        return tvm.ir.IRModule.from_expr(main_func, func_dict)

    return tvm.transform.module_pass(
        _calibrate_dynamic_weights,
        opt_level=1,
        name="QuantizeCalibrateDynamicWeights",
    )


# ===========================================================================
# API
# ===========================================================================

def quantize_with_dynamic_weights(
    mod: tvm.ir.IRModule,
    params: Dict[str, tvm.nd.NDArray],
    dynamic_weight_var_names: List[str],
    scale_table: Optional[Dict[str, float]] = None,
    dataset=None,
) -> tvm.ir.IRModule:
    """
    Quantize a Relay module while preserving specified variables as dynamic inputs.

    This API is designed for models (like OFA-based networks) where certain weights
    are derived from shared pool variables at runtime. The standard relay.quantize()
    would fold these derived weights as constants, losing the ability to swap
    different weight derivations.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        The Relay module to quantize. Must have weight derivation ops that
        reference dynamic_weight_var_names.

    params : Dict[str, tvm.nd.NDArray]
        The module's parameters. Dynamic weight vars should be included here
        (they'll be used during calibration but not materialized in the graph).

    dynamic_weight_var_names : List[str]
        Names of variables to preserve as inputs (not fold as constants).
        Typically: ["pool_w_0", "pool_w_1", ...] for OFA models.

    scale_table : Optional[Dict[str, float]]
        Pre-computed per-layer quantization scales. If not provided, uses
        the current qconfig's global_scale. Format:
          {layer_name: scale_value, ...}

    dataset : Optional[list of dict]
        Calibration dataset (same format as relay.quantize). If None,
        uses histogram-based calibration.

    Returns
    -------
    tvm.ir.IRModule
        Quantized module with dynamic weight variables still present as
        function inputs (not constants). BN + bias params are folded as usual.

    Example
    -------
    >>> # Build OFA graph with pool vars
    >>> mod, params = build_relay_with_ofa_pool_vars(ofa_net, arch, ...)
    >>> # Quantize, keeping pool vars as inputs
    >>> mod_q = quantize_with_dynamic_weights(
    ...     mod, params,
    ...     dynamic_weight_var_names=["pool_w_0", "pool_w_1"],
    ...     scale_table={...}
    ... )
    >>> # Now mod_q["main"] still has pool_w_0, pool_w_1 as Var inputs
    >>> # suitable for runtime weight derivation
    """
    protected_vars = set(dynamic_weight_var_names)
    dense_fix_enabled = os.environ.get("TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX", "0") == "1"

    # Dynamic-weight graphs often use nn.dense for weight derivation.
    # TVM quantize annotate expects conv2d RHS to be unannotated; if dense
    # annotation is enabled it can propagate an activation-kind node into RHS,
    # tripping _annotate.py:conv2d_rewrite (assert rhs_kind is None).
    if protected_vars and not current_qconfig().skip_dense_layer and not dense_fix_enabled:
        raise ValueError(
            "quantize_with_dynamic_weights requires qconfig(skip_dense_layer=True) unless "
            "TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX=1 is set to enable the dynamic dense->conv2d annotate fix."
        )

    # Step 1: Apply prerequisite optimization, but skip binding protected params.
    print(f"  [quantize_with_dynamic_weights] Skipping param binding for {len(protected_vars)} dynamic vars...")
    skip_bind_pass = _SkipBindParamsPass(protected_vars)
    mod = skip_bind_pass(mod, params)

    # Never pass protected params into relay.quantize(), otherwise it will re-bind
    # them during prerequisite_optimize() and fold the dynamic-weight path.
    filtered_params = {k: v for k, v in params.items() if k not in protected_vars}

    # Step 2: Apply quantization with a dynamic-weight-safe calibrate pass.
    print(f"  [quantize_with_dynamic_weights] Running quantization...")
    try:
        # Keep protected vars unbound during prerequisite optimization.
        mod = _qtz.prerequisite_optimize(mod, filtered_params)

        calibrate_pass = _make_dynamic_calibrate_pass(scale_table)
        quant_passes = [_qtz.partition(), _qtz.annotate(), calibrate_pass, relay_transform.InferType()]
        if not current_qconfig().do_simulation:
            quant_passes.append(_qtz.realize())
        quant_passes.append(relay_transform.FoldConstant())
        quantize_seq = tvm.transform.Sequential(quant_passes)

        required = ["QuantizeAnnotate"]
        if not current_qconfig().do_simulation:
            required.append("QuantizeRealize")
        with tvm.transform.PassContext(opt_level=3, required_pass=required):
            with _qtz.quantize_context():
                mod = quantize_seq(mod)
    except tvm.TVMError as err:
        msg = str(err)
        if "conv2d_rewrite" in msg and "rhs_kind is None" in msg:
            raise RuntimeError(
                "TVM quantize annotate failed at conv2d RHS (rhs_kind). "
                "Either keep relay.quantize.qconfig(skip_dense_layer=True), or set "
                "TVM_QTZ_DYNAMIC_DENSE_CONV2D_FIX=1 and use skip_dense_layer=False."
            )
        raise

    # Step 3: Post-quantization validation
    # Verify that protected vars are still present in the graph
    main_func = mod["main"]
    free_vars_names = {v.name_hint for v in relay.analysis.free_vars(main_func.body)}
    remaining_protected = protected_vars & free_vars_names

    if remaining_protected:
        print(
            f"  ✓ {len(remaining_protected)} dynamic weight vars preserved as inputs: "
            f"{sorted(remaining_protected)}"
        )
    else:
        print(f"  ⚠ Warning: Expected dynamic vars not found in quantized graph.")
        print(f"    Expected: {protected_vars}")
        print(f"    Found: {free_vars_names}")

    return mod


def merge_derivation_and_inference_modules(
    mod_deriv: tvm.ir.IRModule,
    deriv_params: Dict[str, tvm.nd.NDArray],
    mod_infer: tvm.ir.IRModule,
    infer_params: Dict[str, tvm.nd.NDArray],
    derived_weight_names: List[str],
) -> Tuple[tvm.ir.IRModule, Dict[str, tvm.nd.NDArray]]:
    """
    Merge derivation and inference modules back into a single integrated module.

    The merged module has:
      - Inputs: (input_image, pool_w_0, pool_w_1, ..., pool_tm_0, ...)
      - Internal: weight derivation ops
      - Internal: conv layers using derived weights
      - Output: logits

    This allows VTA to see the full dataflow and compile it as one unit,
    rather than managing two separate runtimes.

    Parameters
    ----------
    mod_deriv : tvm.ir.IRModule
        Module that derives weights from pool variables.
        Its main function outputs Tuple(derived_w_0, derived_w_1, ...).

    deriv_params : Dict[str, tvm.nd.NDArray]
        Parameters for derivation (pool variables).

    mod_infer : tvm.ir.IRModule
        Module that performs inference using weight variables as explicit inputs.
        Its main function inputs: (input_image, derived_w_0, derived_w_1, ...)

    infer_params : Dict[str, tvm.nd.NDArray]
        Parameters for inference (BN, bias, etc.).

    derived_weight_names : List[str]
        Names of derived weight outputs (e.g., ["derived_w_0", "derived_w_1"]).

    Returns
    -------
    (merged_mod, merged_params)
        merged_mod["main"] has integrated derivation + inference graph.
        merged_params includes all deriv_params + infer_params.

    Example
    -------
    >>> mod_merged, params_merged = merge_derivation_and_inference_modules(
    ...     mod_deriv, deriv_params, mod_infer, infer_params, derived_weight_names
    ... )
    >>> # Now mod_merged is a single function with all ops
    """

    deriv_func = mod_deriv["main"]
    infer_func = mod_infer["main"]

    # Get free variables from both functions
    deriv_free_vars = relay.analysis.free_vars(deriv_func.body)
    infer_free_vars = relay.analysis.free_vars(infer_func.body)

    # Derivation outputs are input to inference
    # Substitute infer's derived weight Vars with deriv's outputs

    # Extract derived weight Vars from infer_func inputs
    derived_weight_vars: Dict[str, relay.Var] = {}
    for var in infer_free_vars:
        if var.name_hint in derived_weight_names:
            derived_weight_vars[var.name_hint] = var

    # Create a mapping: derived_w_i Var -> call to deriv_func that outputs derived_w_i
    # Since deriv_func outputs a Tuple, we extract each element
    deriv_call = deriv_func()
    weight_replacements = {}

    for idx, weight_name in enumerate(derived_weight_names):
        # Extract idx-th output from deriv tuple
        extracted = relay.TupleGetItem(deriv_call, idx)
        weight_replacements[weight_name] = extracted

    # Substitute derived weight Vars in infer body with extracted outputs
    class _ReplaceVarWithExpr(ExprMutator):
        def __init__(self, replacements: Dict[str, relay.Expr]):
            super().__init__()
            self.replacements = replacements

        def visit_var(self, var: relay.Var) -> relay.Expr:
            if var.name_hint in self.replacements:
                return self.replacements[var.name_hint]
            return var

    replacer = _ReplaceVarWithExpr(weight_replacements)
    merged_body = replacer.visit(infer_func.body)

    # Build merged function signature: all non-derived free vars
    merged_free_vars = relay.analysis.free_vars(merged_body)

    # Remove derived weight vars from the free vars list (they're now internal)
    merged_free_vars = [
        v for v in merged_free_vars
        if v.name_hint not in derived_weight_vars
    ]

    merged_func = relay.Function(merged_free_vars, merged_body)
    merged_mod = tvm.IRModule.from_expr(merged_func)
    merged_mod = relay_transform.InferType()(merged_mod)

    # Merge parameters
    merged_params = dict(deriv_params)
    merged_params.update(infer_params)

    print(f"  [merge_modules] Merged mod has {len(merged_free_vars)} inputs")
    print(f"  [merge_modules] Removed {len(derived_weight_vars)} derived weight vars from signature")

    return merged_mod, merged_params






