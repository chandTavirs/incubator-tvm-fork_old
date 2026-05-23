"""Integration utilities for GEMM_Mat_Trf matrix transform feature.

This module provides high-level utilities to integrate matrix transform
materialization into existing VTA workflows without modifying core scripts.
"""

from __future__ import absolute_import, print_function

import os
import sys
import numpy as np
from tvm import relay

from .relay_passes.lower_gemm_mat_trf import lower_gemm_mat_trf
from .runtime_gemm_mat_trf import vta_gemm_mat_trf_cpu, batch_vta_gemm_mat_trf_cpu


def identify_transform_matrices(transform_matrices_dict):
    """Identify which pool variables are transformation matrices.
    
    Parameters
    ----------
    transform_matrices_dict : dict
        Dictionary mapping variable names to transformation matrices
    
    Returns
    -------
    set
        Set of variable names that are transformation matrices
    """
    return set(transform_matrices_dict.keys())


def apply_gemm_mat_trf_pass(
    mod,
    transform_var_names=None,
    cpu_materialization=True,
    precomputed=None,
    verbose=False,
):
    """Apply GEMM_Mat_Trf Relay lowering pass to a module.

    Parameters
    ----------
    mod : tvm.IRModule
    transform_var_names : set, optional
        Names of pool vars that are transform matrices (accepts 'pool_X' or 'X').
    cpu_materialization : bool
        True  → substitute relay.const(precomputed[name]) where available,
                 else leave as nn.dense for CPU.
        False → wrap in composite function "vta.gemm_mat_trf" for VTA intrinsic.
    precomputed : dict, optional
        Output of materialize_transforms_cpu: {transform_name -> np.ndarray}.
        Only used when cpu_materialization=True.
    verbose : bool

    Returns
    -------
    tvm.IRModule
    """
    return lower_gemm_mat_trf(
        mod,
        transform_var_names=transform_var_names,
        cpu_materialization=cpu_materialization,
        precomputed=precomputed,
        verbose=verbose,
    )


def materialize_transforms_cpu(
    pool_var_dict,
    transform_matrix_names,
    input_block_slices=None,
    input_scale=1.0,
    transform_scale=1.0,
    verbose=True
):
    """Pre-compute matrix transforms on CPU for specified pool variables.
    
    This function materializes dense-transform subgraphs on the CPU side before
    uploading pool variables to the device, reducing device memory overhead.
    
    Parameters
    ----------
    pool_var_dict : dict
        Dictionary of pool variable name -> tensor array
    transform_matrix_names : set or list
        Names of transformation matrices to use
    input_block_slices : dict, optional
        Mapping of transform names to input block slices for each transform
    input_scale : float
        Quantization scale for input blocks
    transform_scale : float
        Quantization scale for transformation matrices
    verbose : bool
        Whether to print progress information
    
    Returns
    -------
    dict
        Dictionary of materialized transform results
    """
    results = {}
    
    if verbose:
        print(f"[GEMM_Mat_Trf] Materializing {len(transform_matrix_names)} transforms on CPU...")
    
    for trf_name in sorted(transform_matrix_names):
        if trf_name not in pool_var_dict:
            if verbose:
                print(f"  [warn] Transform matrix '{trf_name}' not found in pool; skipping")
            continue
        
        trf_matrix = pool_var_dict[trf_name]
        
        # If we have specific input blocks for this transform, use them
        if input_block_slices and trf_name in input_block_slices:
            inp_blocks = input_block_slices[trf_name]
            output = vta_gemm_mat_trf_cpu(inp_blocks, trf_matrix)
            results[trf_name] = output
            if verbose:
                print(f"  {trf_name}: input{inp_blocks.shape} @ trf{trf_matrix.shape} -> {output.shape}")
        else:
            if verbose:
                print(f"  {trf_name}: trf{trf_matrix.shape} (no input slices provided)")
    
    if verbose:
        print(f"[GEMM_Mat_Trf] Materialized {len(results)} transforms")
    
    return results


def create_gemm_mat_trf_config(
    transform_matrix_dict,
    enable=True,
    cpu_materialization=True,
    verbose=True
):
    """Create a configuration dict for GEMM_Mat_Trf feature usage.
    
    This is useful for passing feature configuration through workflows.
    
    Parameters
    ----------
    transform_matrix_dict : dict
        Dictionary of transformation matrices
    enable : bool
        Whether to enable GEMM_Mat_Trf feature
    cpu_materialization : bool
        Use CPU runtime materialization if enabled
    verbose : bool
        Print debug information
    
    Returns
    -------
    dict
        Configuration dictionary
    """
    return {
        'enable': enable,
        'cpu_materialization': cpu_materialization,
        'transform_var_names': set(transform_matrix_dict.keys()),
        'num_transforms': len(transform_matrix_dict),
        'verbose': verbose,
    }


def maybe_apply_transforms(mod, config, transform_var_dict=None):
    """Conditionally apply GEMM_Mat_Trf feature based on configuration.
    
    Convenience function to apply the feature only if enabled in config.
    
    Parameters
    ----------
    mod : tvm.IRModule
        Relay module
    config : dict
        Configuration dict from create_gemm_mat_trf_config
    transform_var_dict : dict, optional
        Dictionary of transformation matrices (used for materialization)
    
    Returns
    -------
    tuple of (tvm.IRModule, dict or None)
        (transformed_module, materialization_results or None)
    """
    if not config.get('enable', False):
        return mod, None
    
    if config.get('verbose', False):
        print("[GEMM_Mat_Trf] Feature enabled")
    
    # Apply Relay lowering pass
    mod = apply_gemm_mat_trf_pass(
        mod,
        transform_var_names=config.get('transform_var_names'),
        cpu_materialization=config.get('cpu_materialization', True)
    )
    
    # Pre-materialize transforms if dict provided
    materialization_results = None
    if transform_var_dict is not None and config.get('cpu_materialization', False):
        materialization_results = materialize_transforms_cpu(
            transform_var_dict,
            config.get('transform_var_names', set()),
            verbose=config.get('verbose', False)
        )
    
    return mod, materialization_results


# Convenience exports
__all__ = [
    'identify_transform_matrices',
    'apply_gemm_mat_trf_pass',
    'materialize_transforms_cpu',
    'create_gemm_mat_trf_config',
    'maybe_apply_transforms',
]

