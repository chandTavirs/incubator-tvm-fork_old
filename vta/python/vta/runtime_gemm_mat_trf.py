"""CPU-side runtime materializer for GEMM_Mat_Trf matrix transforms.

This module provides CPU-side implementations of matrix transform operations
that can be executed before uploading pool variables to the VTA device.
"""

from __future__ import absolute_import, print_function

import numpy as np


def vta_gemm_mat_trf_cpu(input_blocks, transform_matrix, output_buffer=None):
    """CPU-side matrix transform materialization.
    
    Computes: output = input_blocks @ transform_matrix.T
    
    This is useful for materializing dense transform subgraphs on the CPU
    before uploading results to the VTA device, avoiding unnecessary device
    transfers and compute overhead.
    
    Parameters
    ----------
    input_blocks : np.ndarray
        Input tensor shape (batch, in_dim), dtype int8 or float32
    transform_matrix : np.ndarray
        Transformation matrix shape (out_dim, in_dim), dtype int8 or float32
    output_buffer : np.ndarray, optional
        Pre-allocated output buffer. If None, allocates new array.
    
    Returns
    -------
    np.ndarray
        Transformed output shape (batch, out_dim)
    """
    # Ensure inputs are numpy arrays
    inp = np.asarray(input_blocks)
    trf = np.asarray(transform_matrix)
    
    # Validate shapes
    if inp.ndim != 2:
        raise ValueError(f"input_blocks must be 2D, got shape {inp.shape}")
    if trf.ndim != 2:
        raise ValueError(f"transform_matrix must be 2D, got shape {trf.shape}")
    
    batch, in_dim = inp.shape
    out_dim, trf_in_dim = trf.shape
    
    if in_dim != trf_in_dim:
        raise ValueError(
            f"input_blocks in_dim {in_dim} != transform_matrix in_dim {trf_in_dim}"
        )
    
    # Determine output dtype
    if output_buffer is not None:
        out_dtype = output_buffer.dtype
    else:
        # Use int32 for int8 inputs (accumulation), else preserve
        if inp.dtype == np.int8 and trf.dtype == np.int8:
            out_dtype = np.int32
        else:
            out_dtype = np.float32
    
    # Allocate output if not provided
    if output_buffer is None:
        output = np.zeros((batch, out_dim), dtype=out_dtype)
    else:
        output = output_buffer
        if output.shape != (batch, out_dim):
            raise ValueError(
                f"output_buffer shape {output.shape} does not match expected ({batch}, {out_dim})"
            )
    
    # Perform matrix multiply
    # output[i, j] = sum_k(input[i, k] * transform[j, k])
    if inp.dtype == np.int8 and trf.dtype == np.int8:
        # Signed int8 compute with int32 accumulation (like VTA hardware)
        inp_s32 = inp.astype(np.int32)
        trf_s32 = trf.astype(np.int32)
        output[:] = np.dot(inp_s32, trf_s32.T)
    else:
        # Float compute
        inp_f32 = inp.astype(np.float32)
        trf_f32 = trf.astype(np.float32)
        output[:] = np.dot(inp_f32, trf_f32.T)
    
    return output


def batch_vta_gemm_mat_trf_cpu(input_blocks_list, transform_matrices_list, output_buffers=None):
    """Batch CPU-side matrix transforms for multiple (input, transform) pairs.
    
    Parameters
    ----------
    input_blocks_list : list of np.ndarray
        List of input tensors
    transform_matrices_list : list of np.ndarray
        List of transformation matrices
    output_buffers : list of np.ndarray, optional
        List of pre-allocated output buffers
    
    Returns
    -------
    list of np.ndarray
        List of transformed outputs
    """
    if output_buffers is None:
        output_buffers = [None] * len(input_blocks_list)
    
    results = []
    for inp, trf, out_buf in zip(input_blocks_list, transform_matrices_list, output_buffers):
        result = vta_gemm_mat_trf_cpu(inp, trf, out_buf)
        results.append(result)
    
    return results


def quantize_and_transform(input_blocks_fp32, transform_matrix_fp32, input_scale=1.0, trf_scale=1.0):
    """Quantize float32 inputs to int8, then apply matrix transform.
    
    Useful for materializing transforms in quantized graphs where pool variables
    are float32 but need to be quantized before sending to VTA.
    
    Parameters
    ----------
    input_blocks_fp32 : np.ndarray
        Float32 input tensor
    transform_matrix_fp32 : np.ndarray
        Float32 transformation matrix
    input_scale : float
        Quantization scale for input (typically inverse of network quantization scale)
    trf_scale : float
        Quantization scale for transform matrix
    
    Returns
    -------
    np.ndarray
        Transformed output in int32
    """
    # Quantize inputs
    inp_i8 = np.round(np.asarray(input_blocks_fp32) * input_scale).astype(np.int8)
    trf_i8 = np.round(np.asarray(transform_matrix_fp32) * trf_scale).astype(np.int8)
    
    # Apply transform
    return vta_gemm_mat_trf_cpu(inp_i8, trf_i8)

