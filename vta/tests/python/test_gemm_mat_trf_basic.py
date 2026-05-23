"""Smoke test for GEMM_Mat_Trf feature - basic correctness validation."""

from __future__ import absolute_import, print_function

import numpy as np
import tvm
from tvm import te

try:
    import vta
    from vta.intrin import gemm_mat_trf
    from vta.runtime_gemm_mat_trf import vta_gemm_mat_trf_cpu
    HAS_VTA = True
except ImportError:
    HAS_VTA = False


def test_cpu_gemm_mat_trf_basic():
    """Test basic CPU-side matrix transform."""
    print("\n[Test] CPU GEMM_Mat_Trf Basic")
    
    # Create simple test case
    batch = 2
    in_dim = 4
    out_dim = 3
    
    # Input blocks: (batch, in_dim)
    input_blocks = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ], dtype=np.int8)
    
    # Transformation matrix: (out_dim, in_dim)
    transform_matrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 1, 1],
    ], dtype=np.int8)
    
    # Expected output (int32 accumulation)
    # out[0, 0] = 1*1 + 2*0 + 3*1 + 4*0 = 4
    # out[0, 1] = 1*0 + 2*1 + 3*0 + 4*1 = 6
    # out[0, 2] = 1*1 + 2*1 + 3*1 + 4*1 = 10
    # out[1, 0] = 5*1 + 6*0 + 7*1 + 8*0 = 12
    # out[1, 1] = 5*0 + 6*1 + 7*0 + 8*1 = 14
    # out[1, 2] = 5*1 + 6*1 + 7*1 + 8*1 = 26
    expected = np.array([
        [4, 6, 10],
        [12, 14, 26],
    ], dtype=np.int32)
    
    # Compute using CPU runtime
    output = vta_gemm_mat_trf_cpu(input_blocks, transform_matrix)
    
    print(f"  Input shape: {input_blocks.shape}, dtype: {input_blocks.dtype}")
    print(f"  Transform shape: {transform_matrix.shape}, dtype: {transform_matrix.dtype}")
    print(f"  Output shape: {output.shape}, dtype: {output.dtype}")
    print(f"  Expected:\n{expected}")
    print(f"  Got:\n{output}")
    
    if np.allclose(output, expected):
        print("  PASS: Output matches expected")
        return True
    else:
        print("  FAIL: Output does not match expected")
        print(f"  Difference:\n{output - expected}")
        return False


def test_cpu_gemm_mat_trf_float():
    """Test CPU-side matrix transform with float32."""
    print("\n[Test] CPU GEMM_Mat_Trf Float32")
    
    batch = 2
    in_dim = 3
    out_dim = 2
    
    input_blocks = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=np.float32)
    
    transform_matrix = np.array([
        [0.5, 0.5, 0.5],
        [1.0, -1.0, 0.5],
    ], dtype=np.float32)
    
    output = vta_gemm_mat_trf_cpu(input_blocks, transform_matrix)
    
    # Expected: out[0,0] = 0.5*1 + 0.5*2 + 0.5*3 = 3.0
    #           out[0,1] = 1.0*1 - 1.0*2 + 0.5*3 = 1.0 - 2.0 + 1.5 = 0.5
    #           out[1,0] = 0.5*4 + 0.5*5 + 0.5*6 = 7.5
    #           out[1,1] = 1.0*4 - 1.0*5 + 0.5*6 = 4.0 - 5.0 + 3.0 = 2.0
    expected = np.array([
        [3.0, 0.5],
        [7.5, 2.0],
    ], dtype=np.float32)
    
    print(f"  Input shape: {input_blocks.shape}, dtype: {input_blocks.dtype}")
    print(f"  Transform shape: {transform_matrix.shape}, dtype: {transform_matrix.dtype}")
    print(f"  Output shape: {output.shape}, dtype: {output.dtype}")
    print(f"  Expected:\n{expected}")
    print(f"  Got:\n{output}")
    
    if np.allclose(output, expected, atol=1e-5):
        print("  PASS: Output matches expected")
        return True
    else:
        print("  FAIL: Output does not match expected")
        print(f"  Difference:\n{np.abs(output - expected)}")
        return False


def test_batch_transforms():
    """Test batch processing of multiple transforms."""
    print("\n[Test] Batch CPU GEMM_Mat_Trf")
    
    from vta.runtime_gemm_mat_trf import batch_vta_gemm_mat_trf_cpu
    
    # Create two transform cases
    input_list = [
        np.array([[1, 2], [3, 4]], dtype=np.int8),
        np.array([[5, 6, 7]], dtype=np.int8),
    ]
    transform_list = [
        np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int8),  # 3x2
        np.array([[2, 1, 0], [1, 1, 1]], dtype=np.int8),    # 2x3
    ]
    
    results = batch_vta_gemm_mat_trf_cpu(input_list, transform_list)
    
    print(f"  Batch processed {len(results)} transforms")
    for i, result in enumerate(results):
        print(f"    Result {i}: shape {result.shape}, dtype {result.dtype}")
    
    if len(results) == 2 and results[0].shape == (2, 3) and results[1].shape == (1, 2):
        print("  PASS: Batch processing correct")
        return True
    else:
        print("  FAIL: Batch processing incorrect")
        return False


def test_vta_intrin_exists():
    """Test that VTA gemm_mat_trf intrinsic is defined."""
    if not HAS_VTA:
        print("\n[Test] VTA intrinsic - SKIPPED (VTA not available)")
        return True
    
    print("\n[Test] VTA gemm_mat_trf Intrinsic")
    
    try:
        env = vta.get_env()
        intrin = gemm_mat_trf(env, mock=True)
        print(f"  Intrinsic created: {intrin}")
        print("  PASS: gemm_mat_trf intrinsic available")
        return True
    except Exception as e:
        print(f"  FAIL: Could not create gemm_mat_trf intrinsic: {e}")
        return False


def run_all_tests():
    """Run all smoke tests."""
    print("=" * 72)
    print("GEMM_Mat_Trf Smoke Tests")
    print("=" * 72)
    
    results = {
        'cpu_basic': test_cpu_gemm_mat_trf_basic(),
        'cpu_float': test_cpu_gemm_mat_trf_float(),
        'batch': test_batch_transforms(),
        'vta_intrin': test_vta_intrin_exists(),
    }
    
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
    
    return all(results.values())


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)


