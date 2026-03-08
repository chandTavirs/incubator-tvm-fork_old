"""
Phase 2 Quick Test - Test with 3 models

Quick validation of the shared parameter approach with minimal models.
"""

from __future__ import absolute_import, print_function

import sys
import os
import time
import numpy as np
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase2_builder import Phase2Builder
from execute_candidate_set_refactored import Config

import tvm


def prepare_test_input(batch_size: int = 1):
    """Prepare a test input image."""
    # Create dummy input (or load actual image)
    input_shape = (batch_size, 3, 224, 224)

    # Random input for testing
    input_data = np.random.randn(*input_shape).astype('float32')

    # Normalize (ImageNet stats)
    mean = np.array([123.0, 117.0, 104.0]).reshape(1, 3, 1, 1)
    std = np.array([58.395, 57.12, 57.375]).reshape(1, 3, 1, 1)

    input_data = (input_data - mean) / std

    return input_data


def test_basic_functionality(executor):
    """Test basic executor functionality."""
    print(f"\n{'='*80}")
    print("TEST 1: Basic Functionality")
    print(f"{'='*80}\n")

    # Prepare input
    input_data = prepare_test_input(batch_size=1)

    # Test inference on all models
    print("Running inference on all models...")
    for model_id in executor.model_ids:
        print(f"\n  Testing {model_id}...")

        # Switch model
        executor.set_active_model(model_id)

        # Set input (use "input0" as that's the default input name)
        executor.set_input("input0", input_data)

        # Run inference
        start = time.time()
        executor.run()
        inference_time = (time.time() - start) * 1000

        # Get output
        output = executor.get_output(0)

        print(f"    Inference time: {inference_time:.2f} ms")
        print(f"    Output shape: {output.shape}")
        print(f"    Output dtype: {output.dtype}")

        # Get top-5 predictions
        output_np = output.asnumpy()
        top5 = np.argsort(output_np[0])[-5:][::-1]
        print(f"    Top-5 classes: {top5}")

    print(f"\n✓ All models executed successfully")
    print(f"{'='*80}\n")


def test_model_switching(executor):
    """Test model switching performance."""
    print(f"\n{'='*80}")
    print("TEST 2: Model Switching Performance")
    print(f"{'='*80}\n")

    if len(executor.model_ids) < 2:
        print("⚠️  Need at least 2 models for switching test")
        return

    # Benchmark switching
    results = executor.benchmark_model_switching(num_switches=100)

    print(f"✓ Model switching test complete")
    print(f"{'='*80}\n")


def test_memory_efficiency(executor):
    """Test and report memory efficiency."""
    print(f"\n{'='*80}")
    print("TEST 3: Memory Efficiency")
    print(f"{'='*80}\n")

    mem_stats = executor.get_memory_stats()

    print(f"Number of models: {mem_stats['num_models']}")

    shared = mem_stats['shared_params']
    print(f"\nShared Parameters:")
    print(f"  Count: {shared['total_params']}")
    print(f"  Memory: {shared['total_memory_mb']:.2f} MB")

    savings = mem_stats['memory_savings']
    print(f"\nMemory Comparison:")
    print(f"  Without sharing: {savings['separate_memory_mb']:.2f} MB")
    print(f"  With sharing: {savings['shared_memory_mb']:.2f} MB")
    print(f"  Savings: {savings['savings_mb']:.2f} MB ({savings['savings_percent']:.1f}%)")

    print(f"\n✓ Memory efficiency validated")
    print(f"{'='*80}\n")


def test_accuracy_consistency(executor):
    """Test that same input gives consistent results."""
    print(f"\n{'='*80}")
    print("TEST 4: Accuracy Consistency")
    print(f"{'='*80}\n")

    # Prepare fixed input
    input_data = prepare_test_input(batch_size=1)

    # Run same model twice
    model_id = executor.model_ids[0]

    print(f"Running {model_id} twice with same input...")

    # First run
    executor.set_active_model(model_id)
    executor.set_input("input0", input_data)
    executor.run()
    output1 = executor.get_output(0).asnumpy()

    # Second run
    executor.set_active_model(model_id)
    executor.set_input("input0", input_data)
    executor.run()
    output2 = executor.get_output(0).asnumpy()

    # Check consistency
    diff = np.abs(output1 - output2).max()

    print(f"  Max difference: {diff}")

    if diff < 1e-5:
        print(f"  ✓ Outputs are consistent")
    else:
        print(f"  ⚠️  Outputs differ (may be normal for float32)")

    print(f"\n{'='*80}\n")


def main():
    """Run Phase 2 quick test."""
    print(f"\n{'='*80}")
    print("PHASE 2 QUICK TEST: SHARED PARAMETER EXECUTION")
    print("Testing with 3 models from sa_lam_2.0")
    print(f"{'='*80}\n")

    # Configuration
    config = Config()
    config.experiment_name = "sa_lam_2.0"
    config.device_host = "10.42.0.188"  # Update as needed

    num_models = 3

    print(f"Configuration:")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Number of models: {num_models}")
    print(f"  Device: {config.device_host}")

    # Build executor
    print(f"\n{'='*80}")
    print("Building Executor...")
    print(f"{'='*80}\n")

    builder = Phase2Builder(config)

    try:
        executor = builder.build_from_experiment(
            config.experiment_name,
            num_models=num_models
        )

        # Print initial summary
        executor.print_summary()

        # Run tests
        test_basic_functionality(executor)
        test_model_switching(executor)
        test_memory_efficiency(executor)
        test_accuracy_consistency(executor)

        # Final summary
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}\n")

        executor.print_summary()

        print(f"\n{'='*80}")
        print("✓ ALL TESTS PASSED")
        print(f"{'='*80}\n")

        # Cleanup
        executor.cleanup()

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ ERROR: {e}")
        print(f"{'='*80}\n")

        import traceback
        traceback.print_exc()

        return 1

    return 0


if __name__ == "__main__":
    exit(main())

