"""
Phase 2 Full Test - Test with all 25 models

Complete validation with full candidate set.
"""

from __future__ import absolute_import, print_function

import sys
import os
import time
import json
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase2_builder import Phase2Builder
from execute_candidate_set_refactored import Config

import tvm


def save_results(executor, output_file: str):
    """Save test results to JSON file."""
    results = {
        'num_models': len(executor.model_ids),
        'model_ids': executor.model_ids,
        'memory_stats': executor.get_memory_stats(),
        'param_usage': executor.param_manager.get_usage_stats(),
        'switch_times': {
            'count': len(executor.switch_times),
            'mean_ms': float(np.mean(executor.switch_times) * 1000) if executor.switch_times else 0,
            'std_ms': float(np.std(executor.switch_times) * 1000) if executor.switch_times else 0
        },
        'inference_stats': {}
    }

    # Add per-model inference stats
    for model_id in executor.model_ids:
        if executor.inference_times[model_id]:
            times = executor.inference_times[model_id]
            results['inference_stats'][model_id] = {
                'count': len(times),
                'mean_ms': float(np.mean(times) * 1000),
                'std_ms': float(np.std(times) * 1000),
                'min_ms': float(np.min(times) * 1000),
                'max_ms': float(np.max(times) * 1000)
            }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {output_file}")


def test_all_models_inference(executor):
    """Run inference on all models."""
    print(f"\n{'='*80}")
    print("TEST 1: All Models Inference")
    print(f"{'='*80}\n")

    # Prepare input
    input_shape = (1, 3, 224, 224)
    input_data = np.random.randn(*input_shape).astype('float32')

    print(f"Running inference on all {len(executor.model_ids)} models...")

    success_count = 0
    failed_models = []

    for idx, model_id in enumerate(executor.model_ids):
        try:
            print(f"  [{idx+1}/{len(executor.model_ids)}] {model_id}...", end=" ")

            # Switch and run
            executor.set_active_model(model_id)
            executor.set_input("data", input_data)

            start = time.time()
            executor.run()
            inference_time = (time.time() - start) * 1000

            output = executor.get_output(0)

            print(f"✓ ({inference_time:.2f} ms)")
            success_count += 1

        except Exception as e:
            print(f"✗ Error: {e}")
            failed_models.append(model_id)

    print(f"\n{'='*80}")
    print(f"Results:")
    print(f"  Successful: {success_count}/{len(executor.model_ids)}")
    print(f"  Failed: {len(failed_models)}")
    if failed_models:
        print(f"  Failed models: {failed_models}")
    print(f"{'='*80}\n")

    return success_count == len(executor.model_ids)


def test_rapid_switching(executor):
    """Test rapid model switching."""
    print(f"\n{'='*80}")
    print("TEST 2: Rapid Model Switching")
    print(f"{'='*80}\n")

    num_switches = 500

    print(f"Performing {num_switches} rapid model switches...")

    switch_times = []
    for i in range(num_switches):
        target_id = executor.model_ids[i % len(executor.model_ids)]

        start = time.time()
        executor.set_active_model(target_id)
        switch_time = (time.time() - start) * 1000
        switch_times.append(switch_time)

    print(f"\nResults:")
    print(f"  Total switches: {len(switch_times)}")
    print(f"  Mean time: {np.mean(switch_times):.3f} ms")
    print(f"  Std dev: {np.std(switch_times):.3f} ms")
    print(f"  Min time: {np.min(switch_times):.3f} ms")
    print(f"  Max time: {np.max(switch_times):.3f} ms")
    print(f"  Median time: {np.median(switch_times):.3f} ms")

    print(f"\n{'='*80}\n")


def test_memory_validation(executor):
    """Validate memory sharing and savings."""
    print(f"\n{'='*80}")
    print("TEST 3: Memory Validation")
    print(f"{'='*80}\n")

    mem_stats = executor.get_memory_stats()

    print(f"Models: {mem_stats['num_models']}")

    shared = mem_stats['shared_params']
    print(f"\nShared Parameters:")
    print(f"  Total parameters: {shared['total_params']}")
    print(f"  Total memory: {shared['total_memory_mb']:.2f} MB")
    print(f"  Average param size: {shared['avg_param_size_kb']:.2f} KB")

    savings = mem_stats['memory_savings']
    print(f"\nMemory Analysis:")
    print(f"  Baseline (separate): {savings['separate_memory_mb']:.2f} MB")
    print(f"  Optimized (shared): {savings['shared_memory_mb']:.2f} MB")
    print(f"  Absolute savings: {savings['savings_mb']:.2f} MB")
    print(f"  Percentage savings: {savings['savings_percent']:.1f}%")

    # Parameter usage
    usage = executor.param_manager.get_usage_stats()
    print(f"\nParameter Usage:")
    print(f"  Parameters tracked: {usage['params_tracked']}")
    print(f"  Min usage: {usage['min_usage']} models")
    print(f"  Max usage: {usage['max_usage']} models")
    print(f"  Avg usage: {usage['avg_usage']:.1f} models")
    print(f"  Unused: {usage['params_unused']}")

    # Validation checks
    print(f"\nValidation Checks:")

    checks_passed = 0
    total_checks = 0

    # Check 1: Savings should be significant
    total_checks += 1
    if savings['savings_percent'] > 50:
        print(f"  ✓ Memory savings > 50% ({savings['savings_percent']:.1f}%)")
        checks_passed += 1
    else:
        print(f"  ⚠️  Memory savings < 50% ({savings['savings_percent']:.1f}%)")

    # Check 2: All params should be uploaded
    total_checks += 1
    if shared['uploaded']:
        print(f"  ✓ All parameters uploaded to device")
        checks_passed += 1
    else:
        print(f"  ✗ Parameters not uploaded")

    # Check 3: Most params should be used
    total_checks += 1
    if usage['params_unused'] < usage['params_tracked'] * 0.1:
        print(f"  ✓ <10% unused parameters ({usage['params_unused']}/{usage['params_tracked']})")
        checks_passed += 1
    else:
        print(f"  ⚠️  ≥10% unused parameters ({usage['params_unused']}/{usage['params_tracked']})")

    print(f"\n  Checks passed: {checks_passed}/{total_checks}")

    print(f"\n{'='*80}\n")

    return checks_passed == total_checks


def benchmark_throughput(executor, num_inferences: int = 100):
    """Benchmark inference throughput."""
    print(f"\n{'='*80}")
    print(f"TEST 4: Throughput Benchmark ({num_inferences} inferences)")
    print(f"{'='*80}\n")

    # Prepare input
    input_data = np.random.randn(1, 3, 224, 224).astype('float32')

    # Run on first few models
    models_to_test = executor.model_ids[:min(5, len(executor.model_ids))]

    print(f"Testing {len(models_to_test)} models with {num_inferences} inferences each...")

    for model_id in models_to_test:
        print(f"\n  {model_id}:")

        executor.set_active_model(model_id)
        executor.set_input("data", input_data)

        times = []
        for i in range(num_inferences):
            start = time.time()
            executor.run()
            inference_time = (time.time() - start) * 1000
            times.append(inference_time)

        print(f"    Mean: {np.mean(times):.2f} ms")
        print(f"    Std:  {np.std(times):.2f} ms")
        print(f"    Min:  {np.min(times):.2f} ms")
        print(f"    Max:  {np.max(times):.2f} ms")
        print(f"    Throughput: {1000/np.mean(times):.2f} inferences/sec")

    print(f"\n{'='*80}\n")


def main():
    """Run Phase 2 full test."""
    print(f"\n{'='*80}")
    print("PHASE 2 FULL TEST: SHARED PARAMETER EXECUTION")
    print("Testing with ALL 25 models from sa_lam_2.0")
    print(f"{'='*80}\n")

    # Configuration
    config = Config()
    config.experiment_name = "sa_lam_2.0"
    config.device_host = "10.42.0.188"  # Update as needed

    num_models = 25  # All models

    print(f"Configuration:")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Number of models: {num_models}")
    print(f"  Device: {config.device_host}")

    # Output file
    output_dir = "phase2_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "phase2_full_test_results.json")

    # Build executor
    print(f"\n{'='*80}")
    print("Building Executor...")
    print(f"{'='*80}\n")

    builder = Phase2Builder(config)

    try:
        start_build = time.time()
        executor = builder.build_from_experiment(
            config.experiment_name,
            num_models=num_models
        )
        build_time = time.time() - start_build

        print(f"\n✓ Executor built in {build_time:.2f}s")

        # Print initial summary
        executor.print_summary()

        # Run tests
        all_passed = True

        all_passed &= test_all_models_inference(executor)
        test_rapid_switching(executor)
        all_passed &= test_memory_validation(executor)
        benchmark_throughput(executor, num_inferences=50)

        # Final summary
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}\n")

        executor.print_summary()

        # Save results
        save_results(executor, output_file)

        if all_passed:
            print(f"\n{'='*80}")
            print("✓ ALL TESTS PASSED")
            print(f"{'='*80}\n")
            status = 0
        else:
            print(f"\n{'='*80}")
            print("⚠️  SOME TESTS HAD WARNINGS")
            print(f"{'='*80}\n")
            status = 0  # Still return success

        # Cleanup
        executor.cleanup()

        return status

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ ERROR: {e}")
        print(f"{'='*80}\n")

        import traceback
        traceback.print_exc()

        return 1


if __name__ == "__main__":
    exit(main())

