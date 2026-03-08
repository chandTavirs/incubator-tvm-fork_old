"""
Aggregate benchmark results from all chunks into final JSON and CSV files.

Usage:
    python aggregate_benchmark_results.py --input_dir benchmark_results --output_dir benchmark_results
"""

import os
import sys
import json
import csv
import argparse
import glob
from typing import List, Dict, Any, Tuple
from collections import defaultdict


def load_chunk_results(chunk_file: str) -> List[Dict[str, Any]]:
    """Load results from a single chunk JSONL file."""
    results = []
    with open(chunk_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                results.append(obj)
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse line in {chunk_file}: {e}")
                continue
    return results


def aggregate_all_chunks(input_dir: str) -> Tuple[List[Dict], Dict]:
    """
    Aggregate results from all chunk files.
    
    Returns:
        (all_results, statistics)
    """
    chunk_files = sorted(glob.glob(os.path.join(input_dir, "benchmark_chunk_*.jsonl")))
    
    if not chunk_files:
        print(f"Error: No chunk files found in {input_dir}")
        return [], {}
    
    print(f"Found {len(chunk_files)} chunk files")
    
    all_results = []
    chunk_stats = []
    
    for chunk_file in chunk_files:
        chunk_name = os.path.basename(chunk_file)
        print(f"  Loading {chunk_name}...")
        
        chunk_results = load_chunk_results(chunk_file)
        all_results.extend(chunk_results)
        
        # Calculate chunk statistics
        successful = [r for r in chunk_results if r.get('success', False)]
        failed = [r for r in chunk_results if not r.get('success', False)]
        
        chunk_stat = {
            'chunk_file': chunk_name,
            'total': len(chunk_results),
            'successful': len(successful),
            'failed': len(failed)
        }
        
        if successful:
            exec_times = [r['mean_exec_time_ms'] for r in successful]
            chunk_stat['min_exec_time_ms'] = min(exec_times)
            chunk_stat['max_exec_time_ms'] = max(exec_times)
            chunk_stat['avg_exec_time_ms'] = sum(exec_times) / len(exec_times)
        
        chunk_stats.append(chunk_stat)
    
    # Calculate overall statistics
    successful_all = [r for r in all_results if r.get('success', False)]
    failed_all = [r for r in all_results if not r.get('success', False)]
    
    statistics = {
        'total_models': len(all_results),
        'successful_models': len(successful_all),
        'failed_models': len(failed_all),
        'success_rate': len(successful_all) / len(all_results) if all_results else 0.0,
        'chunks_processed': len(chunk_files),
        'chunk_statistics': chunk_stats
    }
    
    if successful_all:
        exec_times = [r['mean_exec_time_ms'] for r in successful_all]
        std_times = [r['std_exec_time_ms'] for r in successful_all]
        compile_times = [r['compile_time_s'] for r in successful_all]
        
        statistics['execution_time_stats'] = {
            'min_ms': min(exec_times),
            'max_ms': max(exec_times),
            'avg_ms': sum(exec_times) / len(exec_times),
            'median_ms': sorted(exec_times)[len(exec_times) // 2]
        }
        
        statistics['std_time_stats'] = {
            'min_ms': min(std_times),
            'max_ms': max(std_times),
            'avg_ms': sum(std_times) / len(std_times)
        }
        
        statistics['compile_time_stats'] = {
            'min_s': min(compile_times),
            'max_s': max(compile_times),
            'avg_s': sum(compile_times) / len(compile_times),
            'total_s': sum(compile_times)
        }
    
    return all_results, statistics


def save_aggregated_jsonl(results: List[Dict], output_path: str):
    """Save aggregated results to JSONL file (similar to transfer_meta.jsonl format)."""
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"  ✓ Saved {len(results)} results to {output_path}")


def save_aggregated_csv(results: List[Dict], output_path: str):
    """Save aggregated results to CSV file."""
    if not results:
        print("  Warning: No results to save to CSV")
        return
    
    # Determine all possible fields
    fieldnames = ['model_id', 'mean_exec_time_ms', 'std_exec_time_ms', 
                  'compile_time_s', 'success', 'error']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for result in results:
            row = {
                'model_id': result.get('model_id', ''),
                'mean_exec_time_ms': f"{result.get('mean_exec_time_ms', 0):.4f}" if result.get('success') else 'N/A',
                'std_exec_time_ms': f"{result.get('std_exec_time_ms', 0):.4f}" if result.get('success') else 'N/A',
                'compile_time_s': f"{result.get('compile_time_s', 0):.2f}",
                'success': result.get('success', False),
                'error': result.get('error', '')
            }
            writer.writerow(row)
    
    print(f"  ✓ Saved {len(results)} results to {output_path}")


def save_statistics(statistics: Dict, output_path: str):
    """Save statistics to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"  ✓ Saved statistics to {output_path}")


def print_summary(statistics: Dict):
    """Print summary of aggregated results."""
    print("\n" + "=" * 80)
    print("AGGREGATION SUMMARY")
    print("=" * 80)
    print(f"Total models: {statistics['total_models']}")
    print(f"Successful: {statistics['successful_models']}")
    print(f"Failed: {statistics['failed_models']}")
    print(f"Success rate: {statistics['success_rate']*100:.1f}%")
    print(f"Chunks processed: {statistics['chunks_processed']}")
    
    if 'execution_time_stats' in statistics:
        exec_stats = statistics['execution_time_stats']
        print(f"\nExecution Time Statistics:")
        print(f"  Min: {exec_stats['min_ms']:.2f}ms")
        print(f"  Max: {exec_stats['max_ms']:.2f}ms")
        print(f"  Avg: {exec_stats['avg_ms']:.2f}ms")
        print(f"  Median: {exec_stats['median_ms']:.2f}ms")
    
    if 'compile_time_stats' in statistics:
        compile_stats = statistics['compile_time_stats']
        print(f"\nCompile Time Statistics:")
        print(f"  Min: {compile_stats['min_s']:.2f}s")
        print(f"  Max: {compile_stats['max_s']:.2f}s")
        print(f"  Avg: {compile_stats['avg_s']:.2f}s")
        print(f"  Total: {compile_stats['total_s']:.2f}s ({compile_stats['total_s']/3600:.2f}h)")
    
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Aggregate benchmark results from all chunks')
    parser.add_argument('--input_dir', type=str, default='benchmark_results',
                       help='Input directory containing chunk results')
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                       help='Output directory for aggregated results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Aggregating Benchmark Results")
    print("=" * 80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load and aggregate all chunks
    all_results, statistics = aggregate_all_chunks(args.input_dir)
    
    if not all_results:
        print("Error: No results to aggregate")
        sys.exit(1)
    
    # Sort results by model_id for consistency
    all_results.sort(key=lambda x: x.get('model_id', ''))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save aggregated results
    print("\nSaving aggregated results...")
    
    # JSONL format (similar to transfer_meta.jsonl)
    jsonl_output = os.path.join(args.output_dir, "benchmark_results_all.jsonl")
    save_aggregated_jsonl(all_results, jsonl_output)
    
    # CSV format
    csv_output = os.path.join(args.output_dir, "benchmark_results_all.csv")
    save_aggregated_csv(all_results, csv_output)
    
    # Statistics
    stats_output = os.path.join(args.output_dir, "benchmark_statistics.json")
    save_statistics(statistics, stats_output)
    
    # Save successful and failed models separately
    successful = [r for r in all_results if r.get('success', False)]
    failed = [r for r in all_results if not r.get('success', False)]
    
    if successful:
        success_output = os.path.join(args.output_dir, "benchmark_successful.jsonl")
        save_aggregated_jsonl(successful, success_output)
    
    if failed:
        failed_output = os.path.join(args.output_dir, "benchmark_failed.jsonl")
        save_aggregated_jsonl(failed, failed_output)
    
    # Print summary
    print_summary(statistics)
    
    print("✓ Aggregation complete!")
    print(f"\nOutput files:")
    print(f"  All results (JSONL): {jsonl_output}")
    print(f"  All results (CSV): {csv_output}")
    print(f"  Statistics: {stats_output}")
    if successful:
        print(f"  Successful only: {success_output}")
    if failed:
        print(f"  Failed only: {failed_output}")
    print()


if __name__ == "__main__":
    main()

