# Phase A Enhanced: OFA-Level Weight Sharing Analysis
_Generated: 2026-03-08 16:46:57_

## Summary
This analysis examines weight sharing at the **PyTorch/OFA level** (before quantization and VTA packing) to determine the TRUE memory savings potential.

## Results
| Metric | Value |
|---|---|
| Subnets analyzed | 3 |
| Total weight tensors | 229 |
| Unique weight tensors | 147 |
| Shared tensors (>1 subnet) | 51 |
| Total weight bytes | 1002.74 MB |
| Unique weight bytes | 974.14 MB |
| **Memory savings** | **28.60 MB (2.9%)** |

## Top 20 Most Shared Weight Tensors
| Rank | Hash | Shape | dtype | Size (KB) | Used by N subnets | Example Layer |
|---|---|---|---|---|---|---|
| 1 | `f42ee29201e5` | [10, 1024] | float32 | 40.0 | 3 | `fc.weight` |
| 2 | `0353bcc82457` | [32, 3, 7, 7] | float32 | 18.4 | 3 | `first_conv.weight` |
| 3 | `8816d480cbac` | [1024] | float32 | 4.0 | 3 | `blocks.3.0.main_path.1.weight` |
| 4 | `34013eee1c64` | [1024] | float32 | 4.0 | 3 | `blocks.3.0.main_path.1.bias` |
| 5 | `ae247afa8f71` | [1024] | float32 | 4.0 | 3 | `blocks.3.0.main_path.1.running_mean` |
| 6 | `3ecf90bfc04e` | [1024] | float32 | 4.0 | 3 | `blocks.3.0.main_path.1.running_var` |
| 7 | `69fe87709b5a` | [1024] | float32 | 4.0 | 3 | `blocks.3.0.main_path.4.weight` |
| 8 | `6b8f781a1d16` | [1024] | float32 | 4.0 | 3 | `blocks.3.0.main_path.4.bias` |
| 9 | `ea98ba6eca0f` | [1024] | float32 | 4.0 | 3 | `blocks.3.0.main_path.4.running_mean` |
| 10 | `5da33934228b` | [1024] | float32 | 4.0 | 3 | `blocks.3.0.main_path.4.running_var` |
| 11 | `7803f6297e3d` | [512] | float32 | 2.0 | 3 | `blocks.2.0.main_path.1.weight` |
| 12 | `d5a2d04057c8` | [512] | float32 | 2.0 | 3 | `blocks.2.0.main_path.1.bias` |
| 13 | `dfa8b9dd3a7e` | [512] | float32 | 2.0 | 3 | `blocks.2.0.main_path.1.running_mean` |
| 14 | `ec2db0dfdb1d` | [512] | float32 | 2.0 | 3 | `blocks.2.0.main_path.1.running_var` |
| 15 | `e6eb46b149e9` | [512] | float32 | 2.0 | 3 | `blocks.2.1.main_path.1.weight` |
| 16 | `b6b7d86fe3c9` | [512] | float32 | 2.0 | 3 | `blocks.2.1.main_path.1.bias` |
| 17 | `0387280f8768` | [512] | float32 | 2.0 | 3 | `blocks.2.1.main_path.1.running_mean` |
| 18 | `2fe1bb3bb147` | [512] | float32 | 2.0 | 3 | `blocks.2.1.main_path.1.running_var` |
| 19 | `1d3816b313d4` | [256] | float32 | 1.0 | 3 | `blocks.1.0.main_path.1.weight` |
| 20 | `55d1aff205ee` | [256] | float32 | 1.0 | 3 | `blocks.1.0.main_path.1.bias` |

## Per-Subnet Sharing
| Subnet | Total Tensors | Shared | Private | Sharing % | Total Size (MB) |
|---|---|---|---|---|---|
| arch_20250927_180844_0578 | 55 | 37 | 18 | 67.3% | 342.31 |
| arch_20250927_180844_0034 | 88 | 47 | 41 | 53.4% | 298.15 |
| arch_20250927_180844_0600 | 86 | 49 | 37 | 57.0% | 362.28 |

## Layer Configuration Sharing
Top layer configurations by instance count:

| Rank | Configuration | Instances | Unique Weights | Sharing % |
|---|---|---|---|---|
| 1 | `conv_in3_out32_k7_s2_g1` | 3 | 1 | 66.7% |

## Interpretation
❌ **LOW** — Limited benefit from weight deduplication.

## Next Steps for Option B
⚠️  Consider focusing on Option A (multi-runtime with separate parameters) or optimizing other aspects like graph structure sharing.
