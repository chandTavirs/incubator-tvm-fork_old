# Phase A: Weight Identity Verification — Summary
_Generated: 2026-03-08 16:05:24_

## Q1 + Q4: Compilation Determinism
### arch_20250927_180844_0578
- **Q1** (bitwise identical across two compilations): ✅ PASS
- **Q4** (param ordering stable): ✅ PASS
- Params compiled: 35

## Q2 + Q3: Cross-Subnet Weight Identity
| Metric | Value |
|---|---|
| Subnets analyzed | 3 |
| Total param tensors | 146 |
| Unique param tensors | 123 |
| Shared tensors (>1 subnet) | 18 |
| Total weight bytes | 250.93 MB |
| Unique weight bytes | 243.70 MB |
| **Memory savings** | **2.9%** |
| Q2 confirmed | ✅ YES |

### Top 10 Most Shared Weight Tensors
| Hash (prefix) | Shape | dtype | Size (KB) | Used by N subnets |
|---|---|---|---|---|
| `4d133f494522` | [32, 1, 1] | float32 | 0.1 | 3 |
| `9428114d6ae3` | [32, 3, 7, 7] | float32 | 18.4 | 3 |
| `f42ee29201e5` | [10, 1024] | float32 | 40.0 | 3 |
| `4b5484b8e351` | [32, 1, 1, 1, 16] | int32 | 2.0 | 3 |
| `07b1ad4305b2` | [10] | float32 | 0.0 | 3 |
| `e7b88aafa24c` | [32, 32, 1, 1, 16, 16] | int8 | 256.0 | 2 |
| `5314ea8c5598` | [32, 1, 1, 1, 16] | int32 | 2.0 | 2 |
| `fb1dbddea102` | [32, 8, 5, 5, 16, 16] | int8 | 1600.0 | 2 |
| `6f28cc36625b` | [32, 8, 5, 5, 16, 16] | int8 | 1600.0 | 2 |
| `afb16330fae0` | [32, 8, 5, 5, 16, 16] | int8 | 1600.0 | 2 |

### Per-Subnet Sharing Breakdown
| Subnet | Total Params | Shared | Private | Sharing % |
|---|---|---|---|---|
| arch_20250927_180844_0578 | 35 | 12 | 23 | 34.3% |
| arch_20250927_180844_0034 | 53 | 13 | 40 | 24.5% |
| arch_20250927_180844_0600 | 58 | 16 | 42 | 27.6% |

## Pairwise Param Match: `arch_20250927_180844_0578` vs `arch_20250927_180844_0034`
- Matched: 7   Unique to arch_20250927_180844_0578: 28

| arch_20250927_180844_0578 param | Shape | Match in arch_20250927_180844_0034 |
|---|---|---|
| `p22` | [16, 32, 7, 7, 16, 16] | _(unique)_ |
| `p1` | [32, 1, 1] | `p1` |
| `p0` | [32, 3, 7, 7] | `p0` |
| `p33` | [10, 1024] | `p51` |
| `p21` | [32, 1, 1, 1, 16] | `p37` |
| `p10` | [4, 2, 1, 1, 16, 16] | _(unique)_ |
| `p32` | [32, 32, 1, 1, 16, 16] | `p47` |
| `p8` | [16, 1, 3, 3, 16, 16] | _(unique)_ |
| `p30` | [64, 1, 1, 1, 16] | _(unique)_ |
| `p16` | [32, 16, 1, 1, 16, 16] | _(unique)_ |
| `p28` | [64, 32, 7, 7, 16, 16] | _(unique)_ |
| `p15` | [32, 1, 1, 1, 16] | _(unique)_ |
| `p4` | [2, 1, 1, 1, 16] | _(unique)_ |
| `p14` | [32, 16, 3, 3, 16, 16] | _(unique)_ |
| `p19` | [32, 8, 5, 5, 16, 16] | _(unique)_ |
| `p17` | [32, 8, 5, 5, 16, 16] | _(unique)_ |
| `p11` | [4, 2, 1, 1, 16, 16] | _(unique)_ |
| `p24` | [16, 32, 7, 7, 16, 16] | _(unique)_ |
| `p27` | [64, 1, 1, 1, 16] | _(unique)_ |
| `p12` | [4, 2, 1, 1, 16, 16] | _(unique)_ |
| `p2` | [2, 1, 3, 3, 16, 16] | _(unique)_ |
| `p29` | [64, 32, 7, 7, 16, 16] | _(unique)_ |
| `p20` | [32, 8, 5, 5, 16, 16] | _(unique)_ |
| `p18` | [32, 8, 5, 5, 16, 16] | _(unique)_ |
| `p31` | [32, 32, 1, 1, 16, 16] | `p46` |
| `p26` | [64, 1, 1, 1, 16] | _(unique)_ |
| `p23` | [16, 32, 7, 7, 16, 16] | _(unique)_ |
| `p3` | [2, 1, 3, 3, 16, 16] | _(unique)_ |
| `p13` | [4, 2, 1, 1, 16, 16] | _(unique)_ |
| `p9` | [16, 1, 1, 1, 16] | _(unique)_ |
| `p25` | [16, 32, 7, 7, 16, 16] | _(unique)_ |
| `p5` | [16, 1, 3, 3, 16, 16] | _(unique)_ |
| `p34` | [10] | `p52` |
| `p6` | [16, 1, 3, 3, 16, 16] | _(unique)_ |
| `p7` | [16, 1, 3, 3, 16, 16] | _(unique)_ |

## Final Verdict for Option B
- Q1 (deterministic compilation):  ✅
- Q2 (identical tensors across subnets): ✅
- Q3 (memory savings potential):  2.9%
- Q4 (stable param ordering):  ✅

**✅ All foundational checks PASSED. Option B is viable. Proceed to Phase B.**
