# Option B: True OFA Weight Sharing — Step-by-Step Implementation Plan

## High-Level Goal

Create a single multi-graph executor that:
1. Loads **OFA base weights + transformation matrices ONCE**
2. Compiles **all subnet graphs** (compute graphs, no weights baked in)
3. At inference time: selects a subnet by `model_id`, applies the weight
   transformation for that subnet **on-device or on CPU just-in-time**, and runs

---

## Full System Understanding

### Current Compilation Pipeline (What We're Changing)

```
OFA checkpoint
    |
    v
OFA model loaded in PyTorch        ← load once, currently done per-subnet
    |
    | set_active_subnet(arch)
    v
StaticResNetFromArch               ← standalone PyTorch model per subnet
    |
    | load_weights_from_ofa_checkpoint()
    |   - channel slice: base_weight[:out_ch, :in_ch, :, :]
    |   - kernel crop: center-crop if ks < max_ks
    |   - matrix transform: F.linear(filter, Nto(N-2)_matrix)  ← LEARNT PARAMS
    v
Transformed weights (float32)      ← already baked into StaticResNet
    |
    | torch.jit.trace()
    v
Scripted model (frozen)
    |
    | relay.frontend.from_pytorch()
    v
Relay IRModule + params dict       ← params are float32 transformed weights
    |
    | relay.quantize.quantize()     ← weights become int8
    v
Quantized Relay module
    |
    | graph_pack()                  ← VTA tiling, BLOCK_IN=BLOCK_OUT tiling
    v
Packed relay_prog
    |
    | relay.build()
    v
(graph_json, lib, params_dict)     ← params_dict contains PACKED int8 weights
    |
    | runtime.set_input(**params)   ← upload to FPGA/sim
    v
GraphModule — ready to run
```

### What "Baking In" Means

After `relay.quantize.quantize()` + `graph_pack()` + `relay.build()`:
- Weights are **quantized int8** and **tiled** for VTA
- They live in `params_dict` returned by `relay.build`
- The **graph_json** references these params **by name** (p0, p1, …)
- `runtime.set_input(**params)` uploads them to the device

The connection to OFA base weights **is already lost at the PyTorch level**.
But we can **re-establish** it by intercepting before `pytorch_to_relay`.

---

## Option B: Core Insight

Instead of passing a `StaticResNetFromArch` (which has fixed, transformed weights)
into the TVM pipeline, we will:

1. **Keep the OFA model alive** with its base weights + transform matrices
2. **Modify the compilation flow** so that each subnet graph is compiled with
   **symbolic (placeholder) weights** — or with the correctly-transformed weights
   computed from the live OFA model
3. **At runtime, share weights across subnets** by deduplicating the **post-quantization,
   post-packing** weight tensors (the actual tensors VTA needs)

This is achievable because:
- All subnets share the same OFA checkpoint → same `base_conv.weight` source
- Transform matrices are **the same objects** in the loaded OFA model
- After quantization + packing, **many subnets produce identical weight tensors**
  for layers with the same configuration (same channels, same kernel, same decomp)

---

## Implementation Phases

---

### Phase A: Understand & Verify Weight Identity After Compilation
**Goal**: Confirm that identically-configured layers in different subnets produce
bitwise-identical quantized+packed weight tensors.

#### A.1 — Compile 2 subnets, inspect params dicts
- Take `arch_20250927_180844_0578` (35 params) and `arch_20250927_180844_0455` (32 params)
- These are the two smallest subnets — maximum overlap
- Print the shape + hash of every param tensor from `relay.build` output for both
- Identify which `(pN_in_subnet1, pM_in_subnet2)` pairs are numerically identical

#### A.2 — Build a param identity map
- For each pair of subnets, build a table:
  ```
  subnet1.p0  ↔  subnet2.p3  [IDENTICAL]
  subnet1.p1  ↔  subnet2.p4  [IDENTICAL]
  subnet1.p2  ↔  (unique)
  ```
- Understand the naming: TVM names params `p0`, `p1`, … in graph order
- The same OFA layer with the same config → identical post-compiled weight tensor

#### A.3 — Verify the identity is stable across compilations
- Compile the same subnet twice independently
- Confirm the params are bitwise identical (deterministic compilation)
- This is necessary for the deduplication strategy to be reliable

**Deliverable**: A script `verify_param_identity.py` that outputs a param identity report.

---

### Phase B: OFA-Aware Weight Extractor
**Goal**: Build a component that, given the loaded OFA model and a subnet arch dict,
can produce the SAME transformed float32 weight tensors that `StaticResNet.load_weights()` would produce — but as numpy arrays, without needing a full StaticResNet instantiation.

#### B.1 — `OFAWeightExtractor` class
Location: `graph_switching_phase2/ofa_weight_extractor.py`

```
class OFAWeightExtractor:
    ofa_model: OFADynamicResnetAllMod  ← kept alive, SHARED across all subnets
    
    def get_layer_weight(block_idx, residual_idx, conv_idx, arch) → np.ndarray
        # Mirrors _extract_active_weight_from_dynamic_conv() but:
        # - takes arch config directly
        # - returns numpy (not torch tensor)
        # - is stateless (no side effects on ofa_model)
    
    def get_all_weights_for_subnet(arch) → Dict[str, np.ndarray]
        # Returns ALL weights for a subnet, keyed by a canonical layer ID:
        # e.g. "first_conv", "block0_res0_conv0", "block0_res0_bn0", etc.
        # This canonical key is STABLE across subnets and compilations
```

Key point: the canonical layer ID must be **architecture-aware**, not position-based
(p0, p1, …), because different subnets have different numbers of layers.

#### B.2 — Canonical Layer Naming Scheme
Define a naming convention that identifies a layer by its position in the OFA hierarchy:
```
first_conv_weight
first_bn_weight / first_bn_bias / first_bn_running_mean / first_bn_running_var

block{B}_res{R}_conv{C}_weight      ← conv weight
block{B}_res{R}_bn{C}_weight        ← BN weight
block{B}_res{R}_bn{C}_bias          ← BN bias
block{B}_res{R}_bn{C}_mean          ← BN running_mean
block{B}_res{R}_bn{C}_var           ← BN running_var
block{B}_res{R}_shortcut_weight     ← shortcut conv weight (if exists)

fc_weight / fc_bias
```

These are the **semantic identifiers** — independent of compilation param order.

#### B.3 — Verify OFAWeightExtractor matches StaticResNet
Write a unit test:
- Compile a subnet using StaticResNet (existing flow) → get params dict
- Extract weights using OFAWeightExtractor for the same subnet
- Verify they match numerically (before quantization)

---

### Phase C: Param-to-Canonical Mapping at Compile Time
**Goal**: During TVM compilation, record which `pN` in the graph corresponds to which
canonical layer ID from the OFA hierarchy.

#### C.1 — The Mapping Problem
When `relay.frontend.from_pytorch` converts the scripted model, it assigns names
`p0`, `p1`, `p2`, … to parameters in **graph traversal order**.
This order depends on the subnet architecture (different subnets = different graph order).

We need a reliable way to say: `p7 in subnet_arch_0578 = block1_res0_bn0_weight`.

#### C.2 — Approach: Trace the param names before quantization
After `relay.frontend.from_pytorch()` → we have `params` dict with keys like `p0`, `p1`, …
and `values` as numpy arrays. We also have, from OFAWeightExtractor, a dict with canonical
names and the SAME values. We can match by **value identity** (array comparison):

```python
canonical_weights = extractor.get_all_weights_for_subnet(arch)
relay_params = relay.frontend.from_pytorch(...)  # keys: p0, p1, ...

relay_to_canonical = {}
for pname, pval in relay_params.items():
    for cname, cval in canonical_weights.items():
        if arrays_match(pval, cval):
            relay_to_canonical[pname] = cname
            break
```

This match is done ONCE per subnet at compile time. It produces the mapping we need.

#### C.3 — Handle quantized / packed params
After `relay.build()`, the returned `params` dict contains **quantized int8 packed** tensors.
These have names like `p0`, `p1`, … as well (same names, different values — now int8).
We need the `relay_to_canonical` mapping to survive through quantization.

**Solution**: The graph_json returned by `relay.build()` preserves the param names.
So the mapping `relay_param_name → canonical_name` built in C.2 is valid for the final compiled params too.

#### C.4 — `CompiledModelInfo` dataclass extension
Extend the existing `CompiledModel` dataclass to carry the mapping:
```python
@dataclass
class CompiledModelInfo:
    model_id: str
    arch: Dict
    graph: str              # graph JSON
    lib: tvm.runtime.Module
    params: Dict[str, tvm.nd.NDArray]     # post-build quantized+packed params
    relay_to_canonical: Dict[str, str]    # p0 -> "block1_res0_conv0_weight"
    canonical_to_relay: Dict[str, str]    # inverse
```

---

### Phase D: Global Weight Pool
**Goal**: A single data structure that stores each unique post-quantization weight tensor
ONCE, deduplicated by canonical name.

#### D.1 — `GlobalWeightPool` class
Location: `graph_switching_phase2/global_weight_pool.py`

```
class GlobalWeightPool:
    # canonical_name → weight tensor (int8 packed, numpy)
    pool: Dict[str, np.ndarray]
    
    # subnet_id → {relay_param_name → canonical_name}
    subnet_param_map: Dict[str, Dict[str, str]]
    
    def add_compiled_model(model_info: CompiledModelInfo):
        # For each param in model_info.params:
        #   - Look up its canonical name via model_info.relay_to_canonical
        #   - If canonical name NOT in pool: add it
        #   - If canonical name IS in pool: verify it matches (consistency check)
        # Store the subnet_param_map entry
    
    def get_canonical_weight(canonical_name) → np.ndarray
    
    def get_subnet_weights(subnet_id) → Dict[str, np.ndarray]
        # Returns {relay_param_name: weight} for the given subnet
        # By looking up through the canonical pool
    
    def memory_report() → str
        # Reports: total unique weights, memory saved, dedup ratio
```

#### D.2 — Why This is True Sharing
- `block1_res0_conv0_weight` for subnet_0578 and subnet_0455 (if both use block1 with
  same out_ch=128, in_ch=64, ks=3) produce **bitwise identical quantized tensors**
- They will have **canonical name = `block1_res0_conv0_weight`**
- They will map to the **same entry** in `GlobalWeightPool`
- Only ONE copy of this tensor ever exists in memory / on device

---

### Phase E: Device Weight Upload
**Goal**: Upload each unique weight tensor to the device ONCE.

#### E.1 — `DeviceWeightCache`
Location: `graph_switching_phase2/device_weight_cache.py`

```
class DeviceWeightCache:
    ctx: tvm context
    device_tensors: Dict[str, tvm.nd.NDArray]  # canonical_name → on-device tensor
    
    def upload_from_pool(pool: GlobalWeightPool):
        # For each canonical_name in pool:
        #   Upload once, store tvm.nd.NDArray handle
        # Print: "Uploaded N unique tensors, M bytes total"
    
    def get_device_tensor(canonical_name) → tvm.nd.NDArray
```

#### E.2 — Memory layout consideration
On VTA, weights are in device DRAM. The `tvm.nd.NDArray` handle points to a specific
device memory region. When multiple runtimes call `set_input(param_name, tensor)`,
TVM copies the tensor data into the runtime's own buffer.

**Important**: We need to verify whether `set_input` does a copy or just stores a
reference. If it copies, we can't truly share device memory, but we still benefit
from only uploading each unique tensor once (reduced upload bandwidth).

To be checked:
```
tvm/src/runtime/graph_runtime.cc:  SetInput() → copies into internal buffer
tvm/python/tvm/contrib/graph_runtime.py: set_input() calls _get_input(k).copyfrom(tensor)
```
→ Yes, `set_input` does a `copyfrom` (copies). So runtimes have independent copies.
But we still benefit: upload unique weights once, then `set_input` from the device cache
(device-to-device copy = fast, no PCIe/network transfer per subnet).

---

### Phase F: Multi-Runtime Executor (True Shared Weight Design)
**Goal**: A single executor object that manages N runtimes, all sharing weights from
the `GlobalWeightPool`.

#### F.1 — `MultiGraphExecutor` class
Location: `graph_switching_phase2/multi_graph_executor.py`

```
class MultiGraphExecutor:
    weight_pool: GlobalWeightPool
    device_cache: DeviceWeightCache
    runtimes: Dict[str, GraphModule]    # subnet_id → runtime
    subnet_infos: Dict[str, CompiledModelInfo]
    
    def add_subnet(model_info: CompiledModelInfo):
        # 1. Add to global weight pool (dedup)
        weight_pool.add_compiled_model(model_info)
    
    def finalize():
        # 1. Upload all unique weights to device (once)
        device_cache.upload_from_pool(weight_pool)
        # 2. For each subnet: create GraphModule, set all inputs from device cache
        for subnet_id, model_info in subnet_infos.items():
            runtime = graph_runtime.create(model_info.graph, model_info.remote_lib, ctx)
            for pname, cname in model_info.relay_to_canonical.items():
                device_tensor = device_cache.get_device_tensor(cname)
                runtime.set_input(pname, device_tensor)
            runtimes[subnet_id] = runtime
    
    def run(subnet_id: str, input_data: np.ndarray) → np.ndarray:
        # 1. Set ONLY the input tensor (weights already loaded)
        runtime = runtimes[subnet_id]
        runtime.set_input(input_name, tvm.nd.array(input_data, ctx))
        # 2. Run
        runtime.run()
        # 3. Return output
        return runtime.get_output(0).asnumpy()
    
    def switch_and_run(subnet_id: str, input_data: np.ndarray) → np.ndarray:
        # Fast path: weights pre-loaded, just swap input and run
        return self.run(subnet_id, input_data)
    
    def memory_report()
    def timing_report()
```

#### F.2 — Initialization Flow
```
OFA model loaded (ONCE)
    ↓
OFAWeightExtractor(ofa_model) created (ONCE)
    ↓
For each subnet arch:
    1. canonical_weights = extractor.get_all_weights_for_subnet(arch)
    2. mod, pre_build_params = pytorch_to_relay(StaticResNet(arch), ...)
    3. relay_to_canonical = match_params_to_canonical(pre_build_params, canonical_weights)
    4. relay_prog = apply_quantization_and_packing(mod, pre_build_params, ...)
    5. graph, lib, post_build_params = relay.build(relay_prog, ...)
    6. model_info = CompiledModelInfo(relay_to_canonical=relay_to_canonical, 
                                      params=post_build_params, ...)
    7. executor.add_subnet(model_info)
    ↓
executor.finalize()
    → upload unique weights to device
    → set_input for all runtimes
    ↓
Ready: executor.run(subnet_id, input)
```

---

### Phase G: Integration & Testing

#### G.1 — Unit Tests
- `test_ofa_weight_extractor.py`: verify extracted weights match StaticResNet
- `test_param_mapping.py`: verify relay_to_canonical mapping is correct
- `test_global_weight_pool.py`: verify deduplication works correctly
- `test_device_weight_cache.py`: verify upload + retrieval

#### G.2 — Quick Integration Test
File: `graph_switching_phase2/test_phase2_option_b_quick.py`
- Use 3 subnets from `sa_lam_2.0`
- Build MultiGraphExecutor
- Print memory report (unique weights vs total params)
- Run inference on each subnet
- Verify output shapes are correct

#### G.3 — Full Test
File: `graph_switching_phase2/test_phase2_option_b_full.py`
- All 25 subnets from `sa_lam_2.0`
- Benchmark:
  - Setup time (compilation + upload)
  - Inference time per subnet
  - Total device memory used (unique weights)
  - Compare vs naive approach (25 separate runtimes with duplicate weights)

---

## File Structure

```
graph_switching_phase2/
    ├── ofa_weight_extractor.py      [NEW - Phase B]
    ├── global_weight_pool.py        [NEW - Phase D]
    ├── device_weight_cache.py       [NEW - Phase E]
    ├── multi_graph_executor.py      [REWRITE - Phase F]
    ├── phase2_builder.py            [UPDATE - Phase F.2]
    │
    ├── verify_param_identity.py     [NEW - Phase A]
    ├── test_ofa_weight_extractor.py [NEW - Phase G.1]
    ├── test_param_mapping.py        [NEW - Phase G.1]
    ├── test_global_weight_pool.py   [NEW - Phase G.1]
    ├── test_phase2_option_b_quick.py [NEW - Phase G.2]
    ├── test_phase2_option_b_full.py  [NEW - Phase G.3]
    │
    ├── (keep existing files for reference)
    ├── multi_runtime_executor.py    [OLD - keep for reference]
    ├── phase2_builder.py            [OLD - keep for reference, rename]
    └── shared_param_manager.py      [OLD - superseded]
```

---

## Implementation Order

```
Phase A  →  Phase B  →  Phase B.3 (verify)
                ↓
            Phase C  →  Phase C.4
                ↓
            Phase D  →  Phase D.2
                ↓
            Phase E  →  Phase E.2 (verify set_input behavior)
                ↓
            Phase F  →  Phase G
```

Do NOT skip Phase A. It is the foundation. If compiled weights are NOT bitwise identical
for same-configuration layers, the whole deduplication strategy needs adjustment.

---

## Key Questions to Answer in Phase A

1. **Are quantized weights deterministic?**
   Same subnet compiled twice → same params? (Should be yes)

2. **Are same-config layers in different subnets identical post-compilation?**
   block1_res0_conv0 in subnet_0578 = block1_res0_conv0 in subnet_0455?
   (Should be yes, same OFA base weights, same transformation)

3. **How many unique weight tensors exist across all 25 subnets?**
   This tells us the theoretical memory savings.

4. **Does the param order (p0, p1, ...) change between compilations of the same subnet?**
   (Should not, but must verify for mapping stability)

---

## Expected Memory Savings

Based on Phase 1 analysis:
- 25 subnets, 26-61 params each
- 100% parameter sharing ratio (all params come from same OFA weights)
- BUT: parameters with different shapes (different subnet config) are NOT identical

Conservative estimate:
- 60-70% of parameters are identical across subnets (same layer config)
- Memory savings: 40-60% on device weight storage
- More importantly: **zero per-subnet upload cost** — all weights pre-loaded

---

## What We Are NOT Doing (Scope)

- NOT implementing on-device weight transformation (would require custom TVM ops)
- NOT modifying TVM's quantization pipeline
- NOT changing the PyTorch model structure
- NOT implementing dynamic weight loading during inference

These could be future optimizations if this approach proves the concept.

---

## Definition of Success

✅ All 25 subnets can run inference via a single `MultiGraphExecutor.run(subnet_id, input)`  
✅ Weights for each subnet are pre-loaded at startup (zero upload at inference time)  
✅ Device memory for weights is < sum of all individual subnet weight sizes  
✅ Inference latency matches or beats the current sequential approach  
✅ Switching between subnets takes ~0ms overhead (just change which runtime runs)

