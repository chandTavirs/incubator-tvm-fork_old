# Phase B: OFA Weight Pool Runtime Implementation Plan

**Date:** March 8, 2026  
**Goal:** Implement VTA runtime that shares OFA base weights across all subnet executions  
**Status:** Planning

---

## 🎯 Core Concept

### Current (Incorrect) Approach
```python
# Each subnet has materialized, independent weights
subnet1 = StaticResNetFromArch(arch1)  # Transforms & stores weights
subnet2 = StaticResNetFromArch(arch2)  # Transforms & stores weights (duplicate memory!)
```

### Target (Correct) Approach
```python
# Single OFA weight pool shared by all subnets
ofa_pool = load_ofa_checkpoint()  # Base weights: conv.weight [max_out, max_in, max_k, max_k]

# Subnet runtimes derive weights dynamically during inference
subnet1_runtime.run(input)  # Uses slice + transform of ofa_pool weights
subnet2_runtime.run(input)  # Uses different slice/transform of SAME ofa_pool weights
```

---

## 📊 Key Insights from OFA Implementation

### 1. Base Weight Storage
```python
# From DynamicConv2DAll.__init__()
self.base_conv = nn.Conv2d(max_in_channel, max_out_channel, max_kernel_size, ...)
# base_conv.weight shape: [max_out, max_in, max_k, max_k]
```

### 2. Transform Matrices
```python
# For kernel size reduction (e.g., 7×7 → 5×5)
# Registered as parameters:
self.register_parameter("7to5_matrix", Parameter(torch.eye(5*5)))  # [25, 25]
self.register_parameter("5to3_matrix", Parameter(torch.eye(3*3)))  # [9, 9]

# Transform operation:
_input_filter = _input_filter.view(-1, src_ks**2)  # [out*in, 49] for 7×7
_output_filter = F.linear(_input_filter, transform_matrix)  # [out*in, 25] for 5×5
_output_filter = _output_filter.view(out_ch, in_ch, target_ks, target_ks)
```

### 3. Weight Derivation Logic (from `get_active_weights()`)
```python
# Step 1: Slice channels
active_weights = base_conv.weight[
    out_start:out_end,    # Output channel slice
    in_start:in_end,      # Input channel slice
    k_start:k_end,        # Kernel height slice
    k_start:k_end         # Kernel width slice
]

# Step 2: Apply kernel transformation (if needed)
if active_kernel_size < max_kernel_size:
    # Iteratively transform from max_kernel → active_kernel
    for src_ks, target_ks in transform_sequence:
        filter = filter.view(-1, src_ks**2)
        filter = F.linear(filter, transform_matrix[f"{src_ks}to{target_ks}"])
        filter = filter.view(out_ch, in_ch, target_ks, target_ks)
```

### 4. Decomposition Types
```python
decompose_type == 0: No decomposition (single conv)
decompose_type == 1: Split output channels in 2 → concat
decompose_type == 2: Split output channels in 4 → concat
decompose_type == 3: Split input channels in 2 → sum
decompose_type == 4: Split input channels in 4 → sum
```

---

## 🏗️ Architecture Design

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     VTA Device (FPGA)                        │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │         OFA Weight Pool (DRAM)                     │    │
│  │  - base_conv_weights[layer_id][max_out, max_in, 7, 7]  │
│  │  - transform_matrices["7to5", "5to3", etc.]        │    │
│  │  - bn_params, biases, etc.                         │    │
│  └────────────────────────────────────────────────────┘    │
│                           ↓                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Subnet 1   │  │  Subnet 2   │  │  Subnet N   │         │
│  │  Runtime    │  │  Runtime    │  │  Runtime    │         │
│  │             │  │             │  │             │         │
│  │  - Graph    │  │  - Graph    │  │  - Graph    │         │
│  │  - Deriv    │  │  - Deriv    │  │  - Deriv    │         │
│  │    Metadata │  │    Metadata │  │    Metadata │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Weight Derivation Metadata (Per Subnet Layer)
```json
{
  "subnet_arch_0578": {
    "layer_configs": [
      {
        "layer_id": "conv1",
        "base_weight": "first_conv.weight",
        "out_channel_slice": [0, 32],
        "in_channel_slice": [0, 3],
        "kernel_size": 7,
        "transform": null,
        "decompose_type": 0
      },
      {
        "layer_id": "blocks.0.conv",
        "base_weight": "blocks.0.conv.weight",
        "out_channel_slice": [0, 64],
        "in_channel_slice": [0, 32],
        "kernel_size": 5,
        "transform": "7to5",
        "decompose_type": 2
      }
    ]
  }
}
```

---

## 🔧 Implementation Steps

### Step 1: OFA Weight Pool Extraction (Host Side)
**File:** `ofa_weight_pool_builder.py`

```python
def extract_ofa_weight_pool(ofa_checkpoint_path):
    """
    Extract base weights and transform matrices from OFA checkpoint.
    
    Returns:
        {
            "base_weights": {
                "first_conv.weight": ndarray[32, 3, 7, 7],
                "blocks.0.conv.weight": ndarray[128, 64, 7, 7],
                ...
            },
            "transform_matrices": {
                "7to5_matrix": ndarray[25, 49],
                "5to3_matrix": ndarray[9, 25],
                ...
            },
            "bn_params": {...},
            "biases": {...}
        }
    """
```

**Implementation:**
1. Load OFA checkpoint
2. Iterate through all layers
3. For DynamicConv2DAll layers:
   - Extract `base_conv.weight` (max size weights)
   - Extract all transform matrices (7to5, 5to3, etc.)
4. For BatchNorm layers: extract weight, bias, running_mean, running_var
5. Return as structured dict

---

### Step 2: Subnet Weight Derivation Metadata (Host Side)
**File:** `subnet_weight_derivation_extractor.py`

```python
def extract_derivation_metadata(ofa_net, subnet_arch):
    """
    For a given subnet architecture, extract how each layer's weights
    should be derived from OFA base weights.
    
    Returns:
        {
            "layer_0": {
                "base_weight_key": "first_conv.weight",
                "out_slice": (0, 32),
                "in_slice": (0, 3),
                "kernel_size": 7,
                "transform_sequence": [],  # No transform needed
                "decompose_type": 0
            },
            "layer_1": {
                "base_weight_key": "blocks.0.conv.weight",
                "out_slice": (0, 64),
                "in_slice": (0, 32),
                "kernel_size": 5,
                "transform_sequence": ["7to5"],
                "decompose_type": 2
            }
        }
    """
```

**Implementation:**
1. Set OFA to active subnet: `ofa_net.set_active_subnet(subnet_arch)`
2. Hook into each DynamicConv2DAll layer's forward pass
3. Capture:
   - `active_in_channel`, `active_out_channel`
   - `active_kernel_size`
   - `decompose_type`
   - Transform sequence (from `get_active_weights()` logic)
4. Map to base weight keys
5. Return metadata dict

---

### Step 3: Relay Graph Modification (Compiler Side)
**File:** `relay_ofa_graph_builder.py`

**Current Relay Graph (Materialized Weights):**
```python
%1 = nn.conv2d(%input, %weight_p0, ...)  # weight_p0 is a constant
```

**Target Relay Graph (Dynamic Derivation):**
```python
# For layer with kernel transform 7→5 and decompose_type=2

# Base weight from OFA pool
%base_weight = var("ofa_pool.blocks.0.conv.weight")  # [128, 64, 7, 7]

# Slice output channels (split into 4 for decompose_type=2)
%slice_out_0 = strided_slice(%base_weight, [0, 0, 0, 0], [32, 64, 7, 7])
%slice_out_1 = strided_slice(%base_weight, [32, 0, 0, 0], [64, 64, 7, 7])
%slice_out_2 = strided_slice(%base_weight, [64, 0, 0, 0], [96, 64, 7, 7])
%slice_out_3 = strided_slice(%base_weight, [96, 0, 0, 0], [128, 64, 7, 7])

# Apply kernel transformation (7×7 → 5×5) to each slice
%transform_matrix = var("ofa_pool.7to5_matrix")  # [25, 49]
%transformed_0 = relay.ofa_kernel_transform(%slice_out_0, %transform_matrix, src_k=7, dst_k=5)
%transformed_1 = relay.ofa_kernel_transform(%slice_out_1, %transform_matrix, src_k=7, dst_k=5)
%transformed_2 = relay.ofa_kernel_transform(%slice_out_2, %transform_matrix, src_k=7, dst_k=5)
%transformed_3 = relay.ofa_kernel_transform(%slice_out_3, %transform_matrix, src_k=7, dst_k=5)

# Perform convolutions
%conv_0 = nn.conv2d(%input, %transformed_0, ...)
%conv_1 = nn.conv2d(%input, %transformed_1, ...)
%conv_2 = nn.conv2d(%input, %transformed_2, ...)
%conv_3 = nn.conv2d(%input, %transformed_3, ...)

# Concatenate outputs (decompose_type=2)
%output = concatenate([%conv_0, %conv_1, %conv_2, %conv_3], axis=1)
```

**Implementation:**
1. Parse subnet relay graph (from quantized model)
2. Identify conv2d ops with constant weight parameters
3. Replace each conv2d with:
   - OFA pool weight variable reference
   - Slice operations (strided_slice)
   - Transform operations (custom relay op or nn.dense)
   - Decomposed convolutions (if needed)
   - Concat/sum operations (if decomposed)
4. Return modified relay graph

---

### Step 4: Custom Relay Op for Kernel Transform
**File:** `tvm/relay/op/ofa_ops.py` (new file)

```python
@relay.op.register("ofa.kernel_transform")
def kernel_transform_compute(attrs, inputs, output_type):
    """
    Transform conv kernel from src_k×src_k to dst_k×dst_k using transform matrix.
    
    Inputs:
        inputs[0]: weight tensor [out_ch, in_ch, src_k, src_k]
        inputs[1]: transform matrix [dst_k*dst_k, src_k*src_k]
    
    Attrs:
        src_k: source kernel size
        dst_k: destination kernel size
    
    Output:
        transformed weight [out_ch, in_ch, dst_k, dst_k]
    """
    # Implementation using relay ops
```

**Alternative:** Use `nn.dense` instead of custom op
```python
# Reshape weight: [out, in, src_k, src_k] → [out*in, src_k*src_k]
%reshaped = reshape(%weight, [-1, src_k*src_k])

# Apply transformation: [out*in, src_k*src_k] @ [dst_k*dst_k, src_k*src_k]^T
%transformed = nn.dense(%reshaped, %transform_matrix)  # [out*in, dst_k*dst_k]

# Reshape back: [out*in, dst_k*dst_k] → [out, in, dst_k, dst_k]
%result = reshape(%transformed, [out_ch, in_ch, dst_k, dst_k])
```

---

### Step 5: VTA Runtime Extension
**File:** `tvm/python/tvm/contrib/ofa_graph_runtime.py` (new file)

```python
class OFAGraphRuntime:
    """
    Extended GraphRuntime that supports OFA weight pool.
    """
    
    def __init__(self, ofa_pool, ctx, remote):
        """
        Args:
            ofa_pool: dict of base weights and transform matrices
            ctx: VTA context
            remote: RPC remote
        """
        self.ofa_pool = {}
        
        # Upload OFA pool to VTA device (once!)
        for key, weight_np in ofa_pool.items():
            self.ofa_pool[key] = tvm.nd.array(weight_np, ctx=ctx)
        
        self.subnets = {}  # model_id → GraphRuntime
    
    def add_subnet(self, model_id, graph_json, lib, params_metadata):
        """
        Add a subnet runtime that references OFA pool.
        
        Args:
            model_id: unique subnet identifier
            graph_json: relay graph JSON (with OFA pool references)
            lib: compiled library
            params_metadata: derivation metadata (not materialized weights!)
        """
        # Create graph runtime
        runtime = graph_runtime.create(graph_json, lib, ctx=self.ctx)
        
        # Set OFA pool as input (all subnets share these!)
        for key, nd_array in self.ofa_pool.items():
            runtime.set_input(key, nd_array)
        
        self.subnets[model_id] = runtime
    
    def run(self, model_id, input_data):
        """
        Execute inference for a specific subnet.
        """
        runtime = self.subnets[model_id]
        runtime.set_input("input", input_data)
        runtime.run()
        return runtime.get_output(0)
```

---

### Step 6: Integration with Existing Pipeline
**File:** `phase_b_ofa_runtime_builder.py`

```python
class OFARuntimeBuilder:
    """
    Builds OFA-aware VTA runtime from candidate set.
    """
    
    def __init__(self, ofa_checkpoint_path, env, remote, ctx):
        self.ofa_checkpoint_path = ofa_checkpoint_path
        self.env = env
        self.remote = remote
        self.ctx = ctx
        
        # Step 1: Extract OFA weight pool
        self.ofa_pool = extract_ofa_weight_pool(ofa_checkpoint_path)
        
        # Step 2: Upload to VTA device
        self.upload_ofa_pool()
    
    def add_subnet(self, model_id, subnet_arch):
        """
        Compile subnet with OFA weight references.
        """
        # 1. Extract derivation metadata
        metadata = extract_derivation_metadata(self.ofa_net, subnet_arch)
        
        # 2. Build relay graph with OFA pool references
        relay_mod = build_relay_graph_with_ofa_refs(subnet_arch, metadata)
        
        # 3. Compile for VTA
        with vta.build_config(disabled_pass={"AlterOpLayout", "FoldConstant"}):
            graph, lib, params = relay.build(relay_mod, target=self.env.target)
        
        # 4. Add to runtime
        self.runtime.add_subnet(model_id, graph, lib, metadata)
```

---

## 🧪 Validation Plan

### Test 1: Single Layer Derivation
**Goal:** Verify transform + slice produces correct weights

```python
# PyTorch reference
ofa_net.set_active_subnet(arch)
pytorch_weight = ofa_net.layers[0].get_active_weights()

# VTA derivation
vta_weight = derive_weight_from_pool(
    ofa_pool["layer_0.weight"],
    metadata["layer_0"]
)

assert np.allclose(pytorch_weight.numpy(), vta_weight.numpy())
```

### Test 2: Single Subnet Inference
**Goal:** Verify full subnet produces correct output

```python
# PyTorch reference
ofa_net.set_active_subnet(arch)
pytorch_out = ofa_net(input)

# VTA with OFA pool
vta_out = ofa_runtime.run("subnet_0", input)

assert np.allclose(pytorch_out.numpy(), vta_out.numpy(), rtol=1e-2)
```

### Test 3: Multi-Subnet Switching
**Goal:** Verify OFA pool is shared across subnets

```python
# Add 3 subnets
ofa_runtime.add_subnet("subnet_0", arch0)
ofa_runtime.add_subnet("subnet_1", arch1)
ofa_runtime.add_subnet("subnet_2", arch2)

# Run all 3
out0 = ofa_runtime.run("subnet_0", input)
out1 = ofa_runtime.run("subnet_1", input)
out2 = ofa_runtime.run("subnet_2", input)

# Verify each matches PyTorch reference
# Verify memory usage (should be ~OFA pool size + 3× small overhead)
```

---

## 📈 Expected Outcomes

### Memory Savings
```
Baseline (Static Models):
- Subnet 1: 350 MB
- Subnet 2: 300 MB  
- Subnet 3: 360 MB
- Total: 1010 MB

OFA Pool Approach:
- OFA Pool: 400 MB (max-size base weights)
- Subnet 1 metadata: 1 MB
- Subnet 2 metadata: 1 MB
- Subnet 3 metadata: 1 MB
- Total: 403 MB

Savings: 60% reduction! (607 MB saved)
```

### Runtime Overhead
- Transform ops: ~2-3ms per layer
- Slice ops: <0.1ms per layer
- Total overhead: ~5ms per inference (acceptable per your requirement)

---

## 🚧 Implementation Challenges

### Challenge 1: Relay Graph Modification
**Issue:** Current compilation folds constants. Need to preserve weight references.

**Solution:** Disable `FoldConstant` pass, use variables for OFA pool.

### Challenge 2: Transform Matrix Application
**Issue:** Relay may not have efficient matrix ops for weight transformation.

**Solution:** Use `nn.dense` or implement custom VTA operator.

### Challenge 3: Decomposed Convolutions
**Issue:** Need to split single conv2d into multiple conv2d + concat/sum.

**Solution:** Graph rewrite pass that expands conv2d based on `decompose_type`.

### Challenge 4: Quantization
**Issue:** OFA weights are float32, VTA needs int8.

**Solution:** Apply quantization AFTER derivation in the relay graph.

---

## 📋 Next Actions

1. **Create proof-of-concept for Step 1** (OFA pool extraction)
2. **Implement Step 2** (derivation metadata extraction)
3. **Test single layer transform** (7×7 → 5×5)
4. **Build relay graph modifier** (Step 3)
5. **Integrate and test end-to-end**

---

**Status:** Ready to implement Phase B 🚀

