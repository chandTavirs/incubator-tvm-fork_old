# OFA Weight Sharing Architecture - Complete Understanding

## Overview

All subnets in the OFA model share the **SAME base weights**. Different subnet weights are **derived** from these base weights through transformations, NOT stored separately.

## How OFA Weight Sharing Works

### 1. Base Weight Storage

```python
# In OFA's DynamicConv2DAll layer:
base_conv.weight  # Shape: [max_out_channels, max_in_channels, max_kernel_size, max_kernel_size]
                  # Example: [512, 512, 7, 7]
```

**All subnets reference this SINGLE tensor.**

### 2. Weight Derivation Process

When a subnet needs weights (e.g., 256 out channels, 128 in channels, 5x5 kernel):

```python
def _extract_active_weight_from_dynamic_conv(dynamic_conv, in_channels, out_channels, kernel_size):
    # Step 1: Extract active channels
    base_weight = dynamic_conv.base_conv.weight
    active_weight = base_weight[:out_channels, :in_channels, :, :]
    
    # Step 2: Extract active kernel size (center crop)
    if kernel_size < max_kernel_size:
        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        active_weight = active_weight[:, :, start:end, start:end]
    
    # Step 3: Apply transformation matrices
    # Transform 7x7 -> 5x5 -> 3x3 if needed
    if kernel_size < max_kernel_size:
        # Reshape for linear transformation
        _filter = active_weight.view(out_channels, in_channels, -1)
        _filter = _filter.view(-1, _filter.size(2))
        
        # Apply learned transformation matrix
        transform_matrix = dynamic_conv.7to5_matrix  # Learned parameter
        _filter = F.linear(_filter, transform_matrix)
        
        # Reshape back
        active_weight = _filter.view(out_channels, in_channels, 5, 5)
    
    return active_weight
```

### 3. Transformation Matrices

**These are learned parameters stored in the OFA model:**

```python
dynamic_conv.7to5_matrix  # Shape: [25, 49] - transforms 7x7 to 5x5
dynamic_conv.5to3_matrix  # Shape: [9, 25] - transforms 5x5 to 3x3
```

**These matrices are shared across ALL layers** with the same kernel size transformation.

## Current Implementation Issues

### Phase 1 Analyzer (WRONG)
```python
# Current: Compares parameter names
shared = set()
for param_name in graph1.params:
    if param_name in graph2.params:
        shared.add(param_name)
```

**Problem**: This doesn't understand that:
- All `p0`, `p1`, `p2` etc. in different graphs are derived from the SAME OFA base weight
- Parameters aren't "shared" - they're **derived transformations** of the same source

### Phase 2 Multi-Runtime (WRONG)
```python
# Current: Loads transformed weights separately per model
for model in models:
    runtime = create_runtime(model.graph, model.lib, ctx)
    runtime.set_input(**model.params)  # Uploads separate copies!
```

**Problem**: This uploads transformed weights separately, missing the opportunity for true sharing.

## Correct Approach for Phase 2

### Option A: Pre-compute All Transformed Weights Once

```python
class OFAWeightManager:
    def __init__(self, ofa_model):
        self.ofa_model = ofa_model
        self.base_weights = {}  # OFA base weights
        self.transform_matrices = {}  # 7to5, 5to3 matrices
        self.subnet_weight_cache = {}  # Cached transformed weights
        
    def extract_base_weights(self):
        """Extract ALL base weights and transformation matrices from OFA model."""
        for name, layer in self.ofa_model.named_modules():
            if isinstance(layer, DynamicConv2DAll):
                self.base_weights[name] = layer.base_conv.weight
                
                # Extract transformation matrices
                if hasattr(layer, '7to5_matrix'):
                    self.transform_matrices['7to5'] = layer.7to5_matrix
                if hasattr(layer, '5to3_matrix'):
                    self.transform_matrices['5to3'] = layer.5to3_matrix
    
    def get_subnet_weights(self, subnet_arch):
        """Get transformed weights for a specific subnet."""
        if subnet_arch.id in self.subnet_weight_cache:
            return self.subnet_weight_cache[subnet_arch.id]
        
        # Transform base weights for this subnet
        weights = {}
        for layer_name, layer_config in subnet_arch.layers:
            base_weight = self.base_weights[layer_name]
            transformed = self._apply_transformation(
                base_weight, 
                layer_config.out_channels,
                layer_config.in_channels,
                layer_config.kernel_size
            )
            weights[layer_name] = transformed
        
        self.subnet_weight_cache[subnet_arch.id] = weights
        return weights
```

### Option B: On-Demand Transformation (Memory Efficient)

```python
class OFAWeightManager:
    def setup_base_weights_on_device(self, remote, ctx):
        """Upload ONLY base weights + transform matrices to device."""
        # Upload base OFA weights (once)
        self.device_base_weights = {}
        for name, weight in self.base_weights.items():
            self.device_base_weights[name] = tvm.nd.array(weight, ctx)
        
        # Upload transform matrices (once)
        self.device_transforms = {}
        for name, matrix in self.transform_matrices.items():
            self.device_transforms[name] = tvm.nd.array(matrix, ctx)
    
    def transform_on_device(self, subnet_arch):
        """Apply transformations ON DEVICE before inference."""
        # This would require custom TVM ops to do the transformation
        pass
```

### Option C: Hybrid (RECOMMENDED)

```python
class SharedOFAExecutor:
    def __init__(self, ofa_model, remote, ctx):
        self.ofa_model = ofa_model
        self.remote = remote
        self.ctx = ctx
        
        # Step 1: Extract base weights ONCE
        self.base_weights = self._extract_ofa_base_weights()
        self.transform_matrices = self._extract_transform_matrices()
        
        # Step 2: Pre-compute ALL subnet weights ONCE (CPU)
        self.subnet_weights = {}  # {subnet_id: {param_name: weight}}
        
        # Step 3: Upload to device ONCE
        self.device_weights = {}  # {param_hash: tvm.nd.NDArray}
        
    def add_subnet(self, subnet_id, arch, compiled_model):
        """Add a subnet and compute/upload its transformed weights."""
        # Compute transformed weights
        weights = self._transform_weights_for_arch(arch)
        self.subnet_weights[subnet_id] = weights
        
        # Upload unique weights to device
        for param_name, weight in weights.items():
            weight_hash = self._hash_array(weight)
            if weight_hash not in self.device_weights:
                self.device_weights[weight_hash] = tvm.nd.array(weight, self.ctx)
        
        # Create runtime
        runtime = graph_runtime.create(compiled_model.graph, compiled_model.lib, self.ctx)
        
        # Link to shared device weights
        for param_name, weight in weights.items():
            weight_hash = self._hash_array(weight)
            runtime.set_input(param_name, self.device_weights[weight_hash])
        
        return runtime
    
    def run_inference(self, subnet_id, input_data):
        """Run inference - weights already loaded!"""
        runtime = self.runtimes[subnet_id]
        runtime.set_input('input0', input_data)  # Only set input
        runtime.run()
        return runtime.get_output(0)
```

## Memory Savings Analysis

### Current Approach (Separate Weights)
```
Total memory = sum(all transformed weights for all subnets)
Example for 25 subnets: 25 × 10MB = 250MB
```

### Shared Base Weights Approach
```
Memory = OFA base weights + transform matrices + unique transformed weights
Example: 50MB (base) + 1MB (matrices) + 50MB (unique transforms) = 101MB
Savings: ~60%
```

### Why Savings Occur:
1. **Many layers share identical configurations** across subnets
2. **Transform matrices are tiny** compared to full weights
3. **Base weights stored once** instead of redundantly

## Action Items

### Phase 1 - Fix Analysis
- [x] Understand that all weights derive from OFA base
- [ ] Analyze actual weight derivation patterns
- [ ] Report: "All subnets share 100% of BASE weights"
- [ ] Report: "X unique transformed weight configurations"

### Phase 2 - Implement Shared Execution
- [ ] Create `OFAWeightManager` class
- [ ] Extract base weights + transformation matrices from OFA
- [ ] Pre-compute transformed weights for all subnets
- [ ] Deduplicate identical transformed weights
- [ ] Upload only unique weights to device
- [ ] Link runtimes to shared device weights
- [ ] Test inference with multiple subnet switching

## Questions Answered

**Q: Should we load the full OFA checkpoint weights once?**
A: YES - Load base weights + transformation matrices once.

**Q: Then each subnet's runtime just references/transforms the relevant portions?**
A: YES - Either pre-transform all and deduplicate, OR transform on-demand.

**Q: This would be true weight sharing rather than copying parameters per model?**
A: YES - Store base weights once, derive subnet weights through transformations.

**Q: Are transform matrices stored in subnet definition?**
A: NO - They're stored in the OFA model's `DynamicConv2DAll` layers.

**Q: Applied during weight loading?**
A: YES - In `StaticResNetFromArch.load_weights_from_ofa_checkpoint()`.

**Q: Need to expose for weight sharing?**
A: YES - We need to extract and reuse them for all subnets.

