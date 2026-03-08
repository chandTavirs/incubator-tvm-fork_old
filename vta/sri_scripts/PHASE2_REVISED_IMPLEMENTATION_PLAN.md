# Phase 2: Revised Implementation Plan - True OFA Weight Sharing

## Problem Analysis

### Current Flow (Inefficient)
```
OFA Model (base weights)
    ↓
[For each subnet]
  StaticResNet.load_weights() → Transform weights
    ↓
  PyTorch JIT trace → Captures transformed weights
    ↓
  relay.from_pytorch → Extracts weights as params dict
    ↓
  TVM Compile → Creates graph
    ↓
  runtime.set_input(**params) → Upload weights to device
    ↓
[End for]

Result: 25 separate uploads of transformed weights
```

### Root Cause
**By the time we reach TVM, the connection to OFA base weights is lost.**  
Each subnet has independent transformed weights.

## Revised Approach

### Strategy: Extract and Share at TVM Level

Since TVM compilation requires **already-transformed weights**, we need to:

1. **Transform all subnet weights ONCE** (on CPU)
2. **Deduplicate identical weights** across subnets
3. **Upload unique weights ONCE** to device
4. **Map subnet parameters** to shared device weights

### Implementation

#### Step 1: OFA Weight Extractor

```python
class OFAWeightExtractor:
    """Extract and transform OFA weights for all subnets."""
    
    def __init__(self, ofa_model_path: str):
        self.ofa_model = self._load_ofa_model(ofa_model_path)
        self.base_weights = {}
        self.transform_matrices = {}
        
    def extract_base_weights(self):
        """Extract all base weights from OFA model."""
        for name, module in self.ofa_model.named_modules():
            if isinstance(module, DynamicConv2DAll):
                self.base_weights[name] = module.base_conv.weight.detach().cpu().numpy()
                
                # Extract transform matrices
                for ks_pair in ['7to5', '5to3']:
                    matrix_name = f"{ks_pair}_matrix"
                    if hasattr(module, matrix_name):
                        self.transform_matrices[f"{name}_{ks_pair}"] = \
                            getattr(module, matrix_name).detach().cpu().numpy()
    
    def transform_weights_for_subnet(self, arch: Dict) -> Dict[str, np.ndarray]:
        """Transform OFA base weights for a specific subnet architecture."""
        # This replicates StaticResNet.load_weights logic but returns numpy arrays
        weights = {}
        
        # Transform each layer's weights
        for layer_idx, layer_config in enumerate(self._parse_arch(arch)):
            base_weight = self.base_weights[layer_config['ofa_layer_name']]
            
            # Apply channel slicing
            transformed = base_weight[
                :layer_config['out_channels'],
                :layer_config['in_channels'],
                :, :
            ]
            
            # Apply kernel size transformation
            if layer_config['kernel_size'] < 7:
                transformed = self._apply_kernel_transform(
                    transformed,
                    layer_config['ofa_layer_name'],
                    7,
                    layer_config['kernel_size']
                )
            
            weights[f"layer{layer_idx}_weight"] = transformed
        
        return weights
```

#### Step 2: Weight Deduplication Manager

```python
class SharedWeightManager:
    """Manage weight sharing across multiple subnets."""
    
    def __init__(self):
        self.weight_pool = {}  # {weight_hash: numpy_array}
        self.subnet_weight_map = {}  # {subnet_id: {param_name: weight_hash}}
        
    def add_subnet_weights(self, subnet_id: str, weights: Dict[str, np.ndarray]):
        """Add subnet weights and deduplicate."""
        weight_map = {}
        
        for param_name, weight_array in weights.items():
            # Hash the weight array
            weight_hash = self._hash_array(weight_array)
            
            # Add to pool if new
            if weight_hash not in self.weight_pool:
                self.weight_pool[weight_hash] = weight_array
            
            # Map subnet param to hash
            weight_map[param_name] = weight_hash
        
        self.subnet_weight_map[subnet_id] = weight_map
        
    def _hash_array(self, array: np.ndarray) -> str:
        """Create hash of numpy array."""
        return hashlib.sha256(array.tobytes()).hexdigest()[:16]
    
    def get_unique_weights(self) -> Dict[str, np.ndarray]:
        """Get all unique weights across all subnets."""
        return self.weight_pool
    
    def get_subnet_weight_mapping(self, subnet_id: str) -> Dict[str, str]:
        """Get mapping from subnet param names to shared weight hashes."""
        return self.subnet_weight_map[subnet_id]
```

#### Step 3: Multi-Runtime with Shared Weights

```python
class SharedWeightMultiRuntime:
    """Multi-runtime executor with shared weight pool."""
    
    def __init__(self, remote, ctx):
        self.remote = remote
        self.ctx = ctx
        self.device_weight_pool = {}  # {weight_hash: tvm.nd.NDArray}
        self.runtimes = {}  # {subnet_id: GraphModule}
        self.weight_manager = SharedWeightManager()
        
    def upload_shared_weights(self):
        """Upload unique weights to device ONCE."""
        unique_weights = self.weight_manager.get_unique_weights()
        
        print(f"\nUploading {len(unique_weights)} unique weights to device...")
        for weight_hash, weight_array in unique_weights.items():
            self.device_weight_pool[weight_hash] = tvm.nd.array(weight_array, self.ctx)
        
        print(f"Total unique weights uploaded: {len(self.device_weight_pool)}")
        
    def add_runtime(self, subnet_id: str, graph: str, lib: tvm.runtime.Module):
        """Create runtime and link to shared weights."""
        # Create runtime
        runtime = graph_runtime.create(graph, lib, self.ctx)
        
        # Get weight mapping for this subnet
        weight_mapping = self.weight_manager.get_subnet_weight_mapping(subnet_id)
        
        # Link runtime params to shared device weights
        for param_name, weight_hash in weight_mapping.items():
            shared_weight = self.device_weight_pool[weight_hash]
            runtime.set_input(param_name, shared_weight)
        
        self.runtimes[subnet_id] = runtime
        return runtime
    
    def run_inference(self, subnet_id: str, input_data: np.ndarray) -> np.ndarray:
        """Run inference on specific subnet."""
        runtime = self.runtimes[subnet_id]
        
        # Set ONLY input data (weights already loaded)
        runtime.set_input('input0', tvm.nd.array(input_data, self.ctx))
        
        # Run
        runtime.run()
        
        # Get output
        return runtime.get_output(0).asnumpy()
```

#### Step 4: Integration Flow

```python
def build_shared_weight_executor(
    ofa_model_path: str,
    subnet_architectures: List[Dict],
    remote, ctx, env, target, config
):
    """Build multi-runtime executor with shared OFA weights."""
    
    print("=" * 80)
    print("Phase 2: Building Shared Weight Executor")
    print("=" * 80)
    
    # Step 1: Extract OFA base weights
    print("\n[1/5] Extracting OFA base weights...")
    extractor = OFAWeightExtractor(ofa_model_path)
    extractor.extract_base_weights()
    print(f"  Extracted {len(extractor.base_weights)} base weight tensors")
    
    # Step 2: Transform weights for all subnets
    print("\n[2/5] Transforming weights for all subnets...")
    weight_manager = SharedWeightManager()
    
    for subnet_id, arch in subnet_architectures:
        weights = extractor.transform_weights_for_subnet(arch)
        weight_manager.add_subnet_weights(subnet_id, weights)
    
    unique_count = len(weight_manager.get_unique_weights())
    total_count = sum(len(weight_manager.get_subnet_weight_mapping(sid)) 
                     for sid in [s[0] for s in subnet_architectures])
    
    print(f"  Total parameters: {total_count}")
    print(f"  Unique parameters: {unique_count}")
    print(f"  Sharing ratio: {(1 - unique_count/total_count)*100:.1f}%")
    
    # Step 3: Compile all subnets (graph + lib only, we have weights)
    print("\n[3/5] Compiling subnet graphs...")
    compiled_models = {}
    
    for subnet_id, arch in subnet_architectures:
        # Create StaticResNet but DON'T load weights yet
        subnet_model = create_empty_static_resnet(arch)
        
        # Load weights manually from our transformed set
        subnet_weights = {
            name: weight_manager.weight_pool[weight_hash]
            for name, weight_hash in weight_manager.get_subnet_weight_mapping(subnet_id).items()
        }
        load_weights_into_model(subnet_model, subnet_weights)
        
        # Compile to TVM
        mod, params = pytorch_to_relay(subnet_model, [1, 3, 224, 224], "input0")
        relay_prog = apply_quantization_and_packing(mod, params, env, "resnet18", config)
        graph, lib, params = build_relay_graph(relay_prog, target, env.target_host, params, env, config)
        
        compiled_models[subnet_id] = (graph, lib)
    
    # Step 4: Create shared runtime executor
    print("\n[4/5] Creating shared runtime executor...")
    executor = SharedWeightMultiRuntime(remote, ctx)
    executor.weight_manager = weight_manager
    
    # Step 5: Upload shared weights and create runtimes
    print("\n[5/5] Uploading shared weights...")
    executor.upload_shared_weights()
    
    for subnet_id, (graph, lib) in compiled_models.items():
        # Upload lib
        lib_path = f"graphlib_{subnet_id}.tar"
        lib.export_library(lib_path)
        remote.upload(lib_path)
        remote_lib = remote.load_module(lib_path)
        
        # Create runtime linked to shared weights
        executor.add_runtime(subnet_id, graph, remote_lib)
    
    print("\n✓ Shared weight executor ready!")
    print(f"  Total runtimes: {len(executor.runtimes)}")
    print(f"  Shared weights on device: {len(executor.device_weight_pool)}")
    
    return executor
```

## Expected Results

### Memory Savings

```
Before (separate weights):
  Subnet 1: 35 params × 10KB = 350KB
  Subnet 2: 53 params × 10KB = 530KB
  Subnet 3: 58 params × 10KB = 580KB
  Total: 1,460KB

After (shared weights):
  Unique weights: 70 params × 10KB = 700KB
  Total: 700KB
  
Savings: 52%
```

### Inference Switching Speed

```
Before:
  Switch model → Upload new weights (10-50ms) → Run inference

After:
  Switch model → Run inference (0ms overhead)
```

## Implementation Steps

1. **[DONE]** Understand OFA weight transformation
2. **[TODO]** Create `OFAWeightExtractor` class
3. **[TODO]** Create `SharedWeightManager` class  
4. **[TODO]** Create `SharedWeightMultiRuntime` class
5. **[TODO]** Test with 3 subnets (quick test)
6. **[TODO]** Scale to 25 subnets (full test)
7. **[TODO]** Benchmark memory usage
8. **[TODO]** Benchmark inference switching speed

## Open Questions

1. **Can we modify TVM params after compilation?**
   - Need to verify if we can replace param tensors in a compiled graph
   
2. **How does TVM handle shared memory references?**
   - Need to ensure multiple runtimes can reference same device memory

3. **Does quantization affect weight deduplication?**
   - Quantized weights might be different even if source weights are same

