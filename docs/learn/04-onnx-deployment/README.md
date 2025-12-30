# Module 04: ONNX & Model Deployment

**Estimated Time:** 1 day (6-8 hours)

## 🎯 Learning Objectives

- ✅ Understand what ONNX is and why it's used
- ✅ Export PyTorch models to ONNX format
- ✅ Verify ONNX model correctness (numerical equivalence)
- ✅ Run ONNX inference in Python (test before C++)
- ✅ Inspect ONNX model structure (inputs, outputs, ops)
- ✅ Profile inference performance
- ✅ Debug common ONNX export issues

## What is ONNX?

**ONNX** (Open Neural Network Exchange) = Universal format for neural networks

```
┌────────────────────────────────────────────────────────────┐
│             WHY ONNX?                                       │
│  ┌─────────────────┐       ┌──────────────────────┐        │
│  │  PyTorch Model  │  →→→  │  ONNX File (.onnx)   │  →→→   │
│  │  (Python)       │       │  (Universal Format)  │        │
│  └─────────────────┘       └──────────────────────┘        │
│                                     ↓                       │
│                           ┌────────────────────────┐        │
│                           │  ONNX Runtime          │        │
│                           │  (Python, C++, C#, JS) │        │
│                           └────────────────────────┘        │
│                                                             │
│  Benefits:                                                  │
│  • Platform-independent (Windows/Linux/Mac)                │
│  • Language-independent (Python → C++)                     │
│  • Optimized inference (faster than native PyTorch)       │
│  • Smaller deployment (no training code needed)           │
└────────────────────────────────────────────────────────────┘
```

### ONNX vs Alternatives

| Method | Pros | Cons | Our Choice? |
|--------|------|------|-------------|
| **ONNX** | Cross-platform, optimized, small | Extra export step | ✅ YES |
| PyTorch C++ API | Direct integration | Large binaries, complex | ❌ No |
| TorchScript | PyTorch native | Still Python-dependent | ❌ No |
| TensorFlow Lite | Mobile-optimized | TF ecosystem only | ❌ No |
| Direct Python calls | Easy | Too slow for real-time | ❌ No |

---

## Hands-On Exercises

### Exercise 1: Export Simple Model (30 min)

Create a minimal model and export it:

```python
import torch
import torch.nn as nn

# Define simple model
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)

# Create and export
model = TinyModel()
model.eval()

# Dummy input (required for export)
dummy_input = torch.randn(1, 10)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'tiny_model.onnx',
    input_names=['input'],
    output_names=['output'],
    opset_version=14,
    verbose=True
)

print("✓ Model exported to tiny_model.onnx")
```

**Verify export:**
```python
import onnx

# Load and check
onnx_model = onnx.load('tiny_model.onnx')
onnx.checker.check_model(onnx_model)
print("✓ Model is valid!")

# Print structure
print(onnx.helper.printable_graph(onnx_model.graph))
```

### Exercise 2: Test ONNX Inference (Python) (45 min)

Before going to C++, verify ONNX works in Python:

```python
import torch
import onnxruntime as ort
import numpy as np

# 1. Get PyTorch prediction
model = TinyModel()
model.eval()
test_input = torch.randn(1, 10)

with torch.no_grad():
    pytorch_output = model(test_input).numpy()

print(f"PyTorch output: {pytorch_output}")

# 2. Get ONNX Runtime prediction
ort_session = ort.InferenceSession('tiny_model.onnx')

# Prepare inputs (must be numpy arrays)
ort_inputs = {
    'input': test_input.numpy()
}

# Run inference
ort_outputs = ort_session.run(None, ort_inputs)
onnx_output = ort_outputs[0]

print(f"ONNX output: {onnx_output}")

# 3. Verify numerical equivalence
diff = np.abs(pytorch_output - onnx_output).max()
print(f"Max difference: {diff}")

assert diff < 1e-5, f"Outputs don't match! Diff: {diff}"
print("✓ Outputs match! ONNX export is correct.")
```

### Exercise 3: Export Project Model (1 hour)

Export the actual planner model:

```bash
cd /home/ANT.AMAZON.COM/basancht/plan_ga

# First, train a small model on synthetic data
python training/neural_network/dataset.py  # Creates /tmp/synthetic_trajectories.pkl

python training/train_nn.py \
  --data /tmp/synthetic_trajectories.pkl \
  --config training/config/nn_config.yaml \
  --output models/test_planner.onnx \
  --checkpoint models/test_checkpoint.pth
```

**Verify the export:**
```python
import onnx
import onnxruntime as ort
import numpy as np

# Load model
onnx_model = onnx.load('models/test_planner.onnx')
onnx.checker.check_model(onnx_model)

# Check inputs/outputs
print("Inputs:")
for input_tensor in onnx_model.graph.input:
    print(f"  {input_tensor.name}: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")

print("\nOutputs:")
for output_tensor in onnx_model.graph.output:
    print(f"  {output_tensor.name}: {[d.dim_value for d in output_tensor.type.tensor_type.shape.dim]}")

# Expected:
# Inputs:
#   costmap_input: [1, 1, 50, 50]
#   robot_state_input: [1, 9]
#   goal_relative_input: [1, 3]
#   costmap_metadata_input: [1, 2]
# Outputs:
#   output: [1, 60]
```

### Exercise 4: Profile Inference Speed (1 hour)

Compare PyTorch vs ONNX Runtime performance:

```python
import torch
import onnxruntime as ort
import numpy as np
import time

# Load PyTorch model
from training.neural_network.model import create_model
import yaml

with open('training/config/nn_config.yaml') as f:
    config = yaml.safe_load(f)

pytorch_model = create_model(config)
pytorch_model.load_state_dict(torch.load('models/test_checkpoint.pth'))
pytorch_model.eval()

# Load ONNX model
ort_session = ort.InferenceSession('models/test_planner.onnx')

# Create test inputs
batch_size = 1
costmap = torch.randn(batch_size, 1, 50, 50)
robot_state = torch.randn(batch_size, 9)
goal_relative = torch.randn(batch_size, 3)
costmap_metadata = torch.randn(batch_size, 2)

# Warmup (JIT compilation)
for _ in range(10):
    with torch.no_grad():
        _ = pytorch_model(costmap, robot_state, goal_relative, costmap_metadata)
    _ = ort_session.run(None, {
        'costmap_input': costmap.numpy(),
        'robot_state_input': robot_state.numpy(),
        'goal_relative_input': goal_relative.numpy(),
        'costmap_metadata_input': costmap_metadata.numpy()
    })

# Benchmark PyTorch
num_runs = 100
start = time.time()
for _ in range(num_runs):
    with torch.no_grad():
        _ = pytorch_model(costmap, robot_state, goal_relative, costmap_metadata)
pytorch_time = (time.time() - start) / num_runs

# Benchmark ONNX
start = time.time()
for _ in range(num_runs):
    _ = ort_session.run(None, {
        'costmap_input': costmap.numpy(),
        'robot_state_input': robot_state.numpy(),
        'goal_relative_input': goal_relative.numpy(),
        'costmap_metadata_input': costmap_metadata.numpy()
    })
onnx_time = (time.time() - start) / num_runs

print(f"PyTorch: {pytorch_time*1000:.2f}ms per inference")
print(f"ONNX: {onnx_time*1000:.2f}ms per inference")
print(f"Speedup: {pytorch_time/onnx_time:.2f}x")
print(f"\nCan we hit 10-20 Hz? (50-100ms budget)")
print(f"  PyTorch: {'✓ YES' if pytorch_time < 0.1 else '✗ NO'}")
print(f"  ONNX: {'✓ YES' if onnx_time < 0.1 else '✗ NO'}")
```

### Exercise 5: Inspect Model with Netron (30 min)

Visualize the ONNX model graphically:

```bash
# Install netron
pip install netron

# Launch visualization
netron models/test_planner.onnx
```

**Tasks:**
1. Find the CNN layers (Conv2d operations)
2. Trace data flow from costmap_input to output
3. Count total number of operations
4. Identify the policy head layers

**Questions:**
- How many Conv2d layers are there? (Should be 3)
- What's the output shape after each conv?
- Where do the 4 inputs get concatenated?

### Exercise 6: Debug Export Issue (1 hour)

Intentionally break the export and fix it:

```python
# broken_export.py
import torch
import torch.nn as nn

class BrokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x):
        # Issue 1: Using Python list (not supported in ONNX)
        results = []
        for i in range(3):
            results.append(self.fc(x))
        
        # Issue 2: Python len() (not supported)
        output_size = len(results)
        
        # Issue 3: Dynamic control flow
        if x.sum() > 0:
            return torch.stack(results).sum(dim=0)
        else:
            return x

model = BrokenModel()
dummy = torch.randn(1, 10)

try:
    torch.onnx.export(model, dummy, 'broken.onnx', opset_version=14)
except Exception as e:
    print(f"Export failed: {e}")
```

**Your task:** Fix the model to export successfully

<details>
<summary><b>Hint</b></summary>

Replace:
- Python loops → vectorized operations
- len() → tensor.shape
- Dynamic control → torch.where()
</details>

<details>
<summary><b>Solution</b></summary>

```python
class FixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x):
        # Fix: Use vectorized operations
        result1 = self.fc(x)
        result2 = self.fc(x)
        result3 = self.fc(x)
        
        # Stack and sum
        stacked = torch.stack([result1, result2, result3])
        summed = stacked.sum(dim=0)
        
        # Fix: Use torch.where instead of if/else
        output = torch.where(
            (x.sum() > 0).unsqueeze(-1),  # Condition
            summed,  # If true
            x  # If false
        )
        
        return output
```
</details>

---

## Code Walkthrough

### ONNX Export in train_nn.py

File: `training/train_nn.py` (line ~250)

```python
def export_to_onnx(model, config, output_path):
    """Export trained model to ONNX format."""
    model.eval()
    
    # Create dummy inputs matching expected shapes
    batch_size = 1
    dummy_costmap = torch.randn(batch_size, 1, 50, 50)
    dummy_robot_state = torch.randn(batch_size, 9)
    dummy_goal = torch.randn(batch_size, 3)
    dummy_metadata = torch.randn(batch_size, 2)
    
    dummy_inputs = (
        dummy_costmap,
        dummy_robot_state,
        dummy_goal,
        dummy_metadata
    )
    
    # Define input/output names (CRITICAL for C++)
    input_names = [
        'costmap_input',
        'robot_state_input',
        'goal_relative_input',
        'costmap_metadata_input'
    ]
    output_names = ['output']
    
    # Export with dynamic axes (allow batch size to vary)
    dynamic_axes = {
        'costmap_input': {0: 'batch'},
        'robot_state_input': {0: 'batch'},
        'goal_relative_input': {0: 'batch'},
        'costmap_metadata_input': {0: 'batch'},
        'output': {0: 'batch'}
    }
    
    torch.onnx.export(
        model,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,  # ONNX Runtime 1.16.3 supports opset 14
        do_constant_folding=True,  # Optimization
        verbose=False
    )
    
    print(f"✓ Model exported to {output_path}")
    
    # Verify export
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verified")
```

**Key Points:**
1. **input_names/output_names MUST match C++ code**
2. **opset_version** determines available operations
3. **dynamic_axes** allows flexible batch sizes
4. **do_constant_folding** = compile-time optimization

---

## Common ONNX Issues & Solutions

### Issue 1: Input Name Mismatch

**Error:**
```
RuntimeError: Input name 'input_0' not found in model
```

**Cause:** C++ code uses different names than ONNX export

**Solution:** Ensure names match exactly:
```python
# Python export
input_names = ['costmap_input', ...]  # Must match C++

# C++ (src/plan_ga_planner/include/plan_ga_planner/onnx_inference.h)
std::vector<std::string> input_names = {"costmap_input", ...};
```

### Issue 2: Unsupported Operation

**Error:**
```
RuntimeError: ONNX export failed: Couldn't export operator aten::some_op
```

**Solution:** Use ONNX-compatible operations only. Check [ONNX operator list](https://github.com/onnx/onnx/blob/main/docs/Operators.md)

Common issues:
- `tensor.item()` → Use indexing instead
- Python loops → Vectorize with torch operations
- `if x > 0: ...` → Use `torch.where()`

### Issue 3: Shape Mismatch

**Error:**
```
Shape mismatch: Expected [1, 60], got [60]
```

**Solution:** Ensure batch dimension is preserved:
```python
# Wrong
output = model(input).squeeze()  # Removes batch dim!

# Correct
output = model(input)  # Keep [batch, 60]
```

---

## Quiz

1. **What is ONNX?**
   a) A Python library
   b) Universal neural network format
   c) C++ compiler
   d) Docker container

2. **Why export to ONNX instead of using PyTorch C++ API?**
   a) Smaller binary size
   b) Cross-platform compatibility
   c) Simpler integration
   d) All of the above

3. **What does `opset_version=14` mean?**
   a) ONNX format version (determines available ops)
   b) Model version number
   c) PyTorch version requirement
   d) Optimization level

4. **Why verify numerical equivalence?**
   a) Ensure export didn't change model behavior
   b) Required by ONNX standard
   c) Improve performance
   d) Reduce model size

5. **What happens if input names don't match?**
   a) Slow inference
   b) Runtime error in C++
   c) Wrong predictions
   d) Model won't load

<details>
<summary><b>Show Answers</b></summary>

1. b) Universal neural network format
2. d) All of the above
3. a) ONNX format version (determines available ops)
4. a) Ensure export didn't change model behavior
5. b) Runtime error in C++ (input names must match exactly)
</details>

---

## ✅ Checklist

- [ ] Understand what ONNX is and why it's needed
- [ ] Successfully export simple model to ONNX
- [ ] Verify ONNX output matches PyTorch
- [ ] Profile inference speed (ONNX vs PyTorch)
- [ ] Inspect model with Netron
- [ ] Debug and fix export issues
- [ ] Quiz score 80%+

---

## 📚 Resources

- [ONNX Official Docs](https://onnx.ai/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTorch ONNX Export Tutorial](https://pytorch.org/docs/stable/onnx.html)
- [Netron Model Visualizer](https://netron.app/)
- [ONNX Operator Support](https://github.com/onnx/onnx/blob/main/docs/Operators.md)

---

## 🎉 Next Steps

You can now export models to ONNX! Next, learn how to use them in C++.

**→ [Continue to Module 05: C++ Integration](../05-cpp-integration/)**
