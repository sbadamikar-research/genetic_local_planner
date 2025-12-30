# Training

## Overview

The training pipeline consists of two phases:
1. **GA Training**: Evolve optimal trajectories across diverse scenarios
2. **NN Distillation**: Train neural network to mimic GA behavior

**Output**: `models/planner_policy.onnx` (required for deployment)

## Phase 1: GA Training

### Quick Start

```bash
conda activate plan_ga

python training/train_ga.py \
  --config training/config/ga_config.yaml \
  --output models/checkpoints/ga_trajectories.pkl \
  --num_scenarios 1000 \
  --num_workers 8 \
  --checkpoint_interval 100
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--config` | GA configuration file | Required |
| `--output` | Output file for trajectories | Required |
| `--num_scenarios` | Number of training scenarios | 1000 |
| `--num_workers` | Parallel workers | 8 |
| `--checkpoint_interval` | Save every N scenarios | 100 |
| `--resume` | Resume from checkpoint | None |

### Expected Output

```
Scenario 1/1000: Best fitness=12.3, Goal distance=0.15m, Collision rate=0%
Scenario 2/1000: Best fitness=10.8, Goal distance=0.22m, Collision rate=5%
...
Checkpoint saved: models/checkpoints/ga_trajectories_100.pkl
...
Training complete! Saved 1000 trajectories to models/checkpoints/ga_trajectories.pkl
```

### Configuration

Edit `training/config/ga_config.yaml`:

**Population Parameters**:
```yaml
ga:
  population_size: 100      # Individuals per generation
  elite_size: 10           # Top performers preserved
  mutation_rate: 0.1       # Probability of mutation
  crossover_rate: 0.8      # Probability of crossover
  num_generations: 100     # Generations per scenario
```

**Fitness Weights**:
```yaml
fitness_weights:
  goal_distance: 1.0       # Reaching goal
  collision: 10.0          # Avoiding obstacles
  smoothness: 0.5          # Smooth trajectories
  time_efficiency: 0.3     # Faster is better
```

**Robot Configuration**:
```yaml
robot:
  footprint:               # Vertices in meters
    - [-0.2, -0.2]
    - [0.2, -0.2]
    - [0.2, 0.2]
    - [-0.2, 0.2]
  max_v_x: 1.0
  max_v_y: 0.5             # Set to 0.0 for differential drive
  max_omega: 1.0
```

### Resume from Checkpoint

```bash
python training/train_ga.py \
  --config training/config/ga_config.yaml \
  --output models/checkpoints/ga_trajectories.pkl \
  --resume models/checkpoints/ga_trajectories_500.pkl \
  --num_scenarios 1000
```

### Monitoring

**Check Progress**:
```bash
ls -lh models/checkpoints/
```

**Inspect Checkpoint**:
```python
import pickle
with open('models/checkpoints/ga_trajectories.pkl', 'rb') as f:
    data = pickle.load(f)
print(f"Scenarios: {len(data)}")
print(f"First scenario keys: {data[0].keys()}")
```

## Phase 2: NN Training

### Quick Start

```bash
conda activate plan_ga

python training/train_nn.py \
  --data models/checkpoints/ga_trajectories.pkl \
  --config training/config/nn_config.yaml \
  --output models/planner_policy.onnx \
  --checkpoint models/checkpoints/best_model.pth
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data` | GA trajectories pickle file | Required |
| `--config` | NN configuration file | Required |
| `--output` | Output ONNX model path | Required |
| `--checkpoint` | Save best PyTorch checkpoint | None |
| `--resume` | Resume from checkpoint | None |

### Expected Output

```
Loading dataset from models/checkpoints/ga_trajectories.pkl...
Loaded 1000 trajectories, filtered to 750 (top 75%)
Train: 600 samples, Validation: 150 samples

Epoch 1/50: Train Loss=0.1234, Val Loss=0.1456
Epoch 2/50: Train Loss=0.0987, Val Loss=0.1123
...
New best model! Val Loss: 0.0145
Epoch 50/50: Train Loss=0.0023, Val Loss=0.0145

Exporting to ONNX...
ONNX export successful: models/planner_policy.onnx
Verification passed: PyTorch and ONNX outputs match!
```

### Configuration

Edit `training/config/nn_config.yaml`:

**Model Architecture**:
```yaml
model:
  costmap_size: 50
  num_control_steps: 20
  hidden_dim: 256
  cnn:
    channels: [1, 32, 64, 128]
    kernel_sizes: [5, 3, 3]
    strides: [2, 2, 2]
```

**Training Parameters**:
```yaml
training:
  batch_size: 32
  epochs: 50
  learning_rate: 1e-3
  train_split: 0.8
  filter_percentile: 25    # Use top 75% of trajectories
```

**Early Stopping**:
```yaml
training:
  early_stopping:
    patience: 10
    min_delta: 1e-4
```

## Verification

### Test ONNX Model

```bash
python -c "
import onnx
model = onnx.load('models/planner_policy.onnx')
print('Inputs:', [i.name for i in model.graph.input])
print('Outputs:', [o.name for o in model.graph.output])
"
```

Expected output:
```
Inputs: ['costmap_input', 'robot_state_input', 'goal_relative_input', 'costmap_metadata_input']
Outputs: ['output']
```

### Test Inference

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession('models/planner_policy.onnx')

# Create dummy inputs
costmap = np.random.rand(1, 1, 50, 50).astype(np.float32)
robot_state = np.random.rand(1, 9).astype(np.float32)
goal = np.random.rand(1, 3).astype(np.float32)
metadata = np.array([[0.05, 0.8]], dtype=np.float32)

# Run inference
outputs = session.run(
    None,
    {
        'costmap_input': costmap,
        'robot_state_input': robot_state,
        'goal_relative_input': goal,
        'costmap_metadata_input': metadata
    }
)

print(f"Output shape: {outputs[0].shape}")  # Should be (1, 60)
```

## Troubleshooting

### GA Training Slow

**Issue**: Training takes too long

**Solution**:
- Increase `--num_workers` (match CPU cores)
- Reduce `num_generations` in config
- Reduce `population_size` in config

### Poor Fitness Values

**Issue**: GA not finding good solutions

**Solution**:
- Increase `num_generations`
- Increase `population_size`
- Adjust `fitness_weights` (increase goal_distance weight)
- Check robot velocity limits are reasonable

### NN Training Poor Validation Loss

**Issue**: Validation loss not improving

**Solution**:
- Increase `filter_percentile` (use better GA trajectories)
- Enable data augmentation in config
- Reduce model complexity (smaller `hidden_dim`)
- Increase training data (more GA scenarios)

### ONNX Export Fails

**Issue**: Error during ONNX export

**Solution**:
```bash
# Check PyTorch version
python -c "import torch; print(torch.__version__)"

# Check ONNX version
python -c "import onnx; print(onnx.__version__)"

# Use compatible opset
# Edit nn_config.yaml: onnx.opset_version: 14
```

## Performance Tuning

### Training Time Estimation

| Component | Scenarios/Epochs | CPU Cores | Est. Time |
|-----------|------------------|-----------|-----------|
| GA Training | 1000 scenarios | 8 cores | 4-8 hours |
| NN Training | 50 epochs | Any | 30-60 min |

### Hardware Recommendations

- **CPU**: 8+ cores for parallel GA fitness evaluation
- **RAM**: 16GB recommended (8GB minimum)
- **GPU**: Optional for NN training (minimal benefit)
- **Disk**: 10GB for checkpoints and models

## Next Steps

- **Deploy model**: See [deployment.md](deployment.md)
- **Configure planner**: See [configuration.md](configuration.md)
- **Tune parameters**: Experiment with `ga_config.yaml` and `nn_config.yaml`
