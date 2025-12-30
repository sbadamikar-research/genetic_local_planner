# Module 03: Neural Networks for Robot Control

**Estimated Time:** 1 day (6-8 hours)

## 🎯 Learning Objectives

- ✅ Understand CNN architecture for costmap processing
- ✅ Design MLP for state/goal encoding
- ✅ Train models in PyTorch with MSE loss
- ✅ Implement knowledge distillation (GA → NN)
- ✅ Tune network architecture and hyperparameters
- ✅ Understand why NNs are faster than GAs at inference

## Why Distill GA into NN?

| Method | Training Time | Inference Time | Use Case |
|--------|---------------|----------------|----------|
| GA | Hours | Seconds per decision | Training |
| NN | Minutes | Milliseconds | Real-time control |

**Key Insight:** GAs find good solutions (offline), NNs execute them fast (online).

---

## Architecture Overview

```
Inputs (4 separate tensors):
├─ costmap: [batch, 1, 50, 50]  →  CNN Encoder    → [batch, 256]
├─ robot_state: [batch, 9]      →                   
├─ goal_relative: [batch, 3]    →  MLP Encoder    → [batch, 256]
└─ costmap_metadata: [batch, 2] →                   
                                        │
                                        ↓
                               Concatenate: [batch, 512]
                                        │
                                        ↓
                               Policy Head (MLP)
                                        │
                                        ↓
                         Output: [batch, 60] (20 steps × 3 controls)
```

---

## Hands-On Exercises

### Exercise 1: Train a Simple NN (30 min)

First, create synthetic data to test the pipeline:

```bash
cd training/neural_network
python dataset.py  # Creates synthetic data
```

Then train:

```bash
python ../train_nn.py \
  --data /tmp/synthetic_trajectories.pkl \
  --config ../config/nn_config.yaml \
  --output /tmp/test_model.onnx \
  --checkpoint /tmp/test_checkpoint.pth
```

**Questions:**
1. How many epochs until convergence?
2. What's the final train/val loss?
3. Does the model overfit?

### Exercise 2: Visualize CNN Filters (1 hour)

After training, visualize what the CNN learned:

```python
import torch
from training.neural_network.model import PlannerPolicy, create_model
import matplotlib.pyplot as plt

# Load model
config = {...}  # Load from yaml
model = create_model(config)
model.load_state_dict(torch.load('checkpoint.pth'))

# Extract first conv layer
conv1_weights = model.costmap_encoder.conv_layers[0].weight.data
# Shape: [32, 1, 5, 5]

# Plot filters
fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < 32:
        ax.imshow(conv1_weights[i, 0], cmap='gray')
    ax.axis('off')
plt.suptitle('First Layer CNN Filters')
plt.show()
```

**Questions:**
1. Do filters look random or structured?
2. Can you identify edge detectors?
3. What happens if you train longer?

### Exercise 3: Architecture Experiments (2 hours)

Modify `training/config/nn_config.yaml` and compare:

**Experiment 1:** CNN depth
```yaml
# Shallow
cnn:
  channels: [1, 32, 64]
  
# Deep  
cnn:
  channels: [1, 32, 64, 128, 256]
```

**Experiment 2:** Hidden dimensions
```yaml
# Small
hidden_dim: 128

# Large
hidden_dim: 512
```

**Experiment 3:** Dropout
```yaml
training:
  dropout: 0.0   # No regularization
  dropout: 0.3   # Strong regularization
```

Track:
- Train/val loss curves
- Parameter count
- Inference time
- Overfitting behavior

### Exercise 4: Implement Custom Loss (1.5 hours)

Add weighted MSE loss in `training/neural_network/train.py`:

```python
def weighted_mse_loss(predictions, targets):
    """
    Weight recent timesteps more than distant ones.
    Recent controls are executed first, so prioritize accuracy.
    
    Args:
        predictions: [batch, 60]
        targets: [batch, 60]
    
    Returns:
        loss: Weighted MSE
    """
    # Reshape to [batch, 20, 3]
    pred_seq = predictions.view(-1, 20, 3)
    target_seq = targets.view(-1, 20, 3)
    
    # Create weights: [1.0, 0.95, 0.90, ..., 0.05]
    weights = torch.linspace(1.0, 0.05, 20).to(predictions.device)
    weights = weights.view(1, 20, 1)  # Broadcast shape
    
    # Compute weighted MSE
    errors = (pred_seq - target_seq) ** 2
    weighted_errors = errors * weights
    loss = weighted_errors.mean()
    
    return loss
```

Compare with standard MSE.

### Exercise 5: Data Augmentation (1 hour)

Add rotational augmentation in `dataset.py`:

```python
def augment_costmap(costmap, robot_state, goal_relative, angle):
    """
    Rotate costmap, robot state, and goal by angle.
    
    Increases training data diversity.
    """
    # Rotate costmap
    from scipy.ndimage import rotate
    costmap_rot = rotate(costmap, angle, reshape=False, order=1)
    
    # Rotate robot orientation
    robot_state_rot = robot_state.copy()
    robot_state_rot[2] += np.radians(angle)  # theta
    
    # Rotate goal
    goal_rot = goal_relative.copy()
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    goal_rot[0] = cos_a * goal_relative[0] - sin_a * goal_relative[1]
    goal_rot[1] = sin_a * goal_relative[0] + cos_a * goal_relative[1]
    goal_rot[2] += angle_rad
    
    return costmap_rot, robot_state_rot, goal_rot
```

Apply during training for 4x data.

---

## Code Walkthrough

### Key Files

1. **training/neural_network/model.py** (381 lines)
   - `CostmapEncoder`: CNN for spatial features
   - `StateEncoder`: MLP for robot/goal state
   - `PolicyHead`: Output decoder
   - `PlannerPolicy`: Complete model

2. **training/neural_network/dataset.py** (330 lines)
   - `TrajectoryDataset`: Load from pickle
   - Fitness filtering (remove bottom 25%)
   - Train/val splitting
   - Tensor conversion

3. **training/train_nn.py** (main script)
   - Training loop with validation
   - Early stopping
   - Learning rate scheduling
   - ONNX export

---

## Quiz

1. **Why use CNNs for costmaps?**
   a) CNNs are always better
   b) Exploit spatial structure
   c) Faster than MLPs
   d) Required by ONNX

2. **What is knowledge distillation?**
   a) Training NN to mimic another model
   b) Compressing model size
   c) Removing layers
   d) Quantization

3. **Why filter low-fitness trajectories?**
   a) Reduce dataset size
   b) Train on good behaviors only
   c) Speed up training
   d) Prevent overfitting

4. **What does MSE loss measure?**
   a) Classification accuracy
   b) Average squared error
   c) Cross-entropy
   d) Collision rate

5. **Why separate train/val sets?**
   a) Faster training
   b) Detect overfitting
   c) Required by PyTorch
   d) Better accuracy

<details>
<summary><b>Show Answers</b></summary>

1. b) Exploit spatial structure (costmaps are images)
2. a) Training NN to mimic another model (GA in this case)
3. b) Train on good behaviors only (garbage in = garbage out)
4. b) Average squared error
5. b) Detect overfitting (if val loss increases, model memorizing)
</details>

---

## ✅ Checklist

- [ ] Understand CNN+MLP architecture
- [ ] Train model on synthetic data
- [ ] Visualize learned features
- [ ] Experiment with architecture changes
- [ ] Implement custom loss or augmentation
- [ ] Quiz score 80%+

---

## 📚 Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) (interactive visualization)
- [Knowledge Distillation Paper](https://arxiv.org/abs/1503.02531) (Hinton et al.)

---

## 🎉 Next Steps

You can now train neural networks! Next, learn how to export them for C++ deployment.

**→ [Continue to Module 04: ONNX Deployment](../04-onnx-deployment/)**
