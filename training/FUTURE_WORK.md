# Future Work: Advanced NN Training Features

This document outlines advanced features that were deferred from the initial neural network training implementation. These features are designed to improve model quality, robustness, and training efficiency.

**Current Status**: Core training functionality is complete and functional.

**Priority**: Implement these features incrementally as needed based on model performance.

---

## Table of Contents

1. [Data Augmentation](#1-data-augmentation)
2. [Advanced Metrics Tracking](#2-advanced-metrics-tracking)
3. [Simulator Validation During Training](#3-simulator-validation-during-training)
4. [TensorBoard Logging](#4-tensorboard-logging)
5. [Advanced Loss Functions](#5-advanced-loss-functions)
6. [Hyperparameter Tuning](#6-hyperparameter-tuning)
7. [Ensemble Methods](#7-ensemble-methods)
8. [Multi-GA Distillation](#8-multi-ga-distillation)

---

## 1. Data Augmentation

### Overview
Data augmentation artificially increases dataset diversity by applying transformations to training samples. This improves model generalization and reduces overfitting.

### Rotation Augmentation

**Implementation Strategy:**
- Apply random rotations to costmap (90°, 180°, 270°)
- Adjust robot state and goal accordingly:
  - Rotate `theta` in robot_state
  - Rotate `goal_relative` coordinates
  - Rotate control sequence velocities (v_x, v_y)

**Code Structure:**
```python
# training/neural_network/augmentation.py

class RotationAugment:
    def __init__(self, angles=[90, 180, 270]):
        self.angles = angles

    def __call__(self, costmap, robot_state, goal_relative, control_sequence):
        # Randomly select rotation angle
        angle = np.random.choice(self.angles)

        # Rotate costmap
        costmap_rotated = rotate_costmap(costmap, angle)

        # Adjust robot state theta
        robot_state_rotated = robot_state.copy()
        robot_state_rotated[2] += np.deg2rad(angle)  # theta

        # Rotate goal relative position
        goal_rotated = rotate_vector(goal_relative, angle)

        # Rotate control velocities in control_sequence
        control_rotated = rotate_controls(control_sequence, angle)

        return costmap_rotated, robot_state_rotated, goal_rotated, control_rotated
```

**Integration:**
- Wrap `TrajectoryDataset` with augmentation transform
- Apply only during training, not validation
- Use `transform` parameter in dataset constructor

**Benefits:**
- 4× effective dataset size
- Better generalization to different robot orientations
- Reduced overfitting

**Configuration Addition to `nn_config.yaml`:**
```yaml
augmentation:
  rotate: true
  rotation_angles: [90, 180, 270]
```

---

### Gaussian Noise Augmentation

**Implementation Strategy:**
- Add Gaussian noise to costmap during training
- Simulate sensor noise and measurement uncertainty
- Noise applied independently to each pixel

**Code:**
```python
class GaussianNoiseAugment:
    def __init__(self, sigma=5.0):
        self.sigma = sigma

    def __call__(self, costmap):
        # Costmap values are in [0, 1]
        noise = np.random.normal(0, self.sigma / 255.0, costmap.shape)
        costmap_noisy = np.clip(costmap + noise, 0, 1)
        return costmap_noisy
```

**Benefits:**
- Improves robustness to noisy sensor data
- Better real-world deployment performance
- Reduces overfitting to clean training data

**Configuration:**
```yaml
augmentation:
  add_noise: true
  noise_sigma: 5.0  # Std dev for Gaussian noise
```

---

### Flip Augmentation (Omnidirectional Robots Only)

**Note**: Only applicable for omnidirectional robots, not differential drive.

**Implementation:**
- Horizontal flip: mirror costmap, adjust v_y, goal_y
- Vertical flip: mirror costmap, adjust v_x, goal_x

**Configuration:**
```yaml
augmentation:
  flip_horizontal: true  # Set to true for omnidirectional
  flip_vertical: true
```

---

## 2. Advanced Metrics Tracking

### Per-DOF MSE Tracking

**Overview:**
Track Mean Squared Error separately for each control degree of freedom (v_x, v_y, omega). This helps identify which controls are hardest for the model to learn.

**Implementation:**

**Modify `validate_epoch()` in `train_nn.py`:**
```python
def validate_epoch_with_metrics(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    per_dof_errors = {'v_x': [], 'v_y': [], 'omega': []}

    with torch.no_grad():
        for batch in dataloader:
            costmap, robot_state, goal_relative, costmap_metadata, control_target = batch

            # Move to device
            costmap = costmap.to(device)
            # ... (move other tensors)

            # Forward pass
            control_pred = model(costmap, robot_state, goal_relative, costmap_metadata)

            # Overall loss
            loss = criterion(control_pred, control_target)
            total_loss += loss.item()

            # Reshape predictions for per-DOF analysis
            # (batch, 60) -> (batch, 20, 3)
            control_pred_reshaped = control_pred.view(-1, 20, 3)
            control_target_reshaped = control_target.view(-1, 20, 3)

            # Compute per-DOF MSE
            for i, dof in enumerate(['v_x', 'v_y', 'omega']):
                dof_error = (control_pred_reshaped[:, :, i] - control_target_reshaped[:, :, i]).pow(2).mean()
                per_dof_errors[dof].append(dof_error.item())

    # Compute averages
    avg_loss = total_loss / len(dataloader)
    avg_per_dof = {dof: np.mean(errors) for dof, errors in per_dof_errors.items()}

    return avg_loss, avg_per_dof
```

**Logging:**
```python
print(f"  Val Loss:   {val_loss:.6f}")
print(f"    v_x MSE:  {per_dof_mse['v_x']:.6f}")
print(f"    v_y MSE:  {per_dof_mse['v_y']:.6f}")
print(f"    omega MSE: {per_dof_mse['omega']:.6f}")
```

**Benefits:**
- Identify problematic controls (e.g., angular velocity harder to predict)
- Guide architecture changes (e.g., separate heads for different DOFs)
- Enable weighted loss functions

---

### Distribution Analysis

**Track statistics for predictions vs ground truth:**
- Mean and std for each DOF
- Min/max values
- Distribution histograms

**Use Case:**
- Detect if model is biased (e.g., predicting too-low angular velocities)
- Identify if model output range matches target range

---

### Convergence Analysis

**Track and visualize:**
- Training and validation loss curves
- Learning rate schedule over time
- Gradient norms per layer
- Weight update magnitudes

**Implementation:**
- Save metrics to JSON file after each epoch
- Plot with matplotlib at end of training

---

## 3. Simulator Validation During Training

### Overview
Run the trained policy in the Stage simulator every N epochs to evaluate real-world performance. This provides metrics beyond MSE loss.

### Metrics to Track
- **Success rate**: % of navigation goals reached
- **Collision rate**: % of episodes with collisions
- **Path efficiency**: Actual path length / optimal path length
- **Time to goal**: Average time to reach goal
- **Smoothness**: Average control change magnitude

### Implementation Strategy

**Requirements:**
- Implement `training/simulator/stage_wrapper.py` (Stage simulator interface)
- Create test scenarios (different obstacle configurations)
- Define success criteria (distance to goal, no collisions)

**Code Structure:**
```python
# training/simulator/validator.py

class SimulatorValidator:
    def __init__(self, stage_wrapper, num_scenarios=20):
        self.stage = stage_wrapper
        self.num_scenarios = num_scenarios

    def validate_policy(self, model, device):
        """Run policy in simulator and compute metrics."""
        success_count = 0
        collision_count = 0

        for scenario_id in range(self.num_scenarios):
            # Load scenario
            self.stage.load_scenario(scenario_id)

            # Run policy until goal or timeout
            while not done:
                # Get state from simulator
                costmap = self.stage.get_costmap()
                robot_state = self.stage.get_robot_state()
                goal_relative = self.stage.get_goal_relative()

                # Predict control
                with torch.no_grad():
                    control = model(costmap, robot_state, goal_relative, costmap_metadata)

                # Execute first control step
                self.stage.execute_control(control[0:3])

                # Check termination
                if self.stage.goal_reached():
                    success_count += 1
                    break
                if self.stage.collision_detected():
                    collision_count += 1
                    break

        return {
            'success_rate': success_count / self.num_scenarios,
            'collision_rate': collision_count / self.num_scenarios
        }
```

**Integration in `train_nn.py`:**
```python
# After validation epoch, every N epochs
if (epoch + 1) % sim_validation_interval == 0:
    print("  Running simulator validation...")
    sim_metrics = validator.validate_policy(model, device)
    print(f"    Success rate: {sim_metrics['success_rate']:.2%}")
    print(f"    Collision rate: {sim_metrics['collision_rate']:.2%}")
```

**Benefits:**
- Early detection of degenerate policies (e.g., always outputs zero velocity)
- Real-world performance metric alongside MSE loss
- Enables policy-based early stopping (e.g., stop if success rate > 95%)

**Configuration:**
```yaml
validation:
  run_sim_validation: true
  sim_validation_interval: 5  # Every 5 epochs
  sim_num_scenarios: 20
```

---

## 4. TensorBoard Logging

### Overview
TensorBoard provides rich visualizations for training monitoring. It's especially useful for debugging and comparing experiments.

### Scalar Logs
- Training loss per epoch
- Validation loss per epoch
- Learning rate schedule
- Per-DOF MSE (v_x, v_y, omega)
- Simulator metrics (success rate, collision rate)

### Histograms
- Weight distributions per layer
- Gradient norms per layer
- Activation statistics

### Images
- Sample costmap inputs
- Predicted vs ground truth control sequences (line plots)
- Trajectory visualizations in 2D

### Graph Visualization
- Model architecture graph
- Operation dependencies

### Implementation

**Add to `train_nn.py`:**
```python
from torch.utils.tensorboard import SummaryWriter

# Create writer
writer = SummaryWriter(log_dir=f'runs/experiment_{timestamp}')

# In training loop
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('LR', current_lr, epoch)

# Per-DOF metrics
for dof, mse in per_dof_mse.items():
    writer.add_scalar(f'MSE_DOF/{dof}', mse, epoch)

# Histograms
for name, param in model.named_parameters():
    writer.add_histogram(name, param, epoch)
    if param.grad is not None:
        writer.add_histogram(f'{name}.grad', param.grad, epoch)

# Images (example: sample costmap)
sample_costmap = next(iter(val_loader))[0][0]  # First sample
writer.add_image('Costmap/sample', sample_costmap, epoch)

writer.close()
```

**Viewing Logs:**
```bash
tensorboard --logdir=runs
# Open browser to http://localhost:6006
```

**Benefits:**
- Real-time training monitoring
- Easy comparison between experiments
- Debugging tool for training issues
- Publication-quality plots

**Configuration:**
```yaml
logging:
  tensorboard: true
  log_dir: 'runs'
  log_interval: 10  # Log every N batches
```

---

## 5. Advanced Loss Functions

### Huber Loss (Robust to Outliers)

**Motivation:**
MSE is sensitive to outliers in GA data. Huber loss is more robust.

**Implementation:**
```python
criterion = nn.SmoothL1Loss()  # PyTorch's Huber loss
```

**Benefits:**
- Less sensitive to occasional bad GA trajectories
- More stable training

---

### Weighted MSE (Prioritize Critical Controls)

**Motivation:**
Not all controls are equally important. Prioritize angular velocity (omega) for collision avoidance.

**Implementation:**
```python
class WeightedMSELoss(nn.Module):
    def __init__(self, weights={'v_x': 1.0, 'v_y': 1.0, 'omega': 2.0}):
        super().__init__()
        self.weights = weights

    def forward(self, pred, target):
        # Reshape: (batch, 60) -> (batch, 20, 3)
        pred = pred.view(-1, 20, 3)
        target = target.view(-1, 20, 3)

        # Compute weighted MSE
        loss = 0.0
        for i, dof in enumerate(['v_x', 'v_y', 'omega']):
            dof_loss = (pred[:, :, i] - target[:, :, i]).pow(2).mean()
            loss += self.weights[dof] * dof_loss

        return loss
```

**Configuration:**
```yaml
training:
  loss: WeightedMSE
  loss_weights:
    v_x: 1.0
    v_y: 1.0
    omega: 2.0  # Prioritize angular velocity
```

---

### Temporal Consistency Loss

**Motivation:**
Penalize jerky control sequences. Smooth trajectories are safer and more efficient.

**Implementation:**
```python
class TemporalConsistencyLoss(nn.Module):
    def __init__(self, lambda_smooth=0.1):
        super().__init__()
        self.lambda_smooth = lambda_smooth

    def forward(self, pred, target):
        # MSE loss
        mse_loss = (pred - target).pow(2).mean()

        # Smoothness penalty: sum of squared differences between consecutive controls
        pred_reshaped = pred.view(-1, 20, 3)
        smoothness = (pred_reshaped[:, 1:, :] - pred_reshaped[:, :-1, :]).pow(2).mean()

        total_loss = mse_loss + self.lambda_smooth * smoothness
        return total_loss
```

**Configuration:**
```yaml
training:
  loss: TemporalConsistency
  lambda_smooth: 0.1
```

---

### Safety-Aware Loss

**Motivation:**
Penalize controls that would lead to collisions. Use costmap information.

**Implementation:**
```python
def safety_loss(pred_controls, costmap, robot_footprint):
    """
    Compute loss based on predicted collision risk.

    Simulate trajectory with predicted controls and check costmap.
    Penalize trajectories passing through high-cost regions.
    """
    # This requires trajectory simulation during training
    # Complex but potentially very effective
    pass
```

---

## 6. Hyperparameter Tuning

### Parameters to Tune
- **Learning rate**: 1e-2, 1e-3, 1e-4, 1e-5
- **Batch size**: 16, 32, 64, 128
- **Weight decay**: 0, 1e-6, 1e-5, 1e-4
- **Dropout**: 0.0, 0.1, 0.2, 0.3
- **Architecture**:
  - CNN channels: [1, 16, 32, 64] vs [1, 32, 64, 128]
  - MLP hidden dims: [64, 128] vs [128, 256]
  - Policy head depth: 2 vs 3 layers

### Tools

**Ray Tune Integration:**
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def train_with_config(config):
    # Training loop with hyperparameters from config
    pass

search_space = {
    'lr': tune.loguniform(1e-5, 1e-2),
    'batch_size': tune.choice([16, 32, 64, 128]),
    'weight_decay': tune.loguniform(1e-6, 1e-4)
}

analysis = tune.run(
    train_with_config,
    config=search_space,
    num_samples=20,
    scheduler=ASHAScheduler()
)
```

**Optuna Integration:**
```python
import optuna

def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    # Train and return validation loss
    val_loss = train_model(lr, batch_size)
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

---

## 7. Ensemble Methods

### Overview
Train multiple models with different random seeds and average their predictions at inference time. This improves robustness and reduces variance.

### Implementation

**Training:**
```bash
for seed in {42, 123, 456, 789, 1011}; do
    python training/train_nn.py \
        --data models/checkpoints/ga_trajectories.pkl \
        --config training/config/nn_config.yaml \
        --output models/planner_policy_seed${seed}.onnx \
        --seed $seed
done
```

**Inference (C++ modification required):**
```cpp
// Load multiple ONNX models
std::vector<Ort::Session> ensemble_models;
for (const auto& model_path : ensemble_paths) {
    ensemble_models.push_back(createSession(model_path));
}

// Predict with ensemble
std::vector<ControlSequence> predictions;
for (auto& model : ensemble_models) {
    predictions.push_back(model.run(inputs));
}

// Average predictions
ControlSequence final_control = averagePredictions(predictions);
```

**Trade-offs:**
- **Pro**: Better generalization, reduced variance, higher success rate
- **Con**: 3-5× inference cost (may violate 10-20 Hz requirement)

**Recommendation:**
- Test on hardware first
- If inference is fast enough (<10ms per model), ensemble is viable
- Otherwise, use for offline evaluation only

---

## 8. Multi-GA Distillation

### Overview
Train the neural network on trajectories from multiple GA configurations or robot types. This creates a more versatile policy.

### Scenarios

**Different GA Configurations:**
- Conservative GA (high collision penalty)
- Aggressive GA (low time penalty)
- Smooth GA (high smoothness weight)

**Different Robot Types:**
- Differential drive (v_y = 0)
- Omnidirectional (v_y ≠ 0)

**Different Environments:**
- Dense obstacles
- Sparse obstacles
- Narrow corridors

### Implementation

**Data Collection:**
```bash
# Run GA with different configs
python training/train_ga.py --config configs/ga_conservative.yaml --output data/conservative.pkl
python training/train_ga.py --config configs/ga_aggressive.yaml --output data/aggressive.pkl
python training/train_ga.py --config configs/ga_smooth.yaml --output data/smooth.pkl
```

**Dataset Mixing:**
```python
# Combine datasets
conservative = pickle.load(open('data/conservative.pkl', 'rb'))
aggressive = pickle.load(open('data/aggressive.pkl', 'rb'))
smooth = pickle.load(open('data/smooth.pkl', 'rb'))

mixed_data = conservative + aggressive + smooth
pickle.dump(mixed_data, open('data/mixed.pkl', 'wb'))
```

**Training:**
```bash
python training/train_nn.py --data data/mixed.pkl --config nn_config.yaml
```

**Benefits:**
- More diverse navigation behaviors
- Better generalization to novel scenarios
- Single model for multiple robot configurations

**Considerations:**
- Ensure data balance (equal samples from each GA)
- May increase training time
- Potential for conflicting behaviors (needs careful fitness design)

---

## Implementation Priority

### Phase 1 (High Priority - If Overfitting Occurs)
1. **Data Augmentation** (rotation, noise)
2. **Per-DOF Metrics** (identify weak controls)

### Phase 2 (Medium Priority - If Training Issues)
3. **Huber Loss** (if outliers detected)
4. **TensorBoard Logging** (for debugging)

### Phase 3 (Low Priority - For Refinement)
5. **Hyperparameter Tuning** (if performance plateau)
6. **Temporal Consistency Loss** (if trajectories jerky)
7. **Simulator Validation** (once Stage wrapper implemented)

### Phase 4 (Optional - For Research)
8. **Ensemble Methods** (if computational budget allows)
9. **Multi-GA Distillation** (if multiple scenarios needed)

---

## Integration Checklist

Before implementing any feature:
- [ ] Verify core training works with baseline config
- [ ] Identify specific issue (overfitting, poor performance, etc.)
- [ ] Choose appropriate feature from this document
- [ ] Test on small dataset first
- [ ] Compare metrics before/after
- [ ] Update `nn_config.yaml` with new parameters
- [ ] Document results in training logs

---

## References

- **Data Augmentation**: [Shorten & Khoshgoftaar, 2019](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0)
- **Per-DOF Metrics**: [Bojarski et al., 2016](https://arxiv.org/abs/1604.07316) (NVIDIA End-to-End Learning)
- **Ensemble Methods**: [Dietterich, 2000](https://link.springer.com/chapter/10.1007/3-540-45014-9_1)
- **Huber Loss**: [Huber, 1964](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full)

---

## Conclusion

This document provides a roadmap for incrementally improving the neural network training pipeline. Start with the core implementation, evaluate performance, and add features as needed based on empirical results.

**Current baseline**: Core training with MSE loss, early stopping, and ONNX export.

**Next milestone**: Achieve <0.01 validation MSE and >90% simulator success rate.

**Long-term goal**: Robust, generalizable navigation policy deployable on real robots.
