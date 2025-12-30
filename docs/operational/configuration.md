# Configuration

## Parameter Files

| File | Purpose |
|------|---------|
| `samples/configs/planner_params_ros1.yaml` | ROS1 runtime parameters |
| `samples/configs/planner_params_ros2.yaml` | ROS2 runtime parameters |
| `training/config/ga_config.yaml` | GA training hyperparameters |
| `training/config/nn_config.yaml` | NN training configuration |

## Runtime Parameters (ROS1/ROS2)

### Model Configuration

```yaml
model_path: "/models/planner_policy.onnx"
```

**Description**: Absolute path to trained ONNX model

**Notes**:
- Must match training output
- Container mounts `models/` to `/models`

### Planning Parameters

```yaml
num_control_steps: 20        # Number of waypoints in trajectory
control_frequency: 10.0      # Planning rate (Hz)
time_horizon: 2.0           # Look-ahead time (seconds)
```

**Recommendations**:
- **Fast robot**: Increase `control_frequency` to 20 Hz
- **Slow CPU**: Decrease to 5 Hz
- **Longer paths**: Increase `time_horizon` to 3.0s
- **Tight spaces**: Decrease `time_horizon` to 1.5s

**Constraint**: `num_control_steps = control_frequency × time_horizon`

### Velocity Limits

```yaml
max_v_x: 1.0      # Forward speed (m/s)
min_v_x: -0.5     # Backward speed (m/s)
max_v_y: 0.5      # Lateral speed (m/s, 0.0 for differential drive)
min_v_y: -0.5     # Lateral speed (m/s, 0.0 for differential drive)
max_omega: 1.0    # Rotation speed (rad/s)
min_omega: -1.0   # Rotation speed (rad/s)
```

**IMPORTANT**: Must match your robot's capabilities and training configuration

**For Differential Drive**:
```yaml
max_v_y: 0.0
min_v_y: 0.0
```

**For Omnidirectional**:
```yaml
max_v_y: 0.5  # Adjust to robot specs
min_v_y: -0.5
```

### Costmap Parameters

```yaml
costmap_window_size: 50  # Grid size (pixels)
```

**Description**: Size of local costmap window extracted for planning

**Notes**:
- Must match training: `50x50` pixels
- Resolution determined by local costmap configuration
- Typical: `0.05 m/pixel` → `2.5m × 2.5m` window

### Safety Parameters

```yaml
lethal_cost_threshold: 253       # Costs ≥ this are considered lethal
enable_collision_check: true     # Post-planning validation
```

**Recommendations**:
- Keep `lethal_cost_threshold: 253` (ROS standard)
- Set `enable_collision_check: false` only if NN is highly reliable

### Goal Tolerance

```yaml
xy_goal_tolerance: 0.1     # Position tolerance (meters)
yaw_goal_tolerance: 0.1    # Orientation tolerance (radians)
```

**Tuning**:
- **Large robot**: Increase `xy_goal_tolerance` to 0.15-0.2
- **Precision tasks**: Decrease to 0.05
- **No orientation constraint**: Increase `yaw_goal_tolerance` to 3.14

### Debugging

```yaml
debug_mode: false              # Verbose logging
publish_cost_cloud: false      # Visualize costmap as point cloud
publish_local_plan: true       # Publish trajectory to /local_plan
local_plan_topic: "local_plan" # Topic name
```

**For Debugging**:
```yaml
debug_mode: true
publish_cost_cloud: true
```

**For Production**:
```yaml
debug_mode: false
publish_cost_cloud: false
```

## Training Configuration

### GA Configuration (`ga_config.yaml`)

#### Population Parameters

```yaml
ga:
  population_size: 100      # Individuals per generation
  elite_size: 10           # Best preserved each generation
  mutation_rate: 0.1       # Gene mutation probability
  crossover_rate: 0.8      # Parent crossover probability
  num_generations: 100     # Generations per scenario
```

**Tuning Guidelines**:

| Symptom | Adjustment |
|---------|------------|
| Slow convergence | Increase `population_size` to 150-200 |
| Stuck in local minima | Increase `mutation_rate` to 0.2 |
| Good solutions lost | Increase `elite_size` to 15-20 |
| Overfitting | Decrease `num_generations` to 50 |

#### Fitness Weights

```yaml
fitness_weights:
  goal_distance: 1.0       # Reaching goal
  collision: 10.0          # Obstacle avoidance
  smoothness: 0.5          # Trajectory smoothness
  time_efficiency: 0.3     # Speed incentive
```

**Effect of Weights**:
- **goal_distance**: Higher → more goal-oriented (may cut corners)
- **collision**: Higher → more conservative (may get stuck)
- **smoothness**: Higher → smoother paths (may be slower)
- **time_efficiency**: Higher → faster trajectories (may be jerky)

**Recommended Profiles**:

**Aggressive**:
```yaml
goal_distance: 1.5
collision: 5.0
smoothness: 0.2
time_efficiency: 0.8
```

**Conservative**:
```yaml
goal_distance: 0.8
collision: 15.0
smoothness: 1.0
time_efficiency: 0.1
```

**Balanced** (default):
```yaml
goal_distance: 1.0
collision: 10.0
smoothness: 0.5
time_efficiency: 0.3
```

#### Robot Configuration

```yaml
robot:
  footprint:
    - [-0.2, -0.2]  # Rear-left
    - [0.2, -0.2]   # Front-left
    - [0.2, 0.2]    # Front-right
    - [-0.2, 0.2]   # Rear-right

  max_v_x: 1.0
  max_v_y: 0.5      # 0.0 for differential drive
  max_omega: 1.0

  max_acc_x: 2.0    # Max acceleration (m/s²)
  max_acc_theta: 2.0  # Max angular acceleration (rad/s²)
```

**CRITICAL**: Must match physical robot

**Footprint Shapes**:

**Circular** (radius 0.3m):
```yaml
footprint:
  - [0.3, 0.0]
  - [0.212, 0.212]
  - [0.0, 0.3]
  - [-0.212, 0.212]
  - [-0.3, 0.0]
  - [-0.212, -0.212]
  - [0.0, -0.3]
  - [0.212, -0.212]
```

**Rectangular** (0.6m × 0.4m):
```yaml
footprint:
  - [-0.3, -0.2]
  - [0.3, -0.2]
  - [0.3, 0.2]
  - [-0.3, 0.2]
```

#### Scenario Generation

```yaml
scenarios:
  num_training: 1000         # Total scenarios to generate
  goal_distance_min: 1.5     # Minimum goal distance (m)
  goal_distance_max: 3.0     # Maximum goal distance (m)
  random_seed: 42            # For reproducibility (null = random)
```

**Recommendations**:
- Start with 100 scenarios for quick testing
- Use 1000+ for production models
- Vary goal distances to cover use cases

### NN Configuration (`nn_config.yaml`)

#### Model Architecture

```yaml
model:
  costmap_size: 50          # Must match GA config
  num_control_steps: 20     # Must match GA config
  hidden_dim: 256          # Feature dimension

  cnn:
    channels: [1, 32, 64, 128]  # Conv layers
    kernel_sizes: [5, 3, 3]
    strides: [2, 2, 2]

  policy_head:
    hidden_dims: [256, 256]   # MLP layers
```

**Model Size Trade-offs**:

| Component | Smaller (faster) | Larger (better) |
|-----------|------------------|-----------------|
| `hidden_dim` | 128 | 512 |
| `cnn.channels` | [1, 16, 32, 64] | [1, 64, 128, 256] |
| `policy_head` | [128, 128] | [512, 512, 256] |

#### Training Parameters

```yaml
training:
  batch_size: 32
  epochs: 50
  learning_rate: 1e-3
  train_split: 0.8              # 80% train, 20% validation
  filter_percentile: 25         # Use top 75% GA trajectories
```

**Tuning**:
- **Overfitting**: Increase `filter_percentile` to 50 (top 50%)
- **Underfitting**: Decrease `filter_percentile` to 10 (top 90%)
- **Slow training**: Increase `batch_size` to 64
- **Memory issues**: Decrease `batch_size` to 16

#### Early Stopping

```yaml
training:
  early_stopping:
    patience: 10              # Epochs without improvement
    min_delta: 1e-4          # Minimum improvement threshold
```

**Effect**:
- Higher `patience` → more training, risk overfitting
- Lower `patience` → stops early, may underfit

## Parameter Validation

### Check Consistency

**Runtime vs Training**:
```bash
# Compare parameters
diff -y \
  <(grep -A5 "num_control_steps\|max_v_x\|max_omega" training/config/ga_config.yaml) \
  <(grep -A5 "num_control_steps\|max_v_x\|max_omega" samples/configs/planner_params_ros1.yaml)
```

**Must Match**:
- `num_control_steps`
- Velocity limits (`max_v_x`, `max_omega`, etc.)
- `costmap_window_size`

### Test Configuration

**ROS1**:
```bash
rosparam load samples/configs/planner_params_ros1.yaml
rosparam get /PlanGAROS1Plugin
```

**ROS2**:
```bash
ros2 param load /controller_server samples/configs/planner_params_ros2.yaml
ros2 param dump /controller_server
```

## Common Configuration Patterns

### High-Speed Robot

```yaml
# Runtime
max_v_x: 2.0
control_frequency: 20.0
time_horizon: 2.0
num_control_steps: 40

# Training
ga:
  num_generations: 150  # Need more exploration
fitness_weights:
  smoothness: 1.0       # Emphasize smoothness
  time_efficiency: 0.5
```

### Tight Spaces

```yaml
# Runtime
max_v_x: 0.5
time_horizon: 1.5
xy_goal_tolerance: 0.05

# Training
fitness_weights:
  collision: 20.0       # Very conservative
  smoothness: 0.8
costmap:
  num_obstacles_max: 12  # Dense environments
```

### Omnidirectional Robot

```yaml
# Runtime & Training
max_v_x: 1.0
max_v_y: 1.0          # Enable lateral movement
min_v_y: -1.0
```

## Next Steps

- **Deploy planner**: See [deployment.md](deployment.md)
- **Troubleshooting**: See [troubleshooting.md](troubleshooting.md)
- **Retrain with new config**: See [training.md](training.md)
