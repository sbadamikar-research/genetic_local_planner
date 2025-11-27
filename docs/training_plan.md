# Training Plan - GA-Based ROS Local Planner

## Table of Contents
1. [Overview](#overview)
2. [Stage Simulator Setup](#stage-simulator-setup)
3. [Genetic Algorithm Training](#genetic-algorithm-training)
4. [Neural Network Training](#neural-network-training)
5. [Model Export and Validation](#model-export-and-validation)
6. [Iterative Improvement](#iterative-improvement)

---

## Overview

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: GA Training (1-2 weeks)                            │
│ ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│ │ Generate     │ -> │ Evolve GA    │ -> │ Collect Best │   │
│ │ Scenarios    │    │ Population   │    │ Trajectories │   │
│ └──────────────┘    └──────────────┘    └──────────────┘   │
│        1000+ scenarios, 50-100 generations each             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: NN Distillation (1 week)                           │
│ ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│ │ Prepare      │ -> │ Train Neural │ -> │ Export ONNX  │   │
│ │ Dataset      │    │ Network      │    │ Model        │   │
│ └──────────────┘    └──────────────┘    └──────────────┘   │
│        Supervised learning, 50 epochs                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Validation & Deployment                            │
│ ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│ │ Verify ONNX  │ -> │ Test in      │ -> │ Deploy to    │   │
│ │ Accuracy     │    │ Simulation   │    │ ROS Plugins  │   │
│ └──────────────┘    └──────────────┘    └──────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Training Environment

**Host Machine:**
- Python 3.10 with PyTorch, ONNX
- Miniconda environment: `plan_ga`
- Lightweight simulator (no ROS required)

**Output:**
- GA trajectory dataset: `models/checkpoints/all_trajectories.pkl`
- Trained NN: `models/distilled_policy.pth`
- ONNX model: `models/planner_policy.onnx`

---

## Stage Simulator Setup

### Lightweight Python Simulator

We use a lightweight Python-based simulator to avoid ROS overhead during training. This significantly speeds up the GA evolution process.

#### Components

1. **Costmap** (`training/simulator/costmap.py`):
   - 50×50 grid, 0.05m resolution (2.5m × 2.5m local window)
   - Values: 0-255 (0=free, 254=lethal, 253=inscribed, 1-252=inflation)
   - Exponential decay: `cost = 252 * exp(-decay_factor * distance)`

2. **Robot Model** (`training/simulator/robot_model.py`):
   - Differential drive dynamics
   - Velocity limits: v_x ∈ [-0.5, 1.0], v_y ∈ [-0.5, 0.5], ω ∈ [-1.0, 1.0]
   - Footprint: Polygon vertices (e.g., square robot)
   - Integration: Euler method with dt=0.1s

3. **Collision Checker** (`training/simulator/collision_checker.py`):
   - Ray-casting or grid-based collision detection
   - Check robot footprint against costmap
   - Return cost value at robot base_link

4. **Stage Wrapper** (`training/simulator/stage_wrapper.py`):
   - Unified interface for simulation
   - Forward simulation: Execute control sequence
   - Record trajectory: poses, velocities, costmap values

### Costmap Generation

#### Option 1: Procedural Generation (Recommended for Training)

```python
def generateRandomCostmap(width=50, height=50, resolution=0.05,
                          num_obstacles=5, inflation_radius=0.5):
    """
    Generate random costmap with obstacles.

    Args:
        width, height: Grid dimensions (pixels)
        resolution: Meters per pixel
        num_obstacles: Number of random obstacles
        inflation_radius: Inflation radius in meters

    Returns:
        costmap: 50×50 numpy array with values 0-255
    """
    # Initialize free space
    costmap = np.zeros((height, width), dtype=np.uint8)

    # Place random obstacles
    for _ in range(num_obstacles):
        # Random circle or rectangle
        center_x = np.random.randint(10, width-10)
        center_y = np.random.randint(10, height-10)
        radius = np.random.randint(2, 8)

        # Mark lethal cells
        for y in range(height):
            for x in range(width):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < radius:
                    costmap[y, x] = 254  # Lethal

    # Apply inflation
    costmap = applyInflation(costmap, inflation_radius, resolution)

    return costmap
```

#### Option 2: Load from Stage World Files

For validation, load actual Stage world files:

```python
def loadStageWorld(world_file):
    """Parse Stage .world file and extract obstacle positions."""
    # Parse .world file
    # Convert obstacles to costmap grid
    # Apply inflation
    return costmap
```

### Training Scenario Generation

```python
def generateTrainingScenario():
    """
    Generate a random training scenario.

    Returns:
        scenario: Dict with costmap, start_pose, goal_pose
    """
    # Generate costmap
    costmap = generateRandomCostmap(num_obstacles=np.random.randint(3, 8))

    # Sample free start position
    start_pose = sampleFreePosition(costmap)

    # Sample goal at reasonable distance (1.5-3.0 meters)
    while True:
        goal_pose = sampleFreePosition(costmap)
        dist = np.linalg.norm(goal_pose[:2] - start_pose[:2])
        if 1.5 <= dist <= 3.0:
            break

    return {
        'costmap': costmap,
        'start_pose': start_pose,  # [x, y, theta]
        'goal_pose': goal_pose,
        'metadata': {
            'inflation_decay': 0.8,
            'resolution': 0.05
        }
    }
```

---

## Genetic Algorithm Training

### GA Hyperparameters

**Rationale and Selection:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Population Size | 100 | Balance between exploration (diversity) and computation time. Larger = better exploration but slower. |
| Elite Size | 10 | Preserve best 10% of population. Ensures good solutions aren't lost during evolution. |
| Mutation Rate | 0.1 | 10% of genes mutated. High enough for exploration, low enough to preserve good solutions. |
| Crossover Rate | 0.8 | 80% of offspring created via crossover. Strong exploitation of good gene combinations. |
| Generations | 50-100 | Typically converges in 50 gens for simple scenarios, 100 for complex. Monitor fitness plateaus. |
| Tournament Size | 3 | For selection. Larger = more selection pressure toward best solutions. |
| Training Scenarios | 1000+ | More scenarios = better generalization. Aim for diverse obstacle configurations. |

### Chromosome Encoding

**Direct Control Sequence:**

```python
chromosome.genes = [
    v_x_0, v_y_0, omega_0,   # Timestep 0
    v_x_1, v_y_1, omega_1,   # Timestep 1
    ...
    v_x_19, v_y_19, omega_19 # Timestep 19
]
# Total: 20 timesteps × 3 DOF = 60 genes
```

**Advantages:**
- Direct representation (no decoding needed)
- Smooth control transitions possible
- Easy to apply crossover/mutation

**Gene Bounds:**
- v_x: [-0.5, 1.0] m/s
- v_y: [-0.5, 0.5] m/s (set to 0 for differential drive)
- omega: [-1.0, 1.0] rad/s

### Fitness Function

**Multi-objective with weighted sum:**

```python
fitness = -(
    w_goal * goal_distance +
    w_collision * collision_cost +
    w_smoothness * jerk +
    w_time * path_length
)
# Negative because we minimize
```

**Default Weights:**

```yaml
fitness_weights:
  goal_distance: 1.0      # Primary objective
  collision: 10.0         # Heavy penalty (10× goal weight)
  smoothness: 0.5         # Moderate smoothing
  time_efficiency: 0.3    # Mild preference for direct paths
```

**Weight Tuning Strategy:**

1. **Start with goal distance only**: `[1.0, 0, 0, 0]`
   - Verify GA can reach goal in free space

2. **Add collision penalty**: `[1.0, 10.0, 0, 0]`
   - Increase collision weight until obstacles are avoided
   - Try: 5.0, 10.0, 20.0

3. **Add smoothness**: `[1.0, 10.0, 0.5, 0]`
   - Reduce jerky motions
   - Balance with goal-reaching

4. **Add time efficiency**: `[1.0, 10.0, 0.5, 0.3]`
   - Encourage shorter paths
   - Keep weight low to avoid overly aggressive paths

### Genetic Operators

#### Crossover (Uniform)

```python
def uniformCrossover(parent1, parent2):
    """
    Randomly swap genes between parents.

    For each gene:
        - 50% chance: child1 gets parent1's gene
        - 50% chance: child1 gets parent2's gene
    """
    child1 = parent1.copy()
    child2 = parent2.copy()

    for i in range(len(parent1.genes)):
        if random.random() < 0.5:
            child1.genes[i] = parent2.genes[i]
            child2.genes[i] = parent1.genes[i]

    return child1, child2
```

#### Mutation (Gaussian)

```python
def gaussianMutation(chromosome, mutation_rate=0.1, sigma=0.1):
    """
    Add Gaussian noise to genes.

    Args:
        mutation_rate: Probability of mutating each gene
        sigma: Standard deviation of Gaussian noise
    """
    for i in range(len(chromosome.genes)):
        if random.random() < mutation_rate:
            # Add Gaussian noise
            chromosome.genes[i] += random.gauss(0, sigma)

    # Clip to bounds
    chromosome.clip()
```

### Parallel Fitness Evaluation

**Speed up training 10×+ using multiprocessing:**

```python
from multiprocessing import Pool

def evaluatePopulationParallel(population, simulator, num_workers=8):
    """
    Evaluate fitness for all chromosomes in parallel.

    Args:
        population: List of chromosomes
        simulator: Simulation environment
        num_workers: Number of parallel processes
    """
    with Pool(num_workers) as pool:
        # Each worker evaluates a subset of population
        results = pool.starmap(
            evaluateFitness,
            [(chrom, simulator) for chrom in population]
        )

    # Assign fitness to chromosomes
    for chrom, (fitness, metrics) in zip(population, results):
        chrom.fitness = fitness
        chrom.metrics = metrics
```

### Training Script Usage

```bash
# Activate conda environment
conda activate plan_ga

# Train GA on 1000 scenarios
python training/train_ga.py \
    --config training/config/ga_config.yaml \
    --num_scenarios 1000 \
    --output models/checkpoints/ \
    --workers 8

# Monitor progress
# - Tensorboard logs saved to models/checkpoints/logs/
tensorboard --logdir models/checkpoints/logs/

# Expected runtime: 6-12 hours (depending on CPU)
```

### Monitoring Training

**Key Metrics:**

1. **Best Fitness**: Should steadily improve over generations
2. **Average Fitness**: Population quality indicator
3. **Goal Reached %**: Percentage of scenarios where goal tolerance met
4. **Collision Rate**: Should decrease to near 0%
5. **Convergence**: Fitness plateaus after ~50-80 generations

**Example Output:**

```
Scenario 1/1000:
  Generation 0: Best=-15.23, Avg=-28.45, Goal=12%, Collision=45%
  Generation 10: Best=-8.67, Avg=-18.32, Goal=35%, Collision=25%
  Generation 20: Best=-4.21, Avg=-10.15, Goal=68%, Collision=8%
  Generation 50: Best=-1.83, Avg=-4.92, Goal=95%, Collision=2%
  ✓ Converged at generation 52

Saved trajectory to models/checkpoints/trajectory_0001.pkl
```

---

## Neural Network Training

### Network Architecture

**Design Rationale:**

1. **CNN for Costmap**: Spatial features (obstacles, free space)
   - Conv2D layers extract local patterns
   - Learns obstacle avoidance implicitly

2. **MLP for State**: Kinematic/dynamic information
   - Current velocity/acceleration
   - Goal direction and distance

3. **Fusion**: Combine perception + state
   - Late fusion (after separate encoding)
   - Allows independent learning

4. **Output**: Direct control prediction
   - 20 timesteps × 3 DOF = 60 outputs
   - Tanh activation + scaling for bounds

**Detailed Architecture:**

```
Input:
  - costmap: (1, 50, 50) float32, normalized [0, 1]
  - robot_state: (9,) float32
  - goal_relative: (3,) float32
  - costmap_metadata: (2,) float32

Costmap Encoder (CNN):
  Conv2D(1 -> 32, kernel=5, stride=2)  →  (32, 25, 25)
  ReLU
  Conv2D(32 -> 64, kernel=3, stride=2) →  (64, 13, 13)
  ReLU
  Conv2D(64 -> 128, kernel=3, stride=2) → (128, 7, 7)
  ReLU
  Flatten → (6272,)
  Linear(6272 -> 256)
  ReLU

State Encoder (MLP):
  Concat(robot_state, goal_relative, costmap_metadata) → (14,)
  Linear(14 -> 128)
  ReLU
  Linear(128 -> 256)
  ReLU

Fusion:
  Concat(costmap_features, state_features) → (512,)
  Linear(512 -> 256)
  ReLU
  Linear(256 -> 256)
  ReLU

Output Head:
  Linear(256 -> 60)  # 20 steps × 3 controls
  Reshape(60 -> 20, 3)
  Tanh activation + scaling

Output:
  control_sequence: (20, 3) float32
```

**Model Size**: ~2.5M parameters, ~10MB ONNX file

### Dataset Preparation

```python
# Load GA trajectories
with open('models/checkpoints/all_trajectories.pkl', 'rb') as f:
    trajectories = pickle.load(f)

# Filter low-quality trajectories (bottom 25%)
fitness_threshold = np.percentile([t['fitness'] for t in trajectories], 25)
trajectories = [t for t in trajectories if t['fitness'] >= fitness_threshold]

# Data augmentation
augmented = []
for traj in trajectories:
    # Original
    augmented.append(traj)

    # Random rotation (0°, 90°, 180°, 270°)
    for angle in [90, 180, 270]:
        aug_traj = rotateCostmapAndPoses(traj, angle)
        augmented.append(aug_traj)

    # Add Gaussian noise to costmap
    noisy_traj = addCostmapNoise(traj, sigma=5.0)
    augmented.append(noisy_traj)

# Split train/val (80/20)
split_idx = int(0.8 * len(augmented))
train_data = augmented[:split_idx]
val_data = augmented[split_idx:]

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
```

### Training Hyperparameters

```yaml
training:
  batch_size: 32
  epochs: 50
  learning_rate: 1e-3
  optimizer: Adam
  loss: MSELoss
  scheduler: ReduceLROnPlateau
    patience: 5
    factor: 0.5
  early_stopping:
    patience: 10
    min_delta: 1e-4
```

**Rationale:**

- **Batch size 32**: Balance between stability and memory
- **Learning rate 1e-3**: Standard Adam starting point
- **MSE loss**: Regression task (predict continuous controls)
- **ReduceLROnPlateau**: Adaptive learning rate when validation plateaus
- **Early stopping**: Prevent overfitting

### Training Script Usage

```bash
# Train neural network
python training/train_nn.py \
    --data models/checkpoints/all_trajectories.pkl \
    --output models/distilled_policy.pth \
    --onnx_output models/planner_policy.onnx \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3

# Monitor with Tensorboard
tensorboard --logdir models/logs/

# Expected runtime: 2-4 hours (CPU), 30-60 min (GPU)
```

### Monitoring Training

**Key Metrics:**

1. **Training Loss**: Should decrease steadily
2. **Validation Loss**: Should track training loss (gap indicates overfitting)
3. **Learning Rate**: Decreases when validation plateaus
4. **Control MSE**: Per-DOF error (v_x, v_y, omega)

**Example Output:**

```
Epoch 1/50:
  Train Loss: 0.1523, Val Loss: 0.1687
  Control MSE: v_x=0.0234, v_y=0.0189, omega=0.0312

Epoch 10/50:
  Train Loss: 0.0432, Val Loss: 0.0489
  Control MSE: v_x=0.0089, v_y=0.0067, omega=0.0123
  LR: 1e-3

Epoch 20/50:
  Train Loss: 0.0198, Val Loss: 0.0221
  Control MSE: v_x=0.0041, v_y=0.0032, omega=0.0067
  LR reduced to 5e-4

Epoch 35/50:
  Train Loss: 0.0087, Val Loss: 0.0095
  Control MSE: v_x=0.0018, v_y=0.0015, omega=0.0029
  ✓ Best model saved

Early stopping triggered at epoch 45
Best validation loss: 0.0091 (epoch 38)
```

---

## Model Export and Validation

### ONNX Export

```bash
# Export happens automatically during training
python training/train_nn.py ... --onnx_output models/planner_policy.onnx

# Or export manually
python training/utils/export_onnx.py \
    --model models/distilled_policy.pth \
    --output models/planner_policy.onnx \
    --verify
```

### Verification

**1. Accuracy Verification:**

```python
# Compare PyTorch vs ONNX outputs
import torch
import onnxruntime as ort

# Load PyTorch model
model = PlannerPolicyNet()
model.load_state_dict(torch.load('models/distilled_policy.pth'))
model.eval()

# Load ONNX model
session = ort.InferenceSession('models/planner_policy.onnx')

# Test inputs
costmap = torch.randn(1, 1, 50, 50)
robot_state = torch.randn(1, 9)
goal_relative = torch.randn(1, 3)
costmap_metadata = torch.randn(1, 2)

# PyTorch inference
with torch.no_grad():
    pytorch_output = model(costmap, robot_state, goal_relative, costmap_metadata)

# ONNX inference
onnx_output = session.run(None, {
    'costmap': costmap.numpy(),
    'robot_state': robot_state.numpy(),
    'goal_relative': goal_relative.numpy(),
    'costmap_metadata': costmap_metadata.numpy()
})[0]

# Compare
diff = np.abs(pytorch_output.numpy() - onnx_output).max()
print(f"Max difference: {diff:.8f}")

# Should be < 1e-5
assert diff < 1e-5, "ONNX export verification failed!"
```

**2. Inference Latency Benchmark:**

```python
import time

# Warm-up
for _ in range(10):
    session.run(None, inputs)

# Benchmark
num_iters = 100
start = time.time()
for _ in range(num_iters):
    session.run(None, inputs)
end = time.time()

avg_latency = (end - start) / num_iters * 1000  # ms
print(f"Average inference latency: {avg_latency:.2f} ms")

# Target: < 30ms for 10-20 Hz planning
assert avg_latency < 30, f"Latency too high: {avg_latency:.2f} ms"
```

**3. Simulation Validation:**

```python
# Test trained policy in simulator
success_rate, avg_path_length = testPolicyInSimulation(
    onnx_model='models/planner_policy.onnx',
    num_scenarios=100,
    max_steps=200
)

print(f"Success Rate: {success_rate:.1%}")
print(f"Average Path Length: {avg_path_length:.2f} m")

# Target: >90% success rate
```

---

## Iterative Improvement

### Failure Analysis

**Identify failure modes:**

```python
# Collect failures during validation
failures = []
for scenario in test_scenarios:
    result = runScenario(scenario, model)
    if not result.success:
        failures.append({
            'scenario': scenario,
            'reason': result.failure_reason,  # collision, timeout, etc.
            'trajectory': result.trajectory
        })

# Analyze common failure patterns
collision_failures = [f for f in failures if f['reason'] == 'collision']
timeout_failures = [f for f in failures if f['reason'] == 'timeout']

print(f"Collisions: {len(collision_failures)}")
print(f"Timeouts: {len(timeout_failures)}")
```

### Retraining Strategy

**1. Augment Dataset with Failures:**

```python
# Generate more scenarios similar to failures
for failure in failures:
    # Extract failure characteristics
    obstacle_density = analyzeObstacleDensity(failure['scenario'])
    goal_distance = analyzeGoalDistance(failure['scenario'])

    # Generate similar scenarios
    new_scenarios = generateSimilarScenarios(
        obstacle_density=obstacle_density,
        goal_distance=goal_distance,
        count=10
    )

    # Retrain GA on these scenarios
    for scenario in new_scenarios:
        trainGA(scenario)
```

**2. Adjust Fitness Weights:**

```yaml
# If too many collisions, increase collision weight
fitness_weights:
  collision: 20.0  # Was 10.0

# If goals not reached, increase goal weight
fitness_weights:
  goal_distance: 2.0  # Was 1.0
```

**3. Fine-tune Neural Network:**

```python
# Load pre-trained model
model.load_state_dict(torch.load('models/distilled_policy.pth'))

# Fine-tune on new data
train(model, new_data, epochs=10, lr=1e-4)

# Export updated model
exportToONNX(model, 'models/planner_policy_v2.onnx')
```

### Validation Protocol

**Before deploying new model:**

1. **Unit Tests**: Verify ONNX export accuracy
2. **Simulation Tests**: 100+ scenarios, >90% success rate
3. **Latency Tests**: <30ms average inference time
4. **Comparison Tests**: Match or beat previous model
5. **Stage/Gazebo Tests**: Real ROS environment validation

---

## Performance Targets

### GA Training

- **Convergence**: <100 generations per scenario
- **Success Rate**: >95% goals reached (in training)
- **Collision Rate**: <5%
- **Training Time**: <12 hours for 1000 scenarios (8-core CPU)

### Neural Network Training

- **Validation MSE**: <0.01
- **Success Rate**: >90% (matches GA performance)
- **Training Time**: <4 hours (CPU), <1 hour (GPU)

### Inference

- **Latency**: <30ms (average)
- **Planning Frequency**: 10-20 Hz (consistent)
- **Accuracy**: Max diff from PyTorch <1e-5

### End-to-End Performance

- **Goal Success Rate**: >85% (real scenarios)
- **Collision Rate**: <5%
- **Path Quality**: Comparable to DWA
- **Smoothness**: Low jerk, human-like motion

---

## Next Steps

After successful training:

1. **Deploy to ROS**: Follow [Development Plan](development_plan.md)
2. **Benchmark**: Compare with DWA/TEB planners
3. **Collect Real Data**: Record real robot failures
4. **Retrain**: Incorporate real-world data
5. **Optimize**: Reduce model size or increase accuracy

For deployment instructions, see [Development Plan](development_plan.md).
