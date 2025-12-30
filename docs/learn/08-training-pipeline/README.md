# Module 08: Training Pipeline & Experiments

**Estimated Time:** 1 day (6-8 hours)

## 🎯 Learning Objectives

- ✅ Execute complete GA training pipeline
- ✅ Train neural network on GA-generated trajectories
- ✅ Export models to ONNX format and verify correctness
- ✅ Run hyperparameter experiments (population size, mutation rate, etc.)
- ✅ Analyze training results and convergence behavior
- ✅ Debug common training failures
- ✅ Understand the end-to-end data flow from scenarios to deployed model
- ✅ Optimize hyperparameters for better performance

## Why Train Custom Models?

**Pre-trained models don't exist for our specific:**
- Robot footprint (size, shape)
- Velocity limits (max speed, acceleration)
- Environment (obstacle density, corridor width)
- Safety preferences (how close to walls?)

**Training advantages:**
- Tailored behavior for your robot
- Optimal for your specific task
- Can retrain when requirements change
- Understand what the model learned

**For this project:** Train GA to find good trajectories → Distill into fast NN → Deploy to C++

---

## Key Concepts

### End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│              TRAINING PIPELINE (Host Machine)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STEP 1: Generate Scenarios                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Random navigation scenarios (1000+)                    │    │
│  │  • Costmap with obstacles                               │    │
│  │  • Start position (0, 0)                                │    │
│  │  • Goal position (random, 1.5-3.0m away)                │    │
│  └────────────────────────────────────────────────────────┘    │
│                          ↓                                       │
│  STEP 2: Evolve Trajectories (GA)                              │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  For each scenario:                                     │    │
│  │  • Initialize population of control sequences          │    │
│  │  • Evolve for 100 generations                          │    │
│  │  • Select best trajectory                              │    │
│  │  Output: control_sequence + fitness                    │    │
│  └────────────────────────────────────────────────────────┘    │
│                          ↓                                       │
│  STEP 3: Collect Data                                          │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Save to ga_trajectories.pkl:                          │    │
│  │  {                                                      │    │
│  │    "costmap": [50, 50],                                │    │
│  │    "robot_state": [x, y, θ, v_x, v_y, ω, ...],        │    │
│  │    "goal_relative": [dx, dy, dθ],                      │    │
│  │    "control_sequence": [20, 3],  # 20 steps           │    │
│  │    "fitness": 0.85                                     │    │
│  │  }                                                      │    │
│  └────────────────────────────────────────────────────────┘    │
│                          ↓                                       │
│  STEP 4: Train Neural Network                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  PyTorch training:                                      │    │
│  │  • Load ga_trajectories.pkl                           │    │
│  │  • Filter low-fitness samples (bottom 25%)            │    │
│  │  • Train CNN+MLP to predict controls                  │    │
│  │  • MSE loss on control sequences                      │    │
│  │  • 50 epochs with early stopping                      │    │
│  └────────────────────────────────────────────────────────┘    │
│                          ↓                                       │
│  STEP 5: Export to ONNX                                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  torch.onnx.export()                                    │    │
│  │  • Verify numerical equivalence (PyTorch vs ONNX)     │    │
│  │  • Save as planner_policy.onnx (~2-5 MB)              │    │
│  └────────────────────────────────────────────────────────┘    │
│                          ↓                                       │
│  STEP 6: Deploy to C++                                         │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Copy to Docker container:                             │    │
│  │  models/planner_policy.onnx → /models/                │    │
│  │                                                         │    │
│  │  C++ plugin loads ONNX Runtime                         │    │
│  │  • 10-20 Hz inference                                  │    │
│  │  • <5ms latency                                        │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Hyperparameter Impact

```
GENETIC ALGORITHM HYPERPARAMETERS:

population_size: 100
├─ Too small (50): Fast but limited diversity → poor solutions
└─ Too large (500): Slow, may not converge within time budget

mutation_rate: 0.1
├─ Too small (0.01): Premature convergence (stuck in local optimum)
└─ Too large (0.5): Too random, no convergence

crossover_rate: 0.8
├─ Promotes exploration via gene mixing
└─ Balance with mutation for optimal search

num_generations: 100
├─ Convergence typically happens around gen 30-70
└─ More generations = better solutions (diminishing returns)

NEURAL NETWORK HYPERPARAMETERS:

batch_size: 32
├─ Too small (8): Noisy gradients, slow training
└─ Too large (256): Less frequent updates, memory issues

learning_rate: 1e-3
├─ Too small (1e-5): Slow convergence (hours)
└─ Too large (1e-1): Unstable, diverges

hidden_dim: 256
├─ Too small (64): Underfitting, can't learn complex patterns
└─ Too large (1024): Overfitting, slow inference
```

### Training Convergence Patterns

**Healthy GA convergence:**
```
Fitness over generations:
Gen 0:   -15.3  (random)
Gen 10:  -8.2   (improving)
Gen 30:  -3.1   (converging)
Gen 50:  -1.2   (good solution)
Gen 100: -1.15  (converged, diminishing returns)
```

**Healthy NN training:**
```
Loss over epochs:
Epoch 1:  train=2.35, val=2.41  (initial)
Epoch 10: train=0.85, val=0.92  (learning)
Epoch 30: train=0.32, val=0.38  (good fit)
Epoch 50: train=0.28, val=0.35  (converged)

If val > train significantly: Overfitting!
If both high: Underfitting!
```

---

## Hands-On Exercises

### Exercise 1: Run GA Training on 100 Scenarios (1 hour)

Execute small-scale GA training:

```bash
# Activate conda environment
conda activate plan_ga

# Quick test (10 scenarios, 1 worker)
python training/train_ga.py \
  --config training/config/ga_config.yaml \
  --output models/checkpoints/test_10.pkl \
  --num_scenarios 10 \
  --num_workers 1

# Watch output:
# Scenario 1/10:
#   Generating random navigation scenario...
#   Running GA evolution (pop=100, gen=100)...
#   Gen   1: best_fitness=-12.35, avg=-18.42
#   Gen  10: best_fitness=-5.21,  avg=-9.87
#   Gen  50: best_fitness=-1.82,  avg=-3.45
#   Gen 100: best_fitness=-1.21,  avg=-1.89
#   ✓ Converged! Saving trajectory...
# Scenario 2/10:
#   ...

# Production run (100 scenarios, 8 workers)
python training/train_ga.py \
  --config training/config/ga_config.yaml \
  --output models/checkpoints/ga_100.pkl \
  --num_scenarios 100 \
  --num_workers 8

# Time estimate: ~10-30 minutes (depends on CPU)
```

**Verify output:**
```python
import pickle
import numpy as np

# Load results
with open('models/checkpoints/ga_100.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Total trajectories: {len(data)}")
print(f"First trajectory keys: {data[0].keys()}")

# Check data shapes
print(f"Costmap shape: {data[0]['costmap'].shape}")  # (50, 50)
print(f"Control sequence shape: {data[0]['control_sequence'].shape}")  # (20, 3)
print(f"Fitness: {data[0]['fitness']}")  # Should be negative (lower = better)

# Statistics
fitnesses = [t['fitness'] for t in data]
print(f"Mean fitness: {np.mean(fitnesses):.2f}")
print(f"Best fitness: {np.max(fitnesses):.2f}")  # Remember: less negative is better
```

**Questions:**
1. How long does one scenario take? (~10-60 seconds depending on difficulty)
2. Does GA consistently find solutions? (should be >90% success rate)
3. What's the typical fitness range? (-5 to -1 is good)

### Exercise 2: Train NN on GA Data (1 hour)

Distill GA knowledge into neural network:

```bash
# Train with default config
python training/train_nn.py \
  --data models/checkpoints/ga_100.pkl \
  --config training/config/nn_config.yaml \
  --output models/test_planner.onnx \
  --checkpoint models/checkpoints/test_model.pth

# Watch training:
# Epoch  1/50 - train_loss: 2.351, val_loss: 2.412, lr: 0.001000
# Epoch  5/50 - train_loss: 0.985, val_loss: 1.023, lr: 0.001000
# Epoch 10/50 - train_loss: 0.542, val_loss: 0.598, lr: 0.001000
# Epoch 20/50 - train_loss: 0.312, val_loss: 0.365, lr: 0.000500
# ...
# Epoch 35/50 - train_loss: 0.287, val_loss: 0.341, lr: 0.000250
# ✓ Best model saved at epoch 35 (val_loss=0.341)
# Early stopping triggered (no improvement for 10 epochs)
# Exporting to ONNX...
# ✓ ONNX export successful: models/test_planner.onnx
# ✓ Verification passed (max diff: 1.2e-6)
```

**Load and inspect model:**
```python
import torch
from training.neural_network.model import create_model
import yaml

# Load PyTorch model
with open('training/config/nn_config.yaml') as f:
    config = yaml.safe_load(f)

model = create_model(config)
model.load_state_dict(torch.load('models/checkpoints/test_model.pth'))
model.eval()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")  # ~500K-1M parameters

# Test inference
costmap = torch.randn(1, 1, 50, 50)
robot_state = torch.randn(1, 9)
goal = torch.randn(1, 3)
metadata = torch.randn(1, 2)

with torch.no_grad():
    output = model(costmap, robot_state, goal, metadata)
print(f"Output shape: {output.shape}")  # (1, 60)
```

**Questions:**
1. At what epoch does training converge? (typically 20-40 epochs)
2. Is there overfitting? (val_loss > train_loss by >20%?)
3. How big is the ONNX file? (2-5 MB)

### Exercise 3: Verify ONNX Export and Test in C++ (45 min)

Ensure ONNX model works in C++:

**Step 1: Verify ONNX in Python**
```python
import onnxruntime as ort
import numpy as np

# Load ONNX session
session = ort.InferenceSession('models/test_planner.onnx')

# Check inputs/outputs
for input_node in session.get_inputs():
    print(f"Input: {input_node.name}, shape: {input_node.shape}")

for output_node in session.get_outputs():
    print(f"Output: {output_node.name}, shape: {output_node.shape}")

# Expected:
# Input: costmap_input, shape: [1, 1, 50, 50]
# Input: robot_state_input, shape: [1, 9]
# Input: goal_relative_input, shape: [1, 3]
# Input: costmap_metadata_input, shape: [1, 2]
# Output: output, shape: [1, 60]

# Run inference
inputs = {
    'costmap_input': np.random.randn(1, 1, 50, 50).astype(np.float32),
    'robot_state_input': np.random.randn(1, 9).astype(np.float32),
    'goal_relative_input': np.random.randn(1, 3).astype(np.float32),
    'costmap_metadata_input': np.random.randn(1, 2).astype(np.float32)
}

output = session.run(None, inputs)
print(f"Output: {output[0].shape}")  # (1, 60)
```

**Step 2: Test in C++ (inside Docker container)**
```bash
# Copy model to container
docker cp models/test_planner.onnx plan_ga_ros1:/models/planner_policy.onnx

# Attach to container
docker exec -it plan_ga_ros1 bash

# Inside container: Build project
cd /catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make

# Test plugin loads
source devel/setup.bash
rospack plugins --attrib=plugin nav_core | grep plan_ga
# Should show: plan_ga_ros1

# Launch minimal test (if you have a test launch file)
# roslaunch plan_ga_ros1 test_planner.launch
```

**Questions:**
1. Do input/output names match C++ code? (Critical!)
2. Does ONNX inference work without errors?
3. What's the inference latency? (should be <10ms in Python, <5ms in C++)

### Exercise 4: GA Hyperparameter Experiments (1.5 hours)

Compare different GA configurations:

**Experiment Setup:**
```bash
# Create variant configs
cp training/config/ga_config.yaml training/config/ga_pop50.yaml
cp training/config/ga_config.yaml training/config/ga_pop200.yaml
cp training/config/ga_config.yaml training/config/ga_mut01.yaml
cp training/config/ga_config.yaml training/config/ga_mut03.yaml
```

**Edit configs:**
```yaml
# ga_pop50.yaml: Small population
ga:
  population_size: 50   # (was 100)

# ga_pop200.yaml: Large population
ga:
  population_size: 200  # (was 100)

# ga_mut01.yaml: Low mutation
ga:
  mutation_rate: 0.01   # (was 0.1)

# ga_mut03.yaml: High mutation
ga:
  mutation_rate: 0.3    # (was 0.1)
```

**Run experiments:**
```bash
# Baseline
python training/train_ga.py --config training/config/ga_config.yaml \
  --output models/checkpoints/baseline.pkl --num_scenarios 50 --num_workers 8

# Variants
for config in ga_pop50 ga_pop200 ga_mut01 ga_mut03; do
  python training/train_ga.py \
    --config training/config/${config}.yaml \
    --output models/checkpoints/${config}.pkl \
    --num_scenarios 50 --num_workers 8
done
```

**Analyze results:**
```python
import pickle
import matplotlib.pyplot as plt
import numpy as np

configs = ['baseline', 'ga_pop50', 'ga_pop200', 'ga_mut01', 'ga_mut03']
results = {}

for config in configs:
    with open(f'models/checkpoints/{config}.pkl', 'rb') as f:
        data = pickle.load(f)
    results[config] = {
        'fitnesses': [t['fitness'] for t in data],
        'mean_fitness': np.mean([t['fitness'] for t in data]),
        'best_fitness': np.max([t['fitness'] for t in data])
    }

# Plot comparison
plt.figure(figsize=(12, 5))

# Fitness distributions
plt.subplot(1, 2, 1)
for config in configs:
    plt.hist(results[config]['fitnesses'], alpha=0.5, label=config, bins=20)
plt.xlabel('Fitness')
plt.ylabel('Count')
plt.legend()
plt.title('Fitness Distribution Comparison')

# Mean fitness bar chart
plt.subplot(1, 2, 2)
means = [results[c]['mean_fitness'] for c in configs]
plt.bar(configs, means)
plt.ylabel('Mean Fitness')
plt.title('Mean Fitness by Configuration')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('ga_comparison.png')
plt.show()

# Print summary
for config in configs:
    print(f"{config}:")
    print(f"  Mean fitness: {results[config]['mean_fitness']:.3f}")
    print(f"  Best fitness: {results[config]['best_fitness']:.3f}")
```

**Questions:**
1. Which configuration works best? (baseline likely good)
2. How does population size affect fitness? (larger = better but slower)
3. What happens with extreme mutation rates? (0.01: stagnates, 0.3: too random)

### Exercise 5: NN Architecture Experiments (1.5 hours)

Compare network architectures:

**Create variant configs:**
```yaml
# nn_small.yaml: Smaller network
model:
  hidden_dim: 128  # (was 256)
  cnn:
    channels: [1, 16, 32, 64]  # (was [1, 32, 64, 128])

# nn_large.yaml: Larger network
model:
  hidden_dim: 512  # (was 256)
  cnn:
    channels: [1, 64, 128, 256]  # (was [1, 32, 64, 128])

# nn_nodropout.yaml: No regularization
training:
  dropout: 0.0  # (was 0.0, but add to baseline)

# nn_dropout.yaml: With regularization
training:
  dropout: 0.3  # (was 0.0)
```

**Run training:**
```bash
for config in nn_config nn_small nn_large nn_dropout; do
  python training/train_nn.py \
    --data models/checkpoints/ga_100.pkl \
    --config training/config/${config}.yaml \
    --output models/${config}.onnx \
    --checkpoint models/${config}.pth
done
```

**Compare results:**
```python
import json

configs = ['nn_config', 'nn_small', 'nn_large', 'nn_dropout']

for config in configs:
    # Load training log (if saved)
    with open(f'models/{config}_log.json') as f:
        log = json.load(f)

    final_train_loss = log['train_losses'][-1]
    final_val_loss = log['val_losses'][-1]
    num_params = log['num_parameters']

    print(f"{config}:")
    print(f"  Train loss: {final_train_loss:.4f}")
    print(f"  Val loss: {final_val_loss:.4f}")
    print(f"  Overfitting: {(final_val_loss - final_train_loss):.4f}")
    print(f"  Parameters: {num_params:,}")
```

**Questions:**
1. Does larger network overfit? (check val_loss - train_loss)
2. Does dropout help? (reduces overfitting if present)
3. What's the parameter count trade-off? (larger = slower inference)

### Exercise 6: Complete End-to-End Pipeline (2 hours)

Run the full pipeline from scratch:

```bash
# Step 1: Generate training data (1000 scenarios)
python training/train_ga.py \
  --config training/config/ga_config.yaml \
  --output models/checkpoints/full_1000.pkl \
  --num_scenarios 1000 \
  --num_workers 8 \
  --checkpoint_interval 100

# Time: ~1-2 hours

# Step 2: Train neural network
python training/train_nn.py \
  --data models/checkpoints/full_1000.pkl \
  --config training/config/nn_config.yaml \
  --output models/planner_policy.onnx \
  --checkpoint models/best_model.pth

# Time: ~20-40 minutes

# Step 3: Copy to Docker container
docker cp models/planner_policy.onnx plan_ga_ros1:/models/

# Step 4: Build C++ plugin
docker exec -it plan_ga_ros1 bash
cd /catkin_ws
source /opt/ros/noetic/setup.bash && catkin_make
source devel/setup.bash

# Step 5: Test (if you have Stage simulator)
# roslaunch plan_ga_ros1 navigation.launch
```

**Validation checklist:**
- [ ] GA training completed without errors
- [ ] NN training converged (val_loss < 0.5)
- [ ] ONNX export verified (numerical match)
- [ ] C++ plugin compiles successfully
- [ ] Plugin loads in move_base
- [ ] (Optional) Navigation works in simulation

---

## Code Walkthrough

### GA Training Script

**File:** `training/train_ga.py`

Key functions:
```python
def generate_scenario(config, scenario_id):
    """Create random navigation scenario."""
    # 1. Generate random costmap with obstacles
    costmap = generate_random_costmap(
        width=50, height=50,
        num_obstacles=np.random.randint(3, 8)
    )

    # 2. Random goal position (1.5-3.0m away)
    goal_distance = np.random.uniform(1.5, 3.0)
    goal_angle = np.random.uniform(-np.pi, np.pi)
    goal = [goal_distance * np.cos(goal_angle),
            goal_distance * np.sin(goal_angle)]

    # 3. Create environment
    env = NavigationEnvironment(costmap, start=[0, 0], goal=goal)

    return env

def run_ga_on_scenario(env, config):
    """Evolve trajectory for one scenario."""
    # Initialize GA
    ga = GeneticAlgorithm(
        population_size=config['ga']['population_size'],
        mutation_rate=config['ga']['mutation_rate'],
        # ...
    )

    # Run evolution
    for generation in range(config['ga']['num_generations']):
        # Evaluate fitness (parallel)
        ga.evaluate(env)

        # Selection, crossover, mutation
        ga.evolve()

        # Check convergence
        if ga.converged():
            break

    # Return best trajectory
    best_chromosome = ga.get_best()
    return {
        'costmap': env.costmap,
        'control_sequence': best_chromosome.genes,
        'fitness': best_chromosome.fitness,
        # ...
    }
```

**Checkpoint mechanism:**
```python
# Every N scenarios, save progress
if scenario_num % checkpoint_interval == 0:
    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Checkpoint saved: {len(trajectories)} trajectories")

# Resume from checkpoint
if os.path.exists(output_path):
    with open(output_path, 'rb') as f:
        trajectories = pickle.load(f)
    print(f"Resuming from scenario {len(trajectories)}")
```

### NN Training Script

**File:** `training/train_nn.py`

Training loop:
```python
def train_epoch(model, dataloader, optimizer, criterion):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in dataloader:
        # Unpack batch
        costmap, robot_state, goal, metadata, target = batch

        # Forward pass
        prediction = model(costmap, robot_state, goal, metadata)
        loss = criterion(prediction, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# Main training loop
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss = validate(model, val_loader, criterion)

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), checkpoint_path)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
```

ONNX export:
```python
def export_to_onnx(model, output_path):
    """Export model to ONNX format."""
    model.eval()

    # Create dummy inputs
    dummy_costmap = torch.randn(1, 1, 50, 50)
    dummy_state = torch.randn(1, 9)
    dummy_goal = torch.randn(1, 3)
    dummy_metadata = torch.randn(1, 2)

    # Export
    torch.onnx.export(
        model,
        (dummy_costmap, dummy_state, dummy_goal, dummy_metadata),
        output_path,
        input_names=['costmap_input', 'robot_state_input',
                     'goal_relative_input', 'costmap_metadata_input'],
        output_names=['output'],
        dynamic_axes={'costmap_input': {0: 'batch'},
                      'robot_state_input': {0: 'batch'},
                      'goal_relative_input': {0: 'batch'},
                      'costmap_metadata_input': {0: 'batch'},
                      'output': {0: 'batch'}},
        opset_version=14
    )

    # Verify
    import onnxruntime as ort
    ort_session = ort.InferenceSession(output_path)

    # Compare outputs
    with torch.no_grad():
        pytorch_out = model(dummy_costmap, dummy_state, dummy_goal, dummy_metadata)

    ort_inputs = {
        'costmap_input': dummy_costmap.numpy(),
        'robot_state_input': dummy_state.numpy(),
        'goal_relative_input': dummy_goal.numpy(),
        'costmap_metadata_input': dummy_metadata.numpy()
    }
    ort_out = ort_session.run(None, ort_inputs)[0]

    diff = np.abs(pytorch_out.numpy() - ort_out).max()
    assert diff < 1e-5, f"Verification failed: max diff = {diff}"
    print(f"✓ ONNX export verified (max diff: {diff:.2e})")
```

---

## Quiz

1. **What does the GA optimize?**
   a) Neural network weights
   b) Control sequences (trajectories)
   c) Costmap obstacles
   d) Robot footprint

2. **Why filter low-fitness trajectories before NN training?**
   a) Reduce dataset size
   b) Train on good behaviors only (garbage in = garbage out)
   c) Speed up training
   d) Prevent overfitting

3. **What's the purpose of ONNX export?**
   a) Compress model size
   b) Enable Python/C++ interoperability
   c) Faster training
   d) Better accuracy

4. **What indicates overfitting?**
   a) Train loss > val loss
   b) Val loss > train loss significantly
   c) Both losses are high
   d) Both losses are low

5. **How many scenarios should you train on?**
   a) 10 (quick test)
   b) 100 (minimal)
   c) 1000+ (production)
   d) 10,000+ (overkill)

<details>
<summary><b>Show Answers</b></summary>

1. b) Control sequences (GA finds good trajectories for training data)
2. b) Train on good behaviors only (don't learn from failures)
3. b) Enable Python/C++ interoperability (train in PyTorch, deploy in C++)
4. b) Val loss > train loss significantly (model memorizing training data)
5. c) 1000+ for production (100 for testing, 10 for debugging)
</details>

---

## ✅ Checklist

- [ ] Run GA training successfully (100+ scenarios)
- [ ] Train neural network on GA data
- [ ] Export to ONNX and verify correctness
- [ ] Test ONNX model in C++ container
- [ ] Run hyperparameter experiments (GA and NN)
- [ ] Analyze training convergence
- [ ] Complete end-to-end pipeline (GA → NN → ONNX → C++)
- [ ] Understand data flow and model performance
- [ ] Quiz score 80%+

---

## 📚 Resources

- [Genetic Algorithms Theory](https://en.wikipedia.org/wiki/Genetic_algorithm)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/tune-performance.html)
- [Hyperparameter Tuning Best Practices](https://docs.ray.io/en/latest/tune/index.html)
- [Training/Validation Split Guidelines](https://developers.google.com/machine-learning/crash-course/training-and-test-sets/splitting-data)
- `training/GA_FUTURE_WORK.md` (10 enhancement ideas)

---

## 🎉 Next Steps

You can now train custom navigation policies from scratch!

Finally, explore capstone projects to extend the system with advanced features.

**→ [Continue to Module 09: Capstone Projects](../09-capstone/)**
