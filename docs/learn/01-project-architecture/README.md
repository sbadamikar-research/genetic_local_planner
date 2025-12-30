# Module 01: Project Architecture

**Estimated Time:** 1 day (6-8 hours)

## 🎯 Learning Objectives

- ✅ Understand the overall system architecture
- ✅ Navigate the codebase (~5000 lines)
- ✅ Understand the training → deployment pipeline
- ✅ Learn coding standards used in the project
- ✅ Trace data flow through the system

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Directory Structure](#directory-structure)
3. [Training Pipeline](#training-pipeline)
4. [Deployment Pipeline](#deployment-pipeline)
5. [Key Design Decisions](#key-design-decisions)
6. [Exercises](#exercises)

---

## 1. System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE                             │
│  ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐   │
│  │  Scenarios   │→  │ GA Evolution  │→  │ Optimal          │   │
│  │ Generation   │   │ (1000×50 gen) │   │ Trajectories     │   │
│  └──────────────┘   └───────────────┘   └──────────────────┘   │
│          │                                        │              │
│          v                                        v              │
│  ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐   │
│  │ Python       │   │ Multiprocess  │   │ Dataset          │   │
│  │ Simulator    │   │ Evaluation    │   │ (PKL file)       │   │
│  └──────────────┘   └───────────────┘   └──────────────────┘   │
│                                                   │              │
│                                                   v              │
│                          ┌───────────────────────────────┐      │
│                          │  Neural Network Training      │      │
│                          │  (PyTorch, supervised)        │      │
│                          └───────────────────────────────┘      │
│                                        │                         │
│                                        v                         │
│                          ┌───────────────────────────────┐      │
│                          │  ONNX Export                  │      │
│                          │  planner_policy.onnx          │      │
│                          └───────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    │ model file
                                    v
┌─────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT PHASE                            │
│  ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐   │
│  │  ROS         │→  │ C++ Plugin    │→  │ Robot Control    │   │
│  │  Costmap     │   │ (ONNX Runtime)│   │ Commands         │   │
│  └──────────────┘   └───────────────┘   └──────────────────┘   │
│          │                  │                     │              │
│          v                  v                     v              │
│  ┌──────────────┐   ┌───────────────┐   ┌──────────────────┐   │
│  │ move_base/   │   │ 10-20 Hz      │   │ cmd_vel topic    │   │
│  │ Nav2         │   │ inference     │   │ (Twist msgs)     │   │
│  └──────────────┘   └───────────────┘   └──────────────────┘   │
│                                                                  │
│  Container: Docker (ROS1 Noetic or ROS2 Humble)                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Language | Purpose | Lines |
|-----------|----------|---------|-------|
| GA Training | Python | Evolve optimal trajectories | ~900 |
| Simulator | Python | Fast environment for GA | ~1200 |
| NN Training | Python | Distill GA into network | ~500 |
| Core Planner | C++ | ROS-agnostic inference engine | ~1500 |
| ROS1 Plugin | C++ | move_base integration | ~800 |
| ROS2 Plugin | C++ | Nav2 integration | ~900 |

**Total:** ~5800 lines of production code

---

## 2. Directory Structure

```
plan_ga/
├── training/                    # Python training pipeline
│   ├── ga/                      # Genetic algorithm (4 files)
│   │   ├── chromosome.py        # Control sequence encoding
│   │   ├── fitness.py           # Multi-objective evaluation
│   │   ├── evolution.py         # GA main loop
│   │   └── operators.py         # Selection, crossover, mutation
│   ├── simulator/               # Pure Python simulator (4 files)
│   │   ├── costmap.py          # Procedural generation
│   │   ├── robot_model.py      # Dynamics/kinematics
│   │   ├── collision_checker.py # Footprint collision
│   │   └── environment.py      # Navigation wrapper
│   ├── neural_network/          # NN training (3 files)
│   │   ├── model.py            # CNN + MLP architecture
│   │   ├── dataset.py          # PyTorch dataset
│   │   └── __init__.py
│   ├── config/
│   │   ├── ga_config.yaml      # GA hyperparameters
│   │   └── nn_config.yaml      # NN architecture config
│   ├── train_ga.py             # GA training script
│   └── train_nn.py             # NN training script
│
├── src/                         # C++ deployment code
│   ├── plan_ga_planner/        # Core library (ROS-agnostic)
│   │   ├── include/plan_ga_planner/
│   │   │   ├── types.h         # Data structures
│   │   │   ├── onnx_inference.h # ONNX Runtime wrapper
│   │   │   ├── costmap_processor.h
│   │   │   ├── trajectory_generator.h
│   │   │   ├── collision_checker.h
│   │   │   └── planner_core.h  # Main orchestration
│   │   └── src/                # Implementations
│   ├── plan_ga_ros1/           # ROS1 plugin
│   │   ├── include/plan_ga_ros1/
│   │   │   └── plan_ga_ros1_plugin.h
│   │   ├── src/
│   │   │   └── plan_ga_ros1_plugin.cpp
│   │   ├── package.xml
│   │   ├── plan_ga_plugin.xml
│   │   └── CMakeLists.txt
│   └── plan_ga_ros2/           # ROS2 plugin (similar structure)
│
├── docker/                     # Development containers
│   ├── ros1/
│   │   ├── Dockerfile
│   │   ├── build.sh
│   │   ├── run.sh
│   │   ├── stop.sh
│   │   └── remove.sh
│   └── ros2/                   # (same structure)
│
├── models/
│   ├── checkpoints/            # GA training checkpoints
│   └── planner_policy.onnx     # Trained model (after training)
│
├── samples/
│   └── configs/
│       ├── planner_params_ros1.yaml
│       └── planner_params_ros2.yaml
│
├── docs/
│   └── learn/                  # This course!
│
├── CLAUDE.md                   # Project context
├── README.md                   # Quick start guide
└── environment.yml             # Conda environment
```

---

## 3. Training Pipeline

### Data Flow

```
1. Scenario Generation (train_ga.py)
   ├─> Random costmap (50×50 grid, 0.05m resolution)
   ├─> Start position (center, random orientation)
   └─> Goal position (1-3m away, random angle)

2. GA Evolution (ga/evolution.py)
   ├─> Initialize population (random control sequences)
   ├─> For each generation:
   │   ├─> Evaluate fitness (parallel, 8 workers)
   │   ├─> Select parents (tournament)
   │   ├─> Crossover + Mutation
   │   └─> Keep elites
   └─> Return best chromosome

3. Fitness Evaluation (ga/fitness.py + simulator/)
   ├─> Simulate trajectory (environment.py)
   ├─> Check collisions (collision_checker.py)
   ├─> Compute metrics:
   │   ├─> Goal distance
   │   ├─> Collision penalty
   │   ├─> Smoothness
   │   └─> Path length
   └─> Weighted fitness score

4. Dataset Creation (train_ga.py)
   ├─> Extract 50×50 costmap window
   ├─> Normalize robot state
   ├─> Compute goal in robot frame
   ├─> Save as PKL: {costmap, state, goal, controls, fitness}
   └─> Repeat for 1000+ scenarios

5. NN Training (train_nn.py)
   ├─> Load trajectories from PKL
   ├─> Filter low-fitness samples (bottom 25%)
   ├─> Train/val split (80/20)
   ├─> Train with MSE loss
   ├─> Early stopping on validation loss
   └─> Export to ONNX

6. ONNX Export
   ├─> Define input names (costmap_input, robot_state_input, ...)
   ├─> Define output name (output)
   ├─> Set opset version 14
   └─> Save to models/planner_policy.onnx
```

### Configuration Files

**ga_config.yaml:**
```yaml
ga:
  population_size: 100
  elite_size: 10
  mutation_rate: 0.1
  crossover_rate: 0.8
  num_generations: 50
  time_horizon: 2.0
  control_frequency: 10.0

fitness_weights:
  goal_distance: 1.0
  collision: 10.0
  smoothness: 0.5
  time_efficiency: 0.3

robot:
  footprint: [[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]]
  max_v_x: 1.0
  min_v_x: -0.5
  max_v_y: 0.5
  min_v_y: -0.5
  max_omega: 1.0
  min_omega: -1.0
```

**nn_config.yaml:**
```yaml
model:
  costmap_size: 50
  num_control_steps: 20
  hidden_dim: 256
  cnn:
    channels: [1, 32, 64, 128]
    kernel_sizes: [5, 3, 3]
    strides: [2, 2, 2]
  mlp:
    input_dim: 14  # 9 (state) + 3 (goal) + 2 (metadata)
    hidden_dims: [128, 256]
  policy_head:
    hidden_dims: [256, 256]
    output_dim: 60  # 20 steps × 3 controls

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  early_stopping_patience: 10
  filter_percentile: 25  # Remove bottom 25% by fitness
  train_split: 0.8
```

---

## 4. Deployment Pipeline

### C++ Code Architecture

```
┌──────────────────────────────────────────────────────────┐
│                ROS Plugin (ros1 or ros2)                 │
│  ┌────────────────────────────────────────────────────┐  │
│  │ - Implements BaseLocalPlanner (ROS1)               │  │
│  │   or Controller (ROS2)                             │  │
│  │ - Reads ROS costmap                                │  │
│  │ - Publishes cmd_vel                                │  │
│  │ - Handles lifecycle (configure/activate/cleanup)   │  │
│  └────────────────────────────────────────────────────┘  │
│                          │                               │
│                          v                               │
│  ┌────────────────────────────────────────────────────┐  │
│  │            PlannerCore (core library)              │  │
│  │  ┌──────────────────────────────────────────────┐  │  │
│  │  │ computeVelocity():                           │  │  │
│  │  │  1. Extract costmap window (50×50)           │  │  │
│  │  │  2. Prepare model inputs                     │  │  │
│  │  │  3. Call ONNX inference                      │  │  │
│  │  │  4. Decode control sequence                  │  │  │
│  │  │  5. Simulate trajectory                      │  │  │
│  │  │  6. Check collisions                         │  │  │
│  │  │  7. Return first control                     │  │  │
│  │  └──────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────┘  │
│                          │                               │
│                          v                               │
│  ┌────────────────────────────────────────────────────┐  │
│  │          ONNXInference (onnx_inference.h)          │  │
│  │  - Loads .onnx model file                          │  │
│  │  - Creates ONNX Runtime session                    │  │
│  │  - Prepares input tensors                          │  │
│  │  - Runs inference                                  │  │
│  │  - Extracts output tensors                         │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  Supporting modules:                                     │
│  - CostmapProcessor: 50×50 window extraction            │
│  - TrajectoryGenerator: Forward simulation              │
│  - CollisionChecker: Footprint-based validation         │
└──────────────────────────────────────────────────────────┘
```

### Key Data Structures (types.h)

```cpp
struct Pose {
    double x, y, theta;
};

struct Velocity {
    double v_x, v_y, omega;
};

struct Costmap {
    std::vector<uint8_t> data;  // Flattened 2D grid
    int width, height;
    double resolution;
    double origin_x, origin_y;
    double inflation_decay;
};

struct ControlStep {
    double v_x, v_y, omega;
    double dt;
};

using ControlSequence = std::vector<ControlStep>;
```

---

## 5. Key Design Decisions

### Why This Architecture?

**1. Separate Training from Deployment**
- **Reason:** Python for fast prototyping, C++ for real-time performance
- **Benefit:** Best of both worlds
- **Trade-off:** Need cross-language integration (ONNX)

**2. ROS-Agnostic Core Library**
- **Reason:** Share code between ROS1 and ROS2
- **Benefit:** Write once, deploy twice
- **Implementation:** `plan_ga_planner/` has no ROS dependencies

**3. Multi-Objective Fitness Function**
- **Reason:** Balance goal reaching, safety, smoothness, efficiency
- **Implementation:** Weighted sum with tunable weights
- **Alternative:** Pareto optimization (see GA_FUTURE_WORK.md)

**4. ONNX for Model Export**
- **Reason:** Cross-platform, well-supported, fast
- **Alternatives considered:**
  - PyTorch C++ API: Too complex, large binary size
  - TensorFlow Lite: Less flexible than ONNX Runtime
  - Direct Python calls: Too slow for 10-20 Hz

**5. Docker for Development**
- **Reason:** ROS installation is complex and system-dependent
- **Benefit:** Reproducible environments, easy ROS1/ROS2 switching
- **Trade-off:** Slight overhead, learning curve

**6. Parallel GA Evaluation**
- **Reason:** Fitness evaluation is embarrassingly parallel
- **Implementation:** Python multiprocessing (8 workers)
- **Speedup:** ~6x on 8-core CPU

---

## 6. Exercises

### Exercise 1: Codebase Exploration (30 min)

Run these commands and analyze output:

```bash
# Count lines per component
echo "=== GA Components ===" && find training/ga -name "*.py" | xargs wc -l | tail -1
echo "=== Simulator ===" && find training/simulator -name "*.py" | xargs wc -l | tail -1
echo "=== Neural Network ===" && find training/neural_network -name "*.py" | xargs wc -l | tail -1
echo "=== C++ Core ===" && find src/plan_ga_planner -name "*.cpp" -o -name "*.h" | xargs wc -l | tail -1
echo "=== ROS1 Plugin ===" && find src/plan_ga_ros1 -name "*.cpp" -o -name "*.h" | xargs wc -l | tail -1
echo "=== ROS2 Plugin ===" && find src/plan_ga_ros2 -name "*.cpp" -o -name "*.hpp" | xargs wc -l | tail -1

# Find all main entry points
find . -name "train_*.py"
find . -name "*_plugin.cpp" -o -name "*_plugin.hpp"

# List all config files
find . -name "*.yaml"
```

**Questions:**
1. Which component has the most code? Why?
2. How many configuration files are there?
3. What's the naming pattern for plugins?

### Exercise 2: Trace a Control Sequence (45 min)

Follow the data flow of a control sequence:

**Step 1:** Start in `training/train_ga.py` (line ~307)
```python
best_chromosome, fitness_history = ga.run(environment, ...)
```

**Step 2:** Jump to `training/ga/evolution.py` (line ~158)
```python
for generation in range(num_generations):
    population = self.evolve_generation(population, environment, num_workers)
```

**Step 3:** Look at `evolve_generation` (line ~81)
```python
evaluate_population_parallel(population, environment, self.fitness_evaluator, num_workers)
```

**Step 4:** Check `training/ga/fitness.py` (line ~151)
```python
def evaluate_population_parallel(population, environment, evaluator, num_workers):
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_evaluate_worker, args_list)
```

**Step 5:** Examine `_evaluate_worker` (line ~120)
```python
def _evaluate_worker(args):
    chromosome_genes, velocity_limits, num_steps, env_state, weights = args
    chromosome = Chromosome(num_steps, velocity_limits)
    chromosome.genes = chromosome_genes
    environment = NavigationEnvironment(config)
    environment.reset(costmap, start_state, goal_x, goal_y, goal_theta)
    evaluator = FitnessEvaluator(weights)
    fitness = evaluator.evaluate(chromosome, environment)
    return fitness, chromosome.fitness_components
```

**Questions:**
1. Why use multiprocessing for fitness evaluation?
2. What data needs to be serialized for worker processes?
3. Where does the chromosome's control sequence get executed?

**Hint:** Check `training/simulator/environment.py::simulate_control_sequence`

### Exercise 3: Model Interface Verification (30 min)

Verify ONNX input/output matches C++ expectations:

**Python side** (training/neural_network/model.py:~254):
```python
def forward(self, costmap, robot_state, goal_relative, costmap_metadata):
    # ... processing ...
    return control_sequence  # [batch, 60]
```

**ONNX export** (training/train_nn.py):
```python
input_names = ['costmap_input', 'robot_state_input', 'goal_relative_input', 'costmap_metadata_input']
output_names = ['output']
```

**C++ side** (src/plan_ga_planner/include/plan_ga_planner/onnx_inference.h):
```cpp
std::vector<std::string> input_names = {
    "costmap_input", "robot_state_input", 
    "goal_relative_input", "costmap_metadata_input"
};
std::vector<std::string> output_names = {"output"};
```

**Task:** Create a diagram showing data shapes at each stage:
1. Python tensors → ONNX → C++ tensors
2. Label all dimensions

### Exercise 4: Configuration Tuning Experiment (1 hour)

Modify `training/config/ga_config.yaml`:

**Experiment 1:** Population size
```yaml
# Try these values:
population_size: 50   # Small
population_size: 100  # Medium
population_size: 200  # Large
```

**Experiment 2:** Fitness weights
```yaml
# Default
fitness_weights:
  goal_distance: 1.0
  collision: 10.0
  
# Aggressive
fitness_weights:
  goal_distance: 2.0
  collision: 20.0
  
# Conservative
fitness_weights:
  goal_distance: 0.5
  collision: 5.0
```

**Run (don't actually run yet, just plan):**
```bash
python training/train_ga.py --config configs/ga_experiment1.yaml --output results/exp1.pkl --num_scenarios 10
```

**Questions:**
1. What do you expect to happen with larger populations?
2. How will aggressive fitness weights affect behavior?
3. What metrics would you track to compare results?

### Exercise 5: Code Review Challenge (45 min)

Review this code from `training/ga/operators.py` (line ~41):

```python
def uniform_crossover(parent1: Chromosome, parent2: Chromosome,
                     crossover_rate: float = 0.8) -> Chromosome:
    offspring = parent1.copy()
    
    if np.random.random() < crossover_rate:
        mask = np.random.random(parent1.genes.shape) < 0.5
        offspring.genes[mask] = parent2.genes[mask]
    
    offspring.fitness = -np.inf
    offspring.fitness_components = {}
    
    return offspring
```

**Questions:**
1. Why reset fitness to `-np.inf` after crossover?
2. What does the `mask` do?
3. If `crossover_rate=0`, what gets returned?
4. How would you modify this for 3-parent crossover?

---

## Quiz

1. **How many main components are in the system?**
   a) 3 (GA, NN, C++)
   b) 4 (GA, Simulator, NN, C++)
   c) 6 (GA, Simulator, NN, Core, ROS1, ROS2)
   d) 2 (Python, C++)

2. **What is the ONNX model's output shape?**
   a) [1, 20, 3]
   b) [1, 60]
   c) [batch, 20, 3]
   d) [batch, 60]

3. **Why is the core planner ROS-agnostic?**
   a) To avoid ROS dependencies
   b) To share code between ROS1 and ROS2
   c) To enable unit testing
   d) All of the above

4. **Where does parallel execution happen?**
   a) C++ ONNX inference
   b) Python GA fitness evaluation
   c) Docker containers
   d) Neural network training

5. **What format is used for configuration?**
   a) JSON
   b) XML
   c) YAML
   d) TOML

<details>
<summary><b>Show Answers</b></summary>

1. c) 6 components
2. d) [batch, 60] (flattened)
3. d) All of the above
4. b) Python GA fitness evaluation (multiprocessing)
5. c) YAML
</details>

---

## ✅ Checklist

- [ ] Understand high-level architecture diagram
- [ ] Can navigate directory structure confidently
- [ ] Traced data flow from training to deployment
- [ ] Understand why ONNX is used
- [ ] Completed all exercises
- [ ] Quiz score 80%+

---

## 📚 Further Reading

- [ONNX Documentation](https://onnx.ai/onnx/)
- [ROS Navigation Stack](http://wiki.ros.org/navigation)
- [PyTorch to ONNX Guide](https://pytorch.org/docs/stable/onnx.html)

---

## 🎉 Next Steps

You now understand how the pieces fit together! Time to dive into genetic algorithms.

**→ [Continue to Module 02: Genetic Algorithms](../02-genetic-algorithms/)**
