# Claude Code Session Context

**Project**: GA-Based ROS Local Planner
**Last Updated**: 2025-12-26
**Status**: C++ Complete, Training Pipeline Complete, Awaiting Execution

---

## Project Overview

A hybrid genetic algorithm + neural network local planner for ROS robots. The system trains navigation policies using GA with Stage simulator, distills them into neural networks via ONNX, and deploys as high-performance C++ plugins for both ROS1 Noetic and ROS2 Humble.

**Architecture**:
- **Training** (Python on host): GA → NN distillation → ONNX export
- **Deployment** (C++ in Docker): ONNX Runtime inference in ROS plugins
- **Frequency**: 10-20 Hz planning rate
- **Input**: 50x50 costmap window, robot state, goal relative position
- **Output**: Control sequence (20 steps of v_x, v_y, omega)

---

## Current Status

### ✅ Completed

#### 1. C++ Core Planner (ROS-Agnostic)
**Location**: `src/plan_ga_planner/`
**Files**: 6 headers + 5 implementations (~1500 lines)

- `types.h`: Data structures (Pose, Velocity, Costmap, ControlSequence, etc.)
- `onnx_inference.h/cpp`: ONNX Runtime wrapper for model loading/inference
- `costmap_processor.h/cpp`: 50x50 window extraction and normalization
- `trajectory_generator.h/cpp`: Forward simulation with Euler integration
- `collision_checker.h/cpp`: Footprint-based collision detection
- `planner_core.h/cpp`: Main orchestration interface

**Key Features**:
- ONNX Runtime 1.16.3 integration
- Multi-step control sequence generation
- Differential drive kinematics
- Optional collision validation
- Goal reached checking

#### 2. ROS1 Noetic Plugin
**Location**: `src/plan_ga_ros1/`
**Files**: 1 header + 1 implementation + 3 config files (~800 lines)

- `plan_ga_ros1_plugin.h/cpp`: Implements `nav_core::BaseLocalPlanner`
- `package.xml`: Catkin package manifest
- `plan_ga_plugin.xml`: Plugin descriptor for pluginlib
- `CMakeLists.txt`: Catkin build configuration

**Integration**: Drop-in replacement for DWA/TEB in move_base

#### 3. ROS2 Humble Plugin
**Location**: `src/plan_ga_ros2/`
**Files**: 1 header + 1 implementation + 3 config files (~900 lines)

- `plan_ga_ros2_plugin.hpp/cpp`: Implements `nav2_core::Controller`
- Full lifecycle node support (configure/activate/deactivate/cleanup)
- Dynamic speed limit interface
- `package.xml`: Ament package manifest
- `CMakeLists.txt`: Colcon build configuration

**Integration**: Compatible with Nav2 controller_server

#### 4. Docker Infrastructure
**Location**: `docker/`

**ROS1 Container** (`docker/ros1/`):
- Dockerfile: ros:noetic-ros-core + nav-core + ONNX Runtime 1.16.3
- `build.sh`: Image building with DNS configuration
- `run.sh`: Start container in detached mode
- `stop.sh`: Stop container gracefully
- `remove.sh`: Remove container completely

**ROS2 Container** (`docker/ros2/`):
- Dockerfile: ros:humble-ros-core + nav2-core + ONNX Runtime 1.16.3
- Same script set as ROS1

**Volume Mounts**:
- `src/plan_ga_planner/` → `/{workspace}/src/plan_ga/plan_ga_planner`
- `src/plan_ga_ros{1|2}/` → `/{workspace}/src/plan_ga/plan_ga_ros{1|2}`
- `models/` → `/models` (for ONNX model)
- `samples/` → `/samples` (configs/launch files)

**Network**: Host mode for ROS communication

#### 5. VS Code Dev Container Configurations
**Location**: `.devcontainer/`

- `devcontainer.json`: Default (ROS1)
- `ros1/devcontainer.json`: ROS1 Noetic config
- `ros2/devcontainer.json`: ROS2 Humble config
- `README.md`: Complete usage guide

**Features**:
- Automatic extension installation (C++, Python, CMake, Git Graph)
- Pre-configured IntelliSense with correct include paths
- Lifecycle automation (auto-sourcing ROS)
- Workspace: `/catkin_ws` or `/ros2_ws`

#### 6. Documentation
- `README.md`: Quick start guide
- `docs/development_plan.md`: Complete deployment guide with VS Code integration
- `docs/training_plan.md`: GA/NN training protocol (created earlier)
- `CPP_IMPLEMENTATION_STATUS.md`: Implementation status tracker
- `docker/README.md`: Container management reference
- `.devcontainer/README.md`: Dev Container usage guide

#### 7. Configuration Files
- `samples/configs/planner_params_ros1.yaml`: ROS1 parameters
- `samples/configs/planner_params_ros2.yaml`: ROS2 parameters (aligned with implementation)
- `training/config/ga_config.yaml`: GA hyperparameters
- `training/config/nn_config.yaml`: NN architecture config
- `environment.yml`: Conda environment for training

#### 8. Genetic Algorithm Training
**Location**: `training/ga/`
**Files**: 4 modules + main script (~900 lines)

- `chromosome.py` (276 lines): Control sequence encoding with validation
  - Direct gene encoding (20 steps × 3 DOF)
  - Velocity limit clamping
  - Fitness tracking and comparison operators
  - Deep copy support for genetic operations
- `fitness.py` (345 lines): Multi-objective fitness evaluation
  - Weighted fitness: goal_distance + collision + smoothness + time_efficiency
  - Parallel evaluation with multiprocessing (8 workers)
  - Goal reached bonus, collision penalties
  - Detailed fitness component tracking
- `evolution.py` (343 lines): Complete GA evolution loop
  - Population initialization and management
  - Generation evolution (evaluate → select → crossover → mutate)
  - Elitism preservation
  - Statistics tracking and convergence monitoring
- `operators.py` (335 lines): Genetic operators
  - Tournament selection (tournament_size=5)
  - Uniform and single-point crossover
  - Gaussian and uniform mutation
  - Elitism selection for best chromosomes
  - Roulette wheel selection (fitness proportionate)

**Key Features**:
- Parallel fitness evaluation (multiprocessing)
- Configurable population size, mutation rate, crossover rate
- Multi-objective fitness with tunable weights
- Comprehensive test code in each module

#### 9. Python Simulator
**Location**: `training/simulator/`
**Files**: 4 modules (~1200 lines)

- `costmap.py`: Costmap generation and inflation
  - Random obstacle placement with configurable density
  - Distance-based cost inflation with exponential decay
  - 50×50 grid at 0.05m resolution
  - Procedural scenario generation
- `robot_model.py`: Robot dynamics and kinematics
  - RobotState: Full state representation (pose, velocities, accelerations)
  - RobotModel: Forward dynamics with Euler integration
  - Differential and omnidirectional drive support
  - Velocity limit enforcement
- `collision_checker.py`: Footprint-based collision detection
  - Polygon footprint representation
  - Grid-based collision checking
  - Lethal (254) and inscribed (253) thresholds
  - Efficient batch checking
- `environment.py`: Navigation environment wrapper
  - Complete scenario management (costmap, start, goal)
  - Control sequence simulation
  - Trajectory generation with collision detection
  - Fitness metrics computation (goal distance, smoothness, path length)
  - 50×50 costmap window extraction for NN training

**Key Features**:
- ROS-independent (pure Python/NumPy)
- Fast simulation for parallel GA evaluation
- Procedural scenario generation for diverse training
- Direct compatibility with NN training data format

#### 10. GA Training Script
**Location**: `training/train_ga.py`
**Lines**: 343

Main training orchestration script with:
- Scenario generation with configurable difficulty
- Parallel GA evolution (configurable workers)
- Periodic checkpointing (every N scenarios)
- Resume capability from checkpoints
- Statistics tracking (fitness, goal distance, collision rate)
- Trajectory export in NN-ready format (costmap, robot_state, goal_relative, control_sequence)

**Usage**:
```bash
python training/train_ga.py \
  --config training/config/ga_config.yaml \
  --output models/checkpoints/ga_trajectories.pkl \
  --num_scenarios 1000 \
  --num_workers 8 \
  --checkpoint_interval 100
```

#### 11. Training Documentation
- `training/GA_FUTURE_WORK.md`: Comprehensive guide for 10 future enhancements
  - Real-time visualization (Matplotlib/Pygame)
  - TensorBoard logging
  - Stage simulator integration
  - Adaptive parameters (mutation/crossover)
  - Multi-objective Pareto optimization (NSGA-II)
  - Convergence detection and early stopping
  - Advanced operators (BLX-α, polynomial mutation)
  - Curriculum learning
  - Coevolution
  - Hyperparameter tuning with Optuna

#### 12. Neural Network Training
**Location**: `training/neural_network/`
**Files**: 3 modules + main script (~500 lines)

- `model.py` (381 lines): CNN + MLP architecture for trajectory prediction
  - CostmapEncoder: CNN for 50×50 costmap (4 conv layers → 256D features)
  - StateEncoder: MLP for robot state + goal + metadata (14D → 256D)
  - PolicyHead: Fusion MLP with residual connections (512D → 60D output)
  - PlannerPolicy: Complete model matching ONNX interface
  - 4 separate inputs, 1 output (20 control steps × 3 DOF flattened)
- `dataset.py` (330 lines): PyTorch dataset for GA trajectories
  - TrajectoryDataset: Loads from pickle with fitness filtering
  - Train/validation splitting with random seed
  - Automatic shape validation and tensor conversion
  - Dataset statistics computation
  - Synthetic data generation for testing
- `train_nn.py` (main script): Complete training pipeline
  - Training loop with validation
  - Early stopping and learning rate scheduling
  - Model checkpointing (save best model)
  - ONNX export with input/output naming
  - Progress tracking with tqdm
- `__init__.py`: Module exports

**Key Features**:
- Multi-input architecture matching C++ ONNX interface
- Batch normalization and dropout for regularization
- Residual connections in policy head
- MSE loss for regression task
- Adam optimizer with learning rate scheduling
- Configurable via YAML (nn_config.yaml)
- ONNX export with opset 14

**Usage**:
```bash
python training/train_nn.py \
  --data models/checkpoints/ga_trajectories.pkl \
  --config training/config/nn_config.yaml \
  --output models/planner_policy.onnx \
  --checkpoint models/checkpoints/best_model.pth
```

### ⏳ Pending

#### 1. Training Execution
**Goal**: Run complete training pipeline and generate ONNX model

**Tasks**:
1. Run GA training to collect 1000+ trajectories
2. Train NN to distill GA behavior
3. Verify ONNX export and model performance

#### 2. ONNX Model
**Location**: `models/`
**Status**: `checkpoints/` directory exists but empty

**Missing**: `models/planner_policy.onnx` (required for C++ plugin runtime)

#### 3. Testing
- Build verification in Docker containers (not yet tested)
- Unit tests for C++ components (none written)
- Integration tests with Stage simulator (not yet done)
- Performance profiling (pending trained model)

---

## Development Environment

### Host Machine
- **OS**: Linux 6.8.0-87-generic
- **Location**: `/home/ANT.AMAZON.COM/basancht/plan_ga`
- **Network**: Amazon corporate network (DNS: 10.4.4.10)
- **Python**: Miniconda environment `plan_ga`
- **Docker**: Installed, daemon configured for corporate network

### Docker Images Built
- `plan_ga_ros1:latest` - ROS1 Noetic development
- `plan_ga_ros2:latest` - ROS2 Humble development

### Git Status
- **Branch**: main
- **Recent Commits** (last 10):
  1. docs: update development workflow for background containers
  2. docs(docker): add comprehensive container management guide
  3. feat(docker): update container scripts for background execution
  4. feat(devcontainer): add VS Code Dev Container configurations
  5. docs(readme): update ROS2 build instructions with verification
  6. config(ros2): update planner parameters to match implementation
  7. docs: add C++ implementation status tracking document
  8. feat(ros2): add ROS2 Humble controller plugin
  9. feat(ros1): add ROS1 Noetic local planner plugin
  10. feat(planner): add ROS-agnostic core planner library

### Uncommitted Changes
- New: `training/neural_network/__init__.py` (module exports)
- Modified: `CLAUDE.md` (updated with NN training completion)

---

## Key Technical Details

### Coding Standards (Strictly Followed)
- **Functions**: camelCase (e.g., `computeVelocity`, `isGoalReached`)
- **Variables**: snake_case (e.g., `current_pose`, `control_sequence`)
- **Classes/Structs**: ProperCase (e.g., `PlannerCore`, `ONNXInference`)
- **Member variables**: trailing_underscores_ (e.g., `config_`, `initialized_`)

### Control Sequence Format
- 20 control steps (configurable)
- Each step: `{v_x, v_y, omega, dt}`
- Default: 10 Hz control frequency, 2.0s time horizon

### ONNX Model Interface
**Inputs** (4 tensors):
1. `costmap_input`: [1, 1, 50, 50] - normalized costmap window
2. `robot_state_input`: [1, 9] - [x, y, theta, v_x, v_y, omega, a_x, a_y, a_theta]
3. `goal_relative_input`: [1, 3] - [dx, dy, dtheta] in robot frame
4. `costmap_metadata_input`: [1, 2] - [resolution, inflation_decay]

**Output** (1 tensor):
- `output`: [1, 60] - flattened control sequence (20 steps × 3 values)

### Build Commands

**ROS1**:
```bash
cd /catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
rospack plugins --attrib=plugin nav_core | grep plan_ga
```

**ROS2**:
```bash
cd /ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
ros2 pkg list | grep plan_ga
```

### Docker Workflow

**Start container** (background):
```bash
cd docker/ros1  # or docker/ros2
./run.sh
```

**Attach**:
```bash
docker exec -it plan_ga_ros1 bash  # or plan_ga_ros2
# OR use VS Code: F1 → "Attach to Running Container"
```

**Stop/Remove**:
```bash
./stop.sh    # Stop but preserve build artifacts
./remove.sh  # Complete removal
```

---

## Important Decisions Made

### 1. Architecture Split
- **Rationale**: Separate training (Python/host) from deployment (C++/Docker)
- **Benefit**: Training flexibility + deployment performance

### 2. ROS-Agnostic Core
- **Rationale**: Share implementation between ROS1 and ROS2
- **Location**: `src/plan_ga_planner/`
- **Benefit**: Single codebase, easier maintenance

### 3. ONNX for Model Transfer
- **Rationale**: Cross-platform, well-supported, fast inference
- **Version**: ONNX Runtime 1.16.3
- **Alternative Considered**: Direct PyTorch C++ (rejected: complexity)

### 4. Background Docker Containers
- **Rationale**: Better VS Code integration, persistent build artifacts
- **Change**: From `-it --rm` to `-d` with explicit lifecycle
- **Benefit**: Containers survive terminal closes, faster restarts

### 5. Minimal Extensions
- **Rationale**: User requested Microsoft-only + Git Graph
- **List**: cpptools, cmake-tools, python, git-graph
- **Removed**: ROS extension, GitLens, spell checker, etc.

### 6. Workspace Mounting Strategy
- **Rationale**: Preserve ROS workspace structure, add project access
- **Approach**: Mount individual directories, not entire project root
- **Workspace Folder**: `/catkin_ws` or `/ros2_ws` (not overwritten)

### 7. DNS Configuration
- **Issue**: Corporate network DNS resolution failures
- **Solution**: `--network host` in Docker build
- **DNS Servers**: 10.4.4.10 (corporate), 8.8.8.8, 1.1.1.1 (fallbacks)

---

## Known Issues

### 1. Docker Build DNS
- **Issue**: snapshots.ros.org resolution fails on corporate network
- **Solution**: Docker daemon.json DNS config + --network host
- **Status**: Resolved

### 2. Non-Homogeneous Workspace
- **Issue**: Both ROS1 and ROS2 packages in src/ caused catkin error
- **Solution**: Separate volume mounts for each ROS version
- **Status**: Resolved

### 3. No GPU Required
- **Clarified**: Entire project runs on CPU
- **Training**: GA is CPU-based, NN training optional GPU
- **Deployment**: ONNX Runtime CPU inference at 10-20 Hz

---

## Next Steps (Priority Order)

### 1. Execute Training Pipeline
**Goal**: Generate trained ONNX model from scratch

**Tasks**:
1. Run GA training to collect 1000+ navigation scenarios
   ```bash
   python training/train_ga.py --config training/config/ga_config.yaml \
     --output models/checkpoints/ga_trajectories.pkl --num_scenarios 1000
   ```
2. Train neural network to distill GA behavior
   ```bash
   python training/train_nn.py --data models/checkpoints/ga_trajectories.pkl \
     --config training/config/nn_config.yaml --output models/planner_policy.onnx
   ```
3. Verify ONNX model exports correctly and matches C++ interface

**Estimated Effort**: Large (GA training ~hours depending on CPU cores, NN training ~30min-1hr)

### 2. Verify C++ Build
**Goal**: Confirm plugins compile successfully

**Tasks**:
1. Start ROS1 container: `cd docker/ros1 && ./run.sh`
2. Build: `catkin_make`
3. Verify plugin: `rospack plugins --attrib=plugin nav_core | grep plan_ga`
4. Repeat for ROS2 with `colcon build`

**Estimated Effort**: Small (likely works, minor fixes possible)

### 3. Create Launch Files
**Goal**: Enable easy testing with Stage simulator

**Tasks**:
1. Create `samples/launch/test_planner_ros1.launch`
2. Create `samples/launch/test_planner_ros2.launch.py`
3. Configure move_base/Nav2 to use plan_ga plugin
4. Set up Stage world files

**Estimated Effort**: Medium

### 4. Integration Testing
**Goal**: Verify end-to-end functionality

**Tasks**:
1. Load trained ONNX model
2. Launch Stage simulator
3. Launch move_base/Nav2 with plan_ga plugin
4. Send navigation goals
5. Verify trajectory planning and execution

**Estimated Effort**: Medium (depends on model quality)

### 5. Performance Profiling
**Goal**: Ensure 10-20 Hz target met

**Tasks**:
1. Measure inference time per planning cycle
2. Profile CPU usage
3. Optimize if needed (likely not necessary)

**Estimated Effort**: Small

---

## File Structure Summary

```
plan_ga/
├── .devcontainer/           # VS Code Dev Container configs (ROS1/ROS2)
├── docker/                  # Docker infrastructure
│   ├── ros1/               # ROS1 Noetic container
│   │   ├── Dockerfile      # Base image + dependencies
│   │   ├── build.sh        # Image building
│   │   ├── run.sh          # Start in background
│   │   ├── stop.sh         # Stop container
│   │   └── remove.sh       # Remove container
│   ├── ros2/               # ROS2 Humble container (same structure)
│   └── README.md           # Container management guide
├── docs/
│   ├── development_plan.md # Complete deployment guide + VS Code setup
│   └── training_plan.md    # GA/NN training protocol
├── models/
│   └── checkpoints/        # GA training checkpoints (empty)
│   # Missing: planner_policy.onnx (required for runtime)
├── samples/
│   └── configs/
│       ├── planner_params_ros1.yaml
│       └── planner_params_ros2.yaml
├── src/
│   ├── plan_ga_planner/    # Core library (ROS-agnostic, ~1500 lines)
│   │   ├── include/plan_ga_planner/
│   │   │   ├── types.h
│   │   │   ├── onnx_inference.h
│   │   │   ├── costmap_processor.h
│   │   │   ├── trajectory_generator.h
│   │   │   ├── collision_checker.h
│   │   │   └── planner_core.h
│   │   └── src/            # Implementations
│   ├── plan_ga_ros1/       # ROS1 plugin (~800 lines)
│   │   ├── include/plan_ga_ros1/
│   │   │   └── plan_ga_ros1_plugin.h
│   │   ├── src/
│   │   │   └── plan_ga_ros1_plugin.cpp
│   │   ├── package.xml
│   │   ├── plan_ga_plugin.xml
│   │   └── CMakeLists.txt
│   └── plan_ga_ros2/       # ROS2 plugin (~900 lines)
│       ├── include/plan_ga_ros2/
│       │   └── plan_ga_ros2_plugin.hpp
│       ├── src/
│       │   └── plan_ga_ros2_plugin.cpp
│       ├── package.xml
│       ├── plan_ga_plugin.xml
│       └── CMakeLists.txt
├── training/               # Python training code (COMPLETE)
│   ├── ga/                 # GA components (~900 lines)
│   │   ├── chromosome.py   # Control sequence encoding
│   │   ├── fitness.py      # Multi-objective evaluation
│   │   ├── evolution.py    # GA evolution loop
│   │   ├── operators.py    # Selection, crossover, mutation
│   │   └── __init__.py     # Module exports
│   ├── simulator/          # Python simulator (~1200 lines)
│   │   ├── costmap.py      # Procedural generation with inflation
│   │   ├── robot_model.py  # Dynamics and kinematics
│   │   ├── collision_checker.py  # Footprint-based detection
│   │   ├── environment.py  # Navigation wrapper
│   │   └── __init__.py     # Module exports
│   ├── neural_network/     # NN training (~500 lines)
│   │   ├── model.py        # CNN + MLP architecture
│   │   ├── dataset.py      # PyTorch dataset loader
│   │   └── __init__.py     # Module exports
│   ├── config/
│   │   ├── ga_config.yaml  # GA hyperparameters
│   │   └── nn_config.yaml  # NN architecture config
│   ├── train_ga.py         # GA training script (343 lines)
│   ├── train_nn.py         # NN training script (main)
│   └── GA_FUTURE_WORK.md   # Enhancement guide (765 lines)
├── CLAUDE.md               # This file
├── CPP_IMPLEMENTATION_STATUS.md
├── README.md
└── environment.yml         # Conda environment for training
```

---

## How to Resume Work

### If Continuing C++ Development:
1. Start container: `cd docker/ros1 && ./run.sh`
2. Attach VS Code: F1 → "Attach to Running Container" → plan_ga_ros1
3. Or terminal: `docker exec -it plan_ga_ros1 bash`
4. Build: `cd /catkin_ws && catkin_make`

### If Executing Training Pipeline:
1. Activate conda: `conda activate plan_ga`
2. Run GA training: `python training/train_ga.py --config training/config/ga_config.yaml --output models/checkpoints/ga_trajectories.pkl --num_scenarios 1000 --num_workers 8`
3. Run NN training: `python training/train_nn.py --data models/checkpoints/ga_trajectories.pkl --config training/config/nn_config.yaml --output models/planner_policy.onnx`
4. Verify ONNX model: Check `models/planner_policy.onnx` exists and has correct interface

### If Testing Integration:
1. First ensure ONNX model exists: `models/planner_policy.onnx`
2. Start container and build
3. Create launch file in `samples/launch/`
4. Test with Stage simulator

---

## Questions to Consider

1. **GA Fitness Function**: Should we add more objectives or weights?
2. **NN Architecture**: CNN size sufficient for 50x50 costmap?
3. **Control Frequency**: Is 10 Hz adequate or should we target 20 Hz?
4. **Simulator**: Stage sufficient or need Gazebo for 3D?
5. **Training Data**: How many GA generations needed for good coverage?

---

## Useful Commands Reference

### Docker
```bash
# List containers
docker ps -a | grep plan_ga

# View logs
docker logs plan_ga_ros1

# Inspect
docker inspect plan_ga_ros1

# Remove all
docker rm -f plan_ga_ros1 plan_ga_ros2
```

### Git
```bash
# Status
git status

# Log
git log --oneline -10

# Show commit
git show HEAD
```

### Python Training (when implemented)
```bash
conda activate plan_ga
python training/train_ga.py --config training/config/ga_config.yaml
python training/train_nn.py --data models/checkpoints/trajectories.pkl --onnx_output models/planner_policy.onnx
```

---

**Note**: This project is at a clean checkpoint. All C++ code is complete and committed. The next major phase is implementing the Python training pipeline to generate the ONNX model that the C++ plugins require at runtime.
