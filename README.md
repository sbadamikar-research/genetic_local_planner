# GA-Based ROS Local Planner

A genetic algorithm-based local planner for ROS that learns navigation policies from costmap data, distills them into neural networks, and deploys as real-time C++ plugins for both ROS1 Noetic and ROS2 Humble.

## Project Overview

- **Training**: Python GA with lightweight Stage simulator wrapper (host machine with Miniconda)
- **Model Transfer**: Neural network policy distillation → ONNX export
- **Deployment**: C++ ONNX Runtime inference in ROS1/ROS2 plugins (Docker containers)
- **Target Frequency**: 10-20 Hz planning rate

## Quick Start

### 1. Environment Setup

#### Host Machine (Python Training)

```bash
# Create conda environment
conda env create -f environment.yml
conda activate plan_ga
```

#### Docker Containers (ROS Development)

```bash
# Build ROS1 container
cd docker/ros1
./build.sh

# Build ROS2 container
cd docker/ros2
./build.sh
```

### 2. Build C++ Planner

#### ROS1 Build

```bash
# Run ROS1 container
cd docker/ros1
./run.sh

# Inside container:
cd /catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash

# Verify plugin is registered
rospack plugins --attrib=plugin nav_core | grep plan_ga
```

#### ROS2 Build

```bash
# Run ROS2 container
cd docker/ros2
./run.sh

# Inside container:
cd /ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# Verify plugin
ros2 pkg list | grep plan_ga
```

### 3. Training Workflow (Host Machine)

```bash
# Activate environment
conda activate plan_ga

# Step 1: Train genetic algorithm (TODO - not yet implemented)
python training/train_ga.py \
  --config training/config/ga_config.yaml \
  --output models/checkpoints/ga_trajectories.pkl

# Step 2: Train neural network (distill GA behavior)
python training/train_nn.py \
  --data models/checkpoints/ga_trajectories.pkl \
  --config training/config/nn_config.yaml \
  --output models/planner_policy.onnx \
  --checkpoint models/checkpoints/best_model.pth

# The trained ONNX model (models/planner_policy.onnx) is now ready for C++ deployment
```

### 4. ROS1 Development (Docker)

```bash
# Start ROS1 container (runs in background)
cd docker/ros1
./run.sh

# Attach to container
docker exec -it plan_ga_ros1 bash

# Inside container:
cd /catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash

# Verify plugin registration
rospack plugins --attrib=plugin nav_core | grep plan_ga

# Stop container when done
cd docker/ros1
./stop.sh
```

### 5. ROS2 Development (Docker)

```bash
# Start ROS2 container (runs in background)
cd docker/ros2
./run.sh

# Attach to container
docker exec -it plan_ga_ros2 bash

# Inside container:
cd /ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# Verify plugin registration
ros2 pkg list | grep plan_ga
ros2 plugin list | grep plan_ga

# Stop container when done
cd docker/ros2
./stop.sh
```

## Project Structure

```
plan_ga/
├── docker/                  # Docker containers for ROS1/ROS2
├── docs/                    # Documentation
├── models/                  # Trained models (ONNX)
│   └── checkpoints/         # Training checkpoints
├── training/                # Python training code
│   ├── ga/                  # Genetic algorithm (TODO)
│   ├── simulator/           # Stage wrapper (TODO)
│   ├── neural_network/      # NN distillation (✓ COMPLETE)
│   │   ├── model.py         # PyTorch model architecture
│   │   └── dataset.py       # GA trajectory dataset loader
│   ├── train_nn.py          # NN training script (✓ COMPLETE)
│   ├── FUTURE_WORK.md       # Advanced features roadmap
│   ├── config/              # Training configurations
│   │   ├── ga_config.yaml   # GA hyperparameters
│   │   └── nn_config.yaml   # NN architecture and training params
│   └── utils/               # Utilities
├── src/                     # C++ source code (✓ COMPLETE)
│   ├── plan_ga_planner/     # Core planner (ROS-agnostic)
│   ├── plan_ga_ros1/        # ROS1 plugin
│   └── plan_ga_ros2/        # ROS2 plugin
├── tests/                   # Tests
└── samples/                 # Example configs/launch files
```

## Key Features

- **Multi-objective GA**: Optimizes for goal distance, collision avoidance, smoothness, and time efficiency
- **Neural Network Distillation**: CNN + MLP architecture for fast inference
- **ONNX Deployment**: Cross-platform model deployment with ONNX Runtime
- **Dual ROS Support**: Compatible with both ROS1 Noetic and ROS2 Humble
- **Docker-based Development**: Clean separation of training and deployment environments

## Neural Network Implementation

### Architecture

The neural network consists of three main components:

1. **CostmapEncoder** (CNN): Processes 50×50 costmap images
   - Conv layers: [1→32→64→128] channels
   - Kernel sizes: [5, 3, 3], strides: [2, 2, 2]
   - Output: 256-dimensional feature vector

2. **StateEncoder** (MLP): Processes robot state + goal + metadata
   - Input: 14 dimensions (9 robot state + 3 goal + 2 metadata)
   - Hidden layers: [14 → 128 → 256]
   - Output: 256-dimensional feature vector

3. **PolicyHead** (MLP): Generates control sequences
   - Input: 512 dimensions (concatenated costmap + state features)
   - Hidden layers: [512 → 256 → 256 → 60]
   - Output: 60 dimensions (20 steps × 3 controls: v_x, v_y, omega)

**Total Parameters**: ~1.95 million

### ONNX Interface

The model expects **4 separate input tensors** to match the C++ plugin interface:

- `costmap`: [1, 1, 50, 50] - normalized costmap window
- `robot_state`: [1, 9] - [x, y, theta, v_x, v_y, omega, a_x, a_y, alpha]
- `goal_relative`: [1, 3] - [dx, dy, dtheta] in robot frame
- `costmap_metadata`: [1, 2] - [inflation_decay, resolution]

**Output:**
- `control_sequence`: [1, 60] - flattened control sequence

### Training

The training script (`training/train_nn.py`) includes:
- MSE loss with early stopping (patience=10)
- Adam optimizer with ReduceLROnPlateau scheduler
- Train/validation split (80/20)
- Automatic ONNX export with verification
- Console logging of training progress

**Training Command:**
```bash
python training/train_nn.py \
  --data models/checkpoints/ga_trajectories.pkl \
  --config training/config/nn_config.yaml \
  --output models/planner_policy.onnx \
  --checkpoint models/checkpoints/best_model.pth
```

### Testing the Model

Test the architecture without training data:
```bash
# Test model architecture
conda activate plan_ga
python training/neural_network/model.py

# Test dataset loader with synthetic data
python training/neural_network/dataset.py
```

### Advanced Features

See `training/FUTURE_WORK.md` for planned enhancements:
- Data augmentation (rotation, noise)
- Per-DOF metrics tracking
- TensorBoard logging
- Advanced loss functions (Huber, weighted MSE, temporal consistency)
- Hyperparameter tuning
- Ensemble methods

## Coding Standards

- **Functions**: camelCase (e.g., `computeVelocity`)
- **Variables**: snake_case (e.g., `current_pose`)
- **Classes/Structs**: ProperCase (e.g., `PlannerCore`)
- **Member variables**: trailing_underscores_ (e.g., `config_`)

## Documentation

- [Development Plan](docs/development_plan.md) - Setup, build, and deployment guide
- [Training Plan](docs/training_plan.md) - GA/NN training protocol and simulation details

## License

[Your License Here]

## Contributors

[Your Name/Team]
