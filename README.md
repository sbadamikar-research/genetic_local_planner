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

# Train genetic algorithm
python training/train_ga.py --config training/config/ga_config.yaml --output models/checkpoints/

# Train neural network
python training/train_nn.py --data models/checkpoints/all_trajectories.pkl --output models/distilled_policy.pth --onnx_output models/planner_policy.onnx
```

### 3. ROS1 Development (Docker)

```bash
# Run ROS1 container
cd docker/ros1
./run.sh

# Inside container:
cd /catkin_ws
catkin_make
source devel/setup.bash
roslaunch plan_ga_ros1 test_planner.launch
```

### 4. ROS2 Development (Docker)

```bash
# Run ROS2 container
cd docker/ros2
./run.sh

# Inside container:
cd /ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# Verify plugin registration
ros2 pkg list | grep plan_ga
ros2 plugin list | grep plan_ga

# Launch test (when available)
ros2 launch plan_ga_ros2 test_planner.launch.py
```

## Project Structure

```
plan_ga/
├── docker/                  # Docker containers for ROS1/ROS2
├── docs/                    # Documentation
├── models/                  # Trained models (ONNX)
├── training/                # Python training code
│   ├── ga/                  # Genetic algorithm
│   ├── simulator/           # Stage wrapper
│   ├── neural_network/      # NN distillation
│   └── utils/               # Utilities
├── src/                     # C++ source code
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
