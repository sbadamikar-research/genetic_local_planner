# Operational Documentation

Quick reference for deploying and running the GA-based ROS local planner.

## Documentation Structure

- **[Setup](setup.md)** - Environment setup and prerequisites
- **[Building](building.md)** - Build instructions for ROS1 and ROS2
- **[Docker](docker.md)** - Container management and development workflow
- **[Training](training.md)** - Execute the training pipeline (GA → NN → ONNX)
- **[Deployment](deployment.md)** - Integrate with move_base/Nav2
- **[Configuration](configuration.md)** - Parameter tuning and customization
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

## Quick Start

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate plan_ga

# 2. Train model
python training/train_ga.py --config training/config/ga_config.yaml --output models/checkpoints/ga_trajectories.pkl
python training/train_nn.py --data models/checkpoints/ga_trajectories.pkl --config training/config/nn_config.yaml --output models/planner_policy.onnx

# 3. Build ROS plugin
cd docker/ros1  # or ros2
./build.sh
./run.sh
# Inside container:
catkin_make  # or: colcon build

# 4. Deploy and test
# See deployment.md for integration instructions
```

## Architecture Overview

```
Training (Python/Host)          Deployment (C++/Docker)
┌──────────────────────┐       ┌──────────────────────┐
│ GA Evolution         │       │ ONNX Runtime         │
│ ↓                    │  →    │ ↓                    │
│ NN Distillation      │  →    │ ROS1/ROS2 Plugin     │
│ ↓                    │       │ ↓                    │
│ ONNX Export          │       │ move_base/Nav2       │
└──────────────────────┘       └──────────────────────┘
```

## System Requirements

- **OS**: Linux (tested on Ubuntu 20.04/22.04)
- **CPU**: Multi-core recommended (training uses 8+ workers)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 20GB free space
- **Docker**: Version 20.10 or later
- **Python**: 3.10 (via Miniconda)

## Support

- **Issues**: Check [troubleshooting.md](troubleshooting.md)
- **Learning**: See [../learn/](../learn/) for comprehensive course
- **Code**: Review `src/` for implementation details
