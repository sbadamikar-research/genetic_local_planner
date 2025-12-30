# Setup

## Prerequisites

### Required Software

- **Git**: Version control
- **Docker**: Container runtime (version 20.10+)
- **Miniconda3**: Python environment management

### Verify Installation

```bash
git --version
docker --version
conda --version

# Test Docker permissions (should work without sudo)
docker run hello-world
```

## Environment Setup

### 1. Clone Repository

```bash
cd /path/to/your/workspace
git clone <repository-url>
cd plan_ga
```

### 2. Create Python Environment

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate plan_ga

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import onnx; print(f'ONNX: {onnx.__version__}')"
```

### 3. Docker Network Configuration (Corporate Networks)

If on a corporate network with custom DNS:

```bash
# Edit Docker daemon config
sudo nano /etc/docker/daemon.json
```

Add DNS servers:

```json
{
  "dns": ["10.4.4.10", "8.8.8.8", "1.1.1.1"]
}
```

Restart Docker:

```bash
sudo systemctl restart docker
```

## Directory Structure

```
plan_ga/
├── docker/              # Container definitions
│   ├── ros1/           # ROS1 Noetic
│   └── ros2/           # ROS2 Humble
├── models/             # Trained models
│   └── checkpoints/    # Training checkpoints
├── src/                # C++ source code
│   ├── plan_ga_planner/    # Core library
│   ├── plan_ga_ros1/       # ROS1 plugin
│   └── plan_ga_ros2/       # ROS2 plugin
├── training/           # Python training code
│   ├── ga/            # Genetic algorithm
│   ├── simulator/     # Python simulator
│   ├── neural_network/    # NN training
│   └── config/        # Training configs
└── samples/           # Example configs
    └── configs/       # ROS parameter files
```

## Verification

### Test Python Environment

```bash
conda activate plan_ga
python training/simulator/costmap.py  # Should have test code at bottom
```

### Test Docker

```bash
cd docker/ros1
./build.sh  # Should complete without errors
```

## Next Steps

- **Build ROS plugins**: See [building.md](building.md)
- **Run training**: See [training.md](training.md)
- **Docker workflow**: See [docker.md](docker.md)
