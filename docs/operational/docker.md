# Docker

## Container Lifecycle

### Initial Setup

```bash
cd docker/ros1  # or docker/ros2

# Build image (first time only)
./build.sh

# Start container
./run.sh
```

### Daily Workflow

```bash
# Check if container is running
docker ps | grep plan_ga

# Start container (if stopped)
./run.sh

# Attach via terminal
docker exec -it plan_ga_ros1 bash  # or plan_ga_ros2

# Stop when done
./stop.sh
```

### Clean Slate

```bash
# Remove container completely
./remove.sh

# Rebuild from scratch
./build.sh
./run.sh
```

## Container Scripts

### ROS1 Container

| Script | Description |
|--------|-------------|
| `./build.sh` | Build Docker image from Dockerfile |
| `./run.sh` | Start container in background (creates if needed) |
| `./stop.sh` | Stop running container (preserves filesystem) |
| `./remove.sh` | Delete container completely |

Container name: `plan_ga_ros1`
Workspace: `/catkin_ws`

### ROS2 Container

| Script | Description |
|--------|-------------|
| `./build.sh` | Build Docker image from Dockerfile |
| `./run.sh` | Start container in background (creates if needed) |
| `./stop.sh` | Stop running container (preserves filesystem) |
| `./remove.sh` | Delete container completely |

Container name: `plan_ga_ros2`
Workspace: `/ros2_ws`

## Volume Mounts

Both containers mount the following directories:

```
Host                              Container
────────────────────────────────  ─────────────────────────────────────
./src/plan_ga_planner/       →    /{workspace}/src/plan_ga/plan_ga_planner
./src/plan_ga_ros{1|2}/      →    /{workspace}/src/plan_ga/plan_ga_ros{1|2}
./models/                    →    /models
./samples/                   →    /samples
```

**Note**: Changes in mounted directories persist on the host.

## VS Code Integration

### Attach to Running Container

1. Ensure container is running: `./run.sh`
2. Open VS Code
3. Press `F1`
4. Select: **"Dev Containers: Attach to Running Container"**
5. Choose: `plan_ga_ros1` or `plan_ga_ros2`

### Dev Container Configuration

Pre-configured dev containers available in `.devcontainer/`:
- `ros1/devcontainer.json` - ROS1 Noetic
- `ros2/devcontainer.json` - ROS2 Humble

**To use:**
1. Copy desired config to `.devcontainer/devcontainer.json`
2. F1 → **"Dev Containers: Reopen in Container"**

## Manual Docker Commands

### Inspect Container

```bash
# View logs
docker logs plan_ga_ros1

# Container details
docker inspect plan_ga_ros1

# Resource usage
docker stats plan_ga_ros1
```

### Execute Commands

```bash
# Run single command
docker exec plan_ga_ros1 bash -c "source /opt/ros/noetic/setup.bash && roscore"

# Interactive shell
docker exec -it plan_ga_ros1 bash
```

### Network

Both containers use `--network host` for ROS communication.

```bash
# Check network mode
docker inspect plan_ga_ros1 | grep NetworkMode
```

## Multiple Containers

You can run both ROS1 and ROS2 containers simultaneously:

```bash
# Terminal 1
cd docker/ros1
./run.sh
docker exec -it plan_ga_ros1 bash

# Terminal 2
cd docker/ros2
./run.sh
docker exec -it plan_ga_ros2 bash
```

## Troubleshooting

### Container Won't Start

```bash
# Check if port conflict
docker ps -a

# Remove existing container
./remove.sh

# Rebuild
./build.sh
./run.sh
```

### Build Fails

```bash
# Check DNS configuration
cat /etc/docker/daemon.json

# Try build with host network
docker build --network host -t plan_ga_ros1 .
```

### Out of Disk Space

```bash
# Clean up dangling images
docker system prune

# Remove unused volumes
docker volume prune

# Check disk usage
docker system df
```

### Permission Issues

```bash
# Ensure user is in docker group
sudo usermod -aG docker $USER
newgrp docker
```

## Image Management

```bash
# List images
docker images | grep plan_ga

# Remove image
docker rmi plan_ga_ros1:latest

# Rebuild image
cd docker/ros1
./build.sh
```

## Next Steps

- **Build plugins**: See [building.md](building.md)
- **Configure VS Code**: See `.devcontainer/README.md`
- **Deploy planner**: See [deployment.md](deployment.md)
