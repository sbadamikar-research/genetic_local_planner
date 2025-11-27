# Docker Container Management

Quick reference for managing the plan_ga ROS Docker containers.

## Container Scripts

### ROS1 Container

```bash
cd docker/ros1

./build.sh   # Build the Docker image (first time or after Dockerfile changes)
./run.sh     # Start container in background (creates if doesn't exist)
./stop.sh    # Stop the running container
./remove.sh  # Remove the container completely
```

### ROS2 Container

```bash
cd docker/ros2

./build.sh   # Build the Docker image (first time or after Dockerfile changes)
./run.sh     # Start container in background (creates if doesn't exist)
./stop.sh    # Stop the running container
./remove.sh  # Remove the container completely
```

## Container Workflow

### Initial Setup

```bash
# 1. Build the image
cd docker/ros1  # or docker/ros2
./build.sh

# 2. Start the container
./run.sh
```

### Daily Development

```bash
# Start container if not running
./run.sh

# Attach via terminal
docker exec -it plan_ga_ros1 bash  # or plan_ga_ros2

# OR attach via VS Code
# Open Command Palette (F1)
# "Dev Containers: Attach to Running Container"
# Select plan_ga_ros1 or plan_ga_ros2
```

### When Done

```bash
# Stop container (preserves build artifacts)
./stop.sh

# OR remove completely (clean slate next time)
./remove.sh
```

## Manual Docker Commands

If you prefer not to use the scripts:

### Start Container
```bash
# ROS1
docker start plan_ga_ros1

# ROS2
docker start plan_ga_ros2
```

### Attach to Container
```bash
# ROS1
docker exec -it plan_ga_ros1 bash

# ROS2
docker exec -it plan_ga_ros2 bash
```

### Stop Container
```bash
# ROS1
docker stop plan_ga_ros1

# ROS2
docker stop plan_ga_ros2
```

### Remove Container
```bash
# ROS1
docker rm -f plan_ga_ros1

# ROS2
docker rm -f plan_ga_ros2
```

### Check Status
```bash
# List all plan_ga containers
docker ps -a | grep plan_ga

# List running plan_ga containers
docker ps | grep plan_ga

# Inspect container
docker inspect plan_ga_ros1  # or plan_ga_ros2
```

## Container Details

### Volume Mounts

Both containers mount the following directories from the host:

```
Host                           → Container
─────────────────────────────────────────────────────────────
src/plan_ga_planner/           → /catkin_ws or /ros2_ws/src/plan_ga/plan_ga_planner
src/plan_ga_ros1 or ros2/      → /catkin_ws or /ros2_ws/src/plan_ga/plan_ga_ros1|ros2
models/                        → /models
samples/                       → /samples
```

### Key Points

- **Source code** is volume mounted, so edits persist to host
- **Build artifacts** stay in container (devel/ or install/)
- **Containers** run in background with `tail -f /dev/null`
- **Network mode** is host for easy ROS communication
- **Container names** are fixed: `plan_ga_ros1` and `plan_ga_ros2`

## Troubleshooting

### Container Already Exists
```bash
# Error: "The container name "/plan_ga_ros1" is already in use"
# Solution: Use ./run.sh which handles this automatically
# OR manually remove: docker rm -f plan_ga_ros1
```

### Container Stopped Unexpectedly
```bash
# Check logs
docker logs plan_ga_ros1

# Restart
./run.sh  # or docker start plan_ga_ros1
```

### Can't Find Container
```bash
# List all containers (including stopped)
docker ps -a

# If truly missing, recreate
./remove.sh  # ensure clean slate
./run.sh     # create new container
```

### Want Fresh Start
```bash
# Remove container and rebuild image
./remove.sh
./build.sh
./run.sh
```

## VS Code Integration

See [Development Plan](../docs/development_plan.md#vs-code-development-with-docker) for detailed VS Code setup instructions.

Quick steps:
1. Install "Dev Containers" extension
2. Run `./run.sh` to start container
3. In VS Code: `F1` → "Attach to Running Container" → Select container
4. Open folder: `/catkin_ws/src/plan_ga` or `/ros2_ws/src/plan_ga`
