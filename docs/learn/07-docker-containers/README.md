# Module 07: Docker & Dev Containers

**Estimated Time:** 1 day (6-8 hours)

## 🎯 Learning Objectives

- ✅ Understand Docker fundamentals (images, containers, volumes)
- ✅ Build ROS Docker images from Dockerfiles
- ✅ Manage container lifecycle (start, stop, attach)
- ✅ Use volume mounts for live code development
- ✅ Configure VS Code Dev Containers
- ✅ Debug Docker networking and DNS issues
- ✅ Understand the project's Docker strategy

## Why Docker for Robotics?

**The Problem:** "Works on my machine"
- ROS1 Noetic requires Ubuntu 20.04
- ROS2 Humble requires Ubuntu 22.04
- ONNX Runtime, specific library versions, etc.

**The Solution:** Docker containers
- Reproducible environment across machines
- Isolate dependencies from host system
- Easy CI/CD integration
- Switch between ROS1/ROS2 instantly

**For this project:** Our Docker setup lets you develop C++ plugins without polluting your host machine with ROS dependencies.

---

## Key Concepts

### Docker Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     DOCKER ARCHITECTURE                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HOST MACHINE (Your Linux/Mac/Windows)                        │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Docker Daemon (dockerd)                                  │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │  DOCKER IMAGE (read-only template)                 │  │ │
│  │  │  ┌──────────────────────────────────────────────┐  │  │ │
│  │  │  │ Layer 4: ONNX Runtime                        │  │  │ │
│  │  │  ├──────────────────────────────────────────────┤  │  │ │
│  │  │  │ Layer 3: ROS packages (nav-core, move_base) │  │  │ │
│  │  │  ├──────────────────────────────────────────────┤  │  │ │
│  │  │  │ Layer 2: Build tools (cmake, g++)           │  │  │ │
│  │  │  ├──────────────────────────────────────────────┤  │  │ │
│  │  │  │ Layer 1: Base OS (Ubuntu 20.04, ROS Noetic)│  │  │ │
│  │  │  └──────────────────────────────────────────────┘  │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  │                          │                                 │ │
│  │                          ↓ docker run                      │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │  CONTAINER (running instance)                      │  │ │
│  │  │  • Writable layer on top of image                  │  │ │
│  │  │  • Isolated filesystem, network, processes         │  │ │
│  │  │  • Can mount host directories (volumes)            │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  VOLUMES (shared data between host ↔ container)               │
│  Host: /home/user/plan_ga/src/                                │
│    ↕                                                           │
│  Container: /catkin_ws/src/                                    │
│  • Changes in host reflect in container (and vice versa)      │
│  • Persist after container stops                              │
└────────────────────────────────────────────────────────────────┘
```

**Key Terminology:**
- **Image**: Blueprint (like a class in OOP)
- **Container**: Running instance (like an object)
- **Volume**: Shared folder between host and container
- **Dockerfile**: Recipe to build an image
- **Registry**: Image storage (Docker Hub, ghcr.io)

### Dockerfile Anatomy

Our Dockerfile builds a complete ROS1 development environment:

```dockerfile
FROM ros:noetic-ros-core        # Base image (Ubuntu 20.04 + ROS)
                                # ↓
RUN apt-get update              # Install build tools
RUN apt-get install build-essential cmake
                                # ↓
RUN apt-get install ros-noetic-nav-core  # ROS dependencies
                                # ↓
RUN wget onnxruntime-1.16.3.tgz # Install ONNX Runtime
RUN tar -xzf ...
                                # ↓
RUN mkdir -p /catkin_ws/src     # Create workspace
WORKDIR /catkin_ws              # Set default directory
                                # ↓
CMD ["/bin/bash"]               # Default command when container starts
```

**Layer caching:** Docker caches each step. If nothing changed, reuses cached layer (fast rebuilds!).

### Volume Mounting Strategy

**Why not mount entire project?**

```
❌ BAD: Mount entire project root
Host:      /home/user/plan_ga  →  Container: /catkin_ws
Problem:   Overwrites container's /catkin_ws, breaks ROS setup!

✅ GOOD: Mount only source directories
Host:      /home/user/plan_ga/src/plan_ga_planner  →  Container: /catkin_ws/src/plan_ga/plan_ga_planner
Host:      /home/user/plan_ga/src/plan_ga_ros1     →  Container: /catkin_ws/src/plan_ga/plan_ga_ros1
Host:      /home/user/plan_ga/models                →  Container: /models
Result:    Container workspace intact, code is live-editable!
```

**Benefits:**
- Edit code on host (your favorite editor)
- Changes immediately visible in container
- Build artifacts stay in container (no pollution on host)
- Preserve container's `/catkin_ws/devel`, `/catkin_ws/build`

### Network Modes

```
┌──────────────────────────────────────────────────────────┐
│               DOCKER NETWORK MODES                        │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  BRIDGE (default):                                       │
│  ┌──────────────┐          ┌─────────────┐              │
│  │ Container    │ NAT ───> │ Host        │ ───> Internet│
│  │ 172.17.0.2   │          │ 192.168.1.x │              │
│  └──────────────┘          └─────────────┘              │
│  • Isolated network, containers can't see host processes │
│  • Need port forwarding (-p 11311:11311 for ROS Master) │
│  • Good for production, restrictive for development     │
│                                                           │
│  HOST (what we use):                                     │
│  ┌──────────────┐                                        │
│  │ Container    │ === (shares host network stack)       │
│  │ localhost    │                                        │
│  └──────────────┘                                        │
│  • Container uses host's IP and ports directly          │
│  • No port forwarding needed                            │
│  • ROS nodes on host can talk to nodes in container     │
│  • Perfect for ROS development!                         │
└──────────────────────────────────────────────────────────┘
```

**Why host mode for ROS?**
- ROS1 uses ROS Master on port 11311
- Nodes discover each other via hostname/IP
- Host mode = seamless communication

### Dev Containers vs Manual Docker

| Approach | Workflow | Best For |
|----------|----------|----------|
| **Manual Docker** | `./run.sh` → `docker exec -it bash` | Quick testing, CI/CD |
| **Dev Containers** | VS Code auto-attaches, IntelliSense works | Active development |

**Dev Container advantages:**
- IntelliSense with correct include paths
- Integrated terminal (already inside container)
- Extensions auto-install
- Git integration
- Debugging support

---

## Hands-On Exercises

### Exercise 1: Build Docker Images (30 min)

Build the ROS1 and ROS2 images:

```bash
cd /home/ANT.AMAZON.COM/basancht/plan_ga

# Build ROS1 image
cd docker/ros1
./build.sh

# Watch the build process:
# [1/N] FROM ros:noetic-ros-core
# [2/N] RUN apt-get update
# [3/N] RUN apt-get install build-essential
# ...

# Verify image exists
docker images | grep plan_ga_ros1
# plan_ga_ros1   latest   abc123   5 minutes ago   2.1GB

# Build ROS2 image
cd ../ros2
./build.sh

# Check both images
docker images | grep plan_ga
```

**What just happened?**
1. Downloaded base image (`ros:noetic-ros-core` ~500MB)
2. Installed build tools and ROS packages
3. Downloaded and extracted ONNX Runtime
4. Created workspace directory
5. Saved as `plan_ga_ros1:latest` (~2GB)

**Questions:**
1. Why is the image so large? (OS + ROS + libraries)
2. What happens if you run `./build.sh` again? (Uses cached layers, fast!)
3. Where are images stored? (`/var/lib/docker/`)

### Exercise 2: Run and Attach to Containers (30 min)

Start containers in background:

```bash
cd /home/ANT.AMAZON.COM/basancht/plan_ga/docker/ros1
./run.sh

# Output:
# Creating and starting new container 'plan_ga_ros1'...
# Container started in background.
# Attach with: docker exec -it plan_ga_ros1 bash

# Check container is running
docker ps
# CONTAINER ID   IMAGE              STATUS    NAMES
# 7f8a9b2c1d3e   plan_ga_ros1:latest  Up 5s   plan_ga_ros1

# Attach to container
docker exec -it plan_ga_ros1 bash

# You're now INSIDE the container!
root@hostname:/catkin_ws#

# Check ROS installation
source /opt/ros/noetic/setup.bash
roscore &
# ROS Master starts on localhost:11311

# Check mounted volumes
ls src/plan_ga/
# plan_ga_planner  plan_ga_ros1

# Exit container (keeps running in background)
exit

# Stop container
cd docker/ros1
./stop.sh

# Remove container completely
./remove.sh
```

**Container lifecycle commands:**
```bash
# List all containers (running + stopped)
docker ps -a

# Start stopped container
docker start plan_ga_ros1

# Stop running container
docker stop plan_ga_ros1

# Remove container (must stop first)
docker rm plan_ga_ros1

# View container logs
docker logs plan_ga_ros1

# Inspect container details
docker inspect plan_ga_ros1
```

**Questions:**
1. What's the difference between `docker run` and `docker exec`?
   - `run`: Creates new container
   - `exec`: Runs command in existing container
2. Where do files created in container live?
   - In container's filesystem (lost when removed unless in volume)
3. Can you run multiple containers from same image?
   - Yes! Each container is independent

### Exercise 3: Test Volume Mounts and Live Updates (45 min)

Verify live code editing works:

```bash
# Start container
cd /home/ANT.AMAZON.COM/basancht/plan_ga/docker/ros1
./run.sh

# Attach
docker exec -it plan_ga_ros1 bash

# Inside container: Check mounted code
root@:/catkin_ws# cat src/plan_ga/plan_ga_planner/include/plan_ga_planner/types.h | head -n 5
# Should show first 5 lines of types.h

# On host (different terminal): Edit file
cd /home/ANT.AMAZON.COM/basancht/plan_ga
echo "// Test comment" >> src/plan_ga_planner/include/plan_ga_planner/types.h

# Back in container: See change immediately
root@:/catkin_ws# tail -n 1 src/plan_ga/plan_ga_planner/include/plan_ga_planner/types.h
# // Test comment   <--- Appears instantly!

# Remove test comment (host)
sed -i '$ d' src/plan_ga_planner/include/plan_ga_planner/types.h
```

**Verify build artifacts stay in container:**

```bash
# Inside container
cd /catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make

# Check build artifacts
ls build/ devel/
# build/: CMake files, object files
# devel/: Compiled libraries

# On host: Build artifacts NOT present
ls /home/ANT.AMAZON.COM/basancht/plan_ga/
# No build/ or devel/ directory (good!)

# Inside container: Remove build
rm -rf build/ devel/

# Rebuild (fast, no network needed)
catkin_make
```

**Questions:**
1. Why keep build artifacts in container? (Avoid polluting host with binaries)
2. What happens if container is removed? (Build artifacts lost, code safe in volume)
3. Can multiple containers share same volume? (Yes, but be careful with concurrent writes!)

### Exercise 4: Use VS Code Dev Containers (1 hour)

Seamless development inside containers:

**Step 1: Install Dev Containers extension**

```bash
# In VS Code
# 1. Open Extensions (Ctrl+Shift+X)
# 2. Search "Dev Containers"
# 3. Install "Dev Containers" by Microsoft
```

**Step 2: Open project in Dev Container**

```bash
# Method 1: From scratch
# 1. F1 → "Dev Containers: Open Folder in Container"
# 2. Select /home/ANT.AMAZON.COM/basancht/plan_ga
# 3. Choose: ".devcontainer/ros1/devcontainer.json"
# → VS Code rebuilds window inside container!

# Method 2: Attach to running container
# 1. Start container: ./docker/ros1/run.sh
# 2. F1 → "Dev Containers: Attach to Running Container"
# 3. Select "plan_ga_ros1"
# → VS Code attaches to existing container
```

**Step 3: Verify environment**

```bash
# Open integrated terminal (Ctrl+`)
# Already inside container at /catkin_ws!

# Check ROS
echo $ROS_DISTRO
# noetic

# Check extensions installed
# C/C++ IntelliSense
# CMake Tools
# Python
# Git Graph

# Test IntelliSense
# Open src/plan_ga_planner/include/plan_ga_planner/types.h
# Hover over "std::vector" → Shows documentation!
# Ctrl+Click on "Pose" → Jumps to definition!
```

**Step 4: Build from VS Code**

```bash
# In terminal:
source /opt/ros/noetic/setup.bash
catkin_make

# Or use CMake Tools extension:
# Ctrl+Shift+P → "CMake: Configure"
# → Detects catkin project
```

**Questions:**
1. What's stored in `.devcontainer/devcontainer.json`? (Config: image, mounts, extensions)
2. Can you use debugger in Dev Container? (Yes! C++ and Python debugging work)
3. What if you need root access? (`"remoteUser": "root"` in config)

### Exercise 5: Build Project Inside Container (1 hour)

Complete build workflow:

```bash
# Attach to container (VS Code or docker exec)
docker exec -it plan_ga_ros1 bash

# Verify environment
cd /catkin_ws
ls src/plan_ga/
# plan_ga_planner  plan_ga_ros1

# Source ROS
source /opt/ros/noetic/setup.bash

# Clean build
rm -rf build/ devel/
catkin_make

# Expected output:
# ####
# #### Running command: "cmake /catkin_ws/src ..."
# ####
# -- Found catkin packages: plan_ga_planner, plan_ga_ros1
# -- Configuring done
# -- Generating done
# -- Build files written to: /catkin_ws/build
# ####
# #### Running command: "make -j8 ..."
# ####
# [ 25%] Building CXX object plan_ga_planner/...
# [ 50%] Linking CXX shared library libplan_ga_planner.so
# [ 75%] Building CXX object plan_ga_ros1/...
# [100%] Linking CXX shared library libplan_ga_ros1_plugin.so

# Source workspace
source devel/setup.bash

# Verify plugin loads
rospack plugins --attrib=plugin nav_core | grep plan_ga
# plan_ga_ros1 /catkin_ws/src/plan_ga/plan_ga_ros1/plan_ga_plugin.xml

# Check library exists
ls devel/lib/libplan_ga_ros1_plugin.so
# Success!
```

**Common build errors and fixes:**

```bash
# Error: "Could not find ONNXRUNTIME"
# Fix: Check environment variable
echo $ONNXRUNTIME_ROOT
# Should be: /opt/onnxruntime-linux-x64-1.16.3

# Error: "ros/ros.h: No such file or directory"
# Fix: Source ROS
source /opt/ros/noetic/setup.bash

# Error: "Could not find package 'nav_core'"
# Fix: Install ROS dependencies
apt-get update && apt-get install ros-noetic-nav-core
```

**Questions:**
1. Why source `/opt/ros/noetic/setup.bash` before building? (Sets ROS environment variables)
2. What's in `devel/setup.bash`? (Environment for your custom packages)
3. Can you build on host without Docker? (Yes, but need exact dependencies)

### Exercise 6: Debug DNS and Network Issues (1 hour)

Corporate networks can block Docker. Here's how to fix:

**Issue 1: DNS resolution fails during build**

```bash
# Symptom:
./build.sh
# Error: Could not resolve 'archive.ubuntu.com'

# Diagnosis:
docker run -it ubuntu:20.04 bash
root@container# ping google.com
# ping: google.com: Temporary failure in name resolution

# Fix 1: Use host network for build
cd docker/ros1
# Edit Dockerfile: no changes needed
# Edit build.sh:
docker build --network host -t plan_ga_ros1:latest .

# Fix 2: Configure Docker DNS
sudo nano /etc/docker/daemon.json
{
  "dns": ["10.4.4.10", "8.8.8.8", "1.1.1.1"]
}
sudo systemctl restart docker

# Rebuild
./build.sh
```

**Issue 2: ROS nodes can't communicate**

```bash
# Symptom:
# Node on host can't see node in container

# Diagnosis:
# In container:
roscore &
rostopic list
# /rosout

# On host (different terminal):
export ROS_MASTER_URI=http://localhost:11311
rostopic list
# Error: Cannot contact ROS Master

# Fix: Use host network mode (already set in run.sh!)
docker run --network host ...
# Now container shares host's network
```

**Issue 3: ONNX Runtime download fails**

```bash
# Symptom:
Downloading onnxruntime-1.16.3.tgz... Failed

# Fix 1: Download manually and copy
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
# Copy into Docker build context
cp onnxruntime-linux-x64-1.16.3.tgz /home/ANT.AMAZON.COM/basancht/plan_ga/docker/ros1/

# Edit Dockerfile:
# COPY onnxruntime-linux-x64-1.16.3.tgz /tmp/
# RUN tar -xzf /tmp/onnxruntime-linux-x64-1.16.3.tgz -C /opt/

# Fix 2: Use mirror
# Edit Dockerfile, change URL to mirror site
```

**Issue 4: Permission denied on mounted volumes**

```bash
# Symptom:
# Can't write to /catkin_ws/src/plan_ga/

# Cause: User ID mismatch
# Host user: UID 1000
# Container user: UID 0 (root)

# Fix: Run container as host user
docker run --user $(id -u):$(id -g) ...
# Or use rootless Docker
```

---

## Code Walkthrough

### Dockerfile Analysis

**File:** `docker/ros1/Dockerfile`

```dockerfile
# Line 1: Base image
FROM ros:noetic-ros-core
# Ubuntu 20.04 + minimal ROS Noetic
# Size: ~500 MB

# Lines 4-9: Build essentials
RUN apt-get update && apt-get install -y \
    build-essential \  # gcc, g++, make
    cmake \            # CMake 3.16+
    git \              # Version control
    wget \             # Download files
    && rm -rf /var/lib/apt/lists/*  # Clean cache to reduce image size

# Lines 12-20: ROS dependencies
RUN apt-get update && apt-get install -y \
    ros-noetic-nav-core \          # Base local planner interface
    ros-noetic-costmap-2d \        # Costmap representation
    ros-noetic-tf2-ros \           # Transform library
    ros-noetic-tf2-geometry-msgs \ # TF message conversions
    ros-noetic-pluginlib \         # Plugin loading
    ros-noetic-stage-ros \         # Stage simulator (optional)
    ros-noetic-move-base \         # Navigation stack
    && rm -rf /var/lib/apt/lists/*

# Lines 23-25: Install ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz && \
    tar -xzf onnxruntime-linux-x64-1.16.3.tgz -C /opt/ && \
    rm onnxruntime-linux-x64-1.16.3.tgz  # Clean up tarball

# Line 27: Set environment variable
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.16.3
# CMake will use this to find ONNX Runtime

# Lines 30-31: Create workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws  # Default directory when entering container

# Line 34: Auto-source ROS
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Line 36: Default command
CMD ["/bin/bash"]  # Start bash shell
```

**Layer caching:** If you rebuild and only change line 30, Docker reuses cached layers 1-27!

### Container Startup Script

**File:** `docker/ros1/run.sh`

```bash
#!/bin/bash
cd "$(dirname "$0")/../.."  # Go to project root

# Check if container exists
if docker ps -a --format '{{.Names}}' | grep -q '^plan_ga_ros1$'; then
    # Container exists, is it running?
    if docker ps --format '{{.Names}}' | grep -q '^plan_ga_ros1$'; then
        echo "Container is running. Attach with: docker exec -it plan_ga_ros1 bash"
    else
        # Container stopped, restart it
        docker start plan_ga_ros1
    fi
else
    # Create new container
    docker run -d \                          # Detached (background) mode
        --name plan_ga_ros1 \                # Container name
        -v "$(pwd)"/src/plan_ga_planner:/catkin_ws/src/plan_ga/plan_ga_planner \  # Volume mounts
        -v "$(pwd)"/src/plan_ga_ros1:/catkin_ws/src/plan_ga/plan_ga_ros1 \
        -v "$(pwd)"/models:/models \
        -v "$(pwd)"/samples:/samples \
        --network host \                     # Use host network
        plan_ga_ros1:latest \                # Image to use
        tail -f /dev/null                    # Keep container alive
    # Note: `tail -f /dev/null` is a trick to keep container running indefinitely
fi
```

**Why `-d` (detached) mode?**
- Container runs in background
- Survives terminal close
- Can attach/detach freely
- VS Code can attach to it

**Alternative (not used):**
```bash
docker run -it --rm ...  # Interactive, auto-remove when exit
# Problem: Container dies when you close terminal!
```

### Dev Container Configuration

**File:** `.devcontainer/ros1/devcontainer.json`

```json
{
  "name": "ROS1 Noetic - Plan GA Development",
  "image": "plan_ga_ros1:latest",           // Use pre-built image
  "workspaceFolder": "/catkin_ws",          // Open here

  "mounts": [                               // Volume mounts
    "source=${localWorkspaceFolder}/src/plan_ga_planner,target=/catkin_ws/src/plan_ga/plan_ga_planner,type=bind",
    // ... more mounts ...
  ],

  "runArgs": ["--network=host"],            // Docker run args

  "customizations": {
    "vscode": {
      "extensions": [                       // Auto-install extensions
        "ms-vscode.cpptools",               // C++ IntelliSense
        "ms-vscode.cmake-tools",            // CMake support
        "ms-python.python",                 // Python support
        "mhutchie.git-graph"                // Git visualization
      ],
      "settings": {                         // IDE settings
        "C_Cpp.default.includePath": [      // IntelliSense search paths
          "${workspaceFolder}/**",
          "/opt/ros/noetic/include/**",
          "/opt/onnxruntime-linux-x64-1.16.3/include/**"
        ],
        "C_Cpp.default.cppStandard": "c++17"
      }
    }
  },

  "postCreateCommand": "source /opt/ros/noetic/setup.bash",  // Run after container created
  "postAttachCommand": "source /opt/ros/noetic/setup.bash && cd /catkin_ws"  // Run when attaching
}
```

**Key benefit:** IntelliSense works! Knows where ROS headers are.

---

## Quiz

1. **What's the difference between an image and a container?**
   a) No difference
   b) Image is blueprint, container is running instance
   c) Image is smaller than container
   d) Container is stored on Docker Hub

2. **Why use volume mounts instead of COPY in Dockerfile?**
   a) Volumes are faster
   b) Volumes allow live code editing without rebuilding image
   c) COPY is deprecated
   d) Volumes use less disk space

3. **What does `--network host` do?**
   a) Disables networking
   b) Container shares host's network stack (same IP, ports)
   c) Creates bridge network
   d) Enables IPv6

4. **Why keep build artifacts (build/, devel/) in container?**
   a) Better performance
   b) Avoid polluting host with platform-specific binaries
   c) Required by Docker
   d) Smaller image size

5. **What happens when you remove a container?**
   a) Image is deleted
   b) Volumes are deleted
   c) Container's filesystem is deleted, volumes persist
   d) All Docker data is lost

<details>
<summary><b>Show Answers</b></summary>

1. b) Image is blueprint (like a class), container is running instance (like an object)
2. b) Volumes allow live code editing without rebuilding (edit on host, reflects in container)
3. b) Container shares host's network stack (perfect for ROS communication)
4. b) Avoid polluting host with platform-specific binaries (Linux binaries won't run on Mac)
5. c) Container's filesystem deleted, volumes persist (your code is safe!)
</details>

---

## ✅ Checklist

- [ ] Understand Docker images vs containers vs volumes
- [ ] Built ROS1 and ROS2 Docker images successfully
- [ ] Started/stopped containers with run.sh
- [ ] Verified live code editing with volume mounts
- [ ] Used VS Code Dev Containers for development
- [ ] Built project inside container successfully
- [ ] Debugged at least one networking or DNS issue
- [ ] Quiz score 80%+

---

## 📚 Resources

- [Docker Official Documentation](https://docs.docker.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [ROS Docker Images](https://hub.docker.com/_/ros/) (official)
- [Docker Networking](https://docs.docker.com/network/)
- [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)

---

## 🎉 Next Steps

You now understand Docker and can develop ROS projects in isolated, reproducible environments!

Next, learn how to execute the complete training pipeline to generate the ONNX model.

**→ [Continue to Module 08: Training Pipeline & Experiments](../08-training-pipeline/)**
