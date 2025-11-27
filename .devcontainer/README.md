# Dev Container Configurations

This directory contains VS Code Dev Container configurations for ROS1 and ROS2 development.

## Available Configurations

```
.devcontainer/
├── devcontainer.json       # Default (ROS1 Noetic)
├── ros1/
│   └── devcontainer.json   # ROS1 Noetic configuration
└── ros2/
    └── devcontainer.json   # ROS2 Humble configuration
```

## Prerequisites

1. **VS Code** with **Dev Containers** extension installed
   - Extension ID: `ms-vscode-remote.remote-containers`

2. **Docker images built**:
   ```bash
   # Build ROS1 image
   cd docker/ros1
   ./build.sh

   # Build ROS2 image
   cd docker/ros2
   ./build.sh
   ```

## Usage

### Method 1: Open Default Configuration (ROS1)

1. Open project root in VS Code
2. Press `F1` or `Ctrl+Shift+P`
3. Select: **"Dev Containers: Reopen in Container"**
4. VS Code will rebuild and open in ROS1 container

### Method 2: Select Specific Configuration

1. Open project root in VS Code
2. Press `F1` or `Ctrl+Shift+P`
3. Select: **"Dev Containers: Open Folder in Container..."**
4. Navigate to `.devcontainer/ros1` or `.devcontainer/ros2`
5. Select the folder

### Method 3: Use Configuration Picker

1. Open project root in VS Code
2. Press `F1` or `Ctrl+Shift+P`
3. Select: **"Dev Containers: Rebuild and Reopen in Container"**
4. If multiple configurations exist, you'll be prompted to choose

## What Gets Installed

Both configurations automatically install:

### VS Code Extensions
- **C/C++ Tools** - IntelliSense and debugging
- **CMake Tools** - CMake integration
- **Python** - Python language support
- **ROS** - ROS-specific features
- **GitLens** - Git integration
- **Code Spell Checker** - Spell checking
- **IntelliCode** - AI-assisted coding

### Configured Settings
- C++ standard: C++17
- Include paths for ROS, ONNX Runtime, and workspace
- File associations (*.h → cpp, *.launch → xml)
- ROS distro environment variables
- Proper IntelliSense configuration

## Workspace Structure

### ROS1 Container
- **Workspace**: `/catkin_ws/src/plan_ga`
- **Build system**: catkin_make
- **Source setup**: `source /opt/ros/noetic/setup.bash`

### ROS2 Container
- **Workspace**: `/ros2_ws/src/plan_ga`
- **Build system**: colcon
- **Source setup**: `source /opt/ros/humble/setup.bash`

## Volume Mounts

Both configurations mount:
```
Host                    → Container
────────────────────────────────────────────
src/plan_ga_planner/    → /{workspace}/src/plan_ga/plan_ga_planner
src/plan_ga_ros{1|2}/   → /{workspace}/src/plan_ga/plan_ga_ros{1|2}
models/                 → /models
samples/                → /samples
```

Changes in VS Code are immediately reflected on the host.

## Building in Dev Container

### ROS1
```bash
# In VS Code integrated terminal
cd /catkin_ws
source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash
```

### ROS2
```bash
# In VS Code integrated terminal
cd /ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## IntelliSense

IntelliSense is pre-configured with correct include paths:
- ✅ ROS headers (`/opt/ros/{distro}/include`)
- ✅ ONNX Runtime headers (`/opt/onnxruntime-linux-x64-1.16.3/include`)
- ✅ Workspace headers (`/catkin_ws` or `/ros2_ws`)
- ✅ Project headers (`${workspaceFolder}`)

No additional configuration needed!

## Lifecycle Hooks

### postCreateCommand
Runs once when container is created:
- Sources ROS setup.bash
- Adds source command to ~/.bashrc

### postStartCommand
Runs each time container starts:
- Changes to workspace directory
- Sources ROS setup.bash

### postAttachCommand
Runs each time you attach to container:
- Sources ROS setup.bash
- Changes to workspace directory

## Switching Between ROS1 and ROS2

### Option 1: Close and Reopen
1. Close VS Code window
2. Open project root again
3. Use "Open Folder in Container"
4. Select different configuration

### Option 2: Use Command Palette
1. `F1` → "Dev Containers: Open Attached Container Configuration..."
2. Select different configuration
3. Rebuild container

### Option 3: Multiple Windows
- Open two VS Code instances
- One with ROS1 configuration
- One with ROS2 configuration
- Develop for both simultaneously!

## Network Configuration

Both containers use `--network=host` for:
- Easy ROS communication (no port mapping needed)
- Access to ROS Master on host
- Seamless multi-container ROS networks

## Troubleshooting

### Container Fails to Start

**Issue**: Docker image not found
```
Solution: Build the image first
cd docker/ros1  # or ros2
./build.sh
```

**Issue**: Port/name already in use
```
Solution: Stop existing containers
docker stop plan_ga_ros1_devcontainer
docker rm plan_ga_ros1_devcontainer
```

### IntelliSense Not Working

**Issue**: Red squiggles under includes
```
Solution: Reload VS Code window
F1 → "Developer: Reload Window"
```

**Issue**: Wrong include paths
```
Solution: Check C_Cpp.default.includePath in settings
F1 → "C/C++: Edit Configurations (JSON)"
```

### Extensions Not Installing

**Issue**: Extensions fail to install
```
Solution: Install manually
1. Open Extensions panel (Ctrl+Shift+X)
2. Search for extension ID
3. Install in container
```

### Build Issues

**Issue**: catkin_make or colcon fails
```
Solution: Ensure ROS is sourced
source /opt/ros/noetic/setup.bash  # ROS1
source /opt/ros/humble/setup.bash  # ROS2
```

## Advanced: Custom Configuration

To customize the devcontainer:

1. Edit `.devcontainer/{ros1|ros2}/devcontainer.json`
2. Add extensions, settings, or lifecycle commands
3. Rebuild container: `F1` → "Rebuild Container"

### Example: Add More Extensions
```json
"extensions": [
  "ms-vscode.cpptools",
  "your-extension-id-here"
]
```

### Example: Change Default Shell
```json
"settings": {
  "terminal.integrated.defaultProfile.linux": "zsh"
}
```

### Example: Add Environment Variables
```json
"containerEnv": {
  "MY_VARIABLE": "value"
}
```

## Benefits Over Manual Attachment

✅ **Automatic extension installation**
✅ **Pre-configured IntelliSense**
✅ **Consistent environment across team**
✅ **Automatic workspace setup**
✅ **No manual sourcing needed**
✅ **Reproducible development environment**

## Comparison with ./run.sh Method

| Feature | ./run.sh + Attach | Dev Container |
|---------|-------------------|---------------|
| Container management | Manual | Automatic |
| Extension installation | Manual | Automatic |
| IntelliSense setup | Manual | Pre-configured |
| ROS sourcing | Manual each time | Automatic |
| Team sharing | Instructions only | Config file |
| VS Code integration | Basic | Full |

Both methods work - use what fits your workflow!
