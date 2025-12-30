# Module 05: C++ Integration & ONNX Runtime

**Estimated Time:** 1 day (6-8 hours)

## 🎯 Learning Objectives

- ✅ Understand C++ basics for this project
- ✅ Load and run ONNX models in C++
- ✅ Build ROS plugins (ROS1 and ROS2)
- ✅ Debug C++/Python integration issues
- ✅ Profile C++ inference performance
- ✅ Understand plugin architecture (pluginlib)

## Key Concepts

### Why C++ for Deployment?

| Aspect | Python | C++ | Winner |
|--------|--------|-----|--------|
| Inference Speed | ~10-20ms | ~2-5ms | ✅ C++ |
| Memory Usage | High (interpreter) | Low | ✅ C++ |
| Real-time Capable | No (GIL) | Yes | ✅ C++ |
| Development Speed | Fast | Slower | Python |
| ROS Integration | Good | Native | ✅ C++ |

**For 10-20 Hz control:** C++ is essential!

---

## Hands-On Exercises

### Exercise 1: Build Core Planner Library (1 hour)

First, build the ROS-agnostic core library:

```bash
# Start ROS1 container
cd docker/ros1
./build.sh  # If not built
./run.sh    # Start container

# Attach to container
docker exec -it plan_ga_ros1 bash

# Inside container
cd /catkin_ws
source /opt/ros/noetic/setup.bash

# Build
catkin_make

# Check build success
ls devel/lib/libplan_ga_planner.so
# Should exist if successful
```

**If build fails**, check:
1. ONNX Runtime installed? `ldconfig -p | grep onnx`
2. Correct CMake? `cmake --version` (need 3.10+)
3. Check error logs in build output

### Exercise 2: Test ONNX Inference in C++ (1.5 hours)

Create a test program:

```cpp
// test_onnx_cpp.cpp
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>

int main() {
    // 1. Create ONNX Runtime environment
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    
    // 2. Session options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED
    );
    
    // 3. Load model
    const char* model_path = "/models/planner_policy.onnx";
    Ort::Session session(env, model_path, session_options);
    
    // 4. Get input/output info
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    
    std::cout << "Inputs: " << num_input_nodes << std::endl;
    std::cout << "Outputs: " << num_output_nodes << std::endl;
    
    // 5. Prepare inputs (dummy data)
    std::vector<float> costmap_data(1 * 1 * 50 * 50, 0.5f);
    std::vector<float> robot_state_data(1 * 9, 0.0f);
    std::vector<float> goal_data(1 * 3, 1.0f);
    std::vector<float> metadata_data(1 * 2, 0.05f);
    
    // 6. Create input tensors
    auto memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );
    
    std::vector<int64_t> costmap_shape = {1, 1, 50, 50};
    std::vector<int64_t> state_shape = {1, 9};
    std::vector<int64_t> goal_shape = {1, 3};
    std::vector<int64_t> metadata_shape = {1, 2};
    
    Ort::Value costmap_tensor = Ort::Value::CreateTensor<float>(
        memory_info, costmap_data.data(), costmap_data.size(),
        costmap_shape.data(), costmap_shape.size()
    );
    
    Ort::Value state_tensor = Ort::Value::CreateTensor<float>(
        memory_info, robot_state_data.data(), robot_state_data.size(),
        state_shape.data(), state_shape.size()
    );
    
    Ort::Value goal_tensor = Ort::Value::CreateTensor<float>(
        memory_info, goal_data.data(), goal_data.size(),
        goal_shape.data(), goal_shape.size()
    );
    
    Ort::Value metadata_tensor = Ort::Value::CreateTensor<float>(
        memory_info, metadata_data.data(), metadata_data.size(),
        metadata_shape.data(), metadata_shape.size()
    );
    
    // 7. Input/output names (MUST match Python export!)
    const char* input_names[] = {
        "costmap_input", "robot_state_input",
        "goal_relative_input", "costmap_metadata_input"
    };
    const char* output_names[] = {"output"};
    
    // 8. Run inference
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(costmap_tensor));
    input_tensors.push_back(std::move(state_tensor));
    input_tensors.push_back(std::move(goal_tensor));
    input_tensors.push_back(std::move(metadata_tensor));
    
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, input_tensors.data(), 4,
        output_names, 1
    );
    
    // 9. Extract output
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    std::cout << "Output shape: [" << output_shape[0] << ", " 
              << output_shape[1] << "]" << std::endl;
    std::cout << "First 5 values: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << output_data[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "✓ Inference successful!" << std::endl;
    
    return 0;
}
```

**Compile and run:**
```bash
g++ test_onnx_cpp.cpp -o test_onnx \
  -I/usr/include/onnxruntime \
  -lonnxruntime \
  -std=c++17

./test_onnx
```

### Exercise 3: Study Core Planner Code (1.5 hours)

Read and understand key files:

**File 1: `src/plan_ga_planner/include/plan_ga_planner/onnx_inference.h`**

```cpp
class ONNXInference {
public:
    ONNXInference(const std::string& model_path);
    
    // Run inference with multiple inputs
    std::vector<float> infer(
        const std::vector<float>& costmap,      // [1, 1, 50, 50] flattened
        const std::vector<float>& robot_state,  // [1, 9]
        const std::vector<float>& goal_relative, // [1, 3]
        const std::vector<float>& costmap_metadata // [1, 2]
    );
    
private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};
```

**Trace the code flow:**
1. Constructor loads model: `onnx_inference.cpp:30`
2. `infer()` prepares tensors: `onnx_inference.cpp:80`
3. Runs ONNX session: `onnx_inference.cpp:150`
4. Returns output vector: `onnx_inference.cpp:180`

**File 2: `src/plan_ga_planner/include/plan_ga_planner/planner_core.h`**

```cpp
class PlannerCore {
public:
    bool initialize(const std::string& model_path, const PlannerConfig& config);
    
    // Main planning function
    Velocity computeVelocity(
        const Pose& current_pose,
        const Velocity& current_velocity,
        const Pose& goal_pose,
        const Costmap& costmap
    );
    
private:
    std::unique_ptr<ONNXInference> onnx_inference_;
    std::unique_ptr<CostmapProcessor> costmap_processor_;
    std::unique_ptr<TrajectoryGenerator> trajectory_generator_;
    std::unique_ptr<CollisionChecker> collision_checker_;
};
```

**Task:** Draw a flowchart of `computeVelocity()`:
1. Extract 50×50 costmap window
2. Prepare ONNX inputs
3. Run inference → get control sequence
4. Simulate trajectory
5. Check collisions
6. Return first control step

### Exercise 4: Build and Test ROS1 Plugin (2 hours)

```bash
# Inside ROS1 container
cd /catkin_ws
source /opt/ros/noetic/setup.bash

# Build
catkin_make

# Source workspace
source devel/setup.bash

# Verify plugin loads
rospack plugins --attrib=plugin nav_core | grep plan_ga

# Should output:
# plan_ga_ros1 /catkin_ws/src/plan_ga/plan_ga_ros1/plan_ga_plugin.xml
```

**If plugin doesn't appear:**
1. Check `plan_ga_plugin.xml` exists
2. Verify `package.xml` has `<export>` tag
3. Re-run `catkin_make`

**Test plugin (requires model file):**
```bash
# First, copy ONNX model to container
# (On host)
docker cp models/planner_policy.onnx plan_ga_ros1:/models/

# (In container)
# Launch with your plugin
# roslaunch your_robot_bringup navigation.launch local_planner:=plan_ga_ros1/PlanGAPlanner
```

### Exercise 5: Profile C++ vs Python (1 hour)

Compare inference times:

```cpp
// profile_inference.cpp
#include <chrono>
#include <iostream>
#include "plan_ga_planner/onnx_inference.h"

int main() {
    ONNXInference inference("/models/planner_policy.onnx");
    
    // Prepare dummy inputs
    std::vector<float> costmap(1*1*50*50, 0.5f);
    std::vector<float> robot_state(9, 0.0f);
    std::vector<float> goal(3, 1.0f);
    std::vector<float> metadata(2, 0.05f);
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        inference.infer(costmap, robot_state, goal, metadata);
    }
    
    // Benchmark
    const int num_runs = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_runs; ++i) {
        inference.infer(costmap, robot_state, goal, metadata);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    double avg_time_ms = duration.count() / (num_runs * 1000.0);
    
    std::cout << "Average inference time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Max frequency: " << 1000.0 / avg_time_ms << " Hz" << std::endl;
    
    if (avg_time_ms < 50.0) {
        std::cout << "✓ Can achieve 20 Hz!" << std::endl;
    } else if (avg_time_ms < 100.0) {
        std::cout << "✓ Can achieve 10 Hz" << std::endl;
    } else {
        std::cout << "✗ Too slow for real-time" << std::endl;
    }
    
    return 0;
}
```

---

## Code Walkthrough

### Plugin Architecture (ROS1)

**File:** `src/plan_ga_ros1/src/plan_ga_ros1_plugin.cpp`

Key methods:
```cpp
bool PlanGAROS1Plugin::initialize(std::string name, tf2_ros::Buffer* tf, 
                                    costmap_2d::Costmap2DROS* costmap_ros) {
    // 1. Save TF and costmap references
    tf_ = tf;
    costmap_ros_ = costmap_ros;
    
    // 2. Load parameters from ROS parameter server
    ros::NodeHandle private_nh("~/" + name);
    std::string model_path;
    private_nh.param("model_path", model_path, 
                     std::string("/models/planner_policy.onnx"));
    
    // 3. Initialize core planner
    PlannerConfig config;
    // ... load config from parameters ...
    
    planner_core_ = std::make_unique<PlannerCore>();
    if (!planner_core_->initialize(model_path, config)) {
        ROS_ERROR("Failed to initialize planner core");
        return false;
    }
    
    initialized_ = true;
    return true;
}

bool PlanGAROS1Plugin::computeVelocityCommands(geometry_msgs::Twist& cmd_vel) {
    // 1. Get robot pose from TF
    geometry_msgs::PoseStamped robot_pose;
    costmap_ros_->getRobotPose(robot_pose);
    
    // 2. Convert ROS costmap to internal format
    costmap_2d::Costmap2D* costmap = costmap_ros_->getCostmap();
    Costmap internal_costmap = convertCostmap(costmap);
    
    // 3. Get goal pose
    Pose goal = getGoal();  // From global plan
    
    // 4. Call core planner
    Velocity velocity = planner_core_->computeVelocity(
        toPose(robot_pose), current_velocity_, goal, internal_costmap
    );
    
    // 5. Convert to ROS Twist message
    cmd_vel.linear.x = velocity.v_x;
    cmd_vel.linear.y = velocity.v_y;
    cmd_vel.angular.z = velocity.omega;
    
    return true;
}
```

**Key Points:**
- Plugin implements `nav_core::BaseLocalPlanner` interface
- `initialize()` loads model and config
- `computeVelocityCommands()` called at control frequency (~10 Hz)
- Converts between ROS types and internal types

---

## Common C++ Issues

### Issue 1: Linker Error - ONNX Runtime Not Found

**Error:**
```
undefined reference to `Ort::Session::Session(...)`
```

**Solution:**
```cmake
# In CMakeLists.txt
find_library(ONNXRUNTIME_LIB onnxruntime)
target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_LIB})
```

### Issue 2: Segmentation Fault

**Cause:** Usually dereferencing null pointer or accessing invalid memory

**Debug:**
```bash
# Run with gdb
gdb --args ./your_program

# When it crashes:
(gdb) bt  # backtrace
(gdb) frame 0  # inspect frame
(gdb) print variable_name  # check values
```

### Issue 3: Model Not Found

**Error:**
```
terminate called after throwing an instance of 'Ort::Exception'
  what():  Load model from /models/planner_policy.onnx failed
```

**Solution:**
1. Check file exists: `ls /models/planner_policy.onnx`
2. Check permissions: `chmod 644 /models/planner_policy.onnx`
3. Use absolute path in config

---

## Quiz

1. **Why use C++ for deployment instead of Python?**
   a) C++ is easier to write
   b) Lower inference latency (2-5ms vs 10-20ms)
   c) Better ML libraries
   d) ROS requirement

2. **What does `Ort::Session` do?**
   a) Train neural networks
   b) Load and run ONNX models
   c) Export models
   d) Manage ROS topics

3. **What interface does the ROS1 plugin implement?**
   a) `rclcpp::Node`
   b) `nav_core::BaseLocalPlanner`
   c) `geometry_msgs::Twist`
   d) `costmap_2d::Costmap2D`

4. **Why must input names match between Python and C++?**
   a) ONNX Runtime requirement for correct tensor routing
   b) Better performance
   c) ROS convention
   d) No reason

5. **What's a typical C++ inference time target?**
   a) 100ms (10 Hz)
   b) 50ms (20 Hz)
   c) 2-5ms (200-500 Hz capability, run at 10-20 Hz)
   d) 1ms (1000 Hz)

<details>
<summary><b>Show Answers</b></summary>

1. b) Lower inference latency
2. b) Load and run ONNX models
3. b) `nav_core::BaseLocalPlanner`
4. a) ONNX Runtime requirement for correct tensor routing
5. c) 2-5ms (fast enough for 200-500 Hz, run at 10-20 Hz for safety)
</details>

---

## ✅ Checklist

- [ ] Build core planner library successfully
- [ ] Run ONNX inference in C++
- [ ] Understand plugin architecture
- [ ] Build and load ROS1/ROS2 plugin
- [ ] Profile inference performance
- [ ] Quiz score 80%+

---

## 📚 Resources

- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/)
- [pluginlib Tutorial](http://wiki.ros.org/pluginlib/Tutorials)
- [nav_core::BaseLocalPlanner](http://docs.ros.org/en/noetic/api/nav_core/html/classnav__core_1_1BaseLocalPlanner.html)
- [C++ Debugging with GDB](https://www.gnu.org/software/gdb/documentation/)

---

## 🎉 Next Steps

You can now build C++ plugins! Next, learn the ROS navigation stack.

**→ [Continue to Module 06: ROS Fundamentals](../06-ros-fundamentals/)**
