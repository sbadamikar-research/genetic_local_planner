cmake_minimum_required(VERSION 3.8)
project(plan_ga_ros2)

# C++17 standard
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pluginlib REQUIRED)

# Find ONNX Runtime
set(ONNXRUNTIME_ROOT "/opt/onnxruntime-linux-x64-1.16.3" CACHE PATH "Path to ONNX Runtime")
if(EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    message(STATUS "Found ONNX Runtime at: ${ONNXRUNTIME_ROOT}")
    set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
else()
    message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}")
endif()

# Set dependencies for ament
set(dependencies
  rclcpp
  rclcpp_lifecycle
  nav2_core
  nav2_costmap_2d
  nav2_util
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pluginlib
)

# Include directories
include_directories(
  include
  ${CMAKE_CURRENT_SOURCE_DIR}/../plan_ga_planner/include
  ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Build plan_ga_planner core library (ROS-agnostic)
add_library(plan_ga_planner_core SHARED
    ../plan_ga_planner/src/onnx_inference.cpp
    ../plan_ga_planner/src/costmap_processor.cpp
    ../plan_ga_planner/src/trajectory_generator.cpp
    ../plan_ga_planner/src/collision_checker.cpp
    ../plan_ga_planner/src/planner_core.cpp
)
target_link_libraries(plan_ga_planner_core
    ${ONNXRUNTIME_LIB}
)

# Build ROS2 plugin
add_library(plan_ga_ros2_plugin SHARED
    src/plan_ga_ros2_plugin.cpp
)
target_link_libraries(plan_ga_ros2_plugin
    plan_ga_planner_core
)
ament_target_dependencies(plan_ga_ros2_plugin
    ${dependencies}
)

# Prevent pluginlib from using boost
target_compile_definitions(plan_ga_ros2_plugin PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")

# Install libraries
install(TARGETS plan_ga_planner_core plan_ga_ros2_plugin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install include directories
install(DIRECTORY include/
    DESTINATION include
)

# Install plugin description
install(FILES plan_ga_plugin.xml
    DESTINATION share/${PROJECT_NAME}
)

# Install ONNX Runtime library (for deployment)
install(FILES ${ONNXRUNTIME_LIB}
    DESTINATION lib
)

# Export plugin
pluginlib_export_plugin_description_file(nav2_core plan_ga_plugin.xml)

# Export dependencies
ament_export_include_directories(include)
ament_export_libraries(plan_ga_planner_core plan_ga_ros2_plugin)
ament_export_dependencies(${dependencies})

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
