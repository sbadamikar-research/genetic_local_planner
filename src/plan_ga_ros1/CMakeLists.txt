cmake_minimum_required(VERSION 3.10)
project(plan_ga_ros1)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Find catkin packages
find_package(catkin REQUIRED COMPONENTS
    roscpp
    nav_core
    costmap_2d
    tf2_ros
    tf2_geometry_msgs
    geometry_msgs
    nav_msgs
    pluginlib
)

# Find ONNX Runtime
set(ONNXRUNTIME_ROOT "/opt/onnxruntime-linux-x64-1.16.3" CACHE PATH "Path to ONNX Runtime")
if(EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    message(STATUS "Found ONNX Runtime at: ${ONNXRUNTIME_ROOT}")
    set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
else()
    message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}")
endif()

# Catkin package
catkin_package(
    INCLUDE_DIRS include
    LIBRARIES plan_ga_ros1_plugin plan_ga_planner_core
    CATKIN_DEPENDS
        roscpp
        nav_core
        costmap_2d
        tf2_ros
        tf2_geometry_msgs
        geometry_msgs
        nav_msgs
        pluginlib
)

# Include directories
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../plan_ga_planner/include
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Build plan_ga_planner core library (ROS-agnostic)
add_library(plan_ga_planner_core
    ../plan_ga_planner/src/onnx_inference.cpp
    ../plan_ga_planner/src/costmap_processor.cpp
    ../plan_ga_planner/src/trajectory_generator.cpp
    ../plan_ga_planner/src/collision_checker.cpp
    ../plan_ga_planner/src/planner_core.cpp
)
target_link_libraries(plan_ga_planner_core
    ${ONNXRUNTIME_LIB}
)

# Build ROS1 plugin
add_library(plan_ga_ros1_plugin
    src/plan_ga_ros1_plugin.cpp
)
target_link_libraries(plan_ga_ros1_plugin
    plan_ga_planner_core
    ${catkin_LIBRARIES}
)
add_dependencies(plan_ga_ros1_plugin
    ${catkin_EXPORTED_TARGETS}
)

# Install
install(TARGETS plan_ga_planner_core plan_ga_ros1_plugin
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(FILES plan_ga_plugin.xml
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# Install ONNX Runtime library (for deployment)
install(FILES ${ONNXRUNTIME_LIB}
    DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
