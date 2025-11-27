FROM ros:humble-ros-core

# Install build essentials
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Install ROS2 dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-nav2-core \
    ros-humble-nav2-costmap-2d \
    ros-humble-tf2-ros \
    ros-humble-nav2-bringup \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz && \
    tar -xzf onnxruntime-linux-x64-1.16.3.tgz -C /opt/ && \
    rm onnxruntime-linux-x64-1.16.3.tgz

ENV ONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.16.3

# Create workspace
RUN mkdir -p /ros2_ws/src
WORKDIR /ros2_ws

# Source ROS
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
