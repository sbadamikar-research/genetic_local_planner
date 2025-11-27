FROM ros:noetic-ros-core

# Install build essentials
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install ROS1 dependencies
RUN apt-get update && apt-get install -y \
    ros-noetic-nav-core \
    ros-noetic-costmap-2d \
    ros-noetic-tf2-ros \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-pluginlib \
    ros-noetic-stage-ros \
    ros-noetic-move-base \
    && rm -rf /var/lib/apt/lists/*

# Install ONNX Runtime
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz && \
    tar -xzf onnxruntime-linux-x64-1.16.3.tgz -C /opt/ && \
    rm onnxruntime-linux-x64-1.16.3.tgz

ENV ONNXRUNTIME_ROOT=/opt/onnxruntime-linux-x64-1.16.3

# Create workspace
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

# Source ROS
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
