#!/bin/bash

if docker ps -a --format '{{.Names}}' | grep -q '^plan_ga_ros2$'; then
    echo "Removing container 'plan_ga_ros2'..."
    docker rm -f plan_ga_ros2
    echo "Container removed."
else
    echo "Container 'plan_ga_ros2' does not exist."
fi
