#!/bin/bash

if docker ps --format '{{.Names}}' | grep -q '^plan_ga_ros2$'; then
    echo "Stopping container 'plan_ga_ros2'..."
    docker stop plan_ga_ros2
    echo "Container stopped."
else
    echo "Container 'plan_ga_ros2' is not running."
fi
