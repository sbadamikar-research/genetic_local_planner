#!/bin/bash

if docker ps -a --format '{{.Names}}' | grep -q '^plan_ga_ros1$'; then
    echo "Removing container 'plan_ga_ros1'..."
    docker rm -f plan_ga_ros1
    echo "Container removed."
else
    echo "Container 'plan_ga_ros1' does not exist."
fi
