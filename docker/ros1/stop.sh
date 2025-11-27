#!/bin/bash

if docker ps --format '{{.Names}}' | grep -q '^plan_ga_ros1$'; then
    echo "Stopping container 'plan_ga_ros1'..."
    docker stop plan_ga_ros1
    echo "Container stopped."
else
    echo "Container 'plan_ga_ros1' is not running."
fi
