#!/bin/bash
cd "$(dirname "$0")/../.."
docker build -t plan_ga_ros2:latest -f docker/ros2/Dockerfile .
