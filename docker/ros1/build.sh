#!/bin/bash
cd "$(dirname "$0")/../.."
docker build \
    --network host \
    -t plan_ga_ros1:latest \
    -f docker/ros1/Dockerfile .
