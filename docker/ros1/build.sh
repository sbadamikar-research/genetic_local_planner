#!/bin/bash
cd "$(dirname "$0")/../.."
docker build -t plan_ga_ros1:latest -f docker/ros1/Dockerfile .
