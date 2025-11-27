#!/bin/bash
cd "$(dirname "$0")/../.."
docker run -it --rm \
    --name plan_ga_ros2 \
    -v "$(pwd)"/src:/ros2_ws/src/plan_ga \
    -v "$(pwd)"/models:/models \
    -v "$(pwd)"/samples:/samples \
    --network host \
    plan_ga_ros2:latest
