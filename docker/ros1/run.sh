#!/bin/bash
cd "$(dirname "$0")/../.."
docker run -it --rm \
    --name plan_ga_ros1 \
    -v "$(pwd)"/src:/catkin_ws/src/plan_ga \
    -v "$(pwd)"/models:/models \
    -v "$(pwd)"/samples:/samples \
    --network host \
    plan_ga_ros1:latest
