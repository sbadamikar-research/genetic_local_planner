#!/bin/bash
cd "$(dirname "$0")/../.."

# Check if container is already running
if docker ps -a --format '{{.Names}}' | grep -q '^plan_ga_ros1$'; then
    echo "Container 'plan_ga_ros1' already exists."
    if docker ps --format '{{.Names}}' | grep -q '^plan_ga_ros1$'; then
        echo "Container is running. Attach with: docker exec -it plan_ga_ros1 bash"
    else
        echo "Starting existing container..."
        docker start plan_ga_ros1
        echo "Container started. Attach with: docker exec -it plan_ga_ros1 bash"
    fi
else
    echo "Creating and starting new container 'plan_ga_ros1'..."
    docker run -d \
        --name plan_ga_ros1 \
        -v "$(pwd)"/src/plan_ga_planner:/catkin_ws/src/plan_ga/plan_ga_planner \
        -v "$(pwd)"/src/plan_ga_ros1:/catkin_ws/src/plan_ga/plan_ga_ros1 \
        -v "$(pwd)"/models:/models \
        -v "$(pwd)"/samples:/samples \
        --network host \
        plan_ga_ros1:latest \
        tail -f /dev/null
    echo "Container started in background."
    echo "Attach with: docker exec -it plan_ga_ros1 bash"
    echo "Or use VS Code: Dev Containers -> Attach to Running Container -> plan_ga_ros1"
fi
