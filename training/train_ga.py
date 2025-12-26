"""
Main GA training script for trajectory generation.

This script orchestrates the genetic algorithm training process:
1. Generate random navigation scenarios
2. Evolve optimal trajectories using GA
3. Export best trajectories in format for neural network training
4. Save results with periodic checkpointing

Usage:
    python training/train_ga.py \
        --config training/config/ga_config.yaml \
        --output models/checkpoints/ga_trajectories.pkl \
        --num_scenarios 1000 \
        --num_workers 8
"""

import argparse
import pickle
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from simulator import (
    Costmap,
    generate_random_costmap,
    RobotState,
    NavigationEnvironment
)
from ga import GeneticAlgorithm


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_scenario(config: dict, scenario_id: int) -> Dict:
    """
    Generate random navigation scenario.

    Creates costmap with random obstacles, start position at center,
    and goal at random distance/angle.

    Args:
        config: Configuration dict
        scenario_id: Scenario ID for reproducible seeding

    Returns:
        scenario: Dict with 'costmap', 'start_state', 'goal_x', 'goal_y', 'goal_theta'
    """
    # Set random seed for reproducibility
    random_seed = config['scenarios'].get('random_seed', 42)
    np.random.seed(random_seed + scenario_id)

    # Costmap configuration
    costmap_config = config['costmap']
    width = costmap_config['width']
    height = costmap_config['height']
    resolution = costmap_config['resolution']
    inflation_radius = costmap_config['inflation_radius']
    inflation_decay = costmap_config['inflation_decay']

    # Random number of obstacles
    num_obstacles = np.random.randint(
        costmap_config['num_obstacles_min'],
        costmap_config['num_obstacles_max'] + 1
    )

    # Generate costmap
    costmap = generate_random_costmap(
        width=width,
        height=height,
        resolution=resolution,
        num_obstacles=num_obstacles,
        obstacle_radius_range=(3, 6),
        inflation_radius=inflation_radius,
        inflation_decay=inflation_decay,
        free_radius_center=0.5
    )

    # Start at center with random orientation
    center_x = width * resolution / 2.0
    center_y = height * resolution / 2.0
    start_theta = np.random.uniform(-np.pi, np.pi)
    start_state = RobotState(x=center_x, y=center_y, theta=start_theta)

    # Goal at random distance and angle
    scenario_config = config['scenarios']
    goal_distance = np.random.uniform(
        scenario_config['goal_distance_min'],
        scenario_config['goal_distance_max']
    )
    goal_angle = np.random.uniform(-np.pi, np.pi)

    goal_x = center_x + goal_distance * np.cos(goal_angle)
    goal_y = center_y + goal_distance * np.sin(goal_angle)
    goal_theta = goal_angle  # Align with direction to goal

    return {
        'costmap': costmap,
        'start_state': start_state,
        'goal_x': goal_x,
        'goal_y': goal_y,
        'goal_theta': goal_theta,
        'scenario_id': scenario_id
    }


def trajectory_to_dict(chromosome, environment: NavigationEnvironment,
                      scenario_id: int) -> Dict:
    """
    Convert GA chromosome to dictionary format for NN training.

    This format matches the requirements in neural_network/dataset.py.

    Args:
        chromosome: Best chromosome from GA
        environment: NavigationEnvironment with scenario
        scenario_id: Scenario identifier

    Returns:
        trajectory_dict: Dict with all required fields for NN training
    """
    # Get control sequence
    control_sequence = chromosome.get_control_sequence()

    # Extract costmap window (50x50 around start)
    costmap_window = environment.get_costmap_window(
        environment.start_state, window_size=50
    )

    # Robot state array
    robot_state = environment.start_state.to_array()

    # Goal in robot frame
    goal_relative = environment.get_relative_goal(environment.start_state)

    # Costmap metadata
    costmap_metadata = np.array([
        environment.costmap.inflation_decay,
        environment.costmap.resolution
    ], dtype=np.float32)

    return {
        'scenario_id': scenario_id,
        'costmap': costmap_window,  # (50, 50) float32 [0,1]
        'robot_state': robot_state,  # (9,) float32
        'goal_relative': goal_relative,  # (3,) float32
        'costmap_metadata': costmap_metadata,  # (2,) float32
        'control_sequence': control_sequence,  # (20, 3) float32
        'fitness': chromosome.fitness,
        'fitness_components': chromosome.fitness_components
    }


def save_checkpoint(trajectories: List[Dict], checkpoint_path: str):
    """
    Save trajectories to checkpoint file.

    Args:
        trajectories: List of trajectory dicts
        checkpoint_path: Path to save checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    with open(checkpoint_path, 'wb') as f:
        pickle.dump(trajectories, f)


def load_checkpoint(checkpoint_path: str) -> List[Dict]:
    """
    Load trajectories from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        trajectories: List of trajectory dicts
    """
    with open(checkpoint_path, 'rb') as f:
        trajectories = pickle.load(f)
    return trajectories


def print_statistics(trajectories: List[Dict]):
    """
    Print training statistics.

    Args:
        trajectories: List of trajectory dicts
    """
    fitnesses = [t['fitness'] for t in trajectories]
    goal_distances = [t['fitness_components']['goal_distance'] for t in trajectories]
    collisions = [t['fitness_components']['collision'] for t in trajectories]
    goal_reached = [t['fitness_components']['goal_reached'] for t in trajectories]

    print("\n" + "="*60)
    print("TRAINING STATISTICS")
    print("="*60)
    print(f"Total trajectories: {len(trajectories)}")
    print(f"\nFitness:")
    print(f"  Mean: {np.mean(fitnesses):.3f} ± {np.std(fitnesses):.3f}")
    print(f"  Min: {np.min(fitnesses):.3f}")
    print(f"  Max: {np.max(fitnesses):.3f}")
    print(f"  Median: {np.median(fitnesses):.3f}")
    print(f"\nGoal Distance:")
    print(f"  Mean: {np.mean(goal_distances):.3f}m ± {np.std(goal_distances):.3f}m")
    print(f"  Min: {np.min(goal_distances):.3f}m")
    print(f"  Max: {np.max(goal_distances):.3f}m")
    print(f"\nSuccess Metrics:")
    print(f"  Collision rate: {np.sum(collisions) / len(collisions) * 100:.1f}%")
    print(f"  Goal reached rate: {np.sum(goal_reached) / len(goal_reached) * 100:.1f}%")
    print("="*60 + "\n")


def main():
    """Main training loop."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train GA for trajectory generation')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to GA config YAML file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for trajectory pickle file')
    parser.add_argument('--num_scenarios', type=int, default=1000,
                       help='Number of scenarios to generate (default: 1000)')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='Number of parallel workers (default: 8)')
    parser.add_argument('--checkpoint_interval', type=int, default=100,
                       help='Save checkpoint every N scenarios (default: 100)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint file')
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)

    # Display configuration
    print(f"\nGA Configuration:")
    print(f"  Population size: {config['ga']['population_size']}")
    print(f"  Elite size: {config['ga']['elite_size']}")
    print(f"  Num generations: {config['ga']['num_generations']}")
    print(f"  Mutation rate: {config['ga']['mutation_rate']}")
    print(f"  Crossover rate: {config['ga']['crossover_rate']}")
    print(f"\nTraining Configuration:")
    print(f"  Num scenarios: {args.num_scenarios}")
    print(f"  Num workers: {args.num_workers}")
    print(f"  Checkpoint interval: {args.checkpoint_interval}")
    print(f"  Output path: {args.output}")

    # Create GA and environment
    print(f"\nInitializing GA...")
    ga = GeneticAlgorithm(config)
    environment = NavigationEnvironment(config)

    # Resume from checkpoint if specified
    trajectories = []
    start_scenario = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trajectories = load_checkpoint(args.resume)
        start_scenario = len(trajectories)
        print(f"Loaded {start_scenario} trajectories")

    # Training loop
    print(f"\nStarting training...")
    print(f"Scenarios: {start_scenario} → {args.num_scenarios}")

    checkpoint_path = Path(args.output).parent / "checkpoint_latest.pkl"

    for scenario_id in tqdm(range(start_scenario, args.num_scenarios),
                           desc="Training scenarios"):
        # Generate scenario
        scenario = generate_scenario(config, scenario_id)

        # Reset environment
        environment.reset(
            scenario['costmap'],
            scenario['start_state'],
            scenario['goal_x'],
            scenario['goal_y'],
            scenario['goal_theta']
        )

        # Run GA evolution
        best_chromosome, fitness_history = ga.run(
            environment,
            num_generations=config['ga']['num_generations'],
            num_workers=args.num_workers,
            verbose=False  # Suppress per-generation output
        )

        # Convert to trajectory dict
        trajectory_dict = trajectory_to_dict(
            best_chromosome, environment, scenario_id
        )

        trajectories.append(trajectory_dict)

        # Periodic checkpointing
        if (scenario_id + 1) % args.checkpoint_interval == 0:
            save_checkpoint(trajectories, checkpoint_path)
            tqdm.write(f"Checkpoint saved at scenario {scenario_id + 1}")

            # Print intermediate statistics
            recent_trajectories = trajectories[-args.checkpoint_interval:]
            recent_fitnesses = [t['fitness'] for t in recent_trajectories]
            tqdm.write(f"  Recent fitness: {np.mean(recent_fitnesses):.3f} ± "
                      f"{np.std(recent_fitnesses):.3f}")

    # Save final results
    print(f"\nSaving final trajectories to {args.output}...")
    save_checkpoint(trajectories, args.output)

    # Print statistics
    print_statistics(trajectories)

    print(f"✓ Training complete!")
    print(f"Trajectories saved to: {args.output}")
    print(f"\nNext step: Train neural network using:")
    print(f"  python training/train_nn.py \\")
    print(f"    --data {args.output} \\")
    print(f"    --config training/config/nn_config.yaml \\")
    print(f"    --output models/planner_policy.onnx")


if __name__ == "__main__":
    main()
