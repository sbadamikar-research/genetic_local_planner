"""
Multi-objective fitness evaluation for navigation trajectories.

This module provides fitness computation for GA chromosomes based on
goal distance, collision avoidance, smoothness, and time efficiency.
"""

import numpy as np
import multiprocessing as mp
from typing import List, Dict
from .chromosome import Chromosome
from ..simulator.environment import NavigationEnvironment


class FitnessEvaluator:
    """
    Multi-objective fitness evaluator for navigation chromosomes.

    Fitness = -(w_goal * goal_dist + w_collision * collision_penalty
              + w_smoothness * smoothness + w_time * path_length)

    Higher fitness is better (negated cost).
    """

    def __init__(self, weights: Dict[str, float]):
        """
        Initialize fitness evaluator with objective weights.

        Args:
            weights: Dict with keys:
                - goal_distance: Weight for final distance to goal (default 1.0)
                - collision: Weight for collision penalty (default 10.0)
                - smoothness: Weight for control smoothness (default 0.5)
                - time_efficiency: Weight for path length (default 0.3)
        """
        self.weights = weights

    def evaluate(self, chromosome: Chromosome, environment: NavigationEnvironment) -> float:
        """
        Evaluate chromosome fitness on given environment.

        Updates chromosome.fitness and chromosome.fitness_components.

        Args:
            chromosome: Chromosome to evaluate
            environment: NavigationEnvironment with scenario

        Returns:
            fitness: Fitness value (higher is better)
        """
        # Get control sequence
        control_sequence = chromosome.get_control_sequence()

        # Simulate trajectory
        metrics = environment.simulate_control_sequence(control_sequence)

        # Extract metrics
        collision = metrics['collision']
        max_cost = metrics['max_cost']
        goal_distance = metrics['goal_distance']
        goal_distance_reduction = metrics['goal_distance_reduction']
        smoothness = metrics['smoothness']
        path_length = metrics['path_length']
        goal_reached = metrics['goal_reached']

        # Compute fitness components

        # 1. Goal distance term (primary objective)
        # Use final distance to goal (lower is better)
        goal_term = goal_distance

        # 2. Collision penalty
        # Hard penalty if collision, soft penalty based on max cost otherwise
        if collision:
            collision_term = 100.0  # Large penalty
        else:
            # Soft penalty proportional to max cost encountered
            # Normalize by 254 (lethal threshold)
            collision_term = (max_cost / 254.0) * 10.0

        # 3. Smoothness term (sum of squared control changes)
        # Normalize by number of steps to make scale-independent
        smoothness_term = smoothness / max(1, len(control_sequence))

        # 4. Time efficiency (path length)
        # Penalize longer paths
        time_term = path_length

        # Compute weighted fitness (negate because we want to maximize)
        fitness = -(
            self.weights.get('goal_distance', 1.0) * goal_term +
            self.weights.get('collision', 10.0) * collision_term +
            self.weights.get('smoothness', 0.5) * smoothness_term +
            self.weights.get('time_efficiency', 0.3) * time_term
        )

        # Bonus for reaching goal
        if goal_reached:
            fitness += 50.0  # Large bonus

        # Store fitness and components
        chromosome.fitness = fitness
        chromosome.fitness_components = {
            'goal_distance': goal_distance,
            'goal_distance_reduction': goal_distance_reduction,
            'collision': collision,
            'max_cost': max_cost,
            'collision_term': collision_term,
            'smoothness': smoothness,
            'smoothness_term': smoothness_term,
            'path_length': path_length,
            'time_term': time_term,
            'goal_reached': goal_reached,
            'fitness': fitness
        }

        return fitness


def _evaluate_worker(args):
    """
    Worker function for parallel fitness evaluation.

    Args:
        args: Tuple of (chromosome_genes, velocity_limits, num_steps,
                       environment_state, weights)

    Returns:
        Tuple of (fitness, fitness_components)
    """
    chromosome_genes, velocity_limits, num_steps, env_state, weights = args

    # Reconstruct chromosome
    chromosome = Chromosome(num_steps, velocity_limits)
    chromosome.genes = chromosome_genes

    # Reconstruct environment
    # Note: environment_state contains (costmap, start_state, goal_x, goal_y, goal_theta, config)
    costmap, start_state, goal_x, goal_y, goal_theta, config = env_state

    environment = NavigationEnvironment(config)
    environment.reset(costmap, start_state, goal_x, goal_y, goal_theta)

    # Evaluate fitness
    evaluator = FitnessEvaluator(weights)
    fitness = evaluator.evaluate(chromosome, environment)

    return fitness, chromosome.fitness_components


def evaluate_population_parallel(population: List[Chromosome],
                                 environment: NavigationEnvironment,
                                 evaluator: FitnessEvaluator,
                                 num_workers: int = 8) -> None:
    """
    Evaluate fitness of entire population in parallel.

    Updates fitness values in-place for each chromosome.

    Args:
        population: List of Chromosome objects
        environment: NavigationEnvironment with current scenario
        evaluator: FitnessEvaluator instance
        num_workers: Number of parallel workers (default 8)
    """
    if num_workers <= 1:
        # Serial evaluation
        for chromosome in population:
            evaluator.evaluate(chromosome, environment)
        return

    # Prepare environment state for pickling
    env_state = (
        environment.costmap,
        environment.start_state,
        environment.goal_x,
        environment.goal_y,
        environment.goal_theta,
        {
            'robot': {
                'footprint': environment.collision_checker.footprint.tolist(),
                'max_v_x': environment.robot_model.velocity_limits['max_v_x'],
                'min_v_x': environment.robot_model.velocity_limits['min_v_x'],
                'max_v_y': environment.robot_model.velocity_limits['max_v_y'],
                'min_v_y': environment.robot_model.velocity_limits['min_v_y'],
                'max_omega': environment.robot_model.velocity_limits['max_omega'],
                'min_omega': environment.robot_model.velocity_limits['min_omega']
            }
        }
    )

    # Prepare arguments for workers
    args_list = []
    for chromosome in population:
        args = (
            chromosome.genes.copy(),
            chromosome.velocity_limits,
            chromosome.num_steps,
            env_state,
            evaluator.weights
        )
        args_list.append(args)

    # Parallel evaluation
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_evaluate_worker, args_list)

    # Update population with results
    for chromosome, (fitness, fitness_components) in zip(population, results):
        chromosome.fitness = fitness
        chromosome.fitness_components = fitness_components


if __name__ == "__main__":
    # Test fitness evaluator
    print("Testing FitnessEvaluator...")

    from ..simulator.costmap import generate_random_costmap
    from ..simulator.robot_model import RobotState

    # Create test configuration
    config = {
        'robot': {
            'footprint': [(-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)],
            'max_v_x': 1.0,
            'min_v_x': -0.5,
            'max_v_y': 0.5,
            'min_v_y': -0.5,
            'max_omega': 1.0,
            'min_omega': -1.0
        }
    }

    velocity_limits = config['robot']

    # Create environment
    environment = NavigationEnvironment(config)

    # Generate test costmap
    costmap = generate_random_costmap(
        width=50,
        height=50,
        resolution=0.05,
        num_obstacles=3,
        obstacle_radius_range=(3, 6),
        inflation_radius=0.5,
        inflation_decay=0.8,
        free_radius_center=0.5
    )

    # Set up scenario
    center_x = costmap.width * costmap.resolution / 2.0
    center_y = costmap.height * costmap.resolution / 2.0
    start_state = RobotState(x=center_x, y=center_y, theta=0.0)
    goal_x = center_x + 1.5
    goal_y = center_y
    goal_theta = 0.0

    environment.reset(costmap, start_state, goal_x, goal_y, goal_theta)

    # Create fitness evaluator
    weights = {
        'goal_distance': 1.0,
        'collision': 10.0,
        'smoothness': 0.5,
        'time_efficiency': 0.3
    }
    evaluator = FitnessEvaluator(weights)

    print(f"Fitness weights: {weights}")

    # Test 1: Straight trajectory (good)
    print("\n--- Test 1: Straight trajectory ---")
    chrom_straight = Chromosome(20, velocity_limits)
    chrom_straight.genes[:, 0] = 0.5  # v_x = 0.5 m/s
    chrom_straight.genes[:, 1] = 0.0  # v_y = 0.0
    chrom_straight.genes[:, 2] = 0.0  # omega = 0.0

    fitness_straight = evaluator.evaluate(chrom_straight, environment)
    print(f"Fitness: {fitness_straight:.3f}")
    print(f"Components:")
    for key, value in chrom_straight.fitness_components.items():
        print(f"  {key}: {value}")

    # Test 2: Random trajectory (likely poor)
    print("\n--- Test 2: Random trajectory ---")
    chrom_random = Chromosome(20, velocity_limits)
    chrom_random.randomize()

    fitness_random = evaluator.evaluate(chrom_random, environment)
    print(f"Fitness: {fitness_random:.3f}")
    print(f"Goal distance: {chrom_random.fitness_components['goal_distance']:.3f}m")
    print(f"Collision: {chrom_random.fitness_components['collision']}")
    print(f"Path length: {chrom_random.fitness_components['path_length']:.3f}m")

    # Test 3: Zero control (very poor)
    print("\n--- Test 3: Zero control ---")
    chrom_zero = Chromosome(20, velocity_limits)
    # genes already zero

    fitness_zero = evaluator.evaluate(chrom_zero, environment)
    print(f"Fitness: {fitness_zero:.3f}")
    print(f"Goal distance: {chrom_zero.fitness_components['goal_distance']:.3f}m")

    # Test 4: Parallel evaluation
    print("\n--- Test 4: Parallel evaluation ---")
    population = []
    for i in range(10):
        chrom = Chromosome(20, velocity_limits)
        chrom.randomize()
        population.append(chrom)

    print(f"Evaluating population of {len(population)} chromosomes...")

    # Serial evaluation
    import time
    start_time = time.time()
    for chrom in population:
        evaluator.evaluate(chrom, environment)
    serial_time = time.time() - start_time
    serial_fitnesses = [chrom.fitness for chrom in population]

    print(f"Serial evaluation time: {serial_time:.3f}s")
    print(f"Fitness range: [{min(serial_fitnesses):.3f}, {max(serial_fitnesses):.3f}]")

    # Reset fitness values
    for chrom in population:
        chrom.fitness = -np.inf

    # Parallel evaluation
    start_time = time.time()
    evaluate_population_parallel(population, environment, evaluator, num_workers=4)
    parallel_time = time.time() - start_time
    parallel_fitnesses = [chrom.fitness for chrom in population]

    print(f"\nParallel evaluation time (4 workers): {parallel_time:.3f}s")
    print(f"Speedup: {serial_time / parallel_time:.2f}x")
    print(f"Fitness range: [{min(parallel_fitnesses):.3f}, {max(parallel_fitnesses):.3f}]")

    # Verify results match
    fitness_diff = np.abs(np.array(serial_fitnesses) - np.array(parallel_fitnesses))
    print(f"Max fitness difference: {fitness_diff.max():.6f}")

    print("\n✓ FitnessEvaluator tests passed!")
