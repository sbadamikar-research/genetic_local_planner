"""
Genetic algorithm evolution loop.

This module provides the main GA class that orchestrates the evolution
process through generations of selection, crossover, and mutation.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from .chromosome import Chromosome
from .fitness import FitnessEvaluator, evaluate_population_parallel
from .operators import (
    tournament_selection,
    uniform_crossover,
    gaussian_mutation,
    elitism_selection
)
from ..simulator.environment import NavigationEnvironment


class GeneticAlgorithm:
    """
    Genetic algorithm for trajectory optimization.

    Evolves population of control sequences to maximize fitness.
    """

    def __init__(self, config: dict):
        """
        Initialize genetic algorithm with configuration.

        Args:
            config: Configuration dict with sections:
                - ga: population_size, elite_size, mutation_rate, crossover_rate,
                      num_generations, time_horizon, control_frequency
                - fitness_weights: goal_distance, collision, smoothness, time_efficiency
                - robot: velocity_limits and footprint
        """
        self.ga_config = config['ga']
        self.robot_config = config['robot']

        # GA parameters
        self.population_size = self.ga_config['population_size']
        self.elite_size = self.ga_config['elite_size']
        self.mutation_rate = self.ga_config['mutation_rate']
        self.crossover_rate = self.ga_config['crossover_rate']
        self.num_generations = self.ga_config['num_generations']

        # Control sequence parameters
        time_horizon = self.ga_config['time_horizon']
        control_frequency = self.ga_config['control_frequency']
        self.num_steps = int(time_horizon * control_frequency)

        # Velocity limits
        self.velocity_limits = {
            'max_v_x': self.robot_config['max_v_x'],
            'min_v_x': self.robot_config['min_v_x'],
            'max_v_y': self.robot_config['max_v_y'],
            'min_v_y': self.robot_config['min_v_y'],
            'max_omega': self.robot_config['max_omega'],
            'min_omega': self.robot_config['min_omega']
        }

        # Fitness evaluator
        self.fitness_evaluator = FitnessEvaluator(config['fitness_weights'])

    def initialize_population(self) -> List[Chromosome]:
        """
        Create random initial population.

        Returns:
            population: List of random chromosomes
        """
        population = []
        for _ in range(self.population_size):
            chromosome = Chromosome(self.num_steps, self.velocity_limits)
            chromosome.randomize()
            population.append(chromosome)
        return population

    def evolve_generation(self, population: List[Chromosome],
                         environment: NavigationEnvironment,
                         num_workers: int = 8) -> List[Chromosome]:
        """
        Evolve population by one generation.

        Algorithm:
        1. Evaluate fitness (parallel)
        2. Select elites
        3. Generate offspring (selection + crossover + mutation)
        4. Combine elites + offspring

        Args:
            population: Current population
            environment: NavigationEnvironment with scenario
            num_workers: Number of parallel workers for fitness evaluation

        Returns:
            next_population: New population for next generation
        """
        # 1. Evaluate fitness
        evaluate_population_parallel(
            population, environment, self.fitness_evaluator, num_workers
        )

        # 2. Select elites
        elites = elitism_selection(population, self.elite_size)

        # 3. Generate offspring
        offspring = []
        num_offspring = self.population_size - self.elite_size

        for _ in range(num_offspring):
            # Selection
            parent1 = tournament_selection(population, tournament_size=5)
            parent2 = tournament_selection(population, tournament_size=5)

            # Crossover
            child = uniform_crossover(parent1, parent2, self.crossover_rate)

            # Mutation
            gaussian_mutation(child, self.mutation_rate, mutation_strength=0.1)

            offspring.append(child)

        # 4. Combine elites and offspring
        next_population = elites + offspring

        return next_population

    def run(self, environment: NavigationEnvironment, num_generations: int = None,
            num_workers: int = 8, verbose: bool = True,
            callback: Optional[Callable] = None) -> Tuple[Chromosome, List[float]]:
        """
        Run GA evolution for multiple generations.

        Args:
            environment: NavigationEnvironment with scenario
            num_generations: Number of generations (default from config)
            num_workers: Number of parallel workers
            verbose: Print progress (default True)
            callback: Optional callback function called after each generation.
                      Signature: callback(generation, population, environment, best_chromosome)
                      Should return True to continue, False to stop early.

        Returns:
            Tuple of:
                - best_chromosome: Best chromosome found
                - fitness_history: List of best fitness per generation
        """
        if num_generations is None:
            num_generations = self.num_generations

        # Initialize population
        population = self.initialize_population()

        # Fitness history
        fitness_history = []
        best_chromosome = None

        # Evolution loop
        for generation in range(num_generations):
            # Evolve generation
            population = self.evolve_generation(population, environment, num_workers)

            # Track best
            best_in_gen = max(population, key=lambda c: c.fitness)
            fitness_history.append(best_in_gen.fitness)

            if best_chromosome is None or best_in_gen.fitness > best_chromosome.fitness:
                best_chromosome = best_in_gen.copy()

            # Call callback if provided
            if callback is not None:
                should_continue = callback(generation, population, environment, best_chromosome)
                if not should_continue:
                    print(f"Training stopped by callback at generation {generation}")
                    break

            # Progress
            if verbose:
                avg_fitness = np.mean([c.fitness for c in population])
                std_fitness = np.std([c.fitness for c in population])
                print(f"Gen {generation+1}/{num_generations}: "
                      f"Best={best_in_gen.fitness:.3f}, "
                      f"Avg={avg_fitness:.3f}±{std_fitness:.3f}")

        return best_chromosome, fitness_history

    def get_statistics(self, population: List[Chromosome]) -> Dict:
        """
        Compute population statistics.

        Args:
            population: List of chromosomes (must have fitness)

        Returns:
            stats: Dict with mean, std, min, max fitness
        """
        fitnesses = np.array([c.fitness for c in population])

        return {
            'mean': float(np.mean(fitnesses)),
            'std': float(np.std(fitnesses)),
            'min': float(np.min(fitnesses)),
            'max': float(np.max(fitnesses)),
            'median': float(np.median(fitnesses))
        }


if __name__ == "__main__":
    # Test genetic algorithm
    print("Testing GeneticAlgorithm...")

    from ..simulator.costmap import generate_random_costmap
    from ..simulator.robot_model import RobotState

    # Create test configuration
    config = {
        'ga': {
            'population_size': 20,
            'elite_size': 2,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'num_generations': 10,
            'time_horizon': 2.0,
            'control_frequency': 10.0
        },
        'fitness_weights': {
            'goal_distance': 1.0,
            'collision': 10.0,
            'smoothness': 0.5,
            'time_efficiency': 0.3
        },
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

    # Create GA
    ga = GeneticAlgorithm(config)
    print(f"GA configuration:")
    print(f"  Population size: {ga.population_size}")
    print(f"  Elite size: {ga.elite_size}")
    print(f"  Num steps: {ga.num_steps}")

    # Create environment
    environment = NavigationEnvironment(config)

    # Generate test costmap (easy scenario)
    costmap = generate_random_costmap(
        width=50,
        height=50,
        resolution=0.05,
        num_obstacles=2,
        obstacle_radius_range=(3, 5),
        inflation_radius=0.5,
        inflation_decay=0.8,
        free_radius_center=0.5
    )

    # Set up scenario (goal straight ahead)
    center_x = costmap.width * costmap.resolution / 2.0
    center_y = costmap.height * costmap.resolution / 2.0
    start_state = RobotState(x=center_x, y=center_y, theta=0.0)
    goal_x = center_x + 1.0  # 1m ahead
    goal_y = center_y
    goal_theta = 0.0

    environment.reset(costmap, start_state, goal_x, goal_y, goal_theta)

    print(f"\nScenario:")
    print(f"  Start: ({start_state.x:.2f}, {start_state.y:.2f})")
    print(f"  Goal: ({goal_x:.2f}, {goal_y:.2f})")
    print(f"  Distance: {environment._compute_distance_to_goal(start_state):.2f}m")

    # Test 1: Initialize population
    print("\n--- Test 1: Initialize population ---")
    population = ga.initialize_population()
    print(f"Population size: {len(population)}")
    print(f"First chromosome valid: {population[0].is_valid()}")
    print(f"Genes shape: {population[0].genes.shape}")

    # Test 2: Single generation
    print("\n--- Test 2: Single generation evolution ---")
    next_pop = ga.evolve_generation(population, environment, num_workers=2)
    print(f"Next population size: {len(next_pop)}")

    # Compute statistics
    stats = ga.get_statistics(next_pop)
    print(f"Fitness statistics:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std: {stats['std']:.3f}")
    print(f"  Min: {stats['min']:.3f}")
    print(f"  Max: {stats['max']:.3f}")

    # Test 3: Full evolution run
    print("\n--- Test 3: Full evolution run (10 generations) ---")
    best, history = ga.run(environment, num_generations=10, num_workers=2, verbose=True)

    print(f"\nEvolution complete!")
    print(f"Best fitness: {best.fitness:.3f}")
    print(f"Best fitness components:")
    for key, value in best.fitness_components.items():
        print(f"  {key}: {value}")

    print(f"\nFitness improvement:")
    print(f"  Initial best: {history[0]:.3f}")
    print(f"  Final best: {history[-1]:.3f}")
    print(f"  Improvement: {history[-1] - history[0]:.3f}")

    # Test 4: Verify best trajectory
    print("\n--- Test 4: Verify best trajectory ---")
    control_seq = best.get_control_sequence()
    metrics = environment.simulate_control_sequence(control_seq)

    print(f"Best trajectory:")
    print(f"  Final goal distance: {metrics['goal_distance']:.3f}m")
    print(f"  Collision: {metrics['collision']}")
    print(f"  Goal reached: {metrics['goal_reached']}")
    print(f"  Path length: {metrics['path_length']:.3f}m")

    # Test 5: Difficult scenario
    print("\n--- Test 5: Difficult scenario (more obstacles) ---")
    difficult_costmap = generate_random_costmap(
        width=50,
        height=50,
        resolution=0.05,
        num_obstacles=5,
        obstacle_radius_range=(4, 7),
        inflation_radius=0.5,
        inflation_decay=0.8,
        free_radius_center=0.5
    )

    environment.reset(difficult_costmap, start_state, goal_x, goal_y, goal_theta)

    best_difficult, history_difficult = ga.run(
        environment, num_generations=10, num_workers=2, verbose=False
    )

    print(f"Difficult scenario results:")
    print(f"  Best fitness: {best_difficult.fitness:.3f}")
    print(f"  Goal distance: {best_difficult.fitness_components['goal_distance']:.3f}m")
    print(f"  Collision: {best_difficult.fitness_components['collision']}")

    print("\n✓ GeneticAlgorithm tests passed!")
