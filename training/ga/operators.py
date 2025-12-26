"""
Genetic operators for chromosome evolution.

This module provides selection, crossover, and mutation operators
for the genetic algorithm.
"""

import numpy as np
from typing import List
from .chromosome import Chromosome


def tournament_selection(population: List[Chromosome],
                        tournament_size: int = 5) -> Chromosome:
    """
    Select chromosome using tournament selection.

    Randomly sample tournament_size chromosomes and return the best.

    Args:
        population: List of Chromosome objects (must have fitness)
        tournament_size: Number of chromosomes in tournament (default 5)

    Returns:
        winner: Selected chromosome (not a copy)
    """
    if tournament_size > len(population):
        tournament_size = len(population)

    # Random sample without replacement
    tournament_indices = np.random.choice(
        len(population), size=tournament_size, replace=False
    )
    tournament = [population[i] for i in tournament_indices]

    # Return best in tournament
    winner = max(tournament, key=lambda c: c.fitness)
    return winner


def uniform_crossover(parent1: Chromosome, parent2: Chromosome,
                     crossover_rate: float = 0.8) -> Chromosome:
    """
    Perform uniform crossover between two parents.

    Each gene has 50% chance of coming from each parent.
    Crossover happens with probability crossover_rate.

    Args:
        parent1: First parent chromosome
        parent2: Second parent chromosome
        crossover_rate: Probability of crossover (default 0.8)

    Returns:
        offspring: New chromosome (child)
    """
    # Create offspring
    offspring = parent1.copy()

    # Crossover with probability crossover_rate
    if np.random.random() < crossover_rate:
        # Uniform crossover: each gene has 50% chance from each parent
        mask = np.random.random(parent1.genes.shape) < 0.5
        offspring.genes[mask] = parent2.genes[mask]

    # Reset fitness (needs re-evaluation)
    offspring.fitness = -np.inf
    offspring.fitness_components = {}

    return offspring


def single_point_crossover(parent1: Chromosome, parent2: Chromosome,
                           crossover_rate: float = 0.8) -> Chromosome:
    """
    Perform single-point crossover between two parents.

    Select random timestep; take genes before from parent1, after from parent2.

    Args:
        parent1: First parent chromosome
        parent2: Second parent chromosome
        crossover_rate: Probability of crossover (default 0.8)

    Returns:
        offspring: New chromosome (child)
    """
    offspring = parent1.copy()

    if np.random.random() < crossover_rate:
        # Random crossover point
        crossover_point = np.random.randint(1, parent1.num_steps)

        # Take genes after crossover point from parent2
        offspring.genes[crossover_point:] = parent2.genes[crossover_point:]

    offspring.fitness = -np.inf
    offspring.fitness_components = {}

    return offspring


def gaussian_mutation(chromosome: Chromosome, mutation_rate: float = 0.1,
                     mutation_strength: float = 0.1) -> None:
    """
    Apply Gaussian mutation to chromosome in-place.

    Each gene is mutated with probability mutation_rate by adding
    Gaussian noise scaled by mutation_strength.

    Args:
        chromosome: Chromosome to mutate (modified in-place)
        mutation_rate: Probability of mutating each gene (default 0.1)
        mutation_strength: Standard deviation of Gaussian noise (default 0.1)
    """
    # Mutation mask
    mutation_mask = np.random.random(chromosome.genes.shape) < mutation_rate

    # Apply Gaussian noise
    noise = np.random.randn(*chromosome.genes.shape) * mutation_strength
    chromosome.genes[mutation_mask] += noise[mutation_mask]

    # Clamp to valid range
    chromosome.clamp()

    # Reset fitness
    chromosome.fitness = -np.inf
    chromosome.fitness_components = {}


def uniform_mutation(chromosome: Chromosome, mutation_rate: float = 0.1) -> None:
    """
    Apply uniform mutation to chromosome in-place.

    Each gene is mutated with probability mutation_rate by replacing
    with a random value within valid range.

    Args:
        chromosome: Chromosome to mutate (modified in-place)
        mutation_rate: Probability of mutating each gene (default 0.1)
    """
    mutation_mask = np.random.random(chromosome.genes.shape) < mutation_rate

    # For each DOF, generate random values within limits
    if np.any(mutation_mask[:, 0]):
        chromosome.genes[mutation_mask[:, 0], 0] = np.random.uniform(
            chromosome.velocity_limits['min_v_x'],
            chromosome.velocity_limits['max_v_x'],
            size=np.sum(mutation_mask[:, 0])
        )

    if np.any(mutation_mask[:, 1]):
        chromosome.genes[mutation_mask[:, 1], 1] = np.random.uniform(
            chromosome.velocity_limits['min_v_y'],
            chromosome.velocity_limits['max_v_y'],
            size=np.sum(mutation_mask[:, 1])
        )

    if np.any(mutation_mask[:, 2]):
        chromosome.genes[mutation_mask[:, 2], 2] = np.random.uniform(
            chromosome.velocity_limits['min_omega'],
            chromosome.velocity_limits['max_omega'],
            size=np.sum(mutation_mask[:, 2])
        )

    chromosome.fitness = -np.inf
    chromosome.fitness_components = {}


def elitism_selection(population: List[Chromosome], elite_size: int) -> List[Chromosome]:
    """
    Select top N chromosomes by fitness (elitism).

    Args:
        population: List of Chromosome objects (must have fitness)
        elite_size: Number of elites to select

    Returns:
        elites: List of elite chromosomes (copies)
    """
    # Sort by fitness (descending)
    sorted_population = sorted(population, key=lambda c: c.fitness, reverse=True)

    # Return top N (as copies)
    elites = [chrom.copy() for chrom in sorted_population[:elite_size]]

    return elites


def roulette_wheel_selection(population: List[Chromosome]) -> Chromosome:
    """
    Select chromosome using roulette wheel (fitness proportionate) selection.

    Handles negative fitness by shifting to positive range.

    Args:
        population: List of Chromosome objects (must have fitness)

    Returns:
        selected: Selected chromosome (not a copy)
    """
    # Get fitness values
    fitnesses = np.array([chrom.fitness for chrom in population])

    # Shift to positive (add minimum + 1 if negative)
    min_fitness = fitnesses.min()
    if min_fitness < 0:
        fitnesses = fitnesses - min_fitness + 1.0

    # Compute selection probabilities
    total_fitness = fitnesses.sum()
    if total_fitness == 0:
        # All equal, use uniform
        return np.random.choice(population)

    probabilities = fitnesses / total_fitness

    # Select
    selected_index = np.random.choice(len(population), p=probabilities)
    return population[selected_index]


if __name__ == "__main__":
    # Test genetic operators
    print("Testing genetic operators...")

    # Create velocity limits
    velocity_limits = {
        'max_v_x': 1.0,
        'min_v_x': -0.5,
        'max_v_y': 0.5,
        'min_v_y': -0.5,
        'max_omega': 1.0,
        'min_omega': -1.0
    }

    num_steps = 20

    # Test 1: Tournament selection
    print("\n--- Test 1: Tournament selection ---")
    population = []
    for i in range(10):
        chrom = Chromosome(num_steps, velocity_limits)
        chrom.randomize()
        chrom.fitness = np.random.randn()  # Random fitness
        population.append(chrom)

    print(f"Population fitnesses: {[f'{c.fitness:.2f}' for c in population]}")

    winner = tournament_selection(population, tournament_size=5)
    print(f"Tournament winner fitness: {winner.fitness:.2f}")

    # Test 2: Uniform crossover
    print("\n--- Test 2: Uniform crossover ---")
    parent1 = Chromosome(num_steps, velocity_limits)
    parent2 = Chromosome(num_steps, velocity_limits)
    parent1.genes.fill(1.0)
    parent2.genes.fill(-0.5)
    parent1.fitness = 10.0
    parent2.fitness = 20.0

    offspring = uniform_crossover(parent1, parent2, crossover_rate=1.0)
    print(f"Parent1 genes (sample): {parent1.genes[0]}")
    print(f"Parent2 genes (sample): {parent2.genes[0]}")
    print(f"Offspring genes (sample): {offspring.genes[0]}")
    print(f"Offspring has mixed genes: {np.any(offspring.genes == 1.0) and np.any(offspring.genes == -0.5)}")

    # Test 3: Single-point crossover
    print("\n--- Test 3: Single-point crossover ---")
    offspring_sp = single_point_crossover(parent1, parent2, crossover_rate=1.0)
    print(f"Offspring genes[0]: {offspring_sp.genes[0]}")
    print(f"Offspring genes[-1]: {offspring_sp.genes[-1]}")

    # Test 4: Gaussian mutation
    print("\n--- Test 4: Gaussian mutation ---")
    chrom = Chromosome(num_steps, velocity_limits)
    chrom.genes.fill(0.0)
    print(f"Before mutation: genes[0] = {chrom.genes[0]}")
    gaussian_mutation(chrom, mutation_rate=0.3, mutation_strength=0.1)
    print(f"After mutation: genes[0] = {chrom.genes[0]}")
    print(f"Genes changed: {np.any(chrom.genes != 0.0)}")
    print(f"Is valid: {chrom.is_valid()}")

    # Test 5: Uniform mutation
    print("\n--- Test 5: Uniform mutation ---")
    chrom2 = Chromosome(num_steps, velocity_limits)
    chrom2.genes.fill(0.0)
    print(f"Before mutation: genes[0] = {chrom2.genes[0]}")
    uniform_mutation(chrom2, mutation_rate=0.3)
    print(f"After mutation: genes[0] = {chrom2.genes[0]}")
    print(f"Is valid: {chrom2.is_valid()}")

    # Test 6: Elitism selection
    print("\n--- Test 6: Elitism selection ---")
    population_sorted = sorted(population, key=lambda c: c.fitness, reverse=True)
    print(f"Top 3 fitnesses: {[f'{c.fitness:.2f}' for c in population_sorted[:3]]}")

    elites = elitism_selection(population, elite_size=3)
    print(f"Elite fitnesses: {[f'{c.fitness:.2f}' for c in elites]}")
    print(f"Elites are copies: {elites[0] is not population_sorted[0]}")

    # Test 7: Roulette wheel selection
    print("\n--- Test 7: Roulette wheel selection ---")
    # Create population with known fitness distribution
    test_pop = []
    for i in range(5):
        chrom = Chromosome(num_steps, velocity_limits)
        chrom.fitness = float(i)  # 0, 1, 2, 3, 4
        test_pop.append(chrom)

    print(f"Population fitnesses: {[c.fitness for c in test_pop]}")

    # Sample multiple times to check distribution
    selections = []
    for _ in range(100):
        selected = roulette_wheel_selection(test_pop)
        selections.append(selected.fitness)

    print(f"Selection counts: {np.bincount([int(f) for f in selections])}")
    print(f"Higher fitness selected more often: {selections.count(4.0) > selections.count(0.0)}")

    # Test 8: Negative fitness handling
    print("\n--- Test 8: Negative fitness handling ---")
    neg_pop = []
    for i in range(5):
        chrom = Chromosome(num_steps, velocity_limits)
        chrom.fitness = float(i) - 2.0  # -2, -1, 0, 1, 2
        neg_pop.append(chrom)

    print(f"Population fitnesses (negative): {[c.fitness for c in neg_pop]}")
    selected = roulette_wheel_selection(neg_pop)
    print(f"Selected fitness: {selected.fitness}")

    print("\n✓ Genetic operators tests passed!")
