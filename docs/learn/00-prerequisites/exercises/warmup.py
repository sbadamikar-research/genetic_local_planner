#!/usr/bin/env python3
"""Python warm-up exercises for Module 00."""

import numpy as np

def fitness_function(x):
    """
    Compute fitness for a 1D optimization problem.
    Goal: Find x that maximizes f(x) = -(x - 5)^2 + 10

    Args:
        x: float, candidate solution

    Returns:
        fitness: float, higher is better
    """
    # TODO: Implement this
    pass

def create_population(size, bounds):
    """
    Create random population within bounds.

    Args:
        size: int, number of individuals
        bounds: tuple (min, max)

    Returns:
        population: np.ndarray of shape (size,)
    """
    # TODO: Implement using np.random.uniform
    pass

def tournament_selection(population, fitnesses, k=3):
    """
    Select best individual from k random candidates.

    Args:
        population: np.ndarray, all individuals
        fitnesses: np.ndarray, fitness values
        k: int, tournament size

    Returns:
        winner: float, selected individual
    """
    # TODO: Implement
    # Hint: Use np.random.choice for random indices
    pass

# Test your implementations
if __name__ == "__main__":
    # Test fitness
    assert fitness_function(5.0) == 10.0, "Fitness at optimum should be 10"
    assert fitness_function(0.0) < 10.0, "Fitness away from optimum should be less"

    # Test population
    pop = create_population(100, (-10, 10))
    assert pop.shape == (100,), "Population shape incorrect"
    assert np.all((pop >= -10) & (pop <= 10)), "Population out of bounds"

    # Test selection
    pop = np.array([1, 2, 3, 4, 5])
    fits = np.array([1, 2, 3, 4, 5])
    winner = tournament_selection(pop, fits, k=3)
    assert winner >= 3, "Tournament should favor high fitness"

    print("✓ All tests passed!")
