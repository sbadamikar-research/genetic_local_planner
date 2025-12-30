#!/usr/bin/env python3
"""Solutions for Python warm-up exercises."""

import numpy as np

def fitness_function(x):
    """Compute fitness: f(x) = -(x - 5)^2 + 10"""
    return -(x - 5)**2 + 10

def create_population(size, bounds):
    """Create random population within bounds."""
    min_val, max_val = bounds
    return np.random.uniform(min_val, max_val, size=size)

def tournament_selection(population, fitnesses, k=3):
    """Tournament selection: pick best from k random individuals."""
    indices = np.random.choice(len(population), size=k, replace=False)
    tournament_fits = fitnesses[indices]
    winner_idx = indices[np.argmax(tournament_fits)]
    return population[winner_idx]

# Test
if __name__ == "__main__":
    assert fitness_function(5.0) == 10.0
    assert fitness_function(0.0) < 10.0
    
    pop = create_population(100, (-10, 10))
    assert pop.shape == (100,)
    assert np.all((pop >= -10) & (pop <= 10))
    
    pop = np.array([1, 2, 3, 4, 5])
    fits = np.array([1, 2, 3, 4, 5])
    winner = tournament_selection(pop, fits, k=3)
    assert winner >= 3
    
    print("✓ All tests passed!")
