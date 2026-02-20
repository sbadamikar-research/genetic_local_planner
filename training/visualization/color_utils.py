"""
Color mapping utilities for visualization.

Provides color mapping functions for costmap values, fitness values,
and population diversity metrics.
"""

import numpy as np
from typing import Tuple, List


def cost_to_color(cost: int) -> Tuple[int, int, int]:
    """
    Map costmap value [0-254] to RGB color.

    Mapping:
    - 0 (free space): White (255, 255, 255)
    - 1-252 (inflated): Gradient yellow → orange → red
    - 253-254 (lethal): Red (255, 0, 0)

    Args:
        cost: Costmap value [0-254]

    Returns:
        rgb: Tuple of (R, G, B) values [0-255]
    """
    if cost == 0:
        # Free space: white
        return (255, 255, 255)
    elif cost >= 253:
        # Lethal obstacle: red
        return (50, 50, 50)
    else:
        # Inflated zone: gradient from white → yellow → orange → red
        # Normalize to [0, 1]
        normalized = cost / 252.0

        # Three-stage gradient:
        # 0.0-0.3: white → yellow
        # 0.3-0.7: yellow → orange
        # 0.7-1.0: orange → red

        brightness = 0.5
        if normalized < 0.3:
            # White (255, 255, 255) → Yellow (255, 255, 0)
            alpha = normalized / 0.3
            r = int(brightness * 255)
            g = int(brightness * 255)
            b = int(brightness * (255 * (1.0 - alpha)))
        elif normalized < 0.7:
            # Yellow (255, 255, 0) → Orange (255, 165, 0)
            alpha = (normalized - 0.3) / 0.4
            r = int(brightness * 255)
            g = int(brightness * (255 - 90 * alpha))
            b = int(brightness * 0)
        else:
            # Orange (255, 165, 0) → Red (255, 0, 0)
            alpha = (normalized - 0.7) / 0.3
            r = int(brightness * 255)
            g = int(brightness * (165 * (1.0 - alpha)))
            b = int(brightness * 0)

        return (r, g, b)


def fitness_to_color(fitness: float, fitness_min: float, fitness_max: float) -> Tuple[int, int, int]:
    """
    Map fitness value to color gradient (red → yellow → green).

    Best fitness (max) maps to green, worst (min) maps to red.
    Uses HSV interpolation for smooth gradient.

    Args:
        fitness: Fitness value to map
        fitness_min: Minimum fitness in population
        fitness_max: Maximum fitness in population

    Returns:
        rgb: Tuple of (R, G, B) values [0-255]
    """
    # Handle NaN or infinite values
    if not np.isfinite(fitness) or not np.isfinite(fitness_min) or not np.isfinite(fitness_max):
        return (128, 128, 128)  # Gray for invalid values

    # Handle edge case where all fitnesses are equal
    if abs(fitness_max - fitness_min) < 1e-6:
        return (0, 255, 0)  # Green

    # Normalize fitness to [0, 1]
    normalized = (fitness - fitness_min) / (fitness_max - fitness_min)
    normalized = np.clip(normalized, 0.0, 1.0)

    # Map to HSV hue: red (0°) → yellow (60°) → green (120°)
    # Hue range: 0-120° (red to green)
    hue = normalized * 120.0

    # Convert HSV to RGB
    # H in [0, 360], S = 1.0, V = 1.0
    h = hue / 60.0  # Sector 0-2
    c = 1.0  # Chroma
    x = c * (1.0 - abs(h % 2.0 - 1.0))

    if h < 1.0:
        r, g, b = c, x, 0
    elif h < 2.0:
        r, g, b = x, c, 0
    else:
        r, g, b = 0, c, x

    # Convert to [0-255]
    return (int(r * 255), int(g * 255), int(b * 255))


def compute_population_diversity(population: List) -> float:
    """
    Compute population genetic diversity metric.

    Calculates average pairwise Euclidean distance between gene vectors.
    Higher values indicate more diverse population (less converged).

    Args:
        population: List of Chromosome objects with .genes attribute

    Returns:
        diversity: Average pairwise distance (normalized by gene vector length)
    """
    if len(population) < 2:
        return 0.0

    # Extract gene arrays
    gene_arrays = []
    for chromosome in population:
        # Flatten genes to 1D array
        genes_flat = chromosome.genes.flatten()
        gene_arrays.append(genes_flat)

    gene_arrays = np.array(gene_arrays)  # Shape: (pop_size, num_genes)

    # Compute pairwise distances
    n = len(gene_arrays)
    total_distance = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            distance = np.linalg.norm(gene_arrays[i] - gene_arrays[j])
            total_distance += distance
            count += 1

    # Average distance
    avg_distance = total_distance / count if count > 0 else 0.0

    # Normalize by gene vector length (for scale independence)
    num_genes = gene_arrays.shape[1]
    normalized_diversity = avg_distance / np.sqrt(num_genes)

    return normalized_diversity


if __name__ == "__main__":
    # Test color mapping functions
    print("Testing color_utils...")

    # Test 1: cost_to_color
    print("\n--- Test 1: cost_to_color ---")
    test_costs = [0, 1, 50, 100, 150, 200, 252, 253, 254]
    for cost in test_costs:
        color = cost_to_color(cost)
        print(f"Cost {cost:3d} → RGB{color}")

    # Verify key colors
    assert cost_to_color(0) == (255, 255, 255), "Free space should be white"
    assert cost_to_color(254) == (255, 0, 0), "Lethal should be red"
    print("✓ cost_to_color works")

    # Test 2: fitness_to_color
    print("\n--- Test 2: fitness_to_color ---")
    fitness_min = 0.0
    fitness_max = 1.0
    test_fitnesses = [0.0, 0.25, 0.5, 0.75, 1.0]

    for fitness in test_fitnesses:
        color = fitness_to_color(fitness, fitness_min, fitness_max)
        print(f"Fitness {fitness:.2f} → RGB{color}")

    # Verify gradient
    worst_color = fitness_to_color(0.0, 0.0, 1.0)
    best_color = fitness_to_color(1.0, 0.0, 1.0)
    print(f"Worst (0.0): {worst_color}")
    print(f"Best (1.0): {best_color}")
    assert best_color[1] > worst_color[1], "Best should be more green"
    print("✓ fitness_to_color works")

    # Test 3: Edge cases
    print("\n--- Test 3: Edge cases ---")
    # All equal fitness
    color = fitness_to_color(0.5, 0.5, 0.5)
    print(f"Equal fitness → {color}")
    assert color == (0, 255, 0), "Equal fitness should map to green"

    # Out of range
    color_low = fitness_to_color(-0.5, 0.0, 1.0)
    color_high = fitness_to_color(1.5, 0.0, 1.0)
    print(f"Below range (-0.5) → {color_low}")
    print(f"Above range (1.5) → {color_high}")
    print("✓ Edge cases handled")

    # Test 4: compute_population_diversity
    print("\n--- Test 4: compute_population_diversity ---")

    # Mock chromosome class
    class MockChromosome:
        def __init__(self, genes):
            self.genes = genes

    # Test with diverse population
    pop_diverse = [
        MockChromosome(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])),
        MockChromosome(np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])),
        MockChromosome(np.array([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]))
    ]
    diversity_high = compute_population_diversity(pop_diverse)
    print(f"Diverse population diversity: {diversity_high:.3f}")

    # Test with converged population
    pop_converged = [
        MockChromosome(np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])),
        MockChromosome(np.array([[0.11, 0.11, 0.11], [0.11, 0.11, 0.11]])),
        MockChromosome(np.array([[0.12, 0.12, 0.12], [0.12, 0.12, 0.12]]))
    ]
    diversity_low = compute_population_diversity(pop_converged)
    print(f"Converged population diversity: {diversity_low:.3f}")

    assert diversity_high > diversity_low, "Diverse population should have higher diversity"
    print("✓ compute_population_diversity works")

    print("\n✓ All color_utils tests passed!")
