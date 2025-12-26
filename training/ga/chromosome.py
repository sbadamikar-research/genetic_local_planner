"""
Chromosome encoding for control sequences.

This module provides the Chromosome class which represents a candidate
solution (control sequence) in the genetic algorithm population.
"""

import numpy as np
from typing import Dict


class Chromosome:
    """
    Represents a control sequence as a GA chromosome.

    Encoding: Direct encoding of control sequence
    - num_steps timesteps (typically 20)
    - 3 degrees of freedom per timestep: [v_x, v_y, omega]
    - Total genes: num_steps × 3 = 60

    Attributes:
        genes: np.ndarray (num_steps, 3), control values
        velocity_limits: Dict with max/min for v_x, v_y, omega
        fitness: float, fitness value (higher is better, -inf initially)
        fitness_components: Dict, breakdown of fitness terms
    """

    def __init__(self, num_steps: int, velocity_limits: Dict[str, float]):
        """
        Initialize chromosome with zero genes.

        Args:
            num_steps: Number of control timesteps
            velocity_limits: Dict with keys:
                - max_v_x, min_v_x: Forward/backward velocity limits (m/s)
                - max_v_y, min_v_y: Lateral velocity limits (m/s)
                - max_omega, min_omega: Angular velocity limits (rad/s)
        """
        self.num_steps = num_steps
        self.velocity_limits = velocity_limits
        self.genes = np.zeros((num_steps, 3), dtype=np.float32)
        self.fitness = -np.inf
        self.fitness_components = {}

    def randomize(self):
        """
        Initialize with random valid velocities.

        Samples uniformly within velocity limits for each DOF.
        """
        for i in range(self.num_steps):
            # Random v_x
            self.genes[i, 0] = np.random.uniform(
                self.velocity_limits['min_v_x'],
                self.velocity_limits['max_v_x']
            )

            # Random v_y
            self.genes[i, 1] = np.random.uniform(
                self.velocity_limits['min_v_y'],
                self.velocity_limits['max_v_y']
            )

            # Random omega
            self.genes[i, 2] = np.random.uniform(
                self.velocity_limits['min_omega'],
                self.velocity_limits['max_omega']
            )

    def clamp(self):
        """
        Clamp all velocities to valid range.

        Ensures genes respect velocity limits after crossover/mutation.
        """
        self.genes[:, 0] = np.clip(
            self.genes[:, 0],
            self.velocity_limits['min_v_x'],
            self.velocity_limits['max_v_x']
        )

        self.genes[:, 1] = np.clip(
            self.genes[:, 1],
            self.velocity_limits['min_v_y'],
            self.velocity_limits['max_v_y']
        )

        self.genes[:, 2] = np.clip(
            self.genes[:, 2],
            self.velocity_limits['min_omega'],
            self.velocity_limits['max_omega']
        )

    def copy(self):
        """
        Create deep copy of chromosome.

        Returns:
            chromosome_copy: New Chromosome instance with copied genes and fitness
        """
        new_chromosome = Chromosome(self.num_steps, self.velocity_limits)
        new_chromosome.genes = self.genes.copy()
        new_chromosome.fitness = self.fitness
        new_chromosome.fitness_components = self.fitness_components.copy()
        return new_chromosome

    def get_control_sequence(self) -> np.ndarray:
        """
        Get control sequence as numpy array.

        Returns:
            control_sequence: np.ndarray (num_steps, 3), [[v_x, v_y, omega], ...]
        """
        return self.genes.copy()

    def set_control_sequence(self, control_sequence: np.ndarray):
        """
        Set genes from control sequence.

        Args:
            control_sequence: np.ndarray (num_steps, 3)
        """
        if control_sequence.shape != (self.num_steps, 3):
            raise ValueError(
                f"Control sequence shape {control_sequence.shape} "
                f"does not match expected ({self.num_steps}, 3)"
            )
        self.genes = control_sequence.astype(np.float32)
        self.clamp()

    def is_valid(self) -> bool:
        """
        Check if chromosome has valid genes.

        Returns:
            valid: True if all genes are within limits and finite
        """
        # Check for NaN or inf
        if not np.all(np.isfinite(self.genes)):
            return False

        # Check velocity limits
        if np.any(self.genes[:, 0] < self.velocity_limits['min_v_x']) or \
           np.any(self.genes[:, 0] > self.velocity_limits['max_v_x']):
            return False

        if np.any(self.genes[:, 1] < self.velocity_limits['min_v_y']) or \
           np.any(self.genes[:, 1] > self.velocity_limits['max_v_y']):
            return False

        if np.any(self.genes[:, 2] < self.velocity_limits['min_omega']) or \
           np.any(self.genes[:, 2] > self.velocity_limits['max_omega']):
            return False

        return True

    def __repr__(self):
        """String representation for debugging."""
        return (f"Chromosome(num_steps={self.num_steps}, "
                f"fitness={self.fitness:.4f}, "
                f"valid={self.is_valid()})")

    def __lt__(self, other):
        """Less-than comparison for sorting by fitness (higher is better)."""
        return self.fitness < other.fitness

    def __le__(self, other):
        """Less-than-or-equal comparison."""
        return self.fitness <= other.fitness

    def __gt__(self, other):
        """Greater-than comparison."""
        return self.fitness > other.fitness

    def __ge__(self, other):
        """Greater-than-or-equal comparison."""
        return self.fitness >= other.fitness


if __name__ == "__main__":
    # Test chromosome
    print("Testing Chromosome...")

    # Create velocity limits
    velocity_limits = {
        'max_v_x': 1.0,
        'min_v_x': -0.5,
        'max_v_y': 0.5,
        'min_v_y': -0.5,
        'max_omega': 1.0,
        'min_omega': -1.0
    }

    # Create chromosome
    num_steps = 20
    chromosome = Chromosome(num_steps, velocity_limits)
    print(f"Created chromosome: {chromosome}")
    print(f"Genes shape: {chromosome.genes.shape}")
    print(f"Initial fitness: {chromosome.fitness}")

    # Test randomization
    print("\nTesting randomization...")
    chromosome.randomize()
    print(f"After randomization - genes sample:")
    print(chromosome.genes[:3])
    print(f"Is valid: {chromosome.is_valid()}")

    # Test velocity limits
    print("\nTesting velocity limits...")
    print(f"v_x range: [{chromosome.genes[:, 0].min():.3f}, {chromosome.genes[:, 0].max():.3f}]")
    print(f"v_y range: [{chromosome.genes[:, 1].min():.3f}, {chromosome.genes[:, 1].max():.3f}]")
    print(f"omega range: [{chromosome.genes[:, 2].min():.3f}, {chromosome.genes[:, 2].max():.3f}]")

    # Test clamping
    print("\nTesting clamping...")
    chromosome.genes[0, 0] = 10.0  # Invalid v_x
    print(f"Before clamp: v_x[0] = {chromosome.genes[0, 0]:.3f}")
    chromosome.clamp()
    print(f"After clamp: v_x[0] = {chromosome.genes[0, 0]:.3f}")
    print(f"Is valid: {chromosome.is_valid()}")

    # Test copy
    print("\nTesting copy...")
    chromosome.fitness = 42.0
    chromosome.fitness_components = {'goal_distance': 1.0}
    copy = chromosome.copy()
    print(f"Original fitness: {chromosome.fitness}")
    print(f"Copy fitness: {copy.fitness}")
    copy.fitness = 100.0
    print(f"After modifying copy - original fitness: {chromosome.fitness}")
    print(f"After modifying copy - copy fitness: {copy.fitness}")

    # Test control sequence accessors
    print("\nTesting control sequence accessors...")
    control_seq = chromosome.get_control_sequence()
    print(f"Control sequence shape: {control_seq.shape}")

    new_seq = np.random.randn(num_steps, 3)
    chromosome.set_control_sequence(new_seq)
    print(f"After setting new sequence - is valid: {chromosome.is_valid()}")

    # Test comparison operators
    print("\nTesting comparison operators...")
    chrom1 = Chromosome(num_steps, velocity_limits)
    chrom2 = Chromosome(num_steps, velocity_limits)
    chrom1.fitness = 10.0
    chrom2.fitness = 20.0
    print(f"chrom1.fitness = {chrom1.fitness}, chrom2.fitness = {chrom2.fitness}")
    print(f"chrom1 < chrom2: {chrom1 < chrom2}")
    print(f"chrom1 > chrom2: {chrom1 > chrom2}")

    # Test sorting
    print("\nTesting sorting...")
    population = [chrom1, chrom2, chromosome]
    population.sort(reverse=True)  # Sort by fitness descending
    print("Sorted population (descending fitness):")
    for i, chrom in enumerate(population):
        print(f"  {i+1}. fitness = {chrom.fitness:.2f}")

    # Test differential drive robot (v_y = 0)
    print("\nTesting differential drive configuration...")
    diff_drive_limits = {
        'max_v_x': 1.0,
        'min_v_x': -0.5,
        'max_v_y': 0.0,  # Differential drive has no lateral motion
        'min_v_y': 0.0,
        'max_omega': 1.0,
        'min_omega': -1.0
    }
    diff_chrom = Chromosome(num_steps, diff_drive_limits)
    diff_chrom.randomize()
    print(f"Differential drive v_y range: [{diff_chrom.genes[:, 1].min():.3f}, {diff_chrom.genes[:, 1].max():.3f}]")
    print(f"Is valid: {diff_chrom.is_valid()}")

    print("\n✓ Chromosome tests passed!")
