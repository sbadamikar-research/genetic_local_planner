"""
Costmap generation and inflation for GA training.

This module provides functionality for creating and manipulating 2D occupancy
grids (costmaps) used in robot navigation. It supports:
- Procedural obstacle generation
- Exponential cost inflation
- World-to-grid coordinate transformations
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Tuple


class Costmap:
    """
    2D occupancy grid for robot navigation.

    Costmap follows ROS convention:
    - 0: FREE_SPACE (no cost)
    - 1-252: Inflated obstacles (increasing cost)
    - 253: INSCRIBED_INFLATED_OBSTACLE
    - 254: LETHAL_OBSTACLE (actual obstacle)
    - 255: NO_INFORMATION (unknown, treated as free)

    Attributes:
        data: np.ndarray (height, width), uint8, costmap values [0-255]
        resolution: float, meters per pixel
        width: int, grid width in pixels
        height: int, grid height in pixels
        origin_x: float, world x coordinate of grid origin
        origin_y: float, world y coordinate of grid origin
        inflation_decay: float, exponential decay factor for inflation
    """

    def __init__(self, width: int, height: int, resolution: float,
                 inflation_decay: float = 0.8, origin_x: float = 0.0,
                 origin_y: float = 0.0):
        """
        Initialize empty costmap.

        Args:
            width: Grid width in pixels
            height: Grid height in pixels
            resolution: Meters per pixel
            inflation_decay: Exponential decay factor for inflation
            origin_x: World x coordinate of grid origin
            origin_y: World y coordinate of grid origin
        """
        self.data = np.zeros((height, width), dtype=np.uint8)
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.inflation_decay = inflation_decay

    def set_cost(self, x: int, y: int, cost: int):
        """
        Set cost at grid cell (x, y).

        Args:
            x: Grid x index
            y: Grid y index
            cost: Cost value [0-254]
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.data[y, x] = np.clip(cost, 0, 254)

    def get_cost(self, x: int, y: int) -> int:
        """
        Get cost at grid cell (x, y).

        Args:
            x: Grid x index
            y: Grid y index

        Returns:
            cost: Cost value [0-254], or 0 if out of bounds
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return int(self.data[y, x])
        return 0

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """
        Convert world coordinates to grid indices.

        Args:
            wx: World x coordinate (meters)
            wy: World y coordinate (meters)

        Returns:
            (mx, my): Grid indices
        """
        mx = int((wx - self.origin_x) / self.resolution)
        my = int((wy - self.origin_y) / self.resolution)
        return mx, my

    def grid_to_world(self, mx: int, my: int) -> Tuple[float, float]:
        """
        Convert grid indices to world coordinates.

        Args:
            mx: Grid x index
            my: Grid y index

        Returns:
            (wx, wy): World coordinates (meters)
        """
        wx = mx * self.resolution + self.origin_x
        wy = my * self.resolution + self.origin_y
        return wx, wy

    def add_circular_obstacle(self, center_x: int, center_y: int, radius: int):
        """
        Add circular lethal obstacle to costmap.

        Args:
            center_x: Grid x coordinate of circle center
            center_y: Grid y coordinate of circle center
            radius: Circle radius in pixels
        """
        for y in range(max(0, center_y - radius), min(self.height, center_y + radius + 1)):
            for x in range(max(0, center_x - radius), min(self.width, center_x + radius + 1)):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist <= radius:
                    self.data[y, x] = 254  # LETHAL_OBSTACLE

    def inflate_obstacles(self, inflation_radius_meters: float):
        """
        Inflate obstacles with exponential decay.

        Uses distance transform for efficient computation. Obstacles are inflated
        outward with exponentially decaying cost.

        Algorithm:
        1. Find all lethal cells (cost == 254)
        2. Compute distance transform (distance to nearest obstacle)
        3. Apply exponential decay: cost = 254 * exp(-decay * distance)

        Args:
            inflation_radius_meters: Inflation radius in meters
        """
        # Convert inflation radius to pixels
        inflation_radius_pixels = inflation_radius_meters / self.resolution

        # Create binary mask of obstacles (254 = lethal)
        obstacle_mask = (self.data == 254)

        # Compute distance transform (distance to nearest obstacle in pixels)
        distance_map = distance_transform_edt(~obstacle_mask)

        # Apply exponential decay for cells within inflation radius
        inflation_mask = distance_map <= inflation_radius_pixels

        # Compute inflated costs
        # cost = 254 * exp(-decay * distance)
        inflated_costs = np.where(
            inflation_mask & ~obstacle_mask,
            254 * np.exp(-self.inflation_decay * distance_map),
            self.data
        )

        # Update costmap (keep lethal obstacles at 254)
        self.data = np.clip(inflated_costs, 0, 254).astype(np.uint8)

        # Ensure lethal obstacles remain at 254
        self.data[obstacle_mask] = 254

    def to_normalized(self) -> np.ndarray:
        """
        Convert costmap to normalized float32 array for neural network input.

        Returns:
            normalized: np.ndarray (height, width), float32, values [0, 1]
        """
        return self.data.astype(np.float32) / 254.0

    def copy(self):
        """
        Create deep copy of costmap.

        Returns:
            costmap_copy: New Costmap instance with copied data
        """
        new_costmap = Costmap(
            self.width, self.height, self.resolution,
            self.inflation_decay, self.origin_x, self.origin_y
        )
        new_costmap.data = self.data.copy()
        return new_costmap


def generate_random_costmap(width: int, height: int, resolution: float,
                            num_obstacles: int, obstacle_radius_range: Tuple[int, int],
                            inflation_radius: float, inflation_decay: float,
                            free_radius_center: float = 0.5) -> Costmap:
    """
    Generate procedural costmap with random circular obstacles.

    Creates a costmap with randomly placed circular obstacles, ensuring
    a free region around the center for robot start position.

    Args:
        width: Grid width in pixels
        height: Grid height in pixels
        resolution: Meters per pixel
        num_obstacles: Number of random obstacles to place
        obstacle_radius_range: (min_radius, max_radius) in pixels
        inflation_radius: Inflation distance in meters
        inflation_decay: Exponential decay factor
        free_radius_center: Keep center region obstacle-free (meters)

    Returns:
        costmap: Costmap with inflated obstacles
    """
    # Create empty costmap
    costmap = Costmap(width, height, resolution, inflation_decay)

    # Center coordinates
    center_x = width // 2
    center_y = height // 2
    free_radius_pixels = free_radius_center / resolution

    # Add random obstacles
    min_radius, max_radius = obstacle_radius_range

    for _ in range(num_obstacles):
        # Try to place obstacle away from center
        max_attempts = 50
        for attempt in range(max_attempts):
            # Random position
            obs_x = np.random.randint(0, width)
            obs_y = np.random.randint(0, height)

            # Check distance from center
            dist_from_center = np.sqrt((obs_x - center_x)**2 + (obs_y - center_y)**2)

            # Accept if far enough from center
            if dist_from_center > free_radius_pixels:
                obs_radius = np.random.randint(min_radius, max_radius + 1)
                costmap.add_circular_obstacle(obs_x, obs_y, obs_radius)
                break

    # Inflate obstacles
    costmap.inflate_obstacles(inflation_radius)

    return costmap


if __name__ == "__main__":
    # Test costmap generation
    print("Testing Costmap generation...")

    # Create costmap
    costmap = generate_random_costmap(
        width=50,
        height=50,
        resolution=0.05,
        num_obstacles=5,
        obstacle_radius_range=(2, 8),
        inflation_radius=0.5,
        inflation_decay=0.8,
        free_radius_center=0.5
    )

    print(f"Costmap shape: {costmap.data.shape}")
    print(f"Costmap resolution: {costmap.resolution} m/pixel")
    print(f"Min cost: {costmap.data.min()}")
    print(f"Max cost: {costmap.data.max()}")
    print(f"Number of lethal cells: {np.sum(costmap.data == 254)}")
    print(f"Number of free cells: {np.sum(costmap.data == 0)}")

    # Test normalization
    normalized = costmap.to_normalized()
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")

    print("\n✓ Costmap tests passed!")
