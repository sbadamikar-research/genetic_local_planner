"""
Costmap rendering for Pygame visualization.

Provides efficient costmap rendering to Pygame surfaces.
"""

import pygame
import numpy as np
from typing import Tuple
from .color_utils import cost_to_color


class CostmapRenderer:
    """Efficient costmap rendering to Pygame surface."""

    def __init__(self, costmap_size: int = 800):
        """
        Initialize costmap renderer.

        Args:
            costmap_size: Target render size in pixels (square)
        """
        self.costmap_size = costmap_size
        self.color_lut = self._build_color_lookup_table()

    def _build_color_lookup_table(self) -> np.ndarray:
        """
        Build color lookup table for all possible costmap values.

        Returns:
            color_lut: Array of shape (255, 3) with RGB values
        """
        lut = np.zeros((255, 3), dtype=np.uint8)
        for cost in range(255):
            lut[cost] = cost_to_color(cost)
        return lut

    def render_costmap(self, costmap, surface: pygame.Surface):
        """
        Render costmap to surface using vectorized operations.

        Uses pygame.surfarray for efficient numpy → surface transfer.

        Args:
            costmap: Costmap object with .data attribute (NxN uint8 array)
            surface: Pygame surface to render to
        """
        # Get costmap data
        costmap_data = costmap.data.astype(np.uint8)

        # Map costs to colors using lookup table
        # color_image shape: (N, N, 3)
        color_image = self.color_lut[costmap_data]

        # Upscale to render size
        scale_factor = self.costmap_size // costmap_data.shape[0]

        # Repeat pixels for upscaling
        # Use numpy repeat for efficient upscaling
        upscaled = np.repeat(np.repeat(color_image, scale_factor, axis=0),
                            scale_factor, axis=1)

        # Transpose for pygame (height, width, channels) → (width, height, channels)
        # Pygame expects (width, height) but numpy uses (height, width)
        upscaled_transposed = np.transpose(upscaled, (1, 0, 2))

        # Copy to surface using surfarray
        pygame.surfarray.blit_array(surface, upscaled_transposed)

    def world_to_screen(self, wx: float, wy: float, costmap) -> Tuple[int, int]:
        """
        Transform world coordinates → grid → screen pixels.

        Args:
            wx: World x coordinate (meters)
            wy: World y coordinate (meters)
            costmap: Costmap object with resolution attribute

        Returns:
            screen_x, screen_y: Screen pixel coordinates
        """
        # World → grid
        grid_x = int(wx / costmap.resolution)
        grid_y = int(wy / costmap.resolution)

        # Grid → screen (with upscaling)
        scale_factor = self.costmap_size // costmap.width
        screen_x = grid_x * scale_factor
        screen_y = grid_y * scale_factor

        # Clamp to screen bounds
        screen_x = np.clip(screen_x, 0, self.costmap_size - 1)
        screen_y = np.clip(screen_y, 0, self.costmap_size - 1)

        return (screen_x, screen_y)
