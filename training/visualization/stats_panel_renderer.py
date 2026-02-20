"""
Statistics panel rendering for Pygame visualization.

Provides statistics display with text and fitness evolution graph.
"""

import pygame
import numpy as np
from typing import List, Dict


class StatsPanelRenderer:
    """Render statistics panel with text and fitness graph."""

    def __init__(self, panel_width: int = 400, panel_height: int = 800):
        """
        Initialize stats panel renderer.

        Args:
            panel_width: Panel width in pixels
            panel_height: Panel height in pixels
        """
        self.panel_width = panel_width
        self.panel_height = panel_height

        # Initialize fonts (scaled proportionally to panel size)
        pygame.font.init()
        scale_factor = panel_height / 800.0  # Original design was 800px height
        self.font_large = pygame.font.Font(None, int(36 * scale_factor))
        self.font_medium = pygame.font.Font(None, int(28 * scale_factor))
        self.font_small = pygame.font.Font(None, int(22 * scale_factor))

        # Graph area (scaled proportionally)
        self.graph_height = int(200 * scale_factor)
        self.graph_margin = int(20 * scale_factor)

    def render_stats(self, surface: pygame.Surface, stats: Dict,
                    fitness_history: List[float]):
        """
        Render statistics panel showing:
        - Scenario/generation counters
        - Fitness values (best, avg, std)
        - Population diversity
        - Fitness components breakdown
        - Mini fitness evolution graph

        Args:
            surface: Pygame surface (panel_width × panel_height)
            stats: Dict with keys:
                - scenario_id: Scenario number
                - generation: Generation number
                - best_fitness: Best fitness in population
                - avg_fitness: Average fitness
                - std_fitness: Std dev of fitness
                - diversity: Population diversity metric
                - fitness_components: Dict with goal_distance, collision, etc.
            fitness_history: List of fitness values over time (for graph)
        """
        # Fill background
        surface.fill((30, 30, 35))

        # Colors
        text_color = (220, 220, 220)
        highlight_color = (100, 200, 255)

        # Scale factor for layout (based on panel height)
        scale = self.panel_height / 800.0

        y_offset = int(20 * scale)
        margin = int(20 * scale)
        indent = int(30 * scale)

        # Title
        title = self.font_large.render("GA Training", True, highlight_color)
        surface.blit(title, (margin, y_offset))
        y_offset += int(50 * scale)

        # Scenario info
        scenario_text = self.font_medium.render(
            f"Scenario: {stats['scenario_id']}", True, text_color
        )
        surface.blit(scenario_text, (margin, y_offset))
        y_offset += int(35 * scale)

        generation_text = self.font_medium.render(
            f"Generation: {stats['generation']}", True, text_color
        )
        surface.blit(generation_text, (margin, y_offset))
        y_offset += int(45 * scale)

        # Fitness section
        section_title = self.font_medium.render("Fitness", True, highlight_color)
        surface.blit(section_title, (margin, y_offset))
        y_offset += int(35 * scale)

        best_text = self.font_small.render(
            f"Best:  {stats['best_fitness']:.3f}", True, text_color
        )
        surface.blit(best_text, (indent, y_offset))
        y_offset += int(28 * scale)

        avg_text = self.font_small.render(
            f"Avg:   {stats['avg_fitness']:.3f}", True, text_color
        )
        surface.blit(avg_text, (indent, y_offset))
        y_offset += int(28 * scale)

        std_text = self.font_small.render(
            f"Std:   {stats['std_fitness']:.3f}", True, text_color
        )
        surface.blit(std_text, (indent, y_offset))
        y_offset += int(35 * scale)

        # Diversity
        diversity_text = self.font_small.render(
            f"Diversity: {stats['diversity']:.3f}", True, text_color
        )
        surface.blit(diversity_text, (indent, y_offset))
        y_offset += int(45 * scale)

        # Fitness components
        section_title = self.font_medium.render("Components", True, highlight_color)
        surface.blit(section_title, (margin, y_offset))
        y_offset += int(35 * scale)

        if 'fitness_components' in stats and stats['fitness_components']:
            components = stats['fitness_components']

            # Goal distance
            if 'goal_distance' in components:
                goal_text = self.font_small.render(
                    f"Goal: {components['goal_distance']:.3f}m", True, text_color
                )
                surface.blit(goal_text, (indent, y_offset))
                y_offset += int(28 * scale)

            # Collision
            if 'collision' in components:
                collision_text = self.font_small.render(
                    f"Collision: {components['collision']}", True, text_color
                )
                surface.blit(collision_text, (indent, y_offset))
                y_offset += int(28 * scale)

            # Goal reached
            if 'goal_reached' in components:
                reached = "Yes" if components['goal_reached'] else "No"
                reached_text = self.font_small.render(
                    f"Reached: {reached}", True, text_color
                )
                surface.blit(reached_text, (indent, y_offset))
                y_offset += int(28 * scale)

        y_offset += int(20 * scale)

        # Fitness graph
        if fitness_history and len(fitness_history) > 1:
            self._render_fitness_graph(surface, fitness_history, y_offset)

    def _render_fitness_graph(self, surface: pygame.Surface,
                             fitness_history: List[float], y_offset: int):
        """
        Render mini fitness evolution graph.

        Args:
            surface: Pygame surface
            fitness_history: List of fitness values
            y_offset: Y position to start graph
        """
        # Graph dimensions
        scale = self.panel_height / 800.0
        graph_x = self.graph_margin
        graph_y = y_offset
        graph_width = self.panel_width - 2 * self.graph_margin
        graph_height = self.graph_height

        # Draw graph background
        graph_rect = pygame.Rect(graph_x, graph_y, graph_width, graph_height)
        pygame.draw.rect(surface, (20, 20, 25), graph_rect)
        pygame.draw.rect(surface, (100, 100, 110), graph_rect, 1)

        # Title
        title = self.font_small.render("Fitness History", True, (220, 220, 220))
        surface.blit(title, (graph_x + int(5 * scale), graph_y + int(5 * scale)))

        # Compute value range
        fitness_array = np.array(fitness_history)
        fitness_min = np.min(fitness_array)
        fitness_max = np.max(fitness_array)

        # Handle flat fitness
        if fitness_max - fitness_min < 1e-6:
            fitness_min -= 0.1
            fitness_max += 0.1

        # Map fitness to graph coordinates
        num_points = len(fitness_history)
        points = []

        for i, fitness in enumerate(fitness_history):
            # X: evenly spaced across graph width
            x = graph_x + int((i / (num_points - 1)) * graph_width)

            # Y: map fitness to graph height (inverted, high fitness at top)
            normalized = (fitness - fitness_min) / (fitness_max - fitness_min)
            y = graph_y + graph_height - int(normalized * (graph_height - int(30 * scale)))

            points.append((x, y))

        # Draw fitness curve
        if len(points) >= 2:
            pygame.draw.lines(surface, (0, 255, 0), False, points, max(2, int(2 * scale)))

        # Draw min/max labels
        min_label = self.font_small.render(f"{fitness_min:.2f}", True, (180, 180, 180))
        max_label = self.font_small.render(f"{fitness_max:.2f}", True, (180, 180, 180))

        surface.blit(min_label, (graph_x + int(5 * scale), graph_y + graph_height - int(20 * scale)))
        surface.blit(max_label, (graph_x + int(5 * scale), graph_y + int(25 * scale)))
