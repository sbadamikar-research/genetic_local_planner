"""
Real-time Pygame visualizer for GA training.

Provides two visualization modes:
1. Scenario mode: Show final best trajectory per scenario
2. Evolution mode: Show population evolution per generation
"""

import pygame
import numpy as np
import time
from typing import List, Dict, Optional
from pathlib import Path

from .costmap_renderer import CostmapRenderer
from .trajectory_renderer import TrajectoryRenderer
from .stats_panel_renderer import StatsPanelRenderer
from .color_utils import compute_population_diversity


class GAVisualizer:
    """
    Real-time Pygame visualizer for GA training.

    Modes:
    - 'off': No visualization (default)
    - 'scenario': Show final best trajectory per scenario
    - 'evolution': Show population evolution per generation
    """

    def __init__(self, config: dict, mode: str = 'off', viz_freq: int = 1):
        """
        Initialize visualizer with lazy Pygame initialization.

        Args:
            config: Configuration dict (same as GA config)
            mode: Visualization mode ('off', 'scenario', 'evolution')
            viz_freq: Visualize every Nth generation/scenario (default: 1)
        """
        self.config = config
        self.mode = mode
        self.viz_freq = viz_freq
        self.initialized = False

        # Visualization config (with defaults)
        viz_config = config.get('visualization', {})
        self.screen_width = viz_config.get('screen_width', 1200)
        self.screen_height = viz_config.get('screen_height', 900)
        self.costmap_render_size = viz_config.get('costmap_render_size', 800)
        self.fps_cap = viz_config.get('fps_cap', 30)
        self.max_trajectories_shown = viz_config.get('max_trajectories_shown', 20)
        self.trajectory_line_width = viz_config.get('trajectory_line_width', 2)

        # Colors
        self.background_color = tuple(viz_config.get('background_color', [40, 40, 45]))
        self.text_color = tuple(viz_config.get('text_color', [220, 220, 220]))
        self.goal_color = tuple(viz_config.get('goal_color', [255, 255, 0]))
        self.start_color = tuple(viz_config.get('start_color', [0, 255, 255]))

        # Renderers (created on first use)
        self.costmap_renderer = None
        self.trajectory_renderer = None
        self.stats_renderer = None

        # Pygame objects
        self.screen = None
        self.clock = None
        self.font = None

        # State tracking
        self.current_scenario_id = 0
        self.current_generation = 0
        self.fitness_history_global = []  # Global fitness across all scenarios
        self.paused = False
        self.fast_forward = 0  # Skip next N visualizations

    def initialize_pygame(self):
        """Lazy initialization of Pygame (only when mode != 'off')."""
        if self.initialized:
            return

        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("GA Training Visualizer")

        self.clock = pygame.time.Clock()

        # Scale font size based on screen height (original design was 900px)
        scale_factor = self.screen_height / 900.0
        self.font = pygame.font.Font(None, int(24 * scale_factor))

        # Create renderers
        self.costmap_renderer = CostmapRenderer(self.costmap_render_size)
        self.trajectory_renderer = TrajectoryRenderer(self.costmap_render_size)

        # Calculate stats panel size based on layout
        stats_panel_width = self.screen_width - self.costmap_render_size
        stats_panel_height = self.costmap_render_size
        self.stats_renderer = StatsPanelRenderer(stats_panel_width, stats_panel_height)

        self.initialized = True
        print(f"Pygame visualizer initialized (mode: {self.mode})")

    def should_visualize(self, iteration: int) -> bool:
        """
        Check if current iteration should be visualized.

        Args:
            iteration: Current iteration number (generation or scenario)

        Returns:
            should_viz: True if should visualize
        """
        if self.mode == 'off':
            return False

        if self.fast_forward > 0:
            self.fast_forward -= 1
            return False

        return iteration % self.viz_freq == 0

    def on_scenario_start(self, scenario_id: int, scenario: Dict):
        """
        Callback when new scenario starts.

        Args:
            scenario_id: Scenario identifier
            scenario: Dict with 'costmap', 'start_state', 'goal_x', 'goal_y', 'goal_theta'
        """
        self.current_scenario_id = scenario_id
        self.current_generation = 0

    def on_generation_complete(self, generation: int, population: List,
                               environment) -> bool:
        """
        Callback after generation evolution completes (evolution mode).

        Simulates all trajectories, renders top N by fitness.

        Args:
            generation: Generation number
            population: List of Chromosome objects
            environment: NavigationEnvironment with scenario

        Returns:
            should_continue: False if user requests quit
        """
        if not self.should_visualize(generation):
            return True

        if not self.initialized:
            self.initialize_pygame()

        self.current_generation = generation

        # Simulate trajectories for population
        trajectories = []
        for chromosome in population:
            control_seq = chromosome.get_control_sequence()
            metrics = environment.simulate_control_sequence(control_seq)
            trajectories.append((metrics['trajectory'], chromosome.fitness))

        # Filter out trajectories with invalid fitness
        trajectories = [(traj, fit) for traj, fit in trajectories if np.isfinite(fit)]

        # Sort by fitness, take top N
        if trajectories:
            trajectories.sort(key=lambda x: x[1], reverse=True)
            trajectories = trajectories[:self.max_trajectories_shown]

        # Prepare stats
        fitnesses = [c.fitness for c in population if np.isfinite(c.fitness)]
        if not fitnesses:
            # All fitnesses are invalid, skip visualization
            return True

        # Get best chromosome (only from valid fitness chromosomes)
        valid_chromosomes = [c for c in population if np.isfinite(c.fitness)]
        best_chromosome = max(valid_chromosomes, key=lambda c: c.fitness)

        stats = {
            'scenario_id': self.current_scenario_id,
            'generation': generation,
            'best_fitness': max(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'diversity': compute_population_diversity(valid_chromosomes),
            'fitness_components': best_chromosome.fitness_components
        }

        # Update global fitness history
        self.fitness_history_global.append(stats['best_fitness'])

        # Render frame
        self.render_frame(environment, trajectories, stats)

        # Handle events
        return self.handle_events()

    def on_scenario_complete(self, scenario_id: int, best_chromosome,
                            fitness_history: List[float], environment) -> bool:
        """
        Callback after scenario training completes (scenario mode).

        Shows only the best trajectory found.

        Args:
            scenario_id: Scenario identifier
            best_chromosome: Best Chromosome from GA
            fitness_history: List of fitness per generation
            environment: NavigationEnvironment with scenario

        Returns:
            should_continue: False if user requests quit
        """
        if not self.should_visualize(scenario_id):
            return True

        if not self.initialized:
            self.initialize_pygame()

        self.current_scenario_id = scenario_id

        # Check if best chromosome has valid fitness
        if not np.isfinite(best_chromosome.fitness):
            # Skip visualization for invalid fitness
            return True

        # Simulate best trajectory
        control_seq = best_chromosome.get_control_sequence()
        metrics = environment.simulate_control_sequence(control_seq)
        trajectories = [(metrics['trajectory'], best_chromosome.fitness)]

        # Prepare stats
        stats = {
            'scenario_id': scenario_id,
            'generation': len(fitness_history),
            'best_fitness': best_chromosome.fitness,
            'avg_fitness': best_chromosome.fitness,
            'std_fitness': 0.0,
            'diversity': 0.0,
            'fitness_components': best_chromosome.fitness_components
        }

        # Update global fitness history
        self.fitness_history_global.append(stats['best_fitness'])

        # Render frame
        self.render_frame(environment, trajectories, stats)

        # Handle events
        return self.handle_events()

    def render_frame(self, environment, trajectories: List, stats: Dict):
        """
        Main rendering orchestration.

        Layout:
        - Costmap area (0, 0, costmap_render_size, costmap_render_size)
        - Stats panel (costmap_render_size, 0, panel_width, panel_height)
        - Control hints (0, costmap_render_size, screen_width, controls_height)

        Args:
            environment: NavigationEnvironment with costmap and scenario
            trajectories: List of (trajectory, fitness) tuples
            stats: Statistics dict
        """
        # Clear screen
        self.screen.fill(self.background_color)

        # Create costmap surface
        costmap_surface = pygame.Surface((self.costmap_render_size, self.costmap_render_size))

        # Render costmap
        self.costmap_renderer.render_costmap(environment.costmap, costmap_surface)

        # Render start/goal markers
        start_screen = self.costmap_renderer.world_to_screen(
            environment.start_state.x, environment.start_state.y, environment.costmap
        )
        goal_screen = self.costmap_renderer.world_to_screen(
            environment.goal_x, environment.goal_y, environment.costmap
        )

        # Draw start (cyan circle)
        pygame.draw.circle(costmap_surface, self.start_color, start_screen, 8)
        # Draw goal (yellow star - simplified as larger circle)
        pygame.draw.circle(costmap_surface, self.goal_color, goal_screen, 12)
        pygame.draw.circle(costmap_surface, (200, 200, 0), goal_screen, 12, 2)

        # Render trajectories
        if trajectories:
            fitnesses = [f for _, f in trajectories]
            fitness_min = min(fitnesses) if fitnesses else 0
            fitness_max = max(fitnesses) if fitnesses else 1

            for trajectory, fitness in trajectories:
                self.trajectory_renderer.render_trajectory(
                    trajectory, fitness, fitness_min, fitness_max,
                    environment.costmap, costmap_surface,
                    line_width=self.trajectory_line_width
                )

        # Blit costmap to screen
        self.screen.blit(costmap_surface, (0, 0))

        # Render stats panel
        stats_panel_width = self.screen_width - self.costmap_render_size
        stats_panel_height = self.costmap_render_size
        stats_surface = pygame.Surface((stats_panel_width, stats_panel_height))
        self.stats_renderer.render_stats(
            stats_surface, stats, self.fitness_history_global[-100:]  # Last 100 points
        )
        self.screen.blit(stats_surface, (self.costmap_render_size, 0))

        # Render control hints
        self._render_controls()

        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps_cap)

    def _render_controls(self):
        """Render control hints at bottom of screen."""
        controls_y = self.costmap_render_size
        controls_height = self.screen_height - self.costmap_render_size

        # Background
        controls_rect = pygame.Rect(0, controls_y, self.screen_width, controls_height)
        pygame.draw.rect(self.screen, (25, 25, 30), controls_rect)
        pygame.draw.line(self.screen, (60, 60, 65),
                        (0, controls_y), (self.screen_width, controls_y), 2)

        # Control text
        controls = [
            "SPACE: Pause/Resume",
            "ESC: Quit",
            "S: Screenshot",
            "F: Fast-forward (skip 10)"
        ]

        # Scale based on screen size
        scale = self.screen_height / 900.0  # Original design was 900px height
        margin = int(20 * scale)
        row_spacing = int(30 * scale)
        col_spacing = int(300 * scale)

        x_offset = margin
        y_offset = controls_y + row_spacing

        for i, control in enumerate(controls):
            if i == 2:  # Move to second row
                x_offset = margin
                y_offset += row_spacing

            text_surface = self.font.render(control, True, self.text_color)
            self.screen.blit(text_surface, (x_offset, y_offset))
            x_offset += col_spacing

        # Pause indicator
        if self.paused:
            pause_text = self.font.render("[ PAUSED ]", True, (255, 100, 100))
            pause_x = self.screen_width - int(150 * scale)
            pause_y = controls_y + int(40 * scale)
            self.screen.blit(pause_text, (pause_x, pause_y))

    def handle_events(self) -> bool:
        """
        Process Pygame events.

        Controls:
        - SPACE: Pause/resume
        - ESC: Quit training
        - S: Save screenshot
        - F: Fast-forward (skip next 10 visualizations)

        Returns:
            continue_training: False if user requests quit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print(f"Visualizer {'paused' if self.paused else 'resumed'}")
                elif event.key == pygame.K_s:
                    self.save_screenshot()
                elif event.key == pygame.K_f:
                    self.fast_forward = 10
                    print("Fast-forward: skipping next 10 visualizations")

        # Handle pause loop
        while self.paused:
            self.clock.tick(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = False
                        print("Visualizer resumed")
                    elif event.key == pygame.K_ESCAPE:
                        return False

        return True

    def save_screenshot(self):
        """Save current frame as PNG."""
        # Create screenshots directory
        screenshots_dir = Path("screenshots")
        screenshots_dir.mkdir(exist_ok=True)

        # Generate filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = screenshots_dir / f"ga_viz_scenario{self.current_scenario_id}_{timestamp}.png"

        # Save screenshot
        pygame.image.save(self.screen, str(filename))
        print(f"Screenshot saved: {filename}")

    def close(self):
        """Clean up Pygame resources."""
        if self.initialized:
            pygame.quit()
            self.initialized = False
            print("Pygame visualizer closed")


if __name__ == "__main__":
    # Test visualizer
    print("Testing GAVisualizer...")

    import yaml
    from pathlib import Path

    # Load config
    config_path = Path(__file__).parent.parent / "config" / "ga_config.yaml"
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Creating minimal test config...")
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
            },
            'costmap': {
                'width': 50,
                'height': 50,
                'resolution': 0.05,
                'inflation_radius': 0.5,
                'inflation_decay': 0.8
            },
            'visualization': {
                'screen_width': 1800,
                'screen_height': 1350,
                'costmap_render_size': 1200,
                'fps_cap': 30,
                'max_trajectories_shown': 20
            }
        }
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Test 1: Initialize in 'off' mode
    print("\n--- Test 1: Initialize (off mode) ---")
    viz_off = GAVisualizer(config, mode='off')
    assert viz_off.mode == 'off'
    assert not viz_off.initialized
    print("✓ Off mode initialization works")

    # Test 2: Initialize in 'scenario' mode
    print("\n--- Test 2: Initialize (scenario mode) ---")
    viz_scenario = GAVisualizer(config, mode='scenario', viz_freq=1)
    assert viz_scenario.mode == 'scenario'
    assert not viz_scenario.initialized  # Lazy init
    print("✓ Scenario mode initialization works")

    # Test 3: should_visualize
    print("\n--- Test 3: should_visualize ---")
    assert not viz_off.should_visualize(0)  # Off mode
    assert viz_scenario.should_visualize(0)  # viz_freq=1
    assert viz_scenario.should_visualize(1)

    viz_sparse = GAVisualizer(config, mode='evolution', viz_freq=5)
    assert viz_sparse.should_visualize(0)
    assert not viz_sparse.should_visualize(1)
    assert viz_sparse.should_visualize(5)
    print("✓ should_visualize works")

    # Test 4: Full integration test (requires user interaction)
    print("\n--- Test 4: Full render test (manual) ---")
    print("This test requires visual verification.")
    print("A window will open. Press ESC to close.")

    # Import simulator components
    try:
        from ..simulator import (
            generate_random_costmap,
            RobotState,
            NavigationEnvironment
        )
        from ..ga import GeneticAlgorithm

        # Create test scenario
        environment = NavigationEnvironment(config)

        costmap = generate_random_costmap(
            width=50,
            height=50,
            resolution=0.05,
            num_obstacles=3,
            obstacle_radius_range=(3, 5),
            inflation_radius=0.5,
            inflation_decay=0.8,
            free_radius_center=0.5
        )

        center_x = costmap.width * costmap.resolution / 2.0
        center_y = costmap.height * costmap.resolution / 2.0
        start_state = RobotState(x=center_x, y=center_y, theta=0.0)
        goal_x = center_x + 1.5
        goal_y = center_y + 0.5

        environment.reset(costmap, start_state, goal_x, goal_y, 0.0)

        # Create GA
        ga = GeneticAlgorithm(config)

        # Create visualizer
        viz_test = GAVisualizer(config, mode='evolution', viz_freq=1)
        viz_test.on_scenario_start(0, {})

        # Run a few generations
        population = ga.initialize_population()

        for gen in range(5):
            population = ga.evolve_generation(population, environment, num_workers=2)

            # Visualize
            should_continue = viz_test.on_generation_complete(gen, population, environment)

            if not should_continue:
                print("User quit visualization")
                break

        viz_test.close()
        print("✓ Full render test complete")

    except ImportError as e:
        print(f"Skipping integration test (missing dependencies): {e}")

    print("\n✓ GAVisualizer tests passed!")
