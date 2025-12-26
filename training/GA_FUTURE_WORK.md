# GA Training: Future Enhancements

This document outlines advanced features and optimizations that can be added to the genetic algorithm training system. The current implementation prioritizes getting a working pipeline quickly with parallel processing and core functionality. These enhancements can significantly improve training quality, debuggability, and efficiency.

---

## 1. Visualization and Monitoring

### 1.1 Real-Time Training Visualization

**Motivation**: Visual feedback helps understand GA behavior, debug fitness functions, and monitor convergence. YouTube videos of GA training often show animated populations evolving, which provides intuitive insights.

**Implementation**:

#### Option A: Matplotlib Animation (Simple)
```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon

class GAVisualizer:
    def __init__(self, environment, figsize=(12, 8)):
        self.environment = environment
        self.fig, self.axes = plt.subplots(1, 2, figsize=figsize)

        # Left: Costmap + trajectories
        self.ax_map = self.axes[0]
        self.ax_map.set_title("Population Trajectories")

        # Right: Fitness evolution
        self.ax_fitness = self.axes[1]
        self.ax_fitness.set_title("Fitness Over Generations")

        self.fitness_history = []

    def visualize_generation(self, population, generation):
        """Visualize population at current generation."""
        self.ax_map.clear()

        # Draw costmap
        costmap_img = self.environment.costmap.data
        extent = [0, self.environment.costmap.width * self.environment.costmap.resolution,
                 0, self.environment.costmap.height * self.environment.costmap.resolution]
        self.ax_map.imshow(costmap_img, origin='lower', extent=extent,
                          cmap='gray_r', alpha=0.7)

        # Draw start and goal
        self.ax_map.plot(self.environment.start_state.x,
                        self.environment.start_state.y,
                        'go', markersize=10, label='Start')
        self.ax_map.plot(self.environment.goal_x, self.environment.goal_y,
                        'r*', markersize=15, label='Goal')

        # Draw top 5 trajectories
        sorted_pop = sorted(population, key=lambda c: c.fitness, reverse=True)
        colors = plt.cm.viridis(np.linspace(0, 1, 5))

        for i, chrom in enumerate(sorted_pop[:5]):
            control_seq = chrom.get_control_sequence()
            metrics = self.environment.simulate_control_sequence(control_seq)
            trajectory = metrics['trajectory']

            xs = [state.x for state in trajectory]
            ys = [state.y for state in trajectory]

            alpha = 1.0 - (i * 0.15)  # Best trajectory most opaque
            self.ax_map.plot(xs, ys, color=colors[i], alpha=alpha, linewidth=2,
                           label=f"#{i+1}: {chrom.fitness:.1f}")

        self.ax_map.legend()
        self.ax_map.set_xlabel("X (m)")
        self.ax_map.set_ylabel("Y (m)")

        # Update fitness plot
        best_fitness = max(population, key=lambda c: c.fitness).fitness
        avg_fitness = np.mean([c.fitness for c in population])
        self.fitness_history.append((generation, best_fitness, avg_fitness))

        if len(self.fitness_history) > 1:
            self.ax_fitness.clear()
            gens, bests, avgs = zip(*self.fitness_history)
            self.ax_fitness.plot(gens, bests, 'r-', linewidth=2, label='Best')
            self.ax_fitness.plot(gens, avgs, 'b--', label='Average')
            self.ax_fitness.set_xlabel("Generation")
            self.ax_fitness.set_ylabel("Fitness")
            self.ax_fitness.legend()
            self.ax_fitness.grid(True)

        plt.tight_layout()
        plt.pause(0.01)

# Usage in evolution.py:
# visualizer = GAVisualizer(environment)
# for generation in range(num_generations):
#     population = evolve_generation(...)
#     if generation % 5 == 0:  # Visualize every 5 generations
#         visualizer.visualize_generation(population, generation)
```

#### Option B: Pygame (Interactive, Faster)
```python
import pygame
import numpy as np

class PygameVisualizer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("GA Training")
        self.clock = pygame.time.Clock()

    def draw_costmap(self, costmap):
        """Render costmap as pygame surface."""
        # Convert to RGB
        costmap_rgb = np.stack([costmap.data]*3, axis=-1)
        costmap_rgb = 255 - costmap_rgb  # Invert for display

        # Scale to window
        surface = pygame.surfarray.make_surface(costmap_rgb.swapaxes(0, 1))
        surface = pygame.transform.scale(surface, (self.screen.get_width()//2,
                                                   self.screen.get_height()))
        return surface

    def draw_trajectory(self, trajectory, color, surface):
        """Draw trajectory on surface."""
        points = [(state.x * 10, state.y * 10) for state in trajectory]
        if len(points) > 1:
            pygame.draw.lines(surface, color, False, points, 2)

    def update(self, population, environment):
        """Update visualization."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill((255, 255, 255))

        # Draw costmap
        costmap_surface = self.draw_costmap(environment.costmap)
        self.screen.blit(costmap_surface, (0, 0))

        # Draw trajectories
        sorted_pop = sorted(population, key=lambda c: c.fitness, reverse=True)
        for i, chrom in enumerate(sorted_pop[:10]):
            control_seq = chrom.get_control_sequence()
            metrics = environment.simulate_control_sequence(control_seq)
            trajectory = metrics['trajectory']

            # Color from green (best) to red (worst)
            color = (int(255 * i / 10), int(255 * (1 - i / 10)), 0)
            self.draw_trajectory(trajectory, color, self.screen)

        pygame.display.flip()
        self.clock.tick(30)
        return True
```

**Benefits**:
- Intuitive understanding of GA behavior
- Easy debugging of fitness function
- Identifies stuck populations or premature convergence
- Engaging training visualization (like YouTube videos)

**Integration**: Add `--visualize` flag to `train_ga.py`

---

### 1.2 TensorBoard Logging

**Motivation**: Professional training monitoring with rich visualizations, hyperparameter tracking, and experiment comparison.

**Implementation**:
```python
from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger:
    def __init__(self, log_dir='runs/ga_training'):
        self.writer = SummaryWriter(log_dir)

    def log_generation(self, generation, population, best_chromosome):
        """Log generation statistics."""
        # Fitness statistics
        fitnesses = [c.fitness for c in population]
        self.writer.add_scalar('fitness/best', max(fitnesses), generation)
        self.writer.add_scalar('fitness/mean', np.mean(fitnesses), generation)
        self.writer.add_scalar('fitness/std', np.std(fitnesses), generation)

        # Fitness components
        for key, value in best_chromosome.fitness_components.items():
            self.writer.add_scalar(f'components/{key}', value, generation)

        # Population diversity (std of genes)
        all_genes = np.array([c.genes for c in population])
        gene_std = np.std(all_genes, axis=0).mean()
        self.writer.add_scalar('diversity/gene_std', gene_std, generation)

        # Control sequence heatmap (best chromosome)
        control_img = best_chromosome.genes.T  # (3, num_steps)
        self.writer.add_image('controls/best', control_img, generation, dataformats='HW')

    def log_scenario(self, scenario_id, best_fitness, metrics):
        """Log per-scenario results."""
        self.writer.add_scalar('scenario/fitness', best_fitness, scenario_id)
        self.writer.add_scalar('scenario/goal_distance', metrics['goal_distance'], scenario_id)
        self.writer.add_scalar('scenario/collision', int(metrics['collision']), scenario_id)

    def close(self):
        self.writer.close()

# Usage:
# logger = TensorBoardLogger()
# for generation in range(num_generations):
#     population = evolve_generation(...)
#     logger.log_generation(generation, population, best_chromosome)
# logger.close()
```

**View with**: `tensorboard --logdir runs/ga_training`

**Benefits**:
- Track fitness components separately
- Monitor population diversity (detect premature convergence)
- Compare different hyperparameter configurations
- Visualize control sequences as heatmaps

---

## 2. Stage Simulator Integration

**Current**: Lightweight Python simulator (procedural costmaps, simple dynamics)
**Future**: ROS Stage simulator (realistic environments, sensor noise, dynamic obstacles)

### 2.1 Stage Python Wrapper

**Motivation**: Train on realistic world files used in deployment, handle sensor noise and dynamic obstacles.

**Implementation**:
```python
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped
from gazebo_msgs.srv import SetModelState

class StageWrapper:
    def __init__(self, world_file):
        """Initialize Stage simulator."""
        # Launch Stage node
        os.system(f"rosrun stage_ros stageros {world_file} &")
        rospy.init_node('ga_trainer', anonymous=True)

        # Subscribers
        self.costmap_sub = rospy.Subscriber('/costmap', OccupancyGrid, self.costmap_callback)
        self.pose_sub = rospy.Subscriber('/robot_pose', PoseStamped, self.pose_callback)

        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.costmap = None
        self.current_pose = None

    def reset_robot(self, x, y, theta):
        """Teleport robot to start position."""
        # Use SetModelState service
        pass

    def execute_control(self, v_x, v_y, omega, dt):
        """Execute control for dt seconds."""
        msg = Twist()
        msg.linear.x = v_x
        msg.linear.y = v_y
        msg.angular.z = omega

        rate = rospy.Rate(10)
        for _ in range(int(dt * 10)):
            self.cmd_vel_pub.publish(msg)
            rate.sleep()

    def get_costmap(self):
        """Get current costmap from ROS."""
        return self.costmap

    def check_collision(self):
        """Check if robot is in collision."""
        # Read from bumper sensor or check costmap at robot pose
        pass
```

**Benefits**:
- Realistic sensor noise (LIDAR, odometry)
- Dynamic obstacles (moving pedestrians, robots)
- Real-world map files (.world, .yaml)
- Better sim-to-real transfer

**Trade-offs**:
- Slower training (ROS overhead)
- Harder to parallelize (multiple ROS masters)
- More complex setup

---

## 3. Adaptive Parameter Control

### 3.1 Adaptive Mutation Rate

**Motivation**: High mutation early (exploration), low mutation late (exploitation)

**Implementation**:
```python
def adaptive_mutation_rate(generation, max_generations, initial_rate=0.3, final_rate=0.05):
    """Decrease mutation rate over generations."""
    progress = generation / max_generations
    return initial_rate * (1 - progress) + final_rate * progress

# In evolution.py:
mutation_rate = adaptive_mutation_rate(generation, num_generations)
gaussian_mutation(child, mutation_rate, mutation_strength=0.1)
```

**Alternative: Diversity-based**:
```python
def diversity_based_mutation(population, base_rate=0.1):
    """Increase mutation when diversity is low."""
    all_genes = np.array([c.genes for c in population])
    diversity = np.std(all_genes)

    # If diversity < threshold, increase mutation
    if diversity < 0.1:
        return base_rate * 2.0
    return base_rate
```

---

### 3.2 Adaptive Crossover Rate

**Motivation**: More crossover when population is diverse, less when converged.

**Implementation**:
```python
def adaptive_crossover_rate(population, min_rate=0.5, max_rate=0.95):
    """Adjust crossover based on fitness variance."""
    fitnesses = np.array([c.fitness for c in population])
    fitness_std = np.std(fitnesses)

    # Normalize to [0, 1]
    normalized_std = np.clip(fitness_std / 10.0, 0, 1)

    # High diversity → high crossover
    return min_rate + (max_rate - min_rate) * normalized_std
```

---

## 4. Multi-Objective Pareto Optimization

**Current**: Weighted sum fitness (single objective)
**Future**: NSGA-II (discover Pareto frontier of trade-offs)

### 4.1 NSGA-II Implementation

**Motivation**: Explore trade-offs between goal distance, smoothness, time efficiency without hand-tuning weights.

**Implementation** (high-level):
```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem

class NavigationProblem(Problem):
    def __init__(self, environment):
        super().__init__(n_var=60,  # num_steps * 3
                        n_obj=4,    # 4 objectives
                        n_constr=0,
                        xl=-1.0, xu=1.0)  # Variable bounds
        self.environment = environment

    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate population for NSGA-II."""
        fitnesses = []

        for genes in X:
            # Reshape to control sequence
            control_seq = genes.reshape(-1, 3)
            metrics = self.environment.simulate_control_sequence(control_seq)

            # Return multiple objectives (minimize all)
            objectives = [
                metrics['goal_distance'],
                metrics['max_cost'] / 254.0,
                metrics['smoothness'],
                metrics['path_length']
            ]
            fitnesses.append(objectives)

        out["F"] = np.array(fitnesses)

# Usage:
problem = NavigationProblem(environment)
algorithm = NSGA2(pop_size=100)
res = minimize(problem, algorithm, ('n_gen', 100))

# res.F contains Pareto front
# res.X contains corresponding solutions
```

**Benefits**:
- Discover multiple good solutions with different trade-offs
- No need to manually tune weights
- Better exploration of solution space

**Trade-offs**:
- More complex implementation
- Requires `pymoo` library
- Harder to pick "best" solution (need user preference)

---

## 5. Convergence Detection and Early Stopping

**Motivation**: Stop evolution when fitness plateaus to save computation.

**Implementation**:
```python
class ConvergenceDetector:
    def __init__(self, patience=10, min_improvement=0.01):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_fitness = -np.inf
        self.stale_count = 0

    def check(self, current_best_fitness):
        """Check if converged."""
        improvement = current_best_fitness - self.best_fitness

        if improvement > self.min_improvement:
            self.best_fitness = current_best_fitness
            self.stale_count = 0
        else:
            self.stale_count += 1

        return self.stale_count >= self.patience

# In evolution.py:
detector = ConvergenceDetector(patience=15)
for generation in range(num_generations):
    population = evolve_generation(...)
    best_fitness = max(population, key=lambda c: c.fitness).fitness

    if detector.check(best_fitness):
        print(f"Converged at generation {generation}")
        break
```

**Benefits**:
- Faster training (skip unnecessary generations)
- Automatic termination
- Better resource utilization

---

## 6. Advanced Genetic Operators

### 6.1 Blend Crossover (BLX-α)

**Motivation**: Better real-valued crossover than uniform.

**Implementation**:
```python
def blx_alpha_crossover(parent1, parent2, alpha=0.5, crossover_rate=0.8):
    """BLX-α crossover for real-valued genes."""
    offspring = parent1.copy()

    if np.random.random() < crossover_rate:
        for i in range(parent1.num_steps):
            for j in range(3):  # v_x, v_y, omega
                g1 = parent1.genes[i, j]
                g2 = parent2.genes[i, j]

                # Compute blend range
                d = abs(g1 - g2)
                lower = min(g1, g2) - alpha * d
                upper = max(g1, g2) + alpha * d

                # Sample from range
                offspring.genes[i, j] = np.random.uniform(lower, upper)

        offspring.clamp()

    offspring.fitness = -np.inf
    return offspring
```

---

### 6.2 Polynomial Mutation

**Motivation**: Self-adaptive mutation strength.

**Implementation**:
```python
def polynomial_mutation(chromosome, mutation_rate=0.1, eta=20):
    """Polynomial mutation (from NSGA-II)."""
    for i in range(chromosome.num_steps):
        for j in range(3):
            if np.random.random() < mutation_rate:
                gene = chromosome.genes[i, j]
                lower = chromosome.velocity_limits[f'min_{"vxy"[j] if j < 2 else "omega"}']
                upper = chromosome.velocity_limits[f'max_{"vxy"[j] if j < 2 else "omega"}']

                delta1 = (gene - lower) / (upper - lower)
                delta2 = (upper - gene) / (upper - lower)

                rand = np.random.random()
                mut_pow = 1.0 / (eta + 1.0)

                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    deltaq = (val ** mut_pow) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - (val ** mut_pow)

                chromosome.genes[i, j] = gene + deltaq * (upper - lower)

    chromosome.clamp()
    chromosome.fitness = -np.inf
```

---

## 7. Curriculum Learning

**Motivation**: Start with easy scenarios, progressively increase difficulty.

**Implementation**:
```python
def generate_curriculum_scenario(scenario_id, difficulty_level):
    """Generate scenario with controlled difficulty."""
    # difficulty_level: 0 (easy) to 1 (hard)

    # Easy: few obstacles, close goal, large free radius
    # Hard: many obstacles, far goal, small free radius

    num_obstacles = int(2 + difficulty_level * 6)  # 2-8 obstacles
    goal_distance = 1.0 + difficulty_level * 2.0   # 1.0-3.0m
    obstacle_radius_max = int(4 + difficulty_level * 4)  # 4-8 pixels
    free_radius = 0.8 - difficulty_level * 0.3     # 0.8-0.5m

    costmap = generate_random_costmap(
        width=50, height=50, resolution=0.05,
        num_obstacles=num_obstacles,
        obstacle_radius_range=(3, obstacle_radius_max),
        inflation_radius=0.5,
        inflation_decay=0.8,
        free_radius_center=free_radius
    )

    # ... generate start and goal

    return scenario

# In train_ga.py:
for scenario_id in range(num_scenarios):
    difficulty = min(1.0, scenario_id / (num_scenarios * 0.5))  # Ramp up to 50%
    scenario = generate_curriculum_scenario(scenario_id, difficulty)
    # ... run GA
```

**Benefits**:
- Faster convergence (start with easy wins)
- Better generalization
- More robust policies

---

## 8. Coevolution

**Motivation**: Simultaneously evolve controllers and challenging scenarios.

**Implementation**:
```python
def coevolve(controller_population, scenario_population, num_generations):
    """Coevolve controllers and scenarios."""
    for gen in range(num_generations):
        # Evaluate controllers on scenarios
        for controller in controller_population:
            fitness_scores = []
            for scenario in scenario_population:
                score = evaluate(controller, scenario)
                fitness_scores.append(score)
            controller.fitness = np.mean(fitness_scores)

        # Evaluate scenarios (difficulty = how hard for controllers)
        for scenario in scenario_population:
            difficulty_scores = []
            for controller in controller_population:
                score = evaluate(controller, scenario)
                # Hard scenarios have low scores
                difficulty_scores.append(-score)
            scenario.fitness = np.mean(difficulty_scores)

        # Evolve both populations
        controller_population = evolve(controller_population)
        scenario_population = evolve(scenario_population)

    return controller_population, scenario_population
```

**Benefits**:
- Discovers challenging scenarios automatically
- More robust controllers
- Prevents overfitting to fixed scenario distribution

---

## 9. Fitness Landscape Analysis

**Motivation**: Understand and debug fitness function behavior.

**Implementation**:
```python
def analyze_fitness_landscape(environment, num_samples=1000):
    """Sample random control sequences and analyze fitness distribution."""
    fitnesses = []
    goal_distances = []
    collisions = []

    for _ in range(num_samples):
        # Random control sequence
        control_seq = np.random.uniform(-1, 1, (20, 3))
        metrics = environment.simulate_control_sequence(control_seq)

        fitness = compute_fitness(metrics)
        fitnesses.append(fitness)
        goal_distances.append(metrics['goal_distance'])
        collisions.append(metrics['collision'])

    # Visualize distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(fitnesses, bins=50)
    axes[0, 0].set_title("Fitness Distribution")

    axes[0, 1].scatter(goal_distances, fitnesses, alpha=0.5)
    axes[0, 1].set_xlabel("Goal Distance")
    axes[0, 1].set_ylabel("Fitness")

    axes[1, 0].scatter(collisions, fitnesses, alpha=0.5)
    axes[1, 0].set_xlabel("Collision")
    axes[1, 0].set_ylabel("Fitness")

    # Correlation heatmap
    data = np.array([fitnesses, goal_distances, [int(c) for c in collisions]]).T
    corr = np.corrcoef(data.T)
    axes[1, 1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_title("Correlation Matrix")

    plt.tight_layout()
    plt.show()

# Usage:
# analyze_fitness_landscape(environment)
```

**Benefits**:
- Identify fitness function issues (e.g., too easy, deceptive)
- Understand objective correlations
- Debug premature convergence

---

## 10. Hyperparameter Tuning

**Motivation**: Find optimal GA parameters automatically.

**Implementation** (using Optuna):
```python
import optuna

def objective(trial):
    """Optuna objective function."""
    # Suggest hyperparameters
    config = {
        'ga': {
            'population_size': trial.suggest_int('population_size', 50, 200),
            'elite_size': trial.suggest_int('elite_size', 5, 20),
            'mutation_rate': trial.suggest_float('mutation_rate', 0.01, 0.3),
            'crossover_rate': trial.suggest_float('crossover_rate', 0.5, 0.95),
            'num_generations': 50,  # Fixed for tuning
            'time_horizon': 2.0,
            'control_frequency': 10.0
        },
        'fitness_weights': {
            'goal_distance': trial.suggest_float('w_goal', 0.5, 2.0),
            'collision': trial.suggest_float('w_collision', 5.0, 20.0),
            'smoothness': trial.suggest_float('w_smoothness', 0.1, 1.0),
            'time_efficiency': trial.suggest_float('w_time', 0.1, 1.0)
        },
        'robot': {...}  # Fixed robot config
    }

    # Run GA on test scenarios
    ga = GeneticAlgorithm(config)
    environment = NavigationEnvironment(config)

    total_fitness = 0
    num_test_scenarios = 10

    for i in range(num_test_scenarios):
        scenario = generate_scenario(config, i)
        environment.reset(...)
        best, history = ga.run(environment, num_workers=8, verbose=False)
        total_fitness += best.fitness

    return total_fitness / num_test_scenarios

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Best hyperparameters:", study.best_params)
```

**Benefits**:
- Systematic hyperparameter search
- Better performance than manual tuning
- Insight into parameter sensitivity

---

## Summary

This document outlined 10 major categories of enhancements for the GA training system:

1. **Visualization** - Real-time monitoring and debugging
2. **Stage Integration** - Realistic simulation
3. **Adaptive Parameters** - Dynamic mutation/crossover
4. **Multi-Objective** - Pareto optimization
5. **Convergence Detection** - Early stopping
6. **Advanced Operators** - Better crossover/mutation
7. **Curriculum Learning** - Progressive difficulty
8. **Coevolution** - Adversarial scenario generation
9. **Landscape Analysis** - Fitness function debugging
10. **Hyperparameter Tuning** - Automatic optimization

**Implementation Priority**:
1. **High**: Visualization (#1) - Provides immediate value for debugging
2. **High**: Convergence detection (#5) - Saves computation time
3. **Medium**: Adaptive parameters (#3) - Easy to add, moderate benefit
4. **Medium**: TensorBoard (#1.2) - Professional monitoring
5. **Low**: Stage integration (#2) - Complex, requires ROS setup
6. **Low**: Multi-objective (#4) - Research-oriented, not critical for initial training

**Estimated Effort**:
- Visualization: 4-6 hours (Matplotlib) or 8-12 hours (Pygame)
- TensorBoard: 3-4 hours
- Convergence detection: 2-3 hours
- Adaptive parameters: 2-3 hours
- Advanced operators: 4-6 hours
- Stage integration: 20-30 hours (complex)

**Next Steps**:
After completing initial training pipeline, prioritize visualization for better understanding of GA behavior during real training runs on 1000 scenarios.
