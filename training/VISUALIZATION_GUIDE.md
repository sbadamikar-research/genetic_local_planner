# GA Training Visualization Guide

## Overview

The Pygame visualizer provides real-time visualization of genetic algorithm training, helping you understand population evolution, fitness convergence, and trajectory optimization.

## Installation

```bash
# Activate environment
conda activate plan_ga

# Install dependencies
conda env update -f environment.yml --prune
# OR
pip install pygame>=2.5.0 pillow>=10.0.0
```

## Visualization Modes

### 1. Off Mode (Default)
No visualization. Use for production training runs.

```bash
python training/train_ga.py \
  --config training/config/ga_config.yaml \
  --output models/checkpoints/ga_trajectories.pkl \
  --num_scenarios 1000 \
  --num_workers 8
```

### 2. Scenario Mode
Shows final best trajectory per scenario. Lightweight monitoring.

```bash
python training/train_ga.py \
  --config training/config/ga_config.yaml \
  --output models/checkpoints/ga_trajectories.pkl \
  --num_scenarios 1000 \
  --num_workers 8 \
  --visualize scenario \
  --viz-freq 10  # Visualize every 10th scenario
```

**When to use:**
- Long training runs where you want periodic progress checks
- Verifying training is progressing correctly
- Monitoring final trajectory quality

**Performance impact:** Negligible (~0.1-0.3% overhead)

### 3. Evolution Mode
Shows entire population evolving per generation. Detailed debugging.

```bash
python training/train_ga.py \
  --config training/config/ga_config.yaml \
  --output models/checkpoints/ga_trajectories.pkl \
  --num_scenarios 50 \
  --num_workers 8 \
  --visualize evolution \
  --viz-freq 5  # Visualize every 5th generation
```

**When to use:**
- Understanding how GA explores solution space
- Debugging fitness function issues
- Detecting premature convergence
- Tuning GA hyperparameters

**Performance impact:** ~1-3% overhead (depends on viz-freq)

## Visualization Layout

```
┌────────────────────────────────┬────────────────┐
│                                │  STATISTICS    │
│   COSTMAP + TRAJECTORIES       │  - Scenario    │
│   800×800 px                   │  - Generation  │
│                                │  - Fitness     │
│   - White: Free space          │  - Components  │
│   - Red: Obstacles             │  - Graph       │
│   - Cyan circle: Start         │                │
│   - Yellow circle: Goal        │  400×800 px    │
│   - Green trajectories: Best   │                │
│   - Red trajectories: Worst    │                │
│                                │                │
├────────────────────────────────┴────────────────┤
│  CONTROLS: SPACE=pause ESC=quit S=screenshot    │
│  1200×100 px                                    │
└─────────────────────────────────────────────────┘
```

## Interactive Controls

| Key | Action | Description |
|-----|--------|-------------|
| **SPACE** | Pause/Resume | Freeze training, inspect current state |
| **ESC** | Quit | Stop training and exit visualizer |
| **S** | Screenshot | Save PNG to `screenshots/` directory |
| **F** | Fast-forward | Skip next 10 visualizations |

## Statistics Panel

The right panel displays:

1. **Scenario Info**
   - Current scenario number
   - Current generation

2. **Fitness Metrics**
   - Best: Highest fitness in population
   - Avg: Mean fitness
   - Std: Standard deviation

3. **Population Diversity**
   - Average pairwise distance between chromosomes
   - Higher = more exploration
   - Lower = converging to solution

4. **Fitness Components**
   - Goal distance (meters to goal)
   - Collision (boolean)
   - Goal reached (boolean)

5. **Fitness History Graph**
   - Shows last 100 fitness values
   - Visualizes convergence trend

## Configuration

Edit `training/config/ga_config.yaml`:

```yaml
visualization:
  # Rendering settings
  screen_width: 1200
  screen_height: 900
  costmap_render_size: 800
  fps_cap: 30

  # Trajectory rendering
  max_trajectories_shown: 20  # Top N trajectories by fitness
  trajectory_line_width: 2

  # Color scheme
  background_color: [40, 40, 45]
  text_color: [220, 220, 220]
  goal_color: [255, 255, 0]
  start_color: [0, 255, 255]
```

## Usage Tips

### Quick Test
Run a small number of scenarios to verify setup:

```bash
python training/train_ga.py \
  --config training/config/ga_config.yaml \
  --output models/checkpoints/test.pkl \
  --num_scenarios 5 \
  --num_workers 2 \
  --visualize scenario
```

### Debugging GA Hyperparameters
Use evolution mode with high frequency to see detailed behavior:

```bash
python training/train_ga.py \
  --config training/config/ga_config.yaml \
  --output models/checkpoints/debug.pkl \
  --num_scenarios 10 \
  --num_workers 2 \
  --visualize evolution \
  --viz-freq 1  # Every generation
```

### Production Training with Monitoring
Use scenario mode with sparse visualization:

```bash
python training/train_ga.py \
  --config training/config/ga_config.yaml \
  --output models/checkpoints/ga_trajectories.pkl \
  --num_scenarios 1000 \
  --num_workers 8 \
  --visualize scenario \
  --viz-freq 50  # Check every 50 scenarios
```

## Interpreting Visualizations

### Good Training Signs
- **Fitness increasing** steadily in graph
- **Diversity decreasing** over generations (convergence)
- **Green trajectories** (high fitness) dominate in evolution mode
- **Trajectories avoiding obstacles** and reaching goals
- **Smooth trajectory curves** (not jagged)

### Potential Issues
- **Flat fitness graph**: Premature convergence or stuck
  - Solution: Increase mutation rate, population size
- **High diversity persisting**: Not converging
  - Solution: Increase elite size, run more generations
- **Trajectories in obstacles**: Collision penalty too low
  - Solution: Increase collision weight in fitness
- **Erratic trajectories**: Over-exploration
  - Solution: Decrease mutation rate

## Screenshot Analysis

Screenshots are saved to `screenshots/` with naming:
```
ga_viz_scenario42_20260219_143022.png
```

Use screenshots to:
- Document training progress
- Compare different hyperparameter settings
- Create training presentations/reports
- Debug specific problematic scenarios

## Performance Considerations

### Overhead by Mode

| Mode | Overhead | When to Use |
|------|----------|-------------|
| off | 0% | Production runs |
| scenario (freq=1) | 0.3% | Always-on monitoring |
| scenario (freq=10) | <0.1% | Periodic checks |
| evolution (freq=1) | 3-5% | Deep debugging |
| evolution (freq=5) | 1% | Moderate monitoring |

### Optimization Tips

1. **Use viz-freq parameter** to reduce update frequency
2. **Reduce max_trajectories_shown** if evolution mode is slow
3. **Use scenario mode** instead of evolution for long runs
4. **Close visualizer window** early if not needed (ESC)

## Troubleshooting

### "No module named 'pygame'"
```bash
pip install pygame pillow
```

### Window not appearing
- Check that DISPLAY is set (if SSH)
- Try running on local machine
- Verify pygame installation: `python -c "import pygame; print(pygame.ver)"`

### Slow visualization
- Increase viz-freq: `--viz-freq 10`
- Reduce population size in evolution mode
- Lower fps_cap in config: `fps_cap: 15`

### Training hangs after closing window
- Use ESC to quit properly (don't force-close window)
- Check console for error messages

## Module Architecture

```
training/visualization/
├── __init__.py               # Module exports
├── color_utils.py            # Color mapping utilities
├── renderer.py               # Rendering classes
│   ├── CostmapRenderer       # Costmap visualization
│   ├── TrajectoryRenderer    # Trajectory polylines
│   └── StatsPanelRenderer    # Statistics display
└── pygame_visualizer.py      # Main GAVisualizer class
```

### Extension Points

The modular architecture makes it easy to:
- Add new rendering modes
- Export to video (using pygame.movie)
- Integrate with TensorBoard
- Add real-time hyperparameter tuning UI
- Create multi-scenario comparison view

## Example Workflow

1. **Initial test** (verify setup):
   ```bash
   python training/train_ga.py --config ... --num_scenarios 5 --visualize scenario
   ```

2. **Hyperparameter tuning** (understand GA behavior):
   ```bash
   python training/train_ga.py --config ... --num_scenarios 20 --visualize evolution --viz-freq 5
   ```

3. **Full training** (with monitoring):
   ```bash
   python training/train_ga.py --config ... --num_scenarios 1000 --visualize scenario --viz-freq 25
   ```

4. **Review screenshots** in `screenshots/` directory

5. **Analyze results** using printed statistics

## See Also

- `training/GA_FUTURE_WORK.md`: Future enhancement ideas
- `training/config/ga_config.yaml`: Full configuration reference
- `training/train_ga.py`: Main training script
- `training/ga/evolution.py`: GA implementation
