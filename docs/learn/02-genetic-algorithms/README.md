# Module 02: Genetic Algorithms

**Estimated Time:** 1 day (6-8 hours)

## 🎯 Learning Objectives

- ✅ Understand how genetic algorithms work (evolution analogy)
- ✅ Implement GA components: chromosome, fitness, selection, crossover, mutation
- ✅ Tune GA hyperparameters (population size, mutation rate, etc.)
- ✅ Design multi-objective fitness functions
- ✅ Run and visualize GA training
- ✅ Understand when GAs are better than gradient descent

## Key Concepts

### What is a Genetic Algorithm?

**Analogy:** Evolution in nature
- **Population:** Group of candidate solutions
- **Chromosome:** Encoding of one solution (our case: control sequence)
- **Fitness:** How good is this solution? (goal distance, collision, smoothness)
- **Selection:** "Survival of the fittest" - pick good solutions
- **Crossover:** "Reproduction" - combine two parents
- **Mutation:** "Random changes" - explore new solutions
- **Elitism:** Keep the best solutions unchanged

### Why GAs for Robot Navigation?

1. **No gradient needed** - Works with discrete, non-differentiable objectives
2. **Multi-objective** - Balance safety, efficiency, smoothness
3. **Exploration** - Finds diverse solutions
4. **Interpretable** - Can inspect evolved trajectories

---

## Hands-On Exercises

### Exercise 1: Run GA Training (30 min)

```bash
# Run GA on 10 scenarios (quick test)
python training/train_ga.py \
  --config training/config/ga_config.yaml \
  --output results/test_10.pkl \
  --num_scenarios 10 \
  --num_workers 4

# Observe output:
# - Generation progress
# - Best fitness per generation
# - Convergence behavior
```

**Questions:**
1. How many generations does it take to converge?
2. What's the final best fitness?
3. Does fitness improve monotonically?

### Exercise 2: Tune Population Size (1 hour)

Create three configs with different population sizes:

```yaml
# ga_config_pop50.yaml
ga:
  population_size: 50
  num_generations: 30
  
# ga_config_pop100.yaml
ga:
  population_size: 100
  num_generations: 30

# ga_config_pop200.yaml
ga:
  population_size: 200
  num_generations: 30
```

Run all three and plot results:

```python
import pickle
import matplotlib.pyplot as plt

# Load results
with open('results/pop50.pkl', 'rb') as f:
    data50 = pickle.load(f)
    
# Extract fitness
fits50 = [t['fitness'] for t in data50]

# Plot comparison
plt.hist(fits50, alpha=0.5, label='Pop=50')
# ... (repeat for 100, 200)
plt.legend()
plt.show()
```

### Exercise 3: Modify Fitness Function (1.5 hours)

Add a new fitness component in `training/ga/fitness.py`:

```python
# Add energy efficiency term
def compute_energy_cost(trajectory):
    """Penalize high accelerations (energy waste)."""
    energy = 0.0
    for i in range(len(trajectory) - 1):
        dv = trajectory[i+1].v_x - trajectory[i].v_x
        energy += abs(dv)
    return energy

# In FitnessEvaluator.evaluate():
energy_cost = compute_energy_cost(trajectory)
fitness -= self.weights.get('energy', 0.1) * energy_cost
```

Test with:
```yaml
fitness_weights:
  goal_distance: 1.0
  collision: 10.0
  smoothness: 0.5
  time_efficiency: 0.3
  energy: 0.2  # New!
```

### Exercise 4: Visualize Evolution (1 hour)

Create `visualize_ga.py`:

```python
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load GA results
with open('results/ga_trajectories.pkl', 'rb') as f:
    data = pickle.load(f)

# Plot fitness distribution
fitnesses = [t['fitness'] for t in data]
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(fitnesses, bins=50)
plt.xlabel('Fitness')
plt.ylabel('Count')
plt.title('Fitness Distribution')

plt.subplot(1, 3, 2)
goal_dists = [t['fitness_components']['goal_distance'] for t in data]
plt.hist(goal_dists, bins=50)
plt.xlabel('Goal Distance (m)')
plt.title('Goal Distance Distribution')

plt.subplot(1, 3, 3)
collisions = [t['fitness_components']['collision'] for t in data]
collision_rate = sum(collisions) / len(collisions)
plt.bar(['No Collision', 'Collision'], 
        [1-collision_rate, collision_rate])
plt.title(f'Collision Rate: {collision_rate*100:.1f}%')

plt.tight_layout()
plt.savefig('ga_analysis.png')
plt.show()
```

### Exercise 5: Implement Custom Operator (2 hours)

Implement **blend crossover** (BLX-α) in `exercises/blx_crossover.py`:

```python
def blx_alpha_crossover(parent1, parent2, alpha=0.5):
    """
    Blend crossover: offspring genes sampled from expanded range.
    
    For each gene:
    - min_val = min(p1_gene, p2_gene)
    - max_val = max(p1_gene, p2_gene)
    - range_ext = (max_val - min_val) * alpha
    - sample from [min_val - range_ext, max_val + range_ext]
    
    Args:
        parent1, parent2: Chromosome objects
        alpha: Blend factor (default 0.5)
    
    Returns:
        offspring: New Chromosome
    """
    # TODO: Implement this
    pass

# Test it
if __name__ == "__main__":
    # Create test chromosomes
    velocity_limits = {'max_v_x': 1.0, 'min_v_x': -0.5, ...}
    p1 = Chromosome(20, velocity_limits)
    p2 = Chromosome(20, velocity_limits)
    p1.genes.fill(0.5)
    p2.genes.fill(-0.3)
    
    offspring = blx_alpha_crossover(p1, p2, alpha=0.5)
    
    # Check offspring genes are in blended range
    print(f"P1 genes: {p1.genes[0]}")
    print(f"P2 genes: {p2.genes[0]}")
    print(f"Offspring genes: {offspring.genes[0]}")
    # Should be in range [-0.7, 0.9] approximately
```

See `exercises/solutions/blx_crossover_solution.py` after attempting.

---

## Code Walkthrough

### Key Files to Study

1. **training/ga/chromosome.py** (276 lines)
   - How control sequences are encoded
   - Velocity limit clamping
   - Comparison operators for sorting

2. **training/ga/fitness.py** (345 lines)
   - Multi-objective fitness calculation
   - Parallel evaluation with multiprocessing
   - Fitness component tracking

3. **training/ga/evolution.py** (343 lines)
   - Main GA loop
   - Population initialization
   - Generation evolution
   - Statistics tracking

4. **training/ga/operators.py** (335 lines)
   - Tournament selection
   - Uniform/single-point crossover
   - Gaussian/uniform mutation
   - Elitism

### Trace an Evolution

Follow the code flow:
1. `train_ga.py:300` - Call `ga.run()`
2. `evolution.py:150` - Initialize population
3. `evolution.py:158` - For each generation:
   - `evolution.py:102` - Evaluate fitness (parallel)
   - `evolution.py:107` - Select elites
   - `evolution.py:110-124` - Generate offspring
4. `fitness.py:151` - Parallel evaluation setup
5. `fitness.py:120` - Worker function
6. `fitness.py:38` - Evaluate single chromosome
7. `simulator/environment.py:X` - Simulate trajectory
8. Back to `evolution.py:163` - Track best

---

## Quiz

1. **What is a chromosome in this project?**
   a) DNA sequence
   b) Control sequence (20 steps of v_x, v_y, omega)
   c) Robot position
   d) Fitness value

2. **Why use multiprocessing for fitness evaluation?**
   a) Faster than threading
   b) Fitness calculations are independent
   c) Python GIL limitations
   d) All of the above

3. **What does elitism do?**
   a) Select parents for crossover
   b) Preserve best solutions unchanged
   c) Increase mutation rate
   d) Remove worst solutions

4. **Higher mutation rate causes:**
   a) More exploration, slower convergence
   b) Less exploration, faster convergence
   c) No effect
   d) Better fitness always

5. **What's the main advantage of GAs over gradient descent?**
   a) Always faster
   b) Works without gradients
   c) Finds global optimum always
   d) Uses less memory

<details>
<summary><b>Show Answers</b></summary>

1. b) Control sequence
2. d) All of the above
3. b) Preserve best solutions
4. a) More exploration, slower convergence
5. b) Works without gradients (e.g., collision is non-differentiable)
</details>

---

## ✅ Checklist

- [ ] Understand GA concepts (selection, crossover, mutation)
- [ ] Run GA training successfully
- [ ] Tune hyperparameters and observe effects
- [ ] Implement custom fitness component
- [ ] Visualize training results
- [ ] Quiz score 80%+

---

## 📚 Resources

- [Genetic Algorithms Wikipedia](https://en.wikipedia.org/wiki/Genetic_algorithm)
- [Introduction to Evolutionary Computing](https://link.springer.com/book/10.1007/978-3-662-44874-8) (textbook)
- [DEAP Library](https://deap.readthedocs.io/) (Python GA framework)

---

## 🎉 Next Steps

You now understand genetic algorithms! Time to learn how to distill them into neural networks.

**→ [Continue to Module 03: Neural Networks](../03-neural-networks/)**
