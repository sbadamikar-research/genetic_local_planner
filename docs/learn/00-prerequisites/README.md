# Module 00: Prerequisites & Setup

**Estimated Time:** 1 day (6-8 hours)

## 🎯 Learning Objectives

By the end of this module, you will:
- ✅ Set up your development environment (Python, Docker, Git)
- ✅ Understand the tech stack and why each tool is used
- ✅ Verify all installations work correctly
- ✅ Review Python and C++ fundamentals needed for this project
- ✅ Clone and explore the project repository

---

## 📋 Table of Contents

1. [Tech Stack Overview](#tech-stack-overview)
2. [Installation Guide](#installation-guide)
3. [Python Fundamentals Review](#python-fundamentals-review)
4. [C++ Basics Review](#c-basics-review)
5. [Development Tools](#development-tools)
6. [Exercises](#exercises)
7. [Quiz](#quiz)

---

## 1. Tech Stack Overview

### Why These Technologies?

| Technology | Purpose | Why We Need It |
|------------|---------|----------------|
| **Python 3.10** | Training pipeline | Fast prototyping, rich ML ecosystem (PyTorch) |
| **C++17** | Deployment | Real-time performance (10-20 Hz), ROS integration |
| **PyTorch** | Neural networks | Industry-standard, great for research |
| **ONNX** | Model export | Cross-platform inference (Python → C++) |
| **Docker** | Development environment | Reproducibility, isolated ROS installations |
| **ROS1/ROS2** | Robotics middleware | Industry-standard for robot software |
| **Git** | Version control | Collaboration, track changes |

### The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING (Python)                        │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐            │
│  │ Genetic  │ →   │  Neural  │  →  │  ONNX    │            │
│  │Algorithm │     │ Network  │     │ Export   │            │
│  └──────────┘     └──────────┘     └──────────┘            │
│       Host Machine (Linux/Mac/Windows)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               DEPLOYMENT (C++ in Docker)                    │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐            │
│  │  ONNX    │  →  │   ROS    │  →  │  Robot   │            │
│  │ Runtime  │     │ Plugin   │     │ Control  │            │
│  └──────────┘     └──────────┘     └──────────┘            │
│       Docker Container (ROS1 or ROS2)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Installation Guide

### Step 1: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y git curl build-essential
```

**macOS:**
```bash
brew install git
```

**Windows:**
- Install Git from https://git-scm.com/
- Install WSL2 for best experience

### Step 2: Install Miniconda

**Download and install:**
```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Restart terminal or:
source ~/.bashrc
```

**Verify:**
```bash
conda --version
# Should show: conda 23.x.x or newer
```

### Step 3: Clone Repository

```bash
# Navigate to desired location
cd ~/projects  # or wherever you want

# Clone (replace with actual URL)
git clone <repo-url> plan_ga
cd plan_ga

# Check status
git status
git log --oneline -10
```

### Step 4: Create Python Environment

```bash
# Create environment from file
conda env create -f environment.yml

# Activate
conda activate plan_ga

# Verify installation
python -c "import torch, numpy, yaml; print('✓ All imports successful')"
```

**Troubleshooting:**
- If `torch` fails: Check CUDA version or use CPU-only version
- If `yaml` fails: `conda install pyyaml`

### Step 5: Install Docker

**Ubuntu:**
```bash
# Add Docker repository
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group (avoid sudo)
sudo usermod -aG docker $USER

# Log out and back in, then test:
docker run hello-world
```

**macOS:**
- Download Docker Desktop from https://www.docker.com/products/docker-desktop

**Windows (WSL2):**
- Install Docker Desktop with WSL2 backend

**Verify:**
```bash
docker --version
docker ps  # Should not require sudo
```

### Step 6: Verify Everything Works

```bash
# From plan_ga directory
cd /path/to/plan_ga

# Test Python environment
conda activate plan_ga
python -c "import training.ga as ga; print('✓ GA module loads')"

# Test Docker
docker run hello-world

# Check Git
git status

# You're ready! ✨
```

---

## 3. Python Fundamentals Review

### Quick Self-Check

Can you understand this code?

```python
import numpy as np

class Chromosome:
    def __init__(self, genes):
        self.genes = np.array(genes)
        self.fitness = -np.inf

    def mutate(self, rate=0.1):
        mask = np.random.random(self.genes.shape) < rate
        self.genes[mask] += np.random.randn(mask.sum()) * 0.1

    def __lt__(self, other):
        return self.fitness < other.fitness

# Usage
population = [Chromosome(np.random.randn(10)) for _ in range(100)]
population.sort()
best = population[-1]
```

**If yes:** You're ready! Skip to exercises.
**If no:** Review the concepts below.

### Key Concepts Needed

#### 1. NumPy Arrays
```python
import numpy as np

# Creating arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.random.randn(100, 10)  # 100x10 random matrix

# Indexing and slicing
first_row = matrix[0, :]    # First row
column = matrix[:, 5]       # 6th column

# Broadcasting
matrix_scaled = matrix * 2.0  # Multiply all elements

# Boolean masking
mask = matrix > 0
positive_values = matrix[mask]
```

#### 2. Classes and Objects
```python
class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 0.5

    def move(self, direction):
        if direction == 'forward':
            self.y += self.speed
        elif direction == 'backward':
            self.y -= self.speed

    def __repr__(self):
        return f"Robot(x={self.x}, y={self.y})"

# Usage
robot = Robot(0, 0)
robot.move('forward')
print(robot)  # Robot(x=0, y=0.5)
```

#### 3. List Comprehensions
```python
# Traditional loop
squares = []
for i in range(10):
    squares.append(i**2)

# List comprehension (preferred)
squares = [i**2 for i in range(10)]

# With condition
even_squares = [i**2 for i in range(10) if i % 2 == 0]
```

#### 4. File I/O
```python
import pickle
import yaml

# Reading YAML config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Saving/loading pickle
data = {'trajectories': [...], 'fitness': [...]}
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)

with open('data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
```

#### 5. Multiprocessing (Preview)
```python
from multiprocessing import Pool

def compute_fitness(chromosome):
    # Expensive computation
    return sum(chromosome**2)

# Parallel evaluation
chromosomes = [np.random.randn(100) for _ in range(1000)]

with Pool(processes=8) as pool:
    fitnesses = pool.map(compute_fitness, chromosomes)
```

**Resources:**
- NumPy tutorial: https://numpy.org/doc/stable/user/quickstart.html
- Python classes: https://docs.python.org/3/tutorial/classes.html

---

## 4. C++ Basics Review

You don't need to be a C++ expert, but you should recognize these patterns:

### Quick Self-Check

Can you understand this code?

```cpp
#include <vector>
#include <memory>
#include <iostream>

class Planner {
public:
    Planner(int steps) : num_steps_(steps) {
        trajectory_.resize(num_steps_);
    }

    void computeTrajectory(const std::vector<double>& controls) {
        for (size_t i = 0; i < controls.size(); ++i) {
            trajectory_[i] = controls[i] * 2.0;
        }
    }

    const std::vector<double>& getTrajectory() const {
        return trajectory_;
    }

private:
    int num_steps_;
    std::vector<double> trajectory_;
};

int main() {
    auto planner = std::make_unique<Planner>(20);
    std::vector<double> controls = {0.5, 0.6, 0.7};
    planner->computeTrajectory(controls);
    return 0;
}
```

**If yes:** You're ready!
**If no:** Don't worry, we'll explain C++ code when we encounter it.

### Key C++ Concepts (Brief)

- **Classes:** Similar to Python, but with `public`/`private` access
- **Vectors:** Like Python lists, `std::vector<T>`
- **Smart Pointers:** `std::unique_ptr`, `std::shared_ptr` for memory management
- **const:** Means "read-only", used for safety
- **Templates:** Generic code, like `template <typename T>`

**You'll learn by doing!** Don't stress about C++ now.

---

## 5. Development Tools

### VS Code (Recommended)

**Install Extensions:**
1. Python (Microsoft)
2. C/C++ (Microsoft)
3. Docker
4. Remote - Containers
5. Git Graph

**Useful Shortcuts:**
- `Ctrl+P`: Quick file open
- `Ctrl+Shift+P`: Command palette
- `F12`: Go to definition
- `Ctrl+```: Toggle terminal

### Alternative: PyCharm / CLion

Both work great! Use what you're comfortable with.

### Git Basics

```bash
# Check status
git status

# Create branch for experiments
git checkout -b my-experiment

# Commit changes
git add file.py
git commit -m "feat: add new fitness function"

# View history
git log --oneline --graph

# Switch back to main
git checkout main
```

---

## 6. Exercises

### Exercise 1: Environment Verification (20 min)

Run these commands and paste the output:

```bash
# Python version
python --version

# Conda environment
conda env list

# Package versions
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# Docker version
docker --version

# Git version
git --version
```

**Expected output:**
- Python 3.10 or newer
- torch 2.x
- numpy 1.2x
- Docker 20.x or newer

### Exercise 2: Test Imports (10 min)

Create `test_imports.py`:

```python
#!/usr/bin/env python3
"""Test all required imports for the course."""

try:
    import torch
    import numpy as np
    import yaml
    import pickle
    from pathlib import Path

    print("✓ All basic imports successful!")

    # Test torch
    x = torch.randn(10, 10)
    print(f"✓ PyTorch tensor created: {x.shape}")

    # Test numpy
    arr = np.random.rand(5, 5)
    print(f"✓ NumPy array created: {arr.shape}")

    # Test yaml
    config = {'test': 123, 'nested': {'value': 456}}
    yaml_str = yaml.dump(config)
    print(f"✓ YAML serialization works")

    print("\n🎉 All tests passed! You're ready to start learning!")

except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Please check your conda environment")
```

Run it:
```bash
python test_imports.py
```

### Exercise 3: Git Exploration (15 min)

Explore the repository:

```bash
# Count lines of code
find . -name "*.py" | xargs wc -l
find . -name "*.cpp" -o -name "*.hpp" | xargs wc -l

# Find all GA-related files
find . -name "*ga*" -o -name "*genetic*"

# View recent commits
git log --oneline -20

# See what files changed in last commit
git show --name-only HEAD
```

### Exercise 4: Python Warm-Up (30 min)

Create `warmup.py` and implement these functions:

```python
import numpy as np

def fitness_function(x):
    """
    Compute fitness for a 1D optimization problem.
    Goal: Find x that maximizes f(x) = -(x - 5)^2 + 10

    Args:
        x: float, candidate solution

    Returns:
        fitness: float, higher is better
    """
    # TODO: Implement this
    pass

def create_population(size, bounds):
    """
    Create random population within bounds.

    Args:
        size: int, number of individuals
        bounds: tuple (min, max)

    Returns:
        population: np.ndarray of shape (size,)
    """
    # TODO: Implement using np.random.uniform
    pass

def tournament_selection(population, fitnesses, k=3):
    """
    Select best individual from k random candidates.

    Args:
        population: np.ndarray, all individuals
        fitnesses: np.ndarray, fitness values
        k: int, tournament size

    Returns:
        winner: float, selected individual
    """
    # TODO: Implement
    # Hint: Use np.random.choice for random indices
    pass

# Test your implementations
if __name__ == "__main__":
    # Test fitness
    assert fitness_function(5.0) == 10.0, "Fitness at optimum should be 10"
    assert fitness_function(0.0) < 10.0, "Fitness away from optimum should be less"

    # Test population
    pop = create_population(100, (-10, 10))
    assert pop.shape == (100,), "Population shape incorrect"
    assert np.all((pop >= -10) & (pop <= 10)), "Population out of bounds"

    # Test selection
    pop = np.array([1, 2, 3, 4, 5])
    fits = np.array([1, 2, 3, 4, 5])
    winner = tournament_selection(pop, fits, k=3)
    assert winner >= 3, "Tournament should favor high fitness"

    print("✓ All tests passed!")
```

**Solution:** Check `exercises/solutions/warmup_solution.py` after attempting.

---

## 7. Quiz

### Question 1
Why do we use Python for training but C++ for deployment?

a) Python is faster
b) C++ has better ML libraries
c) Python is easier to prototype, C++ is faster at runtime
d) C++ doesn't support neural networks

### Question 2
What is ONNX used for in this project?

a) Training neural networks
b) Exporting models from Python to C++
c) Running genetic algorithms
d) Managing Docker containers

### Question 3
What does `conda activate plan_ga` do?

a) Installs packages
b) Activates the Python environment for this project
c) Starts Docker containers
d) Runs the training script

### Question 4
In the code `arr = np.random.randn(100, 3)`, what is the shape?

a) 3 rows, 100 columns
b) 100 rows, 3 columns
c) 1D array of 103 elements
d) 3D array

### Question 5
What does `git checkout -b my-branch` do?

a) Delete a branch
b) Create and switch to a new branch
c) Merge branches
d) Commit changes

<details>
<summary><b>Show Answers</b></summary>

1. c) Python is easier to prototype, C++ is faster at runtime
2. b) Exporting models from Python to C++
3. b) Activates the Python environment for this project
4. b) 100 rows, 3 columns
5. b) Create and switch to a new branch
</details>

---

## ✅ Checklist

Before moving to Module 01, ensure:

- [ ] Python environment installed and activated
- [ ] Docker installed and working (no sudo needed)
- [ ] Git working and repository cloned
- [ ] All imports in Exercise 2 successful
- [ ] Comfortable with NumPy arrays and Python classes
- [ ] Quiz completed (80%+ correct)

---

## 📚 Additional Resources

### Python
- [Python Official Tutorial](https://docs.python.org/3/tutorial/)
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [Real Python](https://realpython.com/) - Excellent tutorials

### Git
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [Oh Shit, Git!](https://ohshitgit.com/) - Fix common mistakes

### Docker
- [Docker Getting Started](https://docs.docker.com/get-started/)

---

## 🎉 Next Steps

Great job setting up your environment! You're ready to dive into the project.

**→ [Continue to Module 01: Project Architecture](../01-project-architecture/)**
