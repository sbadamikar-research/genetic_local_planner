# Learning Course: GA-Based ROS Local Planner

**A hands-on course for software engineering students**

Welcome! This course teaches you how to build an intelligent robot navigation system from scratch using genetic algorithms, neural networks, and robotics middleware.

---

## 🎯 What You'll Build

By the end of this course, you'll understand how to:
- Evolve optimal robot trajectories using **genetic algorithms**
- Distill GA solutions into fast **neural networks**
- Deploy models to **C++ ROS plugins** using **ONNX**
- Use **Docker** for reproducible development environments
- Integrate with the **ROS navigation stack**

You'll work with a complete, production-ready codebase (~5000 lines).

---

## 📚 Course Structure

Each module takes **~1 day** to complete (6-8 hours of focused work).

| Module | Topic | What You'll Learn | Time |
|--------|-------|-------------------|------|
| [00](./00-prerequisites/) | **Prerequisites & Setup** | Environment setup, Python/C++ basics review | 1 day |
| [01](./01-project-architecture/) | **Project Architecture** | Codebase walkthrough, design decisions | 1 day |
| [02](./02-genetic-algorithms/) | **Genetic Algorithms** | Evolution, fitness functions, tuning | 1 day |
| [03](./03-neural-networks/) | **Neural Networks** | CNN for costmaps, MLP fusion, training | 1 day |
| [04](./04-onnx-deployment/) | **ONNX Deployment** | Model export, cross-platform inference | 1 day |
| [05](./05-cpp-integration/) | **C++ Integration** | Plugin architecture, ONNX Runtime in C++ | 1 day |
| [06](./06-ros-fundamentals/) | **ROS Fundamentals** | Nodes, topics, navigation stack | 1 day |
| [07](./07-docker-containers/) | **Docker Containers** | Containerization, dev environments | 1 day |
| [08](./08-training-pipeline/) | **Training Pipeline** | End-to-end training, hyperparameter tuning | 1 day |
| [09](./09-capstone/) | **Capstone Projects** | Extensions and improvements | 2-3 days |

**Total:** 11-12 days of focused learning

---

## 🧑‍🎓 Prerequisites

### What You Should Know
- **Python:** Functions, classes, NumPy basics
- **Data Structures:** Arrays, lists, dictionaries, basic algorithms
- **Basic ML Concepts:** What is training/testing, loss functions (high-level)
- **Command Line:** Navigate directories, run scripts

### What You'll Learn Here
- Advanced Python (multiprocessing, decorators, generators)
- Machine learning (GAs, CNNs, supervised learning)
- C++ integration with Python
- ROS robotics framework
- Docker containerization
- Software architecture for robotics

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone <repo-url>
cd plan_ga
```

### 2. Install Python Environment
```bash
conda env create -f environment.yml
conda activate plan_ga
```

### 3. Install Docker
Follow instructions at https://docs.docker.com/get-docker/

### 4. Verify Installation
```bash
# Test Python environment
python -c "import torch, numpy; print('Python OK')"

# Test Docker
docker run hello-world
```

### 5. Start Learning!
Begin with [Module 00: Prerequisites](./00-prerequisites/)

---

## 📖 How to Use This Course

### Learning Path

**For Each Module:**
1. **Read the concepts** - Understand the "why" and "how"
2. **Walk through code** - See real implementations
3. **Do exercises** - Hands-on practice (don't skip!)
4. **Take the quiz** - Self-assessment
5. **Check solutions** - After attempting exercises

**Tips:**
- **Type code, don't copy-paste** - Builds muscle memory
- **Experiment freely** - Break things, see what happens
- **Use print statements** - Understand data flow
- **Google is your friend** - Look up unfamiliar concepts
- **Ask questions** - Use issues/discussions on GitHub

### Exercise Types

**🏃 Running Experiments**
```bash
# Example: Test different population sizes
python training/train_ga.py --config configs/ga_config_pop50.yaml
python training/train_ga.py --config configs/ga_config_pop100.yaml
# Compare results!
```

**⚙️ Parameter Tuning**
```yaml
# Modify training/config/ga_config.yaml
mutation_rate: 0.1  # Try 0.05, 0.15, 0.3
crossover_rate: 0.8  # Try 0.5, 0.9
# Observe impact on convergence
```

**🔧 Code Modifications**
```python
# Add new fitness component in training/ga/fitness.py
def compute_energy_cost(trajectory):
    # Your code here
    pass
```

**🐛 Debugging Challenges**
```python
# Fix intentional bugs in exercise files
# exercises/broken_fitness.py
def fitness(x):
    return -abs(x - 5)  # Bug: Why is this wrong?
```

**📊 Visualization**
```python
# Plot training curves
import matplotlib.pyplot as plt
plt.plot(fitness_history)
plt.show()
```

---

## 🎓 Learning Outcomes

After completing this course, you will be able to:

### Technical Skills
- ✅ Implement genetic algorithms for optimization problems
- ✅ Design and train neural networks in PyTorch
- ✅ Export models to ONNX and integrate with C++
- ✅ Build ROS navigation plugins from scratch
- ✅ Use Docker for reproducible development
- ✅ Profile and optimize inference performance

### Conceptual Understanding
- ✅ When to use evolutionary algorithms vs gradient descent
- ✅ How to design fitness functions for multi-objective problems
- ✅ Model distillation and knowledge transfer
- ✅ Trade-offs in robotics: accuracy vs speed vs safety
- ✅ Software architecture for modular robotics systems

### Engineering Practices
- ✅ Read and navigate large codebases (~5K lines)
- ✅ Debug across Python/C++ boundaries
- ✅ Version control with Git (branching, merging)
- ✅ Write reproducible experiments
- ✅ Document code for others

---

## 🚀 Project Highlights

### Real-World Complexity
- **~5000 lines of code** across Python and C++
- **Multi-language integration** (Python ↔ C++ via ONNX)
- **Production-ready** ROS1 and ROS2 plugins
- **Parallel processing** for fast training
- **Containerized** development environment

### Technologies You'll Master
- **Python:** PyTorch, NumPy, multiprocessing, YAML configs
- **C++:** Modern C++17, templates, smart pointers
- **ML:** Genetic algorithms, CNNs, supervised learning
- **Robotics:** ROS1/ROS2, navigation stack, costmaps
- **DevOps:** Docker, VS Code Dev Containers
- **Tools:** Git, CMake, ONNX Runtime, pluginlib

---

## 📊 Assessment

### Module Quizzes
Each module has a short quiz (5-10 questions) to test understanding.

### Coding Exercises
Hands-on exercises range from:
- **Simple:** Modify a parameter, observe output
- **Medium:** Implement a function using pseudocode
- **Hard:** Debug complex multi-file issues
- **Challenge:** Extend the system with new features

### Final Capstone
Choose one of several capstone projects:
1. **Visualization Dashboard** - Real-time GA training visualization
2. **Adaptive Parameters** - Implement self-adjusting mutation rates
3. **Multi-Robot** - Extend to multi-agent coordination
4. **Sim-to-Real** - Deploy on physical robot (if available)

---

## 🤝 Getting Help

### Common Issues
Check [FAQ](./FAQ.md) for solutions to common problems.

### Debugging Tips
1. **Read error messages carefully** - They usually tell you what's wrong
2. **Use print debugging** - Add `print()` statements liberally
3. **Check file paths** - Absolute vs relative paths
4. **Verify installations** - `import` statements failing?
5. **Google the error** - Someone has seen it before

### Community
- **GitHub Issues:** Report bugs or ask questions
- **Discussions:** Share your solutions and learn from others

---

## 📝 Note on Learning Style

This course is designed for **active learning**:
- 📖 **20% Reading** - Conceptual understanding
- 💻 **80% Coding** - Hands-on practice

**Don't just read!** Type out every example, run every experiment, and complete every exercise. That's where real learning happens.

---

## 🎉 Let's Get Started!

Ready to build an intelligent robot navigation system?

**→ [Start with Module 00: Prerequisites](./00-prerequisites/)**

Good luck, and have fun! 🤖🚀
