# PPO for Combinatorics

This repository contains implementations for solving combinatorial problems using reinforcement learning approaches, specifically Cross-Entropy Method (CEM) and Proximal Policy Optimization (PPO).

## Project Structure

### Main Branch
- **`demo_with_pytorch.py`** - PyTorch implementation of the Cross-Entropy Method
- **`analyzed_demo.py`** - Keras/TensorFlow implementation of the Cross-Entropy Method 
- **`reward_plot.py`** - Utility for plotting reward comparisons across different test runs

### Gheb6-gym Branch
- **`gym/`** folder containing PPO implementation using OpenAI Gym environment:
  - `graph_gym_env.py` - Custom Gymnasium environment for graph generation
  - `main_ppo.py` - PPO training script with custom callbacks
  - `ppo_mean_rewards.py` - Plotting script for mean rewards
  - `ppo_best_rewards.py` - Plotting script for best rewards

## Problem Description

The code tackles **Conjecture 2.1** from the paper "Constructions in combinatorics via neural networks and LP solvers" by A Z Wagner. The goal is to find graphs that minimize λ₁ + μ (largest eigenvalue plus matching number) relative to √(N-1) + 1, where finding a positive score indicates a counterexample to the conjecture.

## Approaches

### 1. Cross-Entropy Method (CEM)
- **File**: `demo_with_pytorch.py` (PyTorch) or `analyzed_demo.py` (Keras)
- Uses evolutionary approach to generate graphs
- Tracks elite graphs and their frequencies
- Generates heatmaps showing edge probabilities

### 2. Proximal Policy Optimization (PPO) 
- **Branch**: `Gheb6-gym`
- **File**: `gym/main_ppo.py`
- Uses reinforcement learning with custom Gym environment
- Sequential edge decision making
- Real-time counterexample detection

## Software Requirements

### For CEM Implementation
- Python 3.6+ (for `demo_with_pytorch.py`)
- OR Python 3.6.3 + TensorFlow 1.14.0 + Keras 2.3.1 (for `analyzed_demo.py`)
- PyTorch (for `demo_with_pytorch.py`)
- NetworkX
- NumPy
- Matplotlib

### For PPO Implementation (Gheb6-gym branch)
- Python 3.7+
- Stable-Baselines3
- Gymnasium
- PyTorch
- NetworkX
- NumPy
- Matplotlib

## Key Features

### Cross-Entropy Method
- ✅ N×N heatmap matrix showing average edge probabilities
- ✅ Matplotlib plotting for visualization
- ✅ Dictionary tracking super graphs and frequencies
- ✅ Continues for 1000 steps after counterexample found
- ✅ CSV output with comprehensive statistics
- ✅ Automatic termination on convergence

### PPO Method
- ✅ Custom Gymnasium environment for graph construction
- ✅ Sequential decision making (edge by edge)
- ✅ Two game modes: starting from empty graph (adding edges) or full graph (removing edges)
- ✅ Real-time scoring and counterexample detection
- ✅ Custom callbacks for tracking progress
- ✅ Graph visualization and saving
- ✅ Connectivity constraint handling

## Output Files

### CEM Output
- `results/output.csv` - Training statistics and metrics
- `results/dictionaries/` - Graph frequency dictionaries
- `results/heatmap/` - Edge probability heatmaps
- `results/elite_graph/` - Elite graph sequences
- `saved_graphs/` - Counterexample graphs (if found)

### PPO Output
- `ppo_graph_gym_results/` - Training results and rewards
- `saved_graphs/` - Best scoring graphs
- `ppo_graph_gym_models/` - Trained models

## How to Cite

If you use this code in your research, please cite it as:

```bibtex
@misc{righi2025ppo,
  author       = {Gabriele Righi and Francesco Morandin},
  title        = {{PPO\_for\_combinatorics}},
  year         = {2025},
  howpublished = {\url{https://github.com/Gheb6/PPO_for_combinatorics}}
}
```

## References

Based on the paper: "Constructions in combinatorics via neural networks and LP solvers" by A Z Wagner
- arXiv: https://arxiv.org/abs/2104.14516

## Notes

- The PPO implementation is available in the `Gheb6-gym` branch
- GPU acceleration recommended for faster training
- Both approaches aim to find counterexamples to graph theory conjectures




