# 1D Jump-Diffusion Model Comparison

A numerical comparison of Finite Difference (FD) and Physics-Informed Neural Network (PINN) methods for solving 1D Jump-Diffusion partial integro-differential equations (PIDEs) with application to probability of default calculations.

## Overview

This project implements and compares two approaches for solving the 1D Jump-Diffusion PIDE:
- **Finite Difference Method**: Backward Time Central Space (BTCS) implicit scheme
- **Physics-Informed Neural Network**: Deep learning approach using PyTorch

The underlying stochastic process combines:
- Ornstein-Uhlenbeck mean-reverting diffusion
- Poisson jump process with normal-distributed jump sizes

## Core Files

### Main Scripts
- `jump_diffusion_solver.py` - Standalone FD solver with visualization
- `compare_methods.py` - Comparative analysis of FD vs PINN methods
- `model.py` - PINN neural network architecture (OU_PINN class)
- `config.py` - Shared configuration parameters

### Data Files
- `levy_ou_pinn_model.pth` - Pre-trained PINN model weights
- `probability_of_default_results.json` - FD solver results cache
- `method_comparison_results.json` - Detailed comparison metrics

### Output Directory
- `plots/` - Interactive HTML visualizations from both methods

## Usage

### Run Individual Methods
```bash
# Finite Difference solver
python jump_diffusion_solver.py

# Generates: plots/probability_of_default_pinn_config.html
```

### Run Comparison Analysis
```bash
# Compare FD vs PINN
python compare_methods.py

# Generates: 
#   - plots/comparison_pinn_solution.html
#   - plots/comparison_fd_solution.html  
#   - plots/comparison_absolute_diff.html
#   - plots/comparison_relative_diff.html
#   - plots/comparison_side_by_side.html
#   - method_comparison_results.json
```

## Model Parameters

- Domain: t ∈ [0, 1], x ∈ [-0.5, 2.0]
- PDE coefficients: κ=0.3, θ=0.0, σ=0.2
- Jump parameters: λ=1.0, σ_jump=0.2
- Grid resolution: 1000 × 1000 (spatial × temporal)

## Key Results

The comparison analysis shows:
- PINN inference: ~0.03 seconds
- FD computation: ~98 seconds
- Speedup factor: ~3,600x (PINN faster)
- Solution correlation: >97%
- RMSE: ~0.08

## Dependencies

- numpy, scipy
- matplotlib, plotly
- torch (for PINN)
- tqdm (progress bars)

## Output

All methods generate interactive 3D Plotly visualizations showing the probability of default evolution over time and space. Results are saved as standalone HTML files for analysis and presentation.