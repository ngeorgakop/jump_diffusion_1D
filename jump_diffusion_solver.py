#!/usr/bin/env python3
"""
1D Jump-Diffusion PIDE Solver - Probability of Default
======================================================

This script implements a numerical solver for a 1D jump-diffusion process using
finite difference methods. The solver tracks execution time and provides 3D
visualization of the probability of default evolution.

Mathematical Model:
∂φ/∂t = (σ²/2)∂²φ/∂x² + κ(θ-x)∂φ/∂x + λ∫[φ(x+y,t) - φ(x,t)]f(y)dy

Output: Probability of Default (PD) = 1 - Survival Probability

Author: Refactored from Jupyter notebook
"""

import numpy as np
import time
from math import *
import scipy.stats
from scipy.integrate import quad
import plotly.offline as pyoff
import plotly.graph_objects as go
from tqdm import tqdm
import warnings
import json

warnings.filterwarnings("ignore")


class JumpDiffusionSolver:
    """
    1D Jump-Diffusion PIDE Solver using implicit finite difference scheme.
    Calculates Probability of Default (PD) evolution over time and space.
    """
    
    def __init__(self, spatial_params, temporal_params, model_params):
        """
        Initialize the solver with parameters.
        
        Args:
            spatial_params (dict): Spatial grid parameters
            temporal_params (dict): Temporal grid parameters  
            model_params (dict): Model parameters
        """
        # Spatial grid setup
        self.x0 = spatial_params['x0']
        self.xn = spatial_params['xn'] 
        self.L = spatial_params['L']
        self.xsteps = spatial_params['xsteps']
        self.dx = (self.xn - self.x0) / self.xsteps
        self.x = np.arange(self.x0, self.xn + self.dx, self.dx)
        
        # Temporal grid setup
        self.t0 = temporal_params['t0']
        self.tn = temporal_params['tn']
        self.tsteps = temporal_params['tsteps']
        self.dt = (self.tn - self.t0) / self.tsteps
        self.t = np.arange(self.t0, self.tn + self.dt, self.dt)
        
        # Model parameters
        self.sigma = model_params['sigma']
        self.kappa = model_params['kappa']
        self.theta = model_params['theta']
        self.mu_jump = model_params['mu_jump']
        self.sigma_jump = model_params['sigma_jump']
        self.rate = model_params['rate']
        self.N = model_params['N']
        
        # Initialize solution matrix
        self.phi_matrix = None
        self.solve_time = None
        
        print(f"Grid setup: dx = {self.dx:.6f}, dt = {self.dt:.6f}")
        print(f"Stability ratio r = dt/dx² = {self.dt/(self.dx**2):.6f}")
    
    def jump_distribution_full(self, x_value, mu, sigma_jump, step_x):
        """
        Calculate jump distribution for integral terms using numerical integration.
        """
        lower = x_value - step_x / 2.0
        upper = x_value + step_x / 2.0
        
        def normal_distribution_function(x):
            return scipy.stats.norm.pdf(x, mu, sigma_jump)
        
        res, err = quad(normal_distribution_function, lower, upper)
        return res
    
    def a_coeff(self, x_value, sigma, kappa, theta):
        """Coefficient for BTCS discretization scheme."""
        return (sigma**2) / (self.dx**2)
    
    def b_coeff(self, x_value, sigma, kappa, theta):
        """Coefficient for BTCS discretization scheme."""
        return 0.5 * (sigma**2) * (1/self.dx**2) + 0.5 * (1/self.dx) * kappa * (theta - x_value)
    
    def c_coeff(self, x_value, sigma, kappa, theta):
        """Coefficient for BTCS discretization scheme."""
        return 0.5 * (sigma**2) * (1/self.dx**2) - 0.5 * (1/self.dx) * kappa * (theta - x_value)
    
    def M_matrix(self, x, sigma, kappa, theta):
        """
        Construct M matrix containing coefficients for solution at time t+1 
        in the implicit scheme.
        """
        M = np.zeros(shape=(len(x), len(x)))
        
        for i in range(len(x)):
            # Apply boundary conditions only at extended grid boundaries, not in visualization domain
            if x[i] <= self.x0 + 0.05 or x[i] >= self.xn - 0.05:  # Only at extended boundaries
                M[i, i] = 1.0
            else:
                a_coefficient = self.a_coeff(x[i], sigma, kappa, theta)
                b_coefficient = self.b_coeff(x[i], sigma, kappa, theta)
                c_coefficient = self.c_coeff(x[i], sigma, kappa, theta)
                
                if a_coefficient < 0 or b_coefficient < 0 or c_coefficient < 0:
                    print(f'Warning: Negative coefficient at x[{i}] = {x[i]}')
                
                M[i, (i-1):(i+2)] = [-self.dt * c_coefficient, 
                                      1 + self.dt * a_coefficient, 
                                      -self.dt * b_coefficient]
        return M
    
    def jump_integral(self, x, mu, N, sigma_jump, step_x):
        """Calculate integral term in the PIDE."""
        integral_list = []
        positions = np.arange(int(-N/2 + 1), int(N/2), 1)
        
        for j in positions:
            x_jump = step_x * j
            integral = self.jump_distribution_full(x_jump, mu, sigma_jump, step_x)
            integral_list.append(integral)
        
        return sum(integral_list)
    
    def N_matrix(self, x, poisson_rate, N, mu, sigma_jump, step_x):
        """
        Construct matrix containing coefficients for solution at time t 
        in the implicit scheme.
        """
        N_m = np.zeros(shape=(len(x), len(x)))
        positions = np.arange(int(-N/2 + 1), int(N/2), 1)
        jump_result_list = []
        
        for i in range(len(x)):
            integral_list = []
            # Apply jump terms everywhere except at extended grid boundaries
            if x[i] > self.x0 + 0.05 and x[i] < self.xn - 0.05:
                for j in positions:
                    x_jump = step_x * j
                    if (i + j) >= 0 and (i + j) <= (len(x) - 1):
                        integral = self.jump_distribution_full(x_jump, mu, sigma_jump, step_x)
                        integral_list.append(integral)
                        N_m[i, i + j] = poisson_rate * self.dt * integral
                
                jump_integral = sum(integral_list)
                N_m[i, i] = N_m[i, i] + (1 - poisson_rate * self.dt * jump_integral)
                
                jump_result_list.append(jump_integral)
                if 1 - poisson_rate * self.dt * jump_integral < 0:
                    print('Error: Integral approximation violates stability')
        
        if jump_result_list:
            print(f'Jump integral - Min: {min(jump_result_list):.6f}, Max: {max(jump_result_list):.6f}')
        
        return N_m
    
    def check_stability(self):
        """Check stability conditions for the numerical scheme."""
        int_approx = self.jump_integral(self.x, self.mu_jump, self.N, self.sigma_jump, self.dx)
        
        condition1 = self.sigma**2 < np.max(np.abs(self.dx * self.kappa * (self.theta - self.x)))
        condition2 = self.rate * int_approx * self.dt > 1
        
        if condition1 or condition2:
            print('Warning: Stability conditions may not be satisfied')
            if condition1:
                print(f'  - Condition 1 violated: σ² < max|dx·κ·(θ-x)|')
            if condition2:
                print(f'  - Condition 2 violated: λ·∫·dt > 1')
        else:
            print('Stability conditions satisfied')
        
        return not (condition1 or condition2)
    
    def solve(self):
        """
        Solve the jump-diffusion PIDE and track execution time.
        """
        print("Setting up matrices and initial conditions...")
        start_time = time.time()
        
        # Setup matrices
        M = self.M_matrix(self.x, self.sigma, self.kappa, self.theta)
        N_mat = self.N_matrix(self.x, self.rate, self.N, self.mu_jump, self.sigma_jump, self.dx)
        
        # Boundary condition vector for forcing term
        b = np.zeros(len(self.x))
        
        # Initial condition - adjusted for PINN domain  
        self.phi_matrix = np.zeros(shape=(len(self.x), len(self.t)))
        # Initial condition: survival probability = 1 if x > threshold (0.0), 0 otherwise
        initial_threshold = 0.0  # K_threshold from PINN config
        self.phi_matrix[:, 0] = 1.0 * (self.x > initial_threshold)
        
        # Apply boundary conditions only at extended grid boundaries to avoid artifacts in visualization
        # Lower extended boundary (x ≤ -1.25): survival prob = 0 → PD = 1 after conversion
        # Upper extended boundary (x ≥ 2.75): survival prob = 1 → PD = 0 after conversion
        self.phi_matrix[self.x <= self.x0 + 0.05, :] = 0.0  # Low survival at extended lower boundary
        self.phi_matrix[self.x >= self.xn - 0.05, :] = 1.0  # High survival at extended upper boundary
        
        # Check stability
        self.check_stability()
        
        setup_time = time.time()
        print(f"Setup completed in {setup_time - start_time:.2f} seconds")
        
        # Time-stepping loop
        print("Starting time-stepping iterations...")
        M_inv = np.linalg.inv(M)
        
        iteration_start = time.time()
        for time_step in tqdm(range(1, len(self.t)), desc="Time steps"):
            self.phi_matrix[:, time_step] = M_inv.dot(
                N_mat.dot(self.phi_matrix[:, time_step - 1]) + b
            )
        
        # Convert from survival probability to probability of default
        print("Converting to Probability of Default (PD = 1 - Survival)...")
        self.phi_matrix = 1.0 - self.phi_matrix
        
        iteration_end = time.time()
        total_time = iteration_end - start_time
        iteration_time = iteration_end - iteration_start
        
        self.solve_time = {
            'setup_time': setup_time - start_time,
            'iteration_time': iteration_time,
            'total_time': total_time
        }
        
        print(f"\nSolver completed successfully!")
        print(f"Setup time: {self.solve_time['setup_time']:.2f} seconds")
        print(f"Iteration time: {self.solve_time['iteration_time']:.2f} seconds") 
        print(f"Total time: {self.solve_time['total_time']:.2f} seconds")
        
        return self.phi_matrix
    
    def probability_of_default(self, space, time):
        """Get probability of default at given space and time indices."""
        if self.phi_matrix is None:
            raise ValueError("Must solve the system first!")
        return self.phi_matrix[space, time]
    
    # Keep backward compatibility
    def survival(self, space, time):
        """Get survival probability at given space and time indices (deprecated - use probability_of_default)."""
        return 1.0 - self.probability_of_default(space, time)
    
    def plot_solution(self, save_html=True, filename="plots/jump_diffusion_solution.html"):
        """
        Create 3D surface plot of the probability of default solution.
        """
        if self.phi_matrix is None:
            raise ValueError("Must solve the system first!")
        
        print("Creating 3D visualization...")
        
        # Filter to main domain [xmin, L] - matching PINN domain
        xmin = -0.5  # From PINN config
        accept_x = np.where((self.x >= xmin) & (self.x <= self.L))[0]
        
        # Create meshgrid
        X_indices = accept_x
        T_indices = np.arange(0, len(self.t), 1)
        X, T = np.meshgrid(X_indices, T_indices)
        
        # Get probability of default values
        Phi = self.probability_of_default(X, T)
        
        # Convert indices to actual values
        X_values = self.x0 + self.dx * X
        T_values = self.t0 + self.dt * T
        
        # Create 3D surface plot
        data = go.Surface(
            z=Phi, 
            x=X_values, 
            y=T_values,
            colorscale='Viridis',
            name='Probability of Default'
        )
        
        layout = go.Layout(
            title=dict(
                text=f'1D Jump-Diffusion: Probability of Default Evolution<br>'
                     f'<sub>σ={self.sigma}, κ={self.kappa}, θ={self.theta}, '
                     f'λ={self.rate}, σⱼ={self.sigma_jump}</sub>',
                x=0.5
            ),
            scene=dict(
                xaxis_title='Initial Position',
                yaxis_title='Time until Maturity',
                zaxis_title='Probability of Default',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        
        fig = go.Figure(data=[data], layout=layout)
        
        if save_html:
            pyoff.plot(fig, filename=filename, auto_open=True)
            print(f"Plot saved as {filename}")
        
        return fig
    
    def get_summary_stats(self):
        """Get summary statistics of the probability of default solution."""
        if self.phi_matrix is None:
            raise ValueError("Must solve the system first!")
        
        xmin = -0.5  # From PINN config  
        accept_x = np.where((self.x >= xmin) & (self.x <= self.L))[0]
        domain_solution = self.phi_matrix[accept_x, :]
        
        stats = {
            'min_default_prob': np.min(domain_solution),
            'max_default_prob': np.max(domain_solution),
            'mean_default_prob': np.mean(domain_solution),
            'final_time_mean_default_prob': np.mean(domain_solution[:, -1]),
            'grid_points_spatial': len(accept_x),
            'grid_points_temporal': len(self.t),
            'total_grid_points': len(accept_x) * len(self.t)
        }
        
        return stats
    
    def save_comparison_results(self, filename="finite_difference_results.json"):
        """
        Save timing and solution statistics for comparison with PINN approach.
        """
        if self.phi_matrix is None or self.solve_time is None:
            raise ValueError("Must solve the system first!")
        
        stats = self.get_summary_stats()
        
        # Get solution at specific points for detailed comparison
        xmin = -0.5
        accept_x = np.where((self.x >= xmin) & (self.x <= self.L))[0]
        
        # Sample solution at a few key points and times for comparison
        mid_time_idx = len(self.t) // 2
        final_time_idx = -1
        
        # Select a few spatial points for detailed comparison
        x_sample_indices = [accept_x[0], accept_x[len(accept_x)//4], 
                           accept_x[len(accept_x)//2], accept_x[3*len(accept_x)//4], accept_x[-1]]
        
        sample_solutions = {}
        for i, x_idx in enumerate(x_sample_indices):
            x_val = self.x[x_idx]
            sample_solutions[f'x_{x_val:.3f}'] = {
                'initial_pd': float(self.phi_matrix[x_idx, 0]),
                'mid_time_pd': float(self.phi_matrix[x_idx, mid_time_idx]),
                'final_time_pd': float(self.phi_matrix[x_idx, final_time_idx])
            }
        
        results = {
            'method': 'Finite_Difference_BTCS',
            'parameters': {
                'sigma': self.sigma,
                'kappa': self.kappa, 
                'theta': self.theta,
                'mu_jump': self.mu_jump,
                'sigma_jump': self.sigma_jump,
                'rate': self.rate,
                'domain': {
                    'x_min': float(xmin),
                    'x_max': float(self.L),
                    't_min': float(self.t0),
                    't_max': float(self.tn)
                },
                'grid': {
                    'dx': float(self.dx),
                    'dt': float(self.dt),
                    'x_steps': self.xsteps,
                    't_steps': self.tsteps
                }
            },
            'timing': {
                'setup_time_seconds': self.solve_time['setup_time'],
                'iteration_time_seconds': self.solve_time['iteration_time'],
                'total_time_seconds': self.solve_time['total_time']
            },
            'solution_statistics': stats,
            'sample_solutions': sample_solutions,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Comparison results saved to {filename}")
        return results


def main():
    """
    Main execution function with parameters matching the PINN config.
    """
    print("=" * 60)
    print("1D Jump-Diffusion PIDE Solver - PINN Comparison")
    print("=" * 60)
    
    # Parameters matching the PINN config file
    # PDE Parameters from config
    k = 0.3              # Mean reversion speed (kappa)
    theta_val = 0.0      # Mean reversion level  
    sigma_val = 0.2      # Volatility
    lambda_jump = 1.0    # Jump intensity (rate)
    jump_std = 0.2       # Std dev of jump size (sigma_jump)
    
    # Domain from config
    tmin = 0.0
    tmax = 1.0  
    xmin = -0.5
    xmax = 2.0
    
    # Extended domain for jump handling (extend by ~2 jump standard deviations)
    extension = 4 * jump_std  # 4 * 0.2 = 0.8
    x0_extended = xmin - extension  # -0.5 - 0.8 = -1.3
    xn_extended = xmax + extension  # 2.0 + 0.8 = 2.8
    
    print(f"PINN Config Parameters:")
    print(f"  PDE: k={k}, θ={theta_val}, σ={sigma_val}, λ={lambda_jump}, σⱼ={jump_std}")
    print(f"  Domain: t∈[{tmin}, {tmax}], x∈[{xmin}, {xmax}]")
    print(f"  Extended grid: x∈[{x0_extended}, {xn_extended}] (for jump handling)")
    
    # Define parameters
    spatial_params = {
        'x0': x0_extended,   # Extended lower boundary
        'xn': xn_extended,   # Extended upper boundary  
        'L': xmax,           # Actual domain boundary (matches PINN xmax)
        'xsteps': 1000       # Number of spatial steps
    }
    
    temporal_params = {
        't0': tmin,          # Initial time (matches PINN tmin)
        'tn': tmax,          # Final time (matches PINN tmax)
        'tsteps': 1000       # Number of time steps
    }
    
    model_params = {
        'sigma': sigma_val,     # Volatility (matches PINN)
        'kappa': k,             # Mean reversion rate (matches PINN k)
        'theta': theta_val,     # Long-term mean (matches PINN)
        'mu_jump': 0.0,         # Jump size mean (not specified in PINN config)
        'sigma_jump': jump_std, # Jump size std (matches PINN)
        'rate': lambda_jump,    # Poisson jump rate (matches PINN)
        'N': 150                # Jump integral points
    }
    
    # Create and run solver
    solver = JumpDiffusionSolver(spatial_params, temporal_params, model_params)
    
    # Solve the system
    solution = solver.solve()
    
    # Get summary statistics
    stats = solver.get_summary_stats()
    print("\nSolution Summary:")
    print(f"  Probability of Default Range: [{stats['min_default_prob']:.6f}, {stats['max_default_prob']:.6f}]")
    print(f"  Mean Probability of Default: {stats['mean_default_prob']:.6f}")
    print(f"  Final Time Mean PD: {stats['final_time_mean_default_prob']:.6f}")
    print(f"  Grid Size: {stats['grid_points_spatial']} × {stats['grid_points_temporal']} = {stats['total_grid_points']:,} points")
    
    # Create visualization
    solver.plot_solution(save_html=True, filename="plots/probability_of_default_pinn_config.html")
    
    # Save comparison results
    comparison_results = solver.save_comparison_results("probability_of_default_results.json")
    
    print("\nExecution completed successfully!")
    print("\nFor PINN comparison:")
    print(f"  - Visualization: probability_of_default_pinn_config.html")
    print(f"  - Results data: probability_of_default_results.json")
    print(f"  - Total compute time: {comparison_results['timing']['total_time_seconds']:.2f} seconds")
    
    return solver


if __name__ == "__main__":
    solver = main()