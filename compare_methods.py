#!/usr/bin/env python3
"""
PINN vs Finite Difference Comparison Script
==========================================

This script compares the Physics-Informed Neural Network (PINN) solution 
with the Finite Difference solution for the 1D jump-diffusion PIDE.

Comparisons include:
- Side-by-side 3D visualizations
- Difference plots  
- Quantitative error metrics
- Performance benchmarking
- Sample point comparisons

Author: Comparison analysis script
"""

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
import json
import time
import sys
import os
from pathlib import Path

# Import PINN model (assuming it's in the same directory)
try:
    from model import OU_PINN
    # Import config if available
    try:
        from config import (
            device, DTYPE, xmin, xmax, tmin, tmax,
            HIDDEN_LAYERS, NEURONS_PER_LAYER, MODEL_PATH,
            PDE_CONFIG
        )
    except ImportError:
        # Fallback config values
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        DTYPE = torch.float32
        xmin, xmax = -0.5, 2.0
        tmin, tmax = 0.0, 1.0
        HIDDEN_LAYERS = 6
        NEURONS_PER_LAYER = 30
        MODEL_PATH = "levy_ou_pinn_model.pth"
        PDE_CONFIG = {
            'k': 0.3,
            'theta': 0.0,
            'sigma': 0.2,
            'lambda_jump': 1.0,
            'jump_std': 0.2
        }
except ImportError:
    print("Warning: Could not import PINN model. Please ensure model.py is available.")
    print("Proceeding with finite difference analysis only.")

# Import finite difference solver
from jump_diffusion_solver import JumpDiffusionSolver


class MethodComparator:
    """
    Compare PINN and Finite Difference solutions for jump-diffusion PIDE.
    """
    
    def __init__(self, pinn_model_path="levy_ou_pinn_model.pth", 
                 fd_results_path="probability_of_default_results.json"):
        self.pinn_model_path = pinn_model_path
        self.fd_results_path = fd_results_path
        self.pinn_model = None
        self.fd_results = None
        self.comparison_grid = None
        self.pinn_solution = None
        self.fd_solution = None
        self.comparison_metrics = {}
        
    def load_pinn_model(self):
        """Load the trained PINN model."""
        try:
            print(f"Loading PINN model from {self.pinn_model_path}...")
            self.pinn_model = OU_PINN(
                hidden_layers=HIDDEN_LAYERS, 
                neurons_per_layer=NEURONS_PER_LAYER
            ).to(device)
            
            self.pinn_model.load_state_dict(torch.load(self.pinn_model_path, map_location=device))
            self.pinn_model.eval()
            print("PINN model loaded successfully!")
            return True
            
        except FileNotFoundError:
            print(f"Error: PINN model file not found at {self.pinn_model_path}")
            return False
        except Exception as e:
            print(f"Error loading PINN model: {e}")
            return False
    
    def load_fd_results(self):
        """Load finite difference results."""
        try:
            print(f"Loading finite difference results from {self.fd_results_path}...")
            with open(self.fd_results_path, 'r') as f:
                self.fd_results = json.load(f)
            print("Finite difference results loaded successfully!")
            return True
            
        except FileNotFoundError:
            print(f"Error: Finite difference results not found at {self.fd_results_path}")
            return False
        except Exception as e:
            print(f"Error loading finite difference results: {e}")
            return False
    
    def create_comparison_grid(self, num_points_x=200, num_points_t=200):
        """Create a grid for comparison evaluation."""
        print(f"Creating comparison grid ({num_points_x} x {num_points_t})...")
        
        x_vals = np.linspace(xmin, xmax, num_points_x)
        t_vals = np.linspace(tmin, tmax, num_points_t)
        X, T = np.meshgrid(x_vals, t_vals)
        
        self.comparison_grid = {
            'x_vals': x_vals,
            't_vals': t_vals,
            'X': X,
            'T': T,
            'num_points_x': num_points_x,
            'num_points_t': num_points_t
        }
        
        print(f"Grid created: {num_points_x} x {num_points_t} = {num_points_x * num_points_t:,} points")
    
    def evaluate_pinn_solution(self):
        """Evaluate PINN model on the comparison grid."""
        if self.pinn_model is None or self.comparison_grid is None:
            print("Error: PINN model or comparison grid not available")
            return False
        
        print("Evaluating PINN solution on comparison grid...")
        start_time = time.time()
        
        X, T = self.comparison_grid['X'], self.comparison_grid['T']
        
        # Prepare grid points for PINN model [N, 2] format
        tx_grid = np.stack([T.ravel(), X.ravel()], axis=-1)
        tx_tensor = torch.tensor(tx_grid, dtype=DTYPE, device=device)
        
        # Evaluate in batches to avoid memory issues
        batch_size = 10000
        num_batches = (len(tx_tensor) + batch_size - 1) // batch_size
        phi_pred_list = []
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(tx_tensor))
                batch = tx_tensor[start_idx:end_idx]
                
                batch_pred = self.pinn_model(batch)
                phi_pred_list.append(batch_pred.cpu().numpy())
        
        # Combine batches and reshape
        phi_pred = np.concatenate(phi_pred_list, axis=0)
        self.pinn_solution = phi_pred.reshape(X.shape)
        
        eval_time = time.time() - start_time
        print(f"PINN evaluation completed in {eval_time:.2f} seconds")
        
        # Store timing for comparison plots
        self.pinn_eval_time = eval_time
        
        return True
    
    def evaluate_fd_solution(self):
        """Evaluate finite difference solution on the comparison grid."""
        print("Setting up finite difference solver for comparison...")
        
        # Extract parameters from loaded FD results
        fd_params = self.fd_results['parameters']
        
        # Setup solver with same parameters
        spatial_params = {
            'x0': -1.3,  # Extended boundary
            'xn': 2.8,   # Extended boundary  
            'L': xmax,   # Actual domain boundary
            'xsteps': 1000
        }
        
        temporal_params = {
            't0': tmin,
            'tn': tmax,
            'tsteps': 1000
        }
        
        model_params = {
            'sigma': fd_params['sigma'],
            'kappa': fd_params['kappa'],
            'theta': fd_params['theta'],
            'mu_jump': fd_params['mu_jump'],
            'sigma_jump': fd_params['sigma_jump'],
            'rate': fd_params['rate'],
            'N': 150
        }
        
        print("Running finite difference solver...")
        start_time = time.time()
        
        # Create and solve - use the same solver logic as main script
        solver = JumpDiffusionSolver(spatial_params, temporal_params, model_params)
        solution_matrix = solver.solve()
        
        eval_time = time.time() - start_time
        print(f"Finite difference evaluation completed in {eval_time:.2f} seconds")
        
        # Note: This includes both setup and solving time for FD method
        
        # Interpolate FD solution onto comparison grid
        print("Interpolating finite difference solution to comparison grid...")
        from scipy.interpolate import RegularGridInterpolator
        
        # Filter to domain of interest with small buffer to avoid interpolation artifacts
        buffer = 0.1
        accept_x = np.where((solver.x >= xmin - buffer) & (solver.x <= xmax + buffer))[0]
        x_domain = solver.x[accept_x]
        t_domain = solver.t
        solution_domain = solution_matrix[accept_x, :]
        
        # Create interpolator with better boundary handling
        interpolator = RegularGridInterpolator(
            (t_domain, x_domain), 
            solution_domain.T,  # Transpose for correct orientation
            method='linear',
            bounds_error=False,
            fill_value=None  # Use extrapolation instead of fixed fill value
        )
        
        # Evaluate on comparison grid
        X, T = self.comparison_grid['X'], self.comparison_grid['T']
        points = np.stack([T.ravel(), X.ravel()], axis=-1)
        fd_interp = interpolator(points).reshape(X.shape)
        
        self.fd_solution = fd_interp
        self.fd_solver_time = eval_time
        
        return True
    
    def compute_comparison_metrics(self):
        """Compute quantitative comparison metrics."""
        if self.pinn_solution is None or self.fd_solution is None:
            print("Error: Solutions not available for comparison")
            return
        
        print("Computing comparison metrics...")
        
        # Mask for valid comparison region
        X, T = self.comparison_grid['X'], self.comparison_grid['T']
        valid_mask = (X >= xmin) & (X <= xmax) & (T >= tmin) & (T <= tmax)
        
        pinn_valid = self.pinn_solution[valid_mask]
        fd_valid = self.fd_solution[valid_mask]
        
        # Compute metrics
        diff = pinn_valid - fd_valid
        
        self.comparison_metrics = {
            'mse': float(np.mean(diff**2)),
            'rmse': float(np.sqrt(np.mean(diff**2))),
            'mae': float(np.mean(np.abs(diff))),
            'max_absolute_error': float(np.max(np.abs(diff))),
            'max_relative_error': float(np.max(np.abs(diff) / (np.abs(fd_valid) + 1e-8))),
            'mean_relative_error': float(np.mean(np.abs(diff) / (np.abs(fd_valid) + 1e-8))),
            'correlation': float(np.corrcoef(pinn_valid, fd_valid)[0, 1]),
            'pinn_range': [float(np.min(pinn_valid)), float(np.max(pinn_valid))],
            'fd_range': [float(np.min(fd_valid)), float(np.max(fd_valid))],
            'valid_points': int(np.sum(valid_mask))
        }
        
        print("Comparison metrics computed:")
        print(f"  RMSE: {self.comparison_metrics['rmse']:.6f}")
        print(f"  MAE: {self.comparison_metrics['mae']:.6f}")
        print(f"  Max Error: {self.comparison_metrics['max_absolute_error']:.6f}")
        print(f"  Correlation: {self.comparison_metrics['correlation']:.6f}")
        
        # Display timing comparison
        pinn_time = getattr(self, 'pinn_eval_time', None)
        fd_time = getattr(self, 'fd_solver_time', None)
        
        if pinn_time is not None and fd_time is not None:
            print(f"\nTiming Comparison:")
            print(f"  PINN Inference Time: {pinn_time:.4f} seconds")
            print(f"  FD Total Time: {fd_time:.4f} seconds")
            speedup = fd_time / pinn_time if pinn_time > 0 else 0
            print(f"  Speedup Factor: {speedup:.1f}x {'(PINN faster)' if speedup > 1 else '(FD faster)'}")
    
    def create_comparison_plots(self, base_filename="comparison"):
        """Create separate comparison visualizations."""
        if self.pinn_solution is None or self.fd_solution is None:
            print("Error: Solutions not available for plotting")
            return
        
        print("Creating separate comparison visualizations...")
        
        X, T = self.comparison_grid['X'], self.comparison_grid['T']
        
        # Common scene settings
        scene_settings = dict(
            xaxis_title='Time',
            yaxis_title='Position', 
            zaxis_title='Probability of Default',
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        )
        
        # Common layout settings
        layout_settings = dict(
            height=600,
            width=800,
            showlegend=False,
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        figures = {}
        
        # 1. PINN Solution
        if self.pinn_solution is not None:
            print("  Creating PINN solution plot...")
            fig_pinn = go.Figure()
            fig_pinn.add_trace(
                go.Surface(
                    z=self.pinn_solution, x=T, y=X,
                    colorscale='Viridis',
                    name='PINN Solution',
                    showscale=True
                )
            )
            fig_pinn.update_layout(
                title=dict(
                    text=f'PINN Solution - Probability of Default<br>'
                         f'<sub>Range: [{np.min(self.pinn_solution):.3f}, {np.max(self.pinn_solution):.3f}]</sub>',
                    x=0.5
                ),
                scene=scene_settings,
                **layout_settings
            )
            pinn_filename = f"plots/{base_filename}_pinn_solution.html"
            fig_pinn.write_html(pinn_filename)
            figures['pinn'] = fig_pinn
            print(f"    Saved: {pinn_filename}")
        
        # 2. FD Solution
        print("  Creating FD solution plot...")
        fig_fd = go.Figure()
        fig_fd.add_trace(
            go.Surface(
                z=self.fd_solution, x=T, y=X,
                colorscale='Viridis',
                name='FD Solution',
                showscale=True
            )
        )
        fig_fd.update_layout(
            title=dict(
                text=f'Finite Difference Solution - Probability of Default<br>'
                     f'<sub>Range: [{np.min(self.fd_solution):.3f}, {np.max(self.fd_solution):.3f}]</sub>',
                x=0.5
            ),
            scene=scene_settings,
            **layout_settings
        )
        fd_filename = f"plots/{base_filename}_fd_solution.html"
        fig_fd.write_html(fd_filename)
        figures['fd'] = fig_fd
        print(f"    Saved: {fd_filename}")
        
        # Only create difference plots if PINN solution is available
        if self.pinn_solution is not None:
            # 3. Absolute Difference
            print("  Creating absolute difference plot...")
            abs_diff = np.abs(self.pinn_solution - self.fd_solution)
            fig_abs = go.Figure()
            fig_abs.add_trace(
                go.Surface(
                    z=abs_diff, x=T, y=X,
                    colorscale='Reds',
                    name='Absolute Difference',
                    showscale=True
                )
            )
            fig_abs.update_layout(
                title=dict(
                    text=f'Absolute Difference |PINN - FD|<br>'
                         f'<sub>Max Error: {np.max(abs_diff):.6f}, Mean Error: {np.mean(abs_diff):.6f}</sub>',
                    x=0.5
                ),
                scene={**scene_settings, 'zaxis_title': 'Absolute Error'},
                **layout_settings
            )
            abs_filename = f"plots/{base_filename}_absolute_diff.html"
            fig_abs.write_html(abs_filename)
            figures['absolute'] = fig_abs
            print(f"    Saved: {abs_filename}")
            
            # 4. Relative Difference
            print("  Creating relative difference plot...")
            rel_diff = np.abs(self.pinn_solution - self.fd_solution) / (np.abs(self.fd_solution) + 1e-8)
            fig_rel = go.Figure()
            fig_rel.add_trace(
                go.Surface(
                    z=rel_diff, x=T, y=X,
                    colorscale='Reds',
                    name='Relative Difference',
                    showscale=True
                )
            )
            fig_rel.update_layout(
                title=dict(
                    text=f'Relative Difference |PINN - FD| / |FD|<br>'
                         f'<sub>Max Rel Error: {np.max(rel_diff):.6f}, Mean Rel Error: {np.mean(rel_diff):.6f}</sub>',
                    x=0.5
                ),
                scene={**scene_settings, 'zaxis_title': 'Relative Error'},
                **layout_settings
            )
            rel_filename = f"plots/{base_filename}_relative_diff.html"
            fig_rel.write_html(rel_filename)
            figures['relative'] = fig_rel
            print(f"    Saved: {rel_filename}")
            
            # 5. Side-by-side comparison
            print("  Creating side-by-side comparison...")
            fig_compare = sp.make_subplots(
                rows=1, cols=2,
                subplot_titles=('PINN Solution', 'FD Solution'),
                specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                horizontal_spacing=0.1
            )
            
            fig_compare.add_trace(
                go.Surface(z=self.pinn_solution, x=T, y=X, 
                          colorscale='Viridis', showscale=False),
                row=1, col=1
            )
            fig_compare.add_trace(
                go.Surface(z=self.fd_solution, x=T, y=X,
                          colorscale='Viridis', showscale=True),
                row=1, col=2
            )
            
            # Get timing information
            pinn_time = getattr(self, 'pinn_eval_time', 'N/A')
            fd_time = getattr(self, 'fd_solver_time', 'N/A')
            
            # Format timing display
            pinn_time_str = f"{pinn_time:.4f}s" if isinstance(pinn_time, (int, float)) else str(pinn_time)
            fd_time_str = f"{fd_time:.4f}s" if isinstance(fd_time, (int, float)) else str(fd_time)
            
            fig_compare.update_layout(
                title=dict(
                    text=f'<sub>RMSE: {self.comparison_metrics["rmse"]:.6f}, '
                         f'Correlation: {self.comparison_metrics["correlation"]:.4f}<br>'
                         f'PINN Inference: {pinn_time_str}, '
                         f'FD Total Time: {fd_time_str}</sub>',
                    x=0.5
                ),
                height=600,
                width=1200,
                showlegend=False
            )
            
            fig_compare.update_scenes(**scene_settings)
            
            compare_filename = f"plots/{base_filename}_side_by_side.html"
            fig_compare.write_html(compare_filename)
            figures['comparison'] = fig_compare
            print(f"    Saved: {compare_filename}")
        
        print("All separate visualizations created!")
        return figures
    
    def save_comparison_results(self, filename="method_comparison_results.json"):
        """Save comprehensive comparison results."""
        print(f"Saving comparison results to {filename}...")
        
        # Get timing information
        pinn_timing = getattr(self, 'pinn_eval_time', 'Not measured')
        fd_timing = getattr(self, 'fd_solver_time', self.fd_results['timing']['total_time_seconds'])
        
        # Calculate speedup if both timings are available
        speedup_factor = None
        if isinstance(pinn_timing, (int, float)) and isinstance(fd_timing, (int, float)) and pinn_timing > 0:
            speedup_factor = fd_timing / pinn_timing
        
        results = {
            'comparison_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'grid_size': f"{self.comparison_grid['num_points_x']} x {self.comparison_grid['num_points_t']}",
                'total_comparison_points': self.comparison_grid['num_points_x'] * self.comparison_grid['num_points_t'],
                'domain': {
                    'x_range': [xmin, xmax],
                    't_range': [tmin, tmax]
                }
            },
            'methods': {
                'pinn': {
                    'model_path': self.pinn_model_path,
                    'architecture': f"{HIDDEN_LAYERS} layers, {NEURONS_PER_LAYER} neurons/layer",
                    'evaluation_time': pinn_timing
                },
                'finite_difference': {
                    'results_path': self.fd_results_path,
                    'solver_time': fd_timing,
                    'grid_resolution': f"dx={self.fd_results['parameters']['grid']['dx']:.6f}, dt={self.fd_results['parameters']['grid']['dt']:.6f}"
                }
            },
            'timing_comparison': {
                'pinn_inference_time': pinn_timing,
                'fd_total_time': fd_timing,
                'speedup_factor': speedup_factor,
                'faster_method': 'PINN' if speedup_factor and speedup_factor > 1 else 'FD' if speedup_factor else 'Unknown'
            },
            'comparison_metrics': self.comparison_metrics,
            'sample_point_comparison': self._compare_sample_points()
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Comparison results saved to {filename}")
        return results
    
    def _compare_sample_points(self):
        """Compare solutions at specific sample points."""
        if self.fd_results is None:
            return {}
        
        sample_points = {}
        
        # Use sample points from FD results
        for point_key, fd_values in self.fd_results['sample_solutions'].items():
            x_val = float(point_key.split('_')[1])
            
            # Find closest grid points
            x_idx = np.argmin(np.abs(self.comparison_grid['x_vals'] - x_val))
            
            # Sample at initial, mid, and final times
            t_indices = [0, len(self.comparison_grid['t_vals'])//2, -1]
            t_labels = ['initial_pd', 'mid_time_pd', 'final_time_pd']
            
            point_comparison = {}
            for t_idx, t_label in zip(t_indices, t_labels):
                pinn_val = float(self.pinn_solution[t_idx, x_idx])
                fd_val = fd_values[t_label]
                
                point_comparison[t_label] = {
                    'pinn': pinn_val,
                    'finite_difference': fd_val,
                    'absolute_error': abs(pinn_val - fd_val),
                    'relative_error': abs(pinn_val - fd_val) / (abs(fd_val) + 1e-8)
                }
            
            sample_points[point_key] = point_comparison
        
        return sample_points
    
    def run_full_comparison(self):
        """Run the complete comparison analysis."""
        print("=" * 60)
        print("PINN vs Finite Difference Comparison Analysis")
        print("=" * 60)
        
        # Load models and data
        pinn_loaded = self.load_pinn_model()
        fd_loaded = self.load_fd_results()
        
        if not fd_loaded:
            print("Cannot proceed without finite difference results")
            return None
        
        # Create comparison grid
        self.create_comparison_grid(num_points_x=200, num_points_t=200)
        
        # Evaluate solutions
        if pinn_loaded:
            self.evaluate_pinn_solution()
        
        self.evaluate_fd_solution()
        
        # Compare solutions
        if pinn_loaded and self.pinn_solution is not None:
            self.compute_comparison_metrics()
            self.create_comparison_plots()
            return self.save_comparison_results()
        else:
            print("PINN model not available - saving FD analysis only")
            # Could add FD-only analysis here
            return None


def main():
    """Main execution function."""
    comparator = MethodComparator()
    results = comparator.run_full_comparison()
    
    if results:
        print("\nComparison completed successfully!")
        print("Generated files:")
        print("  - plots/ directory (interactive visualizations)")
        print("  - method_comparison_results.json (detailed metrics)")
    else:
        print("\nComparison completed with limitations (PINN model may not be available)")


if __name__ == "__main__":
    main()