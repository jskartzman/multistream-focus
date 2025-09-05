#!/usr/bin/env python3
"""
HPC script for running change point detection experiments with multiprocessing.
Imports existing implementations and adds multiprocessing capabilities.
"""

import numpy as np
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import argparse
import time
import pickle
from datetime import datetime

# Add parent directory to path to import focus_implementation
import focus_implementation as focus

def run_single_simulation(args):
    """Run a single EDD simulation."""
    algorithm, M, T, nu, mu1, threshold, seed, extra_params = args
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Run the specified algorithm using streaming implementation
    try:
        if algorithm == 'focus_decay':
            result = focus.focus_decay_streaming(M, T, nu, mu1, threshold)
        elif algorithm == 'focus_oracle':
            result = focus.focus_oracle_streaming(M, T, nu, mu1, threshold)
        elif algorithm == 'focus_nonuhat':
            result = focus.focus_nonuhat_streaming(M, T, nu, mu1, threshold)
        elif algorithm == 'xumei':
            lower_bound = extra_params.get('lower_bound', .1)
            # Use centralized xumei streaming from focus_implementation
            result = focus.xumei_streaming(M, T, nu, mu1, threshold, lower_bound)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        if result is None:
            return None
        detection_time, _ = result
        return detection_time
        
    except Exception as e:
        print(f"Error in EDD simulation with seed {seed}: {e}")
        return None

# All streaming algorithms now centralized in focus_implementation.py  
# Use: focus.focus_decay_streaming(), focus.focus_nonuhat_streaming(), focus.xumei_streaming()

def run_experiment(algorithm, nus, thresholds, M=10, T=int(1e6), mu1=1, 
                  sims=50, n_workers=None, extra_params=None):
    """
    Run EDD experiment with multiprocessing.
    
    Parameters:
    -----------
    algorithm : str
        Algorithm to use ('focus_decay', 'focus_oracle', 'focus_nonuhat', 'xumei')
    nus : list
        List of change point locations to test
    thresholds : list
        List of thresholds to test
    M : int
        Number of streams
    T : int
        Time horizon
    mu1 : float
        Post-change mean
    sims : int
        Number of simulations per configuration
    n_workers : int
        Number of parallel workers
    extra_params : dict
        Extra parameters for specific algorithms
    
    Returns:
    --------
    edd_means : np.ndarray
        Expected detection delays
    edd_stds : np.ndarray
        Standard deviations of detection delays
    all_detection_times : list
        All raw detection times
    """
    if n_workers is None:
        n_workers = cpu_count()
    
    if extra_params is None:
        extra_params = {}
    
    print(f"Running {algorithm} EDD experiment")
    print(f"Parameters: M={M}, T={T}, mu1={mu1}, sims={sims}, workers={n_workers}")
    print(f"Change points: {nus}")
    print(f"Thresholds: {thresholds}")
    print("-" * 50)
    
    edd_means = np.zeros((len(nus), len(thresholds)))
    edd_stds = np.zeros((len(nus), len(thresholds)))
    all_detection_times = [[[] for _ in thresholds] for _ in nus]
    
    for idx, nu in enumerate(nus):
        for idx2, threshold in enumerate(thresholds):
            print(f"Running nu={nu}, threshold={threshold:.1f}")
            start_time = time.time()
            
            # Prepare arguments for parallel execution
            sim_args = []
            for sim in range(sims):
                seed = np.random.randint(0, 2**31)
                sim_args.append((algorithm, M, T, nu, mu1, threshold, seed, extra_params))
            
            # Run simulations in parallel
            delays = []
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(run_single_simulation, args) for args in sim_args]
                
                for future in as_completed(futures):
                    detection_time = future.result()
                    if detection_time is not None and detection_time > nu:
                        delays.append(detection_time - nu)
            
            # Store results
            all_detection_times[idx][idx2] = delays
            
            # Calculate statistics
            if len(delays) > 0:
                edd_estimate = np.mean(delays)
                std_estimate = np.std(delays)
                success_rate = len(delays) / sims
            else:
                edd_estimate = np.nan
                std_estimate = np.nan
                success_rate = 0.0
            
            # Store results
            edd_means[idx, idx2] = edd_estimate
            edd_stds[idx, idx2] = std_estimate
            
            elapsed_time = time.time() - start_time
            
            # Print results
            print(f"  Success: {success_rate:.1%}, EDD: {edd_estimate:.1f}, Time: {elapsed_time:.1f}s")
    
    return edd_means, edd_stds, all_detection_times

def run_experiment_streaming(algorithm, nus, thresholds, M=10, T=int(1e6), mu1=1, 
                  sims=50, n_workers=None, extra_params=None):
    """COMPATIBILITY WRAPPER: Maintains old API but uses streaming internally."""
    return run_experiment(algorithm, nus, thresholds, M, T, mu1, sims, n_workers, extra_params)

def save_edd_results(algorithm, nus, thresholds, edd_means, edd_stds, T, sims, 
                    all_detection_times, extra_params=None, n_workers=None, M=10, mu1=1.0, data_dir="data"):
    """Save comprehensive EDD results using pickle."""
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare comprehensive results dictionary
    results_data = {
        'metadata': {
            'algorithm': algorithm,
            'timestamp': timestamp,
            'T': T,
            'sims': sims,
            'n_workers': n_workers,
            'M': M,
            'mu1': mu1,
            'extra_params': extra_params or {},
            'lower_bound': extra_params.get('lower_bound', None) if extra_params else None
        },
        'parameters': {
            'nus': nus,
            'thresholds': thresholds,
            'M': M,
            'mu1': mu1,
            'lower_bound': extra_params.get('lower_bound', None) if extra_params else None
        },
        'results': {
            'edd_means': edd_means,
            'edd_stds': edd_stds,
            'all_detection_times': all_detection_times  # Raw detection times for each config
        },
        'statistics': {
            'success_rates': np.zeros((len(nus), len(thresholds))),
            'min_detection_times': np.zeros((len(nus), len(thresholds))),
            'max_detection_times': np.zeros((len(nus), len(thresholds))),
            'median_detection_times': np.zeros((len(nus), len(thresholds)))
        }
    }
    
    # Calculate additional statistics
    for i, nu in enumerate(nus):
        for j, threshold in enumerate(thresholds):
            detection_times = all_detection_times[i][j]
            if len(detection_times) > 0:
                results_data['statistics']['success_rates'][i, j] = len(detection_times) / sims
                results_data['statistics']['min_detection_times'][i, j] = np.min(detection_times)
                results_data['statistics']['max_detection_times'][i, j] = np.max(detection_times)
                results_data['statistics']['median_detection_times'][i, j] = np.median(detection_times)
            else:
                results_data['statistics']['success_rates'][i, j] = 0.0
                results_data['statistics']['min_detection_times'][i, j] = np.nan
                results_data['statistics']['max_detection_times'][i, j] = np.nan
                results_data['statistics']['median_detection_times'][i, j] = np.nan
    
    # Create short filename (full parameters saved inside pickle file)
    nu_range = f"nu{min(nus):.0f}-{max(nus):.0f}" if len(nus) > 1 else f"nu{nus[0]:.0f}"
    thresh_range = f"th{min(thresholds):.1f}-{max(thresholds):.1f}" if len(thresholds) > 1 else f"th{thresholds[0]:.1f}"
    filename = f"edd_{algorithm}_{nu_range}_{thresh_range}_M{M}_T{T//1000}k_s{sims}_{timestamp}.pkl"
    filepath = os.path.join(data_dir, filename)
    
    # Save using pickle
    with open(filepath, 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"Comprehensive EDD results saved to: {filepath}")
    print(f"File size: {os.path.getsize(filepath) / 1024:.1f} KB")
    
    return filepath

def main():
    parser = argparse.ArgumentParser(description='Run EDD experiments for change point detection algorithms')
    parser.add_argument('--algorithms', type=str, required=True,
                       help='Comma-separated algorithms (e.g., focus_decay,xumei)')
    parser.add_argument('--nus', type=str, default='0,1000,10000',
                       help='Comma-separated change point locations')
    parser.add_argument('--Ms', type=str, default='10',
                       help='Comma-separated number of streams to test')
    parser.add_argument('--threshold-min', type=float, default=1000,
                       help='Minimum threshold to test')
    parser.add_argument('--threshold-max', type=float, default=5000,
                       help='Maximum threshold to test')
    parser.add_argument('--threshold-steps', type=int, default=5,
                       help='Number of threshold steps')
    parser.add_argument('--T', type=int, default=1000000, help='Time horizon')
    parser.add_argument('--mu1', type=float, default=1.0, help='Post-change mean')
    parser.add_argument('--sims', type=int, default=50, help='Number of simulations')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--xumei-lb', type=float, default=0.1, 
                       help='Lower bound for xumei algorithm')
    parser.add_argument('--save', action='store_true', help='Save results to files')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to save data files')
    
    args = parser.parse_args()
    
    # Parse comma-separated arguments
    algorithms = [x.strip() for x in args.algorithms.split(',')]
    nus = [float(x.strip()) for x in args.nus.split(',')]
    Ms = [int(x.strip()) for x in args.Ms.split(',')]
    
    # Generate threshold range
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    
    print(f"Starting EDD experiments for {len(algorithms)} algorithms")
    print(f"Algorithms: {algorithms}")
    print(f"Streams: {Ms}")
    print(f"Change points: {nus}")
    print(f"Thresholds: {len(thresholds)} steps from {args.threshold_min} to {args.threshold_max}")
    print("=" * 60)

    # Run experiments for each algorithm and M combination
    for algorithm in algorithms:
        for M in Ms:
            print(f"\nRunning {algorithm} with M={M}")
            
            # Set up extra parameters for this algorithm
            extra_params = {}
            if algorithm == 'xumei':
                extra_params['lower_bound'] = args.xumei_lb

            # Run experiment
            edd_means, edd_stds, all_detection_times = run_experiment(
                algorithm=algorithm,
                nus=nus,
                thresholds=thresholds,
                M=M,
                T=args.T,
                mu1=args.mu1,
                sims=args.sims,
                n_workers=args.workers,
                extra_params=extra_params
            )
            
            # Save results if requested
            if args.save:
                save_edd_results(algorithm, nus, thresholds, edd_means, edd_stds, args.T, 
                                args.sims, all_detection_times, extra_params, args.workers, M, args.mu1, args.data_dir)
    
    print(f"\nCompleted EDD experiments for all {len(algorithms)} algorithms and {len(Ms)} stream configurations!")
    return

if __name__ == "__main__":
    main() 