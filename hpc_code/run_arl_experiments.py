#!/usr/bin/env python3
"""
HPC script for running Average Run Length (ARL) experiments with multiprocessing.
ARL measures false alarm rates when there is NO change in the data.
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

# Import existing implementations
import focus_implementation as focus
# Removed scipy import that causes DLL loading issues in multiprocessing:
# from scipy.stats import bernoulli # Replaced with np.random.random() < p in focus_implementation.py

def run_single_arl_simulation(args):
    """Run a single ARL simulation."""
    algorithm, M, T, threshold, seed, extra_params = args
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    try:
        if algorithm == 'focus_decay':
            result = focus.focus_decay_streaming(M, T, nu=0, mu1=0, threshold=threshold)
        elif algorithm == 'focus_nonuhat':
            result = focus.focus_nonuhat_streaming(M, T, nu=0, mu1=0, threshold=threshold)
        elif algorithm == 'xumei':
            lower_bound = extra_params.get('lower_bound', 2.0)
            result = focus.xumei_streaming(M, T, nu=0, mu1=0, threshold=threshold, lower_bound=lower_bound)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        if result is None:
            return T  # No detection within time horizon
        detection_time, _ = result
        return detection_time
            
    except Exception as e:
        print(f"Error in ARL simulation with seed {seed}: {e}")
        return None

# All streaming algorithms now centralized in focus_implementation.py
# Use: focus.focus_decay_streaming(), focus.focus_nonuhat_streaming(), focus.xumei_streaming()

def run_arl_experiment(algorithm, Ms, thresholds, T=int(2e5), 
                      sims=50, n_workers=None, extra_params=None):
    """
    Run ARL experiment with multiprocessing.
    
    Parameters:
    -----------
    algorithm : str
        Algorithm to use ('focus_decay', 'focus_nonuhat', 'xumei')
    Ms : list
        List of number of streams to test
    thresholds : list
        List of thresholds to test
    T : int
        Time horizon
    sims : int
        Number of simulations per configuration
    n_workers : int
        Number of parallel workers
    extra_params : dict
        Extra parameters for specific algorithms
    
    Returns:
    --------
    arl_means : np.ndarray
        Average run lengths (time to false alarm)
    arl_stds : np.ndarray
        Standard deviations of run lengths
    all_detection_times : list
        All raw detection times
    """
    if n_workers is None:
        n_workers = cpu_count()
    
    if extra_params is None:
        extra_params = {}
    
    print(f"Running {algorithm} ARL experiment")
    print(f"Parameters: T={T}, sims={sims}, workers={n_workers}")
    print(f"Streams: {Ms}")
    print(f"Thresholds: {thresholds}")
    print("-" * 50)
    
    arl_means = np.zeros((len(Ms), len(thresholds)))
    arl_stds = np.zeros((len(Ms), len(thresholds)))
    all_detection_times = [[[] for _ in thresholds] for _ in Ms]
    
    for idx, M in enumerate(Ms):
        for idx2, threshold in enumerate(thresholds):
            print(f"Running M={M}, threshold={threshold:.1f}")
            start_time = time.time()
            
            # Prepare arguments for parallel execution
            sim_args = []
            for sim in range(sims):
                seed = np.random.randint(0, 2**31)
                sim_args.append((algorithm, M, T, threshold, seed, extra_params))
            
            # Run simulations in parallel
            run_lengths = []
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(run_single_arl_simulation, args) for args in sim_args]
                
                for future in as_completed(futures):
                    detection_time = future.result()
                    if detection_time is not None:
                        run_lengths.append(detection_time)
            
            # Store results
            all_detection_times[idx][idx2] = run_lengths
            
            # Calculate statistics
            if len(run_lengths) > 0:
                arl_estimate = np.mean(run_lengths)
                std_estimate = np.std(run_lengths)
                success_rate = len(run_lengths) / sims
            else:
                arl_estimate = np.nan
                std_estimate = np.nan
                success_rate = 0.0
            
            # Store results
            arl_means[idx, idx2] = arl_estimate
            arl_stds[idx, idx2] = std_estimate
            
            elapsed_time = time.time() - start_time
            
            # Print results
            print(f"  Success: {success_rate:.1%}, ARL: {arl_estimate:.1f}, Time: {elapsed_time:.1f}s")
    
    return arl_means, arl_stds, all_detection_times

def save_arl_results(algorithm, Ms, thresholds, arl_means, arl_stds, T, sims, 
                    all_detection_times, extra_params=None, n_workers=None, data_dir="data"):
    """Save comprehensive ARL results using pickle."""
    
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
            'extra_params': extra_params or {},
            'lower_bound': extra_params.get('lower_bound', None) if extra_params else None
        },
        'parameters': {
            'Ms': Ms,
            'thresholds': thresholds,
            'lower_bound': extra_params.get('lower_bound', None) if extra_params else None
        },
        'results': {
            'arl_means': arl_means,
            'arl_stds': arl_stds,
            'all_detection_times': all_detection_times  # Raw detection times for each config
        },
        'statistics': {
            'success_rates': np.zeros((len(Ms), len(thresholds))),
            'min_detection_times': np.zeros((len(Ms), len(thresholds))),
            'max_detection_times': np.zeros((len(Ms), len(thresholds))),
            'median_detection_times': np.zeros((len(Ms), len(thresholds)))
        }
    }
    
    # Calculate additional statistics
    for i, M in enumerate(Ms):
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
    M_range = f"M{min(Ms)}-{max(Ms)}" if len(Ms) > 1 else f"M{Ms[0]}"
    thresh_range = f"th{min(thresholds):.1f}-{max(thresholds):.1f}" if len(thresholds) > 1 else f"th{thresholds[0]:.1f}"
    filename = f"arl_{algorithm}_{M_range}_{thresh_range}_T{T//1000}k_s{sims}_{timestamp}.pkl"
    filepath = os.path.join(data_dir, filename)
    
    # Save using pickle
    with open(filepath, 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"Comprehensive results saved to: {filepath}")
    print(f"File size: {os.path.getsize(filepath) / 1024:.1f} KB")
    
    return filepath

def main():
    parser = argparse.ArgumentParser(description='Run ARL experiments for change point detection algorithms')
    parser.add_argument('--algorithms', type=str, required=True,
                       help='Comma-separated algorithms (e.g., focus_decay,xumei)')
    parser.add_argument('--Ms', type=str, default='1,3,5,10',
                       help='Comma-separated number of streams to test')
    parser.add_argument('--threshold-min', type=float, default=1000,
                       help='Minimum threshold to test')
    parser.add_argument('--threshold-max', type=float, default=5000,
                       help='Maximum threshold to test')
    parser.add_argument('--threshold-steps', type=int, default=5,
                       help='Number of threshold steps')
    parser.add_argument('--thresholds', type=str, default=None,
                       help='Comma-separated exact threshold values (overrides min/max/steps)')
    parser.add_argument('--T', type=int, default=1000000, help='Time horizon')
    parser.add_argument('--sims', type=int, default=50, help='Number of simulations')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--xumei-lb', type=float, default=2.0, 
                       help='Lower bound for xumei algorithm')
    parser.add_argument('--save', action='store_true', help='Save results to files')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to save data files')
    
    args = parser.parse_args()
    
    # Parse comma-separated arguments
    algorithms = [x.strip() for x in args.algorithms.split(',')]
    Ms = [int(x.strip()) for x in args.Ms.split(',')]
    
    # Generate threshold range
    if args.thresholds:
        thresholds = np.array([float(x.strip()) for x in args.thresholds.split(',')])
    else:
        thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    
    print(f"Starting ARL experiments for {len(algorithms)} algorithms")
    print(f"Algorithms: {algorithms}")
    print(f"Streams: {Ms}")
    if args.thresholds:
        print(f"Thresholds: {len(thresholds)} exact values: {thresholds}")
    else:
        print(f"Thresholds: {len(thresholds)} steps from {args.threshold_min} to {args.threshold_max}")
    print("=" * 60)
    
    # Run experiments for each algorithm
    for algorithm in algorithms:
        print(f"\nRunning {algorithm}")
        
        # Set up extra parameters for this algorithm
        extra_params = {}
        if algorithm == 'xumei':
            extra_params['lower_bound'] = args.xumei_lb

        # Run experiment
        arl_means, arl_stds, all_times = run_arl_experiment(
            algorithm, Ms, thresholds, args.T, args.sims, args.workers, extra_params
        )
        
        # Save results if requested
        if args.save:
            save_arl_results(algorithm, Ms, thresholds, arl_means, arl_stds, 
                            args.T, args.sims, all_times, extra_params, args.workers, args.data_dir)
    
    print(f"\nCompleted ARL experiments for all {len(algorithms)} algorithms!")
    return

if __name__ == "__main__":
    main() 