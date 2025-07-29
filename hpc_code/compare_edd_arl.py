#!/usr/bin/env python3
"""
Comprehensive comparison script for change point detection algorithms.
Runs both EDD and ARL experiments for two algorithms across a range of thresholds.
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

# All streaming algorithms now centralized in focus_implementation.py
# Use: focus.focus_decay_streaming(), focus.focus_nonuhat_streaming(), focus.xumei_streaming()

def run_single_edd_simulation_streaming(args):
    """Run a single EDD simulation using centralized streaming implementations."""
    algorithm, M, T, nu, mu1, threshold, seed, extra_params = args
    
    np.random.seed(seed)
    
    try:
        if algorithm == 'focus_decay':
            result = focus.focus_decay_streaming(M, T, nu, mu1, threshold)
        elif algorithm == 'focus_oracle':
            result = focus.focus_oracle_streaming(M, T, nu, mu1, threshold)
        elif algorithm == 'focus_nonuhat':
            result = focus.focus_nonuhat_streaming(M, T, nu, mu1, threshold)
        elif algorithm == 'xumei':
            lower_bound = extra_params.get('lower_bound', 0.1)
            result = focus.xumei_streaming(M, T, nu, mu1, threshold, lower_bound)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        if result is None:
            return None
        detection_time, _ = result
        return detection_time
    except Exception as e:
        print(f"Error in streaming EDD simulation with seed {seed}: {e}")
        return None

def run_single_arl_simulation_streaming(args):
    """Run a single ARL simulation using centralized streaming implementations."""
    algorithm, M, T, threshold, seed, extra_params = args
    
    np.random.seed(seed)
    
    try:
        if algorithm == 'focus_decay':
            result = focus.focus_decay_streaming(M, T, nu=0, mu1=0, threshold=threshold)
        elif algorithm == 'focus_nonuhat':
            result = focus.focus_nonuhat_streaming(M, T, nu=0, mu1=0, threshold=threshold)
        elif algorithm == 'xumei':
            lower_bound = extra_params.get('lower_bound', 0.1)
            result = focus.xumei_streaming(M, T, nu=0, mu1=0, threshold=threshold, lower_bound=lower_bound)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        if result is None:
            return T  # No detection
        detection_time, _ = result
        return detection_time
    except Exception as e:
        print(f"Error in streaming ARL simulation with seed {seed}: {e}")
        return None

def run_single_edd_simulation(args):
    """COMPATIBILITY WRAPPER: Uses streaming implementation."""
    return run_single_edd_simulation_streaming(args)

def run_single_arl_simulation(args):
    """COMPATIBILITY WRAPPER: Uses streaming implementation.""" 
    return run_single_arl_simulation_streaming(args)

def run_comparison_experiment(algorithms, thresholds, nus=[0, 1000, 10000], Ms=[1, 3, 5, 10],
                             M_edd=10, T=int(1e6), mu1=1.0, sims=50, n_workers=None, 
                             extra_params=None):
    """
    Run comprehensive comparison experiment.
    
    Parameters:
    -----------
    algorithms : list
        List of two algorithm names to compare
    thresholds : list
        List of thresholds to test
    nus : list
        Change point locations for EDD experiments
    Ms : list
        Number of streams for ARL experiments
    M_edd : int
        Number of streams for EDD experiments
    T : int
        Time horizon
    mu1 : float
        Post-change mean for EDD
    sims : int
        Number of simulations per configuration
    n_workers : int
        Number of parallel workers
    extra_params : dict
        Extra parameters for algorithms
    
    Returns:
    --------
    results : dict
        Comprehensive results for both algorithms
    """
    
    if n_workers is None:
        n_workers = cpu_count()
    
    if extra_params is None:
        extra_params = {}
    
    print(f"Running comprehensive comparison: {algorithms}")
    print(f"Thresholds: {thresholds}")
    print(f"EDD: M={M_edd}, nus={nus}")
    print(f"ARL: Ms={Ms}")
    print(f"Parameters: T={T}, mu1={mu1}, sims={sims}")
    print(f"Workers: {n_workers}")
    print(f"Optimization: STREAMING (memory-optimized, early stopping)")
    print("=" * 60)
    
    results = {}
    
    for algorithm in algorithms:
        print(f"\nRunning {algorithm}...")
        
        # Get algorithm-specific parameters
        alg_params = extra_params.get(algorithm, {})
        
        # Initialize results for this algorithm
        results[algorithm] = {
            'edd': {
                'means': np.zeros((len(nus), len(thresholds))),
                'stds': np.zeros((len(nus), len(thresholds))),
                'all_times': [[[] for _ in thresholds] for _ in nus]
            },
            'arl': {
                'means': np.zeros((len(Ms), len(thresholds))),
                'stds': np.zeros((len(Ms), len(thresholds))),
                'all_times': [[[] for _ in thresholds] for _ in Ms]
            }
        }
        
        # Run EDD experiments
        print(f"  EDD Experiments...")
        for i, nu in enumerate(nus):
            for j, threshold in enumerate(thresholds):
                print(f"    EDD: nu={nu}, threshold={threshold}")
                start_time = time.time()
                
                # Prepare EDD simulation arguments
                sim_args = []
                for sim in range(sims):
                    seed = np.random.randint(0, 2**31)
                    sim_args.append((algorithm, M_edd, T, nu, mu1, threshold, seed, alg_params))
                
                # Run EDD simulations with full parallelization (no memory limits)
                delays = []
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [executor.submit(run_single_edd_simulation_streaming, args) for args in sim_args]
                    
                    for future in as_completed(futures):
                        detection_time = future.result()
                        if detection_time is not None and detection_time > nu:
                            delays.append(detection_time - nu)
                
                # Store results
                results[algorithm]['edd']['all_times'][i][j] = delays
                if len(delays) > 0:
                    results[algorithm]['edd']['means'][i, j] = np.mean(delays)
                    results[algorithm]['edd']['stds'][i, j] = np.std(delays)
                else:
                    results[algorithm]['edd']['means'][i, j] = np.nan
                    results[algorithm]['edd']['stds'][i, j] = np.nan
                
                elapsed = time.time() - start_time
                success_rate = len(delays) / sims
                print(f"      Success: {success_rate:.1%}, EDD: {results[algorithm]['edd']['means'][i, j]:.1f}, Time: {elapsed:.1f}s")
        
        # Run ARL experiments
        print(f"  ARL Experiments...")
        for i, M in enumerate(Ms):
            for j, threshold in enumerate(thresholds):
                print(f"    ARL: M={M}, threshold={threshold}")
                start_time = time.time()
                
                # NO MORE WORKER ADJUSTMENTS - streaming eliminates memory constraints
                print(f"      Using {n_workers} workers (streaming optimized)")
                
                # Prepare ARL simulation arguments
                sim_args = []
                for sim in range(sims):
                    seed = np.random.randint(0, 2**31)
                    sim_args.append((algorithm, M, T, threshold, seed, alg_params))
                
                # Run ARL simulations with full parallelization
                run_lengths = []
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    futures = [executor.submit(run_single_arl_simulation_streaming, args) for args in sim_args]
                    
                    for future in as_completed(futures):
                        detection_time = future.result()
                        if detection_time is not None:
                            run_lengths.append(detection_time)
                
                # Store results
                results[algorithm]['arl']['all_times'][i][j] = run_lengths
                if len(run_lengths) > 0:
                    results[algorithm]['arl']['means'][i, j] = np.mean(run_lengths)
                    results[algorithm]['arl']['stds'][i, j] = np.std(run_lengths)
                else:
                    results[algorithm]['arl']['means'][i, j] = np.nan
                    results[algorithm]['arl']['stds'][i, j] = np.nan
                
                elapsed = time.time() - start_time
                success_rate = len(run_lengths) / sims
                print(f"      Success: {success_rate:.1%}, ARL: {results[algorithm]['arl']['means'][i, j]:.1f}, Time: {elapsed:.1f}s")
                
                # Show early stopping benefits
                if len(run_lengths) > 0:
                    avg_stop_time = np.mean(run_lengths)
                    speedup = T / avg_stop_time if avg_stop_time > 0 else 1
                    print(f"      Early stop avg: {avg_stop_time:.0f} (speedup: {speedup:.1f}x)")
    
    return results

def save_comparison_results(algorithms, thresholds, nus, Ms, results, T, sims, 
                           extra_params, n_workers, M_edd, mu1):
    """Save comprehensive comparison results using pickle."""
    
    # Create data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare comprehensive results dictionary
    results_data = {
        'metadata': {
            'algorithms': algorithms,
            'timestamp': timestamp,
            'T': T,
            'sims': sims,
            'n_workers': n_workers,
            'M_edd': M_edd,
            'mu1': mu1,
            'extra_params': extra_params
        },
        'parameters': {
            'thresholds': thresholds,
            'nus': nus,
            'Ms': Ms,
            'M_edd': M_edd,
            'mu1': mu1
        },
        'results': results,
        'statistics': {}
    }
    
    # Calculate additional statistics for each algorithm
    for algorithm in algorithms:
        results_data['statistics'][algorithm] = {
            'edd': {
                'success_rates': np.zeros((len(nus), len(thresholds))),
                'min_times': np.zeros((len(nus), len(thresholds))),
                'max_times': np.zeros((len(nus), len(thresholds))),
                'median_times': np.zeros((len(nus), len(thresholds)))
            },
            'arl': {
                'success_rates': np.zeros((len(Ms), len(thresholds))),
                'min_times': np.zeros((len(Ms), len(thresholds))),
                'max_times': np.zeros((len(Ms), len(thresholds))),
                'median_times': np.zeros((len(Ms), len(thresholds)))
            }
        }
        
        # EDD statistics
        for i, nu in enumerate(nus):
            for j, threshold in enumerate(thresholds):
                times = results[algorithm]['edd']['all_times'][i][j]
                if len(times) > 0:
                    results_data['statistics'][algorithm]['edd']['success_rates'][i, j] = len(times) / sims
                    results_data['statistics'][algorithm]['edd']['min_times'][i, j] = np.min(times)
                    results_data['statistics'][algorithm]['edd']['max_times'][i, j] = np.max(times)
                    results_data['statistics'][algorithm]['edd']['median_times'][i, j] = np.median(times)
                else:
                    results_data['statistics'][algorithm]['edd']['success_rates'][i, j] = 0.0
                    results_data['statistics'][algorithm]['edd']['min_times'][i, j] = np.nan
                    results_data['statistics'][algorithm]['edd']['max_times'][i, j] = np.nan
                    results_data['statistics'][algorithm]['edd']['median_times'][i, j] = np.nan
        
        # ARL statistics
        for i, M in enumerate(Ms):
            for j, threshold in enumerate(thresholds):
                times = results[algorithm]['arl']['all_times'][i][j]
                if len(times) > 0:
                    results_data['statistics'][algorithm]['arl']['success_rates'][i, j] = len(times) / sims
                    results_data['statistics'][algorithm]['arl']['min_times'][i, j] = np.min(times)
                    results_data['statistics'][algorithm]['arl']['max_times'][i, j] = np.max(times)
                    results_data['statistics'][algorithm]['arl']['median_times'][i, j] = np.median(times)
                else:
                    results_data['statistics'][algorithm]['arl']['success_rates'][i, j] = 0.0
                    results_data['statistics'][algorithm]['arl']['min_times'][i, j] = np.nan
                    results_data['statistics'][algorithm]['arl']['max_times'][i, j] = np.nan
                    results_data['statistics'][algorithm]['arl']['median_times'][i, j] = np.nan
    
    # Create filename
    alg_str = "_vs_".join(algorithms)
    thresh_str = f"thresh{thresholds[0]:.0f}-{thresholds[-1]:.0f}"
    filename = f"comparison_{alg_str}_{thresh_str}_T{T}_sims{sims}_{timestamp}.pkl"
    filepath = os.path.join(data_dir, filename)
    
    # Save using pickle
    with open(filepath, 'wb') as f:
        pickle.dump(results_data, f)
    
    print(f"\nComprehensive comparison results saved to: {filepath}")
    print(f"File size: {os.path.getsize(filepath) / 1024:.1f} KB")
    
    return filepath

def main():
    parser = argparse.ArgumentParser(description='Compare two algorithms with EDD and ARL experiments')
    parser.add_argument('--algorithms', type=str, required=True,
                       help='Comma-separated pair of algorithms (e.g., focus_decay,xumei)')
    parser.add_argument('--threshold-min', type=float, default=1000,
                       help='Minimum threshold to test')
    parser.add_argument('--threshold-max', type=float, default=5000,
                       help='Maximum threshold to test')
    parser.add_argument('--threshold-steps', type=int, default=5,
                       help='Number of threshold steps')
    parser.add_argument('--nus', type=str, default='0,1000,10000',
                       help='Change point locations for EDD')
    parser.add_argument('--Ms', type=str, default='1,3,5,10',
                       help='Number of streams for ARL')
    parser.add_argument('--M-edd', type=int, default=10, help='Number of streams for EDD')
    parser.add_argument('--T', type=int, default=1000000, help='Time horizon')
    parser.add_argument('--mu1', type=float, default=1.0, help='Post-change mean')
    parser.add_argument('--sims', type=int, default=50, help='Number of simulations')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers')
    parser.add_argument('--xumei-lb', type=float, default=0.1, 
                       help='Lower bound for xumei algorithm')
    parser.add_argument('--save', action='store_true', help='Save results to files')
    
    args = parser.parse_args()
    
    # Parse algorithms
    algorithms = [x.strip() for x in args.algorithms.split(',')]
    if len(algorithms) != 2:
        raise ValueError("Must specify exactly two algorithms")
    
    # Generate threshold range
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    
    # Parse other parameters
    nus = [float(x) for x in args.nus.split(',')]
    Ms = [int(x) for x in args.Ms.split(',')]
    
    # Set up extra parameters
    extra_params = {}
    for alg in algorithms:
        if alg == 'xumei':
            extra_params[alg] = {'lower_bound': args.xumei_lb}
        else:
            extra_params[alg] = {}
    
    print(f"Starting comprehensive comparison...")
    print(f"Algorithms: {algorithms}")
    print(f"Thresholds: {len(thresholds)} steps from {args.threshold_min} to {args.threshold_max}")
    
    # Run comparison
    results = run_comparison_experiment(
        algorithms=algorithms,
        thresholds=thresholds,
        nus=nus,
        Ms=Ms,
        M_edd=args.M_edd,
        T=args.T,
        mu1=args.mu1,
        sims=args.sims,
        n_workers=args.workers,
        extra_params=extra_params
    )
    
    # Save results
    if args.save:
        save_comparison_results(algorithms, thresholds, nus, Ms, results, args.T, args.sims,
                               extra_params, args.workers, args.M_edd, args.mu1)
    
    print("\nComprehensive comparison completed!")
    return results

if __name__ == "__main__":
    main() 