#!/usr/bin/env python3
"""
Script to load and plot ARL results from pickle files.
Compares different algorithms and parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
from pathlib import Path

def load_pickle_file(filepath):
    """Load a pickle file and return the data."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def get_data_files(data_dir="hpc_code/data"):
    """Get all pickle files in the data directory."""
    pattern = os.path.join(data_dir, "*.pkl")
    return glob.glob(pattern)

def extract_algorithm_info(filename):
    """Extract algorithm and parameter info from filename."""
    # Example: arl_xumei_M1_M3_M5_M10_thresholds5.00_6.00_7.00_T200000_sims10_20250724_144100.pkl
    parts = Path(filename).stem.split('_')
    
    # Find algorithm name
    if 'arl_' in parts[0]:
        algorithm = parts[1]
    else:
        algorithm = parts[0]
    
    # Extract M values
    M_values = []
    for part in parts:
        if part.startswith('M') and part[1:].isdigit():
            M_values.append(int(part[1:]))
    
    # Extract thresholds
    thresholds = []
    in_thresholds = False
    for part in parts:
        if part == 'thresholds':
            in_thresholds = True
            continue
        elif in_thresholds and part.replace('.', '').replace('-', '').isdigit():
            thresholds.append(float(part))
        elif in_thresholds and not part.replace('.', '').replace('-', '').isdigit():
            break
    
    return algorithm, M_values, thresholds

def plot_arl_comparison(data_files, M_to_plot=1, save_plot=True):
    """
    Create ARL comparison plot for a specific number of streams.
    
    Parameters:
    -----------
    data_files : list
        List of pickle file paths
    M_to_plot : int
        Number of streams to plot (will use closest available)
    save_plot : bool
        Whether to save the plot
    """
    
    plt.figure(figsize=(12, 8))
    
    # Colors for different algorithms
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    legend_entries = []
    
    for i, filepath in enumerate(data_files):
        try:
            # Load data
            data = load_pickle_file(filepath)
            
            # Extract info
            algorithm = data['metadata']['algorithm']
            Ms = data['parameters']['Ms']
            thresholds = data['parameters']['thresholds']
            arl_means = data['results']['arl_means']
            arl_stds = data['results']['arl_stds']
            lower_bound = data['parameters'].get('lower_bound', None)
            
            # Find closest M value
            M_idx = min(range(len(Ms)), key=lambda x: abs(Ms[x] - M_to_plot))
            actual_M = Ms[M_idx]
            
            if abs(actual_M - M_to_plot) > 0.1:  # If not close enough, skip
                print(f"Skipping {algorithm} - closest M is {actual_M}, not {M_to_plot}")
                continue
            
            # Get ARL values for this M
            arl_values = arl_means[M_idx, :]
            
            # Calculate min/max from raw detection times
            detection_times = data['results']['all_detection_times'][M_idx]
            min_values = []
            max_values = []
            for times in detection_times:
                if len(times) > 0:
                    min_values.append(np.min(times))
                    max_values.append(np.max(times))
                else:
                    min_values.append(np.nan)
                    max_values.append(np.nan)
            
            # Calculate error bars as distance from mean to min/max
            yerr_min = arl_values - np.array(min_values)
            yerr_max = np.array(max_values) - arl_values
            yerr = np.vstack([yerr_min, yerr_max])  # Stack for asymmetric error bars
            
            # Create legend label
            if algorithm == 'xumei' and lower_bound is not None:
                label = f"{algorithm} (lb={lower_bound})"
            else:
                label = algorithm
            
            # Plot with error bars
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            
            plt.errorbar(thresholds, arl_values, yerr=yerr, 
                        marker=marker, color=color, linewidth=2, markersize=8,
                        capsize=5, capthick=2, label=label)
            
            legend_entries.append(label)
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Average Run Length (ARL)', fontsize=14)
    plt.title(f'ARL Comparison for M={M_to_plot} Streams', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale often works better for ARL
    
    # Add some padding
    plt.tight_layout()
    
    if save_plot:
        plot_filename = f"arl_comparison_M{M_to_plot}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}")
    
    plt.show()

def plot_all_M_values(data_files, save_plots=True):
    """
    Create plots for all available M values.
    
    Parameters:
    -----------
    data_files : list
        List of pickle file paths
    save_plots : bool
        Whether to save the plots
    """
    
    # Find all unique M values across all files
    all_M_values = set()
    for filepath in data_files:
        try:
            data = load_pickle_file(filepath)
            all_M_values.update(data['parameters']['Ms'])
        except:
            continue
    
    all_M_values = sorted(list(all_M_values))
    
    print(f"Found M values: {all_M_values}")
    
    # Create plots for each M value
    for M in all_M_values:
        print(f"\nCreating plot for M={M}")
        plot_arl_comparison(data_files, M, save_plots)

def print_summary(data_files):
    """Print a summary of all available data files."""
    print("Available ARL Results:")
    print("=" * 50)
    
    for filepath in data_files:
        try:
            data = load_pickle_file(filepath)
            
            algorithm = data['metadata']['algorithm']
            Ms = data['parameters']['Ms']
            thresholds = data['parameters']['thresholds']
            sims = data['metadata']['sims']
            T = data['metadata']['T']
            lower_bound = data['parameters'].get('lower_bound', None)
            
            print(f"\nAlgorithm: {algorithm}")
            print(f"  M values: {Ms}")
            print(f"  Thresholds: {thresholds}")
            print(f"  Simulations: {sims}")
            print(f"  Time horizon: {T}")
            if lower_bound is not None:
                print(f"  Lower bound: {lower_bound}")
            print(f"  File: {os.path.basename(filepath)}")
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

def main():
    """Main function to run the plotting script."""
    
    # Get all data files
    data_files = get_data_files()
    
    if not data_files:
        print("No pickle files found in data directory!")
        return
    
    print(f"Found {len(data_files)} data files")
    
    # Print summary
    print_summary(data_files)
    
    # Ask user what to plot
    print("\n" + "=" * 50)
    print("Plotting Options:")
    print("1. Plot for specific M value")
    print("2. Plot for all M values")
    print("3. Just print summary")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        M_value = int(input("Enter M value to plot: "))
        plot_arl_comparison(data_files, M_value)
    elif choice == "2":
        plot_all_M_values(data_files)
    elif choice == "3":
        print("Summary printed above.")
    else:
        print("Invalid choice. Creating default plot for M=1...")
        plot_arl_comparison(data_files, 1)

if __name__ == "__main__":
    main() 