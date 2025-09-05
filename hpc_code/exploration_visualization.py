#!/usr/bin/env python3
"""
Visualization script for FOCuS algorithm exploration behavior.
Shows epsilon(t) over time, change point estimates, and stream selection patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import os
import matplot2tikz
plt.style.use('ggplot')

def generate_streaming_observation(stream, time, nu, mu1, M, mu0=0):
    """
    Generate a single observation for streaming algorithms.
    
    Parameters:
    -----------
    stream : int
        Stream index (stream 1 has change point, others don't)
    time : int
        Current time step
    nu : int
        Change point location
    mu1 : float
        Post-change mean
    M : int
        Total number of streams
    mu0 : float
        Pre-change mean (default: 0)
        
    Returns:
    --------
    float : Single observation
    """
    if stream == 0:
        # Stream 0 (0-indexed) has the change point
        if time < nu:
            return np.random.normal(mu0, 1)  # Pre-change
        else:
            return np.random.normal(mu1, 1)  # Post-change
    else:
        # Other streams have no change
        return np.random.normal(mu0, 1)

def focus_decay_with_tracking(M, T, nu, mu1, threshold):
    """
    Modified FOCuS algorithm that tracks exploration behavior and estimates.
    
    Returns:
    --------
    dict: Contains tracking data for visualization
    """
    # Initialize algorithm state
    S = np.zeros(M)
    N = np.zeros(M)
    quadratics_positive = [[(0,0,0,0)] for m in range(M)]
    quadratics_negative = [[(0,0,0,0)] for m in range(M)]
    glr_previous = np.zeros(M)
    v_previous = np.zeros(M)
    
    # Tracking arrays
    tracking = {
        'time': [],
        'epsilon': [],
        'selected_stream': [],
        'exploration_flag': [],
        'glr_stats': [],
        'change_point_estimates': [],
        'best_stream_estimate': [],
        'observations': []
    }
    
    for t in range(T):
        # Stream selection with epsilon-greedy
        m_t = np.random.choice(np.where(glr_previous == np.max(glr_previous))[0])
        v_t = v_previous[m_t]
        epsilon = min(1, M*(t+1-v_t)**(-1/3))
        exploration = np.random.random() < epsilon
        
        if exploration:
            a_t = np.random.randint(M)
        else:
            a_t = m_t
            
        # Generate single observation on-demand
        X_t = generate_streaming_observation(a_t, t, nu, mu1, M)
        
        # Update statistics
        N[a_t] = N[a_t] + 1
        S[a_t] = S[a_t] + X_t
        
        # Positive update
        k = len(quadratics_positive[a_t])
        quadratic_add = [N[a_t], S[a_t], np.inf, t+1]
        i = k
        while (2*(quadratic_add[1]-quadratics_positive[a_t][i-1][1])-(quadratic_add[0]-quadratics_positive[a_t][i-1][0])*quadratics_positive[a_t][i-1][2])<=0 and i>=1:
            i = i-1
        quadratic_add[2] = max(0, 2*(quadratic_add[0]-quadratics_positive[a_t][i-1][0])/(quadratic_add[3]-quadratics_positive[a_t][i-1][3]))
        quadratics_positive[a_t] = quadratics_positive[a_t][:i].copy()
        quadratics_positive[a_t].append(tuple(quadratic_add))
        
        # Negative update
        k = len(quadratics_negative[a_t])
        quadratic_add = [N[a_t], S[a_t], -np.inf, t+1]
        i = k
        while (2*(quadratic_add[1]-quadratics_negative[a_t][i-1][1])-(quadratic_add[0]-quadratics_negative[a_t][i-1][0])*quadratics_negative[a_t][i-1][2])>=0 and i>=1:
            i = i-1
        quadratic_add[2] = min(0, 2*(quadratic_add[0]-quadratics_negative[a_t][i-1][0])/(quadratic_add[3]-quadratics_negative[a_t][i-1][3]))
        quadratics_negative[a_t] = quadratics_negative[a_t][:i].copy()
        quadratics_negative[a_t].append(tuple(quadratic_add))
        
        # Calculate GLR statistics using the same method as original FOCuS
        glr_current = glr_previous.copy()  # Start with previous values
        change_point_estimates = np.zeros(M)
        
        # Update GLR for the selected stream (a_t) only
        best_glr = 0
        best_tau = t+1
        for quadratic in quadratics_positive[a_t] + quadratics_negative[a_t]:
            if N[a_t]-quadratic[0]>0 and (S[a_t]-quadratic[1])**2/(2*(N[a_t]-quadratic[0]))>best_glr:
                best_glr = (S[a_t]-quadratic[1])**2/(2*(N[a_t]-quadratic[0]))
                best_tau = quadratic[3]
        
        glr_current[a_t] = best_glr
        
        # Simple change point estimates for all streams
        for m in range(M):
            if N[m] > 0:
                change_point_estimates[m] = max(1, t+1 - N[m] + 1)  # Rough estimate
            else:
                change_point_estimates[m] = t+1
        
        # Update for next iteration
        glr_previous = glr_current.copy()
        if best_glr > 0:
            v_previous[a_t] = best_tau
        
        # Store tracking information
        tracking['time'].append(t+1)
        tracking['epsilon'].append(epsilon)
        tracking['selected_stream'].append(a_t)
        tracking['exploration_flag'].append(exploration)
        tracking['glr_stats'].append(glr_current.copy())
        tracking['change_point_estimates'].append(change_point_estimates.copy())
        tracking['best_stream_estimate'].append(np.argmax(glr_current))
        tracking['observations'].append(X_t)
        
        # Debug output for first few steps
        if t < 10:
            print(f"t={t+1}: m_t={m_t}, v_t={v_t}, epsilon={epsilon:.3f}, exploration={exploration}, a_t={a_t}, GLR_max={np.max(glr_current):.3f}")
        
        # Check for detection
        if np.max(glr_current) >= threshold:
            detected_stream = np.argmax(glr_current)
            detected_time = t + 1
            estimated_changepoint = change_point_estimates[detected_stream]
            return detected_time, detected_stream, estimated_changepoint, tracking
    
    # No detection
    return T, -1, -1, tracking

def plot_exploration_analysis(tracking, M, T, nu, mu1, threshold, detection_time=None, detected_stream=None, estimated_changepoint=None):
    """
    Create 4 separate plots of the algorithm behavior and save as TikZ files.
    """
    time_steps = np.array(tracking['time'])
    stream_colors = plt.cm.Set1(np.linspace(0, 1, M))
    
    # Create output directory
    output_dir = 'example_tikz'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Exploration probability epsilon(t)
    fig1 = plt.figure(figsize=(10, 4))
    plt.plot(time_steps, tracking['epsilon'], 'b-', linewidth=2, label='epsilon(t)')
    plt.axvline(x=nu, color='red', linestyle='--', alpha=0.7, label=f'True changepoint (t={nu})')
    if detection_time and detection_time < T:
        plt.axvline(x=detection_time, color='green', linestyle='--', alpha=0.7, label=f'Detection (t={detection_time})')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Exploration Probability epsilon(t)', fontsize=12)
    plt.title(f'Exploration Probability: M={M}, mu1={mu1}, nu={nu}, lambda={threshold}', fontsize=14)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    
    # Save as TikZ
    tikz_filename1 = os.path.join(output_dir, f'exploration_probability_{timestamp}.tex')
    try:
        matplot2tikz.save(tikz_filename1, extra_axis_parameters={'width=\\figW', 'height=\\figH'}, encoding='utf-8')
        print(f"Saved: {tikz_filename1}")
    except UnicodeEncodeError:
        print(f"Warning: Could not save {tikz_filename1} due to encoding issues")
    plt.show()
    
    # Plot 2: Stream selection over time
    fig2 = plt.figure(figsize=(10, 4))
    for m in range(M):
        stream_times = [t for t, s in zip(time_steps, tracking['selected_stream']) if s == m]
        stream_y = [m] * len(stream_times)
        plt.scatter(stream_times, stream_y, c=[stream_colors[m]], alpha=0.6, s=20, label=f'Stream {m+1}')
    
    plt.axvline(x=nu, color='red', linestyle='--', alpha=0.7, label=f'True changepoint (t={nu})')
    if detection_time and detection_time < T:
        plt.axvline(x=detection_time, color='green', linestyle='--', alpha=0.7, label=f'Detection (t={detection_time})')
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Selected Stream', fontsize=12)
    plt.title('Stream Selection Over Time', fontsize=14)
    plt.yticks(list(range(M)), [f'Stream {m+1}' for m in range(M)])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save as TikZ
    tikz_filename2 = os.path.join(output_dir, f'stream_selection_{timestamp}.tex')
    try:
        matplot2tikz.save(tikz_filename2, extra_axis_parameters={'width=\\figW', 'height=\\figH'}, encoding='utf-8')
        print(f"Saved: {tikz_filename2}")
    except UnicodeEncodeError:
        print(f"Warning: Could not save {tikz_filename2} due to encoding issues")
    plt.show()
    
    # Plot 3: GLR statistics over time
    fig3 = plt.figure(figsize=(10, 4))
    glr_matrix = np.array(tracking['glr_stats'])
    for m in range(M):
        color = 'red' if m == 0 else stream_colors[m]  # Highlight the true change stream
        linewidth = 3 if m == 0 else 1
        plt.plot(time_steps, glr_matrix[:, m], color=color, linewidth=linewidth, 
                label=f'Stream {m+1}' + (' (True change)' if m == 0 else ''))
    
    plt.axhline(y=threshold, color='black', linestyle='-', alpha=0.8, label=f'Threshold lambda={threshold}')
    plt.axvline(x=nu, color='red', linestyle='--', alpha=0.7, label=f'True changepoint (t={nu})')
    if detection_time and detection_time < T:
        plt.axvline(x=detection_time, color='green', linestyle='--', alpha=0.7, label=f'Detection (t={detection_time})')
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('GLR Statistics', fontsize=12)
    plt.title('GLR Statistics Over Time', fontsize=14)
    plt.legend()
    plt.yscale('log')
    plt.tight_layout()
    
    # Save as TikZ
    tikz_filename3 = os.path.join(output_dir, f'glr_statistics_{timestamp}.tex')
    try:
        matplot2tikz.save(tikz_filename3, extra_axis_parameters={'width=\\figW', 'height=\\figH'}, encoding='utf-8')
        print(f"Saved: {tikz_filename3}")
    except UnicodeEncodeError:
        print(f"Warning: Could not save {tikz_filename3} due to encoding issues")
    plt.show()
    
    # Plot 4: Change point estimates
    fig4 = plt.figure(figsize=(10, 4))
    cp_matrix = np.array(tracking['change_point_estimates'])
    for m in range(M):
        color = 'red' if m == 0 else stream_colors[m]
        linewidth = 3 if m == 0 else 1
        plt.plot(time_steps, cp_matrix[:, m], color=color, linewidth=linewidth, 
                label=f'Stream {m+1}' + (' (True change)' if m == 0 else ''))
    
    plt.axhline(y=nu, color='red', linestyle='--', alpha=0.7, label=f'True changepoint (t={nu})')
    if detection_time and detection_time < T:
        plt.axvline(x=detection_time, color='green', linestyle='--', alpha=0.7, label=f'Detection (t={detection_time})')
        if estimated_changepoint:
            plt.axhline(y=estimated_changepoint, color='green', linestyle=':', alpha=0.7, 
                      label=f'Estimated changepoint (t={estimated_changepoint})')
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Estimated Change Point', fontsize=12)
    plt.title('Change Point Estimates Over Time', fontsize=14)
    plt.legend()
    plt.tight_layout()
    
    # Save as TikZ
    tikz_filename4 = os.path.join(output_dir, f'changepoint_estimates_{timestamp}.tex')
    try:
        matplot2tikz.save(tikz_filename4, extra_axis_parameters={'width=\\figW', 'height=\\figH'}, encoding='utf-8')
        print(f"Saved: {tikz_filename4}")
    except UnicodeEncodeError:
        print(f"Warning: Could not save {tikz_filename4} due to encoding issues")
    plt.show()
    
    print(f"\nAll TikZ files saved in '{output_dir}/' directory")
    print("To use in LaTeX, define \\figW and \\figH macros, e.g.:")
    print("\\newcommand{\\figW}{0.8\\textwidth}")
    print("\\newcommand{\\figH}{0.3\\textheight}")
    
    return [fig1, fig2, fig3, fig4]

def main():
    """
    Run the exploration visualization with specified parameters.
    """
    # Parameters as requested
    M = 5          # Number of streams
    mu1 = 1.0      # Post-change mean
    nu = 400       # Change point time
    T = 3000        # Total time steps (run past change point)
    threshold = 400.0  # Detection threshold
    
    print(f"=== FOCuS Exploration Visualization ===")
    print(f"Parameters: M={M}, μ₁={mu1}, ν={nu}, T={T}, λ={threshold}")
    print(f"True change stream: 1 (1-indexed, stream 0 in 0-indexed)")
    print()
    
    # Set random seed for reproducible results
    # np.random.seed(42)
    
    # Run the algorithm with tracking
    print("Running FOCuS algorithm with tracking...")
    detection_time, detected_stream, estimated_changepoint, tracking = focus_decay_with_tracking(
        M, T, nu, mu1, threshold
    )
    
    # Print results
    print(f"Results:")
    if detection_time < T:
        print(f"  Detection time: {detection_time}")
        print(f"  Detected stream: {detected_stream+1} (True: 1)")
        print(f"  Estimated changepoint: {estimated_changepoint} (True: {nu})")
        print(f"  Stream detection accuracy: {'✓' if detected_stream == 0 else '✗'}")
        print(f"  Changepoint estimation error: {abs(estimated_changepoint - nu)}")
    else:
        print(f"  No detection within T={T} time steps")
    
    # Create visualization
    print("\nCreating visualization...")
    figures = plot_exploration_analysis(
        tracking, M, T, nu, mu1, threshold, 
        detection_time if detection_time < T else None,
        detected_stream if detection_time < T else None,
        estimated_changepoint if detection_time < T else None
    )
    
    # Note: Individual plots are already saved as TikZ files and shown by plot_exploration_analysis()
    print("Individual plots have been displayed and TikZ files saved.")
    
    # Print some exploration statistics
    total_steps = len(tracking['exploration_flag'])
    exploration_count = sum(tracking['exploration_flag'])
    exploitation_count = total_steps - exploration_count
    
    print(f"\nExploration Statistics:")
    print(f"  Total steps: {total_steps}")
    print(f"  Exploration steps: {exploration_count} ({100*exploration_count/total_steps:.1f}%)")
    print(f"  Exploitation steps: {exploitation_count} ({100*exploitation_count/total_steps:.1f}%)")
    print(f"  Average ε(t): {np.mean(tracking['epsilon']):.3f}")
    print(f"  Final ε(t): {tracking['epsilon'][-1]:.3f}")

if __name__ == "__main__":
    main()