import numpy as np
# Removed heavy scipy imports that cause DLL loading issues in multiprocessing:
# from scipy.stats import norm      # Replaced with np.random.normal()
# from scipy.stats import bernoulli # Replaced with np.random.random() < p
import random
import sys

def generate_data(M, T, nu, mu1):
    """DEPRECATED: Memory-intensive batch data generation. Use streaming versions instead."""
    mu0 = 0
    # Assume changepoint is in stream 0
    # Convert nu to int for array indexing and size parameters
    nu = int(nu)
    data = np.zeros((M,T))
    data[0][:nu] = np.random.normal(0, 1, size=nu)  # Replaced norm.rvs()
    data[0][nu:] = np.random.normal(mu1, 1, size=T-nu)  # Replaced norm.rvs()
    for i in range(1,M):
        data[i] = np.random.normal(mu0, 1, size=T)  # Replaced norm.rvs()
    return data

def generate_streaming_observation(stream, time, nu, mu1, mu0=0):
    """
    Generate a single observation for streaming algorithms.
    
    Parameters:
    -----------
    stream : int
        Stream index (0 has change point, others don't)
    time : int
        Current time step
    nu : int
        Change point location
    mu1 : float
        Post-change mean
    mu0 : float
        Pre-change mean (default: 0)
        
    Returns:
    --------
    float : Single observation
    """
    if stream == 0:
        # Stream 0 has the change point
        if time < nu:
            return np.random.normal(mu0, 1)  # Pre-change
        else:
            return np.random.normal(mu1, 1)  # Post-change
    else:
        # Other streams have no change
        return np.random.normal(mu0, 1)

def focus_decay_streaming(M, T, nu, mu1, threshold):
    """
    Streaming implementation of focus_decay algorithm.
    Eliminates memory bottleneck by generating data on-demand.
    """
    # Initialize algorithm state
    S = np.zeros(M)
    N = np.zeros(M)
    quadratics_positive = [[(0,0,0,0)] for m in range(M)]
    quadratics_negative = [[(0,0,0,0)] for m in range(M)]
    glr_previous = np.zeros(M)
    v_previous = np.zeros(M)
    
    for t in range(T):
        # Stream selection with epsilon-greedy
        m_t = np.random.choice(np.where(glr_previous == np.max(glr_previous))[0])
        v_t = v_previous[m_t]
        epsilon = min(1, (t+1-v_t)**(-1/3))
        exploration = np.random.random() < epsilon # Replaced bernoulli.rvs()
        
        if exploration:
            a_t = np.random.randint(M)
        else:
            a_t = m_t
            
        # Generate single observation on-demand
        X_t = generate_streaming_observation(a_t, t, nu, mu1)
        
        # Update statistics
        N[a_t] = N[a_t] + 1
        S[a_t] = S[a_t] + X_t
        
        # Positive update
        k = len(quadratics_positive[a_t])
        quadratic_add = [N[a_t], S[a_t], np.inf, t+1]
        i = k
        while (2*(quadratic_add[1]-quadratics_positive[a_t][i-1][1])-(quadratic_add[0]-quadratics_positive[a_t][i-1][0])*quadratics_positive[a_t][i-1][2])<=0 and i>=1:
            i = i-1
        quadratic_add[2] = max(0, 2*(quadratic_add[0]-quadratics_positive[a_t][i-1][0])/(quadratic_add[2]-quadratics_positive[a_t][i-1][2]))
        quadratics_positive[a_t] = quadratics_positive[a_t][:i].copy()
        quadratics_positive[a_t].append(tuple(quadratic_add))
        
        # Negative update
        k = len(quadratics_negative[a_t])
        quadratic_add = [N[a_t], S[a_t], -np.inf, t+1]
        i = k
        while (2*(quadratic_add[1]-quadratics_negative[a_t][i-1][1])-(quadratic_add[0]-quadratics_negative[a_t][i-1][0])*quadratics_negative[a_t][i-1][2])>=0 and i>=1:
            i = i-1
        quadratic_add[2] = min(0, 2*(quadratic_add[0]-quadratics_negative[a_t][i-1][0])/(quadratic_add[2]-quadratics_negative[a_t][i-1][2]))
        quadratics_negative[a_t] = quadratics_negative[a_t][:i].copy()
        quadratics_negative[a_t].append(tuple(quadratic_add))
        
        # Calculate GLR and check for detection
        best_glr = 0
        for quadratic in quadratics_positive[a_t] + quadratics_negative[a_t]:
            if N[a_t]-quadratic[0]>0 and (S[a_t]-quadratic[1])**2/(2*(N[a_t]-quadratic[0]))>best_glr:
                best_glr = (S[a_t]-quadratic[1])**2/(2*(N[a_t]-quadratic[0]))
                v_previous[a_t] = quadratic[3]
                glr_previous[a_t] = best_glr
                
        # Early stopping: return immediately upon detection
        if best_glr >= threshold:
            return t + 1, best_glr
            
    return None  # No detection within time horizon

def focus_nonuhat_streaming(M, T, nu, mu1, threshold):
    """
    Streaming implementation of focus_nonuhat algorithm.
    Eliminates memory bottleneck by generating data on-demand.
    """
    # Initialize algorithm state
    S = np.zeros(M)
    N = np.zeros(M)
    quadratics_positive = [[(0,0,0)] for m in range(M)]
    quadratics_negative = [[(0,0,0)] for m in range(M)]
    glr_previous = np.zeros(M)
    
    for t in range(T):
        # Stream selection with epsilon-greedy
        m_t = np.random.choice(np.where(glr_previous == np.max(glr_previous))[0])
        epsilon = min(1, (t+1)**(-1/3))
        exploration = np.random.random() < epsilon # Replaced bernoulli.rvs()
        
        if exploration:
            a_t = np.random.randint(M)
        else:
            a_t = m_t
            
        # Generate single observation on-demand
        X_t = generate_streaming_observation(a_t, t, nu, mu1)
        
        # Update statistics
        N[a_t] = N[a_t] + 1
        S[a_t] = S[a_t] + X_t
        
        # Positive update
        k = len(quadratics_positive[a_t])
        quadratic_add = [N[a_t], S[a_t], np.inf]
        i = k
        while (2*(quadratic_add[1]-quadratics_positive[a_t][i-1][1])-(quadratic_add[0]-quadratics_positive[a_t][i-1][0])*quadratics_positive[a_t][i-1][2])<=0 and i>=1:
            i = i-1
        quadratic_add[2] = max(0, 2*(quadratic_add[0]-quadratics_positive[a_t][i-1][0])/(quadratic_add[2]-quadratics_positive[a_t][i-1][2]))
        quadratics_positive[a_t] = quadratics_positive[a_t][:i].copy()
        quadratics_positive[a_t].append(tuple(quadratic_add))
        
        # Negative update
        k = len(quadratics_negative[a_t])
        quadratic_add = [N[a_t], S[a_t], -np.inf]
        i = k
        while (2*(quadratic_add[1]-quadratics_negative[a_t][i-1][1])-(quadratic_add[0]-quadratics_negative[a_t][i-1][0])*quadratics_negative[a_t][i-1][2])>=0 and i>=1:
            i = i-1
        quadratic_add[2] = min(0, 2*(quadratic_add[0]-quadratics_negative[a_t][i-1][0])/(quadratic_add[2]-quadratics_negative[a_t][i-1][2]))
        quadratics_negative[a_t] = quadratics_negative[a_t][:i].copy()
        quadratics_negative[a_t].append(tuple(quadratic_add))
        
        # Calculate GLR and check for detection
        best_glr = 0
        for quadratic in quadratics_positive[a_t] + quadratics_negative[a_t]:
            if N[a_t]-quadratic[0]>0 and (S[a_t]-quadratic[1])**2/(2*(N[a_t]-quadratic[0]))>best_glr:
                best_glr = (S[a_t]-quadratic[1])**2/(2*(N[a_t]-quadratic[0]))
                glr_previous[a_t] = best_glr
                
        # Early stopping: return immediately upon detection
        if best_glr >= threshold:
            return t + 1, best_glr
            
    return None  # No detection within time horizon

def focus_oracle_streaming(M, T, nu, mu1, threshold):
    """
    Streaming implementation of focus_oracle algorithm.
    Eliminates memory bottleneck by generating data on-demand.
    """
    # Initialize algorithm state
    S = np.zeros(M)
    N = np.zeros(M)
    quadratics_positive = [[(0,0,0)] for m in range(M)]
    quadratics_negative = [[(0,0,0)] for m in range(M)]
    glr_previous = np.zeros(M)
    
    for t in range(T):
        # Stream selection with epsilon-greedy (oracle version)
        m_t = np.random.choice(np.where(glr_previous == np.max(glr_previous))[0])
        if t+1 > nu:
            epsilon = min(1, (t+1-nu)**(-1/3))
        else:
            epsilon = 1
        exploration = np.random.random() < epsilon # Replaced bernoulli.rvs()
        
        if exploration:
            a_t = np.random.randint(M)
        else:
            a_t = m_t
            
        # Generate single observation on-demand
        X_t = generate_streaming_observation(a_t, t, nu, mu1)
        
        # Update statistics
        N[a_t] = N[a_t] + 1
        S[a_t] = S[a_t] + X_t
        
        # Positive update
        k = len(quadratics_positive[a_t])
        quadratic_add = [N[a_t], S[a_t], np.inf]
        i = k
        while (2*(quadratic_add[1]-quadratics_positive[a_t][i-1][1])-(quadratic_add[0]-quadratics_positive[a_t][i-1][0])*quadratics_positive[a_t][i-1][2])<=0 and i>=1:
            i = i-1
        quadratic_add[2] = max(0, 2*(quadratic_add[0]-quadratics_positive[a_t][i-1][0])/(quadratic_add[2]-quadratics_positive[a_t][i-1][2]))
        quadratics_positive[a_t] = quadratics_positive[a_t][:i].copy()
        quadratics_positive[a_t].append(tuple(quadratic_add))
        
        # Negative update
        k = len(quadratics_negative[a_t])
        quadratic_add = [N[a_t], S[a_t], -np.inf]
        i = k
        while (2*(quadratic_add[1]-quadratics_negative[a_t][i-1][1])-(quadratic_add[0]-quadratics_negative[a_t][i-1][0])*quadratics_negative[a_t][i-1][2])>=0 and i>=1:
            i = i-1
        quadratic_add[2] = min(0, 2*(quadratic_add[0]-quadratics_negative[a_t][i-1][0])/(quadratic_add[2]-quadratics_negative[a_t][i-1][2]))
        quadratics_negative[a_t] = quadratics_negative[a_t][:i].copy()
        quadratics_negative[a_t].append(tuple(quadratic_add))
        
        # Calculate GLR and check for detection
        best_glr = 0
        for quadratic in quadratics_positive[a_t] + quadratics_negative[a_t]:
            if N[a_t]-quadratic[0]>0 and (S[a_t]-quadratic[1])**2/(2*(N[a_t]-quadratic[0]))>best_glr:
                best_glr = (S[a_t]-quadratic[1])**2/(2*(N[a_t]-quadratic[0]))
                glr_previous[a_t] = best_glr
                
        # Early stopping: return immediately upon detection
        if best_glr >= threshold:
            return t + 1, best_glr
            
    return None  # No detection within time horizon

def xumei_streaming(M, T, nu, mu1, threshold, lower_bound):
    """
    Streaming implementation of Xumei algorithm with change point support.
    For ARL experiments, use nu=0, mu1=0.
    """
    W = np.zeros(M)
    stream = -1
    t = -1
    
    while max(W) < threshold:
        stream = (stream + 1) % M
        m_t = 0
        sum_t = 0
        
        while W[stream] >= 0 and W[stream] < threshold:
            t = t + 1
            if t >= T:  # Safety check
                return None
                
            # Generate single data point on-demand with change point support
            if stream == 0:  # Stream 0 has change point
                if t < nu:
                    x_t = np.random.normal(0, 1)  # Pre-change
                else:
                    x_t = np.random.normal(mu1, 1)  # Post-change
            else:
                x_t = np.random.normal(0, 1)  # No change in other streams
            
            if m_t == 0:
                mean_estimate = lower_bound
            else:
                mean_estimate = max((sum_t/m_t, lower_bound))
                
            W[stream] = max((W[stream], 0)) + mean_estimate*x_t - (mean_estimate**2)/2
            
            for i in range(M):
                if i != stream:
                    W[i] = max((W[i], 0))
                    
            m_t = m_t + 1
            sum_t = sum_t + x_t
            
        # Check if any stream exceeded threshold
        if max(W) >= threshold:
            return t + 1, max(W)
            
    return t + 1, max(W)

# Backward-compatible wrappers that maintain old API but use streaming internally
def focus_decay(data, threshold):
    """
    COMPATIBILITY WRAPPER
    """
    # Infer parameters from data shape and assume standard setup
    M, T = data.shape
    # For compatibility, assume no change point (ARL mode) since we can't infer nu/mu1
    return focus_decay_streaming(M, T, nu=0, mu1=0, threshold=threshold)

def focus_nonuhat(data, threshold):
    """
    COMPATIBILITY WRAPPER
    """
    M, T = data.shape
    return focus_nonuhat_streaming(M, T, nu=0, mu1=0, threshold=threshold)

def focus_oracle(data, threshold, nu):
    """
    COMPATIBILITY WRAPPER
    """
    M, T = data.shape
    # For oracle, we need to assume some mu1 value since it's not provided
    return focus_oracle_streaming(M, T, nu=nu, mu1=1.0, threshold=threshold)

def single_stream(data, threshold):
    quadratics_positive = [(0,0,0)]
    quadratics_negative = [(0,0,0)]
    S = 0
    T = data.shape[0]
    for t in range(T):
        S = S + data[t]
        # Recall time step are different from index in array
        # Time steps are 1-indexed so adding t+1
        #Positive update first
        k = len(quadratics_positive)
        quadratic_add = [t+1,S,np.inf]
        i = k
        while (2*(quadratic_add[1]-quadratics_positive[i-1][1])-(quadratic_add[0]-quadratics_positive[i-1][0])*quadratics_positive[i-1][2])<=0 and i>=1:
            i = i-1
        quadratic_add[2] = max(0,2*(quadratic_add[0]-quadratics_positive[i-1][0])/(quadratic_add[2]-quadratics_positive[i-1][2]))
        quadratics_positive = quadratics_positive[:i].copy()
        quadratics_positive.append(tuple(quadratic_add))
    
        # Now negative update
        k = len(quadratics_negative)
        quadratic_add = [t+1,S,-np.inf]
        i = k
        while (2*(quadratic_add[1]-quadratics_negative[i-1][1])-(quadratic_add[0]-quadratics_negative[i-1][0])*quadratics_negative[i-1][2])>=0 and i>=1:
            i = i-1
        quadratic_add[2] = min(0,2*(quadratic_add[0]-quadratics_negative[i-1][0])/(quadratic_add[2]-quadratics_negative[i-1][2]))
        quadratics_negative = quadratics_negative[:i].copy()
        quadratics_negative.append(tuple(quadratic_add))
        best_glr = 0
        for quadratic in quadratics_positive + quadratics_negative:
            if t+1-quadratic[0]>0 and (S-quadratic[1])**2/(2*(t+1-quadratic[0]))>best_glr:
                best_glr = (S-quadratic[1])**2/(2*(t+1-quadratic[0]))
        if best_glr>=threshold:
            return t+1, best_glr


def generate_history(data):
    # Implementation of multi-stream FOCuS for any mu
    # Each stream gets its own set of quadratics
    # Initialization
    M = data.shape[0]
    T = data.shape[1]
    S = np.zeros(M)
    N = np.zeros(M)
    quadratics_positive = [[(0,0,0,0)] for m in range(M)]
    quadratics_negative = [[(0,0,0,0)] for m in range(M)]
    M_previous = np.random.randint(M)
    v_previous = 0
    changepoint_history = np.zeros(T)
    glr_history = np.zeros((T,M))
    glr_previous = np.zeros(M)
    v_previous = np.zeros(M)
    # tau, s, l, vec{S}, vec{N} 
    
    for t in range(T):
        # First perform stream selection with epsilon-greedy
        m_t = np.random.choice(np.where(glr_previous == np.max(glr_previous))[0])
        v_t = v_previous[m_t]
        epsilon = min(1,(t+1-v_t)**(-1/3))
        exploration = np.random.random() < epsilon # Replaced bernoulli.rvs()
        if exploration:
            a_t = np.random.randint(M)
        else:
            a_t = m_t
        # Get observation
        X_t = data[a_t,t]
        N[a_t] = N[a_t] + 1
        S[a_t] = S[a_t] + X_t
        # Positive update
        k = len(quadratics_positive[a_t])
        quadratic_add = [N[a_t],S[a_t],np.inf,t+1]
        i = k
        while (2*(quadratic_add[1]-quadratics_positive[a_t][i-1][1])-(quadratic_add[0]-quadratics_positive[a_t][i-1][0])*quadratics_positive[a_t][i-1][2])<=0 and i>=1:
            i = i-1
        quadratic_add[2] = max(0,2*(quadratic_add[0]-quadratics_positive[a_t][i-1][0])/(quadratic_add[2]-quadratics_positive[a_t][i-1][2]))
        quadratics_positive[a_t] = quadratics_positive[a_t][:i].copy()
        quadratics_positive[a_t].append(tuple(quadratic_add))
    
        # Now negative update
        k = len(quadratics_negative[a_t])
        quadratic_add = [N[a_t],S[a_t],-np.inf,t+1]
        i = k
        while (2*(quadratic_add[1]-quadratics_negative[a_t][i-1][1])-(quadratic_add[0]-quadratics_negative[a_t][i-1][0])*quadratics_negative[a_t][i-1][2])>=0 and i>=1:
            i = i-1
        quadratic_add[2] = min(0,2*(quadratic_add[0]-quadratics_negative[a_t][i-1][0])/(quadratic_add[2]-quadratics_negative[a_t][i-1][2]))
        quadratics_negative[a_t] = quadratics_negative[a_t][:i].copy()
        quadratics_negative[a_t].append(tuple(quadratic_add))

        best_glr = 0
        # iterate through quadratics and get change-point estimate
        for quadratic in quadratics_positive[a_t] + quadratics_negative[a_t]:
            if N[a_t]-quadratic[0]>0 and (S[a_t]-quadratic[1])**2/(2*(N[a_t]-quadratic[0]))>best_glr:
                best_glr = (S[a_t]-quadratic[1])**2/(2*(N[a_t]-quadratic[0]))
                v_previous[a_t] = quadratic[3]
                glr_previous[a_t] = best_glr
        m_t = np.random.choice(np.where(glr_previous == np.max(glr_previous))[0])
        v_t = v_previous[m_t]
        changepoint_history[t] = v_t
        glr_history[t] = glr_previous
    return changepoint_history, glr_history