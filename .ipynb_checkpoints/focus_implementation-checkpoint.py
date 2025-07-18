import numpy as np
from scipy.stats import norm
import random
import sys
from scipy.stats import bernoulli

def generate_data(M, T, nu, mu1):
    mu0 = 0
    # Assume changepoint is in stream 0
    data = np.zeros((M,T))
    data[0][:nu] = norm.rvs(size=nu)
    data[0][nu:] = norm.rvs(loc=mu1,size=T-nu)
    for i in range(1,M):
        data[i] = norm.rvs(loc=mu0,size=T)
    return data

def focus_decay(data, threshold):
    # Implementation of multi-stream FOCuS for any mu
    # Each stream gets its own set of quadratics
    # Initialization
    M = data.shape[0]
    T = data.shape[1]
    S = np.zeros(M)
    N = np.zeros(M)
    quadratics_positive = [[(0,0,0,0)] for m in range(M)]
    quadratics_negative = [[(0,0,0,0)] for m in range(M)]
    #M_previous = np.random.randint(M)
    glr_previous = np.zeros(M)
    v_previous = np.zeros(M)
    # tau, s, l, vec{S}, vec{N} 
    
    for t in range(T):
        # First perform stream selection with epsilon-greedy
        m_t = np.random.choice(np.where(glr_previous == np.max(glr_previous))[0])
        v_t = v_previous[m_t]
        epsilon = min(1,(t+1-v_t)**(-1/3))
        exploration = bernoulli.rvs(epsilon)
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
        if best_glr>=threshold:
            return t+1, best_glr

def focus_nonuhat(data, threshold):
    # Implementation of multi-stream FOCuS for any mu
    # Each stream gets its own set of quadratics
    # Initialization
    M = data.shape[0]
    T = data.shape[1]
    S = np.zeros(M)
    N = np.zeros(M)
    quadratics_positive = [[(0,0,0)] for m in range(M)]
    quadratics_negative = [[(0,0,0)] for m in range(M)]
    #M_previous = np.random.randint(M)
    glr_previous = np.zeros(M)
    # tau, s, l, vec{S}, vec{N} 
    
    for t in range(T):
        # First perform stream selection with epsilon-greedy
        m_t = np.random.choice(np.where(glr_previous == np.max(glr_previous))[0])
        epsilon = min(1,(t+1)**(-1/3))
        exploration = bernoulli.rvs(epsilon)
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
        quadratic_add = [N[a_t],S[a_t],np.inf]
        i = k
        while (2*(quadratic_add[1]-quadratics_positive[a_t][i-1][1])-(quadratic_add[0]-quadratics_positive[a_t][i-1][0])*quadratics_positive[a_t][i-1][2])<=0 and i>=1:
            i = i-1
        quadratic_add[2] = max(0,2*(quadratic_add[0]-quadratics_positive[a_t][i-1][0])/(quadratic_add[2]-quadratics_positive[a_t][i-1][2]))
        quadratics_positive[a_t] = quadratics_positive[a_t][:i].copy()
        quadratics_positive[a_t].append(tuple(quadratic_add))
    
        # Now negative update
        k = len(quadratics_negative[a_t])
        quadratic_add = [N[a_t],S[a_t],-np.inf]
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
                glr_previous[a_t] = best_glr
        if best_glr>=threshold:
            return t+1, best_glr

def focus_oracle(data, threshold, nu):
    # Implementation of multi-stream FOCuS for any mu
    # Each stream gets its own set of quadratics
    # Initialization
    # Algorithm knows change-point location ahead of time to test estimation procedure
    M = data.shape[0]
    T = data.shape[1]
    S = np.zeros(M)
    N = np.zeros(M)
    quadratics_positive = [[(0,0,0)] for m in range(M)]
    quadratics_negative = [[(0,0,0)] for m in range(M)]
    #M_previous = np.random.randint(M)
    glr_previous = np.zeros(M)
    # tau, s, l, vec{S}, vec{N} 
    
    for t in range(T):
        # First perform stream selection with epsilon-greedy
        m_t = np.random.choice(np.where(glr_previous == np.max(glr_previous))[0])
        if t+1>nu:
            epsilon = min(1,(t+1-nu)**(-1/3))
        else:
            epsilon = 1
        exploration = bernoulli.rvs(epsilon)
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
        quadratic_add = [N[a_t],S[a_t],np.inf]
        i = k
        while (2*(quadratic_add[1]-quadratics_positive[a_t][i-1][1])-(quadratic_add[0]-quadratics_positive[a_t][i-1][0])*quadratics_positive[a_t][i-1][2])<=0 and i>=1:
            i = i-1
        quadratic_add[2] = max(0,2*(quadratic_add[0]-quadratics_positive[a_t][i-1][0])/(quadratic_add[2]-quadratics_positive[a_t][i-1][2]))
        quadratics_positive[a_t] = quadratics_positive[a_t][:i].copy()
        quadratics_positive[a_t].append(tuple(quadratic_add))
    
        # Now negative update
        k = len(quadratics_negative[a_t])
        quadratic_add = [N[a_t],S[a_t],-np.inf]
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
                glr_previous[a_t] = best_glr
        if best_glr>=threshold:
            return t+1, best_glr

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
        exploration = bernoulli.rvs(epsilon)
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