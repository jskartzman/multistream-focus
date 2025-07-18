import numpy as np
from scipy.stats import norm
import random
import sys
from scipy.stats import bernoulli
from numba import njit

def generate_data(M, T, nu, mu1):
    mu0 = 0
    data = np.zeros((M, T))
    data[0, :nu] = np.random.normal(loc=0, scale=1, size=nu)
    data[0, nu:] = np.random.normal(loc=mu1, scale=1, size=T-nu)
    for i in range(1, M):
        data[i] = np.random.normal(loc=mu0, scale=1, size=T)
    return data

@njit
def find_best_glr(Na, Sa, quads):
    best_glr = 0.0
    v_prev = 0.0
    for i in range(quads.shape[0]):
        delta_N = Na - quads[i, 0]
        if delta_N > 0:
            glr_val = (Sa - quads[i, 1]) ** 2 / (2 * delta_N)
            if glr_val > best_glr:
                best_glr = glr_val
                v_prev = quads[i, 3]
    return best_glr, v_prev

@njit
def focus_decay(data, threshold):
    # Implementation of multi-stream FOCuS for any mu
    # Each stream gets its own set of quadratics
    # Initialization
    M = data.shape[0]
    T = data.shape[1]
    S = np.zeros(M)
    N = np.zeros(M)
    
    quadratics_positive = np.zeros((M,T,4))
    quadratics_negative = np.zeros((M,T,4))
    quad_len_pos = np.ones(M,dtype=np.int16)
    quad_len_neg = np.ones(M,dtype=np.int16)
    glr_previous = np.zeros(M)
    v_previous = 0
    quadratic_add = np.empty(4, dtype=np.float64)
    # tau, s, l, nu
    
    for t in range(1,T+1):
        # First perform stream selection with epsilon-greedy
        epsilon = min(1,M*((t-v_previous)**(-1/3)))
        if np.random.rand() < epsilon:
            a_t = np.random.randint(M)
        else:
            #a_t = np.random.choice(np.where(glr_previous == np.max(glr_previous))[0])
            a_t = np.argmax(glr_previous)
        # Get observation
        X_t = data[a_t,t-1]
        N[a_t] += 1
        S[a_t] += X_t

        # Positive update
        k = quad_len_pos[a_t]
        quadratic_add[0] = N[a_t]
        quadratic_add[1] = S[a_t]
        quadratic_add[2] = np.inf
        quadratic_add[3] = t
        i = k
        while (2*(quadratic_add[1]-quadratics_positive[a_t,i-1,1])-(quadratic_add[0]-quadratics_positive[a_t,i-1,0])*quadratics_positive[a_t,i-1,2])<=0 and i>=1:
            i = i-1
        quadratic_add[2] = max(0,2*(quadratic_add[0]-quadratics_positive[a_t,i-1,0])/(quadratic_add[2]-quadratics_positive[a_t,i-1,2]))
        quadratics_positive[a_t,i+1] = quadratic_add
        quad_len_pos[a_t] = i+1
    
        # Now negative update
        k = quad_len_neg[a_t]
        quadratic_add[0] = N[a_t]
        quadratic_add[1] = S[a_t]
        quadratic_add[2] = -np.inf
        quadratic_add[3] = t
        i = k
        while (2*(quadratic_add[1]-quadratics_negative[a_t,i-1,1])-(quadratic_add[0]-quadratics_negative[a_t,i-1,0])*quadratics_negative[a_t,i-1,2])>=0 and i>=1:
            i = i-1
        quadratic_add[2] = min(0,2*(quadratic_add[0]-quadratics_negative[a_t,i-1,0])/(quadratic_add[2]-quadratics_negative[a_t,i-1,2]))
        quadratics_negative[a_t,i+1] = quadratic_add
        quad_len_neg[a_t] = i+1
        best_glr_pos, v_pos = find_best_glr(N[a_t], S[a_t], quadratics_positive[a_t,:quad_len_pos[a_t]])
        best_glr_neg, v_neg = find_best_glr(N[a_t], S[a_t], quadratics_negative[a_t,:quad_len_neg[a_t]])

        if best_glr_pos > best_glr_neg:
            best_glr = best_glr_pos
            v_previous = v_pos
        else:
            best_glr = best_glr_neg
            v_previous = v_neg

        if best_glr>=threshold:
            return t, best_glr