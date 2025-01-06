import numpy as np
from scipy.stats import norm
import random
import sys
from scipy.stats import bernoulli

def eps_focus(data, epsilon, threshold):
    # Implementation of multi-stream FOCuS for any mu
    # Each stream gets its own set of quadratics
    # Initialization
    M = data.shape[0]
    T = data.shape[1]
    S = np.zeros(M)
    N = np.zeros(M)
    quadratics_positive = [[(0,0,0,S.copy(),N.copy())] for m in range(M)]
    quadratics_negative = [[(0,0,0,S.copy(),N.copy())] for m in range(M)]
    # tau, s, l, vec{S}, vec{N} 
    
    for t in range(T):
        # First perform stream selection with epsilon-greedy
        exploration = bernoulli.rvs(epsilon)
        if exploration:
            a_t = np.random.randint(M)
        else:
            best_glr = 0
            S_nu = np.zeros(M)
            N_nu = np.zeros(M)
            # Find stream and changepoint containing best GLR statistic
            for m in range(M):
                # iterate through stream m's quadratics
                # if no samples then keep going
                if N[m] == 0:
                    continue
                else:
                    for quadratic in quadratics_positive[m] + quadratics_negative[m]:
                        if N[m]-quadratic[0]==0: 
                            continue
                        elif (S[m]-quadratic[1])**2/(2*(N[m]-quadratic[0]))>best_glr:
                            best_glr = (S[m]-quadratic[1])**2/(2*(N[m]-quadratic[0]))
                            S_nu = quadratic[3].copy()
                            N_nu = quadratic[4].copy()
            # Check if there exists any stream which do not have any post-change samples 
            M_0 = np.where((N-N_nu)==0)[0]
            if len(M_0) == 0:
                sample_means = (S-S_nu)/(N-N_nu)
                a_t = np.argmax(np.abs(sample_means))
            else:
                a_t = np.random.choice(M_0)
        # Get observation
        X_t = data[a_t,t]
        N[a_t] = N[a_t] + 1
        S[a_t] = S[a_t] + X_t
        # Positive update
        k = len(quadratics_positive[a_t])
        quadratic_add = [N[a_t],S[a_t],np.inf,N.copy(),S.copy()]
        i = k
        while (2*(quadratic_add[1]-quadratics_positive[a_t][i-1][1])-(quadratic_add[0]-quadratics_positive[a_t][i-1][0])*quadratics_positive[a_t][i-1][2])<=0 and i>=1:
            i = i-1
        quadratic_add[2] = max(0,2*(quadratic_add[0]-quadratics_positive[a_t][i-1][0])/(quadratic_add[2]-quadratics_positive[a_t][i-1][2]))
        quadratics_positive[a_t] = quadratics_positive[a_t][:i].copy()
        quadratics_positive[a_t].append(tuple(quadratic_add))
    
        # Now negative update
        k = len(quadratics_negative[a_t])
        quadratic_add = [N[a_t],S[a_t],-np.inf,N.copy(),S.copy()]
        i = k
        while (2*(quadratic_add[1]-quadratics_negative[a_t][i-1][1])-(quadratic_add[0]-quadratics_negative[a_t][i-1][0])*quadratics_negative[a_t][i-1][2])>=0 and i>=1:
            i = i-1
        quadratic_add[2] = min(0,2*(quadratic_add[0]-quadratics_negative[a_t][i-1][0])/(quadratic_add[2]-quadratics_negative[a_t][i-1][2]))
        quadratics_negative[a_t] = quadratics_negative[a_t][:i].copy()
        quadratics_negative[a_t].append(tuple(quadratic_add))

        best_glr = 0
        # iterate through stream a_t's quadratics
        # if no samples then keep going
        if N[a_t] > 0:
            for quadratic in quadratics_positive[a_t] + quadratics_negative[a_t]:
                if N[a_t]-quadratic[0]>0 and (S[a_t]-quadratic[1])**2/(2*(N[a_t]-quadratic[0]))>best_glr:
                    best_glr = (S[a_t]-quadratic[1])**2/(2*(N[a_t]-quadratic[0]))
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

def changepoint_estimation(data, epsilon):
    M = data.shape[0]
    T = data.shape[1]
    S = np.zeros(M)
    N = np.zeros(M)
    # Add time step location tracker
    quadratics_positive = [[(0,0,0,S.copy(),N.copy(),0)] for m in range(M)]
    quadratics_negative = [[(0,0,0,S.copy(),N.copy(),0)] for m in range(M)]
    changepoint_location = np.zeros(T)
    stream_mle = -np.ones(T)
    
    for t in range(T):
        # Estimate change-point
        changepoint_estimate = 0
        best_glr = 0
        # If no stream is best might as well be random
        stream_best = np.random.randint(M)
        for m in range(M):
            # iterate through stream m's quadratics
            # if no samples then keep going
            if N[m] == 0:
                continue
            else:
                for quadratic in quadratics_positive[m] + quadratics_negative[m]:
                    if N[m]-quadratic[0]==0: 
                        continue
                    elif (S[m]-quadratic[1])**2/(2*(N[m]-quadratic[0]))>best_glr:
                        best_glr = (S[m]-quadratic[1])**2/(2*(N[m]-quadratic[0]))
                        stream_best = m
                        changepoint_estimate = quadratic[5]
        stream_mle[t] = stream_best
        changepoint_location[t] = changepoint_estimate
        # First perform stream selection with epsilon-greedy
        exploration = bernoulli.rvs(epsilon)
        if exploration:
            a_t = np.random.randint(M)
        else:
            best_glr = 0
            S_nu = np.zeros(M)
            N_nu = np.zeros(M)
            # Find stream and changepoint containing best GLR statistic
            for m in range(M):
                # iterate through stream m's quadratics
                # if no samples then keep going
                if N[m] == 0:
                    continue
                else:
                    for quadratic in quadratics_positive[m] + quadratics_negative[m]:
                        if N[m]-quadratic[0]==0: 
                            continue
                        elif (S[m]-quadratic[1])**2/(2*(N[m]-quadratic[0]))>best_glr:
                            best_glr = (S[m]-quadratic[1])**2/(2*(N[m]-quadratic[0]))
                            S_nu = quadratic[3].copy()
                            N_nu = quadratic[4].copy()
            # Check if there exists any stream which do not have any post-change samples 
            M_0 = np.where((N-N_nu)==0)[0]
            if len(M_0) == 0:
                sample_means = (S-S_nu)/(N-N_nu)
                a_t = np.argmax(np.abs(sample_means))
            else:
                a_t = np.random.choice(M_0)
        # Get observation
        X_t = data[a_t,t]
        N[a_t] = N[a_t] + 1
        S[a_t] = S[a_t] + X_t
        # Positive update
        k = len(quadratics_positive[a_t])
        quadratic_add = [N[a_t],S[a_t],np.inf,N.copy(),S.copy(),t+1]
        i = k
        while (2*(quadratic_add[1]-quadratics_positive[a_t][i-1][1])-(quadratic_add[0]-quadratics_positive[a_t][i-1][0])*quadratics_positive[a_t][i-1][2])<=0 and i>=1:
            i = i-1
        quadratic_add[2] = max(0,2*(quadratic_add[0]-quadratics_positive[a_t][i-1][0])/(quadratic_add[2]-quadratics_positive[a_t][i-1][2]))
        quadratics_positive[a_t] = quadratics_positive[a_t][:i].copy()
        quadratics_positive[a_t].append(tuple(quadratic_add))
    
        # Now negative update
        k = len(quadratics_negative[a_t])
        quadratic_add = [N[a_t],S[a_t],-np.inf,N.copy(),S.copy(),t+1]
        i = k
        while (2*(quadratic_add[1]-quadratics_negative[a_t][i-1][1])-(quadratic_add[0]-quadratics_negative[a_t][i-1][0])*quadratics_negative[a_t][i-1][2])>=0 and i>=1:
            i = i-1
        quadratic_add[2] = min(0,2*(quadratic_add[0]-quadratics_negative[a_t][i-1][0])/(quadratic_add[2]-quadratics_negative[a_t][i-1][2]))
        quadratics_negative[a_t] = quadratics_negative[a_t][:i].copy()
        quadratics_negative[a_t].append(tuple(quadratic_add))
    return changepoint_location, stream_mle