# coding: utf-8 # In[87]: 
import numpy as np 

def quick_kendalltau(k, l, pi, sigma): 
    ''' computes quick kendall_tau distances using method described in paper '''
    n = len(pi)
    count = .25 * (n - l) *(n - l - 1)
    for i in xrange(k): 
        for j in xrange(i+1, n): 
            count += int(sigma.index(pi[i]) > sigma.index(pi[j]))
    for j in xrange(l + 1, n): 
        count += int(sigma.index(pi[l]) > sigma.index(pi[j]))
    return count

def quick_src(k, l, pi, sigma):
	n = len(pi)
	count = 0.
	for i in xrange(k-1):
		count += (sigma.index(pi[i]) - i) ** 2
	count += (sigma.index(pi[l]) - (k - 1)) ** 2
	temporary_count = 0.
	for i in range(k, n):
		for j in range(k, n):
			temporary_count += (sigma.index(pi[i]) - j) ** 2
	return count + (1. / float(n - k)) * temporary_count

def find_optimal_theta(pi, sigma_set, dist=quick_src): 
    n = len(pi)
    M = len(sigma_set)
    theta = np.ones(M)
    
    def expon(theta_est, k, j): 
        return np.exp(sum([theta_est[m] * dist(k, j, pi, sigma_set[m]) for m in xrange(M)]))
    
    def gradient(theta_est, theta_est_idx):
        total = 0 
        for k in xrange(n): 
            val_one = -1 * dist(k, k, pi, sigma_set[theta_est_idx])
            numerator = 0
            denominator = 0
            for j in xrange(k, n): 
                numerator += dist(k, j, pi, sigma_set[theta_est_idx]) * expon(theta_est, k, j)
                denominator += expon(theta_est, k, j)
            total += val_one + (numerator / denominator)
        return 0.75*total 
                        
    # let's just do one hundred iteratons to see if we can get anything close to convergence
    # print ([gradient(theta, i) for i in xrange(M)])
    theta_one = np.zeros(M)
    i = 0
    while not(np.isclose(theta, theta_one, rtol=1e-08).all()) and (i < 1000):
        # print [gradient(theta, i) for i in xrange(M)]
        i += 1
        theta_one = theta 
        theta = theta - (([gradient(theta, i) for i in xrange(M)]))
        print theta
    return theta 

find_optimal_theta([1,2,3], [[2,1,3], [2,3,1]], quick_src) 

def sequential_inference(theta, sigma, dist=quick_src): 
    pi = []
    M = len(theta)
    n = len(sigma[0])
    items = list(xrange(1,n + 1))
    def get_min(idx, items, pi): 
        vals = [(elt, sum([theta[m] * dist(idx, idx, pi + [elt], sigma[m]) for m in xrange(M)])) for elt in items]
        return min(vals, key = lambda t: t[1])[0]
    for i in xrange(1, n + 1): 
        val = get_min(i, items, pi)
        pi.append(val)
        items.remove(val)
    return pi

sequential_inference([1,2,1,2], [[1,2,3,4,5],[1,2,4,3,5], [1,5,4,3,2], [1,2,3,4,5]])