# coding: utf-8 # In[87]: 
import numpy as np 
# In[177]: 
def quick_kendalltau(k, l, pi, sigma): 
''' computes quick kendall_tau distances using method described in paper ''' 
	n = len(pi) 
	count = .25 * (n - l) *(n - l - 1) 
	for i in xrange(k - 1): for j in xrange(i+1, n): 
		count += int(sigma.index(pi[i]) > sigma.index(pi[j])) 
		for j in xrange(l + 1, n): 
			count += int(sigma.index(pi[l]) > sigma.index(pi[j])) 
	return count 

# In[199]: 
def find_optimal_theta(pi, sigma_set): 
	n = len(pi) M = len(sigma_set) 
	theta = np.ones(M) 
	def expon(theta_est, k, j): 
		return np.exp(sum([theta_est[m] * quick_kendalltau(k, j, pi, sigma_set[m]) for m in xrange(M)])) 
	def gradient(theta_est, theta_est_idx): 
		total = 0 
		for k in xrange(1, n + 1): 
			val_one = -1 * quick_kendalltau(k, k, pi, sigma_set[theta_est_idx]) 
			numerator = 0 
			denominator = 0 
			for j in xrange(k, n + 1): 
				numerator += quick_kendalltau(k, j, pi, sigma_set[theta_est_idx]) * expon(theta_est, k, j) 
				denominator += expon(theta_est, k, j) 
			total += val_one + (numerator / denominator) 
		return total 
	# let's just do one hundred iterations to see if we can get anything close to convergence 
	# print ([gradient(theta, i) for i in xrange(M)]) 
	theta_one = np.zeros(M) 
	i = 0 
	while not(np.isclose(theta, theta_one, rtol=1e-08)) and (i < 1000): 
		print [gradient(theta, i) for i in xrange(M)] 
		i += 1 
		theta_one = theta 
		theta = theta - (([gradient(theta, i) for i in xrange(M)])) 
	return theta 

# In[200]: 
find_optimal_theta([1,2,3], [[2,1,3]]) 

# In[195]: 
def sequential_inference(theta, sigma): 
	pi = [] 
	M = len(theta) 
	n = len(sigma[0]) 
	items = list(xrange(1,n + 1)) 
	def get_min(idx, items, pi): 
		vals = [(elt, sum([theta[m] * quick_kendalltau(idx, idx, pi + [elt], sigma[m]) for m in xrange(M)])) for elt in items] 
		return min(vals, key = lambda t: t[1])[0] 
	for i in xrange(1, n + 1): 
		val = get_min(i, items, pi) 
		pi.append(val) 
		items.remove(val) 
	return pi 

# In[196]: 
sequential_inference([1,2,1,2], [[1,2,3,4,5],[1,2,4,3,5], [1,5,4,3,2], [1,2,3,4,5]]) # In[ ]: