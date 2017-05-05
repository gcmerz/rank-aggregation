import numpy as np
from itertools import combinations, permutations

def spearman_footrule(x, y):
	num_items = len(x)
	delta = np.arange(num_items)
	pi = np.arange(num_items)
	for i in range(len(x)):
		delta[x[i]] = i
		pi[y[i]] = i
	return np.linalg.norm((delta - pi), ord=3)

def kendall_tau(x, y):
    tau = 0
    n_candidates = len(x)
    for i, j in combinations(range(n_candidates), 2):
        tau += (np.sign(x[i] - x[j]) ==
                -np.sign(y[i] - y[j]))
    return float(tau)

def spearman_footrule_matrix(X):
	return np.array([[spearman_footrule(a, b) for a in X] for b in X])

def kendalltau_matrix(X):
	return np.array([[kendall_tau(a, b) for a in X] for b in X])