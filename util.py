import numpy as np

def spearman_footrule(x, y):
	num_items = len(x)
	delta = np.arange(num_items)
	pi = np.arange(num_items)
	for i in range(num_items):
		delta[x[i]] = i
		pi[y[i]] = i
	return np.linalg.norm(delta - pi, ord=1)

def spearman_footrule_matrix(X):
	return np.array([[spearman_footrule(a, b) for a in X] for b in X])
