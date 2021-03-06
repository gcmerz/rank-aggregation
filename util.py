import numpy as np
from itertools import combinations, permutations

def spearman_footrule(x, y):
	num_items = len(x)
	delta = np.arange(num_items)
	pi = np.arange(num_items)
	for i in range(len(x)):
		delta[x[i]] = i
		pi[y[i]] = i
	return np.linalg.norm((delta - pi), ord=1)

def spearman_rank_correlation(x, y):
	num_items = len(x)
	delta = np.arange(num_items)
	pi = np.arange(num_items)
	for i in range(len(x)):
		delta[x[i]] = i
		pi[y[i]] = i
	return np.linalg.norm((delta - pi) ** 2, ord=1)

def kendall_tau(x, y):
    tau = 0
    n_candidates = len(x)
    for i, j in combinations(range(n_candidates), 2):
        tau += (np.sign(x[i] - x[j]) ==
                -np.sign(y[i] - y[j]))
    return float(tau)

def spearman_footrule_matrix(X):
	return np.array([[spearman_footrule(a, b) for a in X] for b in X])

def spearman_rank_correlation_matrix(X):
	return np.array([[spearman_rank_correlation(a, b) for a in X] for b in X])

def kendalltau_matrix(X):
	return np.array([[kendall_tau(a, b) for a in X] for b in X])

def read_sushi_votes(same=False, fn='sushi3-2016/sushi3a.5000.10.order'):
	votes = []
	with open(fn) as f:
		lines = f.readlines()
		for line in lines[1:]:
			votes.append(np.array(map(int, line.rstrip('\n').split(' ')[2:])))

	r_ids = []
	same = []
	with open('sushi3-2016/sushi3.udata') as f:
		lines = f.readlines()
		for line in lines:
			r_ids.append(map(int, line.split())[8])
			same.append(map(int, line.split())[10])

	if same:
		votes = [votes[i] for i in range(len(votes)) if same[i]]
		r_ids = [r_ids[i] for i in range(len(r_ids)) if same[i]]

	return np.array(votes), np.array(r_ids)
	