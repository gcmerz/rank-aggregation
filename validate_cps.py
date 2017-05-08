from learncps import * 
from election import Election 
import numpy as np
from numpy.random import choice
from random import sample
from util import spearman_rank_correlation, read_sushi_votes


def random_sample(n, n_samples): 
	'''
		function: random_sample
		params: n (int, number of objects in permutation)
				n_samples (int, number of times to sample)
		returns: list of lists, inner list random permutations of n
	'''
	lst = list(xrange(n))
	l = len(lst)
	samples = []
	for _ in xrange(n_samples):
		samples.append(sample(lst, l))
	return samples

def cps_approx_sample(pi, theta, sigma, n_samples, dist = quick_src): 
	''' 
		function: cps_sample
		params: theta, sigma of cps model and n_samples
		returns: number of randomly sampled
	'''
	n = len(sigma[0])
	def expon(i, j, perm): 
		return np.exp(-sum([theta[m] * dist(i, j, perm, sigma[m]) for m in xrange(len(sigma))]))
	def probability(perm): 
		i = len(perm) - 1	
		num = expon(i, i, perm)
		return num 
	def p_of_next_elts(perm, elts):
		ps = []
		for elt in elts: 
			ps.append(probability(perm + [elt]))
		return elts, ps
	def build_sample():
		perm = []
		elts = list(xrange(n))
		for _ in xrange(n): 
			next_elt, ps = p_of_next_elts(perm, elts)
			new_elt = choice(next_elt, 1, ps)[0]
			perm.append(new_elt)	
			elts.remove(new_elt)
		return perm 
	samples = []
	for _ in xrange(n_samples):
		samples.append(build_sample())
	print samples
	return samples


def avg_distance(v1s, v2s, dist_metric = spearman_rank_correlation):
	'''
		function: total_distance
		params: v1s, v2s (two lists of votes)
		returns: total distance between every vote in v1s and every vote in v2s
	'''
	dist = 0 
	ct = 0. 
	for v1 in v1s: 
		for v2 in v2s: 
			dist += dist_metric(v1, v2)
			ct += 1. 
	return dist / ct


if __name__ == '__main__': 
	# generate 100 random samples, twice 
	v1 = random_sample(10, 100)
	v2 = random_sample(10, 100)
	# average distance between 100 randomly generated samples
	print avg_distance(v1, v2)
	# average distance between randomly generated samples and SUSHI
	votes, _ = read_sushi_votes()
	print avg_distance(v1, votes)