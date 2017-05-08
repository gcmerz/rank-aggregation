from learncps import * 
from election import Election 
import numpy as np
from numpy.random import choice
from random import sample
from util import spearman_rank_correlation, read_sushi_votes
import os


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
	return samples

def read_cps_file(fname):
	# time was 1441.78837204
	with open(fname, 'r') as f: 
		lines = f.readlines()
		pi = np.fromstring(lines[0][1:-2], dtype = int, sep = ' ')
		sigma = []
		for i, line in enumerate(lines[1:]):
			i += 1 
			if '[[' in line: 
				sigma.append(np.fromstring(line[3:-2], dtype = int, sep = ' '))
			elif ']]' in line: 
				sigma.append(np.fromstring(line[2:-3], dtype = int, sep = ' '))
				break
			else: 
				sigma.append(np.fromstring(line[2:-2], dtype = int, sep = ' '))
		array_strs = [lines[i + 1][2:-1]] + [lines[i + j][2:-1] for j in xrange(1, len(lines[i + 2: -1]))] + [lines[-2][:-3]]
		theta = np.fromstring((' ').join(array_strs), dtype = float, sep = ' ')
		loss = np.fromstring(lines[-1][2:-2], dtype = float, sep = ',')
	return pi, theta, sigma, loss


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

def run_election(votes): 
	center = Election(votes=votes, num_clusters=1).cluster_centers
	return center[0]


if __name__ == '__main__': 
	# generate 100 random samples, twice 

	# declare number of samples, distance metric, 
	# and number of clusters we're validating
	ns = 1000
	dist = spearman_rank_correlation
	num_clusters = 2 
	num_clusters = str(num_clusters)

	# read in the parameters for the CPS models
	# we learned
	params = []
	total_votes = 0 
	for filename in os.listdir('results'): 
		if filename.endswith(num_clusters + '.txt'): 
			p = read_cps_file('results/' + filename)
			params.append(p)
			total_votes += len(p[2])

	# proprtion of votes to sample from each model 
	proportion = lambda s, n: int((float(len(s)) / total_votes) * n)
	cps_samples = [] 
	random_samples = []
	for param in params: 
			cps_samples.append(cps_approx_sample(param[0], param[1], param[2], proportion(param[2], ns)))
			random_samples.append(random_sample(10, proportion(param[2], ns)))

	# print results
	for i, (r1, c1, p1) in enumerate(zip(random_samples, cps_samples, params)): 
		r1 = run_election(np.array(r1))
		c1 = run_election(np.array(c1))
		print "DISTANCE BETWEEN RANDOM SAMPLE AND CLUSTER %s: %s" % (i, dist(r1, p1[0]))
		print "DISTANCE BETWEEN CPS SAMPLE AND CLUSTER %s: %s" % (i, dist(c1, p1[0]))

	# cps_votes = s1 + s2 + s3
	# print "RANDOM AND SUSHI: %s" % avg_distance(v1, votes)
	# print "RANDOM AND RANDOM: %s" % avg_distance(v1,v2)
	# print "CPS AND RANDOM: %s" % avg_distance(cps_votes, v1)
	# print "CPS AND SUSHI: %s" % avg_distance(cps_votes, votes)
	# print avg_distance(v1, votes)