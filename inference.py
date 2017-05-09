from learncps import sequential_inference, community_sequential_inference, probability_of_ranking
from util import spearman_rank_correlation, spearman_rank_correlation_matrix
from metrics import borda, minimax, instantrunoff
from comparison import convert_votes, convert_to_sushi
import time
import numpy as np

def minimal_average(votes, dist=spearman_rank_correlation_matrix):
	distance_matrix = dist(votes)
	average_distances = np.array([np.average(ds) for ds in distance_matrix])
	return votes[np.argmin(average_distances)]

def check_inference():
	sigmas = []
	thetas = []

	for fn in [('results/3/[0 3 1 6 9 4 8 7 2 5]3.txt', (19,22)), ('results/3/[7 2 0 5 8 1 3 9 6 4]3.txt', (65,76)), ('results/3/[7 4 5 2 1 3 0 8 9 6]3.txt', (69,81))]:
		with open(fn[0]) as f:
			lines = f.readlines()
			votes = []
			for line in lines[1:fn[1][0]]:
				votes.append([int(el) for el in list(line) if not (el == ' ' or el == '[' or el == ']' or el == '\n')])
			sigmas.append(votes)

			theta = []
			for line in lines[fn[1][0]:fn[1][1]]:
				split = [float(el.replace('[', '').replace(']', '')) for el in line.split()]
				for el in split:
					theta.append(el)
			thetas.append(theta)

	inf1 = sequential_inference(thetas[0], sigmas[0], elements=range(10))
	inf2 = sequential_inference(thetas[1], sigmas[1], elements=range(10))
	inf3 = sequential_inference(thetas[2], sigmas[2], elements=range(10))
	
	# our algorithm
	start = time.time()
	comm = community_sequential_inference(thetas, sigmas, elements=range(10))
	print "TIME", time.time() - start

	# other methods
	def aggregate(f, vs, cs):
		return [int(list(el)[0]) for el in f(vs, cs)]
	vs, cs = convert_votes(sigmas[0] + sigmas[1] + sigmas[2])
	start = time.time()
	b = aggregate(borda, vs, cs)
	print "TIME", time.time() - start
	start = time.time()
	simpson = aggregate(minimax, vs, cs)
	print "TIME", time.time() - start
	start = time.time()
	ir = aggregate(instantrunoff, vs, cs)
	ir = ir + list(set(range(10)) - set(ir))
	print "TIME", time.time() - start
	start = time.time()
	ma = minimal_average(sigmas[0] + sigmas[1] + sigmas[2])
	print "TIME", time.time() - start

	print "Community: ", convert_to_sushi(comm)
	print "Borda: ", convert_to_sushi(b)
	print "Simpson: ", convert_to_sushi(simpson)
	print "Instant Runoff: ", convert_to_sushi(ir)
	print "Minimal Average: ", convert_to_sushi(ma)
	print "----"
	print len(sigmas[0]), convert_to_sushi(inf1)
	print len(sigmas[1]), convert_to_sushi(inf2)
	print len(sigmas[2]), convert_to_sushi(inf3)
	print "----"

	def average_dist(votes):
		b_d = []
		s_d = []
		i_d = []
		m_d = []
		c_d = []
		for vote in votes:
			b_d.append(spearman_rank_correlation(vote, b))
			s_d.append(spearman_rank_correlation(vote, simpson))
			i_d.append(spearman_rank_correlation(vote, ir))
			m_d.append(spearman_rank_correlation(vote, ma))
			c_d.append(spearman_rank_correlation(vote, comm))
		print "Borda: ", np.average(b_d)
		print "Simpson: ", np.average(s_d)
		print "Instant Runoff: ", np.average(i_d)
		print "Minimal Average: ", np.average(m_d)
		print "Community: ", np.average(c_d)

	def prob(sigma, theta):
		print probability_of_ranking(b, sigma, theta)
		print probability_of_ranking(simpson, sigma, theta)
		print probability_of_ranking(ir, sigma, theta)
		print probability_of_ranking(ma, sigma, theta)
		print probability_of_ranking(comm, sigma, theta)

	print "Average distance to all voters"
	average_dist(sigmas[0] + sigmas[1] + sigmas[2])
	print "----"

	print "Average distance to voters c1"
	average_dist(sigmas[0])
	prob(sigmas[0], thetas[0])
	print "----"

	print "Average distance to voters c2"
	average_dist(sigmas[1])
	prob(sigmas[1], thetas[1])
	print "----"

	print "Average distance to voters c3"
	average_dist(sigmas[2])
	prob(sigmas[2], thetas[2])
	print "----"

def check_inference2():
	sigmas = []
	thetas = []

	for fn in [('results/[7 5 1 4 8 2 3 6 0 9]2.txt', (39,46)), ('results/[7 0 2 3 8 4 5 1 9 6]2.txt', (63,74))]:
		with open(fn[0]) as f:
			lines = f.readlines()
			votes = []
			for line in lines[1:fn[1][0]]:
				votes.append([int(el) for el in list(line) if not (el == ' ' or el == '[' or el == ']' or el == '\n')])
			sigmas.append(votes)

			theta = []
			for line in lines[fn[1][0]:fn[1][1]]:
				split = [float(el.replace('[', '').replace(']', '')) for el in line.split()]
				for el in split:
					theta.append(el)
			thetas.append(theta)

	inf1 = sequential_inference(thetas[0], sigmas[0], elements=range(10))
	inf2 = sequential_inference(thetas[1], sigmas[1], elements=range(10))
	
	# our algorithm
	start = time.time()
	comm = community_sequential_inference(thetas, sigmas, elements=range(10))
	print "TIME", time.time() - start

	# other methods
	def aggregate(f, vs, cs):
		return [int(list(el)[0]) for el in f(vs, cs)]
	vs, cs = convert_votes(sigmas[0] + sigmas[1])
	start = time.time()
	b = aggregate(borda, vs, cs)
	print "TIME", time.time() - start
	start = time.time()
	simpson = aggregate(minimax, vs, cs)
	print "TIME", time.time() - start
	start = time.time()
	ir = aggregate(instantrunoff, vs, cs)
	ir.append(9)
	print "TIME", time.time() - start
	start = time.time()
	ma = minimal_average(sigmas[0] + sigmas[1])
	print "TIME", time.time() - start

	print "Community: ", convert_to_sushi(comm)
	print "Borda: ", convert_to_sushi(b)
	print "Simpson: ", convert_to_sushi(simpson)
	print "Instant Runoff: ", convert_to_sushi(ir)
	print "Minimal Average: ", convert_to_sushi(ma)
	print "----"
	print len(sigmas[0]), (inf1), "d to c: ", spearman_rank_correlation(inf1, comm), "d to b: ", spearman_rank_correlation(inf1, b)
	print len(sigmas[1]), (inf2), "d to c: ", spearman_rank_correlation(inf2, comm), "d to b: ", spearman_rank_correlation(inf2, b)
	print "----"

	def average_dist(votes):
		b_d = []
		s_d = []
		i_d = []
		m_d = []
		c_d = []
		for vote in votes:
			b_d.append(spearman_rank_correlation(vote, b))
			s_d.append(spearman_rank_correlation(vote, simpson))
			i_d.append(spearman_rank_correlation(vote, ir))
			m_d.append(spearman_rank_correlation(vote, ma))
			c_d.append(spearman_rank_correlation(vote, comm))
		print "Borda: ", np.average(b_d)
		print "Simpson: ", np.average(s_d)
		print "Instant Runoff: ", np.average(i_d)
		print "Minimal Average: ", np.average(m_d)
		print "Community: ", np.average(c_d)

	def prob(sigma, theta):
		print probability_of_ranking(b, sigma, theta)
		print probability_of_ranking(simpson, sigma, theta)
		print probability_of_ranking(ir, sigma, theta)
		print probability_of_ranking(ma, sigma, theta)
		print probability_of_ranking(comm, sigma, theta)

	def diffs(sigmas, thetas):
		print -1 * (probability_of_ranking(b, sigmas[0], thetas[0]) - probability_of_ranking(b, sigmas[1], thetas[1]))
		print -1 * (probability_of_ranking(simpson, sigmas[0], thetas[0]) - probability_of_ranking(simpson, sigmas[1], thetas[1]))
		print -1 * (probability_of_ranking(ir, sigmas[0], thetas[0]) - probability_of_ranking(ir, sigmas[1], thetas[1]))
		print -1 * (probability_of_ranking(ma, sigmas[0], thetas[0]) - probability_of_ranking(ma, sigmas[1], thetas[1]))
		print -1 * (probability_of_ranking(comm, sigmas[0], thetas[0]) - probability_of_ranking(comm, sigmas[1], thetas[1]))

	print "Average distance to all voters"
	average_dist(sigmas[0] + sigmas[1])
	print "----"

	print "Average distance to voters c1"
	average_dist(sigmas[0])
	prob(sigmas[0], thetas[0])
	print "----"

	print "Average distance to voters c2"
	average_dist(sigmas[1])
	prob(sigmas[1], thetas[1])
	print "----"

	diffs(sigmas, thetas)
