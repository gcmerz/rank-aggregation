from learncps import sequential_inference, community_sequential_inference, probability_of_ranking
from util import spearman_rank_correlation
from metrics import borda, minimax, instantrunoff
from comparison import convert_votes, convert_to_sushi
import numpy as np

def check_inference():
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
	comm = community_sequential_inference(thetas, sigmas, elements=range(10))

	# other methods
	def aggregate(f, vs, cs):
		return [int(list(el)[0]) for el in f(vs, cs)]
	vs, cs = convert_votes(sigmas[0] + sigmas[1])
	b = aggregate(borda, vs, cs)
	simpson = aggregate(minimax, vs, cs)
	ir = aggregate(instantrunoff, vs, cs)
	ir.append(9)

	print "Community: ", convert_to_sushi(comm)
	print "Borda: ", convert_to_sushi(b)
	print "Simpson: ", convert_to_sushi(simpson)
	print "Instant Runoff: ", convert_to_sushi(ir)
	print "----"
	print len(sigmas[0]), (inf1), "d to c: ", spearman_rank_correlation(inf1, comm), "d to b: ", spearman_rank_correlation(inf1, b)
	print len(sigmas[1]), (inf2), "d to c: ", spearman_rank_correlation(inf2, comm), "d to b: ", spearman_rank_correlation(inf2, b)
	print "----"

	def average_dist(votes):
		b_d = []
		s_d = []
		i_d = []
		c_d = []
		for vote in votes:
			b_d.append(spearman_rank_correlation(vote, b))
			s_d.append(spearman_rank_correlation(vote, simpson))
			i_d.append(spearman_rank_correlation(vote, ir))
			c_d.append(spearman_rank_correlation(vote, comm))
		print "Borda: ", np.average(b_d)
		print "Simpson: ", np.average(s_d)
		print "Instant Runoff: ", np.average(i_d)
		print "Community: ", np.average(c_d)

	def prob(sigma, theta):
		print probability_of_ranking(b, sigma, theta)
		print probability_of_ranking(simpson, sigma, theta)
		print probability_of_ranking(ir, sigma, theta)
		print probability_of_ranking(comm, sigma, theta)

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

