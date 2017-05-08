from learncps import sequential_inference, community_sequential_inference
from util import spearman_rank_correlation

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
comm = community_sequential_inference(thetas, sigmas, elements=range(10))
print len(sigmas[0]), inf1, spearman_rank_correlation(inf1, comm)
print len(sigmas[1]), inf2, spearman_rank_correlation(inf2, comm)
print comm
