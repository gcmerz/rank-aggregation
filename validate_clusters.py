import numpy as np
from election import Election
from util import read_sushi_votes

votes, r_ids = read_sushi_votes(same=True)

n = 500
nc = 4
props = []
num = [[]] * 10
num_each = np.array([float((r_ids[:n] == i).sum()) for i in range(11)])

E = Election(num_clusters=nc, votes=votes[:n], region_ids=r_ids[:n])
for cluster in E.vote_clusters:
	props.append(np.array([(float((cluster[1] == i).sum())/float(len(cluster[1]))) * n for i in range(11)]))
	print len(cluster[0]), props[-1]
	print "----"

	counts = np.zeros((10, 11))

	for ranking in cluster[0]:
		for i in range(10):
			counts[i][ranking[i]] += 1

	for i in range(10):
		num[i].append(counts[i]/float(len(cluster[1])))

# this metric only makes sense if there are two clusters
if nc == 2:
	print np.linalg.norm(props[0] - props[1], ord=1) / float(n) # score of 1 is perfect separation, 0 means clusters identical
	diff = [np.abs(props[0][i] - props[1][i]) for i in range(len(props[0])) if not i == 3]
	print np.linalg.norm(diff, ord=1) / (float(n) - num_each[3]) # does not consider most populous region
	print "----"

num_east = [np.sum([props[i][j] for j in range(5)]) for i in range(nc)]
num_west = [np.sum([props[i][j] for j in range(5, 11)]) for i in range(nc)]
print num_east, num_west
print "----"
for i in range(11):
	print [num[0][j][i] for j in range(nc)]
print "----"
for i in range(11):
	print [num[-1][j][i] for j in range(nc)]

