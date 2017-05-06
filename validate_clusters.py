import numpy as np
from election import Election

votes = []
with open('sushi3-2016/sushi3a.5000.10.order') as f:
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

votes = np.array(votes)
r_ids = np.array(r_ids)

votes_same = np.array([votes[i] for i in range(len(votes)) if same[i]])
r_ids_same = np.array([r_ids[i] for i in range(len(votes)) if same[i]])

n = 500
props = []
num_first = []
num_last = []
num_each = np.array([float((r_ids_same[:n] == i).sum()) for i in range(11)])
E = Election(num_clusters=2, votes=votes_same[:n], region_ids=r_ids_same[:n])

for cluster in E.vote_clusters:
	props.append(np.array([(float((cluster[1] == i).sum())/float(len(cluster[1]))) * n for i in range(11)]))
	counts_first = np.zeros(11)
	counts_last = np.zeros(11)
	print len(cluster[0]), props[-1]
	for ranking in cluster[0]:
		counts_first[ranking[0]] += 1
		counts_last[ranking[-1]] += 1
	num_first.append(counts_first/float(len(cluster[1])))
	num_last.append(counts_last/float(len(cluster[1])))
	print "----"

print np.linalg.norm(props[0] - props[1], ord=1) / float(n) # score of 1 is perfect separation, 0 means clusters identical
diff = [np.abs(props[0][i] - props[1][i]) for i in range(len(props[0])) if not i == 3]
print np.linalg.norm(diff, ord=1) / (float(n) - num_each[3]) # does not consider most populous region
num_east = [0,0]
num_west = [0,0]
for i in range(2):
	for j in range(11):
		if j < 5:
			num_east[i] += props[i][j]
		else:
			num_west[i] += props[i][j]
print "----"
print num_east, num_west
print np.sum(np.abs(np.diff([num_east, num_west])))/float(n)
print "----"
for i in range(11):
	print num_first[0][i], num_first[1][i]
print "----"
for i in range(11):
	print num_last[0][i], num_last[1][i]
