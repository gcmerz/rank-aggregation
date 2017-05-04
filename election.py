import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from util import spearman_footrule, spearman_footrule_matrix, kendalltau_matrix
from nltk.cluster import KMeansClusterer

class Election(object):
	def __init__(self, votes=None, num_clusters=2, region_ids=None, kmeans=False):
		if votes is None:
			self.__generate_votes()
		else:
			self.votes = votes

		if region_ids is not None:
			self.region_ids = region_ids

		self.__cluster_votes(num_clusters, kmeans)

	def __generate_votes(self):
		self.votes = np.array([np.random.permutation(4) for _ in range(100)])

	def __cluster_votes(self, num_clusters, kmeans=False):
		if not kmeans:
			C = AgglomerativeClustering(n_clusters=num_clusters, affinity=spearman_footrule_matrix, linkage="complete")
			cluster_assignments = C.fit_predict(self.votes)
			self.vote_clusters = []
			for i in range(num_clusters):
				idx = np.array([a for a, b in enumerate(cluster_assignments) if b==i])
				if self.region_ids is None:
					self.vote_clusters.append(self.votes[idx])
				else:
					self.vote_clusters.append((self.votes[idx], np.array(self.region_ids)[idx]))
		else:
			C = KMeansClusterer(num_means=2, distance=spearman_footrule)
			C.cluster_vectorspace(self.votes)
			cluster_assignments = [C.classify_vectorspace(v) for v in self.votes]
			self.vote_clusters = []
			for i in range(num_clusters):
				idx = np.array([a for a, b in enumerate(cluster_assignments) if b==i])
				if self.region_ids is None:
					self.vote_clusters.append(self.votes[idx])
				else:
					self.vote_clusters.append((self.votes[idx], np.array(self.region_ids)[idx]))

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

n = 1792
props = []
num_each = np.array([float((r_ids[:n] == i).sum()) for i in range(11)])
E = Election(num_clusters=2, votes=votes[:n], region_ids=r_ids[:n], kmeans=False)

for cluster in E.vote_clusters:
	props.append(np.array([(float((cluster[1] == i).sum())/float(len(cluster[1]))) * n for i in range(11)]))
	print len(cluster[0]), props[-1]
	for i in cluster[0][:10]:
		print i
	print "----"

print np.linalg.norm(props[0] - props[1], ord=1) / float(n) # score of 1 is perfect separation, 0 means clusters identical
diff = [np.abs(props[0][i] - props[1][i]) for i in range(len(props[0])) if not i == 3]
print np.linalg.norm(diff, ord=1) / (float(n) - num_each[3]) # does not consider most populous region
