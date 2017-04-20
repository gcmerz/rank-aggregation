import numpy as np
from sklearn.cluster import AgglomerativeClustering
from util import spearman_footrule_matrix

class Election(object):
	def __init__(self, num_clusters=2):
		self.__generate_votes()
		self.__cluster_votes(num_clusters)

	def __generate_votes(self):
		self.votes = np.array([np.random.permutation(4) for _ in range(100)])

	def __cluster_votes(self, num_clusters):
		AC = AgglomerativeClustering(n_clusters=num_clusters, affinity=spearman_footrule_matrix, linkage="complete")
		cluster_assignments = AC.fit_predict(self.votes)
		self.vote_clusters = []
		for i in range(num_clusters):
			idx = np.array([a for a, b in enumerate(cluster_assignments) if b==i])
			self.vote_clusters.append(self.votes[idx])

E = Election()
for cluster in E.vote_clusters:
	print cluster
