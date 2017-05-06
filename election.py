import numpy as np
from sklearn.cluster import AgglomerativeClustering
from util import spearman_rank_correlation_matrix, spearman_footrule_matrix, kendalltau_matrix

class Election(object):
	def __init__(self, votes=None, num_clusters=2, region_ids=None, affinity=spearman_rank_correlation_matrix):
		if votes is None:
			self.__generate_votes()
		else:
			self.votes = votes

		self.region_ids = region_ids

		self.affinity = affinity  # see util for possible affinity functions

		self.__cluster_votes(num_clusters)

		self.__find_cluster_centers()

	def __generate_votes(self):
		'''for simulations'''
		self.votes = np.array([np.random.permutation(4) for _ in range(100)])

	def __cluster_votes(self, num_clusters):
		'''uses HAC to cluster'''
		C = AgglomerativeClustering(n_clusters=num_clusters, affinity=self.affinity, linkage="complete")
		cluster_assignments = C.fit_predict(self.votes)
		self.vote_clusters = []
		for i in range(num_clusters):
			idx = np.array([a for a, b in enumerate(cluster_assignments) if b==i])
			if self.region_ids is None:
				self.vote_clusters.append(self.votes[idx])
			else:
				self.vote_clusters.append((self.votes[idx], np.array(self.region_ids)[idx]))

	def __find_cluster_centers(self):
		'''finds vote in each cluster closest to all the other votes on average'''
		self.cluster_centers = []
		for cluster in self.vote_clusters:
			distance_matrix = self.affinity(cluster)
			average_distances = np.array([np.average(ds) for ds in distance_matrix])
			self.cluster_centers.append(cluster[np.argmin(average_distances)])
		return self.cluster_centers
