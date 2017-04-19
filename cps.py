class CPS(object):    
    def __init__(self, sigma, theta):
        
        import numpy as np
        from itertools import permutations
        from math import factorial
        from scipy.stats import spearmanr, kendalltau


        self.sigma = sigma
        self.theta = theta 
        self.n = len(sigma)
        # figure out better way to store this 
        self.p = np.array(list(permutations(sigma)))
        self.idx_dict = {}


    def compute_probabilities(self, distance_metric):
        distance_metrics = {1: spearmanr, 2: kendalltau}
        self.d = distance_metrics[distance_metric]
        prob_dict = {}
        for i, perm in enumerate(self.p):
            self.idx_dict[i] = perm
            prob_dict[i] = self.__cps_prob(np.array(perm))
        # overwrite to be dictionary
        self.p = prob_dict

    def __cps_prob(self, perm):
        num = lambda k, j: np.exp(self.theta * self.__dhat(self.__snk(perm, k, j)))
        denom = lambda k: sum([num(k, j) for j in xrange(k, self.n)])
        # todo: make this not a list comprehension 
        return np.prod([num(k, k) / denom(k) for k in xrange(self.n)])


    def __snk(self, perm, k, j):
        # todo: make this more efficient 
        lst = []
        for arr in self.p: 
            if np.array_equal(arr[:k], perm[:k]) & (arr[j] == perm[j]): 
                lst.append(arr)
        return lst
                
    def __dhat(self, snk):
        # make this not a list comprehension :( 
        return sum([self.d(x, np.array(self.sigma)).correlation for x in snk]) / float(len(snk))
    
    def run_tests(): 
        # assert some stuff
        return    

