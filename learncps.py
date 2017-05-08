# coding: utf-8
from __future__ import print_function
import numpy as np
from election import Election
import time
from multiprocessing import Pool
from util import read_sushi_votes
# import matplotlib.pyplot as plt


def quick_kendalltau(k, l, pi, sigma): 
    ''' 
        function: quick_kendalltau 
        params: k (int), l (int), pi (iterable, permutation of ints 1 -- n), 
            sigma (np.array, permutation of ints 1-- n)
        returns: quick computation of kendall tau coset-distance as
            outlined in "A New Probaiblistic Metric for Rank Aggregation"
    '''
    n = len(pi)
    sigma = list(sigma)
    count = .25 * (n - l) * (n - l - 1)
    for i in xrange(k): 
        for j in xrange(i+1, n): 
            count += int(sigma.index(pi[i]) > sigma.index(pi[j]))
    for j in xrange(l + 1, n): 
        count += int(sigma.index(pi[l]) > sigma.index(pi[j]))
    return count

def quick_src(k, l, pi, sigma):
    ''' 
        function: quick_src
        params: k (int), l (int), pi (iterable, permutation of ints 1 -- n), 
            sigma (np.array, permutation of ints 1-- n)
        returns: quick computation of spearman's rank coorelation coset-distance 
            as outlined in "A New Probaiblistic Metric for Rank Aggregation"
    '''
    n = len(pi)
    sigma = list(sigma)
    count = 0.
    for i in xrange(k):
        count += (sigma.index(pi[i]) - i) ** 2
    count += (sigma.index(pi[l]) - k) ** 2
    temporary_count = 0.
    for i in range(k, n):
        for j in range(k, n):
            temporary_count += (sigma.index(pi[i]) - j) ** 2
    return count + (1. / float(n - k)) * temporary_count

def find_optimal_theta(pi, sigma_set, lr=1.0, dist=quick_src, iterations=None, verbose=False): 
    '''
        function: find_optimal_theta
            given a target permutation and a set of votes, uses MLE / gradient ascent to 
            find the optimal parameters theta for the CPS model
        params: pi (iterable, permutation of ints 1--n)
                sigma_set (np.array of np.arrays, each inner np.array is a permutation 
                        of ints 1--n, corresponds to each users votes)
                lr (learning rate for gradient ascent, float, inbetween 0 and 1)
                iterations (int, number of iterations to perform for gradient ascent)
                verbose (bool, if True prints statements detailing progress)
        returns: tuple of optimal theta (np.array), and final loss (float)
    '''
    n = len(pi)
    M = len(sigma_set)
    
    def expon(theta_est, k, j): 
        # helper function for exponential of coset distances, which is called often
        return np.exp(-1. * sum([theta_est[m] * dist(k, j, pi, sigma_set[m]) for m in xrange(M)]))
    
    def gradient(theta_est, theta_est_idx):
        # calculates the gradient of the estimated theta 
        # (theta_est) and a certain index (theta_est_idx)
        total = 0.
        for k in xrange(n): 
            numerator = sum([dist(k, j, pi, sigma_set[theta_est_idx]) * 
                expon(theta_est, k, j) for j in xrange(k, n)])
            denominator = sum([expon(theta_est, k, j) for j in xrange(k, n)])
            total += (numerator / denominator) - dist(k, k, pi, sigma_set[theta_est_idx])
        return total

    def loss(theta_est):
        # calculates the loss for an estimated theta
        loss = 0.
        for k in range(n):
            loss -= sum([theta_est[m] * dist(k, k, pi, sigma_set[m]) for m in xrange(M)])
            loss -= np.log(sum([expon(theta_est, k, j) for j in xrange(k, n)]))
        return loss

    # gradient ascent, continuously modify learning rate to speed up convergence
    theta = np.zeros(M)
    loss_array = [loss(theta)]
    ctr = 0
    while ctr < iterations and loss_array[-1] < -1e-5:
        if verbose:
            print(ctr, loss_array[-1])
        lr_ = lr
        if loss_array[-1] < -10.:
            lr_ = 0.00004
        elif loss_array[-1] < -1.:
            lr_ = 0.001
        elif loss_array[-1] < -0.1:
            lr_ = 0.004
        elif loss_array[-1] < -0.01:
            lr_ = 0.04
        elif loss_array[-1] < -0.001:
            lr_ = 0.4
        elif loss_array[-1] < -0.0001:
            lr_ = 4.
        elif loss_array[-1] < -0.00001:
            lr_ = 40.
        theta = theta + lr_ * np.array([gradient(theta, i) for i in xrange(M)])
        loss_array.append(loss(theta))
        ctr += 1
    return theta, loss_array 

def sequential_inference(theta, sigma, elements=range(10), dist=quick_src): 
    '''
        function: sequential_inference 
            given dispersion parameters theta and a set of votes sigma, 
            infer the target parameter of the CPS model pi 
        params: theta: float list of dispersion parameters of len(sigma)
                sigma: np.array of np.array of permutations 1--n
                elements: number of elements in the rankings (just in case rankings are weirdly
                    formatted)
                dist: the distance metric to use
        returns: pi, a list of integers that is a permutation of 1--(len(sigma[0]) - 1)
    '''
    pi = []
    M = len(theta)
    n = len(sigma[0])
    D = elements

    def r(l, e):
        temp = l[:]
        temp.remove(e)
        return temp

    for k in range(n):
        obj = np.argmin([sum([theta[m] * dist(k, k, pi + [elt] + r(D, elt), sigma[m]) for m in range(M)]) for elt in D])
        pi.append(D[obj])
        D.remove(D[obj])
        
    return pi

def community_sequential_inference(thetas, sigmas, elements=range(10), dist=quick_src): 
    '''
        function: community_sequential_inference 
            given dispersion parameters thetas for sets of votes sigmas from different communities, 
            infer a single ranking for all communities in the population
        params: thetas: float list of dispersion parameters of len(sigmas[i])
                sigma: np.arrays of np.arrays of permutations 1--n
                elements: number of elements in the rankings (just in case rankings are weirdly
                    formatted)
                dist: the distance metric to use
        returns: pi, a list of integers that is a permutation of 1--(len(sigma[0]) - 1)
    '''
    pi = []
    n = len(elements)
    num_communities = len(thetas)
    D = elements

    def r(l, e):
        temp = l[:]
        temp.remove(e)
        return temp

    for k in range(n):
        sums = np.zeros(len(D))
        for i in range(num_communities):
            M = len(thetas[i])
            sums = sums + np.array([sum([thetas[i][m] * dist(k, k, pi + [elt] + r(D, elt), sigmas[i][m]) for m in range(M)]) for elt in D])
        obj = np.argmin(sums)
        pi.append(D[obj])
        D.remove(D[obj])
        
    return pi

if __name__ == '__main__': 
    num_clusters = 2

    def f(x):
        pi, cluster = x
        theta, la = find_optimal_theta(pi, cluster, lr=40., iterations=1000, verbose = True)
        f = open(str(pi) + str(num_clusters) + '.txt', 'w')
        print(pi, '\n', cluster, '\n', theta, '\n', la, file=f)

    votes, _ = read_sushi_votes(same=True)
    start = time.time()
    E = Election(num_clusters=num_clusters, votes=votes[:500])
    pool = Pool(num_clusters)
    pool.map(f, zip(E.cluster_centers, E.vote_clusters))
    print(time.time() - start)
