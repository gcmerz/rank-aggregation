# coding: utf-8
import numpy as np
from election import Election
import time
from multiprocessing import Pool
from util import read_sushi_votes
# import matplotlib.pyplot as plt

def quick_src(k, l, pi, sigma):
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

def find_optimal_theta(pi, sigma_set, lr=1.0, dist=quick_src, iterations=1000, verbose=False): 
    n = len(pi)
    M = len(sigma_set)
    
    def expon(theta_est, k, j): 
        return np.exp(-1. * sum([theta_est[m] * dist(k, j, pi, sigma_set[m]) for m in xrange(M)]))
    
    def gradient(theta_est, theta_est_idx):
        total = 0.
        for k in xrange(n): 
            numerator = sum([dist(k, j, pi, sigma_set[theta_est_idx]) * expon(theta_est, k, j) for j in xrange(k, n)])
            denominator = sum([expon(theta_est, k, j) for j in xrange(k, n)])
            total += (numerator / denominator) - dist(k, k, pi, sigma_set[theta_est_idx])
        return total

    def loss(theta_est):
        loss = 0.
        for k in range(n):
            loss -= sum([theta_est[m] * dist(k, k, pi, sigma_set[m]) for m in xrange(M)])
            loss -= np.log(sum([expon(theta_est, k, j) for j in xrange(k, n)]))
        return loss

    # gradient ascent
    theta = np.zeros(M)
    loss_array = [loss(theta)]
    ctr = 0
    while ctr < iterations and loss_array[-1] < -1e-5:
        if verbose:
            print ctr, loss_array[-1]
        lr_ = lr
        if loss_array[-1] < -0.1:
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

# theta, la = find_optimal_theta([1,2,3], [[2,1,3], [2,3,1]], 1.0, quick_src, iterations=100, verbose=True)
# print la[-1]
# plt.plot(la)
# plt.show()
# plt.savefig(str(lr) + ".png")

def sequential_inference(theta, sigma, elements=range(10), dist=quick_src): 
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

# theta, _ = find_optimal_theta([1,4,2,3,5], [[1,2,3,4,5],[4,2,1,3,5],[1,5,4,3,2],[1,2,3,4,5]], lr=4., iterations=4000, verbose=True)
# print "SI: ", sequential_inference(theta, [[1,2,3,4,5],[1,2,4,3,5],[1,5,4,3,2],[1,2,3,4,5]], elements=[1,2,3,4,5])

votes, _ = read_sushi_votes(same=True)

def f(x):
    pi, cluster = x
    theta, la = find_optimal_theta(pi, cluster, lr=4., iterations=200)
    pi_pred = sequential_inference(theta, cluster)
    return pi, pi_pred

n = 20
nc = 2
start = time.time()
E = Election(num_clusters=nc, votes=votes[:n])
pool = Pool(nc)
print pool.map(f, zip(E.cluster_centers, E.vote_clusters))
print time.time() - start
