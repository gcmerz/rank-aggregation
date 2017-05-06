# coding: utf-8
import numpy as np
from election import Election
# import matplotlib.pyplot as plt

def quick_kendalltau(k, l, pi, sigma): 
    ''' computes quick kendall_tau distances using method described in paper '''
    n = len(pi)
    sigma = list(sigma)
    count = .25 * (n - l) *(n - l - 1)
    for i in xrange(k): 
        for j in xrange(i+1, n): 
            count += int(sigma.index(pi[i]) > sigma.index(pi[j]))
    for j in xrange(l + 1, n): 
        count += int(sigma.index(pi[l]) > sigma.index(pi[j]))
    return count

def quick_src(k, l, pi, sigma):
    n = len(pi)
    sigma = list(sigma)
    count = 0.
    for i in xrange(k-1):
        count += (sigma.index(pi[i]) - i) ** 2
    count += (sigma.index(pi[l]) - (k - 1)) ** 2
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
            print loss_array[-1]
        theta = theta + lr * np.array([gradient(theta, i) for i in xrange(M)])
        loss_array.append(loss(theta))
        ctr += 1
    return theta, loss_array 

# heta, la = find_optimal_theta([1,2,3], [[2,1,3], [2,3,1]], 1.0, quick_src, iterations=100, verbose=True)
# print la[-1]
# plt.plot(la)
# plt.show()
# plt.savefig(str(lr) + ".png")

def sequential_inference(theta, sigma, dist=quick_src): 
    pi = []
    M = len(theta)
    n = len(sigma[0])
    items = list(xrange(1,n + 1))
    def get_min(idx, items, pi): 
        vals = [(elt, sum([theta[m] * dist(idx, idx, pi + [elt], sigma[m]) for m in xrange(M)])) for elt in items]
        return min(vals, key = lambda t: t[1])[0]
    for i in xrange(1, n + 1): 
        val = get_min(i, items, pi)
        pi.append(val)
        items.remove(val)
    return pi

# sequential_inference([1,2,1,2], [[1,2,3,4,5],[1,2,4,3,5], [1,5,4,3,2], [1,2,3,4,5]])

# read in votes from dataset
votes = []
with open('sushi3-2016/sushi3a.5000.10.order') as f:
    lines = f.readlines()
    for line in lines[1:]:
        votes.append(np.array(map(int, line.rstrip('\n').split(' ')[2:])))
votes = np.array(votes)

n = 20
E = Election(num_clusters=2, votes=votes[:n])
for pi, cluster in zip(E.cluster_centers, E.vote_clusters):
    for lr in [0.001, 0.0025, 0.003, .0035]:
        theta, la = find_optimal_theta(pi, cluster, lr=lr, iterations=10)
        print lr, la[-1]
