from util import * 
from metrics import *
from learncps import *
from collections import Counter


SUSHI = {
	0: 'Shrimp',
	1: 'Sea Eel',
	2: 'Tuna',
	3: 'Squid',
	4: 'Sea Urchin',
	5: 'Salmon Roe',
	6: 'Egg',
	7: 'Fatty Tuna',
	8: 'Tuna Roll',
	9: 'Cucumber',
}

def convert_to_sushi(rank):
	''' 
		function convert_to_sushi
		params: rank, a list (permutation) of numbers 1-9
		returns: a list of the corresponding sushi names in each rank
	''' 
	if type(rank) != list: 
		rank = list(rank)
	if type(rank[0]) == set: 
		rank = [v.pop() for v in rank]
	return [SUSHI[v] for v in rank]

def time_function(f, votes, candidates): 
	'''
		fuction time_function
		params: f, a function that takes one argument 
				arg, the argument that function f takes
		returns: a tuple of the result of f(arg), and the time it took
				to apply f(arg)
	'''
	start = time.time() 
	val = f(votes, candidates)
	end = time.time() - start
	return val, end

def convert_votes(sushi_votes): 
	'''
		function convert_votes
		params: sushi_votes, list of lists. each list a ranking over 
			objects 0 - (len(lst) - 1) 
		returns: votes, candidates as specified format by metrics.py
	'''
	candidates = list(xrange(len(sushi_votes[0])))
	sushi_votes = Counter([tuple(vote) for vote in sushi_votes])
	sushi_votes = list(sushi_votes.iteritems())
	return sushi_votes, candidates

if __name__ == '__main__': 
	votes, ids = read_sushi_votes() 
	mvotes, cands = convert_votes(votes)
	print "Read in Votes, calculating Borda Count"
	result, tme = time_function(borda, mvotes, cands)
	print convert_to_sushi(result)
	print tme
	print "Simpson Method"
	result, tme = time_function(minimax, mvotes, cands)
	print convert_to_sushi(result)
	print tme
	print "Instant Runoff"
	result, tme = time_function(instantrunoff, mvotes, cands)
	print convert_to_sushi(result)
	print tme
