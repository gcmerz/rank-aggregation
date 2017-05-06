import operator

def borda_count(rankings): 
	'''
		simple implementation of borda count
	'''
	# dictionary mapping positions to values
	pos_to_points = {k: v for k, v in zip(xrange(len(rankings[0])), xrange(len(rankings[0]), 0, -1))}
	final = {}
	# sum and put in a dictionary
	for rank in rankings: 
	    for i, item in enumerate(rank): 
	        if item in final: 
	            final[item] += pos_to_points(i)
	        else: 
	            final[item] = pos_to_points(i)
	# sort dictionary and return final output ranking (tie-breaks randomly)
	final_items = sorted(final.items(), key=operator.itemgetter(1), reverse=True)
	return [item[0] for item in final_items]

def condorcet_method(rankings): 
	return None


