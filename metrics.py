import operator

def borda_count(rankings): 
	'''
		simple implementation of borda count
	'''
	# dictionary mapping positions to values
	pos_to_points = {k: v for k, v in zip(xrange(len(rankings[0])), xrange(len(rankings[0]), 0, -1))}
	# get values for each item 
	ranks_points = [[(item, pos_to_points[i]) for i, item in enumerate(lst)] for lst in rankings]
	
	final = {}
	# sum and put in a dictionary
	for rank in ranks_points: 
	    for item, points in rank: 
	        if item in final: 
	            final[item] += points
	        else: 
	            final[item] = points
	# sort dictionary and return final output ranking (tie-breaks randomly)
	final_items = sorted(final.items(), key=operator.itemgetter(1), reverse=True)
	return [item[0] for item in final_items]

def condorcet_method(rankings): 
	return None


