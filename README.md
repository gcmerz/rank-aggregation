# rank-aggregation


Important files: 

* election.py -- code for HAC, "Election" class which outputs clusters and cluster centers
* util.py -- different distance metrics used in election and CPS model, as well as some 
  code for reading from the sushi dataset
* learncps.py -- code for learning CPS model, including gradient ascent / MLE of theta paramaters
* inference.py -- code for testing sequential / community inference for outputting best ranking
* comparison.py -- prints comparison of different metrics 
* requirements.txt -- please run 'pip install -r requirements.txt' before attempting to run any code
* results -- folder of results for learned CPS models, files are named by the target ranking and the
  number of cluster, each file is a text file that contains the parameters for the CPS models
  as well as the losses incurred by learning the optimal theta
  

