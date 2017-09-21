#!usr/bin/python

import os
import sys

#--------------------------------------
# modify these arguments to run the program
machine_file = 'mach' # hosts file
data_file = '/data/webspam_wc_normalized_trigram.svm' #data path
n_cols = ' 16609143 '
n_rows = ' 350000 '
n_servers = ' 2 ' 
n_workers = ' 16 '
n_epoches = ' 1 ' 
n_iters = ' 10 '
rate = ' 1 ' #step size
lam = ' 0.0001 ' #regularization hyperparameter 
param_init = ' 0 ' # parameter initialization. 0--all zero 1--randomize to [0,1]
#--------------------------------------

n_trainers = str(1 + int(n_servers) + int(n_workers))

os.system('mpirun -n ' + n_trainers + ' -f ' + machine_file + ' ./lr ' + ' -n_servers ' + n_servers
+ '-n_workers ' + n_workers + ' -n_epoches' + n_epoches + ' -n_iters' + n_iters + ' -n_cols' + n_cols + ' -n_rows' + n_rows + ' -data_file ' +  data_file
 + ' -rate ' + rate + ' -lambda ' + lam  + ' -param_init ' + param_init )

