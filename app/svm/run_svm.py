#!usr/bin/python

import os
import sys

#--------------------------------------
# modify these arguments to run the program
machine_file = ' mach ' # hosts file
test_data_file = ' /data/rcv1_test.binary ' #test data path, 'null': no test_file
train_data_file = ' /data/rcv1_train.binary ' #train data path. It will read the directory 'train_data_file_/' for partition data.
n_cols = ' 47236 ' #train data feature number
n_rows = ' 20242 ' #train data instance number
n_servers = ' 2 ' 
n_workers = ' 16 '
n_epoches = ' 1 ' 
n_iters = ' 100 '
rate = ' 0.1 ' #step size
lam = ' 0.0001 ' #regularization hyperparameter 
param_init = ' 0 ' # parameter initialization. 0--all zero 1--randomize to [0,1]
#--------------------------------------

n_trainers = str(1 + int(n_servers) + int(n_workers))

os.system('mpirun -n ' + n_trainers + ' -f ' + machine_file + ' ./svm ' + ' -n_servers ' + n_servers
+ '-n_workers ' + n_workers + ' -n_epoches' + n_epoches + ' -n_iters' + n_iters + ' -n_cols' + n_cols 
+ ' -n_rows' + n_rows + ' -test_data_file ' +  test_data_file + ' -train_data_file ' +  train_data_file
 + ' -rate ' + rate + ' -lambda ' + lam  + ' -param_init ' + param_init )

