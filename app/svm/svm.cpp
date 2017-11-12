/**
        * Copyright (c) 2017 LIBBLE team supervised by Dr. Wu-Jun LI at Nanjing University.
        * All Rights Reserved.
        * Licensed under the Apache License, Version 2.0 (the "License");
        * you may not use this file except in compliance with the License.
        * You may obtain a copy of the License at
        *
        * http://www.apache.org/licenses/LICENSE-2.0
        *
        * Unless required by applicable law or agreed to in writing, software
        * distributed under the License is distributed on an "AS IS" BASIS,
        * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        * See the License for the specific language governing permissions and
        * limitations under the License. */

#include <cassert>
#include <iostream>
#include <string>
#include <thread>
#include "mpi.h"

#include "../../src/include_ps.hpp"

using namespace std;

int n_servers = 2, n_workers = 2;
int n_epoches = 100;
int n_iters = 10;
int n_cols = 50;  // num of features for a single data point
int n_rows = 1000;
string test_data_file = "";
string data_file = "";
int batch_size = 10;  // the size of sub-dataset a worker sampled for an epoch
double rate = 0.01;
double lambda = 0.0001;
int mode = 1;
int param_init = 0;

int main(int argc, char **argv) {
    int proc_id, num_procs, provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        printf("MPI do not Support Multiple thread\n");
        exit(0);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int pos;
    if ((pos = arg_parser("-n_servers", argc, argv)) > 0) n_servers = atoi(argv[pos + 1]);
    if ((pos = arg_parser("-n_workers", argc, argv)) > 0) n_workers = atoi(argv[pos + 1]);
    if ((pos = arg_parser("-n_epoches", argc, argv)) > 0) n_epoches = atoi(argv[pos + 1]);
    if ((pos = arg_parser("-n_iters", argc, argv)) > 0) n_iters = atoi(argv[pos + 1]);
    if ((pos = arg_parser("-n_cols", argc, argv)) > 0) n_cols = atoi(argv[pos + 1]);
    if ((pos = arg_parser("-n_rows", argc, argv)) > 0) n_rows = atoi(argv[pos + 1]);
	if ((pos = arg_parser("-test_data_file", argc, argv)) > 0) test_data_file = argv[pos + 1];
    if ((pos = arg_parser("-train_data_file", argc, argv)) > 0) data_file = argv[pos + 1];
    if ((pos = arg_parser("-batch_size", argc, argv)) > 0) batch_size = atoi(argv[pos + 1]);
    if ((pos = arg_parser("-rate", argc, argv)) > 0) rate = atof(argv[pos + 1]);
    if ((pos = arg_parser("-lambda", argc, argv)) > 0) lambda = atof(argv[pos + 1]);
    if ((pos = arg_parser("-mode", argc, argv)) > 0) mode = atoi(argv[pos + 1]);
    if ((pos = arg_parser("-param_init", argc, argv)) > 0) param_init = atoi(argv[pos + 1]);

    n_cols += 1;  // add the bias term

    assert(n_servers + n_workers + 1 == num_procs);

    Model *model_ptr = new SVMModel();
    Comm *comm_ptr = new Comm(n_servers, n_workers, n_cols);

    Trainer *trainer_ptr = nullptr;
    if (proc_id == 0) {
        trainer_ptr = new Coordinator(n_servers, n_workers, n_cols, n_rows, n_epoches, n_iters,
                                      mode, data_file, model_ptr, comm_ptr);
    } else if (proc_id <= n_servers) {
        trainer_ptr = new Server(n_servers, n_workers, n_cols, n_rows, n_epoches, n_iters, mode,
                                 data_file, model_ptr, comm_ptr, proc_id, lambda, rate, param_init);
    } else {
        trainer_ptr =
            new Worker(n_servers, n_workers, n_cols, n_rows, n_epoches, n_iters, mode, data_file,
                       model_ptr, comm_ptr, proc_id - n_servers, batch_size, lambda, rate, test_data_file);
    }

    trainer_ptr->work();  // start working

    MPI_Finalize();

    delete model_ptr;
    delete trainer_ptr;
    delete comm_ptr;
    return 0;
}