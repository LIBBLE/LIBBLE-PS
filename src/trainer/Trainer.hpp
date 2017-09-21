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

#ifndef _TRAINER_HPP_
#define _TRAINER_HPP_

#include <string>
#include "../comm/include_comm.hpp"
#include "../model/include_model.hpp"
#define PRINT_ITER 10  // for SGD

class Trainer {
   protected:
    int num_servers, num_workers;  // number of servers and workers in this system
    int num_cols;                  // number of features
    int num_of_all_data;           // number of data
    int num_epoches;               // number of epoches in the training process
    int num_iters;                 // number of iterations in scope
    std::string data_file;         // file name of the dataset
    Model *model_ptr;
    Comm *comm_ptr;
    int mode;

   public:
    Trainer(int n_ser, int n_wor, int n_c, int n_r, int n_e, int n_i, int mode_, std::string f,
            Model *model_p, Comm *comm_p)
        : num_servers(n_ser),
          num_workers(n_wor),
          num_cols(n_c),
          num_of_all_data(n_r),
          num_epoches(n_e),
          num_iters(n_i),
          mode(mode_),
          data_file(f),
          model_ptr(model_p),
          comm_ptr(comm_p) {}

    // this function include the whole process of working for each participants
    virtual void work() = 0;
};

#endif