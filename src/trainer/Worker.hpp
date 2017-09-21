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

#ifndef _WORKER_HPP_
#define _WORKER_HPP_

#include <cassert>
#include <functional>
#include <random>
#include <vector>
#include "../storage/include_storage.hpp"
#include "../util/include_util.hpp"
#include "Trainer.hpp"

class Worker : public Trainer {
   private:
    int worker_id;
    int batch_size;
    double lambda;
    char info;
    double rate;
    DataSet dataset;
    Parameter params;
    Gradient_Dense grad;
    Gradient_Dense full_grad;

   public:
    Worker(int n_ser, int n_wor, int n_c, int n_r, int n_e, int n_i, int mode_, std::string f,
           Model *model_p, Comm *comm_p, int proc_id, int b_s, double lambda_, double r)
        : Trainer(n_ser, n_wor, n_c, n_r, n_e, n_i, mode_, f, model_p, comm_p),
          worker_id(proc_id),
          batch_size(b_s),
          lambda(lambda_),
          rate(r) {
        params.resize(num_cols);
        grad.resize(num_cols);
        full_grad.resize(num_cols);
    }

    void work() override {
        read_data();
        std::random_device rd;
        std::default_random_engine e(rd());
        std::uniform_int_distribution<> u(0, dataset.get_num_rows() - 1);

        pull();
        double i_loss = calculate_loss();
        report_loss(i_loss);
        for (int i = 0; i < num_iters; i++) {
            MPI_Barrier(MPI_COMM_WORLD);  // start
            pull();

            calculate_part_full_gradient();

            push();

            pull_full_grad();

            local_update_sparse(u, e);

            scope_push();
            MPI_Barrier(MPI_COMM_WORLD);  // end

            pull();
            double loss = calculate_loss();
            report_loss(loss);
        }

        // std::cout << "worker " << worker_id << " done" << std::endl;
    }

    void sample_data(std::vector<int> &sample_ids) {
        //   assert(sample_ids.size() == 0);
        int num_rows = dataset.get_num_rows(), left = batch_size;
        std::default_random_engine e(std::random_device{}());
        for (int i = 0; i < num_rows; i++) {
            int x = e() % (num_rows - i);
            if (x < left) {
                sample_ids.push_back(i);
                left--;
                if (left == 0) break;
            }
        }
    }

    void pull() { comm_ptr->W_recv_params_from_all_S(params); }

    void pull_full_grad() { comm_ptr->W_recv_full_grad_from_all_S(full_grad); }

    void push() { comm_ptr->W_send_grads_to_all_S(grad); }

    void scope_push() { comm_ptr->W_send_params_to_all_S(params); }

    // read data from files
    void read_data() { dataset.read_from_file(data_file, worker_id, num_workers, num_cols); }

    // calculate loss for all data
    double calculate_loss() {
        return model_ptr->compute_loss(dataset, params, num_of_all_data, num_workers, lambda);
    }

    // calculate local full gradient
    void calculate_part_full_gradient() {
        model_ptr->compute_full_gradient(dataset, params, grad, lambda, num_of_all_data);
    }

    void local_update_sparse(std::uniform_int_distribution<> &u, std::default_random_engine &e) {
        model_ptr->local_update_sparse_para(dataset, u, e, params, full_grad, lambda, num_epoches,
                                            rate);
    }

    // report loss to coordinator
    void report_loss(double loss) { comm_ptr->W_send_loss_to_C(loss); }
};

#endif