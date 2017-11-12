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

#ifndef _COMM_HPP_
#define _COMM_HPP_

#include <vector>
#include "mpi.h"

#include "../storage/include_storage.hpp"
#include "../util/include_util.hpp"
#include "protocol.hpp"

/* the only class for doing communications */

class Comm {
   private:
    char info;
    int num_servers, num_workers, num_cols;
    std::vector<int> server_list, worker_list;
    std::vector<double> buffer;
    std::vector<int> buffer_int;

   public:
    Comm(int n_servers, int n_workers, int n_cols)
        : num_servers(n_servers), num_workers(n_workers), num_cols(n_cols) {
        for (int i = 1; i <= num_servers; i++) server_list.push_back(i);
        for (int i = 1; i <= num_workers; i++) worker_list.push_back(num_servers + i);
        buffer.resize(num_cols);
        buffer_int.resize(num_cols);
    }

    std::vector<int> get_server_list() { return server_list; }

    //--------------------coordinator-send

    //--------------------coordinator-receive
    double C_recv_loss_from_all_W() {
        double total_loss = 0, partial_loss = 0;
        for (int w_id : worker_list) {
            MPI_Recv(&partial_loss, 1, MPI_DOUBLE, w_id, WC_LOSS, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            total_loss += partial_loss;
        }
        return total_loss;
    }
	
	double C_recv_accuracy_from_W() {
        double accuracy;
        MPI_Recv(&accuracy, 1, MPI_DOUBLE, worker_list[0], WC_ACCU, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        return accuracy;
    }

    void C_recv_params_from_all_S(Parameter &params) {
        MPI_Status status;
        int pos = 0, recv_num = 0;
        for (int s_id : server_list) {
            /* recv params from each server and concatenate */
            MPI_Recv(&params.parameter[pos], params.parameter.size() - pos, MPI_DOUBLE, s_id,
                     SC_PARAMS, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_DOUBLE, &recv_num);
            pos += recv_num;
        }
    }

    //--------------------server-send
    void S_send_grads_to_all_W(const Gradient_Dense &g) {
        for (int w_id : worker_list) {
            MPI_Send(&g.gradient[0], g.gradient.size(), MPI_DOUBLE, w_id, SW_GRAD, MPI_COMM_WORLD);
        }
    }

    void S_send_params_to_all_W(Parameter &params) {
        const std::vector<double> &v = params.parameter;
        for (int w_id : worker_list) {
            MPI_Send(&v[0], v.size(), MPI_DOUBLE, w_id, SW_PARAMS, MPI_COMM_WORLD);
        }
    }

    void S_send_params_to_C(Parameter &params) {
        const std::vector<double> &v = params.parameter;
        MPI_Send(&v[0], v.size(), MPI_DOUBLE, 0, SC_PARAMS, MPI_COMM_WORLD);
    }

    //--------------------server-receive
    void S_recv_grads_from_all_W(Gradient_Dense &g) {
        g.reset();
        for (int i = 0; i < num_workers; i++) {
            // recv each gradient and add to g->gradient
            MPI_Recv(&buffer[0], buffer.size(), MPI_DOUBLE, MPI_ANY_SOURCE, WS_GRADS,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            vector_add(g.gradient, buffer);
        }
    }

    void S_recv_params_from_all_W(Parameter &params) {
        params.reset();
        for (int i = 0; i < num_workers; i++) {
            MPI_Recv(&buffer[0], buffer.size(), MPI_DOUBLE, MPI_ANY_SOURCE, WS_PARAMS,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            vector_add(params.parameter, buffer);
        }
    }

    //--------------------worker-send

    void W_send_loss_to_C(double loss) {
        MPI_Send(&loss, 1, MPI_DOUBLE, 0, WC_LOSS, MPI_COMM_WORLD);
    }
	
	void W_send_accuracy_to_C(double accuracy) {
        MPI_Send(&accuracy, 1, MPI_DOUBLE, 0, WC_ACCU, MPI_COMM_WORLD);
    }

    void W_send_grads_to_all_S(const Gradient_Dense &grad) {
        /* need to split gradient according to each server's possessions */
        int pos = 0;
        for (int s_id : server_list) {
            int len = get_local_params_size(num_cols, num_servers, s_id);
            MPI_Send(&grad.gradient[pos], len, MPI_DOUBLE, s_id, WS_GRADS, MPI_COMM_WORLD);
            pos += len;
        }
    }

    void W_send_params_to_all_S(const Parameter &params) {
        int pos = 0;
        for (int s_id : server_list) {
            int len = get_local_params_size(num_cols, num_servers, s_id);
            MPI_Send(&params.parameter[pos], len, MPI_DOUBLE, s_id, WS_PARAMS, MPI_COMM_WORLD);
            pos += len;
        }
    }

    //--------------------worker-receive
    void W_recv_params_from_all_S(Parameter &params) {
        int pos = 0;
        for (int s_id : server_list) {  // may optimize to recv unordered
            int len = get_local_params_size(num_cols, num_servers, s_id);
            MPI_Recv(&params.parameter[pos], len, MPI_DOUBLE, s_id, SW_PARAMS, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            pos += len;
        }
    }

    void W_recv_full_grad_from_all_S(Gradient_Dense &grad) {
        int pos = 0;
        for (int s_id : server_list) {
            int len = get_local_params_size(num_cols, num_servers, s_id);
            MPI_Recv(&grad.gradient[pos], len, MPI_DOUBLE, s_id, SW_GRAD, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            pos += len;
        }
    }
};

#endif