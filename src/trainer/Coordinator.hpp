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

#ifndef _COORDINATOR_HPP_
#define _COORDINATOR_HPP_

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include "../util/include_util.hpp"
#include "Trainer.hpp"

class Coordinator : public Trainer {
   private:
    char info;
    Parameter params;

   public:
    Coordinator(int n_ser, int n_wor, int n_c, int n_r, int n_e, int n_i, int mode_, std::string f,
                Model *model_p, Comm *comm_p)
        : Trainer(n_ser, n_wor, n_c, n_r, n_e, n_i, mode_, f, model_p, comm_p) {
        params.resize(num_cols);
    }

    void work() override {
		
        std::chrono::duration<double> total_time = (std::chrono::duration<double>)0;
        double i_loss = gather_loss();
		double accuracy = receive_accuracy();
		std::cout.precision(15);
		if(accuracy != -1) {
			std::cout << "[0.000000s] iter 0 's loss is " << i_loss
			<< ", accuracy is "<< accuracy << std::endl;
		}
		else
			std::cout << "[0.000000s] iter 0 's loss is " << i_loss << std::endl;
        for (int i = 0; i < num_iters; i++) {
            MPI_Barrier(MPI_COMM_WORLD);  // start
            auto start = std::chrono::steady_clock::now();
            MPI_Barrier(MPI_COMM_WORLD);  // end
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> time = end - start;
            total_time += time;
            double loss = gather_loss();
			double accuracy = receive_accuracy();
			/************print***************/
			if(accuracy != -1) {
				std::cout << "[" << total_time.count() << "s] iter " << i + 1 << " 's loss is " << loss
				<< ", accuracy is "<< accuracy << std::endl;
			}
			else{
				std::cout << "[" << total_time.count() << "s] iter " << i + 1 << " 's loss is " << loss
				<< std::endl;
			}
            std::string file = data_file + "_info";
            std::string info = std::to_string(i) + " " + std::to_string(total_time.count()) + " ";
            write_file(file, info, loss, accuracy);
        }

        recv_params_from_servers_and_save();
        // std::cout << "coordinator done" << std::endl;
    }
	
	// gather loss from workers
	double gather_loss() {
        double loss = comm_ptr->C_recv_loss_from_all_W();
        return loss;
    }

    double receive_accuracy() {
        double accuracy = comm_ptr->C_recv_accuracy_from_W();
        return accuracy;
    }

    // receive parameters from servers and save to file
    void recv_params_from_servers_and_save() {
        comm_ptr->C_recv_params_from_all_S(params);
        // save params to file
        params.save_into_file(data_file);
        std::cout << "Already saved parameters into file." << std::endl;
    }
};

#endif