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

#ifndef _SVMMODEL_HPP_
#define _SVMMODEL_HPP_

#include <cmath>
#include "../storage/include_storage.hpp"
#include "Model.hpp"

class SVMModel : public Model {
   public:
    SVMModel() {}
    double compute_loss(const DataSet &ds, const Parameter &params, const int num_of_all_data,
                        const int num_workers, const double lambda) override {
        double loss = 0;
        for (int i = 0; i < ds.num_rows; i++) {
            DataPoint &d = ds.data[i];
            double z = 0;
            for (int j = 0; j < d.key.size(); j++) {
                z += params.parameter[d.key[j]] * d.value[j];
            }
			if(d.label*z <1){
				loss += (1 - d.label*z);
			}
        }
        loss /= (double)num_of_all_data;
		double index = 0.5 * lambda / ((double)num_workers);
        for (int i = 0; i < params.parameter.size(); i++) {
            loss += index * pow(params.parameter[i], 2) ;
        }
        return loss;
    }
	
	void compute_full_gradient(const DataSet &ds, const Parameter &params, Gradient_Dense &g,
                               const int num_of_all_data) override {
        g.reset();
        for (int i = 0; i < ds.num_rows; i++) {
            DataPoint &d = ds.data[i];
            double z = 0;
            for (int j = 0; j < d.key.size(); j++) {
                z += params.parameter[d.key[j]] * d.value[j];
            }
			if(d.label*z <1){
				for (int j = 0; j < d.key.size(); j++) {
					g.gradient[d.key[j]] -= d.label * d.value[j];
				}
			}
        }
		vector_divi(g.gradient, num_of_all_data);
    }
	
	void update(const DataSet &ds, std::uniform_int_distribution<> &u,
                                  std::default_random_engine &e, Parameter &params,
                                  const Gradient_Dense &full_grad, const double lambda,
                                  const int num_epoches, const double rate, const int recover_index,
                                  const int num_of_all_data, const int num_workers) override {
        const std::vector<double> old_params = params.parameter;
        double a = 1, b = 0;
        for (int i = 0; i < num_epoches * (num_of_all_data/num_workers); i++) {
            if(recover_index !=0 && i%recover_index == 0){
                vector_multi_add(params.parameter, a, full_grad.gradient, b);
                a = 1;
                b = 0;
            }
            double z, z1 = 0, z2 = 0;
            const DataPoint &d = ds.data[u(e)];
            for (int j = 0; j < d.key.size(); j++) {
                z1 += (a * params.parameter[d.key[j]] + b * full_grad.gradient[d.key[j]]) *
                      d.value[j];
                z2 += old_params[d.key[j]] * d.value[j];
            }
            b = (1 - lambda * rate) * b - rate;
            a = (1 - lambda * rate) * a;
			
			if(d.label * z1 >1 && d.label * z2 <1){
				for (int j = 0; j < d.key.size(); j++) {
					params.parameter[d.key[j]] -= rate * d.label * d.value[j]/a;
				}
			}
			else if(d.label * z1 <1 && d.label * z2 >1){
				for (int j = 0; j < d.key.size(); j++) {
					params.parameter[d.key[j]] += rate * d.label * d.value[j]/a;
				}
			}
			else;
        }
    }
};

#endif