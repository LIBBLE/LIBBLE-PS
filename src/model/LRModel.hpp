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

#ifndef _LRMODEL_HPP_
#define _LRMODEL_HPP_

#include <cmath>
#include "../storage/include_storage.hpp"
#include "Model.hpp"

class LRModel : public Model {
   public:
    LRModel() {}
    double compute_loss(const DataSet &ds, const Parameter &params, int num_of_all_data,
                        int num_workers, double lambda) override {
        double loss = 0;
        for (int i = 0; i < ds.num_rows; i++) {
            DataPoint &d = ds.data[i];
            double z = 0;
            for (int j = 0; j < d.key.size(); j++) {
                z += params.parameter[d.key[j]] * d.value[j];
            }
            loss += log(1 + exp(-d.label * z));
        }
        loss /= (double)num_of_all_data;
        for (int i = 0; i < params.parameter.size(); i++) {
            loss += 0.5 * lambda * pow(params.parameter[i], 2) / ((double)num_workers);
        }
        return loss;
    }

    void compute_batch_gradient(const DataSet &ds, std::uniform_int_distribution<> &u,
                                std::default_random_engine &e, const Parameter &params,
                                Gradient_Dense &g, const double &lambda,
                                const int &batch_size) override {
        g.reset();
        for (int i = 0; i < batch_size; i++) {
            DataPoint &d = ds.data[u(e)];
            double z = 0;
            for (int j = 0; j < d.key.size(); j++) {
                z += params.parameter[d.key[j]] * d.value[j];
            }
            z = -d.label * (1 - 1 / (1 + exp(-d.label * z))) / batch_size;
            for (int j = 0; j < d.key.size(); j++) {
                g.gradient[d.key[j]] += z * d.value[j];
            }
        }
        for (int i = 0; i < params.parameter.size(); i++) {
            g.gradient[i] += lambda * params.parameter[i];
        }
    }

    void compute_full_gradient(const DataSet &ds, const Parameter &params, Gradient_Dense &g,
                               const double &lambda, const int &num_of_all_data) override {
        g.reset();
        for (int i = 0; i < ds.num_rows; i++) {
            DataPoint &d = ds.data[i];
            double z = 0;
            for (int j = 0; j < d.key.size(); j++) {
                z += params.parameter[d.key[j]] * d.value[j];
            }
            // may divide num_of_all_data here and left only sum up for server
            z = -d.label * (1 - 1 / (1 + exp(-d.label * z)));
            for (int j = 0; j < d.key.size(); j++) {
                g.gradient[d.key[j]] += z * d.value[j];
            }
        }
    }

    void update_parameters(Parameter &params, const Gradient_Dense &g, const double &rate,
                           const double &lambda) override {
        params.subs_gradient(g, rate);
    }

    void local_update_para(const DataSet &ds, std::uniform_int_distribution<> &u,
                           std::default_random_engine &e, Parameter &params,
                           const Gradient_Dense &full_grad, const double &lambda,
                           const int &num_epoches, const double &rate, const int &id) override {
        const std::vector<double> old_params = params.parameter;
        for (int i = 0; i < num_epoches * ds.num_rows; i++) {
            double z, z1 = 0, z2 = 0;
            const DataPoint &d = ds.data[u(e)];
            for (int j = 0; j < d.key.size(); j++) {
                z1 += params.parameter[d.key[j]] * d.value[j];
                z2 += old_params[d.key[j]] * d.value[j];
            }
            z = rate * d.label * (1 / (1 + exp(-d.label * z1)) - 1 / (1 + exp(-d.label * z2)));
            for (int j = 0; j < params.parameter.size(); j++) {
                params.parameter[j] -=
                    rate * (full_grad.gradient[j] + lambda * params.parameter[j]);
            }
            for (int j = 0; j < d.key.size(); j++) {
                params.parameter[d.key[j]] -= z * d.value[j];
            }
        }
    }

    void local_update_sparse_para(const DataSet &ds, std::uniform_int_distribution<> &u,
                                  std::default_random_engine &e, Parameter &params,
                                  const Gradient_Dense &full_grad, const double &lambda,
                                  const int &num_epoches, const double &rate) override {
        const std::vector<double> old_params = params.parameter;
        double a = 1, b = 0;
        for (int i = 0; i < num_epoches * ds.num_rows; i++) {
            double z, z1 = 0, z2 = 0;
            const DataPoint &d = ds.data[u(e)];
            for (int j = 0; j < d.key.size(); j++) {
                z1 += (a * params.parameter[d.key[j]] + b * full_grad.gradient[d.key[j]]) *
                      d.value[j];
                z2 += old_params[d.key[j]] * d.value[j];
            }
            b = (1 - lambda * rate) * b - rate;
            a = (1 - lambda * rate) * a;
            z = rate * d.label * (1 / (1 + exp(-d.label * z1)) - 1 / (1 + exp(-d.label * z2))) / a;
            for (int j = 0; j < d.key.size(); j++) {
                params.parameter[d.key[j]] -= z * d.value[j];
            }
        }
    }
};

#endif